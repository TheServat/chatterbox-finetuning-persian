#!/usr/bin/env bash
# Everything the pod does, from a fresh container to a finished adapter.
#
# Run as a background child of the control server, never as the container's
# foreground process. The control server has to outlive this script: when a run
# fails, the failure is the thing most worth reading, and a container that exits
# on failure takes the explanation with it.
#
# Progress is published two ways. `$WORKSPACE/phase` carries the coarse stage so
# the launcher can tell setup from training, and `$WORKSPACE/status.json` is
# written by the trainer itself with steps, loss and throughput.
#
# Expected environment:
#   HF_TOKEN          read access to the model weights
#   TRAIN_ARGS        arguments passed straight to train.py
#   PERSIST_DIR       network volume mount, if one is attached
#   HOURLY_RATE       GPU price, so the status file can report cost so far

set -uo pipefail

WORKSPACE="${POD_ROOT:-/workspace}"
REPO_DIR="$WORKSPACE/repo"
PERSIST="${PERSIST_DIR:-$WORKSPACE/persist}"
# The cache is named after the corpora it was built from. Without this, a cheap
# yoda-only test would leave a cache that a later full run silently trains on,
# and the difference would never surface as an error.
SOURCES="${DATASET_SOURCES:-yoda narration mana_hf}"
SOURCE_TAG=$(echo "$SOURCES" | tr ' ' '_')
CACHE_ARCHIVE="$PERSIST/preprocess_${SOURCE_TAG}.tar.gz"
METADATA_COPY="$PERSIST/metadata_${SOURCE_TAG}.csv"
MODELS_DIR="$PERSIST/pretrained_models"
OUTPUT_DIR="$PERSIST/chatterbox_output"

exec >>"$WORKSPACE/bootstrap.log" 2>&1

phase() {
    printf '%s' "$1" >"$WORKSPACE/phase"
    echo "=== phase=$1  $(date -Is)"
}

fail() {
    printf '%s' "$*" >"$WORKSPACE/FAILED"
    phase failed
    echo "FAILED: $*"
    exit 1
}

step() { echo; echo "--- $* ---"; }

mkdir -p "$PERSIST" "$WORKSPACE/incoming"

# --------------------------------------------------------------------------
phase setup
step "environment"
nvidia-smi || fail "no GPU visible in this pod"
python3 --version
df -h "$WORKSPACE" "$PERSIST" | tail -3

step "python dependencies"
cd "$REPO_DIR" || fail "repository missing at $REPO_DIR"
pip install --no-cache-dir -q -r requirements.txt || fail "pip install failed"
python3 -c "import torch, transformers, peft, soundfile, num2words; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('bf16 native:', torch.cuda.get_device_capability(0)[0] >= 8)" \
    || fail "dependency import failed"

# --------------------------------------------------------------------------
phase models
step "model weights"
# Kept on the network volume when there is one, so a second run skips the
# 2 GB download entirely.
mkdir -p "$MODELS_DIR"
ln -sfn "$MODELS_DIR" "$REPO_DIR/pretrained_models"
python3 tools/fetch_models.py || fail "fetching model weights failed"

# --------------------------------------------------------------------------
phase data
step "training cache"
mkdir -p "$REPO_DIR/MyTTSDataset"

if [ -f "$CACHE_ARCHIVE" ]; then
    # A previous run on this network volume already did the expensive part.
    echo "cache found on the volume: $(du -h "$CACHE_ARCHIVE" | cut -f1)"
    tar -xzf "$CACHE_ARCHIVE" -C "$REPO_DIR/MyTTSDataset" || fail "extracting the cache failed"
    cp "$METADATA_COPY" "$REPO_DIR/MyTTSDataset/metadata.csv" 2>/dev/null || true
else
    # Built here rather than uploaded. The corpora live on HuggingFace and a
    # datacentre pulls them far faster than a home connection can push 300 MB,
    # and the result is written back to the volume so this happens exactly once.
    phase downloading_datasets
    step "downloading corpora ($SOURCES)"
    for source in $SOURCES; do
        case "$source" in
            yoda)      python3 tools/fetch_datasets.py yoda ;;
            narration) python3 tools/fetch_datasets.py narration ;;
            mana_hf)   python3 tools/fetch_datasets.py MahtaFetrat/Mana-TTS ;;
            youtube)   python3 tools/fetch_datasets.py youtube ;;
        esac || fail "downloading $source failed"
    done
    df -h "$WORKSPACE" | tail -1

    phase building_dataset
    step "building the corpus"
    # shellcheck disable=SC2086
    python3 tools/build_dataset.py --sources $SOURCES \
        --dedupe ${BUILD_ARGS:-} || fail "building the dataset failed"

    phase preprocessing
    step "preprocessing"
    python3 -u -m src.preprocess_ljspeech >>"$WORKSPACE/preprocess.log" 2>&1 \
        || fail "preprocessing failed - see preprocess.log"

    step "saving the cache back to the volume"
    tar -czf "$CACHE_ARCHIVE.part" -C "$REPO_DIR/MyTTSDataset" preprocess \
        && mv "$CACHE_ARCHIVE.part" "$CACHE_ARCHIVE" \
        && cp "$REPO_DIR/MyTTSDataset/metadata.csv" "$METADATA_COPY" \
        && echo "cached $(du -h "$CACHE_ARCHIVE" | cut -f1) for the next run" \
        || echo "WARNING: could not cache the result; the next run will rebuild"

    # The raw audio is the bulk of the disk and is finished with, while the
    # cache that replaces it is a thousandth of its size. Keeping it on the
    # volume would rent 35 GB by the month for something a datacentre re-fetches
    # in minutes, so it goes unless explicitly asked for.
    if [ "${KEEP_DATASETS:-0}" = "1" ]; then
        echo "keeping the raw corpora on the volume (KEEP_DATASETS=1)"
        mkdir -p "$PERSIST/dataset"
        cp -r "$REPO_DIR/dataset/." "$PERSIST/dataset/" 2>/dev/null || true
    fi
    rm -rf "$REPO_DIR/MyTTSDataset/wavs" "$REPO_DIR/dataset" || true
fi

CLIPS=$(find "$REPO_DIR/MyTTSDataset/preprocess" -name '*.pt' | wc -l)
echo "$CLIPS cached clips"
[ "$CLIPS" -gt 0 ] || fail "no preprocessed clips to train on"

# --------------------------------------------------------------------------
phase training
step "training"
mkdir -p "$OUTPUT_DIR"
ln -sfn "$OUTPUT_DIR" "$REPO_DIR/chatterbox_output"

# shellcheck disable=SC2086  # TRAIN_ARGS is deliberately word-split
python3 -u train.py \
    --no-preprocess \
    --status-file "$WORKSPACE/status.json" \
    --hourly-rate "${HOURLY_RATE:-0}" \
    ${TRAIN_ARGS:-} \
    >>"$WORKSPACE/train.log" 2>&1
TRAIN_EXIT=$?

if [ "$TRAIN_EXIT" -ne 0 ]; then
    fail "train.py exited $TRAIN_EXIT - see train.log"
fi

# --------------------------------------------------------------------------
phase packaging
step "packaging results"
ADAPTER="$OUTPUT_DIR/persian_adapter"
[ -d "$ADAPTER" ] || fail "training finished but $ADAPTER is missing"

RESULT="$WORKSPACE/persian_adapter.tar.gz"
tar -czf "$RESULT" -C "$OUTPUT_DIR" persian_adapter || fail "packaging the adapter failed"
cp "$RESULT" "$PERSIST/" 2>/dev/null || true
echo "result: $(du -h "$RESULT" | cut -f1)"

# Samples are small and worth having next to the adapter when judging a run.
if [ -d "$OUTPUT_DIR/inference_samples" ]; then
    tar -czf "$WORKSPACE/samples.tar.gz" -C "$OUTPUT_DIR" inference_samples || true
fi

phase done
touch "$WORKSPACE/DONE"
echo "=== complete $(date -Is) ==="
