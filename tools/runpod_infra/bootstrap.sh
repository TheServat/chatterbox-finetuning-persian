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
df -h "$WORKSPACE" "$PERSIST" | tail -3

step "python interpreter"
# RunPod's PyTorch images keep torch inside a virtualenv while `python3` on PATH
# is the system interpreter, which has never heard of it. Assuming either one
# fails a minute in with ModuleNotFoundError, so the interpreter is chosen by
# asking each candidate whether it can actually import torch.
# The most reliable pointer is pip itself: whichever environment owns the pip on
# PATH is the one the image configured, and its interpreter sits beside it.
# Guessing venv paths found nothing on this image and cost a 2.5 GB reinstall.
PIP_BIN="$(command -v pip3 || command -v pip || true)"
PIP_DIR="${PIP_BIN%/*}"

PY=""
for candidate in "$PIP_DIR/python" "$PIP_DIR/python3" \
                 /venv/main/bin/python /venv/bin/python /opt/conda/bin/python \
                 /usr/local/bin/python3 "$(command -v python3)"; do
    [ -x "$candidate" ] || continue
    if "$candidate" -c "import torch" >/dev/null 2>&1; then
        PY="$candidate"
        echo "torch already present in $PY"
        break
    fi
done
if [ -z "$PY" ]; then
    # Nothing has torch: fall back and let pip provide it. That works, at the
    # cost of a few minutes and a couple of GB.
    PY="$(command -v python3)"
    echo "no preinstalled torch anywhere; installing into $PY"
    echo "  pip on PATH: ${PIP_BIN:-none}"
    [ -n "$PIP_BIN" ] && "$PIP_BIN" --version
fi
export PY
"$PY" --version

step "python dependencies"
cd "$REPO_DIR" || fail "repository missing at $REPO_DIR"

# torch is deliberately excluded from what gets installed here. The image ships
# a build matched to its own driver - cu128 for a CUDA 12.8 host - while
# requirements.txt only says torch>=2.6.0, so pip happily fetched 2.13.0+cu130
# and the first CUDA call died with "driver is too old". The image's torch is
# the correct one; the job is to leave it alone.
grep -viE '^(torch|torchaudio|torchvision)([=<>!~ ]|$)' requirements.txt     > /tmp/requirements-pod.txt

if "$PY" -c "import torch" >/dev/null 2>&1; then
    echo "using the image's torch; installing everything else"
    "$PY" -m pip install --no-cache-dir -q -r /tmp/requirements-pod.txt         || fail "pip install failed"
else
    # Nothing preinstalled: match the wheel to the driver rather than taking
    # whatever is newest, which is how the mismatch happened.
    DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    CUDA_TAG=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    WHEEL_INDEX="https://download.pytorch.org/whl/cu$(echo "$CUDA_TAG" | tr -d '.')"
    echo "no preinstalled torch; driver $DRIVER_CUDA reports CUDA $CUDA_TAG"
    echo "  installing torch from $WHEEL_INDEX"
    "$PY" -m pip install --no-cache-dir -q --index-url "$WHEEL_INDEX" torch torchaudio         || fail "installing a torch matching CUDA $CUDA_TAG failed"
    "$PY" -m pip install --no-cache-dir -q -r /tmp/requirements-pod.txt         || fail "pip install failed"
fi
"$PY" -c "import transformers, peft, soundfile, num2words; print('support libraries ok')" \
    || fail "support libraries failed to import"

step "waiting for CUDA"
# nvidia-smi can answer while CUDA context creation still fails with "CUDA
# unknown error": the driver is visible but the device nodes are not ready.
# It clears within seconds, so it is worth waiting for rather than failing on -
# the previous run died here two minutes into a pod that was about to be fine.
echo "  devices: $(ls /dev/nvidia* 2>/dev/null | tr '\n' ' ')"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

CUDA_READY=0
for attempt in $(seq 1 12); do
    if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        CUDA_READY=1
        echo "  ready after ${attempt} attempt(s)"
        break
    fi
    sleep 10
done

if [ "$CUDA_READY" -ne 1 ]; then
    echo "  still unavailable after two minutes; diagnostics:"
    "$PY" -c "import torch; torch.cuda.init()" 2>&1 | tail -3
    nvidia-smi 2>&1 | head -12
    fail "torch cannot see the GPU (nvidia-smi answers, CUDA init does not)"
fi

"$PY" -c "import torch; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('device:', torch.cuda.get_device_name(0)); \
print('bf16 native:', torch.cuda.get_device_capability(0)[0] >= 8)" \
    || fail "GPU check failed"

# --------------------------------------------------------------------------
phase models
step "link speed"
# Community hosts are individually owned machines, so the connection is an
# unknown until measured - and it decides whether rebuilding the corpus each
# run is cheaper than renting persistent storage at Secure Cloud rates.
LINK_START=$(date +%s)
LINK_URL="https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.safetensors"
LINK_BPS=$(curl -s -o /dev/null -w '%{speed_download}' --max-time 120 \
    -r 0-104857599 "$LINK_URL" 2>/dev/null || echo 0)
echo "  ${LINK_BPS} B/s on a 100 MB fetch, in $(( $(date +%s) - LINK_START ))s"
awk -v bps="$LINK_BPS" 'BEGIN {
    mb = bps / 1048576
    printf "  %.1f MB/s", mb
    if (mb < 10) print " - slow for a datacentre; the corpus rebuild will drag"
    else print " - fast enough to rebuild the corpus cheaply"
}'

step "model weights"
# Kept on the network volume when there is one, so a second run skips the
# 2 GB download entirely.
mkdir -p "$MODELS_DIR"
ln -sfn "$MODELS_DIR" "$REPO_DIR/pretrained_models"
"$PY" tools/fetch_models.py || fail "fetching model weights failed"

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
            yoda)      "$PY" tools/fetch_datasets.py yoda ;;
            narration) "$PY" tools/fetch_datasets.py narration ;;
            mana_hf)   "$PY" tools/fetch_datasets.py MahtaFetrat/Mana-TTS ;;
            youtube)   "$PY" tools/fetch_datasets.py youtube ;;
        esac || fail "downloading $source failed"
    done
    df -h "$WORKSPACE" | tail -1

    phase building_dataset
    step "building the corpus"
    # shellcheck disable=SC2086
    "$PY" tools/build_dataset.py --sources $SOURCES \
        --dedupe ${BUILD_ARGS:-} || fail "building the dataset failed"

    phase preprocessing
    step "preprocessing"
    "$PY" -u -m src.preprocess_ljspeech >>"$WORKSPACE/preprocess.log" 2>&1 \
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
"$PY" -u train.py \
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
