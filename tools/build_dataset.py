"""Turn the Persian corpora into the LJSpeech layout this project trains on.

`src/preprocess_ljspeech.py` expects exactly one thing:

    MyTTSDataset/metadata.csv    id|raw-text|normalized-text, no header
    MyTTSDataset/wavs/<id>.wav

so that is what gets written, rather than teaching the trainer about a new
format. Sources differ in shape - a CSV next to a wav directory, a pipe-
separated manifest, parquet shards with the audio embedded - and each gets a
small reader that yields the same `Clip` record.

Two things keep this cheap. Clips that are already wav files on the same volume
are hard-linked, not copied, so 47,705 ManaTTS files cost zero extra bytes.
Clips embedded in parquet are decoded straight from memory, so the 46 GB
YouTube corpus never gets expanded to wav on disk.

Quality filtering is not optional for TTS. A single mistranscribed or clipped
utterance teaches a mispronunciation that survives the rest of training, and
these corpora are ASR-derived. Every clip must pass duration, text-length,
Persian-ratio and (where the corpus provides a score) perceptual-quality
thresholds before it is written.

    python tools/build_dataset.py --list
    python tools/build_dataset.py --sources mana narration yoda
    python tools/build_dataset.py --sources mana --limit 200 --out MyTTSDataset
    python tools/build_dataset.py --sources yoda --min-quality 3.2 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import unicodedata
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.persian.normalize import (  # noqa: E402
    normalize,
    persian_ratio,
    unknown_characters,
)

TOKENIZER = ROOT / "pretrained_models" / "tokenizer_fa.json"

DEFAULT_OUT = ROOT / "MyTTSDataset"


def show(path: Path) -> str:
    """Project-relative when it can be, absolute otherwise (e.g. a scratch dir)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

# Text that reaches metadata.csv must survive csv.QUOTE_NONE with '|' as the
# separator, so these characters cannot appear in it at all.
_UNSAFE = re.compile(r"[|\r\n\t]+")


@dataclass
class Clip:
    """One utterance, however its corpus happens to store the audio."""

    clip_id: str
    text: str
    speaker: str = "unknown"
    audio_path: Path | None = None     # already a file: hard-link it
    audio_bytes: bytes | None = None   # embedded container: decode and write
    audio_array: object | None = None  # already-decoded samples
    sample_rate: int | None = None     # required alongside audio_array
    quality: float | None = None       # dnsmos / MOS, where the corpus has one
    duration: float | None = None


@dataclass
class Stats:
    seen: int = 0
    written: int = 0
    rejected: Counter = field(default_factory=Counter)
    linked: int = 0
    copied: int = 0
    decoded: int = 0
    seconds: float = 0.0

    def reject(self, reason: str) -> None:
        self.rejected[reason] += 1


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------

def read_mana(root: Path) -> Iterator[Clip]:
    """ManaTTS: raw.csv (audio_path,text,speaker_id,emotion) next to wav/."""
    csv_path = root / "raw.csv"
    wav_dir = root / "wav"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            # The recorded audio_path is from the machine that built the corpus;
            # only the filename is portable.
            name = Path(str(row["audio_path"]).replace("\\", "/")).name
            yield Clip(
                clip_id=f"mana_{Path(name).stem}",
                text=row["text"],
                speaker=row.get("speaker_id") or "mana",
                audio_path=wav_dir / name,
            )


def read_narration(root: Path) -> Iterator[Clip]:
    """PERSIAN_FARSI_NARRATION: <split>_metadata.csv (id|text) + <split>/audio/."""
    for split in ("train", "test"):
        manifest = root / f"{split}_metadata.csv"
        audio_dir = root / split / "audio"
        if not manifest.exists():
            continue
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                clip_id, _, text = line.partition("|")
                yield Clip(
                    clip_id=f"narr_{clip_id}",
                    text=text,
                    # One speaker per recording session; the id prefix is the
                    # closest thing the corpus gives us to a speaker label.
                    speaker=clip_id.rsplit("_part", 1)[0],
                    audio_path=audio_dir / f"{clip_id}.wav",
                )


def _parquet_clips(
    root: Path,
    prefix: str,
    text_column: str,
    audio_column: str,
    speaker_column: str | None = None,
    quality_column: str | None = None,
    key_column: str | None = None,
    rate_column: str | None = None,
    keep: "callable | None" = None,
) -> Iterator[Clip]:
    """Stream clips out of HuggingFace parquet shards, batch by batch."""
    import pyarrow.parquet as pq

    shards = sorted(root.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {root}")

    wanted = [c for c in (text_column, audio_column, speaker_column,
                          quality_column, key_column, rate_column) if c]
    if keep is not None:
        # Filter columns have to be read as well, or the predicate sees nothing.
        wanted += [c for c in getattr(keep, "columns", ())]

    index = 0
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        available = set(parquet.schema_arrow.names)
        columns = [c for c in wanted if c in available]
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                index += 1
                if keep is not None and not keep(row):
                    continue
                audio = row.get(audio_column)
                # Three shapes occur in the wild: a HuggingFace Audio struct, a
                # bare encoded blob, and - in ManaTTS - already-decoded samples
                # with the rate in a separate column.
                data = array = None
                if isinstance(audio, dict):
                    data = audio.get("bytes")
                    array = audio.get("array")
                elif isinstance(audio, (bytes, bytearray)):
                    data = audio
                elif isinstance(audio, (list, tuple)):
                    array = audio
                if data is None and array is None:
                    continue
                key = row.get(key_column) if key_column else None
                yield Clip(
                    clip_id=f"{prefix}_{key or index:06}"
                    if key is None
                    else f"{prefix}_{key}",
                    text=row.get(text_column) or "",
                    speaker=str(row.get(speaker_column) or "unknown")
                    if speaker_column
                    else "unknown",
                    audio_bytes=data,
                    audio_array=array,
                    sample_rate=row.get(rate_column) if rate_column else None,
                    quality=row.get(quality_column) if quality_column else None,
                )


def read_yoda(root: Path) -> Iterator[Clip]:
    """YodaLingua-Farsi: parquet with mp3 bytes, speaker_id and a DNSMOS score."""
    return _parquet_clips(
        root,
        prefix="yoda",
        text_column="text",
        audio_column="mp3",
        speaker_column="speaker_id",
        quality_column="dnsmos",
        key_column="__key__",
    )


def _mana_hf_keep(row: dict) -> bool:
    """ManaTTS ships its own alignment quality, so use it.

    `match_quality` HIGH means the transcript matched the audio confidently and
    `CER` is the character error rate of that match. Anything looser is a
    transcript that does not quite say what the speaker said - the exact kind of
    pair that teaches a mispronunciation.
    """
    if row.get("match_quality") not in (None, "HIGH"):
        return False
    cer = row.get("CER")
    return cer is None or cer <= 0.05


_mana_hf_keep.columns = ("match_quality", "CER")


def read_mana_hf(root: Path) -> Iterator[Clip]:
    """ManaTTS as published on HuggingFace: parquet with decoded samples."""
    return _parquet_clips(
        root,
        prefix="mana",
        text_column="transcript",
        audio_column="audio",
        key_column="file_name",
        rate_column="sample_rate",
        keep=_mana_hf_keep,
    )


def read_youtube(root: Path) -> Iterator[Clip]:
    """Persian-Farsi-Speech: ASR-chunked YouTube, MOS-scored, no speaker labels."""
    return _parquet_clips(
        root,
        prefix="yt",
        text_column="sentence",
        audio_column="audio",
        quality_column="mos_ovr",
    )


SOURCES = {
    "mana": {
        "reader": read_mana,
        "dir": "dataset/persian/manatts_full",
        "default": True,
        "min_quality": None,
        "note": "ManaTTS, CC0, single clean speaker at 44.1 kHz. Stability anchor.",
    },
    "narration": {
        "reader": read_narration,
        "dir": "dataset/persian/PERSIAN_FARSI_NARRATION",
        "default": True,
        "min_quality": None,
        "note": "Narration, 22.05 kHz, ~3k clips.",
    },
    "yoda": {
        "reader": read_yoda,
        "dir": "dataset/persian/YodaLingua-Farsi",
        "default": True,
        "min_quality": 3.0,
        "note": "YodaLingua, CC-BY-4.0, 72 h over 678 speakers at 24 kHz. Primary corpus.",
    },
    "mana_hf": {
        "reader": read_mana_hf,
        "dir": "dataset/persian/Mana-TTS",
        "default": False,
        "min_quality": None,
        "note": "ManaTTS straight from HuggingFace (33 GB of parquet). Use this "
                "when the local wav copy is not available - e.g. on a rented GPU.",
    },
    "youtube": {
        "reader": read_youtube,
        "dir": "dataset/persian/Persian-Farsi-Speech",
        "default": False,
        "min_quality": 3.2,
        "note": "Persian-Farsi-Speech. 16 kHz and ASR-derived, so noisier; "
                "off by default, and worth a high --min-quality when used.",
    },
}


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def clean_text(text: str) -> str:
    return _UNSAFE.sub(" ", str(text)).strip()


class UnknownTextDetector:
    """Finds text the tokenizer cannot represent, the way the tokenizer sees it.

    Two-tier by design, because the two failure modes need opposite treatment:

      * A character that is *silent* - emoji, bullets, decorative symbols - is
        removed by `normalize()`, and the clip is kept. Rejecting it would throw
        away good audio over a mark nobody pronounced.
      * A character that is *pronounced* but unknown - Cyrillic, CJK - has to
        take the clip with it. Stripping it would leave audio whose words are
        missing from the transcript, which is how a model learns to say things
        its input never contained.

    So this catches the second kind, and it does so exactly. A cheap
    per-character membership test is only a first pass: MTLTokenizer applies
    NFKD before encoding, so a character absent from the vocabulary can still
    decompose into pieces that are present. Anything the fast test flags is
    confirmed by actually encoding it and looking for [UNK].
    """

    def __init__(self, tokenizer_path: Path):
        from tokenizers import Tokenizer

        data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        self.vocabulary = set(data["model"]["vocab"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def unknown(self, text: str) -> set[str]:
        """Characters of `text` that would reach the model as [UNK]."""
        suspects = unknown_characters(text, self.vocabulary)
        if not suspects:
            return set()

        confirmed = set()
        for char in suspects:
            prepared = unicodedata.normalize("NFKD", char.lower())
            if "[UNK]" in self.tokenizer.encode(prepared).tokens:
                confirmed.add(char)
        return confirmed


def load_vocabulary() -> UnknownTextDetector | None:
    if not TOKENIZER.exists():
        return None
    return UnknownTextDetector(TOKENIZER)


def link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def write_decoded(clip: "Clip", destination: Path, target_sr: int | None) -> float:
    """Write a clip's embedded audio to a mono wav. Returns duration in seconds."""
    import numpy as np
    import soundfile as sf

    if clip.audio_bytes is not None:
        audio, sample_rate = sf.read(
            io.BytesIO(clip.audio_bytes), dtype="float32", always_2d=True
        )
        audio = audio.mean(axis=1)
    else:
        # Already-decoded samples, so the rate has to come from the corpus.
        if not clip.sample_rate:
            raise ValueError("decoded samples arrived without a sample rate")
        audio = np.asarray(clip.audio_array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        sample_rate = int(clip.sample_rate)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            # Integer PCM stored as numbers; scale to the float range soundfile
            # expects rather than writing something that clips everywhere.
            audio = audio / max(peak, 1.0)

    if target_sr and sample_rate != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    sf.write(destination, audio, sample_rate, subtype="PCM_16")
    return len(audio) / sample_rate


def probe_duration(path: Path) -> float | None:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return info.duration
    except Exception:
        return None


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    wav_dir = out_dir / "wavs"
    metadata_path = out_dir / "metadata.csv"

    if not args.dry_run:
        wav_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = load_vocabulary()
    if vocabulary is None:
        print("WARNING: no tokenizer in pretrained_models/, skipping the "
              "out-of-vocabulary check. Run tools/fetch_models.py first.")

    rows: list[tuple[str, str, str]] = []
    seen_text: set[str] = set()
    per_source: dict[str, Stats] = {}
    speakers: Counter = Counter()

    for name in args.sources:
        spec = SOURCES[name]
        root = ROOT / spec["dir"]
        stats = Stats()
        per_source[name] = stats

        if not root.exists():
            print(f"\n{name}: {spec['dir']} is missing - run tools/fetch_datasets.py")
            continue

        threshold = (
            args.min_quality if args.min_quality is not None else spec["min_quality"]
        )
        print(f"\n{name}: reading {spec['dir']}"
              f"{f' (min quality {threshold})' if threshold else ''}")

        for clip in spec["reader"](root):
            stats.seen += 1
            if args.limit and stats.written >= args.limit:
                break

            if threshold is not None and clip.quality is not None:
                if clip.quality < threshold:
                    stats.reject("quality")
                    continue

            raw = clean_text(clip.text)
            if len(raw) < args.min_chars:
                stats.reject("text too short")
                continue

            normalized = clean_text(normalize(raw))
            if not normalized:
                stats.reject("empty after normalisation")
                continue
            if len(normalized) > args.max_chars:
                stats.reject("text too long")
                continue
            if persian_ratio(normalized) < args.min_persian:
                stats.reject("not Persian enough")
                continue

            if vocabulary is not None:
                unknown = vocabulary.unknown(normalized)
                if unknown:
                    stats.reject(
                        "out-of-vocabulary "
                        + "".join(sorted(f"U+{ord(c):04X}" for c in unknown)[:3])
                    )
                    continue

            if args.dedupe:
                if normalized in seen_text:
                    stats.reject("duplicate text")
                    continue
                seen_text.add(normalized)

            destination = wav_dir / f"{clip.clip_id}.wav"

            if clip.audio_path is not None:
                if not clip.audio_path.exists():
                    stats.reject("audio missing")
                    continue
                duration = probe_duration(clip.audio_path)
                if duration is None:
                    stats.reject("unreadable audio")
                    continue
                if not args.min_seconds <= duration <= args.max_seconds:
                    stats.reject("duration")
                    continue
                if args.dry_run:
                    stats.linked += 1
                elif link_or_copy(clip.audio_path, destination) == "linked":
                    stats.linked += 1
                else:
                    stats.copied += 1
            else:
                if args.dry_run:
                    # Decoding is the expensive half; skip it when only counting.
                    duration = 0.0
                else:
                    try:
                        duration = write_decoded(clip, destination, args.target_sr)
                    except Exception as exc:
                        stats.reject(f"decode failed ({type(exc).__name__})")
                        continue
                    if not args.min_seconds <= duration <= args.max_seconds:
                        destination.unlink(missing_ok=True)
                        stats.reject("duration")
                        continue
                stats.decoded += 1

            stats.seconds += duration or 0.0
            speakers[f"{name}:{clip.speaker}"] += 1
            rows.append((clip.clip_id, raw, normalized))
            stats.written += 1

            if stats.written % 2000 == 0:
                print(f"    {stats.written:,} written ({stats.seconds / 3600:.1f} h)")

        moved = ", ".join(
            f"{count:,} {label}"
            for label, count in (
                ("linked", stats.linked),
                ("copied", stats.copied),
                ("decoded", stats.decoded),
            )
            if count
        )
        print(f"  {stats.written:,} kept of {stats.seen:,} "
              f"({stats.seconds / 3600:.1f} h, {moved or 'nothing written'})")
        for reason, count in stats.rejected.most_common():
            print(f"    rejected {count:>7,}  {reason}")

    if not rows:
        print("\nNothing to write.")
        return 1

    total_hours = sum(s.seconds for s in per_source.values()) / 3600
    print(f"\n{'=' * 66}")
    print(f"total {len(rows):,} clips, {total_hours:.1f} h, "
          f"{len(speakers):,} speaker(s)")

    if args.dry_run:
        print("--dry-run: nothing written.")
        return 0

    # Written by hand rather than through csv.writer: the loader reads this with
    # quoting=QUOTE_NONE, and clean_text has already guaranteed no field can
    # contain a pipe or a newline, so joining is both simpler and exact.
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write("|".join(row) + "\n")

    summary = {
        "clips": len(rows),
        "hours": round(total_hours, 2),
        "speakers": len(speakers),
        "sources": {
            name: {
                "written": s.written,
                "seen": s.seen,
                "linked": s.linked,
                "copied": s.copied,
                "decoded": s.decoded,
                "hours": round(s.seconds / 3600, 2),
                "rejected": dict(s.rejected),
            }
            for name, s in per_source.items()
        },
        "filters": {
            "min_seconds": args.min_seconds,
            "max_seconds": args.max_seconds,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_persian": args.min_persian,
            "min_quality": args.min_quality,
            "dedupe": args.dedupe,
            "target_sr": args.target_sr,
        },
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"wrote {show(metadata_path)}")
    print(f"wrote {show(out_dir / 'dataset_summary.json')}")
    print(f"audio in {show(wav_dir)}")
    return 0


def cmd_list() -> int:
    print(f"{'source':<12} {'default':<8} {'present':<8} note")
    for name, spec in SOURCES.items():
        present = "yes" if (ROOT / spec["dir"]).exists() else "no"
        print(f"{name:<12} {'on' if spec['default'] else 'off':<8} {present:<8} "
              f"{spec['note']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="show available sources")
    parser.add_argument(
        "--sources", nargs="+", choices=list(SOURCES),
        help="sources to include (default: those marked on)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument("--limit", type=int, help="stop after N clips per source")
    parser.add_argument("--dry-run", action="store_true", help="count without writing")

    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument(
        "--max-seconds", type=float, default=20.0,
        help="longer clips blow up activation memory and are rare in training data",
    )
    parser.add_argument("--min-chars", type=int, default=5)
    parser.add_argument(
        "--max-chars", type=int, default=280,
        help="roughly the 256-token text limit in src/config.py",
    )
    parser.add_argument(
        "--min-persian", type=float, default=0.6,
        help="minimum share of letters in Persian script",
    )
    parser.add_argument(
        "--min-quality", type=float,
        help="override the per-source perceptual-quality threshold",
    )
    parser.add_argument(
        "--dedupe", action="store_true",
        help="drop clips whose normalised text was already seen",
    )
    parser.add_argument(
        "--target-sr", type=int,
        help="resample decoded audio to this rate (default: keep the source rate)",
    )
    args = parser.parse_args()

    if args.list:
        return cmd_list()

    if not args.sources:
        args.sources = [n for n, s in SOURCES.items() if s["default"]]

    return build(args)


if __name__ == "__main__":
    sys.exit(main())
