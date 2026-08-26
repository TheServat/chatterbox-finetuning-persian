"""Score the training samples by listening back to them.

Judging progress by ear does not scale and does not compare: generation is
stochastic, so one draw says nothing, and three draws per checkpoint is already
more listening than anyone will do for forty checkpoints. Noise floor is
measurable but only answers whether the clip is clean, not whether it said the
words.

So the audio is transcribed and compared with the sentence it was asked to say.
Wrong or missing words show up as distance; a clip that stopped after "سلام"
scores terribly, which is exactly what it deserved.

Two things make the comparison mean something. Both sides go through the
project's own normaliser, so digits and punctuation cannot skew it. Then both
are folded phonetically, because Persian writes /s/ three ways and /z/ four -
a transcriber writing سدای for صدای has not found a mispronunciation, and
counting it as one would bury the real errors in noise.

What it cannot see: ezafe. The linking vowel is not written, so it is absent
from the transcript as well, and a model that drops it scores the same as one
that does not. That still needs an ear.

Whisper also makes its own mistakes, so the absolute number means little. The
comparison between checkpoints is what to read, and only with the same model
and the same sentence on both sides.

    python tools/score_samples.py                     every sample on disk
    python tools/score_samples.py --checkpoint 2300
    python tools/score_samples.py --model medium      slower, more accurate
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TrainConfig  # noqa: E402
from src.persian.normalize import normalize, phonetic_fold  # noqa: E402

SAMPLE_NAME = re.compile(r"checkpoint-(\d+)(?:_(\d+))?\.wav$")


def edit_distance(a: str, b: str) -> int:
    """Levenshtein, one row at a time - the sentences here are short."""
    if a == b:
        return 0
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def comparable(text: str) -> str:
    """Normalised, folded to sound, and stripped of spacing."""
    folded = phonetic_fold(normalize(text))
    return re.sub(r"[\s‌]+", "", folded)


def rates(reference: str, heard: str) -> tuple[float, float]:
    """(character error rate, word error rate), both against the reference."""
    ref_chars, heard_chars = comparable(reference), comparable(heard)
    cer = edit_distance(ref_chars, heard_chars) / max(len(ref_chars), 1)

    ref_words = phonetic_fold(normalize(reference)).split()
    heard_words = phonetic_fold(normalize(heard)).split()
    wer = edit_distance(ref_words, heard_words) / max(len(ref_words), 1)
    return cer, wer


def samples(directory: Path, only: int | None) -> dict[int, list[Path]]:
    found: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(directory.glob("checkpoint-*.wav")):
        if "trimmed" in path.name:
            continue
        match = SAMPLE_NAME.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if only is None or step == only:
            found[step].append(path)
    return dict(sorted(found.items()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples-dir",
                        default="./chatterbox_output/inference_samples")
    parser.add_argument("--checkpoint", type=int, help="score only this step")
    parser.add_argument("--model", default="small",
                        help="whisper size: tiny, base, small, medium, large-v3")
    parser.add_argument("--threads", type=int, default=2,
                        help="CPU threads; kept low so training keeps its cores")
    parser.add_argument("--text", help="what the samples were asked to say")
    parser.add_argument("--show-transcripts", action="store_true")
    args = parser.parse_args()

    directory = Path(args.samples_dir)
    if not directory.is_dir():
        print(f"no samples at {directory}")
        return 1

    by_step = samples(directory, args.checkpoint)
    if not by_step:
        print("nothing to score")
        return 1

    reference = args.text or TrainConfig().inference_test_text
    print(f"asked for: {reference}")
    print(f"scoring {sum(len(v) for v in by_step.values())} clips across "
          f"{len(by_step)} checkpoint(s) with whisper-{args.model}, on the CPU\n")

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type="int8",
                         cpu_threads=args.threads)

    print(f"{'step':>7} {'draw':>5} {'CER':>7} {'WER':>7}")
    trend = []
    for step, paths in by_step.items():
        scored = []
        for path in paths:
            segments, _ = model.transcribe(str(path), language="fa", beam_size=5)
            heard = " ".join(segment.text.strip() for segment in segments)
            cer, wer = rates(reference, heard)
            scored.append((cer, wer, path, heard))
            draw = SAMPLE_NAME.search(path.name).group(2) or "1"
            print(f"{step:>7} {draw:>5} {cer:7.3f} {wer:7.3f}")
            if args.show_transcripts:
                print(f"        heard: {heard}")

        scored.sort()
        best_cer, _, best_path, best_heard = scored[0]
        median_cer = sorted(c for c, *_ in scored)[len(scored) // 2]
        trend.append((step, median_cer, best_cer, best_path.name, len(scored)))
        print(f"{'':>7} {'best':>5} {best_cer:7.3f}   {best_path.name}")
        if not args.show_transcripts:
            print(f"        heard: {best_heard}")
        print()

    if len(trend) > 1:
        print(f"{'step':>7} {'draws':>6} {'median CER':>11} {'best CER':>9}   best draw")
        for step, median_cer, best_cer, name, draws in trend:
            median = f"{median_cer:11.3f}" if draws >= 3 else f"{'-':>11}"
            print(f"{step:>7} {draws:>6} {median} {best_cer:9.3f}   {name}")

        # A median over one draw is that draw, and setting it beside a median
        # over three reads sampling luck as progress - the very mistake this
        # tool exists to stop. Only like counts are compared.
        comparable_rows = [row for row in trend if row[4] >= 3]
        if len(comparable_rows) > 1:
            first, last = comparable_rows[0], comparable_rows[-1]
            direction = ("improving" if last[1] < first[1]
                         else "worse" if last[1] > first[1] else "unchanged")
            print(f"\nmedian over 3 draws: {first[1]:.3f} at step {first[0]:,} "
                  f"-> {last[1]:.3f} at step {last[0]:,}: {direction}")
        else:
            print(f"\nOnly {len(comparable_rows)} checkpoint(s) have three "
                  "draws, so there is no median trend to read yet.")

        print(f"best of each: {trend[0][2]:.3f} at step {trend[0][0]:,} -> "
              f"{trend[-1][2]:.3f} at step {trend[-1][0]:,}")
        print("\nRead the trend, not the number: whisper's own errors sit in "
              "every row equally, and ezafe is invisible to all of them.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
