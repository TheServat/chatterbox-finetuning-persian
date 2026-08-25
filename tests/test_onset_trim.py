"""The onset trimmer must not eat the first phoneme.

Its first version keyed on zero-crossing rate, reasoning that the decoder emits
a noisy burst before speech begins. Sibilants are noisy too, so a clip opening
on /s/ - "سلام", which is how the Persian test sentence starts - lost its first
consonant, and the sample came back sounding unfinished. These tests pin the
distinction: leading silence goes, speech stays, however noisy it is.

    python tests/test_onset_trim.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import trim_onset_artifact  # noqa: E402

SR = 24000


def zcr(audio: np.ndarray) -> float:
    return float((np.diff(np.sign(audio)) != 0).mean())


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR))


def fricative(seconds: float, level: float = 0.02) -> np.ndarray:
    """Quiet and noisy, the way /s/ actually measures against a vowel."""
    return np.random.default_rng(0).normal(0, level, int(seconds * SR))


def vowel(seconds: float, level: float = 0.30, hz: float = 130.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return level * np.sin(2 * np.pi * hz * t)


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    return condition


def main() -> int:
    ok = True

    clip = np.concatenate([silence(0.10), fricative(0.09), vowel(0.50)])
    kept = trim_onset_artifact(clip, SR)
    cut_ms = (len(clip) - len(kept)) / SR * 1000
    ok &= check("silence + /s/ + vowel: only the silence goes",
                80 <= cut_ms <= 120, f"cut {cut_ms:.0f} ms")
    ok &= check("the fricative survives",
                zcr(kept[: int(0.08 * SR)]) > 0.3,
                f"ZCR {zcr(kept[: int(0.08 * SR)]):.2f}")

    straight = vowel(1.0)
    ok &= check("speech from the first sample: untouched",
                len(trim_onset_artifact(straight, SR)) == len(straight))

    long_pause = np.concatenate([silence(1.2), vowel(0.5)])
    ok &= check("a real pause, longer than any onset: kept",
                len(trim_onset_artifact(long_pause, SR)) == len(long_pause))

    ok &= check("empty input", trim_onset_artifact(np.array([]), SR).size == 0)
    ok &= check("all silence", trim_onset_artifact(silence(0.5), SR).size > 0)

    quiet_start = np.concatenate([silence(0.08), vowel(0.5, level=0.02)])
    kept = trim_onset_artifact(quiet_start, SR)
    ok &= check("a genuinely quiet speaker is not trimmed away",
                len(kept) >= int(0.5 * SR))

    sample = Path(__file__).resolve().parents[1] / \
        "chatterbox_output/inference_samples/checkpoint-500.wav"
    if sample.exists():
        import soundfile as sf

        raw, sr = sf.read(sample)
        cut_ms = (len(raw) - len(trim_onset_artifact(raw, sr))) / sr * 1000
        # The /s/ runs 110-180 ms in this clip; cutting into it is the bug.
        ok &= check("on the real sample, the cut stops before the /s/",
                    cut_ms <= 105, f"cut {cut_ms:.0f} ms, /s/ starts at 110")
    else:
        print("  --   real sample not on disk, skipped")

    print("\nall good" if ok else "\nFAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
