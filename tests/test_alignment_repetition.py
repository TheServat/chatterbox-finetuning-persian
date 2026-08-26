"""The analyzer must not end generation on two ordinary repeated tokens.

AlignmentStreamAnalyzer forces an EOS when the last two speech tokens match,
and chatterbox-tts 0.1.7 ships that test with its `self.complete and` guard
commented out. Ordinary speech repeats tokens constantly - 85% of the clips in
this corpus do, first occurrence a median of 31 tokens in - so generation was
being cut about a second after it began: 0.8 s at step 1250, and 6.2 s at step
1000 only because that draw landed in the tail.

    python tests/test_alignment_repetition.py
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import compat  # noqa: E402
from src.chatterbox_.models.t3.inference.alignment_stream_analyzer import (  # noqa: E402
    AlignmentStreamAnalyzer,
)

VENDORED = ROOT / "src/chatterbox_/models/t3/inference/alignment_stream_analyzer.py"


def upstream_rule(self, logits, next_token=None, *args, **kwargs):
    """Upstream's bookkeeping and repetition test, standing in for the real step."""
    self.generated_tokens.append(next_token)
    if len(self.generated_tokens) > 8:
        self.generated_tokens = self.generated_tokens[-8:]
    fired = (
        len(self.generated_tokens) >= 3
        and len(set(self.generated_tokens[-2:])) == 1
    )
    self.fired = self.fired or fired
    return fired


def run(tokens, complete: bool) -> bool:
    instance = AlignmentStreamAnalyzer.__new__(AlignmentStreamAnalyzer)
    instance.complete = complete
    instance.generated_tokens = []
    instance.fired = False
    for token in tokens:
        AlignmentStreamAnalyzer.step(instance, None, token)
    return instance.fired


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    return condition


def main() -> int:
    ok = True
    source = VENDORED.read_text(encoding="utf-8")

    # Canary: if upstream ever restores this itself, the patch can go.
    ok &= check("upstream still ships the guard commented out",
                "# self.complete and" in source,
                "(if this fails, drop the patch)")
    ok &= check("and still tests only the last two tokens",
                "len(set(self.generated_tokens[-2:])) == 1" in source)

    compat.apply()
    ok &= check("the patch is installed",
                getattr(AlignmentStreamAnalyzer.step, "_repetition_needs_completion", False))
    ok &= check("applying it twice changes nothing",
                compat._patch_alignment_repetition_guard() == [])

    saved = AlignmentStreamAnalyzer.step
    AlignmentStreamAnalyzer.step = upstream_rule
    try:
        compat._patch_alignment_repetition_guard()

        random.seed(0)
        speech = [random.randint(1, 6000) for _ in range(200)]
        speech[30] = speech[29]          # an ordinary repeat, a second in
        speech[120] = speech[119]

        ok &= check("mid-sentence repeats do not end generation",
                    not run(speech, complete=False))
        ok &= check("a run of one token still does not, while text remains",
                    not run([42] * 40, complete=False))

        ok &= check("once the text is spoken, repetition is caught again",
                    run([42, 42, 42, 42], complete=True))
        ok &= check("and varied tokens after completion are left alone",
                    not run([1, 2, 3, 4, 5, 6], complete=True))
    finally:
        AlignmentStreamAnalyzer.step = saved

    # What the corpus actually looks like, so nobody removes this as unnecessary.
    preprocessed = ROOT / "MyTTSDataset/preprocess"
    files = list(preprocessed.glob("*.pt"))[:60] if preprocessed.is_dir() else []
    if files:
        import torch

        with_repeat = 0
        for path in files:
            tokens = torch.load(path, map_location="cpu", weights_only=False)["speech_tokens"].tolist()
            if any(tokens[i] == tokens[i - 1] for i in range(1, len(tokens))):
                with_repeat += 1
        share = with_repeat / len(files)
        ok &= check("real clips do repeat tokens, so the rule would fire on them",
                    share > 0.5, f"{share * 100:.0f}% of {len(files)} clips")
    else:
        print("  --   no preprocessed corpus here, corpus check skipped")

    print("\nall good" if ok else "\nFAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
