"""Seeding the evaluation sample must not disturb training.

Generation is stochastic - do_sample=True at temperature 0.8 - so without a
fixed seed each checkpoint is a different draw, and two samples say nothing
about whether the model improved. A word came out right at step 1000 and wrong
at 1500 with no way to tell the model from the dice.

Seeding globally mid-run would be worse than the problem: the training stream
would jump, changing data order and dropout from that point on. So the sample
saves the RNG state, seeds, and puts it back. This checks both halves.

    python tests/test_sample_seeding.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CALLBACK = Path(__file__).resolve().parents[1] / "src/inference_callback.py"


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    return condition


def sample_like(seed: int):
    """The save/seed/restore dance the callback performs, in miniature."""
    state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        return torch.randn(4)
    finally:
        torch.set_rng_state(state)


def main() -> int:
    ok = True

    # The same seed twice gives the same sample.
    ok &= check("a seeded sample is reproducible",
                torch.equal(sample_like(1234), sample_like(1234)))
    ok &= check("a different seed gives a different one",
                not torch.equal(sample_like(1234), sample_like(99)))

    # And the training stream carries on as if nothing happened.
    torch.manual_seed(7)
    expected = [torch.randn(3) for _ in range(4)]

    torch.manual_seed(7)
    actual = [torch.randn(3)]
    sample_like(1234)                     # a sample happens here
    actual += [torch.randn(3) for _ in range(3)]

    ok &= check("training randomness is untouched by a sample",
                all(torch.equal(a, b) for a, b in zip(expected, actual)))

    # Two samples in a row, as save_steps produces, stay identical.
    torch.manual_seed(7)
    torch.randn(3)
    first = sample_like(1234)
    torch.randn(3)
    second = sample_like(1234)
    ok &= check("consecutive checkpoints are compared on equal terms",
                torch.equal(first, second))

    source = CALLBACK.read_text(encoding="utf-8")
    ok &= check("the callback seeds before generating",
                "torch.manual_seed(" in source)
    ok &= check("and restores what it saved",
                "torch.set_rng_state(rng_state)" in source
                and "set_rng_state_all(cuda_rng_state)" in source)

    from src.config import TrainConfig
    ok &= check("the seed is configurable",
                isinstance(getattr(TrainConfig(), "inference_seed", None), int),
                f"inference_seed={TrainConfig().inference_seed}")

    print("\nall good" if ok else "\nFAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
