"""A checkpoint save must survive having generated a sample first.

T3.inference caches a `patched_model` around the same layers the trainer owns.
safetensors refuses to write two names pointing at one storage, so the next
save raises and the run ends - twice here, most recently at step 1500 after a
sample at 1250, eleven hours before anyone noticed.

The first fix deleted the attribute off the object the callback had to hand.
Under LoRA that object is a PeftModel, which forwards attribute reads to the
model it wraps but not deletes: the delete raised, the fallback set the name on
the wrapper, and the real cache stayed where it was. This reproduces that shape
- wrapper, PEFT-like forwarder, inner model - and saves through safetensors for
real, because only the save can tell the two outcomes apart.

    python tests/test_shared_tensor_save.py
"""

import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compat import drop_inference_cache  # noqa: E402


class Inner(nn.Module):
    """Stands in for T3: real layers, plus the cache inference leaves behind."""

    def __init__(self):
        super().__init__()
        self.tfmr = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))

    def run_inference(self):
        # What T3.inference does: wrap the very same layers and keep them.
        self.patched_model = nn.Sequential(*self.tfmr)
        self.compiled = True


class PeftLike(nn.Module):
    """Forwards reads to the model it wraps, as PEFT does. Deletes it does not."""

    def __init__(self, inner):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = inner

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model.model, name)


class Wrapper(nn.Module):
    def __init__(self, peft):
        super().__init__()
        self.t3 = peft


def save(module) -> tuple[bool, str]:
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp:
        try:
            save_file(module.state_dict(), str(Path(tmp) / "model.safetensors"),
                      metadata={"format": "pt"})
            return True, ""
        except Exception as exc:
            # The message opens with a blank line, so report the first line
            # that actually says something.
            said = [line.strip() for line in str(exc).splitlines() if line.strip()]
            return False, (said[0][:70] if said else type(exc).__name__)


def build():
    inner = Inner()
    return Wrapper(PeftLike(inner)), inner


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    return condition


def main() -> int:
    ok = True

    wrapper, inner = build()
    ok &= check("a clean model saves", save(wrapper)[0])

    inner.run_inference()
    saved, why = save(wrapper)
    ok &= check("after a sample, the save fails as it did in the run",
                not saved, why or "(it saved, so this no longer reproduces)")

    # The old fix, on the object the callback actually holds.
    peft = wrapper.t3
    try:
        delattr(peft, "patched_model")
    except Exception:
        setattr(peft, "patched_model", None)
    ok &= check("deleting the name off the PeftModel does not help",
                not save(wrapper)[0])

    wrapper, inner = build()
    inner.run_inference()
    removed = drop_inference_cache(wrapper.t3)
    ok &= check("drop_inference_cache finds it through the wrapper",
                any("patched_model" in r for r in removed), str(removed))
    saved, why = save(wrapper)
    ok &= check("and the save succeeds", saved, why)

    # Called from the top, and twice, and on a model that never sampled.
    wrapper, inner = build()
    inner.run_inference()
    drop_inference_cache(wrapper)
    ok &= check("works from the outermost module too", save(wrapper)[0])
    ok &= check("a second call is harmless", drop_inference_cache(wrapper) == [])
    wrapper, _ = build()
    ok &= check("no cache to remove: no-op", drop_inference_cache(wrapper) == [])

    # The weights themselves must be untouched.
    wrapper, inner = build()
    before = inner.tfmr[0].weight.clone()
    inner.run_inference()
    drop_inference_cache(wrapper)
    ok &= check("real weights survive", torch.equal(before, inner.tfmr[0].weight))
    ok &= check("and are still in the state dict",
                any("tfmr.0.weight" in k for k in wrapper.state_dict()))

    print("\nall good" if ok else "\nFAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
