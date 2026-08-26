"""Tell a usable checkpoint from one a power cut ate.

A checkpoint written while the mains fail can come back the right size with a
region of it never flushed - zeros where weights should be. That is not a
crash: training resumes from it happily, the loss sits at ln(2455) + ln(8194) =
16.82 because the model now answers uniformly, and every later checkpoint
inherits the damage because frozen weights are copied forward rather than
retrained. It cost three hundred and fifty steps here before anyone noticed,
and it would have cost the whole run.

The tell is specific. Under LoRA the base weights never change, so a base
tensor of all zeros cannot be training - it can only be a bad write. Adapter
tensors are exempt: `lora_B` is deliberately zero at initialisation.

    python tools/verify_checkpoint.py                     check them all
    python tools/verify_checkpoint.py chatterbox_output/checkpoint-1800
    python tools/verify_checkpoint.py --newest-good       print one path
    python tools/verify_checkpoint.py --quarantine        move the bad aside
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import safe_open

ROOT = Path(__file__).resolve().parents[1]
QUARANTINE = "corrupt"


def is_adapter(name: str) -> bool:
    """Adapter tensors may legitimately be all zeros; base weights may not."""
    return "lora_A" in name or "lora_B" in name


def inspect(checkpoint: Path) -> dict:
    """What is wrong with this checkpoint, if anything."""
    report = {
        "path": checkpoint,
        "readable": False,
        "tensors": 0,
        "zeroed_base": [],
        "non_finite": [],
        "missing": [],
    }

    for required in ("model.safetensors", "optimizer.pt", "trainer_state.json"):
        if not (checkpoint / required).exists():
            report["missing"].append(required)
    if report["missing"]:
        return report

    try:
        with safe_open(checkpoint / "model.safetensors", framework="pt") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                report["tensors"] += 1
                if not torch.isfinite(tensor).all():
                    report["non_finite"].append(name)
                elif not is_adapter(name) and float(tensor.abs().max()) == 0.0:
                    report["zeroed_base"].append(name)
        report["readable"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"

    return report


def is_good(report: dict) -> bool:
    return (
        report["readable"]
        and not report["missing"]
        and not report["zeroed_base"]
        and not report["non_finite"]
    )


def describe(report: dict) -> str:
    if report["missing"]:
        return f"incomplete - no {', '.join(report['missing'])}"
    if not report["readable"]:
        return report.get("error", "unreadable")
    faults = []
    if report["zeroed_base"]:
        layers = sorted({
            match.group(1)
            for name in report["zeroed_base"]
            if (match := re.search(r"layers\.(\d+)\.", name))
        }, key=int)
        where = f" (layer{'s' if len(layers) > 1 else ''} {', '.join(layers)})" if layers else ""
        faults.append(f"{len(report['zeroed_base'])} base weights zeroed{where}")
    if report["non_finite"]:
        faults.append(f"{len(report['non_finite'])} non-finite")
    return ", ".join(faults) if faults else f"ok ({report['tensors']} tensors)"


def checkpoints(output_dir: Path) -> list[Path]:
    numbered = [
        (int(match.group(1)), path)
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and (match := re.fullmatch(r"checkpoint-(\d+)", path.name))
    ]
    return [path for _, path in sorted(numbered)]


def newest_good(output_dir: Path) -> Path | None:
    for path in reversed(checkpoints(output_dir)):
        if is_good(inspect(path)):
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", help="checkpoints; default is all of them")
    parser.add_argument("--output-dir", default="./chatterbox_output")
    parser.add_argument("--newest-good", action="store_true",
                        help="print the newest usable checkpoint and nothing else")
    parser.add_argument("--quarantine", action="store_true",
                        help=f"move damaged checkpoints into {QUARANTINE}/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.newest_good:
        found = newest_good(output_dir)
        if not found:
            print("no usable checkpoint", file=sys.stderr)
            return 1
        print(found)
        return 0

    targets = [Path(p) for p in args.paths] or checkpoints(output_dir)
    if not targets:
        print(f"no checkpoints under {output_dir}")
        return 0

    damaged = []
    for path in targets:
        report = inspect(path)
        good = is_good(report)
        print(f"  {'ok  ' if good else 'BAD '} {path.name:<20} {describe(report)}")
        if not good:
            damaged.append(path)

    if damaged and args.quarantine:
        destination = output_dir / QUARANTINE
        destination.mkdir(exist_ok=True)
        print()
        for path in damaged:
            target = destination / path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(path), str(target))
            print(f"  moved {path.name} -> {destination}/")

    if damaged and not args.quarantine:
        print(f"\n{len(damaged)} damaged. --quarantine moves them out of the way "
              "so resume cannot pick one.")

    return 1 if damaged else 0


if __name__ == "__main__":
    sys.exit(main())
