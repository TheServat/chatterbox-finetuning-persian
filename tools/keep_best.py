"""Save the checkpoints worth keeping before rotation deletes them.

transformers keeps the last `save_total_limit` checkpoints and deletes the rest
by step number, which has nothing to do with quality. Saving every 200 steps
with a limit of five leaves a thousand steps of history, and the best
checkpoint measured on this run - step 7200, noise floor 0.00141 - was gone
before anyone thought to keep it. The final checkpoint is not reliably the best
either: on the previous run step 3500 beat the last one on every measure taken.

So the ones that score well are copied into `chatterbox_output/keep/`, where
the trainer does not look and rotation cannot reach.

Ranking uses the noise-floor median each checkpoint logged during training -
the one figure measured for every checkpoint without anyone listening. It says
how clean a clip is, not how well it reads, so this preserves candidates rather
than crowning a winner.

    python tools/keep_best.py                 what is measured, and what is at risk
    python tools/keep_best.py --keep 3        copy the three cleanest
    python tools/keep_best.py --step 7200     copy one by name
    python tools/keep_best.py --watch         keep doing it while training runs
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FLOOR_LINE = re.compile(r"checkpoint-(\d+): noise floor median ([\d.]+)")


def measured() -> dict[int, float]:
    """Noise-floor medians logged during training, by step."""
    floors: dict[int, float] = {}
    for log in list(ROOT.glob("supervised_train*.log")) + list(ROOT.glob("local_train*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for step, value in FLOOR_LINE.findall(text):
            floors[int(step)] = float(value)
    return floors


def on_disk(output_dir: Path) -> dict[int, Path]:
    found = {}
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match and (path / "model.safetensors").exists():
            found[int(match.group(1))] = path
    return found


def kept(keep_dir: Path) -> dict[int, Path]:
    if not keep_dir.is_dir():
        return {}
    return {
        int(match.group(1)): path
        for path in keep_dir.glob("checkpoint-*")
        if (match := re.fullmatch(r"checkpoint-(\d+)", path.name)) and path.is_dir()
    }


def preserve(path: Path, keep_dir: Path) -> str:
    """Copy a checkpoint aside, verifying it first and never half-copying."""
    from tools.verify_checkpoint import describe, inspect, is_good

    report = inspect(path)
    if not is_good(report):
        return f"{path.name}: not copied - {describe(report)}"

    keep_dir.mkdir(parents=True, exist_ok=True)
    target = keep_dir / path.name
    if target.exists():
        return f"{path.name}: already kept"

    # Copy to a temporary name first: a copy interrupted half way through would
    # otherwise look exactly like a finished one, which is the same trap a power
    # cut set with checkpoint-1800.
    staging = keep_dir / f".{path.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(path, staging)
    staging.rename(target)

    size = sum(f.stat().st_size for f in target.iterdir()) / 2**30
    return f"{path.name}: kept ({size:.1f} GB)"


def survey(output_dir: Path, keep_dir: Path) -> None:
    floors, live, safe = measured(), on_disk(output_dir), kept(keep_dir)
    everything = sorted(set(floors) | set(live) | set(safe))
    if not everything:
        print("nothing measured and nothing on disk yet")
        return

    print(f"{'step':>7} {'noise floor':>12} {'on disk':>9} {'kept':>6}")
    for step in everything:
        floor = floors.get(step)
        print(f"{step:>7} {(f'{floor:.5f}' if floor else '-'):>12} "
              f"{('yes' if step in live else 'gone'):>9} "
              f"{('yes' if step in safe else '-'):>6}")

    scored = sorted(((f, s) for s, f in floors.items()), key=lambda x: x[0])
    if scored:
        best_floor, best_step = scored[0]
        where = ("kept" if best_step in safe else
                 "on disk, at risk" if best_step in live else "already deleted")
        print(f"\ncleanest measured: step {best_step:,} at {best_floor:.5f} - {where}")

    at_risk = [s for f, s in scored if s in live and s not in safe]
    if at_risk:
        print(f"{len(at_risk)} measured checkpoint(s) still on disk and unkept; "
              "--keep N copies the cleanest of them")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", default="./chatterbox_output")
    parser.add_argument("--keep", type=int, default=0,
                        help="copy the N cleanest measured checkpoints still on disk")
    parser.add_argument("--step", type=int, action="append",
                        help="copy this step, whatever it scored; repeatable")
    parser.add_argument("--watch", action="store_true",
                        help="keep checking while training runs")
    parser.add_argument("--interval", type=float, default=600)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    keep_dir = output_dir / "keep"

    if not args.keep and not args.step and not args.watch:
        survey(output_dir, keep_dir)
        return 0

    def once() -> None:
        floors, live = measured(), on_disk(output_dir)
        wanted: list[int] = list(args.step or [])
        if args.keep:
            ranked = sorted((f, s) for s, f in floors.items() if s in live)
            wanted += [s for _, s in ranked[: args.keep]]
        for step in dict.fromkeys(wanted):
            if step not in live:
                print(f"checkpoint-{step}: not on disk any more")
                continue
            print(preserve(live[step], keep_dir))

    once()
    if not args.watch:
        return 0

    print(f"\nwatching every {args.interval / 60:.0f} min; stop with Ctrl+C")
    try:
        while True:
            time.sleep(args.interval)
            once()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
