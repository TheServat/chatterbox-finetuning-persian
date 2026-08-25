"""How far along is preprocessing, and how much longer.

Preprocessing writes one `.pt` per clip, so the cache directory is the progress
bar - no log needed, and it works even when the run's own output is buffered or
its terminal is long gone.

The rate is measured over the most recent files rather than the whole run. An
average over everything is dragged down by model loading at the start, and by
any earlier partial run whose files are still sitting in the directory.

    python tools/progress.py            # one look
    python tools/progress.py --watch    # refresh until it finishes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "MyTTSDataset" / "preprocess"
DEFAULT_METADATA = ROOT / "MyTTSDataset" / "metadata.csv"

# Files older than this are assumed to be from a previous run and are excluded
# from the rate estimate, though they still count as done.
RECENT_WINDOW = 2000


def total_expected(metadata: Path) -> int:
    if not metadata.exists():
        return 0
    with metadata.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def snapshot(cache: Path, total: int) -> dict:
    files = list(cache.glob("*.pt"))
    if not files:
        return {"done": 0, "total": total}

    mtimes = sorted(f.stat().st_mtime for f in files)
    now = time.time()

    recent = mtimes[-RECENT_WINDOW:]
    span = recent[-1] - recent[0]
    rate = (len(recent) - 1) / span if span > 0 else 0.0

    remaining = max(0, total - len(files))
    return {
        "done": len(files),
        "total": total,
        "rate": rate,
        "remaining": remaining,
        "eta": remaining / rate if rate > 0 else None,
        "idle": now - mtimes[-1],
        "bytes": sum(f.stat().st_size for f in files[:200]) / min(200, len(files)),
    }


def render(state: dict) -> str:
    done, total = state["done"], state["total"]
    if not done:
        return "no clips processed yet"

    share = 100 * done / total if total else 0
    filled = int(share / 100 * 30)
    bar = "#" * filled + "-" * (30 - filled)

    lines = [f"[{bar}] {done:,} / {total:,}  ({share:.1f}%)"]

    if state.get("idle", 0) > 120:
        lines.append(
            f"  STALLED: nothing written for {state['idle'] / 60:.0f} min"
        )
    elif state.get("eta") is not None:
        finish = time.strftime("%H:%M", time.localtime(time.time() + state["eta"]))
        lines.append(
            f"  {state['rate']:.1f} clips/s   {state['remaining']:,} left   "
            f"about {state['eta'] / 60:.0f} min, done around {finish}"
        )

    if per_clip := state.get("bytes"):
        projected = per_clip * total / 2**20
        lines.append(
            f"  {per_clip / 1024:.1f} KB per clip -> {projected:.0f} MB for the "
            "whole corpus"
        )
    return "\n".join(lines)


def gpu_line() -> str | None:
    """One line of nvidia-smi, to show the job is actually on the GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    util, used, total = (p.strip() for p in result.stdout.strip().split(","))
    return f"  GPU {util}% busy, {int(used) / 1024:.1f} / {int(total) / 1024:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--watch", action="store_true", help="refresh until done")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    if not args.cache.exists():
        print(f"{args.cache} does not exist - preprocessing has not started.")
        return 1

    total = total_expected(args.metadata)
    if not total:
        print(f"Cannot read {args.metadata}; percentages will be unavailable.")

    while True:
        state = snapshot(args.cache, total)
        print(render(state))
        if line := gpu_line():
            print(line)

        if not args.watch or (total and state["done"] >= total):
            break
        print()
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
