"""Keep training going across power cuts, without needing anyone to notice.

Mains power at this site fails often - twenty five unclean shutdowns on record
since June, four of them in one day - and every one leaves training stopped
until a person restarts it by hand. The work itself survives: checkpoints are
written every hundred steps and `--resume` picks up the newest one. What was
missing is something to do the restarting.

So this runs the trainer, waits for it to exit, works out whether it stopped
because the run finished or because something killed it, and starts it again if
there is still work to do.

A loop that restarts on any exit would hammer a real bug forever, so a run that
dies within a minute counts as a fast failure, and three in a row stop the loop
and name the log to read. Only a finished run ends it quietly.

    python tools/keep_training.py                    the usual settings
    python tools/keep_training.py --epochs 10        anything else train.py takes
    python tools/keep_training.py --max-restarts 50

This survives the trainer dying, not the machine going down with it. To carry it
across the reboot as well, start it from Task Scheduler at logon, or put a
shortcut to it in shell:startup.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A run shorter than this did not fail on its own terms; it failed to start.
FAST_FAILURE_SECONDS = 60
FAST_FAILURES_ALLOWED = 3
PAUSE_BETWEEN_RUNS = 15

DEFAULT_ARGS = [
    "--epochs", "2",
    "--batch-size", "4",
    "--grad-accum", "8",
    "--workers", "2",
    "--save-steps", "100",
    "--sample",
    "--resume",
    "--no-preprocess",
]


def newest_checkpoint(output_dir: Path) -> Path | None:
    numbered = [
        (int(match.group(1)), path)
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir()
        and (match := re.fullmatch(r"checkpoint-(\d+)", path.name))
    ]
    return max(numbered)[1] if numbered else None


def progress(output_dir: Path) -> tuple[int, int] | None:
    """(step, max_steps) from the newest checkpoint, or None if unreadable."""
    checkpoint = newest_checkpoint(output_dir)
    if not checkpoint:
        return None
    try:
        state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        return int(state["global_step"]), int(state.get("max_steps") or 0)
    except Exception:
        return None


def say(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def quarantine_damaged(output_dir: Path) -> int:
    """Move aside any checkpoint a power cut left half-written.

    A checkpoint can come back the right size with a region never flushed -
    zeros where weights should be. Training resumes from it without complaint
    and learns nothing, because under LoRA the base weights are frozen and a
    zeroed one is copied forward for ever. That is what happened at step 1800:
    layers 7 to 9 came back empty and three hundred and fifty steps ran against
    a model answering uniformly.

    Only the newest is checked, then the next if that one is bad, because
    reading a 2.25 GB file is not free and corruption lands on whatever was
    being written when the power went.
    """
    sys.path.insert(0, str(ROOT))
    from tools.verify_checkpoint import checkpoints, describe, inspect, is_good

    moved = 0
    for path in reversed(checkpoints(output_dir)):
        report = inspect(path)
        if is_good(report):
            return moved
        say(f"{path.name} is damaged: {describe(report)}")
        destination = output_dir / "corrupt"
        destination.mkdir(exist_ok=True)
        target = destination / path.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(path), str(target))
        moved += 1
        say(f"moved {path.name} out of the way")
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", default="./chatterbox_output")
    parser.add_argument("--max-restarts", type=int, default=100)
    parser.add_argument("--log-prefix", default="supervised_train")
    known, passthrough = parser.parse_known_args()

    output_dir = Path(known.output_dir)
    train_args = list(passthrough) or list(DEFAULT_ARGS)
    if "--resume" not in train_args:
        train_args.append("--resume")

    say(f"train.py {' '.join(train_args)}")
    if quarantine_damaged(output_dir):
        say("resuming from the newest checkpoint that survived")
    if state := progress(output_dir):
        say(f"resuming from step {state[0]:,}"
            + (f" of {state[1]:,}" if state[1] else ""))

    fast_failures = 0
    for attempt in range(1, known.max_restarts + 1):
        log_path = ROOT / f"{known.log_prefix}{attempt}.log"
        say(f"run {attempt}: logging to {log_path.name}")

        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", str(ROOT / "train.py"), *train_args],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
            )
        ran_for = time.time() - started
        quarantine_damaged(output_dir)
        state = progress(output_dir)
        reached = f", reached step {state[0]:,}" if state else ""

        if state and state[1] and state[0] >= state[1]:
            say(f"finished: step {state[0]:,} of {state[1]:,}")
            return 0

        if result.returncode == 0:
            # A clean exit short of max_steps is still done: the trainer stops
            # on its own epoch count, which can be the lower of the two.
            say(f"trainer exited cleanly after {ran_for / 60:.0f}m{reached}")
            return 0

        if ran_for < FAST_FAILURE_SECONDS:
            fast_failures += 1
            say(f"died after {ran_for:.0f}s (exit {result.returncode}) - "
                f"fast failure {fast_failures} of {FAST_FAILURES_ALLOWED}")
            if fast_failures >= FAST_FAILURES_ALLOWED:
                say(f"stopping: three quick failures in a row is a bug, not a "
                    f"power cut. Read {log_path.name}")
                return 1
        else:
            fast_failures = 0
            say(f"stopped after {ran_for / 60:.0f}m "
                f"(exit {result.returncode}){reached}")

        time.sleep(PAUSE_BETWEEN_RUNS)

    say(f"reached the {known.max_restarts}-restart limit")
    return 1


if __name__ == "__main__":
    sys.exit(main())
