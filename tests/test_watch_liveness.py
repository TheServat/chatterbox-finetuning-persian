"""The monitor must not call a live run dead.

Telling live from dead is the whole job here: a directory accumulates a log per
run, and showing a finished one beside a running one is worse than showing
nothing. A clock that steps backwards breaks that - after a restart the system
clock here went back twenty minutes, the previous run's log was left stamped in
the future, and it won "newest" while training stepped normally beside it.

    python tests/test_watch_liveness.py
"""

import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("watch", ROOT / "tools/watch.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def write(directory: Path, name: str, mtime_offset: float) -> Path:
    path = directory / name
    path.write_text("x", encoding="utf-8")
    stamp = time.time() + mtime_offset
    os.utime(path, (stamp, stamp))
    return path


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    return condition


def main() -> int:
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        original_root = watch.ROOT
        watch.ROOT = directory
        try:
            dead = write(directory, "local_train5.log", +1200)   # stamped ahead
            live = write(directory, "local_train6.log", -3)      # actually current

            picked = watch.newest("local_train*.log")
            ok &= check("a log stamped in the future does not win",
                        picked == live, f"picked {picked.name}")

            ok &= check("age never goes negative", watch.age(dead) >= 0,
                        f"{watch.age(dead):.0f}s")
            ok &= check("and the live log reads as fresh", watch.age(live) < 60)

            # The ordinary case must keep working.
            for path in directory.glob("*"):
                path.unlink()
            older = write(directory, "local_train1.log", -9000)
            newer = write(directory, "local_train2.log", -30)
            ok &= check("with sane stamps, newest still wins",
                        watch.newest("local_train*.log") == newer)
            ok &= check("and the older one is not chosen",
                        watch.newest("local_train*.log") != older)

            # If every candidate is skewed, still return something.
            for path in directory.glob("*"):
                path.unlink()
            only = write(directory, "local_train9.log", +5000)
            ok &= check("all stamps skewed: falls back rather than reporting nothing",
                        watch.newest("local_train*.log") == only)

            for path in directory.glob("*"):
                path.unlink()
            ok &= check("nothing matching: None",
                        watch.newest("local_train*.log") is None)
        finally:
            watch.ROOT = original_root

    ok &= check("a missing file is infinitely old",
                watch.age(Path("does-not-exist")) == float("inf"))

    print("\nall good" if ok else "\nFAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
