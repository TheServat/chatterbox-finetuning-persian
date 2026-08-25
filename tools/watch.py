"""What is running right now, in a few lines.

Progress ends up scattered - a tqdm bar redrawing itself in one log, a status
JSON somewhere else, a launcher log per attempt - and none of it reads well
directly. tqdm writes with carriage returns, so a plain `tail` returns one
enormous line.

The harder problem is telling live from dead. A directory accumulates logs from
every run ever started, and showing a finished run beside a running one is worse
than showing nothing: the first version of this printed step 575 from the live
run directly above step 1,000 from a run that had died two hours earlier.

So logs are discovered rather than named, the newest wins, and anything that has
not been written to recently is summarised in one line instead of expanded.
Nothing here talks to a trainer; it only reads files and polls a pod, so it is
safe to run at any time.

    python tools/watch.py            # one look
    python tools/watch.py --watch    # refresh every 30 s
    python tools/watch.py --all      # include finished runs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSION_PATH = ROOT / "runpod_session.json"
STATUS_PATH = ROOT / "chatterbox_output" / "status.json"
SAMPLES_DIR = ROOT / "chatterbox_output" / "inference_samples"

# A log untouched for this long belongs to a finished or dead run.
LIVE_SECONDS = 900

LOSS_LINE = re.compile(r"\{'loss':\s*'?([\d.eE+-]+)'?.*?'epoch':\s*'?([\d.eE+-]+)'?")
TQDM_LINE = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.]+)(s/it|it/s)"
)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def tail_text(path: Path, limit: int = 60000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - limit))
            return handle.read().decode("utf-8", "replace")
    except Exception:
        return ""


def last_match(pattern: re.Pattern, text: str):
    found = None
    for found in pattern.finditer(text.replace("\r", "\n")):
        pass
    return found


def age(path: Path) -> float:
    try:
        return time.time() - path.stat().st_mtime
    except Exception:
        return float("inf")


def human_age(seconds: float) -> str:
    if seconds == float("inf"):
        return "never"
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def newest(pattern: str) -> Path | None:
    """The most recently written file matching a glob, or None."""
    matches = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def bar(fraction: float, width: int = 24) -> str:
    filled = max(0, min(width, int(fraction * width)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# --------------------------------------------------------------------------

def show_local() -> None:
    log = newest("local_train*.log")
    status = read_json(STATUS_PATH)
    log_age = age(log) if log else float("inf")
    live = log_age < LIVE_SECONDS

    if not log:
        print("LOCAL    nothing has been run here")
        return

    if not live:
        print(f"LOCAL    not running   ({log.name}, last written {human_age(log_age)})")
        return

    text = tail_text(log)
    line = f"LOCAL    {log.name}"

    if tq := last_match(TQDM_LINE, text):
        percent, done, total, elapsed, remaining, rate, unit = tq.groups()
        fraction = int(done) / max(int(total), 1)
        print(f"LOCAL    {bar(fraction)} {percent}%   "
              f"step {int(done):,}/{int(total):,}")
        line = f"         elapsed {elapsed}   left {remaining}   {rate} {unit}"
    print(line)

    if loss := last_match(LOSS_LINE, text):
        detail = f"         loss {float(loss.group(1)):.3f}   epoch {float(loss.group(2)):.2f}"
        if status and (gpu := status.get("gpu")):
            detail += (f"   gpu {gpu.get('mem_used_gb', 0):.1f}/"
                       f"{gpu.get('mem_total_gb', 0):.0f} GB")
        print(detail)

    if status:
        if warning := contention(status.get("gpu")):
            print(warning)

    print(f"         written {human_age(log_age)}")


def contention(gpu: dict | None) -> str | None:
    """Whether something other than training is sitting on the card.

    Worth a line of its own because the failure is silent. When a 6 GB card
    fills up, Windows pages VRAM to system RAM rather than raising: utilisation
    still reads 91%, the loss still falls, and only the step time gives it away
    - here 7.4 s/step became 30 when Ollama loaded a 7B model beside training.
    Reading a step time as slow needs a baseline; reading "3.1 GB is not yours"
    does not.

    mem_used_gb covers the whole card and reserved_gb covers this process, so
    the gap is other processes plus our own CUDA context. The context alone is
    a few hundred megabytes, hence a threshold rather than any gap at all.
    """
    if not gpu or "reserved_gb" not in gpu:
        return None  # written by an older run, before this was recorded

    used = gpu.get("mem_used_gb", 0)
    total = gpu.get("mem_total_gb", 0)
    elsewhere = used - gpu["reserved_gb"]

    if elsewhere < 0.8 or not total:
        return None

    line = f"         {elsewhere:.1f} GB on this card is not training"
    if used / total > 0.9:
        line += f"  - card {used / total * 100:.0f}% full, expect slow steps"
    return line


def show_remote() -> None:
    session = read_json(SESSION_PATH)
    log = newest("runpod_*.log")

    if session and session.get("pod_id"):
        show_live_pod(session)
        return

    if not log:
        print("REMOTE   nothing has been run")
        return

    log_age = age(log)
    if log_age < LIVE_SECONDS:
        # No session file, but the log is fresh. Either a launcher is between
        # attempts, or one was killed and its last lines still name a pod that
        # is gone. The account is the only authority on what is actually
        # billing, so ask it rather than inferring from a log.
        running = account_pods()
        if running:
            print(f"REMOTE   {len(running)} pod(s) billing, no session file "
                  f"(started outside this launcher?)")
            for pod in running:
                print(f"         {pod.get('id')}  {pod.get('desiredStatus')}  "
                      f"${pod.get('costPerHr', 0):.2f}/h")
            print("         detail needs a session; "
                  "python -m tools.runpod_infra.launch --cleanup stops them")
        else:
            print(f"REMOTE   nothing running   ({log.name}, {human_age(log_age)})")
            outcome = interesting_lines(tail_text(log))
            if outcome:
                print(f"         last: {outcome[-1][:88]}")
        return

    outcome = interesting_lines(tail_text(log))
    summary = outcome[-1] if outcome else "no outcome recorded"
    print(f"REMOTE   not running   ({log.name}, {human_age(log_age)})")
    print(f"         last: {summary[:88]}")
    if running := account_pods():
        print(f"         WARNING: {len(running)} pod(s) still billing - "
              "python -m tools.runpod_infra.launch --cleanup")


def account_pods() -> list[dict]:
    """What the RunPod account says is running, which is the only thing billing."""
    try:
        sys.path.insert(0, str(ROOT))
        from tools.runpod_infra import api

        return api.list_pods()
    except Exception:
        return []


def interesting_lines(text: str) -> list[str]:
    wanted = re.compile(
        r"STOPPING|finished:|terminated|host fault|escalating|created at|"
        r"phase ->|COULD NOT|budget:"
    )
    return [
        line.strip()
        for line in text.replace("\r", "\n").splitlines()
        if wanted.search(line)
    ]


def show_live_pod(session: dict) -> None:
    import urllib.request

    pod_id = session["pod_id"]
    rate = session.get("hourly_rate", 0)
    elapsed = time.time() - session.get("started_at", time.time())
    header = (f"REMOTE   pod {pod_id}   {session.get('cloud', '?')}   "
              f"${rate:.2f}/h   up {elapsed / 60:.0f}m   "
              f"spent ${elapsed / 3600 * rate:.2f}")

    request = urllib.request.Request(f"{session['proxy']}/status")
    request.add_header("Authorization", f"Bearer {session['control_token']}")
    request.add_header("User-Agent", "chatterbox-watch/1.0")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            status = json.loads(response.read().decode("utf-8"))
    except Exception:
        print(header)
        print("         not answering yet - still booting, or already gone")
        return

    print(header)
    training = status.get("training") or {}
    phase = status.get("phase", "?")

    if training.get("step") is not None:
        total = training.get("max_steps") or 0
        fraction = training["step"] / total if total else 0
        print(f"         {bar(fraction)} {phase}   "
              f"step {training['step']:,}/{total:,}")
        bits = []
        if (loss := training.get("loss")) is not None:
            bits.append(f"loss {loss:.3f}")
        if sps := training.get("samples_per_sec"):
            bits.append(f"{sps} samp/s")
        if eta := training.get("eta_seconds"):
            bits.append(f"eta {int(eta) // 3600}h{int(eta) % 3600 // 60:02d}m")
        if bits:
            print("         " + "   ".join(bits))
    else:
        print(f"         {phase}")
        # Before training there are no steps, so the setup log is the only
        # sign of life - and "downloading" for ten minutes with nothing else
        # is indistinguishable from a hang.
        for line in reversed(status.get("setup_log") or []):
            line = line.strip()
            if line and not line.startswith(("===", "---", "|", "+")):
                print(f"         {line[:88]}")
                break

    if gpu := status.get("gpu"):
        print(f"         gpu {gpu.get('util_percent', 0)}%   "
              f"{gpu.get('mem_used_gb', 0):.1f}/{gpu.get('mem_total_gb', 0):.0f} GB")
    if status.get("error"):
        print(f"         ERROR: {status['error'][:88]}")


def show_samples() -> None:
    if not SAMPLES_DIR.is_dir():
        return
    samples = [p for p in SAMPLES_DIR.glob("*.wav") if "trimmed" not in p.name]
    if not samples:
        print("AUDIO    none yet")
        return
    samples.sort(key=lambda p: p.stat().st_mtime)
    latest = samples[-1]
    print(f"AUDIO    {len(samples)} sample(s), newest {latest.name} "
          f"({human_age(age(latest))})")
    print(f"         {SAMPLES_DIR.relative_to(ROOT)}")


def snapshot(show_all: bool) -> None:
    print("=" * 66)
    print(f"  {time.strftime('%H:%M:%S')}")
    print()
    show_local()
    print()
    show_remote()
    print()
    show_samples()

    if show_all:
        print("\nolder runs:")
        for log in sorted(ROOT.glob("runpod_*.log"),
                          key=lambda p: p.stat().st_mtime, reverse=True)[1:6]:
            lines = interesting_lines(tail_text(log))
            print(f"  {log.name:<24} {human_age(age(log)):>9}  "
                  f"{(lines[-1] if lines else '')[:60]}")
    print("=" * 66)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--watch", action="store_true", help="keep refreshing")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--all", action="store_true", help="also list finished runs")
    args = parser.parse_args()

    while True:
        snapshot(args.all)
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
