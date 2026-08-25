"""One view of everything that is running: local training and any remote run.

Progress ends up scattered - a tqdm bar redrawing itself in one log, a status
JSON somewhere else, a launcher log for the pod - and none of it is pleasant to
read directly. tqdm in particular writes with carriage returns, so a plain
`tail` shows one enormous line.

This pulls the useful parts out of whatever exists and prints a few lines.
Nothing here talks to the trainer; it only reads files, so it is safe to run at
any time and cannot disturb a run.

    python tools/watch.py            # one look
    python tools/watch.py --watch    # refresh every 30 s
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

# Where a training status file might be. The first that exists wins.
STATUS_CANDIDATES = [
    ROOT / "chatterbox_output" / "status.json",
    ROOT / "status.json",
]

LOG_CANDIDATES = [
    ("local", ROOT / "local_train.log"),
    ("remote", ROOT / "runpod_test02.log"),
    ("remote", ROOT / "runpod_test01.log"),
]

# transformers logs a dict per logging_steps; the numbers may be quoted.
LOSS_LINE = re.compile(r"\{'loss':\s*'?([\d.eE+-]+)'?.*?'epoch':\s*'?([\d.eE+-]+)'?")
# tqdm: "  12%|# | 520/4256 [57:55<7:19:20,  7.06s/it]"
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
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", "replace")
    except Exception:
        return ""


def last_match(pattern: re.Pattern, text: str):
    found = None
    for found in pattern.finditer(text.replace("\r", "\n")):
        pass
    return found


def human_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def show_status_file(path: Path) -> bool:
    state = read_json(path)
    if not state:
        return False

    age = time.time() - state.get("updated_at", 0)
    stale = "  (STALE)" if age > 600 else ""
    print(f"  status  {path.name}  updated {human_age(age)}{stale}")

    bits = []
    if (step := state.get("step")) is not None:
        total = state.get("max_steps")
        bits.append(f"step {step:,}/{total:,}" if total else f"step {step:,}")
    if (loss := state.get("loss")) is not None:
        bits.append(f"loss {loss:.3f}")
    if sps := state.get("samples_per_sec"):
        bits.append(f"{sps} samp/s")
    if (progress := state.get("progress")) is not None:
        bits.append(f"{100 * progress:.1f}%")
    if eta := state.get("eta_seconds"):
        bits.append(f"eta {int(eta) // 3600}h{int(eta) % 3600 // 60:02d}m")
    if cost := state.get("cost_so_far"):
        bits.append(f"${cost:.2f}")
    if bits:
        print("          " + "   ".join(bits))

    if gpu := state.get("gpu"):
        print(f"          gpu {gpu.get('name', '?')}  "
              f"{gpu.get('mem_used_gb', 0):.1f}/{gpu.get('mem_total_gb', 0):.0f} GB")
    if state.get("error"):
        print(f"          ERROR: {state['error']}")
    return True


def show_log(label: str, path: Path) -> None:
    text = tail_text(path)
    if not text.strip():
        return

    age = time.time() - path.stat().st_mtime
    stale = "  (no new output)" if age > 600 else ""
    print(f"\n  {label}  {path.name}  written {human_age(age)}{stale}")

    if bar := last_match(TQDM_LINE, text):
        percent, done, total, elapsed, remaining, rate, unit = bar.groups()
        per_step = f"{rate} {unit}"
        print(f"          {percent}%  step {int(done):,}/{int(total):,}  "
              f"elapsed {elapsed}  left {remaining}  {per_step}")

    if loss := last_match(LOSS_LINE, text):
        print(f"          loss {float(loss.group(1)):.3f}  "
              f"epoch {float(loss.group(2)):.3f}")

    # Launcher lines are already one per line and worth showing verbatim.
    interesting = [
        line.strip() for line in text.replace("\r", "\n").splitlines()
        if re.search(r"phase ->|STOPPING|finished|terminated|ERROR|FAILED|"
                     r"created at|budget:|downloaded|cheapest", line)
    ]
    for line in interesting[-4:]:
        print(f"          {line}")


def show_remote_pod() -> bool:
    """Ask a live pod how it is doing, rather than inferring it from a log.

    The launcher writes its pod id and control token to runpod_session.json for
    exactly this: without them the pod is only reachable from the process that
    created it, which is no help when that process is busy watching.
    """
    session = read_json(SESSION_PATH)
    if not session or not session.get("pod_id"):
        # No session, but a pod may still be running - from an older launcher,
        # or one somebody started by hand. Anything billing is worth showing.
        return show_pods_from_api()

    import urllib.request

    pod_id = session["pod_id"]
    request = urllib.request.Request(f"{session['proxy']}/status")
    request.add_header("Authorization", f"Bearer {session['control_token']}")
    request.add_header("User-Agent", "chatterbox-watch/1.0")

    print(f"\n  pod  {pod_id}  ({session.get('cloud', '?')}, "
          f"${session.get('hourly_rate', 0):.2f}/h)")

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            status = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        elapsed = time.time() - session.get("started_at", time.time())
        print(f"          unreachable ({type(error).__name__}) - "
              f"up {elapsed / 60:.0f} min")
        print("          the pod may still be booting; the launcher log knows more")
        return True

    elapsed = time.time() - session.get("started_at", time.time())
    cost = elapsed / 3600 * session.get("hourly_rate", 0)
    print(f"          phase {status.get('phase', '?')}   "
          f"up {elapsed / 60:.0f} min   spent ${cost:.2f}")

    training = status.get("training") or {}
    bits = []
    if (step := training.get("step")) is not None:
        total = training.get("max_steps")
        bits.append(f"step {step:,}/{total:,}" if total else f"step {step:,}")
    if (loss := training.get("loss")) is not None:
        bits.append(f"loss {loss:.3f}")
    if sps := training.get("samples_per_sec"):
        bits.append(f"{sps} samp/s")
    if eta := training.get("eta_seconds"):
        bits.append(f"eta {int(eta) // 3600}h{int(eta) % 3600 // 60:02d}m")
    if bits:
        print("          " + "   ".join(bits))

    if gpu := status.get("gpu"):
        print(f"          gpu {gpu.get('util_percent', '?')}%  "
              f"{gpu.get('mem_used_gb', 0):.1f}/{gpu.get('mem_total_gb', 0):.0f} GB  "
              f"{gpu.get('temperature_c', '?')}C")
    if disk := status.get("disk"):
        print(f"          disk {disk.get('free_gb', 0):.0f} GB free")
    if status.get("error"):
        print(f"          ERROR: {status['error']}")

    for line in (status.get("train_log") or status.get("setup_log") or [])[-3:]:
        print(f"          | {line[:96]}")
    return True


def show_pods_from_api() -> bool:
    """List whatever is running on the account, without needing a control token.

    This is the answer to "is anything costing me money right now", which is a
    different question from "how is the training going" and deserves an answer
    even when the detailed one is unavailable.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from tools.runpod_infra import api

        pods = api.list_pods()
    except Exception:
        return False

    if not pods:
        return False

    print("\n  runpod  (no session file; account-level view only)")
    for pod in pods:
        print(f"          {pod.get('id')}  {pod.get('desiredStatus')}  "
              f"${pod.get('costPerHr', 0):.2f}/h  {pod.get('name', '')}")
    print("          detail needs the control token the launcher holds; "
          "see its log below")
    return True


def show_samples() -> None:
    directory = ROOT / "chatterbox_output" / "inference_samples"
    if not directory.is_dir():
        return
    samples = sorted(directory.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    if not samples:
        return
    print(f"\n  audio samples  {len(samples)} in "
          f"{directory.relative_to(ROOT)}")
    for path in samples[-3:]:
        age = time.time() - path.stat().st_mtime
        print(f"          {path.name:<28} {path.stat().st_size / 1024:6.0f} KB  "
              f"{human_age(age)}")


def snapshot(extra_logs: list[Path]) -> None:
    print("=" * 68)
    print(f"  {time.strftime('%H:%M:%S')}")

    shown = any(show_status_file(p) for p in STATUS_CANDIDATES if p.exists())
    if not shown:
        # The launcher may have been pointed at a scratch directory.
        for path in sorted(ROOT.glob("**/local_status.json"))[:1]:
            show_status_file(path)

    show_remote_pod()

    for label, path in LOG_CANDIDATES:
        if path.exists():
            show_log(label, path)
    for path in extra_logs:
        if path.exists():
            show_log("extra", path)

    show_samples()
    print("=" * 68)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("logs", nargs="*", type=Path, help="extra log files")
    parser.add_argument("--watch", action="store_true", help="keep refreshing")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    while True:
        snapshot(args.logs)
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
