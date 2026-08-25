"""Run the Persian finetune on RunPod, start to finish, without leaving a pod on.

The whole point is that a GPU bills by the second whether or not anything useful
is happening. So this owns the entire lifecycle: choose the GPU on live prices,
create the pod, ship the data up, watch the run, bring the result back, and shut
everything down - including on failure, on a watchdog trip, and on Ctrl-C.

Termination is the part worth being paranoid about. It happens in a `finally`,
and again from a signal handler, and the pod is re-checked afterwards to confirm
it is really gone. A pod that survives a crashed launcher is the one mistake
here that costs real money.

Watching happens over RunPod's HTTPS proxy rather than SSH, because port 22 is
blocked on the network this is driven from. The pod runs a small control server
(`control_server.py`) that reports progress and accepts the data upload.

    python -m tools.runpod_infra.launch --plan          # decide nothing, show costs
    python -m tools.runpod_infra.launch --epochs 3
    python -m tools.runpod_infra.launch --adopt <pod-id>  # re-attach after a crash
    python -m tools.runpod_infra.launch --cleanup       # kill anything left running
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import secrets
import signal
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.runpod_infra import api, plan
# Aliased: the stdlib `secrets` is already imported above for token_urlsafe.
from tools.runpod_infra import secrets as credentials  # noqa: E402

credentials.load()

# CUDA 12.8 covers Ampere through Blackwell, so one image serves every GPU in
# the fallback list, including the RTX 5090.
DEFAULT_IMAGE = "runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204"

# Ordered by estimated total cost for this job, cheapest first. RunPod walks the
# list, so a card being out of stock costs a moment rather than the run.
DEFAULT_GPUS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA GeForce RTX 5090",
]

CONTROL_PORT = 8080
CACHE_NAME = "preprocess_fa.tar.gz"
VOLUME_MOUNT = "/workspace/persist"

DEFAULTS = {
    "pod_ready_timeout": 900,     # RUNNING, from creation
    "server_timeout": 1200,       # control server answering, from RUNNING
    "setup_timeout": 3600,        # through dependencies, weights and extraction
    "stall_timeout": 1200,        # no step progress once training has begun
    "poll_seconds": 20,
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------
# talking to the pod
# --------------------------------------------------------------------------

class Pod:
    def __init__(self, pod_id: str, token: str):
        self.id = pod_id
        self.token = token
        self.base = api.proxy_url(pod_id, CONTROL_PORT)

    def _open(self, path: str, method="GET", data=None, timeout=60,
              headers: dict | None = None):
        request = urllib.request.Request(f"{self.base}{path}", data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("User-Agent", api.USER_AGENT)
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        return urllib.request.urlopen(request, timeout=timeout)

    def alive(self) -> bool:
        try:
            with self._open("/health", timeout=20) as response:
                return response.status == 200
        except Exception:
            return False

    def status(self) -> dict | None:
        try:
            with self._open("/status", timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

    def upload(self, local: Path, remote: str, progress=True) -> bool:
        size = local.stat().st_size
        log(f"uploading {local.name} ({size / 2**20:.0f} MB) -> {remote}")
        started = time.time()
        with local.open("rb") as handle:
            try:
                with self._open(
                    f"/upload?path={remote}", method="PUT", data=handle,
                    timeout=3600,
                    headers={"Content-Length": str(size),
                             "Content-Type": "application/octet-stream"},
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except Exception as error:
                log(f"upload failed: {error}")
                return False
        elapsed = max(time.time() - started, 1e-6)
        if progress:
            log(f"uploaded {body.get('bytes', 0) / 2**20:.0f} MB in "
                f"{elapsed / 60:.1f} min ({size / elapsed / 2**20:.1f} MB/s)")
        return True

    def download(self, remote: str, local: Path) -> bool:
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._open(f"/download?path={remote}", timeout=1800) as response:
                with local.open("wb") as handle:
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
        except Exception as error:
            log(f"download of {remote} failed: {error}")
            return False
        log(f"downloaded {remote} -> {local} ({local.stat().st_size / 2**20:.1f} MB)")
        return True

    def exists(self, remote: str) -> bool:
        try:
            with self._open(f"/ls?path={remote}", timeout=30) as response:
                return response.status == 200
        except Exception:
            return False


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------

_ACTIVE_POD: str | None = None


def _emergency_terminate() -> None:
    """Last line of defence: never leave a GPU running because we crashed."""
    if _ACTIVE_POD:
        try:
            api.terminate_pod(_ACTIVE_POD)
            print(f"\nterminated pod {_ACTIVE_POD}", flush=True)
        except Exception as error:
            print(f"\nCOULD NOT TERMINATE {_ACTIVE_POD}: {error}\n"
                  f"Terminate it by hand at https://console.runpod.io/pods",
                  flush=True)


atexit.register(_emergency_terminate)


def _handle_signal(signum, frame):
    print(f"\nsignal {signum} - shutting the pod down", flush=True)
    _emergency_terminate()
    sys.exit(130)


def ensure_volume(name: str, size_gb: int, datacenter: str | None) -> dict | None:
    existing = api.find_volume(name)
    if existing:
        log(f"reusing network volume {name} "
            f"({existing.get('size')} GB, {existing.get('dataCenterId')})")
        return existing

    if datacenter is None:
        supported = api.datacenters(storage_only=True)
        if not supported:
            log("no datacenter supports network volumes; continuing without one")
            return None
        datacenter = supported[0]["id"]

    log(f"creating network volume {name}: {size_gb} GB in {datacenter}")
    return api.create_volume(name, size_gb, datacenter)


def build_spec(args, control_token: str, volume: dict | None, gpu_ids: list[str]) -> dict:
    train_args = args.train_args or (
        f"--batch-size {args.batch_size} --grad-accum {args.grad_accum} "
        f"--epochs {args.epochs} --workers {args.workers} "
        f"--save-steps {args.save_steps} --resume"
    )
    if args.sample:
        train_args += " --sample"

    # The start command stays small on purpose: it clones the repository and
    # hands over, so the real bootstrap is version-controlled rather than
    # embedded in a pod spec.
    start = (
        "set -x; mkdir -p /workspace/incoming; cd /workspace; "
        "rm -f /workspace/DONE /workspace/FAILED /workspace/status.json; "
        "echo cloning > /workspace/phase; "
        f"git clone --depth 1 -b {args.branch} {args.repo} repo "
        ">> /workspace/bootstrap.log 2>&1 || echo 'git clone failed' > /workspace/FAILED; "
        "nohup bash /workspace/repo/tools/runpod_infra/bootstrap.sh "
        ">> /workspace/bootstrap.log 2>&1 & "
        "exec python3 /workspace/repo/tools/runpod_infra/control_server.py "
        f"--port {CONTROL_PORT} --root /workspace"
    )

    # ManaTTS alone is 33 GB of parquet and expands to roughly the same again in
    # wav, so the disk has to be sized from what is actually being built.
    disk = args.container_disk
    if disk is None:
        disk = 120 if "mana_hf" in args.sources else 60
        if "youtube" in args.sources:
            disk += 60

    spec = {
        "name": args.name,
        "imageName": args.image,
        "cloudType": args.cloud,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": gpu_ids,
        "gpuTypePriority": "custom",
        "containerDiskInGb": disk,
        "ports": [f"{CONTROL_PORT}/http"],
        "dockerStartCmd": ["bash", "-lc", start],
        "env": {
            "POD_ROOT": "/workspace",
            "PERSIST_DIR": VOLUME_MOUNT if volume else "/workspace/persist",
            "CONTROL_TOKEN": control_token,
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "TRAIN_ARGS": train_args,
            "DATASET_SOURCES": args.sources,
            "KEEP_DATASETS": "1" if args.keep_datasets else "0",
            "HOURLY_RATE": str(args.hourly_rate or 0),
            "DATA_TIMEOUT": str(args.setup_timeout),
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            # Only so the pod's watchdog can delete itself if this launcher dies.
            "RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY", ""),
            "IDLE_LIMIT": str(int(args.idle_limit)),
            "MAX_LIFETIME": str(int(args.max_hours * 3600 * 1.5)),
        },
    }

    if volume:
        spec["networkVolumeId"] = volume["id"]
        spec["volumeMountPath"] = VOLUME_MOUNT
        if datacenter := volume.get("dataCenterId"):
            # A network volume only attaches inside its own datacenter.
            spec["dataCenterIds"] = [datacenter]
    else:
        spec["volumeInGb"] = disk

    return spec


def wait_for(condition, timeout: float, poll: float, description: str):
    """Poll until `condition()` is truthy. Returns the value, or None on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = condition()
        if value:
            return value
        remaining = int(deadline - time.time())
        print(f"\r  waiting for {description}... {remaining}s left    ",
              end="", flush=True)
        time.sleep(poll)
    print()
    return None


def render(status: dict, started: float, rate: float) -> str:
    training = status.get("training") or {}
    gpu = status.get("gpu") or {}
    elapsed = time.time() - started
    parts = [f"{elapsed / 60:6.1f}m", f"{status.get('phase', '?'):<16}"]

    if step := training.get("step"):
        total = training.get("max_steps") or 0
        parts.append(f"step {step}/{total}" if total else f"step {step}")
    if (loss := training.get("loss")) is not None:
        parts.append(f"loss {loss:.3f}")
    if sps := training.get("samples_per_sec"):
        parts.append(f"{sps:.1f} samp/s")
    if eta := training.get("eta_seconds"):
        parts.append(f"eta {int(eta) // 3600}h{int(eta) % 3600 // 60:02d}m")
    if gpu:
        parts.append(f"gpu {gpu.get('util_percent', '?')}% "
                     f"{gpu.get('mem_used_gb', 0):.0f}/{gpu.get('mem_total_gb', 0):.0f}G")
    if rate:
        parts.append(f"${elapsed / 3600 * rate:.2f}")
    return "  ".join(parts)


def collect_results(pod: Pod, out_dir: Path, *, expect_adapter: bool) -> bool:
    """Bring everything back. Returns whether the trained adapter arrived intact.

    The adapter is the only irreplaceable output - the logs can be re-read and
    the corpus rebuilt, but hours of training cannot. So it is retried, and its
    archive is opened to prove it is not truncated before anyone acts on a
    "downloaded" message.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for remote, local in (
        ("train.log", out_dir / "train.log"),
        ("bootstrap.log", out_dir / "bootstrap.log"),
        ("preprocess.log", out_dir / "preprocess.log"),
        ("status.json", out_dir / "status.json"),
        ("samples.tar.gz", out_dir / "samples.tar.gz"),
    ):
        pod.download(remote, local)

    if not expect_adapter:
        return True

    archive = out_dir / "persian_adapter.tar.gz"
    for attempt in range(1, 4):
        if pod.download("persian_adapter.tar.gz", archive) and archive.exists():
            try:
                with tarfile.open(archive) as tar:
                    tar.extractall(out_dir, filter="data")
                log(f"adapter verified and extracted to {out_dir / 'persian_adapter'}")
                return True
            except Exception as error:
                log(f"the adapter archive is unreadable ({error}); retrying")
                archive.unlink(missing_ok=True)
        log(f"adapter download attempt {attempt}/3 failed")
        time.sleep(10)

    return False


def monitor(pod: Pod, args, started: float) -> tuple[bool, str]:
    """Watch until the run ends. Returns (succeeded, reason)."""
    last_step = -1
    last_progress_at = time.time()
    last_phase = None

    while True:
        elapsed = time.time() - started

        if args.max_hours and elapsed > args.max_hours * 3600:
            return False, f"wall-clock limit of {args.max_hours} h reached"
        if args.max_cost and args.hourly_rate:
            spent = elapsed / 3600 * args.hourly_rate
            if spent > args.max_cost:
                return False, f"cost limit of ${args.max_cost} reached (${spent:.2f})"

        status = pod.status()
        if status is None:
            # A dropped poll is normal; a dead pod is not.
            state = (api.get_pod(pod.id) or {}).get("desiredStatus", "?")
            if state not in ("RUNNING",):
                return False, f"pod is no longer running (state {state})"
            log("status unavailable, retrying")
            time.sleep(args.poll_seconds)
            continue

        phase = status.get("phase", "?")
        if phase != last_phase:
            log(f"phase -> {phase}")
            last_phase = phase

        if phase == "done":
            return True, "training finished"
        if phase == "failed" or status.get("ok") is False:
            return False, status.get("error") or "the pod reported failure"

        training = status.get("training") or {}
        step = training.get("step", -1)
        if step > last_step:
            last_step, last_progress_at = step, time.time()

        idle = time.time() - last_progress_at
        if phase == "training" and idle > args.stall_timeout:
            return False, f"no progress for {idle / 60:.0f} min"
        if phase in ("setup", "models", "cloning", "booting", "data") \
                and elapsed > args.setup_timeout:
            return False, f"setup exceeded {args.setup_timeout / 60:.0f} min"

        # Downloading and preprocessing the corpus is legitimately slow - tens of
        # GB and tens of thousands of clips - so it gets its own, longer ceiling
        # rather than tripping the setup one.
        if phase in ("downloading_datasets", "building_dataset", "preprocessing") \
                and elapsed > args.build_timeout:
            return False, f"corpus build exceeded {args.build_timeout / 3600:.1f} h"

        print(f"\r  {render(status, started, args.hourly_rate)}   ", end="", flush=True)
        time.sleep(args.poll_seconds)


def cmd_cleanup() -> int:
    pods = api.list_pods()
    if not pods:
        print("no pods running")
    for pod in pods:
        print(f"terminating {pod.get('id')} ({pod.get('name')})")
        api.terminate_pod(pod["id"])

    volumes = api.list_volumes()
    if volumes:
        print("\nnetwork volumes (kept - delete explicitly with --delete-volume):")
        for volume in volumes:
            print(f"  {volume.get('id')}  {volume.get('name')}  "
                  f"{volume.get('size')} GB  {volume.get('dataCenterId')}")
    return 0


def main() -> int:
    global _ACTIVE_POD

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", action="store_true", help="show costs, create nothing")
    parser.add_argument("--cleanup", action="store_true", help="terminate every pod")
    parser.add_argument("--adopt", help="re-attach to an existing pod id")

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--sample", action="store_true",
                        help="synthesise a Persian sample at every checkpoint")
    parser.add_argument("--train-args", help="replace the generated train.py arguments")

    parser.add_argument("--gpu", action="append", help="GPU id, repeatable, in order")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    # Community is roughly half the price of secure for the same card. These runs
    # are short, checkpointed and resumable, so the reliability premium is not
    # worth paying.
    parser.add_argument("--cloud", default="COMMUNITY", choices=["SECURE", "COMMUNITY"])
    parser.add_argument("--container-disk", type=int, default=None,
                        help="ephemeral disk in GB; sized from --sources if unset")
    parser.add_argument("--name", default="chatterbox-fa")
    parser.add_argument("--repo", default="https://github.com/TheServat/chatterbox-finetuning-persian.git")
    parser.add_argument("--branch", default="main")

    parser.add_argument("--volume-name", default="chatterbox-fa")
    parser.add_argument("--volume-gb", type=int, default=20)
    parser.add_argument("--no-volume", action="store_true",
                        help="skip the network volume; nothing survives the pod")
    parser.add_argument("--delete-volume", action="store_true",
                        help="delete the network volume once results are downloaded")
    parser.add_argument("--datacenter", help="force a datacenter id")

    parser.add_argument("--sources", default="yoda narration mana_hf",
                        help="corpora the pod downloads and builds, space separated")
    parser.add_argument("--keep-datasets", action="store_true",
                        help="also keep the raw corpora on the volume; only worth "
                             "it when iterating on the corpus filters")
    parser.add_argument("--upload-cache", action="store_true",
                        help="push the local preprocessed cache instead of letting "
                             "the pod rebuild it; only worth it on a fast uplink")
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "MyTTSDataset" / CACHE_NAME)
    parser.add_argument("--out", type=Path, default=ROOT / "runpod_results")

    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULTS["poll_seconds"])
    parser.add_argument("--setup-timeout", type=float, default=DEFAULTS["setup_timeout"])
    parser.add_argument("--stall-timeout", type=float, default=DEFAULTS["stall_timeout"])
    parser.add_argument("--build-timeout", type=float, default=4 * 3600,
                        help="ceiling on downloading and preprocessing the corpus")
    parser.add_argument("--idle-limit", type=float, default=3600,
                        help="seconds without a check-in before the pod deletes "
                             "itself; the safety net if this machine dies")
    parser.add_argument("--keep-pod", action="store_true",
                        help="leave the pod running at the end (it keeps billing)")
    args = parser.parse_args()
    args.hourly_rate = 0.0

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if args.cleanup:
        return cmd_cleanup()

    estimates = plan.rank(args.epochs, cloud=args.cloud)
    if args.plan:
        for entry in estimates[:8]:
            print(f"  {entry['name']:<16} ${entry['price']:.2f}/h  "
                  f"{entry['hours']:.1f} h  ${entry['cost']:.2f}")
        return 0

    if not estimates:
        log("no suitable GPU is available right now")
        return 1

    chosen = estimates[0]
    args.hourly_rate = chosen["price"]

    # Ask for exactly the cards that were ranked, in ranked order. Passing a
    # fixed list while quoting a ranked estimate lets RunPod hand back a card
    # nobody priced.
    if args.gpu:
        gpu_ids = args.gpu
    else:
        ranked_ids = [e["id"] for e in estimates if e["id"] in DEFAULT_GPUS]
        gpu_ids = ranked_ids or DEFAULT_GPUS
    log(f"cheapest suitable on {args.cloud}: {chosen['name']} at "
        f"${chosen['price']:.2f}/h, "
        f"estimated {chosen['hours']:.1f} h / ${chosen['cost']:.2f}")

    control_token = secrets.token_urlsafe(32)
    started = time.time()

    if args.adopt:
        pod_id = args.adopt
        log(f"adopting pod {pod_id} (its control token must match CONTROL_TOKEN)")
        control_token = os.environ.get("CONTROL_TOKEN", control_token)
    else:
        if args.upload_cache and not args.cache.exists():
            log(f"--upload-cache given but {args.cache} does not exist")
            return 1

        volume = None if args.no_volume else ensure_volume(
            args.volume_name, args.volume_gb, args.datacenter
        )
        spec = build_spec(args, control_token, volume, gpu_ids)
        log(f"creating pod ({args.cloud}, sources: {args.sources})")
        log(f"  GPUs, in order: {', '.join(g.split()[-1] for g in gpu_ids)}")
        if volume:
            log(f"  pinned to {volume.get('dataCenterId')} by the network volume")

        try:
            pod_info = api.create_pod(spec)
        except api.RunPodError as error:
            if "no instances currently available" not in str(error):
                raise
            # A network volume pins the pod to one datacentre, so "nothing
            # available" usually means nothing available *there*, not nowhere.
            log("no instances available for that request")
            if volume:
                log(f"  the volume pins this to {volume.get('dataCenterId')}; "
                    "options:")
                log("    --no-volume            any datacentre, nothing persisted")
                log("    --cloud SECURE         same datacentre, roughly double the rate")
            else:
                log("  options:")
                log("    --cloud SECURE         more expensive but usually available")
                log("    --gpu '<name>'         name a card explicitly")
            log("  or wait: community capacity comes and goes within minutes")
            return 1
        pod_id = pod_info["id"]
        _ACTIVE_POD = pod_id

        # Ground truth. Everything before this was an estimate from a price list
        # that does not always match what gets rented.
        actual = pod_info.get("costPerHr")
        if actual:
            if abs(actual - args.hourly_rate) > 0.01:
                log(f"NOTE: actual rate is ${actual:.2f}/h, not the estimated "
                    f"${args.hourly_rate:.2f}/h")
            args.hourly_rate = actual
        log(f"pod {pod_id} created at ${args.hourly_rate:.2f}/h - "
            f"console: https://console.runpod.io/pods/{pod_id}")

        budget_hours = args.max_cost / args.hourly_rate if args.hourly_rate else 0
        log(f"budget: ${args.max_cost:.2f} = {budget_hours:.1f} h at this rate")

    _ACTIVE_POD = pod_id
    pod = Pod(pod_id, control_token)
    succeeded, reason = False, "did not start"

    try:
        running = wait_for(
            lambda: (api.get_pod(pod_id) or {}).get("desiredStatus") == "RUNNING",
            DEFAULTS["pod_ready_timeout"], 10, "the pod to start",
        )
        if not running:
            return _finish(pod, args, False, "the pod never reached RUNNING", started)

        log("pod is running; waiting for its control server")
        if not wait_for(pod.alive, DEFAULTS["server_timeout"], 10, "the control server"):
            return _finish(pod, args, False, "the control server never answered", started)
        log("control server is up")

        remote_cache = f"persist/{CACHE_NAME}"
        if pod.exists(remote_cache):
            log("cache already on the volume; the pod will use it directly")
        elif args.upload_cache:
            if not pod.upload(args.cache, remote_cache):
                return _finish(pod, args, False, "uploading the cache failed", started)
        else:
            log(f"pod will build the corpus itself from {args.sources}")
            log("this happens once; the result is kept on the volume")

        succeeded, reason = monitor(pod, args, started)
    except KeyboardInterrupt:
        succeeded, reason = False, "interrupted"
    finally:
        print()

    return _finish(pod, args, succeeded, reason, started)


def _finish(pod: Pod, args, succeeded: bool, reason: str, started: float) -> int:
    global _ACTIVE_POD

    elapsed = time.time() - started
    cost = elapsed / 3600 * (args.hourly_rate or 0)
    log(f"{'finished' if succeeded else 'STOPPING'}: {reason}")
    log(f"ran {elapsed / 60:.0f} min, about ${cost:.2f}")

    if status := pod.status():
        if not succeeded:
            for line in (status.get("train_log") or status.get("setup_log") or [])[-15:]:
                print(f"    {line}")

    log("collecting results")
    got_adapter = collect_results(pod, args.out, expect_adapter=succeeded)

    if succeeded and not got_adapter:
        # A few more cents of GPU is a trivial price next to losing the run.
        log("THE TRAINED ADAPTER COULD NOT BE DOWNLOADED - leaving the pod up")
        log(f"  retry:     python -m tools.runpod_infra.launch --adopt {pod.id}")
        log(f"  console:   https://console.runpod.io/pods/{pod.id}")
        log("  it is also on the network volume, so a later pod can fetch it")
        log("  when you have it:  python -m tools.runpod_infra.launch --cleanup")
        _ACTIVE_POD = None
        return 3

    if args.keep_pod:
        log(f"leaving pod {pod.id} running as asked - it is still billing")
        _ACTIVE_POD = None
    else:
        try:
            api.terminate_pod(pod.id)
            log(f"terminated pod {pod.id}")
        except Exception as error:
            log(f"COULD NOT TERMINATE {pod.id}: {error}")
            log("Terminate it by hand: https://console.runpod.io/pods")
            return 2
        _ACTIVE_POD = None

        remaining = [p for p in api.list_pods() if p.get("id") == pod.id]
        log("confirmed gone" if not remaining else "WARNING: pod still listed")

    if args.delete_volume and not args.no_volume:
        if volume := api.find_volume(args.volume_name):
            api.delete_volume(volume["id"])
            log(f"deleted network volume {args.volume_name}")

    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
