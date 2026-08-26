"""Find a datacentre that will actually rent us a GPU right now.

A network volume only attaches inside its own datacentre, so choosing where to
put it decides where every future run can go. RunPod's API reports stock as a
vague "low" or "medium" per GPU model with no regional breakdown, which is not
enough to choose on.

A rejected pod creation, though, is free and immediate - RunPod answers "no
instances currently available" in about two seconds and bills nothing. So
availability is measured by asking for a pod and reading the answer. Any request
that unexpectedly succeeds is terminated at once, in a finally block, because a
probe that leaves a GPU running would cost far more than it discovered.

    python -m tools.runpod_infra.probe                 # storage datacentres
    python -m tools.runpod_infra.probe --cloud SECURE
    python -m tools.runpod_infra.probe --gpu 'NVIDIA GeForce RTX 4090'
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.runpod_infra import api  # noqa: E402
from tools.runpod_infra import secrets as credentials  # noqa: E402

credentials.load()

DEFAULT_GPUS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA GeForce RTX 5090",
]

# Smallest thing that still exercises the same scheduler decision. It never runs
# anything: if one is ever created it is terminated immediately.
PROBE_IMAGE = "runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204"


def probe(datacenter: str, gpu_ids: list[str], cloud: str,
          disk: int = 10, volume_id: str | None = None) -> tuple[bool, str]:
    """Ask for a pod there. Returns (available, detail)."""
    spec = {
        "name": f"probe-{datacenter}",
        "imageName": PROBE_IMAGE,
        "cloudType": cloud,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": gpu_ids,
        "gpuTypePriority": "custom",
        "containerDiskInGb": disk,
        "dataCenterIds": [datacenter],
        # Never starts anything: the scheduler's answer is the whole point.
        "dockerStartCmd": ["bash", "-lc", "sleep 5"],
    }
    if volume_id:
        # Worth probing separately: community hosts are individually owned
        # machines and many cannot mount network storage at all, so attaching a
        # volume can rule out an entire cloud rather than a single host.
        spec["networkVolumeId"] = volume_id
        spec["volumeMountPath"] = "/workspace/persist"

    pod_id = None
    try:
        created = api.create_pod(spec)
        pod_id = created.get("id")
        return True, f"{created.get('costPerHr', 0):.2f}/h"
    except api.RunPodError as error:
        message = str(error)
        # Both of these mean the same thing - nothing free to give - but the
        # second arrives as an HTTP 500, which read as a server fault until the
        # body was looked at. Reporting it as an error made a whole sweep of
        # datacentres look broken when they were merely full.
        if ("no instances currently available" in message
                or "could not find any pods with required specifications" in message):
            return False, "no capacity"
        if "dataCenterIds/items/enum" in message:
            return False, "not a pod datacentre"
        return False, message.split(":")[-1].strip()[:60]
    finally:
        if pod_id:
            # A probe that leaves a GPU running costs more than it learned.
            for attempt in range(3):
                try:
                    api.terminate_pod(pod_id)
                    break
                except Exception:
                    time.sleep(2)
            else:
                print(f"  WARNING: could not terminate probe pod {pod_id} - "
                      "terminate it at https://console.runpod.io/pods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cloud", default="COMMUNITY",
                        choices=["COMMUNITY", "SECURE"])
    parser.add_argument("--gpu", action="append", help="GPU id, repeatable")
    parser.add_argument("--datacenter", help="probe only this one")
    parser.add_argument("--all-datacenters", action="store_true",
                        help="include those without network-volume support")
    parser.add_argument("--disk", type=int, default=10,
                        help="container disk to ask for; the usual reason a "
                             "community request cannot be placed")
    parser.add_argument("--volume", help="attach this network volume id")
    parser.add_argument("--volume-name", help="attach the volume with this name")
    parser.add_argument("--disk-sweep", action="store_true",
                        help="find the largest placeable disk in one datacentre")
    parser.add_argument("--stop-after", type=int, default=0,
                        help="stop once this many datacentres have capacity")
    args = parser.parse_args()

    gpu_ids = args.gpu or DEFAULT_GPUS

    volume_id = args.volume
    if args.volume_name and not volume_id:
        found_volume = api.find_volume(args.volume_name)
        if not found_volume:
            print(f"no volume named {args.volume_name}")
            return 1
        volume_id = found_volume["id"]
    centres = api.datacenters(storage_only=not args.all_datacenters)
    names = [d["id"] for d in centres]
    # GraphQL knows regions the pod endpoint will not accept; asking about
    # those wastes the sweep on schema errors instead of on capacity.
    if accepted := api.pod_datacenters():
        skipped = [n for n in names if n not in accepted]
        names = [n for n in names if n in accepted]
        if skipped:
            print(f"({len(skipped)} region(s) cannot host pods via this API: "
                  f"{', '.join(skipped)})")
    if args.datacenter:
        names = [args.datacenter]

    print(f"{len(names)} datacentre(s), {args.cloud}, "
          f"GPUs: {', '.join(g.split()[-1] for g in gpu_ids)}")
    print("a rejected request is free and takes about two seconds\n")

    if args.disk_sweep:
        target = names[0]
        print(f"disk sweep in {target}\n")
        for size in (20, 40, 60, 80, 100, 150):
            print(f"  {size:>4} GB  ", end="", flush=True)
            available, detail = probe(target, gpu_ids, args.cloud, size, volume_id)
            print("AVAILABLE" if available else f"-  {detail}")
            if not available:
                break
        return 0

    found = []
    for name in names:
        print(f"  {name:<12} ", end="", flush=True)
        available, detail = probe(name, gpu_ids, args.cloud, args.disk, volume_id)
        print("AVAILABLE" if available else f"-  {detail}")
        if available:
            found.append(name)
            if args.stop_after and len(found) >= args.stop_after:
                print("\n(stopping early as asked)")
                break

    print()
    if found:
        print(f"capacity in: {', '.join(found)}")
        print(f"\n  python -m tools.runpod_infra.launch --datacenter {found[0]} ...")
    else:
        print("no capacity anywhere for that request. Options:")
        print("  --cloud SECURE      more expensive, usually available")
        print("  --gpu '<name>'      widen or change the GPU list")
        print("  wait: community capacity moves within minutes")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
