"""Pick a GPU on price, and say what the run will cost before it starts.

Cheapest per hour is the wrong target. This job is compute-bound, not
VRAM-bound - it peaks at 4 GB on the development card - so the number that
decides the bill is hours x rate, and a card at twice the price that finishes
in a third of the time is cheaper *and* done sooner.

The throughput model is anchored on a real measurement rather than spec sheets:
the full 68k-clip corpus on a Quadro RTX 3000, giving 4.9 samples/s at batch 8
with accumulation 4. Every other card is scaled from that by relative bf16
tensor throughput, discounted for the fact that no training loop reaches peak
FLOPS.

Two things also improve on the reference card and are folded into the estimate:
gradient checkpointing can be switched off with 24 GB (it trades roughly a
third of the speed for memory), and Ampere and later run bf16 natively rather
than fp16 with a loss scaler.

    python -m tools.runpod_infra.plan                 # rank what is available
    python -m tools.runpod_infra.plan --epochs 3      # cost for a given run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.runpod_infra import api  # noqa: E402

# Measured on the full 68k-clip corpus on a Quadro RTX 3000, fp16, gradient
# checkpointing on. Two numbers came out of that, and the difference matters:
# 2.50 samples/s at batch 8 with no accumulation, and 4.9 at batch 8 x accum 4,
# because accumulation amortises the optimiser step. The production figure is
# the one used here.
REFERENCE_SAMPLES_PER_SEC = 4.9
REFERENCE_GPU = "Quadro RTX 3000"

# Relative training throughput, normalised to the reference card. Derived from
# dense bf16 tensor throughput and memory bandwidth, then held deliberately
# conservative: real speedups on a 520M model with short sequences land well
# below the FLOPS ratio.
RELATIVE_SPEED = {
    "RTX A4000": 3.0,
    "RTX A5000": 4.5,
    "RTX 3090": 5.5,
    "RTX 4000 Ada": 4.0,
    "A40": 6.0,
    "RTX A6000": 6.0,
    "RTX 4090": 9.0,
    "RTX 5080": 8.0,
    "RTX 5090": 12.0,
    "L40": 8.0,
    "L40S": 9.5,
    "RTX 6000 Ada": 9.0,
    "A100 PCIe": 12.0,
    "A100 SXM": 14.0,
    "H100 NVL": 20.0,
    "H100 SXM": 22.0,
    "H200 SXM": 26.0,
}

# bf16 needs Ampere or newer. fp16 trains this model correctly but needs a loss
# scaler and is the more fragile of the two, so pre-Ampere cards are excluded
# rather than merely deprioritised.
NO_BF16 = ("Tesla V100", "V100 SXM2", "RTX 3070", "RTX 4070 Ti", "RTX A2000")

MIN_VRAM_GB = 16          # 4 GB peak measured, plus room to drop checkpointing
CORPUS_CLIPS = 68_085


def candidates(gpus: list[dict], *, min_vram: int = MIN_VRAM_GB,
               cloud: str = "COMMUNITY") -> list[dict]:
    """GPUs that can run this job, priced for the cloud they will be rented from.

    `lowestPrice` is the cheapest across both clouds, which is the community
    price - renting the same card on SECURE costs roughly twice as much. Ranking
    on the wrong one produced a pod at $0.74/h against a $0.34 estimate.
    """
    ranked = []
    for gpu in gpus:
        # The per-cloud fields and lowestPrice disagree for some cards - H100
        # reports 1.00 against a lowest of 2.69 - and there is no way to tell
        # from here which is stale. Planning takes the higher of the two:
        # overestimating costs nothing, while underestimating is what produced
        # a $0.74/h pod against a $0.34 estimate.
        cloud_price = (gpu.get("secure_price") if cloud == "SECURE"
                       else gpu.get("community_price"))
        price = max(cloud_price or 0, gpu.get("on_demand") or 0) or None
        if not price or gpu["vram_gb"] < min_vram:
            continue
        if any(bad in gpu["name"] for bad in NO_BF16):
            continue
        speed = RELATIVE_SPEED.get(gpu["name"])
        if speed is None:
            continue  # unknown card: no honest estimate, so leave it out
        ranked.append({**gpu, "relative_speed": speed, "price": price})
    return ranked


def estimate(gpu: dict, epochs: float, clips: int = CORPUS_CLIPS) -> dict:
    samples_per_sec = REFERENCE_SAMPLES_PER_SEC * gpu["relative_speed"]
    seconds = clips * epochs / samples_per_sec
    hours = seconds / 3600
    return {
        **gpu,
        "samples_per_sec": samples_per_sec,
        "hours": hours,
        "cost": hours * gpu["price"],
    }


def rank(epochs: float, *, min_vram: int = MIN_VRAM_GB,
         cloud: str = "COMMUNITY") -> list[dict]:
    estimates = [
        estimate(g, epochs)
        for g in candidates(api.gpu_types(), min_vram=min_vram, cloud=cloud)
    ]
    estimates.sort(key=lambda e: e["cost"])
    return estimates


def local_estimate(epochs: float, clips: int = CORPUS_CLIPS) -> dict:
    seconds = clips * epochs / REFERENCE_SAMPLES_PER_SEC
    return {"hours": seconds / 3600, "samples_per_sec": REFERENCE_SAMPLES_PER_SEC}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--clips", type=int, default=CORPUS_CLIPS)
    parser.add_argument("--min-vram", type=int, default=MIN_VRAM_GB)
    parser.add_argument("--all", action="store_true", help="include out-of-stock GPUs")
    parser.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    args = parser.parse_args()

    estimates = [
        estimate(g, args.epochs, args.clips)
        for g in candidates(api.gpu_types(), min_vram=args.min_vram, cloud=args.cloud)
    ]
    estimates.sort(key=lambda e: e["cost"])

    reference = local_estimate(args.epochs, args.clips)
    print(f"{args.clips:,} clips x {args.epochs} epochs")
    print(f"locally on the {REFERENCE_GPU}: {reference['hours']:.1f} h "
          f"at {reference['samples_per_sec']:.2f} samples/s (measured)\n")

    print(f"{'GPU':<16} {'VRAM':>5} {'$/hr':>6} {'est. h':>7} {'est. $':>7} "
          f"{'stock':<12} cloud")
    print("-" * 72)
    for e in estimates:
        stock = (e.get("stock") or "?").lower()
        if not args.all and stock in ("none", "unavailable", "out of stock"):
            continue
        cloud = "secure" if e["secure"] else ""
        cloud += "+community" if e["community"] and cloud else ("community" if e["community"] else "")
        print(f"{e['name']:<16} {e['vram_gb']:>4}G {e['price']:>6.2f} "
              f"{e['hours']:>7.1f} {e['cost']:>7.2f} {stock:<12} {cloud}")

    if estimates:
        best = estimates[0]
        print(f"\ncheapest total: {best['name']} at ${best['cost']:.2f} "
              f"({best['hours']:.1f} h)")
        print(f"speedup over local: {best['relative_speed']:.0f}x, "
              f"saving {reference['hours'] - best['hours']:.0f} h")
        print("\nEstimates scale a real measurement by relative throughput; treat")
        print("them as a ranking, not a promise. The launcher re-measures in the")
        print("first minutes and reports the actual rate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
