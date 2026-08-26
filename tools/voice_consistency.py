"""How close each sample is to the reference voice, and to the other draws.

Pronunciation converged before the voice did: at step 2900 all three draws said
the sentence correctly, and all three sounded like different people. Something
has to say how different, because "they sound different" cannot be tracked
across checkpoints or compared between inference settings.

Chatterbox already carries the instrument. The voice encoder that conditions
generation maps a clip to a speaker embedding, and it is frozen - never trained
here - so it is an outside judge rather than a mirror of what the model
believes. Two numbers come out of it:

  to reference   cosine similarity with the voice the samples were cloned from.
                 This is the one that matters: it is the actual goal.
  between draws  the lowest similarity between any two draws of one checkpoint.
                 Low means the same prompt produced different speakers, which is
                 what a listener hears as the voice wandering.

It runs on the CPU so training keeps the card. Absolute values depend on the
encoder, so read them against each other, not against any published figure.

    python tools/voice_consistency.py
    python tools/voice_consistency.py --checkpoint 2900
    python tools/voice_consistency.py --dir chatterbox_output/seed_spread
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import compat  # noqa: E402,F401
from src.config import TrainConfig  # noqa: E402
from src.utils import load_audio  # noqa: E402

SAMPLE_NAME = re.compile(r"checkpoint-(\d+)(?:_(\d+))?\.wav$")
VE_SAMPLE_RATE = 16000


def embed(encoder, path: Path) -> np.ndarray:
    # load_audio returns (tensor [1, samples], sample_rate)
    wav, _ = load_audio(str(path), VE_SAMPLE_RATE)
    audio = wav.squeeze(0).cpu().numpy()
    with torch.no_grad():
        vector = encoder.embeds_from_wavs([audio], sample_rate=VE_SAMPLE_RATE)
    vector = np.asarray(vector).squeeze()
    return vector / (np.linalg.norm(vector) + 1e-9)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def load_encoder():
    from src.chatterbox_.models.voice_encoder import VoiceEncoder
    from safetensors.torch import load_file

    config = TrainConfig()
    weights = Path(config.model_dir) / "ve.safetensors"
    encoder = VoiceEncoder()
    encoder.load_state_dict(load_file(str(weights)))
    return encoder.to("cpu").eval()


def grouped(directory: Path, only: int | None) -> dict:
    found = defaultdict(list)
    for path in sorted(directory.glob("*.wav")):
        if "trimmed" in path.name:
            continue
        match = SAMPLE_NAME.search(path.name)
        key = int(match.group(1)) if match else path.stem
        if only is not None and key != only:
            continue
        found[key].append(path)
    # Step numbers sort as numbers; anything unparsed sorts after them by name.
    return dict(sorted(
        found.items(),
        key=lambda kv: (0, kv[0], "") if isinstance(kv[0], int) else (1, 0, str(kv[0])),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", default="./chatterbox_output/inference_samples")
    parser.add_argument("--checkpoint", type=int)
    parser.add_argument("--reference")
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"nothing at {directory}")
        return 1

    by_step = grouped(directory, args.checkpoint)
    if not by_step:
        print("no samples to compare")
        return 1

    config = TrainConfig()
    reference_path = Path(args.reference or config.inference_prompt_path)
    print(f"reference voice: {reference_path}")

    encoder = load_encoder()
    reference = embed(encoder, reference_path)

    print(f"\n{'step':>8} {'draws':>6} {'to reference':>14} {'between draws':>15}")
    rows = []
    for step, paths in by_step.items():
        vectors = [embed(encoder, path) for path in paths]
        to_reference = [similarity(reference, v) for v in vectors]
        mean_to_reference = sum(to_reference) / len(to_reference)
        between = (
            min(similarity(a, b) for a, b in combinations(vectors, 2))
            if len(vectors) > 1 else None
        )
        rows.append((step, mean_to_reference, between))
        print(f"{str(step):>8} {len(paths):>6} {mean_to_reference:14.3f} "
              f"{(f'{between:.3f}' if between is not None else '-'):>15}")

    multi = [r for r in rows if r[2] is not None]
    if len(multi) > 1:
        first, last = multi[0], multi[-1]
        print(f"\nto reference:  {first[1]:.3f} at {first[0]} -> {last[1]:.3f} at {last[0]}")
        print(f"between draws: {first[2]:.3f} at {first[0]} -> {last[2]:.3f} at {last[0]}")
        print("\nBoth should climb as the model learns to hold a voice. The "
              "second is what a listener calls the voice wandering between takes.")
    elif multi:
        step, to_reference, between = multi[0]
        print(f"\nAt {step}: draws sit {to_reference:.3f} from the reference and "
              f"{between:.3f} from each other.")
        print("Nothing to compare it against yet - run again once another "
              "checkpoint has three draws.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
