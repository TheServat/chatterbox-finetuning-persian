"""Fold a trained LoRA adapter into a standalone T3 checkpoint.

Useful when the adapter should stop being a separate artifact: for sharing a
single file, for serving without PEFT installed, or for handing the result to
code that only knows how to load a state dict. `infer_fa.py` merges on the fly
anyway, so this is not needed just to listen to the model.

The merged file carries the enlarged vocabulary, so load it with a T3 built at
`cfg.new_vocab_size`, not at the base 2454.

    python merge_lora.py
    python merge_lora.py --adapter chatterbox_output/persian_adapter --out t3_fa.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import compat  # noqa: F401,E402
from src.config import TrainConfig  # noqa: E402
from src.utils import setup_logger  # noqa: E402

logger = setup_logger("merge-lora")


def main() -> int:
    cfg = TrainConfig()

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--adapter", type=Path,
        default=Path(cfg.output_dir) / "persian_adapter",
        help="adapter directory written by train.py",
    )
    parser.add_argument(
        "--out", type=Path,
        help="output .safetensors (default: <output_dir>/t3_fa_merged.safetensors)",
    )
    args = parser.parse_args()

    output = args.out or Path(cfg.output_dir) / "t3_fa_merged.safetensors"

    if not args.adapter.exists():
        logger.error(f"No adapter at {args.adapter}. Train first.")
        return 1

    from peft import PeftModel
    from safetensors.torch import save_file

    logger.info(
        f"Building the base at vocab {cfg.new_vocab_size} from {cfg.t3_filename}"
    )
    if cfg.is_persian:
        from src.persian.engine import load_persian_t3

        t3 = load_persian_t3(
            cfg.model_dir,
            cfg.t3_filename,
            vocab_size=cfg.new_vocab_size,
            tokenizer_path=Path(cfg.model_dir) / cfg.tokenizer_filename,
            # Merging touches weights only; the attention kernel is irrelevant
            # and sdpa builds faster.
            attn_implementation="sdpa",
        )
    else:
        from src.chatterbox_.models.t3.modules.t3_config import T3Config
        from src.chatterbox_.models.t3.t3 import T3
        from safetensors.torch import load_file

        from src.model import resize_and_load_t3_weights

        t3 = T3(hp=T3Config(text_tokens_dict_size=cfg.new_vocab_size))
        t3 = resize_and_load_t3_weights(
            t3, load_file(str(Path(cfg.model_dir) / cfg.t3_filename))
        )
        if cfg.is_turbo and hasattr(t3.tfmr, "wte"):
            logger.info("Turbo: removing the backbone wte to match training")
            del t3.tfmr.wte

    logger.info(f"Applying {args.adapter}")
    merged = PeftModel.from_pretrained(t3, str(args.adapter)).merge_and_unload()

    output.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in merged.state_dict().items()}
    save_file(state, str(output))

    size_gb = output.stat().st_size / 2**30
    logger.info(f"Wrote {output} ({size_gb:.2f} GB, vocab {cfg.new_vocab_size})")
    logger.info(
        "Load it into a T3 built at this vocabulary size - the base 2454 will "
        "not accept it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
