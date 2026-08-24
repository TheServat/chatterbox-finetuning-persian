"""Synthesise Persian speech from a finetuned Chatterbox.

Loads the multilingual base, applies the trained LoRA adapter (or a merged
checkpoint), and writes a wav. Text of any length works: `--long` splits on
Persian sentence boundaries and renders each piece against one set of speaker
conditionals, because a single `generate()` call is capped at 1000 speech
tokens - about 40 seconds at 25 Hz - and drifts well before that.

    python infer_fa.py --text "سلام، حال شما چطور است؟"
    python infer_fa.py --text-file article.txt --long --out article.wav
    python infer_fa.py --text "..." --voice speaker_reference/2.wav
    python infer_fa.py --text "..." --base-only      # untrained, for comparison

Voice: pass `--voice` to clone from a reference clip, or omit it to use the
built-in conditionals from `conds.pt`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import compat  # noqa: F401,E402
from src.config import TrainConfig  # noqa: E402
from src.persian.engine import ChatterboxPersianTTS, split_persian  # noqa: E402
from src.persian.normalize import normalize  # noqa: E402
from src.utils import setup_logger  # noqa: E402

logger = setup_logger("infer-fa")

DEFAULT_TEXT = (
    "سلام، این یک آزمایش برای مدل گفتار فارسی است. "
    "امیدوارم صدای طبیعی و روانی داشته باشد."
)


def load_engine(cfg: TrainConfig, adapter: Path | None, device: str):
    """Base model, plus the adapter if one was trained."""
    engine = ChatterboxPersianTTS.from_local(
        cfg.model_dir,
        device,
        t3_filename=cfg.t3_filename,
        s3gen_filename=cfg.s3gen_filename,
        ve_filename=cfg.ve_filename,
        tokenizer_filename=cfg.tokenizer_filename,
        vocab_size=cfg.new_vocab_size,
    )

    if adapter is not None:
        from peft import PeftModel

        logger.info(f"Applying adapter: {adapter}")
        # merge_and_unload folds the LoRA deltas into the base weights, so
        # generation runs at full speed with no adapter overhead.
        engine.t3 = PeftModel.from_pretrained(engine.t3, str(adapter))
        engine.t3 = engine.t3.merge_and_unload()
        engine.t3.to(device).eval()

    return engine


def main() -> int:
    cfg = TrainConfig()

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Persian text to speak")
    source.add_argument("--text-file", type=Path, help="read the text from a file")

    parser.add_argument("--out", type=Path, default=Path("output_fa.wav"))
    parser.add_argument("--voice", type=Path, help="reference wav to clone")
    parser.add_argument(
        "--adapter", type=Path,
        default=Path(cfg.output_dir) / "persian_adapter",
        help="trained LoRA adapter directory",
    )
    parser.add_argument(
        "--base-only", action="store_true",
        help="skip the adapter - useful to hear what training actually changed",
    )
    parser.add_argument(
        "--long", action="store_true",
        help="split into sentences and join, for text beyond ~40 s of speech",
    )
    parser.add_argument("--max-chunk-chars", type=int, default=200)
    parser.add_argument("--gap", type=float, default=0.12,
                        help="silence between chunks, in seconds")

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, help="fix the sampling seed")
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    else:
        text = args.text or DEFAULT_TEXT

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    adapter = None if args.base_only else args.adapter
    if adapter is not None and not adapter.exists():
        logger.warning(
            f"No adapter at {adapter} - generating from the untrained base. "
            "Persian will not sound right yet; train first, or pass --base-only "
            "to silence this."
        )
        adapter = None

    normalized = normalize(text)
    logger.info(f"Normalised: {normalized[:110]}{'...' if len(normalized) > 110 else ''}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    logger.info(f"Loading engine on {device}...")
    engine = load_engine(cfg, adapter, device)

    params = dict(
        temperature=args.temperature,
        cfg_weight=args.cfg_weight,
        exaggeration=args.exaggeration,
        repetition_penalty=args.repetition_penalty,
    )

    started = time.monotonic()

    if args.long:
        chunks = split_persian(normalized, max_chars=args.max_chunk_chars)
        logger.info(f"{len(chunks)} chunk(s)")
        wav = engine.generate_long(
            normalized,
            audio_prompt_path=args.voice,
            max_chunk_chars=args.max_chunk_chars,
            gap_seconds=args.gap,
            **params,
        )
    else:
        if args.voice:
            engine.prepare_conditionals(
                str(args.voice), exaggeration=args.exaggeration
            )
        wav = engine.generate(normalized, language_id=cfg.language_id, **params)

    audio = wav.squeeze(0).cpu().numpy()
    duration = len(audio) / engine.sr
    elapsed = time.monotonic() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), audio, engine.sr)

    logger.info(
        f"Wrote {args.out}  ({duration:.1f} s of audio in {elapsed:.1f} s, "
        f"{duration / max(elapsed, 1e-6):.2f}x realtime)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
