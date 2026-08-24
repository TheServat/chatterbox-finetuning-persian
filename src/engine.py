"""One place that decides which Chatterbox engine a config asks for.

`train.py`, `src/preprocess_ljspeech.py` and `inference.py` all need the same
answer - Persian, Turbo or English - and used to each work it out themselves.
Keeping it here means adding a mode touches one file.

It also lets preprocessing skip loading T3. Preprocessing only needs the voice
encoder, the S3 tokenizer and the text tokenizer; T3 is 2 GB it never touches,
and preprocessing is the job that runs for hours over tens of thousands of
clips. `load_t3=False` swaps in a stub that carries only the config, which is
the single attribute the preprocessing loop reads from it.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src import compat  # noqa: F401  (applies the training/dtype patches)
from src.chatterbox_.models.t3.modules.t3_config import T3Config
from src.utils import setup_logger

logger = setup_logger(__name__)


class _T3Stub:
    """Stands in for T3 when only its hyperparameters are needed."""

    def __init__(self, hp: T3Config):
        self.hp = hp

    def __getattr__(self, name):
        raise AttributeError(
            f"T3 was not loaded (load_t3=False), so '{name}' is unavailable. "
            "Preprocessing only needs t3.hp; anything else means the engine "
            "should have been built with load_t3=True."
        )


def build_engine(cfg, device: str = "cpu", *, load_t3: bool = True):
    """Construct the engine this config describes, loaded from `cfg.model_dir`."""
    model_dir = Path(cfg.model_dir)

    if cfg.is_persian:
        from src.persian.engine import ChatterboxPersianTTS

        if load_t3:
            logger.info(
                f"Loading Persian engine: {cfg.t3_filename} + "
                f"{cfg.s3gen_filename} (vocab {cfg.new_vocab_size})"
            )
            return ChatterboxPersianTTS.from_local(
                model_dir,
                device,
                t3_filename=cfg.t3_filename,
                s3gen_filename=cfg.s3gen_filename,
                ve_filename=cfg.ve_filename,
                tokenizer_filename=cfg.tokenizer_filename,
                vocab_size=cfg.new_vocab_size,
            )

        logger.info("Loading Persian engine without T3 (preprocessing only)")
        return _build_persian_without_t3(cfg, model_dir, device)

    if cfg.is_turbo:
        from src.chatterbox_.tts_turbo import ChatterboxTurboTTS

        logger.info("Loading Turbo engine")
        return ChatterboxTurboTTS.from_local(str(model_dir), device=device)

    from src.chatterbox_.tts import ChatterboxTTS

    logger.info("Loading English engine")
    return ChatterboxTTS.from_local(str(model_dir), device=device)


def _build_persian_without_t3(cfg, model_dir: Path, device: str):
    from safetensors.torch import load_file as load_safetensors

    from src.chatterbox_.models.s3gen import S3Gen
    from src.chatterbox_.models.voice_encoder import VoiceEncoder
    from src.chatterbox_.mtl_tts import Conditionals
    from src.persian.engine import ChatterboxPersianTTS, PersianMTLTokenizer

    def _load(path: Path):
        if path.suffix == ".safetensors":
            return load_safetensors(str(path))
        return torch.load(path, weights_only=True)

    voice_encoder = VoiceEncoder()
    compat.load_state_dict_tolerant(
        voice_encoder, _load(model_dir / cfg.ve_filename), cfg.ve_filename
    )
    voice_encoder.to(device).eval()

    s3gen = S3Gen()
    skipped = compat.load_state_dict_tolerant(
        s3gen, _load(model_dir / cfg.s3gen_filename), cfg.s3gen_filename
    )
    if skipped:
        logger.info(f"  {cfg.s3gen_filename}: recomputed buffers {skipped}")
    s3gen.to(device).eval()

    tokenizer = PersianMTLTokenizer(str(model_dir / cfg.tokenizer_filename))

    conds = None
    if (builtin := model_dir / "conds.pt").exists():
        conds = Conditionals.load(builtin).to(device)

    stub = _T3Stub(T3Config(text_tokens_dict_size=cfg.new_vocab_size))
    return ChatterboxPersianTTS(stub, s3gen, voice_encoder, tokenizer, device, conds)
