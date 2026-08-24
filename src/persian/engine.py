"""The Persian layer over Chatterbox multilingual.

Nothing here edits `src/chatterbox_/` - see `tools/sync_upstream.py` for why.
Persian is added the way upstream would add a language, but from outside:

  * `fa` is registered in `SUPPORTED_LANGUAGES`, so `generate()` stops
    rejecting it.
  * `PersianMTLTokenizer` runs `src.persian.normalize` before the base
    tokenizer sees the text. Training and inference therefore normalise
    identically, which is the single easiest way to get a model that reads
    beautifully in evaluation and garbles real input.
  * `load_persian_t3` grows the text embedding by one row for `[fa]` and seeds
    it from `[ar]` rather than from the mean of 23 languages.

It also fixes a hard ceiling. `ChatterboxMultilingualTTS.generate` hardcodes
`max_new_tokens=1000`, and S3 speech tokens run at 25 Hz, so a single call can
never produce more than about 40 seconds - and quality degrades well before
that. `generate_long` splits Persian text on real sentence boundaries, renders
each piece against one set of speaker conditionals, and joins the results.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from src import compat  # noqa: F401  (applies the training/dtype patches)
from src.chatterbox_.models.t3.modules.t3_config import T3Config
from src.chatterbox_.models.t3.t3 import T3
from src.chatterbox_.models.tokenizers.tokenizer import MTLTokenizer
from src.chatterbox_.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS
from src.persian.normalize import normalize
from src.persian.tokenizer_fa import FA_TOKEN, SEED_TOKEN

LANGUAGE_ID = "fa"
BASE_VOCAB_SIZE = 2454   # multilingual checkpoints as shipped
FA_VOCAB_SIZE = 2455     # one more row, for [fa]


def register_language() -> None:
    """Teach the upstream engine that Persian exists.

    `generate()` validates `language_id` against this dict and raises for
    anything else, so registration has to happen before the first call. Editing
    a module-level registry is how upstream expects languages to be listed - it
    is a runtime registration, not a source patch, and survives a re-vendor.
    """
    SUPPORTED_LANGUAGES.setdefault(LANGUAGE_ID, "Persian")


register_language()


class PersianMTLTokenizer(MTLTokenizer):
    """MTLTokenizer with Persian normalisation wired into the preprocessing step."""

    def preprocess_text(
        self,
        raw_text: str,
        language_id: str | None = None,
        lowercase: bool = True,
        nfkd_normalize: bool = True,
    ):
        if language_id and language_id.lower() == LANGUAGE_ID:
            # Runs before the base class's NFKD pass, which is what composes
            # alef-madda before anything can strip its madda.
            raw_text = normalize(raw_text)
        return super().preprocess_text(
            raw_text,
            language_id=language_id,
            lowercase=lowercase,
            nfkd_normalize=nfkd_normalize,
        )


def load_persian_t3(
    model_dir: str | Path,
    t3_filename: str,
    *,
    vocab_size: int = FA_VOCAB_SIZE,
    tokenizer_path: str | Path | None = None,
    device: str = "cpu",
) -> T3:
    """Build a T3 sized for Persian and load the multilingual base into it.

    The shipped checkpoint has `BASE_VOCAB_SIZE` text rows. Growing to
    `FA_VOCAB_SIZE` adds exactly one, for `[fa]`, which is then copied from
    `[ar]`: same script and direction, so it starts far closer than the
    centroid of 23 unrelated languages would.
    """
    from safetensors.torch import load_file

    from src.model import resize_and_load_t3_weights

    model_dir = Path(model_dir)
    config = T3Config(text_tokens_dict_size=vocab_size)
    config.use_cache = False

    model = T3(hp=config)

    state = load_file(str(model_dir / t3_filename))
    if "model" in state:
        state = state["model"][0]
    model = resize_and_load_t3_weights(model, state)

    if vocab_size > BASE_VOCAB_SIZE:
        _seed_fa_row(model, tokenizer_path or model_dir / "tokenizer_fa.json")

    return model.to(device)


def _seed_fa_row(model: T3, tokenizer_path: str | Path) -> None:
    """Copy the `[ar]` embedding and output row into the `[fa]` slot.

    Runs after `resize_and_load_t3_weights`, which mean-initialises new rows and
    says so in its log. That mean value is overwritten here, so the log line
    below is the one that describes what `[fa]` actually ends up as.
    """
    import json

    from src.utils import setup_logger

    data = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    if FA_TOKEN not in vocab or SEED_TOKEN not in vocab:
        raise RuntimeError(
            f"{tokenizer_path} lacks {FA_TOKEN} or {SEED_TOKEN}. "
            "Run `python -m src.persian.tokenizer_fa` first."
        )

    fa_id, seed_id = vocab[FA_TOKEN], vocab[SEED_TOKEN]
    with torch.no_grad():
        for module in (model.text_emb, model.text_head):
            weight = module.weight
            if weight.shape[0] > max(fa_id, seed_id):
                weight[fa_id].copy_(weight[seed_id])

    setup_logger(__name__).info(
        f"{FA_TOKEN} (id {fa_id}) seeded from {SEED_TOKEN} (id {seed_id}), "
        "replacing the mean initialisation"
    )


class ChatterboxPersianTTS(ChatterboxMultilingualTTS):
    """Chatterbox multilingual, loaded with the Persian tokenizer and vocabulary."""

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        *,
        t3_filename: str = "t3_mtl23ls_v3.safetensors",
        s3gen_filename: str = "s3gen_v3.pt",
        ve_filename: str = "ve.safetensors",
        tokenizer_filename: str = "tokenizer_fa.json",
        vocab_size: int = FA_VOCAB_SIZE,
    ) -> "ChatterboxPersianTTS":
        from safetensors.torch import load_file as load_safetensors

        from src.chatterbox_.models.s3gen import S3Gen
        from src.chatterbox_.models.voice_encoder import VoiceEncoder
        from src.chatterbox_.mtl_tts import Conditionals

        ckpt_dir = Path(ckpt_dir)

        voice_encoder = VoiceEncoder()
        ve_path = ckpt_dir / ve_filename
        ve_state = (
            load_safetensors(str(ve_path))
            if ve_path.suffix == ".safetensors"
            else torch.load(ve_path, weights_only=True)
        )
        compat.load_state_dict_tolerant(voice_encoder, ve_state, ve_filename)
        voice_encoder.to(device).eval()

        t3 = load_persian_t3(
            ckpt_dir,
            t3_filename,
            vocab_size=vocab_size,
            tokenizer_path=ckpt_dir / tokenizer_filename,
            device=device,
        )
        t3.eval()

        s3gen = S3Gen()
        s3gen_path = ckpt_dir / s3gen_filename
        s3gen_state = (
            load_safetensors(str(s3gen_path))
            if s3gen_path.suffix == ".safetensors"
            else torch.load(s3gen_path, weights_only=True)
        )
        compat.load_state_dict_tolerant(s3gen, s3gen_state, s3gen_filename)
        s3gen.to(device).eval()

        tokenizer = PersianMTLTokenizer(str(ckpt_dir / tokenizer_filename))

        conds = None
        if (builtin := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin).to(device)

        return cls(t3, s3gen, voice_encoder, tokenizer, device, conds=conds)

    # ---------------------------------------------------------------- long form

    def generate_long(
        self,
        text: str,
        *,
        audio_prompt_path: str | Path | None = None,
        max_chunk_chars: int = 200,
        gap_seconds: float = 0.12,
        language_id: str = LANGUAGE_ID,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Render text of any length by rendering sentences and joining them.

        A single `generate()` call is capped at 1000 speech tokens - about 40
        seconds at 25 Hz - and drifts long before reaching that. Conditionals
        are prepared once and reused, so every chunk speaks in the same voice.
        """
        chunks = split_persian(text, max_chars=max_chunk_chars)
        if not chunks:
            raise ValueError("nothing to speak")

        if audio_prompt_path is not None:
            self.prepare_conditionals(
                str(audio_prompt_path),
                exaggeration=generate_kwargs.get("exaggeration", 0.5),
            )
        elif self.conds is None:
            raise ValueError(
                "no speaker conditionals: pass audio_prompt_path, or call "
                "prepare_conditionals() first"
            )

        gap = np.zeros(int(gap_seconds * self.sr), dtype=np.float32)
        pieces: list[np.ndarray] = []

        for index, chunk in enumerate(chunks):
            wav = self.generate(chunk, language_id=language_id, **generate_kwargs)
            audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
            if index:
                pieces.append(gap)
            pieces.append(audio)

        return torch.from_numpy(np.concatenate(pieces)).unsqueeze(0)


# -------------------------------------------------------------------- splitting

# Persian ends sentences with the Latin full stop and exclamation mark but its
# own question mark, and uses the Arabic comma and semicolon for clause breaks.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?؟])\s+")
_CLAUSE_BREAK = re.compile(r"(?<=[،؛:])\s+")


def split_persian(text: str, max_chars: int = 200) -> list[str]:
    """Split Persian text into chunks small enough to render in one call.

    Sentences first; any sentence still over the limit is broken at clause
    marks, and only then at word boundaries. Splitting mid-word would put an
    audible seam in the middle of a word, so that never happens.
    """
    text = normalize(text)
    if not text:
        return []

    chunks: list[str] = []
    for sentence in _SENTENCE_BREAK.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue

        buffer = ""
        for clause in _CLAUSE_BREAK.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) > max_chars:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(_split_on_words(clause, max_chars))
                continue
            if not buffer:
                buffer = clause
            elif len(buffer) + 1 + len(clause) <= max_chars:
                buffer = f"{buffer} {clause}"
            else:
                chunks.append(buffer)
                buffer = clause
        if buffer:
            chunks.append(buffer)

    return chunks


def _split_on_words(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    buffer = ""
    for word in text.split():
        if not buffer:
            buffer = word
        elif len(buffer) + 1 + len(word) <= max_chars:
            buffer = f"{buffer} {word}"
        else:
            chunks.append(buffer)
            buffer = word
    if buffer:
        chunks.append(buffer)
    return chunks
