"""Runtime fixes layered over the vendored upstream package.

`src/chatterbox_/` is a disposable mirror of Resemble AI's release (see
`tools/sync_upstream.py`), so nothing here edits it. Everything the finetuning
pipeline needs on top of stock Chatterbox is applied at import time instead,
which means an upstream upgrade cannot silently drop one of these fixes.

What gets patched and why:

`T3.gradient_checkpointing_enable` / `T3.get_input_embeddings`
    HuggingFace `Trainer` calls both when `gradient_checkpointing=True`. T3 is a
    plain `nn.Module` wrapping a Llama backbone and exposes neither, so training
    fails at startup without them. Gradient checkpointing is not optional here -
    it is what makes a 520M model trainable on a 6 GB card.

`VoiceEncoder.embeds_from_wavs`
    Returns whatever dtype the mel pipeline produced. When that is float64 the
    speaker embedding reaches a float32 `T3Cond` and torch raises a dtype
    mismatch mid-batch, hours into preprocessing. Casting on the boundary is
    cheaper than debugging it later.

progress bars
    `flow_matching` and `t3.inference` wrap their inner loops in `tqdm`, which
    prints a bar per generated chunk. Fine interactively, unreadable in a
    training log with an inference callback. `quiet=True` neutralises them.

Import for the side effect, or call `apply()` explicitly:

    from src import compat
    compat.apply()
"""

from __future__ import annotations

import numpy as np

_APPLIED = False


def _patch_t3_training_hooks() -> list[str]:
    """Give T3 the two methods HuggingFace Trainer expects."""
    from src.chatterbox_.models.t3.t3 import T3

    applied = []

    if not hasattr(T3, "gradient_checkpointing_enable"):
        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.tfmr.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        T3.gradient_checkpointing_enable = gradient_checkpointing_enable
        applied.append("T3.gradient_checkpointing_enable")

    if not hasattr(T3, "get_input_embeddings"):
        def get_input_embeddings(self):
            # The Llama backbone owns the speech embeddings; `text_emb` is T3's
            # own table and the right answer once the backbone's wte is removed
            # (which the Turbo path does).
            try:
                return self.tfmr.get_input_embeddings()
            except (AttributeError, NotImplementedError):
                return self.text_emb

        T3.get_input_embeddings = get_input_embeddings
        applied.append("T3.get_input_embeddings")

    return applied


def _patch_voice_encoder_dtype() -> list[str]:
    """Pin the speaker embedding to float32 on both sides of the encoder."""
    from src.chatterbox_.models.voice_encoder.voice_encoder import VoiceEncoder

    if getattr(VoiceEncoder.embeds_from_wavs, "_float32_pinned", False):
        return []

    original = VoiceEncoder.embeds_from_wavs

    def embeds_from_wavs(self, wavs, sample_rate, *args, **kwargs):
        wavs = [np.asarray(w, dtype=np.float32) for w in wavs]
        embeds = original(self, wavs, sample_rate, *args, **kwargs)
        if isinstance(embeds, np.ndarray) and embeds.dtype != np.float32:
            embeds = embeds.astype(np.float32)
        return embeds

    embeds_from_wavs._float32_pinned = True
    VoiceEncoder.embeds_from_wavs = embeds_from_wavs
    return ["VoiceEncoder.embeds_from_wavs (float32)"]


def _silence_progress_bars() -> list[str]:
    """Replace the inference-loop tqdm wrappers with a pass-through."""
    import src.chatterbox_.models.s3gen.flow_matching as flow_matching
    import src.chatterbox_.models.t3.t3 as t3_module

    def passthrough(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    applied = []
    for module in (flow_matching, t3_module):
        if hasattr(module, "tqdm"):
            module.tqdm = passthrough
            applied.append(f"{module.__name__}.tqdm")
    return applied


def load_state_dict_tolerant(module, state_dict, what: str = "module"):
    """Load weights, allowing exactly the keys the module declares ignorable.

    S3Gen declares `ignore_state_dict_missing = ("tokenizer._mel_filters",
    "tokenizer.window")` - a mel filterbank and a Hann window, both recomputed
    deterministically in `__init__` - but nothing upstream ever acts on that
    declaration. It happens to work because `s3gen.pt` contains them;
    `s3gen_v3.pt` does not (2488 keys against 2490), so a strict load of the v3
    decoder fails outright.

    Plain `strict=False` would fix that and also swallow a genuinely truncated
    checkpoint. This honours the declaration and nothing beyond it: any other
    missing or unexpected key is still an error.
    """
    ignorable = set(getattr(module, "ignore_state_dict_missing", ()))
    result = module.load_state_dict(state_dict, strict=False)

    missing = [k for k in result.missing_keys if k not in ignorable]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"{what}: checkpoint does not match the model.\n"
            f"  missing:    {missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected: {list(result.unexpected_keys)[:8]}"
            f"{' ...' if len(result.unexpected_keys) > 8 else ''}"
        )

    return [k for k in result.missing_keys if k in ignorable]


def apply(quiet: bool = True, verbose: bool = False) -> list[str]:
    """Apply every patch once. Returns what was applied, for logging."""
    global _APPLIED
    if _APPLIED:
        return []

    applied = _patch_t3_training_hooks() + _patch_voice_encoder_dtype()
    if quiet:
        applied += _silence_progress_bars()

    _APPLIED = True
    if verbose:
        for item in applied:
            print(f"[compat] patched {item}")
    return applied


apply()
