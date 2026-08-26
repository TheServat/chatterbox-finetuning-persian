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


def _patch_is_multilingual() -> list[str]:
    """Make `is_multilingual` a lower bound instead of an exact vocabulary size.

    `T3Config.is_multilingual` tests `text_tokens_dict_size == 2454`, and T3
    only builds its `AlignmentStreamAnalyzer` when that is true. The analyzer is
    what watches the text-to-speech attention during generation and forces an
    EOS when decoding stalls or loops - the guard against runaway output on long
    input.

    Adding `[fa]` takes the vocabulary to 2455, so the equality fails and the
    guard silently switches off. Nothing errors; generation just loses its
    safety net. A vocabulary that has *grown* past the multilingual size is
    still multilingual, so the test becomes `>=`.
    """
    from src.chatterbox_.models.t3.modules.t3_config import T3Config

    if getattr(T3Config.is_multilingual, "_lower_bound", False):
        return []

    def is_multilingual(self) -> bool:
        return self.text_tokens_dict_size >= 2454

    prop = property(is_multilingual)
    prop.fget._lower_bound = True
    T3Config.is_multilingual = prop
    return ["T3Config.is_multilingual (>= 2454)"]


def _patch_alignment_repetition_guard() -> list[str]:
    """Hold the analyzer's token-repetition rule until the text has been spoken.

    AlignmentStreamAnalyzer forces an EOS the moment the last two speech tokens
    match, and in chatterbox-tts 0.1.7 as published the guard on that test is
    commented out:

        token_repetition = (
            # self.complete and
            len(self.generated_tokens) >= 3 and
            len(set(self.generated_tokens[-2:])) == 1
        )

    Its two siblings, long_tail and alignment_repetition, both keep theirs and
    both measure from `completed_at`. Two identical tokens in a row are not a
    stall - they are ordinary speech. Measured over 400 clips of this corpus,
    85% contain the pattern, 3.05% of all token positions are one, and the first
    lands a median of 31 tokens in: 1.2 s at 25 tokens a second. So generation
    gets cut about a second after it starts, whatever the model has learned.
    Both samples here fit that: 0.8 s at step 1250, and 6.2 s at step 1000 from
    the lucky tail of the same distribution.

    Rather than restage a ninety-line method that upstream may change, the token
    history is cleared while the text is unfinished, keeping it under the three
    entries the rule needs. This runs before the wrapped call, so it reads
    `complete` as of the previous frame; that delays a real detection by a frame
    or two and costs nothing, since the sibling rules cover a genuine stall.
    """
    from src.chatterbox_.models.t3.inference import alignment_stream_analyzer

    analyzer = alignment_stream_analyzer.AlignmentStreamAnalyzer
    if getattr(analyzer.step, "_repetition_needs_completion", False):
        return []

    original_step = analyzer.step

    def step(self, logits, next_token=None, *args, **kwargs):
        if not self.complete:
            del self.generated_tokens[:]
        return original_step(self, logits, next_token, *args, **kwargs)

    step._repetition_needs_completion = True
    analyzer.step = step
    return ["AlignmentStreamAnalyzer.step (repetition waits for completion)"]


def use_eager_attention(model) -> bool:
    """Switch a built transformer to eager attention.

    The alignment analyzer reads attention weights out of a forward hook, and
    `output_attentions=True` is ignored under sdpa - the weights never
    materialise, so `AlignmentStreamAnalyzer.step` stacks a list of `None`.
    Llama_520M hardcodes `attn_implementation="sdpa"`, and the implementation is
    bound when the module is built, so flipping the config afterwards is not
    enough.

    Eager attention is slower and heavier, so this is for inference only -
    training never constructs the analyzer.
    """
    tfmr = getattr(model, "tfmr", model)
    setter = getattr(tfmr, "set_attn_implementation", None)
    if setter is None:
        return False
    setter("eager")
    return getattr(tfmr.config, "_attn_implementation", None) == "eager"


def drop_inference_cache(module) -> list[str]:
    """Remove the wrappers T3.inference caches, wherever they sit in the tree.

    T3.inference builds a `patched_model` around the very layers the trainer
    owns and leaves it on the T3 instance. From then on every weight appears in
    the state dict twice under two names, and safetensors refuses to write
    tensors that share storage - so the *next* checkpoint save dies, long after
    the sample that caused it. Two runs ended exactly that way, the second at
    step 1500 having sampled at 1250.

    Walking the tree is the point. Under LoRA the caller holds a PeftModel and
    the cache sits on the T3 wrapped inside it, at
    `base_model.model.patched_model`. PEFT forwards attribute *reads* to the
    model it wraps but not deletes, so `delattr(peft_model, "patched_model")`
    raises, and setting the name on the wrapper instead leaves the real cache
    untouched - which is how the first attempt at this failed silently.

    Returns what it removed, so a caller can log it.
    """
    removed = []
    for owner in list(module.modules()):
        for name in ("patched_model", "compiled"):
            if name in owner._modules:
                del owner._modules[name]
            elif name in owner.__dict__:
                del owner.__dict__[name]
            else:
                continue
            removed.append(f"{type(owner).__name__}.{name}")
    return removed


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

    applied = (
        _patch_t3_training_hooks()
        + _patch_voice_encoder_dtype()
        + _patch_is_multilingual()
        + _patch_alignment_repetition_guard()
    )
    if quiet:
        applied += _silence_progress_bars()

    _APPLIED = True
    if verbose:
        for item in applied:
            print(f"[compat] patched {item}")
    return applied


apply()
