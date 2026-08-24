"""Add a `[fa]` language token to Chatterbox's multilingual tokenizer.

Chatterbox multilingual ships 23 language tokens; Persian is not one of them.
The tokenizer already covers every Persian *character* (see
`tests/test_persian_coverage.py`), so the only thing missing is the tag the
model reads as "speak this in language X".

Two ways to supply it:

  reuse `[ar]`      Zero vocabulary change, and the model has real Arabic
                    training behind that token. But Persian and Arabic share
                    only a script, not a phonology - Arabic has pharyngeals and
                    emphatic consonants Persian lacks, and Persian has vowels
                    Arabic does not. Training Persian onto `[ar]` also degrades
                    Arabic, since one embedding then serves two accents.

  add `[fa]`        One extra row (2454 -> 2455) initialised *from* `[ar]`, so
                    it starts where the closest known language sits and then
                    moves freely. Arabic is left untouched.

The second is the default. The initialisation matters: a mean-initialised row
starts at the centroid of 23 unrelated languages, whereas `[ar]` already encodes
"right-to-left Arabic script", which is most of the way there.

    python -m src.persian.tokenizer_fa          # build the tokenizer
    python -m src.persian.tokenizer_fa --check  # report without writing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BASE_TOKENIZER = ROOT / "pretrained_models" / "grapheme_mtl_merged_expanded_v1.json"
FA_TOKENIZER = ROOT / "pretrained_models" / "tokenizer_fa.json"

FA_TOKEN = "[fa]"
SEED_TOKEN = "[ar]"  # nearest neighbour in script, used to seed the embedding


def build_fa_tokenizer(
    base_path: Path = BASE_TOKENIZER,
    out_path: Path = FA_TOKENIZER,
    *,
    write: bool = True,
) -> dict:
    """Add `[fa]` as a special token and return a summary of the result."""
    if not base_path.exists():
        raise FileNotFoundError(
            f"{base_path} is missing. Run `python tools/fetch_models.py` first."
        )

    data = json.loads(base_path.read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]

    if SEED_TOKEN not in vocab:
        raise RuntimeError(
            f"{SEED_TOKEN} is absent from the base tokenizer - upstream changed "
            "its language tags, so the seeding strategy needs revisiting."
        )

    summary = {
        "base_vocab_size": len(vocab),
        "seed_token": SEED_TOKEN,
        "seed_token_id": vocab[SEED_TOKEN],
        "already_present": FA_TOKEN in vocab,
    }

    if FA_TOKEN in vocab:
        summary["fa_token_id"] = vocab[FA_TOKEN]
        summary["new_vocab_size"] = len(vocab)
        return summary

    fa_id = max(vocab.values()) + 1
    vocab[FA_TOKEN] = fa_id

    # The language tags are special tokens, which is why `[ar]` survives BPE as
    # a single unit. Without this entry `[fa]` would be split into four
    # characters and the tag would be meaningless.
    data.setdefault("added_tokens", []).append(
        {
            "id": fa_id,
            "content": FA_TOKEN,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )

    summary["fa_token_id"] = fa_id
    summary["new_vocab_size"] = len(vocab)

    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        summary["written_to"] = str(out_path)

    return summary


def seed_fa_embedding(state_dict: dict, fa_id: int, seed_id: int) -> dict:
    """Copy the seed language row into the new `[fa]` row, in place.

    Call after the embedding and output head have been grown to the new vocab
    size but before training starts. Operates on `text_emb.weight` and
    `text_head.weight`, the only two tensors indexed by text-token id.
    """
    for name in ("text_emb.weight", "text_head.weight"):
        if name not in state_dict:
            continue
        tensor = state_dict[name]
        if tensor.shape[0] <= max(fa_id, seed_id):
            raise ValueError(
                f"{name} has {tensor.shape[0]} rows, too few to hold id {fa_id}. "
                "Resize the embeddings before seeding."
            )
        tensor[fa_id].copy_(tensor[seed_id])
    return state_dict


def verify(out_path: Path = FA_TOKENIZER) -> list[str]:
    """Confirm `[fa]` survives encoding as a single token."""
    from tokenizers import Tokenizer

    problems = []
    tokenizer = Tokenizer.from_file(str(out_path))

    encoded = tokenizer.encode(f"{FA_TOKEN}سلام")
    if not encoded.tokens or encoded.tokens[0] != FA_TOKEN:
        problems.append(
            f"{FA_TOKEN} did not encode as one token: {encoded.tokens[:6]}"
        )

    baseline = tokenizer.encode(f"{SEED_TOKEN}سلام")
    if len(baseline.ids) != len(encoded.ids):
        problems.append(
            f"{FA_TOKEN} and {SEED_TOKEN} tokenise to different lengths "
            f"({len(encoded.ids)} vs {len(baseline.ids)}) - the tag is not "
            "behaving like the other language tags"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="report without writing a file"
    )
    parser.add_argument(
        "--base", type=Path, default=BASE_TOKENIZER, help="base tokenizer json"
    )
    parser.add_argument(
        "--out", type=Path, default=FA_TOKENIZER, help="where to write the result"
    )
    args = parser.parse_args()

    summary = build_fa_tokenizer(args.base, args.out, write=not args.check)

    print(f"base vocabulary      {summary['base_vocab_size']}")
    print(f"seed token           {summary['seed_token']} (id {summary['seed_token_id']})")
    if summary["already_present"]:
        print(f"{FA_TOKEN} already present at id {summary['fa_token_id']} - nothing to do")
        return 0
    print(f"{FA_TOKEN} added at id      {summary['fa_token_id']}")
    print(f"new vocabulary       {summary['new_vocab_size']}")

    if args.check:
        print("\n--check: nothing written.")
        return 0

    print(f"written to           {summary['written_to']}")

    problems = verify(args.out)
    for problem in problems:
        print(f"  FAIL {problem}")
    if problems:
        return 1

    print(f"  ok   {FA_TOKEN} encodes as a single token, like the other tags")
    print(
        f"\nSet new_vocab_size = {summary['new_vocab_size']} in src/config.py, "
        "then train.\nThe embedding rows are seeded from "
        f"{SEED_TOKEN} at model-build time."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
