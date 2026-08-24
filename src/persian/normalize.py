"""Persian (Farsi) text normalisation for Chatterbox finetuning.

Chatterbox's multilingual grapheme tokenizer already covers every Persian
letter, the ZWNJ, Persian digits and Persian punctuation (verified: 0 UNK on
round-trip). What it does *not* do is spell numbers out or unify the several
Unicode ways of writing the same Persian letter, so two identical-sounding
strings can reach the model as different token sequences. That inconsistency is
what this module removes.

The pipeline, in order:

    presentation forms -> NFKC
    strip bidi/format controls (but keep U+200C ZWNJ, which is phonemic)
    fold Arabic letter variants onto their Persian spellings
    drop tatweel and (optionally) harakat
    Arabic-Indic / Persian digits -> ASCII -> spelled-out Persian words
    normalise punctuation and whitespace
    tidy ZWNJ placement

`normalize` is the entry point; `punc_norm_fa` replaces Chatterbox's
English-centric `punc_norm` (which upper-cases the first letter - a no-op on
Persian - and only knows Latin sentence enders).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

ZWNJ = "‌"

# Format characters that carry no sound. ZWNJ is deliberately absent: in Persian
# it separates morphemes (mi-ravam, ketab-ha) and changes the pronunciation.
_BIDI_AND_FORMAT = dict.fromkeys(
    ord(c) for c in "​‍‎‏‪‫‬‭‮﻿⁦⁧⁨⁩"
)

# Arabic spellings that Persian writes differently. Folding these makes the same
# word tokenise identically no matter which keyboard produced it.
_LETTER_FOLD = {
    "ي": "ی",  # ARABIC YEH        -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA      -> FARSI YEH
    "ے": "ی",  # YEH BARREE (Urdu) -> FARSI YEH
    "ك": "ک",  # ARABIC KAF        -> KEHEH
    "ڪ": "ک",  # SWASH KAF         -> KEHEH
    "ة": "ه",  # TEH MARBUTA       -> HEH
    "ۀ": "ه",  # HEH WITH YEH ABOVE-> HEH
    "أ": "ا",  # ALEF WITH HAMZA ABOVE
    "إ": "ا",  # ALEF WITH HAMZA BELOW
    "ٱ": "ا",  # ALEF WASLA
    "ۍ": "ی",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}
# Kept as-is on purpose: A (aa) is a distinct vowel; hamza on waw/yeh spells
# real Persian words (mo'assese, mas'ale).
_LETTER_FOLD_TABLE = str.maketrans(_LETTER_FOLD)

# Harakat / tashkeel. Datasets diacritise inconsistently, so leaving them in
# would split one word across several token sequences.
_HARAKAT = re.compile(r"[ً-ْٰٕٖٟٓٗ٘ـ]")

_PUNCT_MAP = {
    "…": "،",   # ... -> Persian comma
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-",
    "×": " در ",     # x -> "dar"
    "٪": " درصد ",  # % -> "darsad"
    "%": " درصد ",
    "٫": ".",   # Arabic decimal separator
    "٬": ",",   # Arabic thousands separator
    "؛": "؛", "،": "،", "؟": "؟",
}
_PUNCT_TABLE = str.maketrans(_PUNCT_MAP)

# ASCII punctuation that Persian text should carry in its Persian form, so the
# model sees one prosodic cue per pause type instead of two.
_ASCII_TO_FA_PUNCT = {",": "،", ";": "؛", "?": "؟"}
_ASCII_TO_FA_TABLE = str.maketrans(_ASCII_TO_FA_PUNCT)

_SENTENCE_ENDERS = ".!؟،؛:-"

_PERSIAN_LETTERS = re.compile(r"[ء-ۿ]")
_LATIN_LETTERS = re.compile(r"[A-Za-z]")

# A run of digits, optionally with thousands separators and one decimal part.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

_DIGIT_WORDS = ["صفر", "یک", "دو", "سه",
                "چهار", "پنج", "شش",
                "هفت", "هشت", "نه"]

# Read groups longer than this digit-by-digit (phone numbers, IDs, account
# numbers) - "one billion two hundred..." is not what a speaker would say.
_SPELL_OUT_ABOVE = 9


def _num_to_words(token: str) -> str:
    """Spell one numeric token out in Persian words."""
    from num2words import num2words

    token = token.replace(",", "")
    if not token:
        return token

    if "." in token:
        int_part, _, frac_part = token.partition(".")
        head = _num_to_words(int_part) if int_part else _DIGIT_WORDS[0]
        tail = " ".join(_DIGIT_WORDS[int(d)] for d in frac_part)
        return f"{head} ممیز {tail}"  # "momayyez"

    if len(token) > _SPELL_OUT_ABOVE:
        return " ".join(_DIGIT_WORDS[int(d)] for d in token)

    try:
        return num2words(int(token), lang="fa")
    except (NotImplementedError, OverflowError, ValueError):
        return " ".join(_DIGIT_WORDS[int(d)] for d in token)


def spell_numbers(text: str) -> str:
    """Replace every digit run with its Persian reading."""
    return _NUMBER.sub(lambda m: f" {_num_to_words(m.group(0))} ", text)


def _tidy_zwnj(text: str) -> str:
    text = re.sub(ZWNJ + "+", ZWNJ, text)
    text = re.sub(r"\s*" + ZWNJ + r"\s*", ZWNJ, text)          # never beside a space
    text = re.sub(r"(^|\s)" + ZWNJ, r"\1", text)               # never word-initial
    text = re.sub(ZWNJ + r"($|\s)", r"\1", text)               # never word-final
    text = re.sub(ZWNJ + r"([،؛؟.!?:\"'()\[\]-])", r"\1", text)
    return text


def punc_norm_fa(text: str) -> str:
    """Persian counterpart of Chatterbox's `punc_norm`.

    Collapses whitespace, drops space before a closing punctuation mark, and
    guarantees a sentence-final mark so the model gets a consistent stop cue.
    """
    if not text:
        return ""

    text = " ".join(text.split())
    text = re.sub(r"\s+([،؛؟.!?:])", r"\1", text)
    text = re.sub(r"([،؛؟.!?:])(?=[^\s،؛؟.!?:])", r"\1 ", text)
    text = re.sub(r"([،؛؟.!?:])\1+", r"\1", text)
    text = text.strip()

    if text and not text.endswith(tuple(_SENTENCE_ENDERS)):
        text += "."
    return text


def normalize(
    text: str,
    *,
    spell_out_numbers: bool = True,
    strip_diacritics: bool = True,
    fa_punctuation: bool = True,
    final_punctuation: bool = True,
) -> str:
    """Normalise one Persian utterance for training or inference.

    Both sides must use identical settings: text normalised one way at training
    time and another at inference is the classic source of a model that reads
    fluently in evaluation and garbles real input.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    # NFKC has already expanded U+2026 to three dots, so the ellipsis has to be
    # caught here rather than through _PUNCT_MAP.
    text = re.sub(r"\.{2,}", "، ", text)
    text = text.translate(_BIDI_AND_FORMAT)
    text = text.translate(_LETTER_FOLD_TABLE)

    if strip_diacritics:
        text = _HARAKAT.sub("", text)

    text = text.translate(_PUNCT_TABLE)
    if fa_punctuation:
        text = text.translate(_ASCII_TO_FA_TABLE)

    if spell_out_numbers:
        text = spell_numbers(text)

    text = _tidy_zwnj(text)
    text = " ".join(text.split())

    return punc_norm_fa(text) if final_punctuation else text.strip()


def persian_ratio(text: str) -> float:
    """Share of letters that are Persian/Arabic script (0.0 - 1.0)."""
    fa = len(_PERSIAN_LETTERS.findall(text))
    latin = len(_LATIN_LETTERS.findall(text))
    total = fa + latin
    return fa / total if total else 0.0


def unknown_characters(text: str, vocab: Iterable[str]) -> set[str]:
    """Characters of `text` absent from a tokenizer vocabulary."""
    known = set(vocab)
    return {c for c in text if c not in known and not c.isspace()}
