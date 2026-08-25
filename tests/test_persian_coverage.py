"""Exhaustive audit of Persian script coverage.

Two things must hold for every character Persian actually uses:

  1. `normalize()` must not silently delete or corrupt it.
  2. The result must survive Chatterbox's multilingual grapheme tokenizer with
     zero [UNK] - a single unknown character in a training transcript teaches
     the model to mispronounce that word for good.

Run directly (`python tests/test_persian_coverage.py`) for a readable report, or
under pytest. Re-run after every `tools/sync_upstream.py --update`: an upstream
tokenizer change is exactly the sort of thing that would break this quietly.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.persian.normalize import ZWNJ, normalize, unknown_characters  # noqa: E402

# The 32 letters of the Persian alphabet, in alphabetical order.
PERSIAN_ALPHABET = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"

# Written forms beyond the 32: alef-madda is a distinct vowel, and the hamza
# carriers spell everyday Persian words.
PERSIAN_EXTRA = {
    "آ": "alef madda - AAb (water)",
    "ء": "hamza - joz'",
    "ئ": "yeh with hamza - mas'ale, mas'ul",
    "ؤ": "waw with hamza - mo'assese, mo'asser",
    ZWNJ: "ZWNJ - mi-ravam, ketab-ha",
}

# Arabic-only spellings that appear in Persian text and must fold onto a
# Persian letter rather than reach the model as a separate token.
ARABIC_IN_PERSIAN = {
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه",
    "أ": "ا", "إ": "ا", "ٱ": "ا",
}

# Diacritics that must survive, because Persian pronunciation depends on them.
MUST_SURVIVE_MARKS = {
    "\u064B": "fathatan - the adverbial -an ending",
    "\u0653": "maddah - the madda of alef-madda",
}

# Optional short vowels: dropped on purpose, for consistency across corpora.
MUST_BE_STRIPPED_MARKS = {
    "\u064E": "fatha", "\u064F": "damma", "\u0650": "kasra",
    "\u0651": "shadda", "\u0652": "sukun", "\u0670": "superscript alef",
    "\u0640": "tatweel",
}

DIGIT_SETS = {
    "persian": "۰۱۲۳۴۵۶۷۸۹",
    "arabic-indic": "٠١٢٣٤٥٦٧٨٩",
    "ascii": "0123456789",
}

PUNCTUATION = "،؛؟.!?:-«»()\"'"

# Real sentences exercising the features that break naive normalisers.
SENTENCES = [
    ("plain", "سلام، حال شما چطور است؟"),
    ("zwnj", "می‌خواهم نیم‌فاصله‌ها را نگه دارم"),
    ("tanwin", "لطفاً مثلاً تقریباً واقعاً اصلاً حتماً"),
    ("ezafe-heh", "خانۀ من و خانهٔ تو"),
    ("hamza", "مسئله‌ی مؤسسه و جزء آخر"),
    ("alef-madda", "آب، آتش، آسمان و آرام"),
    ("harakat", "مُحَمَّدِ بْنِ عَبْدُاللّٰه"),
    ("arabic-forms", "كتاب‌هاي علمي و مسأله‌ي رياضي"),
    ("digits", "سال ۱۴۰۳ و ۲۵٪ و ۳۶.۵ درجه"),
    ("long-digits", "شماره ۰۹۱۲۳۴۵۶۷۸۹"),
    ("mixed-latin", "این یک تست AI و machine learning است"),
    ("quotes", "«سلام» گفت … و رفت"),
    ("all-letters", "ثابت ژرف ضخیم ظریف غبار قند چشم پرواز گل ذوق"),
]

# Text whose reading is not obvious, with the answer a Persian speaker expects.
# Each of these was wrong at some point during development.
READINGS = [
    ("thousands separator", "مبلغ ۱٬۲۵۰٬۰۰۰ ریال", "یک میلیون و دویست و پنجاه هزار"),
    ("decimal", "وزن ۷٫۵ کیلوگرم", "هفت ممیز پنج"),
    ("percent", "حدود ۴۵٪ رشد", "چهل و پنج درصد"),
    ("ordinal", "رتبه‌ی ۳م را گرفت", "سوم"),
    ("ordinal -min", "۱۰مین دوره", "دهمین"),
    ("clock time", "ساعت ۱۴:۳۰ قرار داریم", "چهارده و سی دقیقه"),
    ("whole hour", "ساعت ۹:۰۰ صبح", "ساعت نه صبح"),
    ("long digit run", "کد ملی ۰۰۱۲۳۴۵۶۷۸", "صفر صفر یک دو"),
    ("tanwin survives", "لطفاً بنشینید", "لطفاً"),
    ("ezafe unified", "خانۀ من", "خانه‌ی من"),
    ("arabic folded", "كتاب‌هاي علمي", "کتاب‌های علمی"),
]

# Characters that must not survive into a transcript: nothing pronounces them,
# and left in they reach the model as [UNK].
SILENT_JUNK = ["😀", "👍", "★", "‍", "﻿"]


def load_vocab() -> dict:
    for candidate in [
        ROOT / "pretrained_models" / "grapheme_mtl_merged_expanded_v1.json",
        ROOT / "pretrained_models" / "tokenizer.json",
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))["model"]["vocab"]
    raise FileNotFoundError(
        "No tokenizer found. Run `python tools/fetch_models.py` first."
    )


def _tokenize(vocab_path: Path, text: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(vocab_path))
    encoded = tok.encode(text.replace(" ", "[SPACE]"))
    return encoded.tokens.count("[UNK]"), len(encoded.ids)


def audit(verbose: bool = True) -> list[str]:
    """Return a list of failure descriptions (empty means everything passed)."""
    vocab = load_vocab()
    failures: list[str] = []

    def check(label: str, source: str, expect_in_output: str | None = None):
        out = normalize(source, final_punctuation=False)
        missing = unknown_characters(out, vocab)
        if missing:
            failures.append(
                f"{label}: out-of-vocab {sorted(f'U+{ord(c):04X}' for c in missing)}"
            )
        if expect_in_output is not None and expect_in_output not in out:
            failures.append(
                f"{label}: expected {expect_in_output!r} to survive, got {out!r}"
            )
        if verbose:
            mark = "FAIL" if missing else "ok  "
            print(f"  {mark} {label:22s} {source!r} -> {out!r}")
        return out

    print("\n[1] Persian alphabet - all 32 letters")
    for ch in PERSIAN_ALPHABET:
        check(unicodedata.name(ch, "?").replace("ARABIC LETTER ", ""), ch, ch)

    print("\n[2] Beyond the 32: madda, hamza carriers, ZWNJ")
    for ch, why in PERSIAN_EXTRA.items():
        check(why, f"با{ch}با", ch)

    print("\n[3] Arabic spellings fold onto their Persian letter")
    for src, dst in ARABIC_IN_PERSIAN.items():
        out = check(f"{src} -> {dst}", f"با{src}با")
        if src in out:
            failures.append(f"fold: {src!r} was not folded (got {out!r})")
        elif dst not in out:
            failures.append(f"fold: {src!r} did not become {dst!r} (got {out!r})")

    print("\n[4] Marks that must survive")
    for ch, why in MUST_SURVIVE_MARKS.items():
        out = check(why, f"لطف\u0627{ch}" if ch == "\u064B" else f"\u0627{ch}با")
        if ch == "\u064B" and ch not in out:
            failures.append(f"tanwin U+064B was stripped: {out!r}")

    print("\n[5] Marks that must be stripped")
    for ch, why in MUST_BE_STRIPPED_MARKS.items():
        out = normalize(f"با{ch}با", final_punctuation=False)
        status = "ok  " if ch not in out else "FAIL"
        if ch in out:
            failures.append(f"{why} U+{ord(ch):04X} was not stripped: {out!r}")
        if verbose:
            print(f"  {status} {why:22s} -> {out!r}")

    print("\n[6] Digits spell out in every script")
    for name, digits in DIGIT_SETS.items():
        out = normalize(f"عدد {digits[7]} است", final_punctuation=False)
        ok = "هفت" in out
        if not ok:
            failures.append(f"{name} digit 7 did not spell out: {out!r}")
        if verbose:
            print(f"  {'ok  ' if ok else 'FAIL'} {name:22s} -> {out!r}")

    print("\n[7] Punctuation stays in vocabulary")
    for ch in PUNCTUATION:
        if ch not in vocab:
            failures.append(f"punctuation {ch!r} (U+{ord(ch):04X}) missing from vocab")
    print(f"  {'ok  ' if not failures else '....'} {len(PUNCTUATION)} marks checked")

    print("\n[8] Whole sentences")
    for label, sentence in SENTENCES:
        check(label, sentence)

    print("\n[9] Readings that must come out a particular way")
    for label, source, expected in READINGS:
        out = normalize(source)
        ok = expected in out
        if not ok:
            failures.append(f"{label}: expected {expected!r} within {out!r}")
        if verbose:
            print(f"  {'ok  ' if ok else 'FAIL'} {label:22s} {source!r} -> {out!r}")

    print("\n[10] Silent characters are removed, not passed through")
    for junk in SILENT_JUNK:
        out = normalize(f"متن{junk}متن", final_punctuation=False)
        ok = junk not in out
        if not ok:
            failures.append(f"U+{ord(junk):04X} survived normalisation: {out!r}")
        if verbose:
            print(f"  {'ok  ' if ok else 'FAIL'} U+{ord(junk):04X} removed -> {out!r}")

    return failures


def test_persian_coverage():
    assert audit(verbose=False) == []


if __name__ == "__main__":
    problems = audit()
    print("\n" + "=" * 70)
    if problems:
        print(f"FAILURES ({len(problems)}):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("All Persian coverage checks passed.")
