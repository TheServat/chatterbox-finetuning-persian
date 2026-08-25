"""Re-apply the current Persian normaliser to an existing corpus.

The normaliser will keep changing - a reading gets fixed, a new edge case turns
up - and every change quietly invalidates part of the cache. `metadata.csv`
holds the raw text alongside the normalised text, so the stale rows can be found
exactly: re-normalise column 2 and compare against column 3.

Audio is untouched, so this is seconds rather than the hours a full rebuild
costs. Cached `.pt` files for changed rows are deleted; re-running
`python -m src.preprocess_ljspeech` then regenerates only those.

    python tools/renormalize.py            # report what changed
    python tools/renormalize.py --apply    # rewrite metadata and drop stale cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.persian.normalize import normalize  # noqa: E402

DEFAULT_METADATA = ROOT / "MyTTSDataset" / "metadata.csv"
DEFAULT_CACHE = ROOT / "MyTTSDataset" / "preprocess"


def scan(metadata: Path) -> tuple[list[tuple[str, str, str]], list[str], int]:
    """Return (changed rows, all rebuilt rows, total). Changed = (id, old, new)."""
    changed: list[tuple[str, str, str]] = []
    rebuilt: list[str] = []
    total = 0

    with metadata.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|", 2)
            if len(parts) != 3:
                continue
            clip_id, raw, current = parts
            total += 1

            # The pipe is the field separator, so it can never appear in a field.
            fresh = normalize(raw).replace("|", " ")
            if fresh != current:
                changed.append((clip_id, current, fresh))
            rebuilt.append(f"{clip_id}|{raw}|{fresh}")

    return changed, rebuilt, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--apply", action="store_true",
        help="rewrite metadata.csv and delete the cache entries that changed",
    )
    parser.add_argument("--show", type=int, default=10, help="examples to print")
    args = parser.parse_args()

    if not args.metadata.exists():
        print(f"{args.metadata} not found - build the corpus first.")
        return 1

    changed, rebuilt, total = scan(args.metadata)

    print(f"{total:,} rows, {len(changed):,} changed "
          f"({100 * len(changed) / total:.2f}%)" if total else "empty corpus")

    for clip_id, old, new in changed[: args.show]:
        print(f"\n  {clip_id}")
        print(f"    was: {old[:100]}")
        print(f"    now: {new[:100]}")
    if len(changed) > args.show:
        print(f"\n  ... and {len(changed) - args.show:,} more")

    if not changed:
        print("\nThe corpus already matches the current normaliser.")
        return 0

    if not args.apply:
        print("\nNothing written. Re-run with --apply to update.")
        return 0

    # Written in full and replaced atomically: a half-rewritten metadata.csv
    # would be worse than a stale one.
    temporary = args.metadata.with_suffix(".csv.tmp")
    temporary.write_text("\n".join(rebuilt) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(args.metadata)
    print(f"\nrewrote {args.metadata.name}")

    dropped = 0
    for clip_id, _, _ in changed:
        cached = args.cache / f"{clip_id}.pt"
        if cached.exists():
            cached.unlink()
            dropped += 1
    print(f"dropped {dropped} stale cache entries")
    print("\nRun `python -m src.preprocess_ljspeech` to regenerate just those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
