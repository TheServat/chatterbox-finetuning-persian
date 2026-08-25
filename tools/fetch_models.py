"""Assemble `pretrained_models/` from whatever is already on this machine.

Chatterbox's weights are ~3.2 GB for the files this project needs, and they are
frequently already present - in the HuggingFace cache, or in another TTS project
on the same disk. Downloading them again is a waste of bandwidth and, more
importantly, of the 60 GB free on this drive.

So the order is: local search paths from the lock file, then the HuggingFace
cache, then the network. Local hits are hard-linked when the source sits on the
same volume, which costs zero extra bytes; otherwise they are copied.

Every candidate is checked against the size recorded in `versions.lock.json`
before being accepted - a half-finished download from a previous session is
exactly the kind of file that would otherwise load as a corrupt checkpoint
twenty minutes into preprocessing. `--verify-hashes` additionally checks
sha256, which is slower but conclusive.

    python tools/fetch_models.py                  # collect the pinned profile
    python tools/fetch_models.py --verify-hashes  # + sha256 every file
    python tools/fetch_models.py --plan           # report, download nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCK_PATH = ROOT / "versions.lock.json"
DEST_DIR = ROOT / "pretrained_models"


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def required_files(lock: dict) -> list[str]:
    """Filenames the pinned profile needs, in the order they are loaded."""
    profile = lock["weights"]["profile"]
    names = [
        profile[key] for key in ("t3", "s3gen", "ve", "conds", "tokenizer")
    ]
    names += profile.get("extras", [])
    return list(dict.fromkeys(names))


def hf_cache_roots() -> list[Path]:
    """Every plausible HuggingFace cache location on this machine."""
    roots = []
    for env in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env)
        if value:
            candidate = Path(value)
            roots += [candidate / "hub", candidate]
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return [r for r in roots if r.is_dir()]


def candidate_sources(lock: dict, filename: str) -> list[Path]:
    """Places this file might already exist, most specific first."""
    found: list[Path] = []

    for raw in lock.get("local_weight_search_paths", []):
        candidate = Path(raw) / filename
        if candidate.is_file():
            found.append(candidate)

    repo_dir = "models--" + lock["weights"]["repo_id"].replace("/", "--")
    for cache_root in hf_cache_roots():
        snapshots = cache_root / repo_dir / "snapshots"
        if snapshots.is_dir():
            for snapshot in snapshots.iterdir():
                candidate = snapshot / filename
                if candidate.is_file():
                    found.append(candidate)

    return found


def sha256_of(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def is_valid(path: Path, meta: dict, check_hash: bool) -> tuple[bool, str]:
    expected_size = meta.get("size")
    if expected_size is not None:
        actual = path.stat().st_size
        if actual != expected_size:
            return False, f"size {actual} != {expected_size}"

    expected_hash = meta.get("sha256")
    if check_hash and expected_hash:
        actual = sha256_of(path)
        if actual != expected_hash:
            return False, f"sha256 {actual[:12]} != {expected_hash[:12]}"
        return True, "size + sha256"

    return True, "size" if expected_size is not None else "present"


def place(source: Path, destination: Path) -> str:
    """Hard-link when possible, copy otherwise. Returns what was done."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        # Different volume, or a filesystem without hard links.
        shutil.copy2(source, destination)
        return "copied"


def salvage_partials(destination: Path) -> None:
    """Reuse bytes an earlier interrupted attempt already fetched.

    `huggingface_hub` leaves its work in `.cache/huggingface/download/*.incomplete`.
    Those are the same bytes from the same offset, so a 1 GB partial is worth
    carrying over rather than re-downloading.
    """
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        return

    cache = destination.parent / ".cache" / "huggingface" / "download"
    if not cache.is_dir():
        return

    candidates = [
        p for p in cache.glob("*.incomplete") if p.stat().st_size > 0
    ]
    if len(candidates) != 1:
        # Two partials and no way to tell which file each belongs to; starting
        # clean is safer than splicing the wrong bytes onto a checkpoint.
        return

    source = candidates[0]
    print(f"    resuming from a previous attempt ({human(source.stat().st_size)})")
    shutil.move(str(source), str(partial))


def _fast_transfer_available() -> bool:
    """Whether hf_transfer is installed and switched on."""
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") != "1":
        return False
    try:
        import hf_transfer  # noqa: F401

        return True
    except ImportError:
        return False


def download(lock: dict, filename: str, destination: Path) -> None:
    """Fetch one file into `pretrained_models/`, by whichever route suits the link.

    Two very different situations, and the wrong choice is expensive in both.

    On a flaky home connection, `hf_hub_download` is the wrong tool: it has no
    read timeout, so a socket that goes quiet without closing blocks forever -
    which happened twice here, both times pinned at exactly 1 GB.
    `tools/download.py` times out and resumes from a Range request instead.

    In a datacentre the opposite holds. That single-threaded resume managed
    about 4 MB/s on a rented pod, turning 3.3 GB of weights into thirteen
    minutes of paid GPU time doing nothing. hf_transfer opens parallel
    connections and is built for exactly that.
    """
    if _fast_transfer_available():
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=lock["weights"]["repo_id"],
            filename=filename,
            revision=lock["weights"]["revision"],
            token=os.environ.get("HF_TOKEN") or None,
            local_dir=str(destination.parent),
        )
        downloaded = Path(path)
        if downloaded.resolve() != destination.resolve():
            place(downloaded, destination)
        return

    from tools.download import download as resumable_download, hf_url

    salvage_partials(destination)
    resumable_download(
        hf_url(lock["weights"]["repo_id"], filename, lock["weights"]["revision"]),
        destination,
        expected_size=lock["weights"]["files"].get(filename, {}).get("size"),
        token=os.environ.get("HF_TOKEN") or None,
    )


def human(num_bytes: int | None) -> str:
    if not num_bytes:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="sha256 every file instead of trusting its size",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="report what would happen, without copying or downloading",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="skip building the [fa] tokenizer at the end",
    )
    args = parser.parse_args()

    lock = load_lock()
    known = lock["weights"]["files"]
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    to_download: list[tuple[str, dict]] = []
    print(f"Target: {DEST_DIR.relative_to(ROOT)}  (revision "
          f"{lock['weights']['revision'][:12]})\n")

    for filename in required_files(lock):
        meta = known.get(filename, {})
        destination = DEST_DIR / filename

        if destination.exists():
            ok, why = is_valid(destination, meta, args.verify_hashes)
            if ok:
                print(f"  have    {filename:42s} ({why})")
                continue
            print(f"  BAD     {filename:42s} {why} - replacing")
            if not args.plan:
                destination.unlink()

        placed = False
        for source in candidate_sources(lock, filename):
            ok, why = is_valid(source, meta, args.verify_hashes)
            if not ok:
                print(f"  skip    {filename:42s} {source} ({why})")
                continue
            if args.plan:
                print(f"  would link {filename:39s} <- {source}")
            else:
                how = place(source, destination)
                print(f"  {how:7s} {filename:42s} <- {source}")
            placed = True
            break

        if not placed:
            to_download.append((filename, meta))
            print(f"  MISSING {filename:42s} ({human(meta.get('size'))} to download)")

    if to_download:
        total = sum(m.get("size") or 0 for _, m in to_download)
        print(f"\n{len(to_download)} file(s) to download, {human(total)} total")
        if args.plan:
            print("--plan: nothing downloaded.")
            return 0
        if not os.environ.get("HF_TOKEN"):
            print("  (no HF_TOKEN set - fine for public repos)")
        for filename, meta in to_download:
            print(f"  downloading {filename} ({human(meta.get('size'))})...")
            download(lock, filename, DEST_DIR / filename)
            ok, why = is_valid(DEST_DIR / filename, meta, args.verify_hashes)
            print(f"    {'ok' if ok else 'FAILED'} ({why})")
            if not ok:
                return 1
    elif args.plan:
        print("\n--plan: everything is already in place.")
        return 0
    else:
        print("\nAll pinned weights are in place.")

    if not args.no_tokenizer:
        print("\nBuilding the Persian tokenizer...")
        sys.path.insert(0, str(ROOT))
        from src.persian.tokenizer_fa import build_fa_tokenizer, verify

        summary = build_fa_tokenizer()
        if summary["already_present"]:
            print(f"  [fa] already at id {summary['fa_token_id']}")
        else:
            print(
                f"  [fa] added at id {summary['fa_token_id']}, "
                f"vocab {summary['base_vocab_size']} -> {summary['new_vocab_size']}"
            )
        problems = verify()
        for problem in problems:
            print(f"  FAIL {problem}")
        if problems:
            return 1
        print(f"  ok   set new_vocab_size = {summary['new_vocab_size']} in src/config.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
