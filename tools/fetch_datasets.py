"""Download the Persian corpora listed in `versions.lock.json`.

Only sources whose `local_dir` is missing are fetched, so re-running is cheap.
Sizes are printed before anything starts: this drive has finite room and
Persian-Farsi-Speech alone is 46 GB, which is not a download to begin by
accident.

    python tools/fetch_datasets.py --list      what is configured, what is local
    python tools/fetch_datasets.py             fetch everything still missing
    python tools/fetch_datasets.py yoda        fetch one source by short name

Set HF_TOKEN in the environment for gated repositories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "versions.lock.json"

# Short names for the command line, mapped to the repo ids in the lock file.
ALIASES = {
    "yoda": "Thomcles/YodaLingua-Farsi",
    "mana": "MahtaFetrat/Mana-TTS",
    "narration": "pymmdrza/PERSIAN_FARSI_NARRATION",
    "youtube": "Thomcles/Persian-Farsi-Speech",
}


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def resolve(name: str, datasets: dict) -> str:
    if name in datasets:
        return name
    if name in ALIASES and ALIASES[name] in datasets:
        return ALIASES[name]
    matches = [k for k in datasets if name.lower() in k.lower()]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        f"Unknown dataset {name!r}. Configured: {', '.join(sorted(datasets))}"
    )


def local_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def remote_size(repo_id: str) -> int:
    import urllib.request

    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main?recursive=1"
    request = urllib.request.Request(url)
    if token := os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        tree = json.loads(response.read().decode("utf-8"))
    return sum(
        (entry.get("lfs") or {}).get("size") or entry.get("size") or 0
        for entry in tree
        if entry.get("type") == "file"
    )


def cmd_list(datasets: dict) -> int:
    reverse = {v: k for k, v in ALIASES.items()}
    print(f"{'alias':<11} {'repo':<38} {'local':>9}  path")
    for repo_id, meta in datasets.items():
        path = ROOT / meta["local_dir"] if not Path(meta["local_dir"]).is_absolute() \
            else Path(meta["local_dir"])
        size = local_size(path)
        state = human(size) if size else "-"
        print(
            f"{reverse.get(repo_id, ''):<11} {repo_id:<38} {state:>9}  "
            f"{meta['local_dir']}"
        )
        if note := meta.get("note"):
            print(f"{'':<11} {note}")
    return 0


def remote_files(repo_id: str) -> list[tuple[str, int]]:
    """(path, size) for every file in a dataset repo."""
    import urllib.request

    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main?recursive=1"
    request = urllib.request.Request(url)
    if token := os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        tree = json.loads(response.read().decode("utf-8"))
    return [
        (e["path"], (e.get("lfs") or {}).get("size") or e.get("size") or 0)
        for e in tree
        if e.get("type") == "file"
    ]


def fetch(repo_id: str, meta: dict, force: bool) -> bool:
    # Deliberately not snapshot_download: it has no read timeout, so a
    # connection that goes quiet without closing blocks forever. That happened
    # here at 14 MB of 1 GB. tools/download.py times out and resumes.
    sys.path.insert(0, str(ROOT))
    from tools.download import download, hf_url

    raw = meta["local_dir"]
    target = Path(raw) if Path(raw).is_absolute() else ROOT / raw

    if meta.get("type") == "local_wav":
        if target.exists():
            print(f"  {repo_id}: already on disk at {raw}")
            return True
        print(f"  {repo_id}: marked local_wav but {raw} is missing - skipping")
        return False

    token = os.environ.get("HF_TOKEN") or None
    files = remote_files(repo_id)

    # "The directory is non-empty" is not the same as "the dataset is here": an
    # interrupted run leaves a README and nothing else, which is exactly what
    # happened before. Completeness is judged per file, against the remote sizes.
    missing = [
        (path, size)
        for path, size in files
        if force
        or not (target / path).exists()
        or ((target / path).stat().st_size != size and size)
    ]

    if not missing:
        print(f"  {repo_id}: complete ({len(files)} files, {human(local_size(target))})")
        return True

    print(f"  {repo_id}: {len(missing)} of {len(files)} files to fetch -> {raw}")
    for path, size in missing:
        print(f"    {path} ({human(size)})")
        download(
            hf_url(repo_id, path, repo_type="dataset"),
            target / path,
            expected_size=size or None,
            token=token,
        )

    print(f"    done ({human(local_size(target))})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", help="dataset aliases or repo ids")
    parser.add_argument("--list", action="store_true", help="report, download nothing")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    datasets = load_lock()["datasets"]

    if args.list:
        return cmd_list(datasets)

    selected = (
        [resolve(name, datasets) for name in args.names]
        if args.names
        else list(datasets)
    )

    pending = []
    for repo_id in selected:
        meta = datasets[repo_id]
        raw = meta["local_dir"]
        target = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        if meta.get("type") != "local_wav" and (
            args.force or not target.exists() or local_size(target) == 0
        ):
            pending.append(repo_id)

    if pending:
        print("To download:")
        total = 0
        for repo_id in pending:
            try:
                size = remote_size(repo_id)
            except Exception as exc:
                size = 0
                print(f"  {repo_id}: size unknown ({type(exc).__name__})")
            total += size
            print(f"  {repo_id:<40} {human(size) if size else '?':>10}")
        print(f"  {'total':<40} {human(total):>10}\n")

    if not os.environ.get("HF_TOKEN"):
        print("(HF_TOKEN not set - fine for public datasets)\n")

    ok = True
    for repo_id in selected:
        ok &= fetch(repo_id, datasets[repo_id], args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
