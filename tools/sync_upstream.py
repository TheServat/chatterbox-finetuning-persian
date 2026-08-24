"""Keep the vendored Chatterbox in step with the latest stable upstream.

The problem this solves: `src/chatterbox_/` is a copy of Resemble AI's package
with a handful of local edits baked in. Hand-edited vendor trees rot - the
current copy is derived from an older release and is missing upstream fixes
(device-aware `map_location`, the sdpa->eager switch the alignment analyzer
needs) purely because nobody could tell the local edits apart from the original.

So the vendor tree is treated as disposable and kept byte-identical to upstream
apart from one unavoidable edit (`__init__.py` looks up its own version through
importlib.metadata, which fails for a copied-in package). Everything we need on
top lives outside it:

    src/compat.py           training hooks and dtype fixes, applied at import
    src/persian/engine.py   the Persian language layer

This is a code-layout decision, not a modelling one: the weights, tokenizer,
loss and training maths are identical either way. It only means an upgrade is a
directory swap plus a test run instead of a three-way merge.

Note the boundary. `src/chatterbox_/` is a replaceable mirror and is never
hand-edited. Every other file in this repo is ours and is edited freely.

Usage:
    python tools/sync_upstream.py --check     what is new upstream
    python tools/sync_upstream.py --update    vendor latest stable, update lock
    python tools/sync_upstream.py --update --version 0.1.8
    python tools/sync_upstream.py --verify    re-run the post-sync checks
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "versions.lock.json"
VENDOR_DIR = ROOT / "src" / "chatterbox_"

PYPI_JSON = "https://pypi.org/pypi/{package}/json"
HF_MODEL_API = "https://huggingface.co/api/models/{repo_id}"
HF_TREE_API = "https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=1"

# `__init__.py` resolves its own version through importlib.metadata, which only
# works for an installed distribution. This is the single edit we make.
VENDOR_INIT_REPLACEMENT = (
    "from .tts import ChatterboxTTS\n"
    "from .vc import ChatterboxVC\n"
    "from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES\n"
)

EXPECTED_MULTILINGUAL_VOCAB = 2454


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def save_lock(lock: dict) -> None:
    LOCK_PATH.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _is_stable(version: str) -> bool:
    """True for a plain release: no rc/a/b/dev/post suffix."""
    return all(part.isdigit() for part in version.split("."))


def latest_stable_version(package: str) -> str:
    data = _get_json(PYPI_JSON.format(package=package))
    stable = [v for v, files in data["releases"].items() if _is_stable(v) and files]
    if not stable:
        return data["info"]["version"]
    return max(stable, key=lambda v: tuple(int(p) for p in v.split(".")))


def _sdist_url(package: str, version: str) -> str:
    data = _get_json(PYPI_JSON.format(package=package))
    for entry in data["releases"][version]:
        if entry["packagetype"] == "sdist":
            return entry["url"]
    raise RuntimeError(f"{package} {version} publishes no source distribution")


def remote_weight_state(repo_id: str) -> tuple[str, dict]:
    """Current revision of the weights repo and its per-file size/sha256."""
    revision = _get_json(HF_MODEL_API.format(repo_id=repo_id))["sha"]
    tree = _get_json(HF_TREE_API.format(repo_id=repo_id, revision=revision))
    files = {}
    for entry in tree:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        files[entry["path"]] = {
            "size": lfs.get("size") or entry.get("size"),
            "sha256": lfs.get("oid"),
        }
    return revision, files


def newest_multilingual_t3(filenames) -> str | None:
    """Pick the highest-versioned multilingual T3 checkpoint present upstream.

    Resemble ships new bases as `t3_mtl23ls_v2`, `_v3`, ... without wiring them
    into the library, so the newest one has to be discovered by filename.
    """
    candidates = []
    for name in filenames:
        if name.startswith("t3_mtl23ls_v") and name.endswith(".safetensors"):
            tag = name[len("t3_mtl23ls_v"):].split(".")[0]
            if tag.isdigit():
                candidates.append((int(tag), name))
    return max(candidates)[1] if candidates else None


def cmd_check(lock: dict) -> int:
    package = lock["chatterbox"]["pypi_package"]
    pinned = lock["chatterbox"]["version"]
    latest = latest_stable_version(package)

    status = "[UPDATE AVAILABLE]" if latest != pinned else "[up to date]"
    print(f"chatterbox  pinned {pinned}   latest stable {latest}   {status}")

    repo_id = lock["weights"]["repo_id"]
    pinned_rev = lock["weights"]["revision"]
    revision, files = remote_weight_state(repo_id)
    status = "[UPDATE AVAILABLE]" if revision != pinned_rev else "[up to date]"
    print(f"weights     pinned {pinned_rev[:12]}  latest {revision[:12]}   {status}")

    newest_t3 = newest_multilingual_t3(files)
    in_use = lock["weights"]["profile"]["t3"]
    if newest_t3 and newest_t3 != in_use:
        print(f"\n  A newer multilingual base exists upstream: {newest_t3}")
        print(f"  (the finetune currently builds on {in_use})")

    # Only LFS entries are model weights; .gitattributes and README are noise.
    untracked = sorted(
        name
        for name, meta in files.items()
        if meta.get("sha256") and name not in lock["weights"]["files"]
    )
    if untracked:
        print("\n  Weights present upstream but absent from the lock file:")
        for name in untracked:
            print(f"    {name}")

    return 0


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a child Python that prints Persian without tripping over cp1252."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        **kwargs,
    )


VENDOR_PROBE = r"""
import sys, json
sys.path.insert(0, ROOT_PLACEHOLDER)
report = {}
try:
    from src.chatterbox_.models.t3.modules.t3_config import T3Config
    from src.chatterbox_.models.tokenizers.tokenizer import MTLTokenizer
    from src.chatterbox_.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    from src.chatterbox_.models.t3.t3 import T3
    report["import"] = "ok"
    report["mtl_vocab"] = T3Config.multilingual().text_tokens_dict_size
    report["languages"] = len(SUPPORTED_LANGUAGES)
    report["has_from_local"] = hasattr(ChatterboxMultilingualTTS, "from_local")
    report["has_encode"] = hasattr(MTLTokenizer, "encode")
    report["has_forward"] = hasattr(T3, "forward")
except Exception as exc:
    report["import"] = "FAILED: %s: %s" % (type(exc).__name__, exc)
print(json.dumps(report))
"""


def _verify_vendor_tree() -> list[str]:
    """Import the vendored package and confirm the pieces we rely on exist."""
    script = VENDOR_PROBE.replace("ROOT_PLACEHOLDER", repr(str(ROOT)))
    result = _run([sys.executable, "-c", script])
    if result.returncode != 0 or not result.stdout.strip():
        return [f"vendored package failed to import:\n{result.stderr.strip()}"]

    report = json.loads(result.stdout.strip().splitlines()[-1])
    if report.get("import") != "ok":
        return [report["import"]]

    problems: list[str] = []
    if report.get("mtl_vocab") != EXPECTED_MULTILINGUAL_VOCAB:
        problems.append(
            f"T3Config.multilingual() vocab is {report['mtl_vocab']}, expected "
            f"{EXPECTED_MULTILINGUAL_VOCAB} - tokenizer and checkpoint no longer "
            "line up, so the Persian embeddings would land on the wrong rows"
        )
    for attr, what in [
        ("has_from_local", "ChatterboxMultilingualTTS.from_local"),
        ("has_encode", "MTLTokenizer.encode"),
        ("has_forward", "T3.forward"),
    ]:
        if not report.get(attr):
            problems.append(f"{what} disappeared upstream - the overlay depends on it")
    return problems


def _verify_persian() -> list[str]:
    """Run the Persian coverage audit against the (possibly new) tokenizer."""
    test = ROOT / "tests" / "test_persian_coverage.py"
    if not test.exists():
        return ["tests/test_persian_coverage.py is missing"]
    result = _run([sys.executable, str(test)], cwd=str(ROOT))
    if result.returncode == 0:
        return []
    tail = (result.stdout + result.stderr).strip().splitlines()[-15:]
    return ["Persian coverage audit failed:\n    " + "\n    ".join(tail)]


def cmd_verify() -> int:
    print("Verifying the vendored tree...")
    problems = _verify_vendor_tree()
    for line in problems:
        print(f"  FAIL {line}")
    if not problems:
        print("  ok   imports, multilingual vocab 2454, public API intact")

    print("Verifying Persian coverage...")
    persian_problems = _verify_persian()
    for line in persian_problems:
        print(f"  FAIL {line}")
    if not persian_problems:
        print("  ok   every Persian character survives normalisation and tokenises")

    total = problems + persian_problems
    print("\nPASS" if not total else f"\nFAILED ({len(total)} problem(s))")
    return 0 if not total else 1


def cmd_update(lock: dict, version: str | None, keep_backup: bool) -> int:
    package = lock["chatterbox"]["pypi_package"]
    target = version or latest_stable_version(package)
    print(f"Vendoring {package} {target}")

    backup = ROOT / "src" / "_chatterbox_backup"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "sdist.tar.gz"
        url = _sdist_url(package, target)
        print(f"  downloading {url.rsplit('/', 1)[-1]}")
        urllib.request.urlretrieve(url, archive)

        with tarfile.open(archive) as tar:
            # filter="data" rejects absolute paths, symlinks and metadata games
            # in the archive. It becomes the default in 3.14; setting it here
            # silences the warning and makes the behaviour explicit.
            tar.extractall(tmp_path, filter="data")

        sources = list(tmp_path.glob("*/src/chatterbox"))
        if not sources:
            print("  ERROR: the sdist has no src/chatterbox directory", file=sys.stderr)
            return 1

        if backup.exists():
            shutil.rmtree(backup)
        if VENDOR_DIR.exists():
            shutil.move(str(VENDOR_DIR), str(backup))

        shutil.copytree(sources[0], VENDOR_DIR)
        (VENDOR_DIR / "__init__.py").write_text(
            VENDOR_INIT_REPLACEMENT, encoding="utf-8"
        )
        for cache in VENDOR_DIR.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        print(f"  vendored into {VENDOR_DIR.relative_to(ROOT)}")

    if cmd_verify() != 0:
        print("\nVerification failed - rolling the vendor tree back.")
        shutil.rmtree(VENDOR_DIR, ignore_errors=True)
        shutil.move(str(backup), str(VENDOR_DIR))
        return 1

    if keep_backup:
        print(f"  previous tree kept at {backup.relative_to(ROOT)}")
    else:
        shutil.rmtree(backup, ignore_errors=True)

    lock["chatterbox"]["version"] = target
    lock["chatterbox"]["synced_on"] = date.today().isoformat()
    lock["chatterbox"]["vendored_from"] = "sdist"

    repo_id = lock["weights"]["repo_id"]
    revision, files = remote_weight_state(repo_id)
    if revision != lock["weights"]["revision"]:
        print(f"  weights repo moved to {revision[:12]}, refreshing file hashes")
        roles = {n: e.get("role") for n, e in lock["weights"]["files"].items()}
        lock["weights"]["revision"] = revision
        lock["weights"]["files"] = {
            name: {**meta, "role": roles.get(name)} for name, meta in files.items()
        }

    newest_t3 = newest_multilingual_t3(files)
    if newest_t3 and newest_t3 != lock["weights"]["profile"]["t3"]:
        print(
            f"\n  NOTE: {newest_t3} is newer than the pinned "
            f"{lock['weights']['profile']['t3']}."
        )
        print("  Switch by editing weights.profile.t3 in versions.lock.json, then")
        print("  re-run `python tools/fetch_models.py`. Retrain after switching:")
        print("  a LoRA adapter is tied to the base it was trained on.")

    save_lock(lock)
    print(f"\nversions.lock.json updated -> chatterbox {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="report what is new")
    group.add_argument("--update", action="store_true", help="vendor latest stable")
    group.add_argument("--verify", action="store_true", help="run post-sync checks")
    parser.add_argument("--version", help="vendor this exact version instead")
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="leave the previous vendor tree in src/_chatterbox_backup",
    )
    args = parser.parse_args()

    if args.verify:
        return cmd_verify()

    lock = load_lock()
    if args.check:
        return cmd_check(lock)
    return cmd_update(lock, args.version, args.keep_backup)


if __name__ == "__main__":
    sys.exit(main())
