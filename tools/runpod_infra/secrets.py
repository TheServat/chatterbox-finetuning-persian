"""Load API credentials from a git-ignored `.env`, so they are typed once.

Both the RunPod key and the HuggingFace token are needed on every launch, and
exporting them by hand each session is the kind of friction that ends with a key
pasted into a tracked file and pushed to a public repository.

So they live in `.env` at the project root, which `.gitignore` excludes. Values
already in the environment win, which keeps CI and one-off overrides working
without touching the file.

    python -m tools.runpod_infra.secrets --check     # what is configured
    python -m tools.runpod_infra.secrets --template  # write a starter .env
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"

KNOWN = {
    "RUNPOD_API_KEY": "RunPod API key, from https://console.runpod.io/user/settings",
    "HF_TOKEN": "HuggingFace token, from https://huggingface.co/settings/tokens",
}

TEMPLATE = """# Credentials for this project. Git-ignored - never commit this file.
#
# RunPod:       https://console.runpod.io/user/settings
# HuggingFace:  https://huggingface.co/settings/tokens
#               read access is enough to download weights and datasets

RUNPOD_API_KEY=
HF_TOKEN=
"""


def load(path: Path = ENV_PATH, *, override: bool = False) -> dict[str, str]:
    """Read `path` into the environment. Returns the names that were set."""
    if not path.exists():
        return {}

    loaded = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if not name or not value:
            continue
        if override or not os.environ.get(name):
            os.environ[name] = value
            loaded[name] = value
    return loaded


def require(name: str) -> str:
    load()
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"{name} is not set.\n"
            f"  {KNOWN.get(name, '')}\n"
            f"Add it to {ENV_PATH} (git-ignored), or export it for this session."
        )
    return value


def mask(value: str) -> str:
    if len(value) <= 12:
        return "set"
    return f"{value[:7]}...{value[-4:]}"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--template", action="store_true", help="write a starter .env")
    parser.add_argument("--check", action="store_true", help="report what is configured")
    args = parser.parse_args()

    if args.template:
        if ENV_PATH.exists():
            print(f"{ENV_PATH} already exists; leaving it alone.")
            return 1
        ENV_PATH.write_text(TEMPLATE, encoding="utf-8")
        print(f"wrote {ENV_PATH} - fill it in. It is git-ignored.")
        return 0

    load()
    print(f"{ENV_PATH}  {'found' if ENV_PATH.exists() else 'MISSING'}")
    for name, description in KNOWN.items():
        value = os.environ.get(name, "")
        print(f"  {name:<16} {mask(value) if value else 'not set':<24} {description}")
    return 0 if all(os.environ.get(n) for n in KNOWN) else 1


if __name__ == "__main__":
    sys.exit(main())
