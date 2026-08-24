"""A resumable HTTP downloader for the large HuggingFace artifacts.

`huggingface_hub` retries failed requests but not a *stalled* one: when the
connection goes quiet without closing, its read blocks forever. On this link
that happened reliably around the 1 GB mark, leaving a 2 GB checkpoint stuck
at 1,024,000,000 bytes with no error and no progress.

So large files are fetched here instead. Two things make it survive the link:

  * a socket timeout, so a quiet connection raises instead of hanging, and
  * `Range: bytes=N-` on retry, so each attempt resumes from what is already on
    disk rather than starting the 2 GB over.

`.part` holds the partial file and is only renamed once the expected size is
reached, so an interrupted run can never leave something that looks complete.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1 << 20          # 1 MiB
READ_TIMEOUT = 30.0      # a quiet socket for this long counts as stalled
DEFAULT_ATTEMPTS = 40


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def _open(url: str, offset: int, token: str | None):
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "chatterbox-finetuning/1.0")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(request, timeout=READ_TIMEOUT)


def download(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    token: str | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    progress: bool = True,
) -> Path:
    """Fetch `url` to `destination`, resuming across stalls and disconnects."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    if destination.exists() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        return destination

    started = time.monotonic()
    last_report = 0.0

    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0

        if expected_size and offset >= expected_size:
            break

        try:
            with _open(url, offset, token) as response:
                # A server that ignores Range restarts the file; honour that
                # rather than appending a second copy onto the first.
                if offset and response.status != 206:
                    offset = 0
                    partial.unlink(missing_ok=True)

                total = expected_size
                if total is None:
                    length = response.headers.get("Content-Length")
                    if length:
                        total = int(length) + offset

                with partial.open("ab" if offset else "wb") as handle:
                    while chunk := response.read(CHUNK):
                        handle.write(chunk)
                        offset += len(chunk)

                        now = time.monotonic()
                        # A carriage-return progress line is unreadable once it
                        # lands in a log file, so report far less often there.
                        interval = 2.0 if sys.stdout.isatty() else 30.0
                        if progress and now - last_report >= interval:
                            last_report = now
                            rate = offset / max(now - started, 1e-6)
                            share = f"/{human(total)}" if total else ""
                            eta = (
                                f"  eta {int((total - offset) / max(rate, 1)) // 60} min"
                                if total and rate > 0
                                else ""
                            )
                            print(
                                f"\r    {human(offset)}{share}  "
                                f"{human(rate)}/s{eta}   ",
                                # A bare carriage return redraws one line in a
                                # terminal but concatenates into an unreadable
                                # single line in a log file.
                                end="" if sys.stdout.isatty() else "\n",
                                flush=True,
                            )

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            grown = (partial.stat().st_size if partial.exists() else 0) - offset
            if progress:
                print(
                    f"\n    attempt {attempt}/{attempts} interrupted "
                    f"({type(exc).__name__}); resuming from "
                    f"{human(partial.stat().st_size if partial.exists() else 0)}"
                )
            # Back off a little, but never long enough to stall the whole run.
            time.sleep(min(2 * attempt, 15))
            continue

        size = partial.stat().st_size
        if expected_size is None or size == expected_size:
            break
        if progress:
            print(
                f"\n    short read ({human(size)} of {human(expected_size)}); "
                f"resuming"
            )
    else:
        raise RuntimeError(f"gave up on {url} after {attempts} attempts")

    size = partial.stat().st_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(
            f"{destination.name}: got {size} bytes, expected {expected_size}"
        )

    if progress:
        elapsed = time.monotonic() - started
        print(f"\r    {human(size)} in {elapsed / 60:.1f} min" + " " * 30)

    partial.replace(destination)
    return destination


def hf_url(repo_id: str, filename: str, revision: str = "main", repo_type: str = "model") -> str:
    prefix = "" if repo_type == "model" else f"{repo_type}s/"
    return f"https://huggingface.co/{prefix}{repo_id}/resolve/{revision}/{filename}"


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("url")
    parser.add_argument("destination")
    parser.add_argument("--size", type=int, help="expected size in bytes")
    args = parser.parse_args()

    try:
        download(
            args.url,
            Path(args.destination),
            expected_size=args.size,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        sys.exit(1)
