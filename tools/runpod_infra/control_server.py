"""The pod's own control surface, reachable over RunPod's HTTPS proxy.

SSH is the usual way to watch a pod, and it is unavailable here: RunPod's SSH
runs on port 22, which is blocked on the network this project is driven from.
RunPod also proxies any exposed HTTP port at
`https://<pod-id>-<port>.proxy.runpod.net`, and that is ordinary HTTPS, so this
serves the same needs over it - progress, file transfer in both directions, and
a way to stop cleanly.

Deliberately standard library only. It has to come up before `pip install` runs,
because the setup phase is exactly when a run is most likely to fail and most
useful to watch.

Every route requires `Authorization: Bearer $CONTROL_TOKEN`. The proxy hostname
is derived from the pod id, which is not secret, so without this the endpoint
would be open to anyone who guessed it.

    python control_server.py --port 8080 --root /workspace
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(os.environ.get("POD_ROOT", "/workspace"))
TOKEN = os.environ.get("CONTROL_TOKEN", "")
STARTED = time.time()

LOG_TAIL_BYTES = 8192
MAX_UPLOAD_BYTES = 8 << 30  # 8 GiB, far above anything this project transfers


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def tail(path: Path, limit: int = LOG_TAIL_BYTES) -> list[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - limit))
            text = handle.read().decode("utf-8", "replace")
        # tqdm redraws with carriage returns; split on those too or the tail is
        # one enormous unreadable line.
        lines = text.replace("\r", "\n").splitlines()
        return [line for line in lines if line.strip()][-40:]
    except Exception:
        return []


def gpu_stats() -> dict | None:
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        util, used, total, temp = (p.strip() for p in result.stdout.strip().split(","))
        return {
            "util_percent": int(util),
            "mem_used_gb": round(int(used) / 1024, 2),
            "mem_total_gb": round(int(total) / 1024, 2),
            "temperature_c": int(temp),
        }
    except Exception:
        return None


def disk_stats(path: Path) -> dict | None:
    try:
        usage = shutil.disk_usage(path)
        return {
            "free_gb": round(usage.free / 2**30, 1),
            "total_gb": round(usage.total / 2**30, 1),
        }
    except Exception:
        return None


def build_status() -> dict:
    phase_file = ROOT / "phase"
    training = read_json(ROOT / "status.json") or {}

    phase = training.get("phase")
    if not phase or phase in ("starting",):
        # Before training starts, bootstrap.sh is the only thing that knows
        # where the run is.
        phase = (phase_file.read_text(encoding="utf-8").strip()
                 if phase_file.exists() else "booting")

    status = {
        "phase": phase,
        "ok": training.get("ok", True),
        "error": training.get("error"),
        "uptime_seconds": round(time.time() - STARTED, 1),
        "training": training or None,
        "gpu": gpu_stats(),
        "disk": disk_stats(ROOT),
        "setup_log": tail(ROOT / "bootstrap.log", 4096),
        "train_log": tail(ROOT / "train.log"),
    }

    failed = ROOT / "FAILED"
    if failed.exists():
        status["ok"] = False
        status["phase"] = "failed"
        status["error"] = status["error"] or failed.read_text(encoding="utf-8")[:800]
    if (ROOT / "DONE").exists():
        status["phase"] = "done"

    return status


class Handler(BaseHTTPRequestHandler):
    server_version = "chatterbox-pod/1.0"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args):  # quieter than the default access log
        pass

    def _authorised(self) -> bool:
        if not TOKEN:
            return True  # no token configured: refuse to pretend there is one
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return False
        return hmac.compare_digest(header[len(prefix):], TOKEN)

    def _send(self, code: int, payload: dict | bytes, content_type="application/json"):
        body = (json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if isinstance(payload, dict) else payload)
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve(self, raw: str) -> Path | None:
        """Resolve a request path inside ROOT, refusing anything that escapes."""
        candidate = (ROOT / raw.lstrip("/")).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError:
            return None
        return candidate

    # -- routes ------------------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/health", "/"):
            # Unauthenticated on purpose: the launcher uses it to tell "the pod
            # is up" from "the pod is up but my token is wrong".
            return self._send(200, {"alive": True, "uptime": round(time.time() - STARTED, 1)})

        if not self._authorised():
            return self._send(401, {"error": "bad or missing bearer token"})

        if parsed.path == "/status":
            return self._send(200, build_status())

        if parsed.path == "/download":
            wanted = parse_qs(parsed.query).get("path", [""])[0]
            target = self._resolve(wanted)
            if target is None:
                return self._send(400, {"error": "path escapes the pod root"})
            if not target.is_file():
                return self._send(404, {"error": f"no such file: {wanted}"})
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(target.stat().st_size))
            self.end_headers()
            with target.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)
            return None

        if parsed.path == "/ls":
            wanted = parse_qs(parsed.query).get("path", ["."])[0]
            target = self._resolve(wanted)
            if target is None or not target.exists():
                return self._send(404, {"error": f"no such path: {wanted}"})
            if target.is_file():
                return self._send(200, {"files": [
                    {"name": target.name, "size": target.stat().st_size}
                ]})
            return self._send(200, {"files": sorted(
                ({"name": p.name, "size": p.stat().st_size if p.is_file() else None,
                  "dir": p.is_dir()} for p in target.iterdir()),
                key=lambda entry: entry["name"],
            )})

        return self._send(404, {"error": "unknown route"})

    def do_PUT(self):
        if not self._authorised():
            return self._send(401, {"error": "bad or missing bearer token"})

        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            return self._send(404, {"error": "unknown route"})

        wanted = parse_qs(parsed.query).get("path", [""])[0]
        target = self._resolve(wanted)
        if target is None:
            return self._send(400, {"error": "path escapes the pod root"})

        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            return self._send(400, {"error": f"bad Content-Length: {length}"})

        target.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the destination and renamed, so a dropped connection
        # cannot leave a truncated file that looks complete.
        partial = target.with_suffix(target.suffix + ".part")
        written = 0
        with partial.open("wb") as handle:
            while written < length:
                chunk = self.rfile.read(min(1 << 20, length - written))
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)

        if written != length:
            partial.unlink(missing_ok=True)
            return self._send(400, {"error": f"got {written} of {length} bytes"})

        partial.replace(target)
        return self._send(200, {"path": str(target), "bytes": written})

    def do_POST(self):
        if not self._authorised():
            return self._send(401, {"error": "bad or missing bearer token"})

        parsed = urlparse(self.path)
        if parsed.path == "/shutdown":
            (ROOT / "STOP").write_text("requested", encoding="utf-8")
            return self._send(200, {"stopping": True})

        return self._send(404, {"error": "unknown route"})


def main() -> int:
    global ROOT

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()

    ROOT = Path(args.root)
    ROOT.mkdir(parents=True, exist_ok=True)

    if not TOKEN:
        print("WARNING: CONTROL_TOKEN is unset - every route is unauthenticated.")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"control server on :{args.port}, root {ROOT}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
