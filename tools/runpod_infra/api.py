"""A small RunPod client covering exactly what this project needs.

RunPod has two APIs and neither covers everything:

  * the REST API (`rest.runpod.io/v1`) creates and manages pods and network
    volumes, and
  * the GraphQL API (`api.runpod.io/graphql`) is the only one that reports GPU
    prices and stock.

So both are wrapped here. The alternative - the official `runpod` SDK - is a
much larger dependency for a handful of calls, and it hides the request bodies
at exactly the moment cost depends on getting them right.

The API key is read from `RUNPOD_API_KEY` and never written to disk. Anything
that persists a pod spec redacts it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

REST_BASE = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"

# RunPod exposes a pod's HTTP port at this address, which is how this project
# talks to a running pod: SSH is on port 22 and that is blocked on some
# networks, while the proxy is ordinary HTTPS.
PROXY_TEMPLATE = "https://{pod_id}-{port}.proxy.runpod.net"


# Cloudflare sits in front of the GraphQL endpoint and rejects requests with
# urllib's default User-Agent (error 1010).
USER_AGENT = "chatterbox-finetuning/1.0 (+https://github.com/TheServat)"


class RunPodError(RuntimeError):
    pass


def redact(text: str) -> str:
    """Strip the API key out of anything that might be printed or logged."""
    key = os.environ.get("RUNPOD_API_KEY", "")
    if key and key in text:
        text = text.replace(key, f"{key[:7]}...REDACTED")
    return text


def api_key() -> str:
    # Credentials live in a git-ignored .env; anything already exported wins.
    from tools.runpod_infra import secrets

    secrets.load()
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise RunPodError(
            "RUNPOD_API_KEY is not set.\n"
            "  PowerShell:  $env:RUNPOD_API_KEY = 'rpa_...'\n"
            "  bash:        export RUNPOD_API_KEY=rpa_...\n"
            "Create one at https://console.runpod.io/user/settings"
        )
    return key


def _request(
    method: str,
    url: str,
    *,
    body: dict | None = None,
    timeout: float = 60.0,
    headers: dict | None = None,
) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {api_key()}")
    request.add_header("Content-Type", "application/json")
    request.add_header("User-Agent", USER_AGENT)
    for name, value in (headers or {}).items():
        request.add_header(name, value)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:600]
        raise RunPodError(
            redact(f"{method} {url} -> HTTP {error.code}: {detail}")
        ) from None
    except urllib.error.URLError as error:
        raise RunPodError(redact(f"{method} {url} unreachable: {error.reason}")) from None


def graphql(query: str, variables: dict | None = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    # The key goes in the query string here: the GraphQL endpoint predates the
    # bearer-token scheme and does not accept the header reliably.
    result = _request("POST", f"{GRAPHQL_URL}?api_key={api_key()}", body=payload)
    if errors := (result or {}).get("errors"):
        raise RunPodError(f"GraphQL: {errors[0].get('message')}")
    return (result or {}).get("data", {})


# --------------------------------------------------------------------------
# GPU types and pricing
# --------------------------------------------------------------------------

GPU_QUERY = """
query {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    securePrice
    communityPrice
    maxGpuCount
    lowestPrice(input: {gpuCount: 1}) {
      uninterruptablePrice
      minimumBidPrice
      stockStatus
    }
  }
}
"""


def gpu_types() -> list[dict]:
    """Every GPU RunPod offers, with its current lowest price and stock."""
    types = graphql(GPU_QUERY).get("gpuTypes", [])
    result = []
    for entry in types:
        price = entry.get("lowestPrice") or {}
        result.append({
            "id": entry["id"],
            "name": entry["displayName"],
            "vram_gb": entry["memoryInGb"],
            "on_demand": price.get("uninterruptablePrice"),
            # Distinct from the secure/community booleans below: a duplicate key
            # let the boolean win, and max(True, 0.34) quietly became $1.00/h.
            "secure_price": entry.get("securePrice"),
            "community_price": entry.get("communityPrice"),
            "spot": price.get("minimumBidPrice"),
            "stock": price.get("stockStatus"),
            "secure": entry.get("secureCloud", False),
            "community": entry.get("communityCloud", False),
        })
    return result


DATACENTER_QUERY = """
query {
  dataCenters { id name location storageSupport listed }
}
"""


_POD_DATACENTERS: list[str] | None = None


def pod_datacenters() -> list[str]:
    """The datacentre ids `create_pod` will actually accept.

    GraphQL lists more of them than the REST pod endpoint takes, so probing
    straight from that list spends most of its attempts on ids the schema
    rejects - which reads as a fault rather than as the wrong question. The
    accepted set is in the OpenAPI enum, and the quickest way to it is to send
    one deliberately invalid id and read the enum out of the 400. That keeps
    working when RunPod adds a region, which a hardcoded list would not.
    """
    global _POD_DATACENTERS
    if _POD_DATACENTERS is not None:
        return _POD_DATACENTERS

    import re

    try:
        create_pod({
            "name": "enum-probe",
            "imageName": "runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204",
            "cloudType": "SECURE", "computeType": "GPU", "gpuCount": 1,
            "gpuTypeIds": ["NVIDIA RTX A5000"], "containerDiskInGb": 10,
            "dataCenterIds": ["__ASK__"],
            "dockerStartCmd": ["bash", "-lc", "true"],
        })
    except Exception as error:
        found = re.search(r"dataCenterIds/items/enum[^']*((?:'[A-Z0-9-]+',?\s*)+)", str(error))
        if found:
            _POD_DATACENTERS = re.findall(r"'([A-Z0-9-]+)'", found.group(1))
            return _POD_DATACENTERS

    _POD_DATACENTERS = []          # unknown: caller should not filter
    return _POD_DATACENTERS


def datacenters(storage_only: bool = False) -> list[dict]:
    entries = graphql(DATACENTER_QUERY).get("dataCenters", []) or []
    if storage_only:
        entries = [d for d in entries if d.get("storageSupport")]
    return entries


# --------------------------------------------------------------------------
# Pods
# --------------------------------------------------------------------------

def create_pod(spec: dict) -> dict:
    return _request("POST", f"{REST_BASE}/pods", body=spec, timeout=120)


def get_pod(pod_id: str) -> dict:
    return _request("GET", f"{REST_BASE}/pods/{pod_id}")


def list_pods() -> list[dict]:
    return _request("GET", f"{REST_BASE}/pods") or []


def stop_pod(pod_id: str) -> Any:
    """Stop the pod but keep it (and its billing for storage) around."""
    return _request("POST", f"{REST_BASE}/pods/{pod_id}/stop")


def terminate_pod(pod_id: str) -> Any:
    """Delete the pod. This is what actually stops GPU billing."""
    return _request("DELETE", f"{REST_BASE}/pods/{pod_id}")


def proxy_url(pod_id: str, port: int) -> str:
    return PROXY_TEMPLATE.format(pod_id=pod_id, port=port)


# --------------------------------------------------------------------------
# Network volumes
# --------------------------------------------------------------------------

def create_volume(name: str, size_gb: int, datacenter_id: str) -> dict:
    return _request(
        "POST",
        f"{REST_BASE}/networkvolumes",
        body={"name": name, "size": size_gb, "dataCenterId": datacenter_id},
    )


def list_volumes() -> list[dict]:
    return _request("GET", f"{REST_BASE}/networkvolumes") or []


def delete_volume(volume_id: str) -> Any:
    return _request("DELETE", f"{REST_BASE}/networkvolumes/{volume_id}")


def find_volume(name: str) -> dict | None:
    for volume in list_volumes():
        if volume.get("name") == name:
            return volume
    return None
