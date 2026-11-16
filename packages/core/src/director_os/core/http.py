"""Cloud ingest HTTP layer — PLAN guardrail 3.

Wikimedia, FOIA portals and CDNs block default client User-Agents; every request
DirectorOS makes therefore carries a descriptive identity. This module is the only
place allowed to build outbound headers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

UA_PRODUCT = "DirectorOS/0.1"
UA_PURPOSE = "(autonomous documentary ingest; respects robots.txt)"

DEFAULT_UA = f"{UA_PRODUCT} {UA_PURPOSE}"


def build_headers(
    contact_email: str | None = None, extra: dict[str, str] | None = None
) -> dict[str, str]:
    """Every outbound request gets the descriptive UA — no exceptions, no override."""
    email = contact_email or os.getenv("CONTACT_EMAIL", "ops@directoros.example")
    headers = {
        "User-Agent": f"{DEFAULT_UA} contact: {email}",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*",
    }
    # guardrail: callers may add cookies/auth headers but may NOT replace the UA
    for k, v in (extra or {}).items():
        if k.lower() == "user-agent":
            continue
        headers[k] = v
    return headers


def download(
    url: str,
    dest: str | Path,
    contact_email: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 120.0,
    chunk_size: int = 1 << 20,
) -> Path:
    """Stream `url` to `dest` with the mandatory DirectorOS identity headers."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = build_headers(contact_email, extra_headers)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, headers=headers, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size):
                f.write(chunk)
    tmp.replace(dest)
    return dest


def fetch_json(url: str, contact_email: str | None = None) -> dict[str, Any]:
    resp = httpx.get(url, headers=build_headers(contact_email), timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()
