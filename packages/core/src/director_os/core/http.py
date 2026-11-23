"""Cloud ingest HTTP layer — PLAN guardrail 3.

Wikimedia, FOIA portals and CDNs block default client User-Agents; every request
DirectorOS makes therefore carries a descriptive identity. This module is the only
place allowed to build outbound headers.
"""

from __future__ import annotations

import os
from collections.abc import Callable
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
    progress_cb: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Stream `url` to `dest` with the mandatory DirectorOS identity headers.

    Kill-safe by construction: bytes land in a pid-scoped ``.part`` file and
    are atomically renamed only after the response completes — a killed
    process (or two racing downloads of the same URL) never corrupts the
    final file. ``progress_cb(received_bytes, total_bytes_or_None)`` fires
    per chunk so the worker can stream download progress into the Studio
    console instead of going silent for the length of a 469 MB bodycam.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = build_headers(contact_email, extra_headers)
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.part")
    received = 0
    with httpx.stream("GET", url, headers=headers, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        length = resp.headers.get("content-length")
        total: int | None = int(length) if length and length.isdigit() else None
        with open(tmp, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size):
                f.write(chunk)
                received += len(chunk)
                if progress_cb is not None:
                    progress_cb(received, total)
    tmp.replace(dest)
    return dest


def fetch_json(url: str, contact_email: str | None = None) -> dict[str, Any]:
    resp = httpx.get(url, headers=build_headers(contact_email), timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()
