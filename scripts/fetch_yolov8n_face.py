#!/usr/bin/env python3
"""Fetch yolov8n-face-lindevs.onnx into data/models/ (gitignored) with SHA-256 pin.

The LocalOnnxCpuBackend's primary detector. 12.2 MB, single-class face model,
$0 — no account, no card, plain GitHub release download with a checksum pin
so the file can never silently change. When the file is absent the compute
backend degrades to packaged YuNet (232 KB, ships with perception).
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx"
SHA256 = "8d0bfb0c3383c5bd7a78dd24ef79a21e2aa456619b6ab5e53867092d1c7dc414"
DEST = Path(__file__).resolve().parents[1] / "data" / "models" / "yolov8n-face-lindevs.onnx"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"already present: {DEST}")
        return 0
    print(f"downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "DirectorOS/0.1 (model fetch)"})
    with urllib.request.urlopen(req, timeout=180) as r, open(DEST, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    if digest != SHA256:
        DEST.unlink(missing_ok=True)
        print(f"SHA-256 MISMATCH: got {digest}, expected {SHA256} — file deleted", file=sys.stderr)
        return 1
    print(f"verified + saved: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
