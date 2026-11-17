#!/usr/bin/env python3
"""Fetch yolov8n-license-plate.onnx into data/models/ (gitignored) with SHA-256 pin.

Second detector of the tracked-redaction family (issue #12: plates, not just
faces). 12.2 MB, single-class plate detector (ml-debi/yolov8-license-plate-detection,
MIT license, ultralytics export, imgsz 640 fixed) — the SAME output layout as the
face model, so `YoloOnnxPlateDetector` is a rename, not a new decode path. Hugging
Face download with a checksum pin so the file can never silently change.

Like the face model, absence degrades gracefully: the worker skips the plate sweep
(face redaction stays mandatory), `capabilities()["plates"]` flips False.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/ml-debi/yolov8-license-plate-detection/resolve/main/best.onnx"
SHA256 = "85d236280a1301ad98907947d284951dd2b20c23a6786ff50f7e6a8ec515bd50"
DEST = Path(__file__).resolve().parents[1] / "data" / "models" / "yolov8n-license-plate.onnx"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"already present: {DEST}")
        return 0
    print(f"downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "DirectorOS/0.1 (model fetch)"})
    with urllib.request.urlopen(req, timeout=300) as r, open(DEST, "wb") as fh:
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
