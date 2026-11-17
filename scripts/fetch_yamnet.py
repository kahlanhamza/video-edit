#!/usr/bin/env python3
"""Fetch yamnet.onnx (+ class map) into data/models/ (gitignored) with SHA-256 pin.

`perception.detect_audio_events` (issue #12): sirens, gunshots, shouting, glass,
alarms on CPU. Google's YAMNet (AudioSet, 521 classes) converted 1:1 to ONNX by
audiomagic/yamnet-onnx (Apache-2.0) — the log-mel frontend is BAKED INTO THE GRAPH,
so the runtime consumes a raw 16 kHz mono waveform (float32) and emits per-0.96-s
frame scores; there is no feature extraction to reimplement (or get subtly wrong).

16 MB — too big to ship inside the package (unlike the 232 KB YuNet), so this is a
fetch-once data file. Absence degrades gracefully: audio event detection is an
optional enrichment, never a pipeline blocker.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/audiomagic/yamnet-onnx/resolve/main"
FILES: list[tuple[str, str, str]] = [
    # (remote name, sha256, dest name)
    (
        "yamnet.onnx",
        "d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e",
        "yamnet.onnx",
    ),
    (
        "yamnet_class_map.csv",
        "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2",
        "yamnet_class_map.csv",
    ),
]
DEST_DIR = Path(__file__).resolve().parents[1] / "data" / "models"


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for remote, sha, local in FILES:
        dest = DEST_DIR / local
        if dest.exists():
            print(f"already present: {dest}")
            continue
        url = f"{BASE}/{remote}"
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "DirectorOS/0.1 (model fetch)"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
        except Exception as e:  # noqa: BLE001 — fetch script reports and exits nonzero
            print(f"download failed: {e}", file=sys.stderr)
            failures += 1
            continue
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != sha:
            dest.unlink(missing_ok=True)
            print(
                f"SHA-256 MISMATCH for {local}: got {digest}, expected {sha} — deleted",
                file=sys.stderr,
            )
            failures += 1
        else:
            print(f"verified + saved: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
