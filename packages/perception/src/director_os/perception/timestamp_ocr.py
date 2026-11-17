"""Bodycam overlay OCR (PLAN §4.6, M4) — the AXON timestamp reads itself.

Job 4 of the vision stack: read `2022-07-04 18:11:55 -0400 / AXON BODY 3` off
the corner of the footage so the Intro Location/Time Card populates itself with
zero manual typing. Two interchangeable backends:

- "gemini"  — one cheap single-frame call through the existing GeminiVisionClient
              (works the moment VISION_API_KEY exists)
- "modal"   — PaddleOCR on the serverless GPU fleet (`deploy/modal/redact_gpu.py`,
              function `ocr_frame`)

Both funnel into the same pure `parse_timestamp_text`, which is hermetic and
pinned by tests; the pipeline never branches on the provider.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class TimestampOcrError(RuntimeError):
    pass


@dataclass(frozen=True)
class BodycamStamp:
    """Parsed overlay: media-wallclock time + device badge."""

    raw_text: str
    iso: str | None = None  # e.g. 2022-07-04T18:11:55-04:00
    device: str | None = None  # e.g. AXON BODY 3
    tz: str | None = None  # e.g. -04:00 / EST
    confidence: float = 0.0


_DATE_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?"
)
_TZ_RE = re.compile(r"([+-])(\d{2}):?(\d{2})\b|\b([A-Z]{2,5})\b(?=\s*/|\s*$)")
_DEVICE_RE = re.compile(
    r"\b(AXON\s+BODY\s*\d|AXON\s+CAPTURE|AXON\s+FLEX|VW\s+ICON)\b", re.IGNORECASE
)


def parse_timestamp_text(text: str, confidence: float = 0.9) -> BodycamStamp | None:
    """Parse a raw OCR string; None when no date-like stamp is present."""
    if not text:
        return None
    m = _DATE_RE.search(text)
    if m is None:
        return None
    y, mo, d, hh, mm, ss = m.groups()
    ss = ss or "00"

    tz: str | None = None
    mtz = _TZ_RE.search(text, m.end())
    if mtz:
        if mtz.group(1):  # numeric offset
            tz = f"{mtz.group(1)}{mtz.group(2)}:{mtz.group(3)}"
        else:  # alphabetic zone (EST, PST, ...)
            tz = mtz.group(4)

    iso = f"{y}-{mo}-{d}T{hh}:{mm}:{ss}"
    if tz:
        iso += tz

    dev = _DEVICE_RE.search(text)
    device = re.sub(r"\s+", " ", dev.group(1)).upper() if dev else None
    return BodycamStamp(
        raw_text=text.strip()[:400],
        iso=iso,
        device=device,
        tz=tz,
        confidence=confidence,
    )


def extract_frame(video_path: str, t_s: float = 0.5, out_png: str | Path | None = None) -> Path:
    """Pull ONE full-resolution frame — the overlay is crisp at the top of a clip."""
    out = Path(out_png or "/tmp/dos_ocr_frame.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, t_s):.3f}", "-i", str(video_path),
        "-frames:v", "1", "-y", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise TimestampOcrError(f"frame extract failed: {r.stderr[-300:]}")
    return out


def read_timestamp_gemini(frame_png: str | Path, client, t_s: float = 0.5) -> BodycamStamp | None:
    """Backend 1: one VLM call — ask for a VERBATIM transcription of the overlay."""
    from director_os.perception.vision import FrameRef, VisionAdapter

    adapter = client if isinstance(client, VisionAdapter) else VisionAdapter(client)
    frame = FrameRef(timestamp_s=t_s, image_path=Path(frame_png))
    insights = adapter.understand(
        "Transcribe EVERY piece of text burned into this bodycam frame VERBATIM "
        "(the timestamp overlay, the device badge like 'AXON BODY 3', watermarks). "
        "Answer ONLY with a JSON array: "
        '[{"timestamp_s": float, "kind": "observation", "description": "<verbatim text>", '
        '"confidence": float 0..1}]',
        [frame],
    )
    best: BodycamStamp | None = None
    for ins in insights:
        stamp = parse_timestamp_text(ins.description, confidence=ins.confidence)
        if stamp is not None and (best is None or stamp.confidence > best.confidence):
            best = stamp
    return best


def read_timestamp_modal(frame_png: str | Path, ocr_fn) -> BodycamStamp | None:
    """Backend 2: the deployed Modal PaddleOCR function (returns {'texts': [...]})."""
    import base64

    payload = {
        "image_b64": base64.b64encode(Path(frame_png).read_bytes()).decode(),
        "mime": "image/png",
    }
    result = ocr_fn.remote(**payload)
    texts = " | ".join(t for t in result.get("texts", []) if t)
    return parse_timestamp_text(texts, confidence=0.95) if texts else None


def read_bodycam_timestamp(
    video_path: str,
    backend: str = "auto",
    client=None,
    ocr_fn=None,
    t_s: float = 0.5,
) -> BodycamStamp | None:
    """Front door: 'auto' prefers the deployed PaddleOCR fleet, falls back to Gemini.

    Raises TimestampOcrError only when NO backend is usable (no client, no fleet);
    a backend that simply finds no stamp returns None — that is a legitimate
    answer for footage without an overlay.
    """
    if backend == "modal" or (backend == "auto" and ocr_fn is not None):
        if ocr_fn is None:
            raise TimestampOcrError("backend 'modal' requested but no OCR function provided")
        frame = extract_frame(video_path, t_s=t_s)
        return read_timestamp_modal(frame, ocr_fn)
    if backend == "gemini" or (backend == "auto" and client is not None):
        if client is None:
            raise TimestampOcrError("no OCR backend available (client=None, ocr_fn=None)")
        frame = extract_frame(video_path, t_s=t_s)
        return read_timestamp_gemini(frame, client, t_s=t_s)
    if backend == "auto":
        raise TimestampOcrError("no OCR backend available (client=None, ocr_fn=None)")
    raise TimestampOcrError(f"unknown OCR backend: {backend}")
