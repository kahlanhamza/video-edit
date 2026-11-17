"""Bodycam overlay OCR (PLAN §4.6, M4) — the AXON timestamp reads itself.

Job 4 of the vision stack: read `2022-07-04 18:11:55 -0400 / AXON BODY 3` off
the corner of the footage so the Intro Location/Time Card populates itself with
zero manual typing — plus street signs / route shields for the location card.
Three interchangeable backends (cost law: local-free first):

- "paddle"  — PaddleOCR on CPU, LOCAL and $0 (lazy import; used when the
              `paddleocr` package is installed — never a default-path dep)
- "gemini"  — one cheap single-frame call through the existing GeminiVisionClient
              (works the moment VISION_API_KEY exists)
- "modal"   — PaddleOCR on the serverless GPU fleet (`deploy/modal/redact_gpu.py`,
              function `ocr_frame`) — dormant until Modal is activated

All backends funnel into the same pure `parse_timestamp_text` /
`parse_overlay_lines`, which are hermetic and pinned by tests; the pipeline
never branches on the provider beyond the availability probe.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TimestampOcrError(RuntimeError):
    pass


@dataclass(frozen=True)
class BodycamStamp:
    """Parsed overlay: media-wallclock time + device badge + location hint."""

    raw_text: str
    iso: str | None = None  # e.g. 2022-07-04T18:11:55-04:00
    device: str | None = None  # e.g. AXON BODY 3
    tz: str | None = None  # e.g. -04:00 / EST
    location_text: str | None = None  # e.g. "N OAK ST" / "US-192 W" / "MILE MARKER 205"
    confidence: float = 0.0


_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?")
_TZ_RE = re.compile(r"([+-])(\d{2}):?(\d{2})\b|\b([A-Z]{2,5})\b(?=\s*/|\s*$)")
_DEVICE_RE = re.compile(
    r"\b(AXON\s+BODY\s*\d|AXON\s+CAPTURE|AXON\s+FLEX|VW\s+ICON)\b", re.IGNORECASE
)

# Street-sign / route / mile-marker grammar (conservative: only lines that are
# unmistakably location-ish become card facts — OCR noise must not ship).
_STREET_RE = re.compile(
    r"\b\d{1,5}\s+(?:[NSEW]\s+)?(?:[A-Z][A-Z'.]+\s+){0,3}"
    r"(?:ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|RD|ROAD|DR|DRIVE|HWY|HIGHWAY|"
    r"LN|LANE|CT|COURT|WAY|PKWY|PARKWAY|PL|PLACE|TER|TERRACE|CIR|CIRCLE|TRL|TRAIL)\b\.?",
)
_ROUTE_RE = re.compile(
    r"\b(?:US|SR|CR|FM|I|IH|RT|STATE\s+ROUTE)[- ]?\d{1,3}\s*[NSEW]?\b", re.IGNORECASE
)
_MARKER_RE = re.compile(r"\bMILE\s*(?:POST|MARKER)?\s*\d{1,4}\b", re.IGNORECASE)


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
        location_text=_location_from_lines([text]),
        confidence=confidence,
    )


def _location_from_lines(lines: list[str]) -> str | None:
    """First unmistakable location line: street address > route shield > marker."""
    for line in lines:
        m = _STREET_RE.search(line)
        if m:
            return re.sub(r"\s+", " ", m.group(0)).strip(". ").upper()
    for line in lines:
        m = _ROUTE_RE.search(line)
        if m:
            return re.sub(r"\s+", " ", m.group(0)).strip(". ").upper()
    for line in lines:
        m = _MARKER_RE.search(line)
        if m:
            return re.sub(r"\s+", " ", m.group(0)).strip(". ").upper()
    return None


def parse_overlay_lines(
    lines: list[str], confidences: list[float] | None = None
) -> BodycamStamp | None:
    """Parse a set of OCR text lines (one overlay frame) into a stamp.

    The timestamp may live on one line and the device badge / street sign on
    others (PaddleOCR returns reading-order blocks). The stamp carries the
    highest-confidence date-bearing line; location rides from ALL lines.
    """
    if not lines:
        return None
    best: BodycamStamp | None = None
    for i, line in enumerate(lines):
        conf = 0.9
        if confidences and i < len(confidences):
            conf = float(confidences[i])
        stamp = parse_timestamp_text(line, confidence=conf)
        if stamp is not None and (best is None or stamp.confidence > best.confidence):
            best = stamp
    if best is None:
        return None
    patched: dict = {}
    if best.device is None:
        for line in lines:  # the badge often sits on its own line
            dev = _DEVICE_RE.search(line)
            if dev:
                patched["device"] = re.sub(r"\s+", " ", dev.group(1)).upper()
                break
    location = _location_from_lines(lines)
    if location and not best.location_text:
        patched["location_text"] = location
    if patched:
        best = BodycamStamp(
            raw_text=best.raw_text,
            iso=best.iso,
            device=patched.get("device", best.device),
            tz=best.tz,
            location_text=patched.get("location_text", best.location_text),
            confidence=best.confidence,
        )
    return best


def extract_frame(video_path: str, t_s: float = 0.5, out_png: str | Path | None = None) -> Path:
    """Pull ONE full-resolution frame — the overlay is crisp at the top of a clip."""
    out = Path(out_png or "/tmp/dos_ocr_frame.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, t_s):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-y",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise TimestampOcrError(f"frame extract failed: {r.stderr[-300:]}")
    return out


# --------------------------------------------------------------- paddle (local)
_PADDLE_ENGINE: Any = None  # one CPU engine per process (init is heavy)


def paddleocr_available() -> bool:
    """True when the `paddleocr` package is importable (lazy — never a default dep)."""
    try:
        import paddleocr  # noqa: F401

        return True
    except ImportError:
        return False


def _paddle_lines_fn(frame_png: str | Path) -> list[str]:
    """Run the LOCAL PaddleOCR engine on CPU; returns reading-order text lines.

    The engine is cached per process (model init costs seconds). Raises
    TimestampOcrError when paddleocr is absent so callers fall to the next
    backend loudly, never silently.
    """
    global _PADDLE_ENGINE
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise TimestampOcrError("paddleocr not installed (pip install paddleocr)") from e
    if _PADDLE_ENGINE is None:
        _PADDLE_ENGINE = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
    result = _PADDLE_ENGINE.ocr(str(frame_png), cls=True)  # type: ignore[union-attr]
    lines: list[str] = []
    # paddleocr returns [[box, (text, conf)], ...] per image (list-of-lists
    # changed across versions — handle both the single-image and wrapped form)
    page = result[0] if result and isinstance(result[0], list) else result
    for item in page or []:
        try:
            text, _conf = item[1]
        except (IndexError, TypeError, ValueError):
            continue
        if text:
            lines.append(str(text))
    return lines


def read_timestamp_paddle(
    frame_png: str | Path,
    lines_fn=None,
    t_s: float = 0.5,
) -> BodycamStamp | None:
    """Backend 0 (LOCAL, $0): PaddleOCR on CPU. `lines_fn` is the port seam —
    tests inject a fake line provider; production uses `_paddle_lines_fn`
    (late-bound so monkeypatching the module attr works)."""
    lines = (lines_fn or _paddle_lines_fn)(frame_png)
    return parse_overlay_lines(lines)


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
    """Front door: 'auto' prefers the LOCAL free PaddleOCR lane, then the
    deployed Modal fleet, then Gemini.

    Raises TimestampOcrError only when NO backend is usable (no package, no
    fleet, no key); a backend that simply finds no stamp returns None — that is
    a legitimate answer for footage without an overlay.
    """
    if backend == "paddle" or (backend == "auto" and paddleocr_available()):
        frame = extract_frame(video_path, t_s=t_s)
        return read_timestamp_paddle(frame, t_s=t_s)
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
