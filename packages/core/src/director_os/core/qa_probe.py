"""FFprobe-based QA probe — the deterministic half of qa.audit_technical (PLAN §3.3G).

Also owns the *probe side* of the audio law (ADR-0005): a rendered file that is not
48 kHz stereo is a failed render, full stop.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SUPPORTED_SAMPLE_RATE = 48_000
SUPPORTED_CHANNELS = 2


class AudioLawError(RuntimeError):
    pass


class ProbeResult(BaseModel):
    duration_s: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_rate: int | None = None
    has_video: bool = False
    has_audio: bool = False


def _ffprobe(path: str | Path, *extra: str) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", *extra, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout)


def has_audio_stream(path: str | Path) -> bool:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return "audio" in (r.stdout or "")


def probe(path: str | Path) -> ProbeResult:
    data = _ffprobe(path, "-show_format", "-show_streams")
    fmt = data["format"]
    res = ProbeResult(
        duration_s=float(fmt["duration"]),
        bit_rate=int(fmt.get("bit_rate") or 0),
    )
    for s in data.get("streams", []):
        if s["codec_type"] == "video" and not res.has_video:
            res.has_video = True
            res.video_codec = s.get("codec_name")
            res.width = int(s["width"])
            res.height = int(s["height"])
            rate = s.get("r_frame_rate", "30/1")
            num, _, den = rate.partition("/")
            res.fps = float(num) / float(den or 1)
        elif s["codec_type"] == "audio" and not res.has_audio:
            res.has_audio = True
            res.audio_codec = s.get("codec_name")
            res.sample_rate = int(s.get("sample_rate") or 0)
            res.channels = int(s.get("channels") or 0)
    return res


def assert_audio_law(result: ProbeResult) -> None:
    """A rendered master that is not 48 kHz stereo fails loudly (ADR-0005)."""
    if not result.has_audio:
        raise AudioLawError("audio law violated: no audio stream")
    if result.sample_rate != SUPPORTED_SAMPLE_RATE or result.channels != SUPPORTED_CHANNELS:
        raise AudioLawError(
            f"audio law violated: got {result.sample_rate} Hz / {result.channels}ch, "
            f"required {SUPPORTED_SAMPLE_RATE} Hz / {SUPPORTED_CHANNELS}ch "
            "(mixed sample rates silently mute downstream narrator clips)"
        )


def blackdetect_seconds(path: str | Path, min_dur: float = 0.5) -> float:
    """Total seconds of black video (ffmpeg blackdetect) — used by qa gates."""
    r = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(path),
            "-vf",
            f"blackdetect=d={min_dur}:pix_th=0.98",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    total = 0.0
    for line in (r.stderr or "").splitlines():
        if "black_start" in line:
            for token in line.split():
                if token.startswith("black_duration="):
                    total += float(token.split("=", 1)[1])
    return total


class QASummary(BaseModel):
    path: str
    probe: ProbeResult
    black_seconds: float = Field(default=0.0)
    audio_law_ok: bool = False


class AuditCheck(BaseModel):
    """One named QA gate result (PLAN §3.3G `qa.audit_technical`)."""

    name: str
    ok: bool
    detail: str = ""


class AuditReport(BaseModel):
    path: str
    ok: bool
    checks: list[AuditCheck] = Field(default_factory=list)

    @property
    def failures(self) -> list[AuditCheck]:
        return [c for c in self.checks if not c.ok]


def audit_technical(
    path: str | Path,
    expected_dur_s: float | None = None,
    max_black_seconds: float = 3.0,
    duration_tolerance: float = 1.0,
) -> AuditReport:
    """Deterministic technical QA of a rendered master (the `qa` pipeline stage).

    Gates: playable streams present, the 48 kHz stereo audio law, black-frame
    budget, and (when a target is given) duration drift vs. the compiled DAG.
    """
    path = Path(path)
    checks: list[AuditCheck] = []

    try:
        info = probe(path)
        checks.append(
            AuditCheck(
                name="streams_present",
                ok=info.has_video and info.has_audio,
                detail=f"video={info.has_video} audio={info.has_audio}",
            )
        )
    except Exception as e:  # unplayable file: everything else is moot
        return AuditReport(
            path=str(path),
            ok=False,
            checks=[AuditCheck(name="streams_present", ok=False, detail=str(e))],
        )

    law_ok = True
    law_detail = ""
    try:
        assert_audio_law(info)
        law_detail = f"{info.sample_rate}Hz/{info.channels}ch"
    except AudioLawError as e:
        law_ok = False
        law_detail = str(e)
    checks.append(AuditCheck(name="audio_law_48k_stereo", ok=law_ok, detail=law_detail))

    black = blackdetect_seconds(path)
    checks.append(
        AuditCheck(
            name="black_frame_budget",
            ok=black <= max_black_seconds,
            detail=f"{black:.2f}s black (max {max_black_seconds:.1f}s)",
        )
    )

    if expected_dur_s is not None:
        drift = abs(info.duration_s - expected_dur_s)
        checks.append(
            AuditCheck(
                name="duration_drift",
                ok=drift <= duration_tolerance,
                detail=f"final={info.duration_s:.2f}s expected={expected_dur_s:.2f}s "
                f"(drift {drift:.2f}s, tol {duration_tolerance:.2f}s)",
            )
        )

    return AuditReport(path=str(path), ok=all(c.ok for c in checks), checks=checks)


def audit_layout_and_policy(
    ass_path: str | Path | None = None,
    cards: list[dict[str, Any]] | None = None,
    blurs: list[dict[str, Any]] | None = None,
    frame: tuple[int, int] = (1920, 1080),
    lane_margins: tuple[int, int] = (46, 96),
) -> AuditReport:
    """Deterministic layout/policy audit (PLAN §3.3G `qa.audit_layout_and_policy`).

    Gates (each becomes a named check):
    - subtitle_lanes: the two-lane margin law holds (MarginV 46/96), type is
      upright bold, and cues within one lane never overlap in time
    - card_policy: every card carries a non-empty title
    - blur_policy: every privacy blur box stays inside the canvas, sigma sane

    The subtitle gate re-reads the *rendered* .ass file with pysubs2 — defense
    in depth against the builder: what ships is what gets audited.
    """
    checks: list[AuditCheck] = []
    any_input = ass_path is not None or cards or blurs

    if ass_path is not None:
        checks.append(_audit_subtitle_lanes(Path(ass_path), lane_margins))

    for card in cards or []:
        fields = card.get("fields") or {}
        title = str(fields.get("title") or "").strip()
        checks.append(
            AuditCheck(
                name="card_policy",
                ok=bool(title),
                detail=(
                    f"card '{card.get('template')}' title ok" if title else "card has empty title"
                ),
            )
        )

    for i, blur in enumerate(blurs or []):
        bbox = blur.get("bbox") or {}
        try:
            x, y, w, h = (int(bbox[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            checks.append(
                AuditCheck(name="blur_policy", ok=False, detail=f"blur {i}: malformed bbox {bbox}")
            )
            continue
        inside = x >= 0 and y >= 0 and x + w <= frame[0] and y + h <= frame[1]
        sigma = blur.get("sigma", 0)
        sigma_ok = isinstance(sigma, (int, float)) and 1 <= sigma <= 200
        ok = inside and sigma_ok
        detail = (
            f"blur {i} bbox {bbox} sigma {sigma}: "
            + ("inside frame" if inside else f"EXCEEDS frame {frame[0]}x{frame[1]}")
            + ("" if sigma_ok else ", sigma out of 1..200")
        )
        checks.append(AuditCheck(name="blur_policy", ok=ok, detail=detail))

    if not any_input:
        checks.append(AuditCheck(name="inputs", ok=False, detail="no ass/cards/blurs provided"))

    return AuditReport(path=str(ass_path or ""), ok=all(c.ok for c in checks), checks=checks)


def _audit_subtitle_lanes(ass_path: Path, lane_margins: tuple[int, int]) -> AuditCheck:
    """Margin law (46/96), upright bold faces, and per-lane no-overlap."""
    import pysubs2

    try:
        subs = pysubs2.SSAFile.from_string(ass_path.read_text(encoding="utf-8"))
    except Exception as e:
        return AuditCheck(name="subtitle_lanes", ok=False, detail=f"ass unreadable: {e}")

    problems: list[str] = []
    lane_styles: dict[int, list[str]] = {1: [], 2: []}
    for style_name, style in subs.styles.items():
        m = re.match(r"Lane([12])_", style_name)
        if not m:
            continue
        lane = int(m.group(1))
        lane_styles[lane].append(style_name)
        expected = lane_margins[lane - 1]
        if style.marginv != expected:
            problems.append(f"{style_name}: MarginV {style.marginv} != law {expected}")
        if not style.bold:
            problems.append(f"{style_name}: not bold (law: upright bold)")
        if style.italic:
            problems.append(f"{style_name}: italic (law: upright)")

    events_by_lane: dict[int, list[pysubs2.SSAEvent]] = {}
    for event in subs.events:
        m = re.match(r"Lane([12])_", event.style or "")
        if not m:
            continue
        events_by_lane.setdefault(int(m.group(1)), []).append(event)

    for lane, events in events_by_lane.items():
        ordered = sorted(events, key=lambda e: e.start)
        for a, b in zip(ordered, ordered[1:], strict=False):
            if b.start < a.end:
                problems.append(
                    f"lane {lane} overlap at {a.end / 1000:.2f}s "
                    f"({b.start / 1000:.2f}s starts before previous cue ends)"
                )
                break

    if problems:
        return AuditCheck(name="subtitle_lanes", ok=False, detail="; ".join(problems[:4]))
    lanes_used = sorted(events_by_lane)
    return AuditCheck(
        name="subtitle_lanes",
        ok=True,
        detail=(
            f"margin law {lane_margins[0]}/{lane_margins[1]} ok; lanes used: {lanes_used or 'none'}"
        ),
    )
