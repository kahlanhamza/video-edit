"""FFprobe-based QA probe — the deterministic half of qa.audit_technical (PLAN §3.3G).

Also owns the *probe side* of the audio law (ADR-0005): a rendered file that is not
48 kHz stereo is a failed render, full stop.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
