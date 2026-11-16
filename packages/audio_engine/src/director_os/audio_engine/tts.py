"""audio.generate_tts — edge-tts narration synthesis (PLAN §4.5, M2).

Sync adapter around edge-tts with two hard properties:

1. Backend injection: the `TtsBackend` port keeps tests hermetic — a fake
   backend provides real audio bytes without network. The real backend lazily
   imports `edge_tts` so the package stays optional at import time.
2. Pause trimming + THE LAW: raw TTS arrives with dead padding on both ends;
   `tts_pause_trim_chain` removes it, and `audio_output_args` pins 48 kHz
   stereo aac on the stem (ADR-0005 audio law), so narration stems drop into
   the mix unchanged.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from director_os.audio_engine.chains import tts_pause_trim_chain
from director_os.core.ffmpeg_args import audio_output_args
from director_os.core.qa_probe import probe

DEFAULT_VOICE = "en-US-GuyNeural"


class TtsError(RuntimeError):
    pass


class TtsBackend(Protocol):
    """Produces audio bytes (mp3/wav) for a text + voice pair."""

    def synthesize(self, text: str, voice: str) -> bytes: ...


class EdgeTtsBackend:
    """Real backend: edge-tts (Microsoft Edge neural voices, no API key)."""

    def synthesize(self, text: str, voice: str) -> bytes:
        try:
            import edge_tts
        except ImportError as e:  # pragma: no cover - exercised only without dep
            raise TtsError("edge-tts is not installed (uv sync should provide it)") from e

        async def _run() -> bytes:
            comms = edge_tts.Communicate(text, voice)
            chunks: list[bytes] = []
            async for chunk in comms.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        try:
            return asyncio.run(_run())
        except Exception as e:
            raise TtsError(f"edge-tts synthesis failed: {e}") from e


@dataclass(frozen=True)
class TtsResult:
    path: Path
    duration_s: float
    voice: str
    text_chars: int
    provider: str


def generate_tts(
    text: str,
    out_path: str | Path,
    voice: str = DEFAULT_VOICE,
    backend: TtsBackend | None = None,
    trim_pauses: bool = True,
) -> TtsResult:
    """Synthesize narration -> trim dead padding -> 48 kHz stereo stem.

    `text` must be non-empty (an empty narration would silently produce a
    silent stem that later fails the QA loudness floor — fail here instead).
    """
    text = text.strip()
    if not text:
        raise TtsError("refusing to synthesize empty narration text")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend = backend or EdgeTtsBackend()
    raw = backend.synthesize(text, voice)
    if not raw:
        raise TtsError("tts backend returned empty audio")

    raw_path = out_path.with_suffix(".raw" + out_path.suffix)
    raw_path.write_bytes(raw)

    # container/codec pairing: an mp3 container refuses AAC — pick a legal pair
    codec, bitrate = (
        ("libmp3lame", "192k") if out_path.suffix.lower() == ".mp3" else ("aac", "192k")
    )
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw_path)]
    if trim_pauses:
        cmd += ["-af", tts_pause_trim_chain()]
    cmd += [*audio_output_args(codec=codec, bitrate=bitrate), str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise TtsError(f"tts trim/master failed: {r.stderr[-400:]}")
    raw_path.unlink(missing_ok=True)

    result = probe(out_path)
    return TtsResult(
        path=out_path,
        duration_s=result.duration_s,
        voice=voice,
        text_chars=len(text),
        provider="edge-tts",
    )
