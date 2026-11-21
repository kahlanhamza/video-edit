"""audio.generate_tts — narration synthesis behind the TtsBackend port.

Port-and-adapter like LLMBackend: a `TtsBackend` port, provider selection via
`build_tts_backend()` (env `TTS_PROVIDER`), and every backend zero-heavy-SDK:

1. `EdgeTtsBackend`     — DEFAULT, $0, no key. Microsoft Edge neural voices
   via `edge-tts`. Great for dev, tests and prototypes; keep it through M2-M5
   so the pipeline burns $0.
2. `ElevenLabsTtsBackend` — OPTIONAL studio grade (`ELEVENLABS_API_KEY` +
   `ELEVENLABS_VOICE_ID`), pure httpx against
   `/v1/text-to-speech/{voice_id}/with-timestamps`: base64 audio + character
   alignment folded into word stamps for karaoke-accurate subtitles. Flip it
   on when the first monetized channel video goes live — one .env key, no
   code change (`TTS_PROVIDER=elevenlabs`).

Both feed the same sync adapter: pause trimming + THE LAW (raw TTS arrives
with dead padding on both ends; `tts_pause_trim_chain` removes it, and
`audio_output_args` pins 48 kHz stereo aac on the stem, ADR-0005), so
narration stems drop into the mix unchanged regardless of provider.
"""

from __future__ import annotations

import asyncio
import base64
import re as _re
import subprocess
import sys as _sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
from director_os.audio_engine.chains import tts_pause_trim_chain
from director_os.core.ffmpeg_args import audio_output_args
from director_os.core.qa_probe import probe

DEFAULT_VOICE = "en-US-GuyNeural"
# Retention law: narration must CATCH attention, never lull it (owner rule
# 2026-11-19 — the +0% default read sleepy and viewers bailed). +14% keeps
# Brian's documentary timbre while landing ~2.9-3.0 words/sec. Override with
# TTS_RATE env ("+0%" .. "+30%"); the rate rides the TTS cache key so a
# change re-synthesizes exactly the affected clips.
DEFAULT_TTS_RATE = "+14%"
# TTS front-end normalization law (owner directive 2026-11-21): the voice
# must SPEAK what a human narrator would say for ANY content. The editor AI
# agent (scripts/tts_norm.py, LLM-driven, validator-guarded, cached) rewrites
# the line before synthesis; when the agent is unreachable the raw line is
# spoken as written — normalization upgrades the voice, it never blocks it.
TTS_NORMALIZE = __import__("os").getenv("TTS_NORMALIZE", "1") == "1"


def agent_normalize(text: str) -> str:
    """Speak-form rewrite via the editor AI agent (fail-open)."""
    if not TTS_NORMALIZE or not text or not text.strip():
        return text
    try:
        scripts_dir = Path(__import__("os").getenv(
            "DIRECTOR_SCRIPTS_DIR", "/home/z/my-project/scripts"
        ))
        if str(scripts_dir) not in _sys.path:
            _sys.path.insert(0, str(scripts_dir))
        from tts_norm import normalize as _norm

        return _norm(text)
    except Exception:  # noqa: BLE001
        return text


def resolve_tts_rate(rate: str | None = None) -> str:
    """Explicit arg > TTS_RATE env > DEFAULT_TTS_RATE (validated shape)."""
    import os as _os

    value = (rate if rate is not None else _os.getenv("TTS_RATE")) or DEFAULT_TTS_RATE
    value = str(value).strip()
    if not _re.match(r"^[+-]\d{1,2}%$", value):
        raise TtsError(f"invalid TTS rate '{value}' (expected e.g. '+14%')")
    return value


ELEVENLABS_DEFAULT_BASE = "https://api.elevenlabs.io"
# public shared voice (Rachel) — overridable via ELEVENLABS_VOICE_ID
ELEVENLABS_DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
ELEVENLABS_DEFAULT_MODEL = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"


class TtsError(RuntimeError):
    pass


class TtsBackend(Protocol):
    """Produces audio bytes (mp3/wav) for a text + voice pair."""

    def synthesize(self, text: str, voice: str, rate: str | None = None) -> bytes: ...


class EdgeTtsBackend:
    """Real backend: edge-tts (Microsoft Edge neural voices, no API key)."""

    provider = "edge-tts"
    supports_rate = True

    def synthesize(self, text: str, voice: str, rate: str | None = None) -> bytes:
        audio, _ = self.synthesize_with_timestamps(text, voice, rate=rate)
        return audio

    def synthesize_with_timestamps(
        self, text: str, voice: str, rate: str | None = None
    ) -> tuple[bytes, tuple[WordStamp, ...]]:
        """Stream audio + WordBoundary events -> (bytes, word stamps).

        WordBoundary offsets are 100-ns ticks from the STREAM start (which is
        the raw clip's clock, pre-trim). `generate_tts` re-bases them onto the
        trimmed stem using the first word's offset.
        """
        try:
            import edge_tts
        except ImportError as e:  # pragma: no cover - exercised only without dep
            raise TtsError("edge-tts is not installed (uv sync should provide it)") from e

        async def _run() -> tuple[bytes, tuple[WordStamp, ...]]:
            comms = edge_tts.Communicate(
                text, voice, rate=resolve_tts_rate(rate), boundary="WordBoundary"
            )
            chunks: list[bytes] = []
            words: list[WordStamp] = []
            async for chunk in comms.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    token = str(chunk.get("text") or "").strip()
                    if not token:
                        continue
                    start_s = float(chunk.get("offset") or 0) / 1e7
                    end_s = start_s + float(chunk.get("duration") or 0) / 1e7
                    words.append(WordStamp(word=token, start_s=start_s, end_s=end_s))
            return b"".join(chunks), tuple(words)

        try:
            return asyncio.run(_run())
        except Exception as e:
            raise TtsError(f"edge-tts synthesis failed: {e}") from e


@dataclass(frozen=True)
class WordStamp:
    """One aligned word (ElevenLabs with-timestamps alignment -> words)."""

    word: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class TtsResult:
    path: Path
    duration_s: float
    voice: str
    text_chars: int
    provider: str
    words: tuple[WordStamp, ...] = ()


def _words_from_character_alignment(
    characters: list[str], starts: list[float], ends: list[float]
) -> tuple[WordStamp, ...]:
    """Fold per-character alignment into word stamps (whitespace is the split).

    A word spans from its first character's start to its last character's
    end; punctuation/space characters ride along inside the current word so
    gaps never swallow a word boundary.
    """
    words: list[WordStamp] = []
    buf = ""
    start: float | None = None
    prev_end = 0.0
    for ch, s, e in zip(characters, starts, ends, strict=False):
        if ch.isspace():
            if buf:
                words.append(
                    WordStamp(word=buf, start_s=float(start or 0.0), end_s=float(prev_end))
                )
                buf = ""
                start = None
            continue
        if start is None:
            start = s
        buf += ch
        prev_end = e
    if buf:
        words.append(WordStamp(word=buf, start_s=float(start or 0.0), end_s=float(prev_end)))
    return tuple(words)


class ElevenLabsTtsBackend:
    """Studio-grade provider — pure httpx, zero SDK, word-level timestamps.

    Endpoint: POST /v1/text-to-speech/{voice_id}/with-timestamps
              (?output_format=mp3_44100_128, header xi-api-key)
    Response: {"audio_base64": ..., "alignment": {"characters": [...],
              "character_start_times_seconds": [...],
              "character_end_times_seconds": [...]}}
    """

    provider = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        voice_id: str = "",
        model_id: str = ELEVENLABS_DEFAULT_MODEL,
        base_url: str = ELEVENLABS_DEFAULT_BASE,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 120.0,
    ):
        if not api_key:
            raise TtsError("ElevenLabsTtsBackend requires ELEVENLABS_API_KEY")
        self.api_key = api_key
        self.voice_id = voice_id or ELEVENLABS_DEFAULT_VOICE_ID
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout_s = timeout_s

    def _resolve_voice(self, voice: str) -> str:
        # a Microsoft-style voice name means "use the configured default"
        return self.voice_id if voice == DEFAULT_VOICE or not voice else voice

    def synthesize(self, text: str, voice: str = "", rate: str | None = None) -> bytes:
        audio, _ = self.synthesize_with_timestamps(text, voice)
        return audio

    def synthesize_with_timestamps(
        self, text: str, voice: str = ""
    ) -> tuple[bytes, tuple[WordStamp, ...]]:
        vid = self._resolve_voice(voice)
        url = f"{self.base_url}/v1/text-to-speech/{vid}/with-timestamps"
        params = {"output_format": ELEVENLABS_OUTPUT_FORMAT}
        payload = {"text": text, "model_id": self.model_id}
        try:
            with httpx.Client(transport=self._transport, timeout=self._timeout_s) as client:
                r = client.post(
                    url, headers={"xi-api-key": self.api_key}, params=params, json=payload
                )
        except httpx.HTTPError as e:
            raise TtsError(f"elevenlabs request failed: {e}") from e
        if r.status_code != 200:
            raise TtsError(f"elevenlabs {r.status_code}: {r.text[:300]}")
        data = r.json()
        audio = base64.b64decode(data.get("audio_base64", ""))
        if not audio:
            raise TtsError("elevenlabs returned empty audio")
        align = data.get("alignment") or {}
        words = _words_from_character_alignment(
            align.get("characters", []),
            align.get("character_start_times_seconds", []),
            align.get("character_end_times_seconds", []),
        )
        return audio, words


def build_tts_backend() -> TtsBackend:
    """Env-driven provider choice (config is data, same as build_backend).

    TTS_PROVIDER: `edge` (default, $0) | `elevenlabs` (needs ELEVENLABS_API_KEY).
    Raises TtsError (never silently falls back) when the requested provider is
    unconfigured — callers decide whether to degrade.
    """
    from director_os.core.config import get_settings

    s = get_settings()
    provider = (s.tts_provider or "edge").lower()
    if provider == "elevenlabs":
        if not s.elevenlabs_api_key:
            raise TtsError(
                "TTS_PROVIDER=elevenlabs requires ELEVENLABS_API_KEY "
                "(+ optional ELEVENLABS_VOICE_ID/ELEVENLABS_MODEL_ID)"
            )
        return ElevenLabsTtsBackend(
            api_key=s.elevenlabs_api_key,
            voice_id=s.elevenlabs_voice_id,
            model_id=s.elevenlabs_model_id,
        )
    return EdgeTtsBackend()


def tts_cache_key(
    text: str,
    voice: str,
    provider: str,
    trim_pauses: bool = True,
    model_id: str = "",
    rate: str | None = None,
) -> str:
    """Content-addressed TTS identity: provider | voice | model | text | trim | rate.

    The SAME law as the segment cache: an unchanged script never re-synthesizes
    (and never re-bills); a one-word edit changes exactly one key. The speaking
    rate rides the key — a pace change re-synthesizes exactly what it affects.
    """
    import hashlib
    import json as _json

    payload = _json.dumps(
        [provider, voice, model_id, " ".join(text.split()), trim_pauses, resolve_tts_rate(rate)],
        sort_keys=True,
    )
    # v3: the pause-trim chain now COMPRESSES internal pauses (>300ms -> 150ms)
    # — a chain change invalidates every cached stem
    return hashlib.sha256(("dos-tts-v3|" + payload).encode()).hexdigest()[:40]


def generate_tts_cached(
    out_dir: str | Path,
    text: str,
    voice: str = DEFAULT_VOICE,
    backend: TtsBackend | None = None,
    trim_pauses: bool = True,
    rate: str | None = None,
) -> tuple[TtsResult, bool]:
    """`generate_tts` behind a content-addressed clip cache.

    Clip + sidecar land in `out_dir/{key}.mp3` + `out_dir/{key}.json`. A hit
    returns the cached stem WITHOUT touching the provider (re-renders of an
    unchanged voiceover are free and byte-stable); a miss synthesizes, then
    persists atomically. Returns (result, cache_hit).
    """
    import json as _json
    import os as _os

    text = text.strip()
    if not text:
        raise TtsError("refusing to synthesize empty narration text")
    text = agent_normalize(text)  # spoken form is the cache identity
    backend = backend or build_tts_backend()
    provider = str(getattr(backend, "provider", "unknown"))
    model_id = str(getattr(backend, "model_id", ""))
    rate = resolve_tts_rate(rate)
    key = tts_cache_key(text, voice, provider, trim_pauses, model_id, rate)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clip = out / f"{key}.mp3"
    sidecar = out / f"{key}.json"

    if clip.exists() and clip.stat().st_size > 0 and sidecar.exists():
        try:
            data = _json.loads(sidecar.read_text())
            result = TtsResult(
                path=clip,
                duration_s=float(data["duration_s"]),
                voice=str(data.get("voice", voice)),
                text_chars=int(data.get("text_chars", len(text))),
                provider=str(data.get("provider", provider)),
                words=tuple(
                    WordStamp(
                        word=str(w["word"]), start_s=float(w["start_s"]), end_s=float(w["end_s"])
                    )
                    for w in data.get("words", [])
                ),
            )
            return result, True
        except (ValueError, KeyError, TypeError):
            pass  # corrupt sidecar: re-synthesize and overwrite below

    result = generate_tts(
        text, clip, voice=voice, backend=backend, trim_pauses=trim_pauses, rate=rate
    )
    tmp = sidecar.with_suffix(".tmp.json")
    tmp.write_text(
        _json.dumps(
            {
                "text": text,
                "voice": result.voice,
                "provider": result.provider,
                "model_id": model_id,
                "rate": rate,
                "duration_s": result.duration_s,
                "text_chars": result.text_chars,
                "words": [
                    {"word": w.word, "start_s": w.start_s, "end_s": w.end_s} for w in result.words
                ],
                "cache_key": key,
            },
            indent=2,
            sort_keys=True,
        )
    )
    _os.replace(tmp, sidecar)
    return result, False


def generate_tts(
    text: str,
    out_path: str | Path,
    voice: str = DEFAULT_VOICE,
    backend: TtsBackend | None = None,
    trim_pauses: bool = True,
    rate: str | None = None,
) -> TtsResult:
    """Synthesize narration -> trim dead padding -> 48 kHz stereo stem.

    `text` must be non-empty (an empty narration would silently produce a
    silent stem that later fails the QA loudness floor — fail here instead).
    `rate` (e.g. "+14%") is the retention-law speaking pace; it rides the
    cache key upstream so a pace change re-synthesizes exactly what it affects.
    """
    text = text.strip()
    if not text:
        raise TtsError("refusing to synthesize empty narration text")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend = backend or build_tts_backend()
    rate = resolve_tts_rate(rate) if getattr(backend, "supports_rate", False) else None

    # word stamps ride along when the provider aligns (edge WordBoundary /
    # ElevenLabs alignment) — call with-timestamps ONCE and reuse its audio
    # bytes (never double-bill)
    words: tuple[WordStamp, ...] = ()
    raw: bytes | None = None
    ts_method = getattr(backend, "synthesize_with_timestamps", None)
    if ts_method is not None:
        try:
            raw, words = ts_method(text, voice, rate=rate)
        except TypeError:
            try:
                raw, words = ts_method(text, voice)  # provider without rate support
            except TypeError:
                raw = None
    if not raw:
        try:
            raw = backend.synthesize(text, voice, rate=rate)
        except TypeError:
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

    # re-base word stamps onto the TRIMMED stem: the head trim removes exactly
    # the dead air before the first sound, and the first word's offset is that
    # same boundary (threshold -50 dB) — shift every stamp back by it and clip
    # to the trimmed duration.
    shifted: tuple[WordStamp, ...] = words
    if words and trim_pauses:
        head = words[0].start_s
        shifted = tuple(
            WordStamp(
                word=w.word,
                start_s=min(max(0.0, w.start_s - head), result.duration_s),
                end_s=min(max(0.0, w.end_s - head), result.duration_s),
            )
            for w in words
            if w.end_s > head
        )
    return TtsResult(
        path=out_path,
        duration_s=result.duration_s,
        voice=voice,
        text_chars=len(text),
        provider=getattr(backend, "provider", "unknown"),
        words=shifted,
    )
