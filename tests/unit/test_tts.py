"""audio.generate_tts — hermetic edge-tts adapter tests (fake backend, real ffmpeg)."""

from __future__ import annotations

import subprocess

import httpx
import pytest
from director_os.audio_engine.tts import (
    EdgeTtsBackend,
    ElevenLabsTtsBackend,
    TtsError,
    WordStamp,
    generate_tts,
)

RAW_S = 2.0
TONE_HEAD_S = 0.4
TONE_END_S = 1.6


def _synthetic_narration_wav_bytes() -> bytes:
    """Real audio: 0.4s silence + 1.2s 220Hz tone + 0.4s silence (2.0s total)."""
    expr = f"if(lt(t\\,{TONE_HEAD_S})\\,0\\,if(lt(t\\,{TONE_END_S})\\,sin(2*PI*220*t)\\,0))"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"aevalsrc={expr}:s=48000:d={RAW_S}",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-",
    ]
    r = subprocess.run(cmd, capture_output=True, check=True)
    return r.stdout


class FakeTtsBackend:
    """Returns deterministic real audio bytes — no network."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, text: str, voice: str) -> bytes:
        self.calls.append((text, voice))
        return _synthetic_narration_wav_bytes()


@pytest.fixture()
def fake_backend():
    return FakeTtsBackend()


def test_generate_tts_trims_pauses_and_pins_audio_law(tmp_path, fake_backend):
    out = tmp_path / "narr" / "line001.mp3"
    result = generate_tts("Bodycam footage shows the alley exit.", out, backend=fake_backend)

    assert fake_backend.calls == [("Bodycam footage shows the alley exit.", result.voice)]
    assert out.exists() and out.stat().st_size > 0
    # dead padding removed (keeps 0.3s head / 0.4s tail pacing room): 2.0s -> ~1.9s
    assert 1.0 < result.duration_s < RAW_S - 0.05
    # THE LAW on the stem: 48 kHz stereo (ADR-0005)
    from director_os.core.qa_probe import probe

    p = probe(out)
    assert p.sample_rate == 48_000
    assert p.channels == 2
    # no .raw litter
    raw = out.with_suffix(".raw.mp3")
    assert not raw.exists()


def test_generate_tts_no_trim_keeps_full_duration(tmp_path, fake_backend):
    out = tmp_path / "line002.mp3"
    result = generate_tts("Second line.", out, backend=fake_backend, trim_pauses=False)
    assert abs(result.duration_s - RAW_S) < 0.3


def test_generate_tts_rejects_empty_text(tmp_path, fake_backend):
    with pytest.raises(TtsError, match="empty"):
        generate_tts("   ", tmp_path / "x.mp3", backend=fake_backend)


def test_generate_tts_rejects_empty_audio(tmp_path):
    class EmptyBackend:
        def synthesize(self, text: str, voice: str) -> bytes:
            return b""

    with pytest.raises(TtsError, match="empty audio"):
        generate_tts("text", tmp_path / "x.mp3", backend=EmptyBackend())


def test_edge_tts_backend_exists_and_is_default():
    """The real backend lazily imports edge_tts; generate_tts defaults to it."""
    from director_os.audio_engine.tts import DEFAULT_VOICE

    assert DEFAULT_VOICE.startswith("en-")
    assert isinstance(EdgeTtsBackend(), EdgeTtsBackend)


# --------------------------------------------------------------------------- providers
# ElevenLabsTtsBackend: pure httpx against /v1/text-to-speech/{vid}/with-timestamps
# (MockTransport keeps tests hermetic — no key burned, no SDK installed).


def _elevenlabs_payload(text: str) -> dict:
    """Real response shape: base64 audio + per-character alignment."""
    import base64

    chars = list(text)
    starts = [i * 0.05 for i in range(len(chars))]
    ends = [s + 0.04 for s in starts]
    return {
        "audio_base64": base64.b64encode(b"ID3fake-mp3-bytes").decode(),
        "alignment": {
            "characters": chars,
            "character_start_times_seconds": starts,
            "character_end_times_seconds": ends,
        },
    }


def _mock_elevenlabs(handler):
    import httpx

    return ElevenLabsTtsBackend(api_key="test-key", transport=httpx.MockTransport(handler))


def test_elevenlabs_word_stamps_from_character_alignment():
    text = "hello officer"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.startswith("/v1/text-to-speech/")
        assert request.url.params["output_format"] == "mp3_44100_128"
        assert request.headers["xi-api-key"] == "test-key"
        assert request.read().decode().count("hello officer") == 1
        return httpx.Response(200, json=_elevenlabs_payload(text))

    backend = _mock_elevenlabs(handler)
    audio, words = backend.synthesize_with_timestamps(text)
    assert audio == b"ID3fake-mp3-bytes"
    assert [w.word for w in words] == ["hello", "officer"]
    # word boundary folds first char start -> last char end
    assert words[0].start_s == pytest.approx(0.0)
    assert words[0].end_s == pytest.approx(0.0 + 4 * 0.05 + 0.04)
    assert words[1].start_s == pytest.approx(6 * 0.05)


def test_elevenlabs_ms_voice_name_falls_back_to_configured_voice():
    """generate_tts's DEFAULT_VOICE (en-US-*) must not leak into the ElevenLabs URL."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["vid"] = request.url.path.split("/")[-2]
        return httpx.Response(200, json=_elevenlabs_payload("ok"))

    backend = _mock_elevenlabs(handler)
    backend.synthesize("ok", "en-US-GuyNeural")
    assert captured["vid"] == "21m00Tcm4TlvDq8ikWAM"


def test_elevenlabs_http_error_raises_tts_error():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": {"status": "invalid_api_key"}})

    backend = _mock_elevenlabs(handler)
    with pytest.raises(TtsError, match="401"):
        backend.synthesize("x")


def test_build_tts_backend_selection(monkeypatch):
    from director_os.audio_engine import tts as tts_mod
    from director_os.core.config import get_settings

    monkeypatch.setenv("TTS_PROVIDER", "edge")
    get_settings.cache_clear()
    assert isinstance(tts_mod.build_tts_backend(), EdgeTtsBackend)

    monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k1")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "v9")
    get_settings.cache_clear()
    b = tts_mod.build_tts_backend()
    assert isinstance(b, ElevenLabsTtsBackend)
    assert b.voice_id == "v9"

    monkeypatch.delenv("ELEVENLABS_API_KEY")
    get_settings.cache_clear()
    with pytest.raises(TtsError, match="ELEVENLABS_API_KEY"):
        tts_mod.build_tts_backend()

    monkeypatch.setenv("TTS_PROVIDER", "edge")
    get_settings.cache_clear()


def test_generate_tts_word_stamps_flow_through(tmp_path):
    """with-timestamps providers must be called ONCE (no double-billing)."""

    class AligningBackend:
        provider = "fake-align"

        def __init__(self):
            self.calls = 0

        def synthesize(self, text: str, voice: str) -> bytes:
            self.calls += 1
            return _synthetic_narration_wav_bytes()

        def synthesize_with_timestamps(self, text: str, voice: str):
            self.calls += 1
            return _synthetic_narration_wav_bytes(), (WordStamp("hi", 0.1, 0.5),)

    b = AligningBackend()
    result = generate_tts("word aligned narration", tmp_path / "n.mp3", backend=b)
    assert b.calls == 1  # with-timestamps path, no second synthesize call
    assert result.provider == "fake-align"
    assert result.words[0].word == "hi"
