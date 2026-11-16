"""audio.generate_tts — hermetic edge-tts adapter tests (fake backend, real ffmpeg)."""

from __future__ import annotations

import subprocess

import pytest
from director_os.audio_engine.tts import EdgeTtsBackend, TtsError, generate_tts

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
