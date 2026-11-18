"""generate_tts_cached — content-addressed voiceover clips (Stage: narration).

The re-render law: an unchanged script is a cache HIT (provider never called,
never billed); a one-word edit is a MISS with a different key.
"""

from __future__ import annotations

import subprocess

import pytest
from director_os.audio_engine.tts import TtsResult, generate_tts_cached, tts_cache_key


def _synth_wav() -> bytes:
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "aevalsrc=0.4*sin(2*PI*440*t):s=48000:d=1.0",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-",
    ]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


class CountingBackend:
    provider = "fake-tts"
    model_id = "fake-model"

    def __init__(self):
        self.calls: list[str] = []

    def synthesize(self, text: str, voice: str) -> bytes:
        self.calls.append(text)
        return _synth_wav()


def test_cache_miss_then_hit(tmp_path):
    backend = CountingBackend()
    r1, hit1 = generate_tts_cached(tmp_path, "Bodycam shows the alley exit.", backend=backend)
    assert hit1 is False
    assert len(backend.calls) == 1
    assert r1.path.exists() and r1.duration_s > 0

    r2, hit2 = generate_tts_cached(tmp_path, "Bodycam shows the alley exit.", backend=backend)
    assert hit2 is True
    assert len(backend.calls) == 1, "unchanged script must never re-synthesize"
    assert r2.path == r1.path
    assert abs(r2.duration_s - r1.duration_s) < 1e-9


def test_one_word_edit_changes_key(tmp_path):
    backend = CountingBackend()
    generate_tts_cached(tmp_path, "Bodycam shows the alley exit.", backend=backend)
    generate_tts_cached(tmp_path, "Bodycam shows the alley exits.", backend=backend)
    assert len(backend.calls) == 2
    keys = {p.name for p in tmp_path.glob("*.mp3")}
    assert len(keys) == 2


def test_whitespace_normalization_shares_key(tmp_path):
    backend = CountingBackend()
    generate_tts_cached(tmp_path, "same  words", backend=backend)
    _, hit = generate_tts_cached(tmp_path, "same words", backend=backend)
    assert hit is True


def test_corrupt_sidecar_resynthesizes(tmp_path):
    backend = CountingBackend()
    r1, _ = generate_tts_cached(tmp_path, "line one", backend=backend)
    # corrupt the sidecar: the cache must self-heal, not crash
    sidecar = r1.path.with_suffix(".json")
    sidecar.write_text("{not json")
    r2, hit = generate_tts_cached(tmp_path, "line one", backend=backend)
    assert hit is False
    assert r2.duration_s > 0


def test_empty_text_rejected(tmp_path):
    with pytest.raises(Exception, match="empty"):
        generate_tts_cached(tmp_path, "   ", backend=CountingBackend())


def test_cache_key_law():
    a = tts_cache_key("hello", "v", "edge", True, "")
    b = tts_cache_key("hello", "v", "edge", True, "")
    c = tts_cache_key("hello", "v", "edge", False, "")
    d = tts_cache_key("hello ", "v", "edge", True, "")
    assert a == b
    assert a != c
    assert a == d  # whitespace-normalized
    assert len(a) == 40


def test_result_shape_on_hit(tmp_path):
    backend = CountingBackend()
    generate_tts_cached(tmp_path, "narration words", backend=backend)
    r, hit = generate_tts_cached(tmp_path, "narration words", backend=backend)
    assert hit and isinstance(r, TtsResult)
    assert r.provider == "fake-tts"
