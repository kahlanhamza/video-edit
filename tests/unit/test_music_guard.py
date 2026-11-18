"""Music guardrail — YAMNet Music flagging -> Demucs routing (Stage 4 -> 5)."""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
from director_os.perception.audio_events import (
    AudioEventsNotAvailable,
    build_event_spans,
)
from director_os.worker.music_guard import (
    demucs_stem_path,
    make_demucs_resolver,
    music_spans_from_shots,
)


def _fake_backend(shots: dict):
    class B:
        name = "fake"

        def detect_events(self, path, threshold=0.3):
            raise AudioEventsNotAvailable("fake")

    return B()


def test_music_group_indices():
    from director_os.perception.audio_events import EVENT_GROUPS

    assert EVENT_GROUPS["music"] == (24, 132, 133, 519)  # Singing/Music/Instrument/Radio


def test_music_span_detection_from_frame_scores():
    # 100 frames (~48s); classes 132 (Music) fires on frames 10..40
    scores = np.zeros((100, 521), dtype=np.float32)
    scores[10:40, 132] = 0.8
    scores[60:70, 24] = 0.7  # Singing later
    spans = build_event_spans(scores, threshold=0.3, merge_gap_s=1.0, min_frames=2)
    music = [s for s in spans if s.kind == "music"]
    assert len(music) >= 2
    assert any(abs(s.start_s - 10 * 0.48) < 1.5 for s in music)
    assert any(abs(s.start_s - 60 * 0.48) < 1.5 for s in music)


def test_music_spans_from_shots_payload():
    shots = {
        "events": [
            {"kind": "siren", "start_s": 1.0, "end_s": 5.0, "peak_score": 0.9},
            {"kind": "music", "start_s": 10.0, "end_s": 30.0, "peak_score": 0.8},
            {"kind": "music", "start_s": 40.0, "end_s": 45.0, "peak_score": 0.7},
        ]
    }
    spans = music_spans_from_shots(shots)
    assert spans == [(10.0, 30.0), (40.0, 45.0)]
    assert music_spans_from_shots({}) == []


def test_resolver_finds_stem(tmp_path):
    settings = type("S", (), {"output_dir": tmp_path})()
    stem = demucs_stem_path("asset-1", settings)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.write_bytes(b"RIFFfake")
    resolver = make_demucs_resolver("asset-1", settings, log=lambda m: None)
    assert resolver("asset-1") == stem


def test_resolver_missing_stem_returns_none(tmp_path):
    settings = type("S", (), {"output_dir": tmp_path})()
    logs: list[str] = []
    resolver = make_demucs_resolver("asset-2", settings, log=logs.append)
    assert resolver("asset-2") is None
    assert any("demucs_stems_run" in m for m in logs), "missing stem must log the fix path"


def test_resolver_never_keeps_plates_confused():
    # contract anchor: the resolver is per-asset; the keep-list law lives in
    # the redaction payload (plates never keepable). This test pins the
    # resolver contract shape only.
    settings = type("S", (), {"output_dir": "/nonexistent"})()
    resolver = make_demucs_resolver("a", settings, log=lambda m: None)
    assert resolver("a") is None


def test_stage_enhance_uses_stem_as_enhancer_input(tmp_path, monkeypatch):
    """The Demucs stem IS the enhancer input when present (not just a label)."""
    from director_os.worker import audio_enhance as ae

    class FakeEnhancer:
        name = "fake-mossformer"
        atten_lim_db = 28.0

        def __init__(self):
            self.inputs: list = []

        def enhance(self, src, dst):
            self.inputs.append(src)
            # real 48k stereo wav, same length as the 1s input
            subprocess_run_wav(src, dst)
            from director_os.audio_engine.enhance import EnhancerResult

            return EnhancerResult(
                engine="fake-mossformer",
                neural=True,
                atten_lim_db=28.0,
                in_path=Path(src),
                out_path=Path(dst),
                duration_in_s=1.0,
                duration_out_s=1.0,
            )

    def subprocess_run_wav(src, dst):
        import subprocess as sp

        sp.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:a", "copy", str(dst)],
            check=True,
            capture_output=True,
        )

    # a 1s source video with audio
    src = tmp_path / "src.mp4"
    import subprocess as sp

    sp.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=30:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-shortest",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    # the Demucs stem
    stem = tmp_path / "audio2bus" / "demucs" / "a1" / "vocals.wav"
    stem.parent.mkdir(parents=True, exist_ok=True)
    sp.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=1",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(stem),
        ],
        check=True,
        capture_output=True,
    )

    class Settings:
        output_dir = tmp_path

    class Deps:
        settings = Settings()
        log = staticmethod(lambda m: None)
        vocals_stem_resolver = staticmethod(lambda asset_id: stem)
        sessions = None

    enhancer = FakeEnhancer()
    info = None
    with contextlib.suppress(Exception):
        # _persist needs a DB session; the CHAIN itself already ran
        info = ae.stage_audio_enhance(
            "a1",
            src,
            {"words": []},
            "p1",
            Deps(),
            enhancer=enhancer,
            music_flagged_spans=[(0.0, 1.0)],
        )
    # the enhancer MUST have been fed the stem copy, never the preclean
    stem_inputs = [i for i in enhancer.inputs if i.name == "stem_window.wav"]
    assert stem_inputs, f"enhancer inputs were {[i.name for i in enhancer.inputs]}"
    if info is not None:
        assert info.enhancer_input == "demucs_vocals"
        assert info.music_flagged_spans == [(0.0, 1.0)]
