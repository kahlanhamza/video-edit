"""Silero VAD streaming-protocol regression tests.

The 2026-11-17 bug: `speech_regions` fed bare 512-sample chunks, but the
official Silero v5 ONNX streaming protocol prepends a 64-sample rolling
context to every chunk (576 samples per step). Without it the model's
probabilities collapse to <0.1 on real speech — the 2-Bus duck gate went
VAD-less and every SNR audit silently measured the wrong speech map.
These tests lock the protocol (fake session) and the perception (real
bundled model + a committed real-speech fixture; both offline).
"""

from __future__ import annotations

import numpy as np
import pytest
from director_os.audio_engine.vad import (
    CHUNK_SAMPLES,
    CONTEXT_SAMPLES,
    VAD_MODEL_PATH,
    SileroVad,
)

FIXTURE = "tests/fixtures/speech_16k_mono.wav"


class _RecordingSession:
    """Minimal ort.InferenceSession stand-in that records input widths."""

    def __init__(self):
        self.inputs: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.prob_schedule = [0.9] * 100  # every chunk "speech"

    class _Tensor:
        shape = [2, 1, 128]  # state spec: shape[1] -> 128

    def get_inputs(self):  # noqa: ANN201 - fake signature
        return [self._Tensor(), self._Tensor(), self._Tensor()]

    def run(self, _names, feeds):  # noqa: ANN001 - fake signature
        window = feeds["input"]
        self.inputs.append(np.array(window))
        self.states.append(np.array(feeds["state"]))
        state_next = np.zeros_like(feeds["state"])
        return np.array([[self.prob_schedule.pop(0)]]), state_next


def _vad_with_session(session: _RecordingSession) -> SileroVad:
    vad = object.__new__(SileroVad)
    vad.session = session  # type: ignore[assignment]
    vad.threshold = 0.5
    return vad


def test_streaming_protocol_prepends_rolling_context() -> None:
    pcm = np.zeros(CHUNK_SAMPLES * 6 + 10, dtype=np.float32)
    pcm[100:200] = 0.5  # mark content so the roll-over is observable
    rec = _RecordingSession()
    spans = _vad_with_session(rec).speech_regions(pcm)

    assert rec.inputs, "no inference steps ran"
    assert len(rec.inputs) == 6  # floor(762 / 512) chunks
    for k, window in enumerate(rec.inputs):
        assert window.shape == (1, CHUNK_SAMPLES + CONTEXT_SAMPLES)
        if k > 0:
            # window's head must be the previous chunk's tail (rolling context)
            prev_tail = rec.inputs[k - 1][0, -CONTEXT_SAMPLES:]
            assert np.array_equal(window[0, :CONTEXT_SAMPLES], prev_tail)
    assert spans == [(0.0, pytest.approx((CHUNK_SAMPLES * 6) / 16_000))]


def test_streaming_protocol_resets_between_calls() -> None:
    pcm = np.zeros(CHUNK_SAMPLES * 2, dtype=np.float32)
    rec = _RecordingSession()
    vad = _vad_with_session(rec)
    vad.speech_regions(pcm)
    first_state = rec.states[0].copy()
    vad.speech_regions(pcm)
    assert np.array_equal(first_state, rec.states[0]), (
        "state leaked across calls — a second read would inherit stale context"
    )


def test_silero_vad_detects_real_speech_fixture() -> None:
    if not VAD_MODEL_PATH.exists():
        pytest.skip("silero_vad.onnx not bundled")
    from director_os.audio_engine.leveling import decode_pcm_mono

    pcm = decode_pcm_mono(FIXTURE, 16_000)
    dur = pcm.size / 16_000
    assert dur > 6.0, "fixture truncated"
    spans = SileroVad().speech_regions(pcm)
    covered = sum(e - s for s, e in spans) / dur
    assert spans, "no speech found in a real-speech fixture (protocol regression)"
    assert covered >= 0.55, f"only {covered:.0%} of a continuous-speech fixture gated as speech"
