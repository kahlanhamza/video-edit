"""Silero VAD (ONNX, MIT) — speech-presence truth for the 2-Bus gate.

ASR word timestamps tell us where WORDS are; Silero VAD tells us where the
VOICE is (breaths, trailing syllables, shouting that ASR missed). The 2-Bus
duck gate takes the UNION of both, so a car blast in an ASR gap never gets
mistaken for speech and a trailing "-yeah?" after the last word stays open.

The model ships inside the package (`audio_engine/models/silero_vad.onnx`,
~2.3 MB — same bundling law as perception's YuNet). If onnxruntime or the
model file is missing, callers degrade to ASR-intervals-only gating LOUDLY
(`VadUnavailableError` -> worker logs it), never silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VAD_MODEL_PATH = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"
VAD_SR = 16_000  # Silero VAD v5 ONNX native rate
CHUNK_SAMPLES = 512  # 32 ms at 16 kHz — the size the ONNX graph was exported for
CONTEXT_SAMPLES = 64  # v5 streaming context: last 64 samples prepended to every chunk
DEFAULT_THRESHOLD = 0.5


class VadUnavailableError(RuntimeError):
    pass


def merge_intervals(
    intervals: list[tuple[float, float]], min_gap_s: float = 0.12
) -> list[tuple[float, float]]:
    """Merge touching/overlapping spans (and spans separated by < min_gap)."""
    spans = sorted((s, e) for s, e in intervals if e > s)
    if not spans:
        return []
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s - out[-1][1] < min_gap_s:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


class SileroVad:
    """Streaming Silero VAD over PCM — deterministic, CPU, no heavy SDK."""

    def __init__(
        self,
        model_path: Path = VAD_MODEL_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover — depends on env install
            raise VadUnavailableError("onnxruntime not installed — Silero VAD unavailable") from e
        if not Path(model_path).exists():
            raise VadUnavailableError(f"silero_vad.onnx missing at {model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts)
        self.threshold = threshold

    def speech_regions(self, pcm_16k: np.ndarray) -> list[tuple[float, float]]:
        """PCM float32 mono @16 kHz -> speech spans in seconds (probability-gated).

        Streaming protocol mirrors the OFFICIAL Silero v5 OnnxWrapper exactly:
        every 512-sample chunk is preceded by the last 64 samples of the
        previous chunk (the rolling context) — the ONNX graph consumes 576
        samples per step. Feeding bare 512-sample chunks starves the conv
        stack and collapses probabilities to <0.1 on real speech (the bug
        this class shipped with on 2026-11-17; found by the MossFormer2 SE
        kernel run's mirrored math contradicting the local gate).
        """
        n = self.session.get_inputs()[1].shape[1]
        state = np.zeros((2, 1, int(n) if isinstance(n, int) else 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        spans: list[tuple[float, float]] = []
        open_s: float | None = None
        x = pcm_16k.astype(np.float32)
        total = (x.size // CHUNK_SAMPLES) * CHUNK_SAMPLES
        for i in range(0, total, CHUNK_SAMPLES):
            window = np.concatenate(
                [context, x[i : i + CHUNK_SAMPLES].reshape(1, -1)], axis=1
            )
            prob, state = self.session.run(
                None,
                {"input": window, "state": state, "sr": np.array(VAD_SR, dtype=np.int64)},
            )
            context = window[:, -CONTEXT_SAMPLES:]
            p = float(prob[0][0])
            t0, t1 = i / VAD_SR, (i + CHUNK_SAMPLES) / VAD_SR
            if p >= self.threshold and open_s is None:
                open_s = t0
            elif p < self.threshold and open_s is not None:
                spans.append((open_s, t1))
                open_s = None
        if open_s is not None:
            spans.append((open_s, total / VAD_SR))
        return merge_intervals(spans)


def speech_gate_intervals(
    word_intervals: list[tuple[float, float]],
    vad_regions: list[tuple[float, float]] | None,
    pad_s: float = 0.10,
) -> list[tuple[float, float]]:
    """The gate truth: union(ASR words, VAD spans), lightly padded, merged.

    `vad_regions=None` (VAD unavailable) degrades to ASR-words-only — the
    caller logs that loudly; the gate stays conservative either way.
    """
    spans = [(max(0.0, s - pad_s), e + pad_s) for s, e in word_intervals if e > s]
    spans += [(s, e) for s, e in (vad_regions or [])]
    return merge_intervals(spans, min_gap_s=0.05)
