"""Audio event detection — sirens, gunshots, shouting, glass, alarms (issue #12).

M4's audio side: `perception.detect_audio_events` per the PLAN. YAMNet (AudioSet,
521 classes) through ONNX Runtime on CPU — the $0 path, zero SDK, zero service.
The ONNX export (audiomagic/yamnet-onnx, Apache-2.0, converted 1:1 from Google's
TF Hub SavedModel) has the log-mel frontend BAKED INTO THE GRAPH: it consumes a
raw 16 kHz mono float32 waveform and emits per-frame scores (one frame per 0.48 s
hop, each covering 0.96 s). There is no feature extraction to reimplement — and
no way to get it subtly wrong.

Consumers:
- `audio_engine.leveling` gains NON-speech loud spans (sirens) as a complement
  to ASR speech intervals (the speech-gated meter already refuses whole-program
  loudness because sirens skew loudnorm — this makes those spans explicit data).
- retention/QA can flag "audio moment" anchors (gunshot, glass) for the Critic.

Port + adapter like everything external: `AudioEventBackend` protocol, real
YAMNet adapter, fakes in tests. The 16 MB model is a fetch-once data file
(`scripts/fetch_yamnet.py`, SHA-pinned) — absence degrades to
`AudioEventsNotAvailable`, never a pipeline blocker (soft enrichment).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

DEFAULT_THRESHOLD = 0.30
DEFAULT_MERGE_GAP_S = 1.0
DEFAULT_MIN_FRAMES = 2
# YAMNet frame cadence (fixed by the graph): 0.96 s window / 0.48 s hop
FRAME_HOP_S = 0.48
FRAME_WINDOW_S = 0.96
SAMPLE_RATE = 16000
# Chunked-inference law: one session.run over a FULL-tape waveform (43 min =
# 42M samples) explodes past ~4 GB inside onnxruntime and the cgroup OOM-kills
# the whole pipeline. 64-window chunks (61.44 s) with a 2-window overlap keep
# identical frame math (chunk and step are whole numbers of window and hop) at
# ~1/45 the peak memory.
CHUNK_WINDOWS = 64
OVERLAP_WINDOWS = 2

# AudioSet class indices (from yamnet_class_map.csv, pinned by the fetch script).
# Groups are OUR canonical event names -> the AudioSet ontology rows that vote
# for them. A group fires when ANY member class crosses the threshold.
# `music` is the Content-ID guardrail (Stage 4->5): car radios / store PA /
# background songs get the flagged ranges routed through Demucs vocal
# separation BEFORE MossFormer2, so copyrighted material never reaches the
# 2-bus master (indices: 24 Singing, 132 Music, 133 Musical instrument,
# 519 Radio).
EVENT_GROUPS: dict[str, tuple[int, ...]] = {
    "siren": (316, 317, 318, 319, 390, 391),  # emergency vehicle + sirens + civil defense
    "gunshot": (421,),  # Gunshot, gunfire
    "shout": (6, 10, 11),  # Shout / Children shouting / Screaming
    "glass": (435,),  # Glass
    "alarm": (382, 304),  # Alarm / Car alarm
    "music": (24, 132, 133, 519),  # Singing / Music / Musical instrument / Radio
}


class AudioEventSpan(BaseModel):
    """One detected sound event over a time window (source-audio seconds)."""

    kind: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    peak_score: float = Field(ge=0.0, le=1.0)


class AudioEventSweep(BaseModel):
    """The complete event map of one audio source."""

    backend: str
    duration_s: float = Field(ge=0.0)
    frame_hop_s: float = FRAME_HOP_S
    threshold: float
    events: list[AudioEventSpan] = Field(default_factory=list)

    def kinds(self) -> set[str]:
        return {e.kind for e in self.events}


class AudioEventsNotAvailable(RuntimeError):
    """The real YAMNet backend cannot run — message says which .env/file flips it."""


class AudioEventBackend(Protocol):
    """Structural port — fakes in tests, YAMNet in production."""

    name: str

    def detect_events(
        self,
        audio_path: Path | str,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> AudioEventSweep: ...


# ------------------------------------------------------------------ span build


def build_event_spans(
    frame_scores: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    min_frames: int = DEFAULT_MIN_FRAMES,
    hop_s: float = FRAME_HOP_S,
) -> list[AudioEventSpan]:
    """Frame scores (n_frames, 521) -> merged event spans. PURE — fully unit-tested.

    Per group: member classes vote per frame (max), frames above `threshold`
    form runs; runs separated by <= `merge_gap_s` merge (a siren wails through
    short dips); runs shorter than `min_frames` are dropped (one-frame blips
    are detector noise, not events). Span time = the union of its frames'
    windows: [first*hop, last*hop + window].
    """
    if frame_scores.ndim != 2:
        raise ValueError(f"frame_scores must be (n_frames, classes), got {frame_scores.shape}")
    n_frames = frame_scores.shape[0]
    gap_frames = max(0, int(round(merge_gap_s / hop_s)))
    spans: list[AudioEventSpan] = []
    for kind, members in EVENT_GROUPS.items():
        idx = [m for m in members if m < frame_scores.shape[1]]
        if not idx:
            continue
        group_score = frame_scores[:, idx].max(axis=1)
        above = group_score >= threshold
        # runs of consecutive True frames
        runs: list[tuple[int, int]] = []
        start = None
        for i, a in enumerate(above):
            if a and start is None:
                start = i
            elif not a and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, n_frames - 1))
        if not runs:
            continue
        # merge runs closer than the gap tolerance (silent frames BETWEEN runs
        # is the honest measure: s - e - 1 silence frames)
        merged: list[list[int]] = [list(runs[0])]
        for s, e in runs[1:]:
            if s - merged[-1][1] - 1 <= gap_frames:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        for s, e in merged:
            if (e - s + 1) < min_frames:
                continue
            spans.append(
                AudioEventSpan(
                    kind=kind,
                    start_s=round(s * hop_s, 3),
                    end_s=round((e + 1) * hop_s + FRAME_WINDOW_S - hop_s, 3),
                    peak_score=round(float(group_score[s : e + 1].max()), 4),
                )
            )
    spans.sort(key=lambda sp: (sp.start_s, sp.kind))
    return spans


# -------------------------------------------------------------------- decode


def decode_audio_pcm16k(audio_path: Path | str) -> np.ndarray:
    """Any media file -> mono 16 kHz float32 waveform (ffmpeg pipe, no temp files)."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise AudioEventsNotAvailable(
            f"audio decode failed ({r.returncode}): {r.stderr[-200:].decode(errors='replace')}"
        )
    wave = np.frombuffer(r.stdout, dtype=np.float32)
    return wave.copy()  # frombuffer is read-only; the session needs the buffer


# ------------------------------------------------------------------- backend


def _default_model_candidates(filename: str) -> list[Path]:
    """Repo-root data/models candidates (module parents, then cwd)."""
    here = Path(__file__).resolve()
    # parents: 0=perception 1=director_os 2=src 3=perception(pkg) 4=packages 5=<repo root>
    out = []
    for up in (5, 4):
        out.append(here.parents[up] / "data" / "models" / filename)
    out.append(Path.cwd() / "data" / "models" / filename)
    return out


def resolve_yamnet_model_path(model_path: str | Path | None = None) -> Path:
    """Search order: explicit arg (must exist) -> env -> repo data/models."""
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"explicit yamnet model not found: {p}")
        return p
    env = os.getenv("DIRECTOR_YAMNET_PATH", "").strip()
    candidates: list[Path] = [Path(env)] if env else []
    candidates.extend(_default_model_candidates("yamnet.onnx"))
    for c in candidates:
        if c.exists():
            return c
    raise AudioEventsNotAvailable(
        "yamnet.onnx not found — run scripts/fetch_yamnet.py or set DIRECTOR_YAMNET_PATH"
    )


class YamnetOnnxBackend:
    """The real adapter: 16 kHz mono waveform in, 521-class frame scores out."""

    name = "yamnet-onnx"

    def __init__(self, model_path: str | Path | None = None):
        resolved = resolve_yamnet_model_path(model_path)
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - exercised only without dep
            raise AudioEventsNotAvailable("onnxruntime is not installed (uv sync)") from e
        self._model_path = resolved
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        self._session = ort.InferenceSession(
            str(resolved), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect_events(
        self,
        audio_path: Path | str,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> AudioEventSweep:
        wave = decode_audio_pcm16k(audio_path)
        duration_s = len(wave) / SAMPLE_RATE
        if len(wave) < SAMPLE_RATE:  # YAMNet needs >= 1 window (0.96 s)
            return AudioEventSweep(
                backend=self.name, duration_s=round(duration_s, 3), threshold=threshold, events=[]
            )
        scores = self._scores_chunked(wave)
        events = build_event_spans(np.asarray(scores), threshold=threshold)
        return AudioEventSweep(
            backend=self.name,
            duration_s=round(duration_s, 3),
            threshold=threshold,
            events=events,
        )

    def _scores_chunked(self, wave: np.ndarray) -> np.ndarray:
        """Full-waveform scores in 64-window steps with a 2-window overlap.

        The graph emits one frame per 0.48 s hop (window j covers
        [j*hop, j*hop+window]), so a chunk starting on a whole number of
        windows re-emits exactly `c0/hop` already-seen leading frames — drop
        them and the concatenated scores equal the single giant run, which
        the memory cgroup cannot survive on a 43-minute tape.
        """
        window_samples = round(FRAME_WINDOW_S * SAMPLE_RATE)
        hop_samples = round(FRAME_HOP_S * SAMPLE_RATE)
        chunk_samples = CHUNK_WINDOWS * window_samples
        if len(wave) <= max(chunk_samples, 4 * window_samples):
            scores, _e, _m = self._session.run(None, {self._input_name: wave})
            return np.asarray(scores)

        step = (CHUNK_WINDOWS - OVERLAP_WINDOWS) * window_samples
        out_frames: list[np.ndarray] = []
        emitted_end = 0  # global frame index one past the last frame already kept
        s0 = 0
        while True:
            if s0 > 0 and len(wave) - s0 < chunk_samples:
                # the runt tail sees edge padding — recompute the end as a FULL
                # chunk aligned to the tape end (real samples to the last frame);
                # the already-emitted leading frames are dropped below
                s0 = len(wave) - chunk_samples
            seg = wave[s0 : s0 + chunk_samples]
            scores, _e, _m = self._session.run(None, {self._input_name: np.ascontiguousarray(seg)})
            first_global = round(s0 / hop_samples)  # this chunk's frame 0 in wave time
            out_frames.append(scores[max(0, emitted_end - first_global) :])
            emitted_end = first_global + len(scores)
            if s0 + chunk_samples >= len(wave):
                break
            s0 += step
        return np.concatenate(out_frames, axis=0)


def make_audio_event_backend() -> AudioEventBackend:
    """YAMNet adapter; raises AudioEventsNotAvailable with the fetch hint when unrunnable."""
    return YamnetOnnxBackend()
