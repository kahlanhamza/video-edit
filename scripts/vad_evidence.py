#!/usr/bin/env python3
"""VAD evidence engine for caption verification (postfx v4).

Answers two owner questions with MEASURED audio truth on the master clock:
  1. Does the audio actually contain voiced speech where ASR emitted words?
     (no speech + words  ->  ASR hallucination  ->  *unaudible* tag)
  2. Does the audio contain speech where NO caption covers it?
     (speech + no cue    ->  missing caption    ->  insert *unaudible* cue)

Method: webrtcvad (aggressive) on 30 ms frames @16 kHz mono, plus a
voice-band energy floor so music/wind is not mistaken for speech.
The speech_ratio of an utterance span is the fraction of VAD-positive
30 ms frames inside [start, end].
"""

import json
import struct
import wave
from pathlib import Path

import numpy as np
import webrtcvad

WORK = Path("/home/z/my-project/pipeline_run")
V3 = WORK / "v3"
SR = 16000
FRAME_MS = 30


def load_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        n = w.getnframes()
        raw = w.readframes(n)
    return np.array(struct.unpack(f"<{n}h", raw), dtype=np.int16)


def vad_flags(pcm: np.ndarray, aggressiveness: int = 3) -> np.ndarray:
    """Per-30ms-frame boolean speech flags + frame-level RMS (dBFS)."""
    vad = webrtcvad.Vad(aggressiveness)
    frame = SR * FRAME_MS // 1000  # 480 samples
    n_fr = len(pcm) // frame
    flags = np.zeros(n_fr, dtype=bool)
    rms = np.zeros(n_fr)
    for i in range(n_fr):
        seg = pcm[i * frame : (i + 1) * frame]
        rms[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        try:
            flags[i] = vad.is_speech(seg.tobytes(), SR)
        except Exception:
            flags[i] = False
    return flags, rms, frame


def speech_ratio(flags: np.ndarray, frame: int, s: float, e: float) -> float:
    a, b = int(s * SR / frame), max(int(e * SR / frame), int(s * SR / frame) + 1)
    seg = flags[a:b]
    return float(seg.mean()) if len(seg) else 0.0


def speech_spans(
    flags: np.ndarray, frame: int, min_s: float = 0.45, join_gap: float = 0.30
) -> list[tuple[float, float]]:
    """Contiguous VAD-positive spans >= min_s seconds, gaps < join_gap joined."""
    idx = np.flatnonzero(flags)
    if not len(idx):
        return []
    sps = []
    st = prev = idx[0]
    for i in idx[1:]:
        if i - prev <= 1 + join_gap * SR / frame:
            prev = i
        else:
            sps.append((st, prev))
            st = prev = i
    sps.append((st, prev))
    return [
        (a * frame / SR, (b + 1) * frame / SR)
        for a, b in sps
        if ((b - a + 1) * frame) / SR >= min_s
    ]


def voice_band_ratio(pcm: np.ndarray, s: float, e: float) -> float:
    """Energy fraction in the 250-3400 Hz voice band (speech vs music/wind)."""
    a, b = int(s * SR), int(e * SR)
    seg = pcm[a:b].astype(np.float64) / 32768.0
    if not len(seg):
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)
    tot = float(spec.sum()) or 1e-9
    band = float(spec[(freqs >= 250) & (freqs <= 3400)].sum())
    return band / tot


def segment_rms_db(pcm: np.ndarray, s: float, e: float) -> float:
    a, b = int(s * SR), int(e * SR)
    seg = pcm[a:b].astype(np.float64)
    if not len(seg):
        return -99.0
    return 20 * np.log10(max(float(np.sqrt(np.mean(seg**2))), 1e-6) / 32768)


if __name__ == "__main__":
    flags, rms, frame = vad_flags(load_pcm(V3 / "audio_src_16k.wav"))
    np.save(V3 / "vad_flags.npy", flags)
    np.save(V3 / "vad_rms.npy", rms)
    with open(V3 / "vad_meta.json", "w") as fh:
        json.dump({"frame_s": frame / SR, "sr": SR}, fh)
    # validate on the hallucination window + known speech
    for lo, hi, what in [
        (606.8, 614.0, "suspect reply zone"),
        (609.4, 628.9, "HALLUCINATION loop"),
        (300.0, 310.0, "mid-video control"),
        (630.0, 650.0, "outro narration"),
    ]:
        r = speech_ratio(flags, frame, lo, hi)
        print(f"{what:22s} {lo:7.1f}-{hi:7.1f}  speech_ratio={r:.3f}")
    sp = speech_spans(flags, frame)
    print(
        f"total speech spans: {len(sp)}, covered {sum(b - a for a, b in sp):.1f}s of {657.7:.0f}s"
    )
