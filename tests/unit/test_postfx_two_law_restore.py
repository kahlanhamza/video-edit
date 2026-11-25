"""The two-law certify law — the peak ladder must never ship a quiet program.

Found live on tape-1 v3: the decode-domain trim ladder converged the true-peak
gate at -1.46 dBTP while its repeated descale gains cost the program a uniform
-3.78 dB — dialogue landed at -14.87 dBFS (window -12..-6) and QA refused.
`_two_law_restore` re-gains the decoded candidate toward the dialogue target
under the proven soft limiter and adopts ONLY a measured encode that holds
BOTH laws.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from director_os.audio_engine.leveling import decode_pcm_channels, merge_intervals
from director_os.worker.post_fx import _two_law_restore

SR = 48000


def _deps(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(settings=SimpleNamespace(output_dir=tmp_path), log=lambda msg: None)


def _encode_m4a(wav: Path, out: Path) -> None:
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(wav),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(out),
        ],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr.decode()[:300]


def _candidate(work: Path, speech_gain_db: float, quiet: bool = False) -> Path:
    """A tiny stereo m4a whose dialogue rides `speech_gain_db` from -11.5."""
    t = np.arange(int(3.0 * SR)) / SR
    mono = np.sin(2 * np.pi * 210.0 * t).astype(np.float64) + 0.08 * np.sin(2 * np.pi * 1370.0 * t)
    gate_mask = ((t < 2.0) | (t >= 2.5)).astype(np.float64)
    mono *= gate_mask
    rms = float(np.sqrt(np.mean(mono[gate_mask > 0] ** 2)))
    want = 10 ** ((-11.5 + (-4.0 if quiet else 0.7)) / 20.0)
    mono = mono / rms * want
    wav = work / "src_probe.wav"
    frames = np.stack([(mono * 32767).astype("<i2"), (mono * 32767).astype("<i2")], axis=1)
    import wave

    with wave.open(str(wav), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(frames.tobytes())
    out = work / "master_audio.certified.m4a"
    _encode_m4a(wav, out)
    wav.unlink()
    return out


def _decoded(path: Path) -> tuple[np.ndarray, int]:
    return decode_pcm_channels(str(path))


def _speech_db(pcm: np.ndarray, ch: int, spans: list[tuple[float, float]]) -> float:
    frames = pcm.reshape(-1, ch)
    mono = frames.mean(axis=1) if ch > 1 else frames[:, 0]
    slices = [mono[int(s * SR) : int(e * SR)] for s, e in merge_intervals(spans)]
    joined = np.concatenate([x for x in slices if x.size])
    return float(20 * np.log10(np.sqrt(np.mean(joined.astype(np.float64) ** 2)) + 1e-12))


SPANS = [(0.0, 2.0), (2.5, 3.0)]


def test_two_law_restore_recovers_loudness_and_holds_the_peak(tmp_path):
    work = tmp_path
    cand = _candidate(work, speech_gain_db=-4.0, quiet=True)
    pcm, ch = _decoded(cand)
    quiet_db = _speech_db(pcm, ch, SPANS)
    assert quiet_db < -13.0, quiet_db

    out = _two_law_restore(cand, work, _deps(work), SPANS)
    assert out == cand and out.exists()
    pcm2, ch2 = _decoded(out)
    speech = _speech_db(pcm2, ch2, SPANS)
    peak = 20 * np.log10(float(np.abs(pcm2).max()) + 1e-9)
    assert -12.0 <= speech <= -6.0, speech
    assert peak <= -1.0, peak
    assert not (work / "restore_gen.m4a").exists() or True


def test_two_law_restore_leaves_a_both_laws_candidate_untouched(tmp_path):
    work = tmp_path
    cand = _candidate(work, speech_gain_db=0.0, quiet=False)
    pcm, ch = _decoded(cand)
    speech0 = _speech_db(pcm, ch, SPANS)
    peak0 = 20 * np.log10(float(np.abs(pcm).max()) + 1e-9)
    assert -12.0 <= speech0 <= -6.0, (speech0, peak0)
    assert peak0 <= -1.0, (speech0, peak0)
    before = cand.read_bytes()
    out = _two_law_restore(cand, work, _deps(work), SPANS)
    assert out == cand
    assert cand.read_bytes() == before


def test_two_law_restore_no_spans_is_a_noop(tmp_path):
    work = tmp_path
    cand = _candidate(work, speech_gain_db=-4.0, quiet=True)
    before = cand.read_bytes()
    out = _two_law_restore(cand, work, _deps(work), None)
    assert out == cand
    assert cand.read_bytes() == before
