"""Master-clock QA speech intervals — the audio gates ride the master clock.

The failure this locks out (found live on tape-1 v2, 2026-11-25): stage_qa fed
SOURCE-clock speech intervals into measurements of the SHIPPED master — a
reordered cut. The spans landed on narration b-roll beds and wrong moments, so
a green enhance stage "failed" QA with a false speech RMS -22 dBFS and a false
clean_snr 9.0 dB. Law: the v6 captions already remap raw speech onto the master
clock (master_cues.py); the acoustic gates measure the same spans the same way.
QA still evaluates the OUTPUT FILE itself (M5-R1) — only the spans change clocks.
"""

from __future__ import annotations

import wave

import numpy as np
from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)
from director_os.worker.pipeline import master_clock_speech_intervals


def _timeline():
    """raw1 (source 100-110s) @ master 0-10, narration @ 10-18, raw2 (300-306) @ 18-24."""
    spec = TimelineSpec(
        name="qa-clock",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw1",
                        source_start_s=100.0,
                        dur_s=10.0,
                        meta=RawClipMeta(asset_id="a"),
                    ),
                    ClipSpec(
                        name="narr1",
                        source_start_s=500.0,
                        dur_s=8.0,
                        meta=NarratorClipMeta(script_text="the narrator explains the case here"),
                    ),
                    ClipSpec(
                        name="raw2",
                        source_start_s=300.0,
                        dur_s=6.0,
                        meta=RawClipMeta(asset_id="a"),
                    ),
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=24.0)]),
        ],
    )
    return build_timeline(spec)


SOURCE_INTERVALS = [
    (102.0, 104.0),  # inside raw1 -> master 2-4
    (301.0, 302.0),  # inside raw2 -> master 19-20
    (502.0, 503.0),  # INSIDE the b-roll source window -> must drop (no raw audio)
    (900.0, 902.0),  # beyond every kept window -> must drop (cut material)
]


def test_remap_moves_source_spans_onto_master_clock():
    out = master_clock_speech_intervals(_timeline(), SOURCE_INTERVALS)
    assert out == [(2.0, 4.0), (19.0, 20.0)]


def test_remap_partial_overlap_keeps_only_the_kept_part():
    # speech straddling a raw clip's edge survives clipped to the window
    out = master_clock_speech_intervals(_timeline(), [(99.0, 101.5)])
    assert out == [(0.0, 1.5)]


def test_remap_no_master_track_returns_source_clock():
    spec = TimelineSpec(
        name="no-master",
        fps=30,
        width=640,
        height=360,
        tracks=[TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=24.0)])],
    )
    out = master_clock_speech_intervals(build_timeline(spec), SOURCE_INTERVALS)
    assert out == sorted(SOURCE_INTERVALS)


def _write_wav(path, pcm, sr=48_000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())


def test_remapped_spans_measure_real_master_audio(tmp_path):
    """The end-to-end lie, at fixture scale: speech only exists at MASTER 2-4s
    and 19-20s. Source-clock spans measure garbage; remapped spans measure the
    actual dialogue."""
    from director_os.audio_engine.leveling import measure_speech_levels

    sr = 48_000
    dur_s = 24.0
    pcm = np.zeros(int(dur_s * sr), dtype=np.float32)
    for a, b in [(2.0, 4.0), (19.0, 20.0)]:
        t = np.arange(int(a * sr), int(b * sr))
        pcm[int(a * sr) : int(b * sr)] = 0.25 * np.sin(2 * np.pi * 440.0 * t / sr)
    wav = tmp_path / "master.wav"
    _write_wav(wav, pcm, sr)

    remapped = master_clock_speech_intervals(_timeline(), SOURCE_INTERVALS)
    good = measure_speech_levels(wav, remapped)
    bad = measure_speech_levels(wav, SOURCE_INTERVALS)

    # the remapped spans land exactly on the synthesized dialogue (merge pads
    # each span 0.15 s per side — the leveling convention: 2x2.3 + 1x1.3 = 3.6)
    assert good["speech_seconds"] == 3.5999791666666665
    assert good["speech_rms_dbfs"] > -20.0  # the real 0.25-amplitude tone
    # the source-clock spans land beyond the 24s master entirely — the lie that
    # failed tape-1 v2's QA on a green enhance stage
    assert bad["speech_rms_dbfs"] < good["speech_rms_dbfs"]


def test_remap_include_narration_blocks_covers_program_speech():
    """The SNR floor must never sample the TTS voice as noise: with
    include_narration_blocks=True each narration block's full master window
    joins the program-speech set (dialogue spans unchanged)."""
    out = master_clock_speech_intervals(
        _timeline(), SOURCE_INTERVALS, include_narration_blocks=True
    )
    assert (2.0, 4.0) in out  # dialogue span kept
    assert (19.0, 20.0) in out  # dialogue span kept
    assert (10.0, 18.0) in out  # narration block now counts as program speech
    assert (502.0, 503.0) not in out  # b-roll SOURCE span still never appears


class _FakeInfo:
    speech_intervals = [(102.0, 104.0)]  # source clock: inside raw1


class _FakeDeps:
    def log(self, msg: str) -> None:
        self.last = msg


def test_leveling_pass_lands_dialogue_on_target(tmp_path):
    """The #15 law on the shipped mix: measured -18-ish dialogue gains to the
    -11 dBFS target and the ceiling brickwall holds the -1.5 dBTP peak."""
    import numpy as np
    from director_os.audio_engine.leveling import (
        LIMITER_LINEAR,
        decode_pcm_mono,
        measure_speech_levels,
    )
    from director_os.worker.post_fx import _leveling_pass

    sr = 48_000
    pcm = np.zeros(int(24.0 * sr), dtype=np.float32)
    # dialogue lives at master 2-4s (raw1 @ source 102-104s), quiet like the
    # found-live failure (-18 dBFS-ish speech RMS on the spoken span)
    t = np.arange(int(2.0 * sr), int(4.0 * sr))
    pcm[int(2.0 * sr) : int(4.0 * sr)] = 0.06 * np.sin(2 * np.pi * 440.0 * t / sr)
    # a loud blip that must get brickwalled after the gain
    pcm[int(5.0 * sr) : int(5.02 * sr)] = 0.95
    enhanced = tmp_path / "enhanced.wav"
    _write_wav(enhanced, pcm, sr)

    work = tmp_path / "work"
    work.mkdir()
    report = _FakeReport()
    deps = _FakeDeps()
    out = _leveling_pass(enhanced, _timeline(), _FakeInfo(), work, report, deps)

    assert report.level_engine == "speech_gated_refined"
    leveled = decode_pcm_mono(out)
    after = measure_speech_levels(out, [(2.0, 4.0)])
    # the one gain moved the spoken span to the -11 target (window guard)
    assert -12.0 <= after["speech_rms_dbfs"] <= -6.0
    # the ceiling law: |out| never exceeds the -1.5 dBTP brickwall
    assert float(np.max(np.abs(leveled))) <= LIMITER_LINEAR + 1e-3


class _FakeReport:
    level_engine = ""
    level_gain_db = 0.0
    level_speech_rms_dbfs = None


def test_audio_duration_coverage_gate_catches_silent_tail(tmp_path):
    """Regression (tape-1 v2, found live): the rendered master's audio ended
    3.85 s before its video and NO gate noticed. The audio stream must cover
    the video stream — a silent tail is a RED, not a surprise."""
    import subprocess

    from director_os.core.qa_probe import audit_technical

    video = tmp_path / "v.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=5:size=160x120:rate=15",
            "-c:v",
            "libx264",
            str(video),
        ],
        check=True,
    )
    short = tmp_path / "short.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            video,
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(short),
        ],
        check=True,
    )
    full = tmp_path / "full.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            video,
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=5",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(full),
        ],
        check=True,
    )
    audit_short = audit_technical(short)
    gate_short = [c for c in audit_short.checks if c.name == "audio_duration_coverage"]
    assert gate_short and not gate_short[0].ok  # the silent tail is RED

    audit_full = audit_technical(full)
    gate_full = [c for c in audit_full.checks if c.name == "audio_duration_coverage"]
    assert gate_full and gate_full[0].ok  # full coverage is GREEN
