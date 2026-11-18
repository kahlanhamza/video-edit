#!/usr/bin/env python3
"""Bus-1 compressor tuning sweep — fix the "punchy wall of voice" artifact.

Measured context (Nov 2026, Eustis MossFormer2 harvest):
  - enhanced speech RMS -25.1 dBFS, peak -6.0 dBFS -> crest 19 dB
  - the 3:1 @ -18 comp barely grabs (crest -> 17.9) so the master limiter
    eats ~10 dB of crest to reach the loudness target -> every syllable
    sliced = "unnaturally punchy" (user report + issue)
  - the loudness law (speech RMS in -12..-6, target now -11) + the -1.5 dBTP
    ceiling require post-comp crest <= ~9.5 dB; the sweep finds comp params
    that deliver that with the SMALLEST limiter GR (most natural dynamics)

Applies EQ + candidate comp to enhanced.wav, measures speech RMS/peak/crest
on the VAD gate spans, then reports: gain needed to land -11, resulting
limiter GR, crest in/out. Best candidate: limiter GR <= 3 dB, crest >= 6.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "audio_engine" / "src"))

from director_os.audio_engine.leveling import measure_speech_levels  # noqa: E402
from director_os.audio_engine.two_bus import SAMPLE_RATE  # noqa: E402
from director_os.audio_engine.vad import SileroVad, speech_gate_intervals  # noqa: E402

WORK = ROOT / "output" / "voice_fix"
ENHANCED = WORK / "enhanced.wav"
TARGET_DBFS = -11.0
CEILING_LINEAR = 0.8414  # -1.5 dBTP

CANDIDATES = [
    # name, threshold_db, ratio, attack_ms, release_ms
    ("U_user_2_1_12", -12.0, 2.0, 15, 150),
    ("A_3_1_24", -24.0, 3.0, 12, 150),
    ("B_3_1_26", -26.0, 3.0, 15, 180),
    ("C_4_1_22", -22.0, 4.0, 10, 120),
    ("D_4_1_26", -26.0, 4.0, 12, 150),
    ("E_6_1_24", -24.0, 6.0, 12, 150),
]


def eq_chain() -> str:
    from director_os.audio_engine.two_bus import bus1_tone_chain

    return bus1_tone_chain()


def main() -> int:
    pcm16 = None
    if (WORK / "preclean.wav").exists():
        from director_os.audio_engine.leveling import decode_pcm_mono

        pcm16 = decode_pcm_mono(str(WORK / "preclean.wav"), sr=16_000)
    gate = speech_gate_intervals([], SileroVad().speech_regions(pcm16)) if pcm16 is not None else []
    print(f"gate spans: {len(gate)}, speech {sum(e - s for s, e in gate):.1f}s")

    base = measure_speech_levels(str(ENHANCED), gate)
    print(
        f"enhanced: rms {base['speech_rms_dbfs']:+.2f} peak {base['speech_peak_dbfs']:+.2f} "
        f"crest {base['speech_peak_dbfs'] - base['speech_rms_dbfs']:.1f} dB"
    )
    print(f"{'candidate':16s} {'rms':>7s} {'peak':>7s} {'crest':>6s} {'gain':>6s} {'limGR':>6s}")
    for name, th, ratio, att, rel in CANDIDATES:
        af = (
            f"{eq_chain()},acompressor=threshold={th:g}dB:ratio={ratio:g}:"
            f"attack={att}:release={rel},aresample={SAMPLE_RATE},"
            "aformat=sample_fmts=s16:channel_layouts=stereo"
        )
        out = WORK / f"sweep_{name}.wav"
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(ENHANCED),
                "-af",
                af,
                "-c:a",
                "pcm_s16le",
                str(out),
            ],
            capture_output=True,
        )
        if r.returncode != 0:
            print(f"{name:16s} FAILED {r.stderr[-200:].decode(errors='replace')}")
            continue
        m = measure_speech_levels(str(out), gate)
        crest = m["speech_peak_dbfs"] - m["speech_rms_dbfs"]
        gain = TARGET_DBFS - m["speech_rms_dbfs"]
        peak_out = m["speech_peak_dbfs"] + gain
        lim_gr = max(0.0, peak_out - (-1.5))
        print(
            f"{name:16s} {m['speech_rms_dbfs']:+7.2f} {m['speech_peak_dbfs']:+7.2f} "
            f"{crest:6.1f} {gain:+6.1f} {lim_gr:6.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
