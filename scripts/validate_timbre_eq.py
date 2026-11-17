"""Validate the Bus-1 timbre-restoration EQ on the real MossFormer2 harvest.

Applies the production `bus1_tone_chain()` (via ffmpeg, exactly as the worker
does) to the real enhanced_48k_mossformer2.wav, then prints the tilt numbers
raw -> enhanced -> EQ-restored. The EQ must move presence/air tilts toward
zero (brighter voice) without re-inflating noise between words.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "audio_engine" / "src"))

from director_os.audio_engine.two_bus import bus1_tone_chain  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_tilt import SR, band_rms_db, decode  # noqa: E402


def tilts(x) -> str:
    low = band_rms_db(x, 120, 250)
    mid = band_rms_db(x, 500, 2000)
    pres = band_rms_db(x, 2000, 5000)
    air = band_rms_db(x, 5000, 12000)
    return f"lowmid={low - mid:+.1f} presence={pres - mid:+.1f} air={air - mid:+.1f} (vs mid)"


def main() -> None:
    enh_path = Path(sys.argv[1])
    out = enh_path.parent / "enhanced_timbre_restored.wav"
    af = f"{bus1_tone_chain()},aresample={SR},aformat=sample_fmts=s16:channel_layouts=stereo"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(enh_path),
            "-af",
            af,
            "-c:a",
            "pcm_s16le",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    enh, restored = decode(str(enh_path)), decode(str(out))
    print(f"BEFORE EQ: {tilts(enh)}")
    print(f"AFTER  EQ: {tilts(restored)}")
    # noise-safety: between-words proxy = overall band levels must stay sane
    probes = (
        ("presence 2-5k", 2000, 5000),
        ("edge 5-8k", 5000, 8000),
        ("air 8-12k", 8000, 12000),
    )
    for name, lo, hi in probes:
        e, r = band_rms_db(enh, lo, hi), band_rms_db(restored, lo, hi)
        print(f"{name}: {e:.1f} -> {r:.1f} dB ({r - e:+.1f})")


if __name__ == "__main__":
    main()
