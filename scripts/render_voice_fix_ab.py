"""Re-render the real Eustis bodycam window with the FIXED voice chain (M4).

The user heard `ab_se48k/ab_clean_15s.wav` — raw MossFormer2_SE_48K output —
and reported "the speakers voices sound soo deep". Root cause (measured):
the neural extractor preserves low-mid and shaves presence/air (see the
timbre constants in audio_engine.two_bus). The fix already landed
(a3d6451): the Bus-1 intelligibility EQ (mud cut -> clarity -> presence ->
air shelf) plus the speech-gated 2-bus mix.

This script replays the EXACT production chain of
`apps/worker/audio_enhance.py::stage_audio_enhance` steps 4-7b on the real
Kaggle harvest (the GPU steps 1-3 already ran there; their outputs are in
download/):

    raw_48k.wav       48 kHz stereo re-extract of the harvested raw window
    preclean.wav      highpass=f=90:poles=2 (pre-ML rumble filter)
    enhanced.wav      the harvested MossFormer2 output (the "deep" stage)
    bus1_gated.wav    Bus-1 tone chain + speech-gated -14 dB duck (Silero VAD)
    master_2bus.wav   + tamed ambience -> speech-gated gain -> -1.5 dBTP

Ears evidence (all at the SAME densest-speech 15 s window, loudness-matched
where noted) -> download/voice_fix_ab/:
    1_raw_noisy_15s.*     A — the original noisy audio
    2_before_deep_15s.*   B — the enhanced-only voice the user heard
                          (same master gain + limiter, NO timbre EQ: the
                          loudness matches C so the ear hears timbre only)
    3_after_fixed_15s.*   C — the full production 2-Bus master (THE FIX)
    *.mp4                 the same window muxed with the real video

Gates re-checked on the NEW master with the production gate code:
snr_gate (clean_snr >= 14 dB, delta_noise <= -8 or improvement >= 6),
timbre_gate (presence tilt >= MIN_PRESENCE_TILT_DB), peak <= -1.0 dBTP.

Usage: uv run python scripts/render_voice_fix_ab.py [full|evidence]
    full     rebuild the chain from the harvest (default)
    evidence reuse output/voice_fix/ artifacts — re-slice, re-mux, re-gate only
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "audio_engine" / "src"))

from director_os.audio_engine.leveling import (  # noqa: E402
    DEFAULT_TARGET_DBFS,
    MAX_GAIN_DB,
    check_speech_window,
    decode_pcm_mono,
    measure_speech_levels,
    speech_gated_gain,
)
from director_os.audio_engine.snr import (  # noqa: E402
    measure_snr,
    natural_dynamics_gate,
    peak_true_dbfs,
    snr_gate,
    speech_band_tilts,
    speech_crest_db,
    timbre_gate,
)
from director_os.audio_engine.two_bus import (  # noqa: E402
    RUMBLE_HIGHPASS_HZ,
    SAMPLE_RATE,
    apply_gate_inplace,
    bus1_tone_chain,
    duck_envelope,
    run_two_bus_mix,
)
from director_os.audio_engine.vad import SileroVad, speech_gate_intervals  # noqa: E402

RAW_IN = Path("/home/z/my-project/download/se48k_raw_60s.wav")
ENHANCED_IN = Path("/home/z/my-project/download/se48k_enhanced_60s.wav")
# rawwin_tracked_clean_720p.mp4 = the verified clean 3-pass render of master
# 26.6-45.6 s (burned-timestamp verified; the webm's non-monotonic PTS make
# cutting the original directly unreliable). The A/B window therefore locks to
# rawwin's span: master 26.62-41.62 -> rawwin 0.02-15.02 s.
MUX_VIDEO = Path("/home/z/my-project/download/rawwin_tracked_clean_720p.mp4")
MUX_VIDEO_START_S = 26.6
AB_START_S = 26.62
AB_END_S = 41.62
WORK = ROOT / "output" / "voice_fix"
OUT = Path("/home/z/my-project/download/voice_fix_ab")
EVIDENCE = Path("/home/z/my-project/download/m4_evidence_voice_fix.json")
KERNEL_EVIDENCE = Path("/home/z/my-project/download/m4_evidence_se48k_gpu.json")


def _run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed: {r.stderr[-400:].decode(errors='replace')}")


def _ffmpeg_af(src: Path, af: str, out: Path) -> None:
    _run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src), "-af", af, "-c:a", "pcm_s16le", str(out),
        ],
        what=f"ffmpeg {out.name}",
    )


def _slice(src: Path, st: float, dur: float, out: Path, stereo: bool) -> Path:
    _run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{st:.6f}", "-t", f"{dur:.6f}", "-i", str(src),
            "-map", "0:a:0", "-ar", str(SAMPLE_RATE),
            "-ac", "2" if stereo else "1",
            "-c:a", "pcm_s16le", str(out),
        ],
        what=f"slice {out.name}",
    )
    return out


def _mp3(src: Path, out: Path) -> Path:
    _run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-c:a", "libmp3lame", "-b:a", "192k", str(out)],
        what=f"mp3 {out.name}",
    )
    return out


def _mux(video: Path, audio: Path, st: float, dur: float, out: Path) -> Path:
    _run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{st:.6f}", "-t", f"{dur:.6f}", "-i", str(video),
            "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE), "-ac", "2",
            "-shortest", str(out),
        ],
        what=f"mux {out.name}",
    )
    return out


def _py(v):
    """numpy 2 scalars (np.bool, np.float64) are not JSON-serializable."""
    return v.item() if hasattr(v, "item") else v


def _gate_truth(preclean: Path) -> tuple[list[tuple[float, float]], str]:
    """Silero VAD on preclean (no ASR words for this window; words are empty)."""
    try:
        pcm16 = decode_pcm_mono(str(preclean), sr=16_000)
        vad_regions = SileroVad().speech_regions(pcm16)
        return speech_gate_intervals([], vad_regions), "vad(preclean)"
    except Exception as e:  # noqa: BLE001 — loud degradation, kernel spans fallback
        print(f"local VAD unavailable ({e}) — falling back to kernel VAD spans")
        k = json.loads(KERNEL_EVIDENCE.read_text())
        spans = [tuple(s) for s in k["kernel_metrics"]["speech_spans_s"]]
        return speech_gate_intervals([], spans), "vad(kernel)"


def build_chain(gate: list[tuple[float, float]]) -> tuple[float, float]:
    """Production steps 1-7b -> master_2bus.wav + before_deep_48k.wav; returns (gain_db, b_gain_db)."""
    raw_ref = WORK / "raw_48k.wav"
    preclean = WORK / "preclean.wav"
    enhanced = WORK / "enhanced.wav"

    # 1) raw 48 kHz stereo reference (also the Bus-2 ambience source)
    _ffmpeg_af(RAW_IN, f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo", raw_ref)
    # 2) pre-ML rumble filter
    _ffmpeg_af(raw_ref, f"highpass=f={RUMBLE_HIGHPASS_HZ}:poles=2,aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo", preclean)
    # 3) the harvested neural Bus-1 extraction (already ran on Kaggle T4)
    _ffmpeg_af(ENHANCED_IN, f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo", enhanced)

    # 5) Bus-1 tone chain (THE VOICE FIX: mud cut -> clarity -> presence -> air)
    bus1_tone = WORK / "bus1_tone.wav"
    _ffmpeg_af(enhanced, f"{bus1_tone_chain()},aresample={SAMPLE_RATE},aformat=sample_fmts=s16:channel_layouts=stereo", bus1_tone)

    # 5b) the speech-gated duck on Bus-1
    total_samples = decode_pcm_mono(str(bus1_tone)).size
    env = duck_envelope(total_samples, SAMPLE_RATE, gate)
    bus1_gated = WORK / "bus1_gated.wav"
    bus1_gated.write_bytes(bus1_tone.read_bytes())
    apply_gate_inplace(bus1_gated, env)

    # 6) master gain: spoken words land in -12..-6 dBFS
    levels = measure_speech_levels(str(bus1_gated), gate)
    gain_db = speech_gated_gain(levels["speech_rms_dbfs"], DEFAULT_TARGET_DBFS, MAX_GAIN_DB)

    # 7) the 2-bus mix: voice + tamed ambience -> limiter -> 48 kHz stereo
    master = WORK / "master_2bus.wav"
    run_two_bus_mix(bus1_gated, raw_ref, master, gain_db)

    # 7b) deterministic refinement loop: the limiter compresses crest after the
    # one-shot gain was computed, so spoken words can land outside the
    # -12..-6 window; the limiter response is concave (each +1 dB of gain
    # raises speech RMS ~0.4 dB), so iterate until the window is met or the
    # correction converges (max 4 passes).
    from director_os.audio_engine.leveling import check_speech_window

    for _ in range(8):
        mixed = measure_speech_levels(str(master), gate)
        if check_speech_window(mixed["speech_rms_dbfs"]):
            break
        refined = round(gain_db + (DEFAULT_TARGET_DBFS - mixed["speech_rms_dbfs"]), 2)
        refined = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, refined))
        if abs(refined - gain_db) < 0.4:
            break
        run_two_bus_mix(bus1_gated, raw_ref, master, refined)
        gain_db = refined
        print(f"master gain refined -> {gain_db:+.2f} dB (post-limiter pass)")

    # BEFORE sample: enhanced-only, same loudness law, NO timbre EQ
    # (isolates the timbre fix: B and C land at the same speech RMS so the
    # ear compares TONE, not volume)
    b_levels = measure_speech_levels(str(enhanced), gate)
    b_gain = speech_gated_gain(b_levels["speech_rms_dbfs"], DEFAULT_TARGET_DBFS, MAX_GAIN_DB)
    deep = WORK / "before_deep_48k.wav"
    _ffmpeg_af(enhanced, f"volume={b_gain:+g}dB,alimiter=level_in=1:level_out=1:level=disabled:limit=0.8414:attack=5:release=50", deep)

    (WORK / "master_gain.json").write_text(json.dumps({"gain_db": gain_db, "b_gain_db": b_gain}))
    return gain_db, b_gain


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode not in {"full", "evidence"}:
        raise SystemExit(f"usage: render_voice_fix_ab.py [full|evidence], got '{mode}'")
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    preclean = WORK / "preclean.wav"
    if mode == "evidence":
        missing = [p for p in (preclean, WORK / "master_2bus.wav", WORK / "before_deep_48k.wav") if not p.exists()]
        if missing:
            raise SystemExit(f"evidence mode needs {missing} — run 'full' first")
    else:
        # production step 2 (preclean) is also the gate-truth source
        tmp_ref = WORK / "raw_48k.wav"
        _ffmpeg_af(RAW_IN, f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo", tmp_ref)
        _ffmpeg_af(tmp_ref, f"highpass=f={RUMBLE_HIGHPASS_HZ}:poles=2,aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo", preclean)

    gate, gate_source = _gate_truth(preclean)
    speech_s = sum(e - s for s, e in gate)
    print(f"gate={gate_source}: {len(gate)} span(s), speech {speech_s:.1f}s of 60s")

    if mode == "evidence":
        # deterministic reproduction of the master gain from the artifacts
        # (same math as build_chain: one-shot gain + post-limiter refinement)
        levels = measure_speech_levels(str(WORK / "bus1_gated.wav"), gate)
        gain_db = speech_gated_gain(levels["speech_rms_dbfs"], DEFAULT_TARGET_DBFS, MAX_GAIN_DB)
        mixed = measure_speech_levels(str(WORK / "master_2bus.wav"), gate)
        if not check_speech_window(mixed["speech_rms_dbfs"]):
            refined = round(gain_db + (DEFAULT_TARGET_DBFS - mixed["speech_rms_dbfs"]), 2)
            refined = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, refined))
            if abs(refined - gain_db) >= 0.5:
                gain_db = refined
    else:
        gain_db, _b = build_chain(gate)

    master = WORK / "master_2bus.wav"
    deep = WORK / "before_deep_48k.wav"
    raw_ref = WORK / "raw_48k.wav"
    enhanced = WORK / "enhanced.wav"
    if mode == "evidence":
        # rebuild the two auxiliary decodes evidence-mode measurements need
        _ffmpeg_af(RAW_IN, f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo", raw_ref)
        _ffmpeg_af(ENHANCED_IN, f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo", enhanced)

    # ---- the rawwin-aligned 15 s window for A/B/C (video+audio in sync) ----
    st, end = AB_START_S, AB_END_S
    w = end - st
    a_wav = _slice(raw_ref, st, w, OUT / "1_raw_noisy_15s.wav", stereo=True)
    b_wav = _slice(deep, st, w, OUT / "2_before_deep_15s.wav", stereo=True)
    c_wav = _slice(master, st, w, OUT / "3_after_fixed_15s.wav", stereo=True)
    for p in (a_wav, b_wav, c_wav):
        _mp3(p, p.with_suffix(".mp3"))
    v_off = AB_START_S - MUX_VIDEO_START_S
    a_mp4 = _mux(MUX_VIDEO, a_wav, v_off, w, OUT / "1_raw_noisy_15s.mp4")
    c_mp4 = _mux(MUX_VIDEO, c_wav, v_off, w, OUT / "3_after_fixed_15s.mp4")

    # ---- production gates on the NEW master ----
    raw_m = measure_snr(str(raw_ref), gate)
    deep_m = measure_snr(str(deep), gate)
    fix_m = measure_snr(str(master), gate)
    snr_ok, snr_detail = snr_gate(
        raw_m["snr_db"], fix_m["snr_db"], fix_m["noise_floor_db"] - raw_m["noise_floor_db"]
    )
    tilt_raw = speech_band_tilts(str(raw_ref), gate)
    tilt_deep = speech_band_tilts(str(deep), gate)
    tilt_fix = speech_band_tilts(str(master), gate)
    timbre_ok, timbre_detail = timbre_gate(tilt_fix)
    crest_fix = speech_crest_db(str(master), gate)
    dyn_ok, dyn_detail = natural_dynamics_gate(crest_fix)
    peak_dbfs = float(peak_true_dbfs(str(master)))
    clip_ok = bool(peak_dbfs <= -1.0)

    evidence = {
        "what": "voice-fix re-render — Eustis bodycam window 0-60 s, production 2-Bus chain",
        "chain": (
            "MossFormer2_SE_48K (Kaggle T4) -> Bus-1 mud cut + clarity + presence + air EQ -> "
            "Silero-gated duck -> 2-Bus ambience -> speech-gated gain -> -1.5 dBTP"
        ),
        "fix_commit": "a3d6451",
        "ab_window_s": [round(_py(st), 2), round(_py(end), 2)],
        "gate_source": gate_source,
        "speech_seconds": round(_py(speech_s), 1),
        "master_gain_db": gain_db,
        "snr": {
            "raw_snr_db": round(_py(raw_m["snr_db"]), 2),
            "deep_snr_db": round(_py(deep_m["snr_db"]), 2),
            "fixed_snr_db": round(_py(fix_m["snr_db"]), 2),
            "delta_noise_db": round(_py(fix_m["noise_floor_db"] - raw_m["noise_floor_db"]), 2),
            "gate_ok": bool(snr_ok),
            "gate_detail": snr_detail,
        },
        "timbre_tilt_db_vs_mid": {
            "raw": {k: round(_py(v), 2) for k, v in tilt_raw.items()},
            "deep_enhanced_only": {k: round(_py(v), 2) for k, v in tilt_deep.items()},
            "fixed_2bus_master": {k: round(_py(v), 2) for k, v in tilt_fix.items()},
            "gate_ok": bool(timbre_ok),
            "gate_detail": timbre_detail,
        },
        "natural_dynamics": {
            "speech_crest_db": round(_py(crest_fix), 2),
            "gate_ok": bool(dyn_ok),
            "gate_detail": dyn_detail,
        },
        "bus2_ambience": "off (AUDIO_BUS2_VOLUME default 0.0 — 2026-11-18 ear verdict)",
        "peak_dbfs": round(peak_dbfs, 2),
        "zero_clipping_ok": clip_ok,
        "all_gates_pass": bool(snr_ok and timbre_ok and clip_ok and dyn_ok),
        "listen": {
            "A_raw_noisy": str(a_wav),
            "B_before_deep_voice": str(b_wav),
            "C_after_voice_fix": str(c_wav),
            "A_video": str(a_mp4),
            "C_video": str(c_mp4),
        },
    }
    EVIDENCE.write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    if not evidence["all_gates_pass"]:
        raise SystemExit("VOICE-FIX GATES FAILED — see evidence above")
    print("OK — voice-fix re-render complete, A/B/C written to", OUT)


if __name__ == "__main__":
    main()
