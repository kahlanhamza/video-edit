"""Stage: audio_prep — the broadcast 2-Bus Speech Isolation pre-pass (M4).

Per RAW ASSET, content-addressed and kill-safe (atomic tmp->rename everywhere,
idempotent re-runs, sidecar JSON receipts):

    raw_48k.wav      full source audio 48 kHz stereo        (SNR raw reference)
    preclean.wav     highpass=f=90:poles=2                  (pre-ML rumble filter)
    enhanced.wav     NEURAL enhancer (deep-filter --atten-lim-db 28 by default,
                     MossFormer2_SE_48K when installed; optional Demucs vocals
                     stem pre-split when music/siren flagged)
    bus1_gated.wav   EQ + compressor + speech-gated -14 dB duck (Silero VAD +
                     ASR words, 150 ms ramps)
    master_2bus.wav  + tamed ambience bus 2 (-24 dB, 150 Hz-4 kHz) -> master
                     gain -> -1.5 dBTP limiter, 48 kHz stereo

THE LAW (`audio.enhance_speech`, mirrored into agent_tool_logs):
- the exact `enhancer_engine` is logged and persisted in the sidecar;
- a missing neural denoiser is NEVER silently swallowed — `resolve_enhancer`
  raises, and afftdn is reachable only by explicit config, where the QA gate
  (`enhancer_engine_used`) fails production runs outside hermetic tests;
- quantitative SNR (raw vs clean on the same ASR word intervals) rides in the
  receipt and `qa.audit_technical` enforces it on the final master.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from director_os.audio_engine.enhance import (
    EnhancerResult,
    SpeechEnhancer,
    resolve_enhancer,
)
from director_os.audio_engine.leveling import (
    DEFAULT_TARGET_DBFS,
    MAX_GAIN_DB,
    decode_pcm_mono,
    measure_speech_levels,
    speech_gated_gain,
)
from director_os.audio_engine.snr import measure_snr, speech_band_tilts
from director_os.audio_engine.two_bus import (
    RUMBLE_HIGHPASS_HZ,
    SAMPLE_RATE,
    apply_gate_inplace,
    bus1_tone_chain,
    duck_envelope,
    run_two_bus_mix,
)
from director_os.audio_engine.vad import SileroVad, speech_gate_intervals

# v2: Bus-1 timbre-restoration chain (presence + air shelf). Bumping the
# version is MANDATORY on any tone-chain change — the cache key must not let
# a re-render silently reuse a master mixed with the old sound.
ENHANCE_CACHE_VERSION = "v2"
DURATION_DRIFT_TOLERANCE_S = 0.05


class AudioEnhanceError(RuntimeError):
    pass


@dataclass
class EnhanceInfo:
    """Everything the renderer, QA gate and cost dashboard need afterwards."""

    asset_id: str
    master_wav: Path
    raw_reference_wav: Path
    engine: str
    neural: bool
    atten_lim_db: float | None
    enhancer_input: str  # "preclean" | "demucs_vocals"
    gate_source: str  # "words+vad" | "words"
    speech_intervals: list[tuple[float, float]]
    raw_snr_db: float
    clean_snr_db: float
    delta_noise_db: float
    master_gain_db: float
    ducked_seconds: float
    total_seconds: float
    cache_key: str
    # timbre receipt (the deep-voice law); defaulted for old-receipt cache hits
    tilt_presence_raw_db: float = 0.0
    tilt_presence_clean_db: float = 0.0
    tilt_air_clean_db: float = 0.0

    def enhanced_audio_map(self) -> dict[str, Path]:
        return {self.asset_id: self.master_wav}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise AudioEnhanceError(f"{what} failed: {r.stderr[-400:].decode(errors='replace')}")


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def word_intervals_from(transcript: dict) -> list[tuple[float, float]]:
    """ASR word spans (the truth SNR + the gate are measured against)."""
    out = []
    for w in transcript.get("words") or []:
        if w.get("text") and w.get("end_s") is not None and w.get("start_s") is not None:
            out.append((float(w["start_s"]), float(w["end_s"])))
    return out


def resolve_speech_gate(
    transcript: dict,
    preclean_wav: Path,
    deps_log=lambda _m: None,
) -> tuple[list[tuple[float, float]], str]:
    """ASR words + Silero VAD (union). VAD failure degrades LOUDLY to words-only."""
    words = word_intervals_from(transcript)
    vad_regions: list[tuple[float, float]] | None = None
    try:
        pcm_16k = decode_pcm_mono(preclean_wav, sr=16_000)
        vad_regions = SileroVad().speech_regions(pcm_16k)
        deps_log(f"  [audio] silero VAD: {len(vad_regions)} voice span(s)")
        return speech_gate_intervals(words, vad_regions), "words+vad"
    except Exception as e:  # noqa: BLE001 — degrade loudly, never crash the render
        deps_log(f"  [audio] VAD unavailable ({e}) — gating on ASR words only")
        return speech_gate_intervals(words, None), "words"


def enhance_cache_key(
    source: Path,
    enhancer_name: str,
    atten_lim_db: float | None,
    enhancer_input: str,
) -> str:
    h = hashlib.sha256(
        (
            f"{ENHANCE_CACHE_VERSION}|{_sha256_file(source)}|{enhancer_name}|"
            f"{atten_lim_db}|{enhancer_input}|{SAMPLE_RATE}"
        ).encode()
    )
    return h.hexdigest()[:40]


def stage_audio_enhance(
    asset_id: str,
    source: Path,
    transcript: dict,
    project_id: str,
    deps,  # PipelineDeps (duck-typed to avoid a cycle)
    enhancer: SpeechEnhancer | None = None,
) -> EnhanceInfo:
    """Run (or reuse) the per-asset 2-bus speech isolation pre-pass."""
    t0 = time.monotonic()
    src = Path(source)
    enhancer = enhancer or resolve_enhancer()
    atten = getattr(enhancer, "atten_lim_db", None)

    # optional Demucs pre-split: music/siren-flagged assets ride the vocals stem
    stem = deps.vocals_stem_resolver(asset_id) if deps.vocals_stem_resolver else None
    enhancer_input = "demucs_vocals" if stem is not None else "preclean"

    key = enhance_cache_key(src, enhancer.name, atten, enhancer_input)
    cache_dir = deps.settings.output_dir / "audio2bus" / key
    master = cache_dir / "master_2bus.wav"
    sidecar = cache_dir / "enhance_receipt.json"
    raw_ref = cache_dir / "raw_48k.wav"

    if master.exists() and sidecar.exists():
        info = _info_from_receipt(json.loads(sidecar.read_text()))
        deps.log(f"  [audio] 2-bus cache hit ({info.engine}) -> {master.name}")
        return info

    cache_dir.mkdir(parents=True, exist_ok=True)
    deps.log(f"  [audio] 2-bus speech isolation: engine={enhancer.name} input={enhancer_input}")

    # 1) raw 48 kHz stereo reference (also the Bus-2 ambience source)
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vn",
            "-map",
            "0:a:0",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(raw_ref),
        ],
        what="raw 48k audio extract",
    )

    # 2) pre-ML rumble filter — vest rubs / engine thump must not hit the model
    preclean = cache_dir / "preclean.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_ref),
            "-af",
            f"highpass=f={RUMBLE_HIGHPASS_HZ}:poles=2,aresample={SAMPLE_RATE},"
            "aformat=sample_fmts=fltp:channel_layouts=stereo",
            "-c:a",
            "pcm_s16le",
            str(preclean),
        ],
        what="90 Hz rumble pre-clean",
    )

    # 3) neural Bus-1 extraction (loud, explicit — never a silent fallback)
    enhanced = cache_dir / "enhanced.wav"
    result: EnhancerResult = enhancer.enhance(preclean, enhanced)
    if result.duration_drift_s > DURATION_DRIFT_TOLERANCE_S:
        raise AudioEnhanceError(
            f"enhancer {result.engine} changed timeline: {result.duration_in_s:.3f}s -> "
            f"{result.duration_out_s:.3f}s (drift {result.duration_drift_s:.3f}s) — refusing "
            "to mix off-timeline audio"
        )
    deps.log(f"  [audio] enhancer_engine={result.engine} neural={result.neural}")

    # 4) gate truth: ASR words + Silero VAD
    gate, gate_source = resolve_speech_gate(transcript, preclean, deps.log)
    deps.log(f"  [audio] gate={gate_source}: {len(gate)} span(s)")

    # 5) Bus-1 tone chain, then the speech-gated duck
    bus1 = cache_dir / "bus1_gated.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(enhanced),
            "-af",
            f"{bus1_tone_chain()},aresample={SAMPLE_RATE},"
            "aformat=sample_fmts=s16:channel_layouts=stereo",
            "-c:a",
            "pcm_s16le",
            str(bus1),
        ],
        what="bus-1 EQ + compressor",
    )

    total_samples = decode_pcm_mono(bus1).size  # mono frame count @48k
    env = duck_envelope(total_samples, SAMPLE_RATE, gate)
    apply_gate_inplace(bus1, env)

    # 6) master gain: spoken words land in -12..-6 dBFS (center -9), clamped
    levels = measure_speech_levels(bus1, gate)
    gain_db = speech_gated_gain(levels["speech_rms_dbfs"], DEFAULT_TARGET_DBFS, MAX_GAIN_DB)

    # 7) the 2-bus mix: voice + tamed ambience -> limiter -> 48 kHz stereo
    run_two_bus_mix(bus1, raw_ref, master, gain_db)

    # 7b) deterministic refinement: the limiter compresses crest after the
    # one-shot gain was computed, so spoken words can land outside the
    # -12..-6 window. Re-measure the MIXED master and correct once.
    from director_os.audio_engine.leveling import check_speech_window

    mixed = measure_speech_levels(str(master), gate)
    if not check_speech_window(mixed["speech_rms_dbfs"]):
        refined = round(gain_db + (DEFAULT_TARGET_DBFS - mixed["speech_rms_dbfs"]), 2)
        refined = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, refined))
        if abs(refined - gain_db) >= 0.5:
            run_two_bus_mix(bus1, raw_ref, master, refined)
            gain_db = refined
            deps.log(f"  [audio] master gain refined -> {gain_db:+.1f} dB (post-limiter)")

    # 8) quantitative SNR receipt (raw vs clean, same word intervals)
    raw_snr = measure_snr(str(raw_ref), gate)
    clean_snr = measure_snr(str(master), gate)
    delta_noise = clean_snr["noise_floor_db"] - raw_snr["noise_floor_db"]

    # 8b) timbre receipt: the deep-voice law (tilt on word spans, raw vs clean)
    tilt_raw = {"tilt_presence_db": 0.0, "tilt_air_db": 0.0, "tilt_lowmid_db": 0.0}
    tilt_clean = dict(tilt_raw)
    try:
        tilt_raw = speech_band_tilts(str(raw_ref), gate)
        tilt_clean = speech_band_tilts(str(master), gate)
        deps.log(
            f"  [audio] timbre tilt (presence/air vs mid): raw {tilt_raw['tilt_presence_db']:+.1f}/"
            f"{tilt_raw['tilt_air_db']:+.1f} dB -> clean {tilt_clean['tilt_presence_db']:+.1f}/"
            f"{tilt_clean['tilt_air_db']:+.1f} dB"
        )
    except ValueError as e:  # no speech inside the gate: nothing to measure
        deps.log(f"  [audio] timbre tilt not measurable: {e}")

    info = EnhanceInfo(
        asset_id=asset_id,
        master_wav=master,
        raw_reference_wav=raw_ref,
        engine=result.engine,
        neural=result.neural,
        atten_lim_db=result.atten_lim_db,
        enhancer_input=enhancer_input,
        gate_source=gate_source,
        speech_intervals=gate,
        raw_snr_db=raw_snr["snr_db"],
        clean_snr_db=clean_snr["snr_db"],
        delta_noise_db=delta_noise,
        tilt_presence_raw_db=tilt_raw["tilt_presence_db"],
        tilt_presence_clean_db=tilt_clean["tilt_presence_db"],
        tilt_air_clean_db=tilt_clean["tilt_air_db"],
        master_gain_db=gain_db,
        ducked_seconds=_ducked_seconds(gate, clean_snr["duration_s"]),
        total_seconds=clean_snr["duration_s"],
        cache_key=key,
    )
    _atomic_json(sidecar, _receipt(info, t0))
    _persist(deps, project_id, asset_id, info, t0)
    deps.log(
        f"  [audio] 2-bus master ready: snr {info.raw_snr_db:.1f} -> {info.clean_snr_db:.1f} dB "
        f"(noise {info.delta_noise_db:+.1f} dB), master gain {gain_db:+.1f} dB"
    )
    return info


def _ducked_seconds(gate: list[tuple[float, float]], total_s: float) -> float:
    open_s = sum(e - s for s, e in gate)
    return max(0.0, total_s - open_s)


def _receipt(info: EnhanceInfo, t0: float) -> dict:
    payload = asdict(info)
    payload["master_wav"] = str(info.master_wav)
    payload["raw_reference_wav"] = str(info.raw_reference_wav)
    return {
        **payload,
        "speech_intervals": [list(p) for p in info.speech_intervals],
        "duration_ms": int((time.monotonic() - t0) * 1000),
        "afftdn_env": os.getenv("DIRECTOR_OS_ALLOW_AFFTDN", ""),
        "params": {
            "duck_db": -14.0,
            "ramp_s": 0.15,
            "bus2_volume": 0.06,
            "limiter_dbtp": -1.5,
            "sample_rate": SAMPLE_RATE,
        },
    }


def _info_from_receipt(payload: dict) -> EnhanceInfo:
    data = {
        k: v
        for k, v in payload.items()
        if k in {f for f in EnhanceInfo.__dataclass_fields__}  # ignore receipt metadata
    }
    data["speech_intervals"] = [tuple(p) for p in data["speech_intervals"]]
    data["master_wav"] = Path(data["master_wav"])
    data["raw_reference_wav"] = Path(data["raw_reference_wav"])
    return EnhanceInfo(**data)


def _persist(deps, project_id: str, asset_id: str, info: EnhanceInfo, t0: float) -> None:
    """media_assets.r2_enhanced_key + agent_tool_logs — the exact engine on record."""

    from director_os.core.db import AgentToolLogRow, MediaAssetRow

    with deps.sessions() as s:
        asset = s.get(MediaAssetRow, asset_id)
        if asset is not None:
            asset.r2_enhanced_key = str(info.master_wav)
        s.add(
            AgentToolLogRow(
                project_id=project_id,
                agent_name="audio_worker",
                tool_name="audio.enhance_speech",
                args_json={
                    "asset_id": asset_id,
                    "enhancer_input": info.enhancer_input,
                    "cache_key": info.cache_key,
                },
                result_summary=(
                    f"enhancer_engine={info.engine} neural={info.neural} "
                    f"atten_lim_db={info.atten_lim_db} gate={info.gate_source} "
                    f"snr_raw={info.raw_snr_db:.1f} snr_clean={info.clean_snr_db:.1f} "
                    f"delta_noise={info.delta_noise_db:+.1f}dB "
                    f"tilt_presence={info.tilt_presence_clean_db:+.1f}dB"
                ),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        )
        s.commit()


def export_enhance_ab(deps, project_id: str, info: EnhanceInfo, video_path: Path | None) -> dict:
    """15 s raw-vs-clean A/B pair next to the pipeline artifacts (ear evidence)."""
    from director_os.audio_engine.ab_export import export_ab_pair

    out_dir = deps.settings.output_dir / "pipeline" / project_id
    ab = export_ab_pair(
        info.raw_reference_wav,
        info.master_wav,
        out_dir,
        speech_intervals=info.speech_intervals,
        video_path=video_path,
    )
    deps.log(
        f"  [audio] A/B export: {ab.raw_path.name} vs {ab.clean_path.name} "
        f"(window {ab.window_start_s:.1f}s-{ab.window_end_s:.1f}s)"
    )
    export_payload = {
        "raw": str(ab.raw_path),
        "clean": str(ab.clean_path),
        "window": [ab.window_start_s, ab.window_end_s],
    }
    if ab.raw_video_path:
        export_payload["raw_video"] = str(ab.raw_video_path)
        export_payload["clean_video"] = str(ab.clean_video_path)
    return export_payload
