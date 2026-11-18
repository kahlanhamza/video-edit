"""Stage: audio_prep — the broadcast 2-Bus Speech Isolation pre-pass (M4).

Per RAW ASSET, content-addressed and kill-safe (atomic tmp->rename everywhere,
idempotent re-runs, sidecar JSON receipts):

    raw_48k.wav      full source audio 48 kHz stereo        (SNR raw reference)
    preclean.wav     highpass=f=90:poles=2                  (pre-ML rumble filter)
    enhanced.wav     NEURAL enhancer (deep-filter --atten-lim-db 28 by default,
                     MossFormer2_SE_48K when installed; Demucs vocals stem as
                     the enhancer INPUT when music is flagged — guardrail:
                     copyrighted background songs never reach the mix)
    bus1_gated.wav   EQ + compressor + speech-gated -14 dB duck (Silero VAD +
                     ASR words, 150 ms ramps)
    master_2bus.wav  + tamed ambience bus 2 (opt-in) -> master gain -> -1.5 dBTP
                     limiter, 48 kHz stereo

LAZY KEPT-WINDOW MODE (`kept_windows`): for long FOIA packets the Director
throws away most of the raw footage — running MossFormer2 over the discarded
85% burns GPU hours for audio nobody renders. When kept windows are passed
(post-planning, the master-track clips), the heavy chain runs PER KEPT WINDOW
(each content-addressed by source hash + window bounds) and the full-timeline
master is ASSEMBLED from window masters + digital-silence gaps, aligned to the
source clock — so the renderer's exact slices keep working unchanged, and the
cut-out footage is never processed. Re-plans only pay for NEW windows.

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
from dataclasses import asdict, dataclass, field
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
from director_os.audio_engine.snr import (
    measure_snr,
    speech_band_tilts,
    speech_crest_db,
)
from director_os.audio_engine.two_bus import (
    RUMBLE_HIGHPASS_HZ,
    SAMPLE_RATE,
    apply_gate_inplace,
    bus1_tone_chain,
    duck_envelope,
    run_two_bus_mix,
)
from director_os.audio_engine.vad import SileroVad, speech_gate_intervals

# v3: Bus-2 ambience default OFF (2026-11-18 ear verdict) + master target -11
# dBFS. Bumping the version is MANDATORY on any mix/tone change — the cache
# key must not let a re-render silently reuse a master mixed with the old
# sound.
ENHANCE_CACHE_VERSION = "v3"
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
    # natural-dynamics receipt (the crush law, user issue 2026-11-18)
    speech_crest_db: float = 0.0
    # lazy kept-window receipt (empty = whole-asset enhance, legacy path)
    kept_windows: list[tuple[float, float]] = field(default_factory=list)
    kept_seconds: float = 0.0
    skipped_seconds: float = 0.0
    # copyrighted-music guardrail receipt (YAMNet Music spans the Director
    # must know about; empty = nothing flagged)
    music_flagged_spans: list[tuple[float, float]] = field(default_factory=list)

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
    kept_windows: list[tuple[float, float]] | None = None,
) -> str:
    windows_tag = ""
    if kept_windows:
        canon = json.dumps([[round(a, 3), round(b, 3)] for a, b in kept_windows])
        windows_tag = "|w" + hashlib.sha256(canon.encode()).hexdigest()[:12]
    h = hashlib.sha256(
        (
            f"{ENHANCE_CACHE_VERSION}|{_sha256_file(source)}|{enhancer_name}|"
            f"{atten_lim_db}|{enhancer_input}|{SAMPLE_RATE}{windows_tag}"
        ).encode()
    )
    return h.hexdigest()[:40]


def merge_windows(
    windows: list[tuple[float, float]], gap_s: float = 0.5
) -> list[tuple[float, float]]:
    """Sort + merge overlapping/near-adjacent keep windows (deterministic)."""
    out: list[list[float]] = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in windows):
        if out and a <= out[-1][1] + gap_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(round(a, 3), round(b, 3)) for a, b in out if b > a]


# ------------------------------------------------------------------ one pass
@dataclass
class _PassResult:
    """One full 2-bus chain over one contiguous material range (pass-local time)."""

    master: Path
    raw_ref: Path
    gate: list[tuple[float, float]]
    snr_raw: dict
    snr_clean: dict
    tilt_raw: dict
    tilt_clean: dict
    crest: float
    engine: str
    neural: bool
    atten_lim_db: float | None
    enhancer_input: str
    gain_db: float


def _extract_raw_48k(src: Path, dst: Path, ss: float | None = None, t: float | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.3f}"]
    if t is not None:
        cmd += ["-t", f"{t:.3f}"]
    cmd += [
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
        str(dst),
    ]
    _run(cmd, what="raw 48k audio extract")


def _extract_stem_window(stem: Path, dst: Path, ss: float, t: float) -> None:
    """Demucs stem window slice — same clock as the raw extract (drift-checked)."""
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{ss:.3f}",
            "-t",
            f"{t:.3f}",
            "-i",
            str(stem),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        what="demucs stem window extract",
    )


def _enhance_pass(
    material: Path,
    transcript: dict,
    cache_dir: Path,
    enhancer: SpeechEnhancer,
    log,
    ss: float | None = None,
    t: float | None = None,
    stem_material: Path | None = None,
    enhancer_input: str = "preclean",
    label: str = "",
) -> _PassResult:
    """The full chain over one contiguous range. Time base = pass-local.

    `material` is the source (whole-asset pass) or the raw asset itself when
    slicing windows (the -ss/-t extraction produces the pass-local raw_48k).
    `stem_material` (optional Demucs vocals stem) becomes the ENHANCER input
    for this range — copyrighted music ranges ride the vocals stem before
    MossFormer2, per the Content-ID guardrail.
    """
    tag = f"{label} " if label else ""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) raw 48 kHz stereo reference (also the Bus-2 ambience source)
    raw_ref = cache_dir / "raw_48k.wav"
    _extract_raw_48k(material, raw_ref, ss=ss, t=t)

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

    # 3) enhancer input: the Demucs vocals stem when music is flagged (the
    #    stem IS the timeline — drift-checked below), else the preclean
    enhancer_src = preclean
    if stem_material is not None:
        enhancer_src = cache_dir / "stem_window.wav"
        if ss is None:
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(stem_material),
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s16le",
                    str(enhancer_src),
                ],
                what="demucs stem copy",
            )
        else:
            _extract_stem_window(stem_material, enhancer_src, ss, float(t or 0.0))

    # 4) neural Bus-1 extraction (loud, explicit — never a silent fallback)
    enhanced = cache_dir / "enhanced.wav"
    result: EnhancerResult = enhancer.enhance(enhancer_src, enhanced)
    if result.duration_drift_s > DURATION_DRIFT_TOLERANCE_S:
        raise AudioEnhanceError(
            f"enhancer {result.engine} changed timeline: {result.duration_in_s:.3f}s -> "
            f"{result.duration_out_s:.3f}s (drift {result.duration_drift_s:.3f}s) — refusing "
            "to mix off-timeline audio"
        )
    log(f"  [audio] {tag}enhancer_engine={result.engine} neural={result.neural}")

    # 5) gate truth: ASR words + Silero VAD (pass-local clock)
    gate, gate_source = resolve_speech_gate(transcript, preclean, log)
    log(f"  [audio] {tag}gate={gate_source}: {len(gate)} span(s)")

    # 6) Bus-1 tone chain, then the speech-gated duck
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

    # 7) master gain: spoken words land in -12..-6 dBFS (center -9), clamped
    levels = measure_speech_levels(bus1, gate)
    gain_db = speech_gated_gain(levels["speech_rms_dbfs"], DEFAULT_TARGET_DBFS, MAX_GAIN_DB)

    # 8) the 2-bus mix: voice + tamed ambience -> limiter -> 48 kHz stereo
    master = cache_dir / "master_seg.wav"
    run_two_bus_mix(bus1, raw_ref, master, gain_db)

    # 8b) deterministic refinement loop: the limiter compresses crest after the
    # one-shot gain was computed, so spoken words can land outside the
    # -12..-6 window. The limiter response is concave (each +1 dB of gain
    # raises speech RMS only ~0.4 dB once it engages), so iterate until the
    # window is met or the correction converges (max 8 passes).
    from director_os.audio_engine.leveling import check_speech_window

    for _ in range(8):
        mixed = measure_speech_levels(str(master), gate)
        if check_speech_window(mixed["speech_rms_dbfs"]):
            break
        refined = round(gain_db + (DEFAULT_TARGET_DBFS - mixed["speech_rms_dbfs"]), 2)
        refined = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, refined))
        if abs(refined - gain_db) < 0.4:
            break
        run_two_bus_mix(bus1, raw_ref, master, refined)
        gain_db = refined
        log(f"  [audio] {tag}master gain refined -> {gain_db:+.1f} dB (post-limiter)")

    # 9) pass receipts
    raw_snr = measure_snr(str(raw_ref), gate)
    clean_snr = measure_snr(str(master), gate)
    tilt_raw = {"tilt_presence_db": 0.0, "tilt_air_db": 0.0, "tilt_lowmid_db": 0.0}
    tilt_clean = dict(tilt_raw)
    try:
        tilt_raw = speech_band_tilts(str(raw_ref), gate)
        tilt_clean = speech_band_tilts(str(master), gate)
    except ValueError as e:  # no speech inside the gate: nothing to measure
        log(f"  [audio] {tag}timbre tilt not measurable: {e}")
    crest = 0.0
    try:
        crest = speech_crest_db(str(master), gate)
    except ValueError as e:
        log(f"  [audio] {tag}speech crest not measurable: {e}")

    return _PassResult(
        master=master,
        raw_ref=raw_ref,
        gate=gate,
        snr_raw=raw_snr,
        snr_clean=clean_snr,
        tilt_raw=tilt_raw,
        tilt_clean=tilt_clean,
        crest=crest,
        engine=result.engine,
        neural=result.neural,
        atten_lim_db=result.atten_lim_db,
        enhancer_input=enhancer_input,
        gain_db=gain_db,
    )


def _shift_transcript(transcript: dict, start_s: float, end_s: float) -> dict:
    """Words inside [start, end] re-based to window-local time (gate truth)."""
    words = []
    for w in transcript.get("words") or []:
        s, e = w.get("start_s"), w.get("end_s")
        if s is None or e is None or not w.get("text"):
            continue
        if e <= start_s or s >= end_s:
            continue
        words.append(
            {
                "text": w["text"],
                "start_s": round(max(0.0, s - start_s), 3),
                "end_s": round(min(e, end_s) - start_s, 3),
                "speaker": w.get("speaker"),
            }
        )
    return {"words": words}


def _assemble_master(
    window_masters: list[tuple[float, float, Path]],
    total_s: float,
    work_dir: Path,
    dst: Path,
    log,
) -> None:
    """Place kept-window masters on the source clock; digital silence in gaps.

    All parts are 48 kHz stereo s16 WAV (the 2-bus law), so the concat is a
    lossless stream copy. The renderer slices this master by source time
    exactly as it slices the whole-asset master — dropped ranges are silence
    the render never touches.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    cursor = 0.0
    for i, (a, b, m) in enumerate(sorted(window_masters, key=lambda w: w[0])):
        if a > cursor + 0.02:
            gap = work_dir / f"gap_{i:03d}.wav"
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                    "-t",
                    f"{a - cursor:.3f}",
                    "-c:a",
                    "pcm_s16le",
                    str(gap),
                ],
                what="silence gap",
            )
            parts.append(gap)
        parts.append(m)
        cursor = max(cursor, b)
    if total_s > cursor + 0.02:
        gap = work_dir / "gap_tail.wav"
        _run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                "-t",
                f"{total_s - cursor:.3f}",
                "-c:a",
                "pcm_s16le",
                str(gap),
            ],
            what="silence tail",
        )
        parts.append(gap)
    if not parts:
        raise AudioEnhanceError("no windows to assemble — refusing to emit an empty master")
    lst = dst.with_suffix(".concat.txt")
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c:a",
            "copy",
            str(dst),
        ],
        what="master assembly (concat)",
    )
    lst.unlink(missing_ok=True)
    log(f"  [audio] assembled full-timeline master from {len(window_masters)} window(s)")


def _window_cache_dir(
    src: Path,
    enhancer_name: str,
    atten: float | None,
    enhancer_input: str,
    start: float,
    end: float,
    root: Path,
) -> tuple[Path, str]:
    canon = (
        f"{ENHANCE_CACHE_VERSION}|{_sha256_file(src)}|{enhancer_name}|{atten}|"
        f"{enhancer_input}|{SAMPLE_RATE}|{start:.3f}|{end:.3f}"
    )
    key = hashlib.sha256(canon.encode()).hexdigest()[:40]
    return root / f"w_{key}", key


def _pass_to_receipt(p: _PassResult) -> dict:
    data = asdict(p)
    data["master"] = str(p.master)
    data["raw_ref"] = str(p.raw_ref)
    data["gate"] = [list(g) for g in p.gate]
    return data


def _pass_from_receipt(data: dict) -> _PassResult:
    return _PassResult(
        master=Path(data["master"]),
        raw_ref=Path(data["raw_ref"]),
        gate=[(float(a), float(b)) for a, b in data["gate"]],
        snr_raw=data["snr_raw"],
        snr_clean=data["snr_clean"],
        tilt_raw=data["tilt_raw"],
        tilt_clean=data["tilt_clean"],
        crest=float(data["crest"]),
        engine=str(data["engine"]),
        neural=bool(data["neural"]),
        atten_lim_db=data["atten_lim_db"],
        enhancer_input=str(data["enhancer_input"]),
        gain_db=float(data["gain_db"]),
    )


def stage_audio_enhance(
    asset_id: str,
    source: Path,
    transcript: dict,
    project_id: str,
    deps,  # PipelineDeps (duck-typed to avoid a cycle)
    enhancer: SpeechEnhancer | None = None,
    kept_windows: list[tuple[float, float]] | None = None,
    music_flagged_spans: list[tuple[float, float]] | None = None,
) -> EnhanceInfo:
    """Run (or reuse) the per-asset 2-bus speech isolation pre-pass.

    `kept_windows` (post-planning keep ranges) flips on LAZY mode: heavy
    neural enhancement runs per KEPT window only; the assembled master keeps
    source-clock alignment with silence over the dropped ranges.
    """
    t0 = time.monotonic()
    src = Path(source)
    enhancer = enhancer or resolve_enhancer()
    atten = getattr(enhancer, "atten_lim_db", None)

    # Demucs pre-split: music-flagged assets route the vocals stem into the
    # enhancer (Content-ID guardrail). The stem is the timeline (verified by
    # the drift check in the pass).
    stem = deps.vocals_stem_resolver(asset_id) if deps.vocals_stem_resolver else None
    enhancer_input = "demucs_vocals" if stem is not None else "preclean"
    if music_flagged_spans:
        log = deps.log
        log(
            f"  [audio] music guardrail: {len(music_flagged_spans)} YAMNet music span(s) "
            f"-> enhancer input {enhancer_input}"
            + ("" if stem is not None else " (no Demucs stem yet — see ship-pack risk flag)")
        )

    key = enhance_cache_key(src, enhancer.name, atten, enhancer_input, kept_windows)
    cache_dir = deps.settings.output_dir / "audio2bus" / key
    master = cache_dir / "master_2bus.wav"
    sidecar = cache_dir / "enhance_receipt.json"
    raw_ref = cache_dir / "raw_48k.wav"

    if master.exists() and sidecar.exists():
        info = _info_from_receipt(json.loads(sidecar.read_text()))
        deps.log(f"  [audio] 2-bus cache hit ({info.engine}) -> {master.name}")
        return info

    windows = merge_windows(kept_windows) if kept_windows else []
    src_dur = 0.0
    try:
        from director_os.core.qa_probe import probe

        src_dur = float(probe(src).duration_s or 0.0)
    except Exception:  # noqa: BLE001 — duration is only used for the tail gap
        src_dur = 0.0

    if not windows:
        # ---------------------------------------------------------------- whole asset
        cache_dir.mkdir(parents=True, exist_ok=True)
        deps.log(f"  [audio] 2-bus speech isolation: engine={enhancer.name} input={enhancer_input}")
        p = _enhance_pass(
            src,
            transcript,
            cache_dir,
            enhancer,
            deps.log,
            stem_material=stem,
            enhancer_input=enhancer_input,
        )
        # canonical whole-asset master name (the cache-hit check looks for it)
        master = cache_dir / "master_2bus.wav"
        p.master.replace(master)
        p.master = master
        info = EnhanceInfo(
            asset_id=asset_id,
            master_wav=p.master,
            raw_reference_wav=p.raw_ref,
            engine=p.engine,
            neural=p.neural,
            atten_lim_db=p.atten_lim_db,
            enhancer_input=p.enhancer_input,
            gate_source="words+vad",
            speech_intervals=p.gate,
            raw_snr_db=p.snr_raw["snr_db"],
            clean_snr_db=p.snr_clean["snr_db"],
            delta_noise_db=p.snr_clean["noise_floor_db"] - p.snr_raw["noise_floor_db"],
            tilt_presence_raw_db=p.tilt_raw["tilt_presence_db"],
            tilt_presence_clean_db=p.tilt_clean["tilt_presence_db"],
            tilt_air_clean_db=p.tilt_clean["tilt_air_db"],
            speech_crest_db=p.crest,
            master_gain_db=p.gain_db,
            ducked_seconds=_ducked_seconds(p.gate, p.snr_clean["duration_s"]),
            total_seconds=p.snr_clean["duration_s"],
            cache_key=key,
            music_flagged_spans=[
                (round(a, 2), round(b, 2)) for a, b in (music_flagged_spans or [])
            ],
        )
        _atomic_json(sidecar, _receipt(info, t0))
        _persist(deps, project_id, asset_id, info, t0)
        deps.log(
            f"  [audio] 2-bus master ready: snr {info.raw_snr_db:.1f} -> "
            f"{info.clean_snr_db:.1f} dB (noise {info.delta_noise_db:+.1f} dB), "
            f"master gain {info.master_gain_db:+.1f} dB, crest {info.speech_crest_db:.1f} dB"
        )
        return info

    # ------------------------------------------------------- lazy kept windows
    total_kept = sum(b - a for a, b in windows)
    skipped = max(0.0, (src_dur or windows[-1][1]) - total_kept)
    deps.log(
        f"  [audio] lazy enhance: {len(windows)} kept window(s), {total_kept:.1f}s kept / "
        f"{skipped:.1f}s skipped (cut-out footage never enhanced)"
    )

    pass_results: list[tuple[float, float, _PassResult, str]] = []
    for start, end in windows:
        wdir, wkey = _window_cache_dir(
            src,
            enhancer.name,
            atten,
            enhancer_input,
            start,
            end,
            deps.settings.output_dir / "audio2bus",
        )
        win_master = wdir / "master_seg.wav"
        win_receipt = wdir / "window_receipt.json"
        if win_master.exists() and win_receipt.exists():
            deps.log(f"  [audio] window {start:.1f}-{end:.1f}s cache hit")
            rec = json.loads(win_receipt.read_text())
            pass_results.append((start, end, _pass_from_receipt(rec["pass"]), wkey))
            continue
        p = _enhance_pass(
            src,
            _shift_transcript(transcript, start, end),
            wdir,
            enhancer,
            deps.log,
            ss=start,
            t=end - start,
            stem_material=stem,
            enhancer_input=enhancer_input,
            label=f"win {start:.1f}-{end:.1f}s",
        )
        _atomic_json(
            win_receipt,
            {
                "pass": _pass_to_receipt(p),
                "window": [start, end],
                "cache_key": wkey,
            },
        )
        pass_results.append((start, end, p, wkey))

    # full-timeline raw reference (QA + A/B export law) — audio-only, cheap
    cache_dir.mkdir(parents=True, exist_ok=True)
    _extract_raw_48k(src, raw_ref)

    # assemble the full-timeline master: kept windows + silence gaps, source clock
    _assemble_master(
        [(a, b, p.master) for a, b, p, _ in pass_results],
        total_s=src_dur or windows[-1][1],
        work_dir=cache_dir,
        dst=master,
        log=deps.log,
    )

    gate_union = sorted((a + s, a + e) for a, b, p, _ in pass_results for s, e in p.gate)
    raw_snr = measure_snr(str(raw_ref), gate_union)
    clean_snr = measure_snr(str(master), gate_union)
    tilt_raw = {"tilt_presence_db": 0.0, "tilt_air_db": 0.0, "tilt_lowmid_db": 0.0}
    tilt_clean = dict(tilt_raw)
    try:
        tilt_raw = speech_band_tilts(str(raw_ref), gate_union)
        tilt_clean = speech_band_tilts(str(master), gate_union)
    except ValueError as e:
        deps.log(f"  [audio] timbre tilt not measurable: {e}")
    crest = 0.0
    try:
        crest = speech_crest_db(str(master), gate_union)
    except ValueError as e:
        deps.log(f"  [audio] speech crest not measurable: {e}")

    # per-window gain (max) keeps the receipt honest about the mix decisions
    gain_db = max(p.gain_db for _, _, p, _ in pass_results)
    info = EnhanceInfo(
        asset_id=asset_id,
        master_wav=master,
        raw_reference_wav=raw_ref,
        engine=pass_results[0][2].engine,
        neural=pass_results[0][2].neural,
        atten_lim_db=pass_results[0][2].atten_lim_db,
        enhancer_input=enhancer_input,
        gate_source="words+vad",
        speech_intervals=gate_union,
        raw_snr_db=raw_snr["snr_db"],
        clean_snr_db=clean_snr["snr_db"],
        delta_noise_db=clean_snr["noise_floor_db"] - raw_snr["noise_floor_db"],
        tilt_presence_raw_db=tilt_raw["tilt_presence_db"],
        tilt_presence_clean_db=tilt_clean["tilt_presence_db"],
        tilt_air_clean_db=tilt_clean["tilt_air_db"],
        speech_crest_db=crest,
        master_gain_db=gain_db,
        ducked_seconds=_ducked_seconds(gate_union, clean_snr["duration_s"]),
        total_seconds=clean_snr["duration_s"],
        cache_key=key,
        kept_windows=[(round(a, 2), round(b, 2)) for a, b in windows],
        kept_seconds=round(total_kept, 2),
        skipped_seconds=round(skipped, 2),
        music_flagged_spans=[(round(a, 2), round(b, 2)) for a, b in (music_flagged_spans or [])],
    )
    _atomic_json(sidecar, _receipt(info, t0))
    _persist(deps, project_id, asset_id, info, t0)
    deps.log(
        f"  [audio] 2-bus master ready (lazy): snr {info.raw_snr_db:.1f} -> "
        f"{info.clean_snr_db:.1f} dB, kept {info.kept_seconds:.1f}s / "
        f"skipped {info.skipped_seconds:.1f}s"
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
        "kept_windows": [list(p) for p in info.kept_windows],
        "music_flagged_spans": [list(p) for p in info.music_flagged_spans],
        "duration_ms": int((time.monotonic() - t0) * 1000),
        "afftdn_env": os.getenv("DIRECTOR_OS_ALLOW_AFFTDN", ""),
        "params": {
            "duck_db": -14.0,
            "ramp_s": 0.15,
            "bus2_volume": "opt-in via AUDIO_BUS2_VOLUME (default 0 = off)",
            "limiter_dbtp": -1.5,
            "sample_rate": SAMPLE_RATE,
        },
    }


def _info_from_receipt(payload: dict) -> EnhanceInfo:
    fields = set(EnhanceInfo.__dataclass_fields__)
    data = {k: v for k, v in payload.items() if k in fields}
    data["speech_intervals"] = [tuple(p) for p in data.get("speech_intervals", [])]
    data["kept_windows"] = [tuple(p) for p in data.get("kept_windows", [])]
    data["music_flagged_spans"] = [tuple(p) for p in data.get("music_flagged_spans", [])]
    data["master_wav"] = Path(data["master_wav"])
    data["raw_reference_wav"] = Path(data["raw_reference_wav"])
    return EnhanceInfo(**data)


def _persist(deps, project_id: str, asset_id: str, info: EnhanceInfo, t0: float) -> None:
    """media_assets.r2_enhanced_key + agent_tool_logs — the exact engine on record."""

    from director_os.core.db import AgentToolLogRow, MediaAssetRow

    lazy = (
        f" kept={info.kept_seconds:.1f}s skipped={info.skipped_seconds:.1f}s"
        if info.kept_windows
        else ""
    )
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
                    "kept_windows": [list(w) for w in info.kept_windows],
                    "music_flagged_spans": [list(w) for w in info.music_flagged_spans],
                },
                result_summary=(
                    f"enhancer_engine={info.engine} neural={info.neural} "
                    f"atten_lim_db={info.atten_lim_db} gate={info.gate_source} "
                    f"snr_raw={info.raw_snr_db:.1f} snr_clean={info.clean_snr_db:.1f} "
                    f"delta_noise={info.delta_noise_db:+.1f}dB "
                    f"tilt_presence={info.tilt_presence_clean_db:+.1f}dB{lazy}"
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
