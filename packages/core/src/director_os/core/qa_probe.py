"""FFprobe-based QA probe — the deterministic half of qa.audit_technical (PLAN §3.3G).

Also owns the *probe side* of the audio law (ADR-0005): a rendered file that is not
48 kHz stereo is a failed render, full stop. Speech-gated loudness (issue #15)
rides here too: RMS measured strictly on ASR spoken-word intervals.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SUPPORTED_SAMPLE_RATE = 48_000
SUPPORTED_CHANNELS = 2


class AudioLawError(RuntimeError):
    pass


class ProbeResult(BaseModel):
    duration_s: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_rate: int | None = None
    has_video: bool = False
    has_audio: bool = False


def _ffprobe(path: str | Path, *extra: str) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", *extra, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout)


def has_audio_stream(path: str | Path) -> bool:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return "audio" in (r.stdout or "")


def probe(path: str | Path) -> ProbeResult:
    data = _ffprobe(path, "-show_format", "-show_streams")
    fmt = data["format"]
    res = ProbeResult(
        duration_s=float(fmt["duration"]),
        bit_rate=int(fmt.get("bit_rate") or 0),
    )
    for s in data.get("streams", []):
        if s["codec_type"] == "video" and not res.has_video:
            res.has_video = True
            res.video_codec = s.get("codec_name")
            res.width = int(s["width"])
            res.height = int(s["height"])
            rate = s.get("r_frame_rate", "30/1")
            num, _, den = rate.partition("/")
            res.fps = float(num) / float(den or 1)
        elif s["codec_type"] == "audio" and not res.has_audio:
            res.has_audio = True
            res.audio_codec = s.get("codec_name")
            res.sample_rate = int(s.get("sample_rate") or 0)
            res.channels = int(s.get("channels") or 0)
    return res


def assert_audio_law(result: ProbeResult) -> None:
    """A rendered master that is not 48 kHz stereo fails loudly (ADR-0005)."""
    if not result.has_audio:
        raise AudioLawError("audio law violated: no audio stream")
    if result.sample_rate != SUPPORTED_SAMPLE_RATE or result.channels != SUPPORTED_CHANNELS:
        raise AudioLawError(
            f"audio law violated: got {result.sample_rate} Hz / {result.channels}ch, "
            f"required {SUPPORTED_SAMPLE_RATE} Hz / {SUPPORTED_CHANNELS}ch "
            "(mixed sample rates silently mute downstream narrator clips)"
        )


def _window_event_seconds(
    path: str | Path, filter_str: str, head_s: float | None, tail_s: float | None
) -> float:
    """Run a detect filter on ONLY the head or tail window; return event seconds.

    `-sseof` + `-t` keeps the scan cheap on long masters and makes the check
    about the OUT-FRAMES specifically (issue #18) instead of the whole file.
    """
    args = ["ffmpeg", "-hide_banner", "-nostats"]
    if head_s is not None:
        args += ["-t", f"{head_s:g}"]
    if tail_s is not None:
        args += ["-sseof", f"-{tail_s:g}"]
    args += ["-i", str(path), "-vf", filter_str, "-f", "null", "-"]
    r = subprocess.run(args, capture_output=True, text=True)
    total = 0.0
    for m in re.finditer(
        r"(?:black_start|freeze_start):\S+\s+(?:black_end|freeze_end):\S+\s+(?:black_duration|freeze_duration):([0-9.]+)",
        r.stderr,
    ):
        total += float(m.group(1))
    return total


def clean_out_frames_check(path: str | Path, window_s: float = 1.5) -> AuditCheck:
    """First/last frames are clean visible frames — no trailing black/frozen tail.

    YouTube renders what's in the container: a trailing black span reads as a
    broken video, a frozen last frame reads as a stall. blackdetect on the
    head AND tail windows + freezedetect on the tail catch all three.
    """
    try:
        head_black = _window_event_seconds(
            path, "blackdetect=d=0.2:pix_th=0.10", head_s=window_s, tail_s=None
        )
        tail_black = _window_event_seconds(
            path, "blackdetect=d=0.2:pix_th=0.10", head_s=None, tail_s=window_s
        )
        tail_freeze = _window_event_seconds(
            path, "freezedetect=n=0.003:d=1", head_s=None, tail_s=max(window_s, 2.0)
        )
        ok = head_black < 0.05 and tail_black < 0.05 and tail_freeze < 0.05
        detail = (
            f"head_black={head_black:.2f}s tail_black={tail_black:.2f}s "
            f"tail_freeze={tail_freeze:.2f}s (windows {window_s:g}s)"
        )
        return AuditCheck(name="clean_out_frames", ok=ok, detail=detail)
    except Exception as e:
        return AuditCheck(name="clean_out_frames", ok=False, detail=str(e))


def blackdetect_seconds(path: str | Path, min_dur: float = 0.5) -> float:
    """Total seconds of black video (ffmpeg blackdetect) — used by qa gates."""
    r = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(path),
            "-vf",
            f"blackdetect=d={min_dur}:pix_th=0.98",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    total = 0.0
    for line in (r.stderr or "").splitlines():
        if "black_start" in line:
            for token in line.split():
                if token.startswith("black_duration="):
                    total += float(token.split("=", 1)[1])
    return total


class QASummary(BaseModel):
    path: str
    probe: ProbeResult
    black_seconds: float = Field(default=0.0)
    audio_law_ok: bool = False


class AuditCheck(BaseModel):
    """One named QA gate result (PLAN §3.3G `qa.audit_technical`)."""

    name: str
    ok: bool
    detail: str = ""


class AuditReport(BaseModel):
    path: str
    ok: bool
    checks: list[AuditCheck] = Field(default_factory=list)

    @property
    def failures(self) -> list[AuditCheck]:
        return [c for c in self.checks if not c.ok]


class AudioVerification(BaseModel):
    """Inputs for the 2-bus acoustic SNR verification (audio.isolate_2bus).

    Carried by the enhance receipt: the raw 48 kHz reference (Bus-2 source),
    the exact enhancer engine used, and the SNR/clipping gate thresholds.
    """

    raw_audio_path: str
    enhancer_engine: str
    enhancer_neural: bool = True
    atten_lim_db: float | None = None
    # Source-clock speech spans for the RAW reference side of the SNR
    # comparison. The shipped master is a reordered cut: its spans live on the
    # MASTER clock (snr_speech_intervals), but the raw reference is a
    # source-clock file — each side is measured on its own clock's spans.
    raw_clock_speech_intervals: list[tuple[float, float]] | None = None
    # None -> hermetic-env check (afftdn allowed only inside tests)
    allow_afftdn: bool | None = None
    min_clean_snr_db: float = 14.0
    max_delta_noise_db: float = -8.0
    min_snr_improvement_db: float = 6.0
    max_peak_dbtp: float = -1.0
    # Enhanced kept windows per clock (SNR region-quantization, 2026-11-25):
    # a full-tape master ships unenhanced gap ambience by design — the floor
    # comparison must stay INSIDE the windows the enhancer touched, else the
    # gap ambience (untouched) poisons the clean floor. None -> whole file.
    enhanced_windows: list[tuple[float, float]] | None = None
    raw_keep_windows: list[tuple[float, float]] | None = None
    receipt_raw_snr_db: float | None = None
    receipt_clean_snr_db: float | None = None
    receipt_delta_noise_db: float | None = None


def audit_technical(
    path: str | Path,
    expected_dur_s: float | None = None,
    max_black_seconds: float = 3.0,
    duration_tolerance: float = 1.0,
    speech_intervals: list[tuple[float, float]] | None = None,
    speech_lo_dbfs: float = -12.0,
    speech_hi_dbfs: float = -6.0,
    audio_verification: AudioVerification | None = None,
    snr_speech_intervals: list[tuple[float, float]] | None = None,
) -> AuditReport:
    """Deterministic technical QA of a rendered master (the `qa` pipeline stage).

    Gates: playable streams present, the 48 kHz stereo audio law, black-frame
    budget, (when a target is given) duration drift vs. the compiled DAG,
    (when ASR speech intervals are given) speech-gated loudness — dialogue RMS
    measured ONLY on spoken spans must sit inside the -12..-6 dBFS window so
    sirens and slams cannot skew the meter (issue #15) — and (when the 2-bus
    verification is given) quantitative SNR proof + clipping + engine law.
    """
    path = Path(path)
    checks: list[AuditCheck] = []

    try:
        info = probe(path)
        checks.append(
            AuditCheck(
                name="streams_present",
                ok=info.has_video and info.has_audio,
                detail=f"video={info.has_video} audio={info.has_audio}",
            )
        )
    except Exception as e:  # unplayable file: everything else is moot
        return AuditReport(
            path=str(path),
            ok=False,
            checks=[AuditCheck(name="streams_present", ok=False, detail=str(e))],
        )

    law_ok = True
    law_detail = ""
    try:
        assert_audio_law(info)
        law_detail = f"{info.sample_rate}Hz/{info.channels}ch"
    except AudioLawError as e:
        law_ok = False
        law_detail = str(e)
    checks.append(AuditCheck(name="audio_law_48k_stereo", ok=law_ok, detail=law_detail))

    checks.append(_stream_uniformity_check(path))
    checks.append(_audio_duration_coverage_check(path))

    checks.append(clean_out_frames_check(path))

    black = blackdetect_seconds(path)
    checks.append(
        AuditCheck(
            name="black_frame_budget",
            ok=black <= max_black_seconds,
            detail=f"{black:.2f}s black (max {max_black_seconds:.1f}s)",
        )
    )

    if expected_dur_s is not None:
        drift = abs(info.duration_s - expected_dur_s)
        checks.append(
            AuditCheck(
                name="duration_drift",
                ok=drift <= duration_tolerance,
                detail=f"final={info.duration_s:.2f}s expected={expected_dur_s:.2f}s "
                f"(drift {drift:.2f}s, tol {duration_tolerance:.2f}s)",
            )
        )

    if speech_intervals:
        try:
            # soft import keeps core standalone-installable without audio_engine
            from director_os.audio_engine.leveling import check_speech_window, measure_speech_levels

            levels = measure_speech_levels(path, speech_intervals)
            speech_db = levels["speech_rms_dbfs"]
            ok = check_speech_window(speech_db, speech_lo_dbfs, speech_hi_dbfs)
            checks.append(
                AuditCheck(
                    name="speech_gated_loudness",
                    ok=ok,
                    detail=(
                        f"speech RMS {speech_db:.1f} dBFS "
                        f"(window {speech_lo_dbfs:g}..{speech_hi_dbfs:g}), "
                        f"file RMS {levels['file_rms_dbfs']:.1f} — siren does not skew the meter"
                    ),
                )
            )
        except Exception as e:
            checks.append(AuditCheck(name="speech_gated_loudness", ok=False, detail=str(e)))

    if audio_verification is not None:
        checks.extend(
            _audio_verification_checks(
                path,
                audio_verification,
                speech_intervals,
                snr_speech_intervals if snr_speech_intervals is not None else speech_intervals,
            )
        )

    return AuditReport(path=str(path), ok=all(c.ok for c in checks), checks=checks)


def _audio_stream_infos(path: Path) -> list[dict]:
    data = _ffprobe(path, "-show_streams")
    return [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]


def _audio_duration_coverage_check(path: Path, tolerance_s: float = 1.0) -> AuditCheck:
    """The audio stream must cover the video stream (no silent tail).

    Found live on tape-1 v2: the rendered master's audio ended 3.85 s before
    its video (the per-clip render under-covers) and NO gate noticed — the
    shipped file had a silent tail no one asked for. The audio stream must
    cover the video stream within `tolerance_s`.
    """
    try:
        data = _ffprobe(path, "-show_streams")
        video = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        audio = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
        if not video or not audio:
            return AuditCheck(name="audio_duration_coverage", ok=False, detail="missing stream")

        def _dur(s: dict) -> float:
            d = s.get("duration")
            if d is None:
                # count packets instead of trusting the container tag
                n = int(s.get("nb_frames") or 0)
                rate = float(s.get("avg_frame_rate") or 0) or float(s.get("r_frame_rate") or 0)
                if s.get("codec_type") == "audio":
                    rate = float(s.get("sample_rate") or 0)
                    return n / rate if rate else 0.0
                return n / rate if rate else 0.0
            return float(d)

        vd, ad = _dur(video[0]), _dur(audio[0])
        gap = vd - ad
        ok = gap <= tolerance_s
        return AuditCheck(
            name="audio_duration_coverage",
            ok=ok,
            detail=(f"video {vd:.2f}s / audio {ad:.2f}s (gap {gap:+.2f}s, tol {tolerance_s:g}s)"),
        )
    except Exception as e:  # noqa: BLE001
        return AuditCheck(name="audio_duration_coverage", ok=False, detail=str(e))


def _stream_uniformity_check(path: Path) -> AuditCheck:
    """EVERY audio stream in the master is 48000 Hz stereo — no stragglers."""
    try:
        streams = _audio_stream_infos(path)
        if not streams:
            return AuditCheck(name="stream_uniformity", ok=False, detail="no audio streams")
        bad = [
            f"stream {i}: {s.get('sample_rate')}Hz/{s.get('channels')}ch"
            for i, s in enumerate(streams)
            if int(s.get("sample_rate") or 0) != SUPPORTED_SAMPLE_RATE
            or int(s.get("channels") or 0) != SUPPORTED_CHANNELS
        ]
        ok = not bad
        detail = (
            f"{len(streams)} audio stream(s) all {SUPPORTED_SAMPLE_RATE} Hz stereo"
            if ok
            else "; ".join(bad)
        )
        return AuditCheck(name="stream_uniformity", ok=ok, detail=detail)
    except Exception as e:  # noqa: BLE001
        return AuditCheck(name="stream_uniformity", ok=False, detail=str(e))


def _audio_verification_checks(
    path: Path,
    verification: AudioVerification,
    speech_intervals: list[tuple[float, float]] | None,
    snr_speech_intervals: list[tuple[float, float]] | None = None,
) -> list[AuditCheck]:
    """The acoustic law for 2-bus renders: engine, SNR, clipping.

    `speech_intervals` (master-clock DIALOGUE spans) drive the timbre and
    crest audits. `snr_speech_intervals` (master-clock PROGRAM speech:
    dialogue + narration blocks) drives the SNR noise-floor comparison —
    the TTS voice is speech, not noise, and the floor must never sample it.
    The RAW reference is measured on its own source-clock spans
    (`raw_clock_speech_intervals`) — each side of the comparison rides its own
    file's clock.

    Soft-imports audio_engine exactly like the speech-gated loudness check —
    core stays standalone-installable without the audio extras.
    """
    from director_os.audio_engine.enhance import engine_is_production_legal
    from director_os.audio_engine.snr import (
        measure_snr,
        natural_dynamics_gate,
        peak_true_dbfs,
        snr_gate,
        speech_band_tilts,
        speech_crest_db,
        timbre_gate,
    )

    checks: list[AuditCheck] = []
    engine = verification.enhancer_engine
    legal = engine_is_production_legal(
        engine, verification.allow_afftdn, neural=verification.enhancer_neural
    )
    checks.append(
        AuditCheck(
            name="enhancer_engine_used",
            ok=legal,
            detail=(
                f"enhancer_engine={engine} neural={verification.enhancer_neural} "
                f"atten_lim_db={verification.atten_lim_db}"
                + (
                    ""
                    if legal
                    else " — afftdn fallback is FORBIDDEN in production renders "
                    "(install deep-filter / clearvoice, or run inside hermetic tests)"
                )
            ),
        )
    )

    clean_spans = snr_speech_intervals if snr_speech_intervals is not None else speech_intervals
    dialogue_spans = speech_intervals if speech_intervals is not None else []
    if clean_spans:
        try:
            raw_spans = (
                verification.raw_clock_speech_intervals
                if verification.raw_clock_speech_intervals
                else clean_spans
            )
            if (
                verification.receipt_clean_snr_db is not None
                and verification.receipt_raw_snr_db is not None
            ):
                # 2-bus receipts law: the whole-mix SNR meter is structurally
                # incompatible with the duck architecture — on the aggregate,
                # the between-word floor IS the ducked bed (the -14 dB duck),
                # so the gate rides the per-window 2-bus receipts measured at
                # enhance time; the enhance_carry_through probe below is the
                # lie-detector that the shipped file really contains those
                # enhanced windows.
                raw_snr = float(verification.receipt_raw_snr_db)
                clean_snr = float(verification.receipt_clean_snr_db)
                delta_noise = (
                    float(verification.receipt_delta_noise_db)
                    if verification.receipt_delta_noise_db is not None
                    else clean_snr - raw_snr
                )
                ok, gate_detail = snr_gate(
                    raw_snr,
                    clean_snr,
                    delta_noise,
                    verification.min_clean_snr_db,
                    verification.max_delta_noise_db,
                    verification.min_snr_improvement_db,
                )
                checks.append(
                    AuditCheck(
                        name="snr_verification",
                        ok=ok,
                        detail=(
                            f"2-bus receipts raw_snr={raw_snr:.1f}dB "
                            f"clean_snr={clean_snr:.1f}dB noise_delta {delta_noise:+.1f} | "
                            + gate_detail
                        ),
                    )
                )
                try:
                    through = measure_snr(
                        str(path), clean_spans, restrict_windows=verification.enhanced_windows
                    )
                    snr_ship = float(through["snr_db"])
                    floor_ship = float(through["noise_floor_db"])
                    # the swap-in lie-detector: nothing forbids the render
                    # side from handing the RAW mix back as "enhanced". The
                    # shipped program's speech-to-gap contrast inside the
                    # enhanced windows must still beat the receipt's raw SNR
                    # by the same improvement law — a raw swap-in reads
                    # snr_ship ~= receipt raw (improvement ~ 0) and dies here
                    # even though every receipt number is untouched.
                    through_ok = (
                        snr_ship >= verification.min_clean_snr_db
                        or (snr_ship - raw_snr) >= verification.min_snr_improvement_db
                    )
                    checks.append(
                        AuditCheck(
                            name="enhance_carry_through",
                            ok=through_ok,
                            detail=(
                                f"shipped program snr {snr_ship:.1f}dB "
                                f"(floor {floor_ship:.1f}) vs receipt raw {raw_snr:.1f}dB | "
                                + (
                                    "the enhanced speech really made it into the shipped bitstream"
                                    if through_ok
                                    else "the shipped windows read as ENHANCEMENT-NEUTRAL: "
                                    "the render shipped the enhancement's input, not its output"
                                )
                            ),
                        )
                    )
                except Exception as ex:  # noqa: BLE001
                    checks.append(
                        AuditCheck(name="enhance_carry_through", ok=False, detail=str(ex))
                    )
            else:
                raw = measure_snr(
                    verification.raw_audio_path,
                    raw_spans,
                    restrict_windows=verification.raw_keep_windows,
                )
                clean = measure_snr(
                    str(path), clean_spans, restrict_windows=verification.enhanced_windows
                )
                delta_noise = clean["noise_floor_db"] - raw["noise_floor_db"]
                ok, gate_detail = snr_gate(
                    raw["snr_db"],
                    clean["snr_db"],
                    delta_noise,
                    verification.min_clean_snr_db,
                    verification.max_delta_noise_db,
                    verification.min_snr_improvement_db,
                )
                checks.append(
                    AuditCheck(
                        name="snr_verification",
                        ok=ok,
                        detail=(
                            f"raw_snr={raw['snr_db']:.1f}dB clean_snr={clean['snr_db']:.1f}dB "
                            f"noise_floor {raw['noise_floor_db']:.1f} -> "
                            f"{clean['noise_floor_db']:.1f}dB (delta {delta_noise:+.1f}) | "
                            + gate_detail
                        ),
                    )
                )
        except Exception as e:  # noqa: BLE001
            checks.append(AuditCheck(name="snr_verification", ok=False, detail=str(e)))

        try:
            tilts = speech_band_tilts(str(path), dialogue_spans)
            ok, tdetail = timbre_gate(tilts)
            checks.append(
                AuditCheck(
                    name="timbre_not_muffled",
                    ok=ok,
                    detail=(
                        f"voice tilt on word spans: "
                        f"presence {tilts['tilt_presence_db']:+.1f}dB "
                        f"air {tilts['tilt_air_db']:+.1f}dB "
                        f"lowmid {tilts['tilt_lowmid_db']:+.1f}dB (vs mid) | " + tdetail
                    ),
                )
            )
        except Exception as e:  # noqa: BLE001
            checks.append(AuditCheck(name="timbre_not_muffled", ok=False, detail=str(e)))

        try:
            crest = speech_crest_db(str(path), dialogue_spans)
            ok, ddetail = natural_dynamics_gate(crest)
            checks.append(
                AuditCheck(
                    name="natural_dynamics",
                    ok=ok,
                    detail=(f"voice dynamics on word spans: crest {crest:.1f} dB | " + ddetail),
                )
            )
        except Exception as e:  # noqa: BLE001
            checks.append(AuditCheck(name="natural_dynamics", ok=False, detail=str(e)))

    try:
        peak = peak_true_dbfs(str(path))
        checks.append(
            AuditCheck(
                name="zero_clipping",
                ok=peak <= verification.max_peak_dbtp,
                detail=(
                    f"sample peak {peak:.2f} dBFS (gate <= {verification.max_peak_dbtp:g} dBTP, "
                    f"limiter ceiling -1.5 dBTP)"
                ),
            )
        )
    except Exception as e:  # noqa: BLE001
        checks.append(AuditCheck(name="zero_clipping", ok=False, detail=str(e)))
    return checks


def audit_layout_and_policy(
    ass_path: str | Path | None = None,
    cards: list[dict[str, Any]] | None = None,
    blurs: list[dict[str, Any]] | None = None,
    frame: tuple[int, int] = (1920, 1080),
    lane_margins: tuple[int, int] = (46, 96),
) -> AuditReport:
    """Deterministic layout/policy audit (PLAN §3.3G `qa.audit_layout_and_policy`).

    Gates (each becomes a named check):
    - subtitle_lanes: the two-lane margin law holds (MarginV 46/96), type is
      upright bold, and cues within one lane never overlap in time
    - card_policy: every card carries a non-empty title
    - blur_policy: every privacy blur box stays inside the canvas, sigma sane

    The subtitle gate re-reads the *rendered* .ass file with pysubs2 — defense
    in depth against the builder: what ships is what gets audited.
    """
    checks: list[AuditCheck] = []
    any_input = ass_path is not None or cards or blurs

    if ass_path is not None:
        checks.append(_audit_subtitle_lanes(Path(ass_path), lane_margins))

    for card in cards or []:
        fields = card.get("fields") or {}
        title = str(fields.get("title") or "").strip()
        checks.append(
            AuditCheck(
                name="card_policy",
                ok=bool(title),
                detail=(
                    f"card '{card.get('template')}' title ok" if title else "card has empty title"
                ),
            )
        )
        bbox = card.get("bbox") or card.get("rect")
        if bbox:
            try:
                bx, by, bw, bh = (int(bbox[k]) for k in ("x", "y", "w", "h"))
                from director_os.vision_fx.safe_zone import check_safe_zone

                ok, detail = check_safe_zone((bx, by, bw, bh), frame[0], frame[1])
                checks.append(
                    AuditCheck(
                        name="card_safe_zone",
                        ok=ok,
                        detail=f"card '{card.get('template')}': {detail}",
                    )
                )
            except (KeyError, TypeError, ValueError):
                checks.append(
                    AuditCheck(
                        name="card_safe_zone", ok=False, detail=f"card malformed bbox {bbox}"
                    )
                )

    for i, blur in enumerate(blurs or []):
        bbox = blur.get("bbox") or {}
        try:
            x, y, w, h = (int(bbox[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            checks.append(
                AuditCheck(name="blur_policy", ok=False, detail=f"blur {i}: malformed bbox {bbox}")
            )
            continue
        inside = x >= 0 and y >= 0 and x + w <= frame[0] and y + h <= frame[1]
        sigma = blur.get("sigma", 0)
        sigma_ok = isinstance(sigma, (int, float)) and 1 <= sigma <= 200
        ok = inside and sigma_ok
        detail = (
            f"blur {i} bbox {bbox} sigma {sigma}: "
            + ("inside frame" if inside else f"EXCEEDS frame {frame[0]}x{frame[1]}")
            + ("" if sigma_ok else ", sigma out of 1..200")
        )
        checks.append(AuditCheck(name="blur_policy", ok=ok, detail=detail))

    if not any_input:
        checks.append(AuditCheck(name="inputs", ok=False, detail="no ass/cards/blurs provided"))

    return AuditReport(path=str(ass_path or ""), ok=all(c.ok for c in checks), checks=checks)


def _audit_subtitle_lanes(ass_path: Path, lane_margins: tuple[int, int]) -> AuditCheck:
    """Margin law (46/96), upright bold faces, and per-lane no-overlap."""
    import pysubs2

    try:
        subs = pysubs2.SSAFile.from_string(ass_path.read_text(encoding="utf-8"))
    except Exception as e:
        return AuditCheck(name="subtitle_lanes", ok=False, detail=f"ass unreadable: {e}")

    problems: list[str] = []
    lane_styles: dict[int, list[str]] = {1: [], 2: []}
    for style_name, style in subs.styles.items():
        m = re.match(r"Lane([12])_", style_name)
        if not m:
            continue
        lane = int(m.group(1))
        lane_styles[lane].append(style_name)
        expected = lane_margins[lane - 1]
        if style.marginv != expected:
            problems.append(f"{style_name}: MarginV {style.marginv} != law {expected}")
        if not style.bold:
            problems.append(f"{style_name}: not bold (law: upright bold)")
        if style.italic:
            problems.append(f"{style_name}: italic (law: upright)")

    events_by_lane: dict[int, list[pysubs2.SSAEvent]] = {}
    for event in subs.events:
        m = re.match(r"Lane([12])_", event.style or "")
        if not m:
            continue
        events_by_lane.setdefault(int(m.group(1)), []).append(event)

    for lane, events in events_by_lane.items():
        ordered = sorted(events, key=lambda e: e.start)
        for a, b in zip(ordered, ordered[1:], strict=False):
            if b.start < a.end:
                problems.append(
                    f"lane {lane} overlap at {a.end / 1000:.2f}s "
                    f"({b.start / 1000:.2f}s starts before previous cue ends)"
                )
                break

    if problems:
        return AuditCheck(name="subtitle_lanes", ok=False, detail="; ".join(problems[:4]))
    lanes_used = sorted(events_by_lane)
    return AuditCheck(
        name="subtitle_lanes",
        ok=True,
        detail=(
            f"margin law {lane_margins[0]}/{lane_margins[1]} ok; lanes used: {lanes_used or 'none'}"
        ),
    )
