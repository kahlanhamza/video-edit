"""Durable staged pipeline — PLAN §7 Stage 1-10 state machine (M2 core).

Each stage persists its product in a DOMAIN table before advancing the job
row, so a worker that dies mid-pipeline resumes without redoing paid work:

- ingest     -> `media_assets` (probe facts)          — idempotent per project+source
- redaction  -> `perception_index.redaction_tracks_json` (sweep plan, keep opt-outs)
- perception -> `perception_index` (ASR transcript)   — idempotent per asset: the
               crash-resume guarantee "never re-run ASR/TTS" comes from here
- proofread  -> `transcript_json["proofread"]` (jargon + Officer/Suspect roles)
- vision     -> `shots_json` (frame quality, YAMNet events incl. Music, VLM
               index + Frames-on-Demand QA, AXON OCR) — every product optional
- planning   -> `timeline_versions` + `projects.active_otio_version_id`
- narration  -> content-addressed TTS clips on disk (re-renders never re-bill)
- audio_prep -> 2-bus masters, LAZY per KEPT window (content-addressed; the
               cut-out footage never runs through the neural enhancer)
- render     -> `segment_cache` (content-addressed)   — misses only, ever
- subtitles  -> subtitles.{json,ass,srt} (proofread words, 2-row karaoke)
- qa         -> `audit_technical` report; failure parks the job in `qa_failed`
- ship_pack  -> output/pipeline/<project>/ship_pack/ (Stage 10 creator bundle)

The ASR callable is injected (`deps.transcribe`): Gladia in production, a stub
in tests. Nothing here talks to the network.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from director_os.core.config import Settings, get_settings
from director_os.core.db import (
    CaseDossierRow,
    CaseSourceRow,
    JobRow,
    MediaAssetRow,
    PerceptionIndexRow,
    ProjectRow,
    TimelineVersionRow,
    init_models,
    make_engine,
    make_session_factory,
)
from director_os.core.http import download
from director_os.core.models import BlurRegion, JobStage, ProjectStatus
from director_os.core.proxy import ProxyBuildError, build_proxies
from director_os.core.qa_probe import AuditReport, audit_technical, probe
from director_os.otio_engine.compiler import compile_timeline
from director_os.otio_engine.contract import CardMeta, RawClipMeta
from director_os.otio_engine.renderer import RenderReport, render_dag
from director_os.otio_engine.segment_cache import AudioRenderSpec, SegmentCache, VideoRenderSpec
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
    save_otio,
    to_json_dict,
)
from director_os.vision_fx.trajectory import PerFrameRedaction
from director_os.worker.progress import RunTracker
from sqlalchemy import select

AUTO_CUT_MAX_SEGMENT_S = 8.0
# sources >= this get the multi-act documentary treatment (LLM story-map
# Director or deterministic fallback); shorter sources keep the v1 auto-cut.
# SCALED LAW (owner rule 2026-11-19): the documentary path is length-adaptive
# (doc_scale.py) — an 8-minute tape plans a ~2.5-min doc, a 43-minute tape a
# 10-15-min doc; nothing in the law assumes one hardcoded source length.
DOCUMENTARY_MIN_SOURCE_S = 480.0
# retention QA binds on every documentary source (floors scale with length)
RETENTION_QA_MIN_SOURCE_S = 480.0
# narration re-time tail pad: sentence-final words never clip (bounded 0.2-3s
# inside retime_narration_to_tts); measured TTS duration + this pad = item len
NARR_TAIL_PAD_S = 0.35


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineDeps:
    sessions: Any  # sessionmaker
    settings: Settings
    transcribe: Callable[[Path, str], Any] | None = None  # -> perception Transcript
    video: VideoRenderSpec = field(default_factory=VideoRenderSpec)
    audio: AudioRenderSpec = field(default_factory=AudioRenderSpec)
    max_workers: int | None = None
    # --- audio law: "loudnorm_48k" (legacy) | "isolate_2bus" (broadcast mix)
    audio_mode: str = "loudnorm_48k"
    speech_enhancer: Any | None = None  # SpeechEnhancer override (tests/fakes)
    vocals_stem_resolver: Callable[[str], Path | None] | None = None  # Demucs pre-split hook
    # --- M4 vision stack (all optional; None/False keeps the CPU-only pipeline)
    redact_backend: str | None = None  # "tracked" (3-pass, issue #14) | "yunet" | "modal"
    vision_enrich: bool = False  # frame-quality score + VLM timeline + timestamp OCR
    # --- M2 close-out: LLM Director autodrive (Gemini controls the edit)
    llm_direct: bool = False  # stage_director after planning (soft-fails to the v1 cut)
    director_max_rounds: int | None = None  # None -> DIRECTOR_MAX_ROUNDS env (default 4)
    # --- M5 story-map Director session (Phase A/B) + CaseDossier injection
    # points; tests fake these, production resolves the live routing chain
    director_backend_factory: Callable[[], Any] | None = None
    complete_json: Callable[[str, str], str] | None = None  # dossier synthesis
    fetch: Callable[[str], Any] | None = None  # case-source fetching (tests)
    # --- POST-FX (owner rule 2026-11-19): tracked blur + audio enhancement +
    # caption burn run ONLY on the assembled deliverable, never the raw source
    post_fx: bool = False
    log: Callable[[str], None] = lambda _msg: None
    # --- live observability (Studio full-visibility mandate): one RunTracker
    # per run streams steps/logs/timeline revisions into job_* rows for SSE
    tracker: RunTracker | None = None


@dataclass
class PipelineResult:
    job_id: str
    project_id: str
    asset_id: str
    otio_version: int
    final_path: Path
    otio_path: Path
    render: RenderReport
    audit: AuditReport
    timeline: Any = None  # the shipped master timeline (act-breakdown reads)
    source_duration_s: float = 0.0
    dossier: dict | None = None
    post_fx: Any = None  # PostFxReport receipt when the post-FX pass ran


def _stop_flag(deps: PipelineDeps, project_id: str) -> Path:
    return deps.settings.output_dir / "pipeline" / project_id / "STOP"


def _check_stop(deps: PipelineDeps, project_id: str) -> None:
    """Cooperative stop: the operator's STOP sentinel parks the job at the
    next stage boundary. The checkpoint law applies — every completed stage
    stays done; Run (resume) clears the flag and continues from the parked
    stage. Never mid-encode: an ffmpeg in flight finishes, the NEXT boundary
    stops."""
    if _stop_flag(deps, project_id).exists():
        raise PipelineError("stopped by operator — press Run to resume (completed stages are kept)")


def _update_job(
    sessions, job_id: str, stage: JobStage, progress: float, error: str | None = None
) -> None:
    """Own the session internally — callers used to pass a bare session from
    `deps.sessions()` that was never closed, leaking one pooled connection
    per stage update (exhausts the pool once the Studio polls progress)."""
    with sessions() as s:
        job = s.get(JobRow, job_id)
        if job is None:
            return
        job.current_stage = stage.value
        job.progress_pct = progress
        job.error_message = error
        s.commit()


def _set_status(s, project_id: str, status: ProjectStatus) -> None:
    project = s.get(ProjectRow, project_id)
    project.status = status.value
    s.commit()


# --------------------------------------------------------------------- stages
def stage_ingest(source: Path, project_id: str, deps: PipelineDeps) -> str:
    """Probe the local source into `media_assets`, build the studio proxy bundle.

    Proxy generation (`ingest.build_proxies`, PLAN §3.3A) is idempotent: an
    existing asset row short-circuits it, and a crashed run resumes because the
    row only commits after the proxy artifacts are on disk.
    """
    with deps.sessions() as s:
        row = s.execute(
            select(MediaAssetRow).where(
                MediaAssetRow.project_id == project_id,
                MediaAssetRow.source_url == str(source),
            )
        ).scalar_one_or_none()
        if row is not None:
            deps.log(f"  [ingest] asset {row.id} already registered — skip probe")
            return row.id

    info = probe(source)
    if not info.has_video:
        raise PipelineError(f"source has no video stream: {source}")
    fps = info.fps or 30.0
    plausible = {23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0}
    if fps not in plausible:
        fps = min(plausible, key=lambda c: abs(c - fps))  # normalize odd rates

    with deps.sessions() as s:
        row = MediaAssetRow(
            project_id=project_id,
            source_url=str(source),
            role="raw_bwc",
            duration_s=info.duration_s,
            fps=fps,
            width=info.width or 0,
            height=info.height or 0,
        )
        s.add(row)
        s.flush()  # need asset_id before naming the proxy directory
        try:
            bundle = build_proxies(
                source, deps.settings.output_dir / "pipeline" / project_id / "assets" / row.id
            )
        except (ProxyBuildError, FileNotFoundError) as e:
            s.rollback()  # never persist an asset without its proxy bundle
            raise PipelineError(f"proxy build failed: {e}") from e
        row.r2_proxy_key = str(bundle.proxy_path)
        row.r2_audio_key = str(bundle.audio_path)
        deps.log(f"  [ingest] proxy 480p + scrubsheet + 64k audio -> assets/{row.id[:8]}")
        _set_status(s, project_id, ProjectStatus.ingested)
        s.commit()
        return row.id


def _asr_input_path(asset_id: str, source: Path, project_id: str, deps: PipelineDeps) -> Path:
    """ASR uploads the COMPRESSED ingest audio, never the raw source.

    The 2026-11-23 incident: a 469 MB bodycam went to Gladia verbatim and the
    provider edge killed every attempt (HTTP 408 / mid-upload disconnect).
    Ingest already ships `audio64k.m4a` (AAC 64 kbps mono — ~20 MB for a
    43-min source) with the proxy bundle; ASR rides that. Legacy assets
    without a bundle get one extracted on the spot, and only a media file
    with no audio stream at all falls back to the raw source.
    """
    with deps.sessions() as s:
        row = s.get(MediaAssetRow, asset_id)
        key = row.r2_audio_key if row is not None else None
    if key:
        candidate = Path(key)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    out_dir = deps.settings.output_dir / "pipeline" / project_id / "assets" / asset_id
    extracted = out_dir / "audio64k.m4a"
    if extracted.exists() and extracted.stat().st_size > 0:
        return extracted
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                str(extracted),
            ],
            what="ASR audio extraction",
        )
        deps.log(f"  [perception] extracted compressed ASR audio -> {extracted.name}")
        # persist so the next run skips the extraction
        with deps.sessions() as s:
            row = s.get(MediaAssetRow, asset_id)
            if row is not None:
                row.r2_audio_key = str(extracted)
                s.commit()
        return extracted
    except Exception as e:  # noqa: BLE001 — ASR can still ride the raw source
        deps.log(f"  [perception] compressed audio extraction failed ({e}) — raw source rides")
        return source


def stage_perception(asset_id: str, source: Path, project_id: str, deps: PipelineDeps) -> dict:
    """Transcribe once; the persisted transcript makes re-runs free (crash resume)."""
    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        has_words = bool((row.transcript_json or {}).get("words")) if row is not None else False
        if has_words:
            deps.log(f"  [perception] transcript for {asset_id[:8]} already persisted — skip ASR")
            return row.transcript_json  # type: ignore[union-attr]

    if deps.transcribe is None:
        raise PipelineError("no transcribe callable provided (Gladia not configured?)")
    asr_media = _asr_input_path(asset_id, source, project_id, deps)
    mib = asr_media.stat().st_size / (1024 * 1024)
    deps.log(f"  [perception] ASR input: {asr_media.name} ({mib:.1f} MiB)")
    transcript = deps.transcribe(asr_media, asset_id)
    payload = transcript.model_dump(mode="json")

    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        if row is None:
            row = PerceptionIndexRow(asset_id=asset_id)
            s.add(row)
        row.transcript_json = payload
        _set_status(s, project_id, ProjectStatus.perceived)
        s.commit()
    deps.log(f"  [perception] ASR persisted ({len(payload.get('words', []))} words)")
    return payload


def stage_proofread(asset_id: str, transcript: dict, project_id: str, deps: PipelineDeps) -> dict:
    """Police-jargon proofread + Officer/Suspect role tagging — persisted, idempotent.

    Two-tier law (the committed v6 "LLM operator supervisor" + the manual
    sandbox workflow):
    1. DETERMINISTIC FLOOR — jargon corrections ("ten four" -> "10-4") re-timed
       via difflib alignment, diarization labels ranked into documentary roles.
       Always available, never billed.
    2. LLM SUPERVISOR (when an LLM rung is reachable) — the notebook pass:
       readable speaker turns go to the LLM in batches of 70 with the case
       facts; it fixes real ASR mishearings ("I got a super suit" -> "I got
       off work"), assigns TRUE Officer/Suspect roles, masks profanity, and
       drops garble (clean_text=""). Every correction is WORD-LOCKED back onto
       the original ASR timestamps — sync cannot drift.

    Product lives in `perception_index.transcript_json["proofread"]` (the same
    durability contract as ASR). Downstream consumers (subtitles, narration,
    Director planning) read the tagged words; crash-resume never re-derives.
    """
    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        existing = (row.transcript_json or {}).get("proofread") if row else None
        if existing and existing.get("words"):
            deps.log(f"  [proofread] {len(existing['words'])} word(s) already persisted — skip")
            return existing

    from director_os.subtitles.proofread import prepare_transcript

    proof = prepare_transcript(transcript)

    # tier 2: the LLM supervisor pass (soft-fail — the floor always stands)
    try:
        backend = _director_backend(deps)
    except Exception as e:  # noqa: BLE001 — supervisor is an upgrade, never a gate
        backend = None
        deps.log(f"  [supervisor] backend unavailable ({e})")
    if backend is not None:
        from director_os.subtitles.supervisor import supervise_transcript

        case_facts = None
        dossier = _load_dossier(project_id, deps)
        if dossier:
            case_facts = {
                k: dossier.get(k)
                for k in ("subject", "location", "date", "agency", "charges_or_outcome")
                if dossier.get(k)
            } or None
        supervised = supervise_transcript(
            {**(transcript or {}), "words": proof["words"]},
            getattr(backend, "complete_json", None),
            case_facts=case_facts,
            log=deps.log,
        )
        if supervised is not None:
            # keep the deterministic jargon corrections the supervisor saw the
            # post-correction words through; merge their corrections list
            supervised["corrections"] = [*proof.get("corrections", []), *supervised["corrections"]]
            proof = supervised

    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        if row is None:
            row = PerceptionIndexRow(asset_id=asset_id)
            s.add(row)
        row.transcript_json = {**(row.transcript_json or {}), "proofread": proof}
        s.commit()
    roles = proof.get("speaker_roles") or {}
    deps.log(
        f"  [proofread] {len(proof['words'])} word(s), {len(proof['corrections'])} correction(s), "
        f"roles: {roles if roles else '(no diarization)'}"
        + (" [supervised]" if proof.get("supervised") else " [deterministic]")
    )
    return proof


def stage_narration(timeline, project_id: str, deps: PipelineDeps) -> list[dict]:
    """Persist narrator TTS clips BEFORE render — re-renders never regenerate.

    Walks NarratorClipMeta clips on the master track; every script is
    content-addressed (`tts_cache_key`: provider|voice|text) into
    output/pipeline/<project>/narration/{key}.mp3 + .json. An unchanged
    voiceover is a cache hit (never re-synthesized, never re-billed); an
    edited script changes exactly one key. The compiled narration segments
    ride the persisted clip (compile_timeline `narration_audio=`), so the
    voiceover in the shipped master IS the cached clip.
    """
    from director_os.audio_engine.tts import generate_tts_cached
    from director_os.otio_engine.contract import NarratorClipMeta, parse_clip_meta
    from director_os.otio_engine.timeline import MASTER_TRACK, is_gap, track_by_name

    master = track_by_name(timeline, MASTER_TRACK)
    clips: list[dict] = []
    if master is not None:
        for item in master:
            if is_gap(item):
                continue
            try:
                meta = parse_clip_meta(item)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(meta, NarratorClipMeta):
                continue
            out_dir = deps.settings.output_dir / "pipeline" / project_id / "narration"
            result, cache_hit = generate_tts_cached(out_dir, meta.script_text, voice=meta.voice_id)
            clips.append(
                {
                    "item": getattr(item, "name", "") or "",
                    "script_text": meta.script_text,
                    "script_chars": len(meta.script_text),
                    "clip": str(result.path),
                    "sidecar": str(result.path.with_suffix(".json")),
                    "duration_s": round(result.duration_s, 3),
                    "provider": result.provider,
                    "voice": meta.voice_id,
                    "cache_hit": cache_hit,
                }
            )
            deps.log(
                f"  [narration] clip {'cache hit' if cache_hit else 'synthesized'} "
                f"({result.duration_s:.1f}s, {result.provider}) -> {result.path.name}"
            )
    if not clips:
        deps.log("  [narration] no narrator clips on the timeline — nothing to synthesize")
        return clips

    manifest = deps.settings.output_dir / "pipeline" / project_id / "narration" / "clips.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # entries carry the script + sidecar path so the POST-FX caption builder can
    # lay TTS word stamps on the master clock without re-synthesizing
    _atomic_json_write(manifest, {"clips": clips})
    return clips


def retime_narration_to_tts(timeline, narration_clips: list[dict]) -> int:
    """Re-time narration segments to their REAL TTS clip durations.

    The plan's dur_s is an LLM estimate; the shipped voiceover is the cached
    TTS clip. Without this pass a short TTS leaves trailing dead air (the
    sleepy-pause complaint) and a long TTS gets its final words CUT OFF.
    Each NarratorClipMeta item is re-timed to tts_duration + NARR_TAIL_PAD_S
    (bounded 0.2-3s pad), b-roll window start unchanged. Returns count.
    """
    import opentimelineio.opentime as ot_time
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.contract import NarratorClipMeta, parse_clip_meta
    from director_os.otio_engine.timeline import is_gap

    dur_by_script = {
        c.get("script_text", ""): float(c.get("duration_s") or 0.0)
        for c in narration_clips or []
        if c.get("script_text")
    }
    if not dur_by_script:
        return 0
    retimed = 0
    try:
        master = master_track(timeline)
    except Exception:
        return 0
    for item in master:
        if is_gap(item):
            continue
        try:
            meta = parse_clip_meta(item)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(meta, NarratorClipMeta):
            continue
        tts_dur = dur_by_script.get(meta.script_text)
        if not tts_dur or tts_dur <= 0:
            continue
        sr = getattr(item, "source_range", None)
        if sr is None:
            continue
        pad = min(max(0.2, NARR_TAIL_PAD_S), 3.0)
        if meta.vo_out_s is not None or meta.vo_in_s > 0:
            # A1 audio trim: the block rides the TRIMMED vo window, never the
            # full stem (a trim that cuts the block to 1s must stay 1s — the
            # 2s floor below exists only to rescue short UNtrimmed estimates).
            vo_in, vo_out = meta.vo_window(tts_dur)
            win = max(0.0, vo_out - vo_in)
            new_dur = max(0.1, win + pad)
        else:
            new_dur = max(2.0, tts_dur + pad)
        rate = sr.duration.rate
        item.source_range = ot_time.TimeRange(
            start_time=sr.start_time,
            # RationalTime carries FRAMES at a rate: value = seconds * rate
            # (the pre-fix code passed seconds as frames and shrank every
            # narration item to 1/30th of its TTS duration)
            duration=ot_time.RationalTime(value=new_dur * rate, rate=rate),
        )
        retimed += 1
    return retimed


def master_spans(timeline) -> list[tuple[float, float]]:
    """[(start, dur), ...] of the master track — the pre-retime snapshot the
    card re-anchor pass needs (call BEFORE retime_narration_to_tts)."""
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.timeline import item_duration_s

    spans: list[tuple[float, float]] = []
    t = 0.0
    for item in master_track(timeline):
        d = item_duration_s(item)
        spans.append((t, d))
        t += d
    return spans


def retime_cards_after_narration(timeline, pre_spans: list[tuple[float, float]]) -> int:
    """Re-anchor overlay cards after the TTS re-time (measured WAV durations).

    Cards were placed at planned (word-count estimated) narration boundaries;
    after `retime_narration_to_tts` the narration items follow the REAL TTS
    durations, so every card would drift by the accumulated delta — the
    acceptance review caught location_intro landing inside the wrong block.
    Each card keeps its anchor ITEM (pre-retime snapshot) and local offset:
    new_pos = new_start[idx] + min(local, new_dur[idx]); the overlay track is
    rebuilt as a gap/card sequence with the same mechanics as add_card.
    """
    import opentimelineio.opentime as ot_time
    import opentimelineio.schema as ot_schema
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.timeline import (
        OVERLAY_TRACK,
        is_gap,
        item_duration_s,
        track_by_name,
    )

    master = master_track(timeline)
    overlay = track_by_name(timeline, OVERLAY_TRACK)
    if overlay is None or not pre_spans:
        return 0

    # post-retime master item starts
    new_starts: list[float] = []
    t = 0.0
    for item in master:
        new_starts.append(t)
        t += item_duration_s(item)

    cards: list[tuple[float, Any]] = []
    t = 0.0
    for item in overlay:
        d = item_duration_s(item)
        meta = item.metadata.get("director_os") or {}
        if meta.get("kind") == "card" and not is_gap(item):
            cards.append((t, item))
        t += d
    if not cards:
        return 0

    moved = 0
    placements: list[tuple[float, Any]] = []
    for pos, card in cards:
        idx = len(pre_spans) - 1
        for i, (s0, d0) in enumerate(pre_spans):
            if s0 - 0.01 <= pos < s0 + max(d0, 0.01):
                idx = i
                break
        local = max(0.0, pos - pre_spans[idx][0])
        new_dur = item_duration_s(master[idx]) if idx < len(master) else 0.0
        if new_dur > 0:
            new_pos = new_starts[idx] + min(local, max(0.5, new_dur - 0.25))
        else:
            new_pos = new_starts[idx]
        new_pos = max(0.0, new_pos)
        if abs(new_pos - pos) > 0.05:
            moved += 1
        placements.append((new_pos, card))

    # rebuild the overlay: gap/card sequence covering the (new) master duration
    total = sum(item_duration_s(i) for i in master)
    placements.sort(key=lambda p: p[0])
    rate = 30  # overlay timings ride the master rate; compiler normalizes on read
    new_items: list[Any] = []
    cursor = 0.0
    for pos, card in placements:
        cd = item_duration_s(card)
        pos = min(pos, max(0.0, total - cd))
        if pos > cursor + 0.01:
            g = ot_schema.Gap()
            g.source_range = ot_time.TimeRange(
                start_time=ot_time.RationalTime(0, rate),
                duration=ot_time.RationalTime((pos - cursor) * rate, rate),
            )
            new_items.append(g)
            cursor = pos
        new_items.append(card)
        cursor += cd
    if total - cursor > 0.01:
        g = ot_schema.Gap()
        g.source_range = ot_time.TimeRange(
            start_time=ot_time.RationalTime(0, rate),
            duration=ot_time.RationalTime((total - cursor) * rate, rate),
        )
        new_items.append(g)
    for it in list(overlay):
        overlay.remove(it)
    for it in new_items:
        overlay.append(it)
    return moved


def _atomic_json_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


# ------------------------------------------------- post-plan sweep scoping law
def _kept_source_ranges(timeline: Any, asset_id: str) -> list[tuple[float, float]]:
    """Kept (start_s, end_s) spans on the SOURCE timeline, merged.

    The post-plan scoping law (operator mandate): detection pays only for
    footage the edit kept. Gaps/cut-out footage are never swept; cards and
    narration items carry no source span. Spans within 2.0s of each other
    merge — a hairline cut should not split one face track (the repair
    bridge window is ~2s).
    """
    from director_os.otio_engine.contract import RawClipMeta, parse_clip_meta
    from director_os.otio_engine.timeline import (
        MASTER_TRACK,
        is_gap,
        item_duration_s,
        track_by_name,
    )

    master = track_by_name(timeline, MASTER_TRACK)
    if master is None:
        return []
    spans: list[tuple[float, float]] = []
    for item in master:
        dur = item_duration_s(item)
        if dur <= 0 or is_gap(item):
            continue
        try:
            meta = parse_clip_meta(item)
        except Exception:  # noqa: BLE001 — non-contract items have no source span
            continue
        if not isinstance(meta, RawClipMeta) or meta.asset_id != asset_id:
            continue
        start = item.source_range.start_time.to_seconds() if item.source_range else 0.0
        spans.append((max(0.0, round(start, 3)), max(0.0, round(start + dur, 3))))
    spans.sort()
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= 2.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _ranges_digest(backend: str, ranges: list[tuple[float, float]] | None) -> str:
    """Sweep cache key: backend + normalized kept ranges (2dp)."""
    import hashlib

    norm = [[round(a, 2), round(b, 2)] for a, b in (ranges or [])]
    return hashlib.sha256(json.dumps([backend, norm]).encode()).hexdigest()[:10]


def _sweep_cache_path(deps: PipelineDeps, project_id: str, asset_id: str, digest: str) -> Path:
    return (
        deps.settings.output_dir
        / "pipeline"
        / project_id
        / f"redact_sweep_{asset_id[:8]}_{digest}.json"
    )


def _persist_redaction_payload(deps: PipelineDeps, asset_id: str, payload: dict[str, Any]) -> None:
    """Merge-persist a sweep payload (studio keep-toggles ride the same column)."""
    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        if row is None:
            row = PerceptionIndexRow(asset_id=asset_id)
            s.add(row)
        row.redaction_tracks_json = {**(row.redaction_tracks_json or {}), **payload}
        s.commit()


@contextlib.contextmanager
def _sweep_guard(lock_path: Path, *, stale_s: float = 1800.0, wait_s: float = 900.0):
    """Cross-job sweep mutex.

    A second queued job on the same asset waits for the winner, then reuses
    its result (cache file / persisted payload) — the "swept the same 43-min
    source twice" burn is impossible by construction. A lock older than
    stale_s belongs to a crashed holder and is stolen; waiting longer than
    wait_s degrades to a sweep-skip (canvas boxes still apply, the pipeline
    never breaks on a lost race).
    """
    import time

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_s
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except OSError:
                age = 0.0
            if age > stale_s:
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"redaction sweep lock stuck: {lock_path.name}") from None
            time.sleep(2.0)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)


def stage_redaction(
    asset_id: str,
    source: Path,
    project_id: str,
    deps: PipelineDeps,
    max_regions: int = 8,
    timeline: Any = None,
) -> list:
    """Auto-detect redaction regions — idempotent, pluggable backend (M4).

    POST-PLAN SCOPING LAW (operator mandate): the sweep runs AFTER the LLM
    plan and pays ONLY for footage the edit kept — a 43-min source that keeps
    8 min sweeps 8 min, never 77k raw frames. `timeline` carries the kept
    spans; None sweeps the whole source (legacy direct calls/tests).

    backend "tracked" (issue #14, the default when the YOLO model is present):
    the deterministic 3-pass pipeline — per-span CFR normalize, imgsz-960/
    conf-0.22 YOLO + GMC tracking, One-Euro trajectory repair — persisted as
    a per-frame plan PLUS the static regions the studio canvas already draws.
    backend "modal": YOLOv8-face + plate + ByteTrack on the serverless GPU
    fleet (the GPU lane absorbs the cost, sweep stays full-source).
    backend "yunet": the zero-download CPU sweep (ranges-scoped). A backend
    that cannot run is a skipped stage, never a broken pipeline.

    Cross-job dedupe: results are keyed by (backend, kept-ranges digest) — a
    second queued job waits on the lock, then reuses the winner's sweep (cache
    file + persisted payload), never re-decodes.
    """
    backend = deps.redact_backend or "yunet"
    keep_ranges = _kept_source_ranges(timeline, asset_id) if timeline is not None else None
    ranges_digest = _ranges_digest(backend, keep_ranges)
    if keep_ranges is not None and not keep_ranges:
        deps.log("  [redaction] plan kept no raw footage — sweep skipped entirely")
        return []

    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        payload_existing = dict(row.redaction_tracks_json or {}) if row else {}
        existing = payload_existing.get("regions")
        asset = s.get(MediaAssetRow, asset_id)
    if existing and payload_existing.get("ranges_digest") == ranges_digest:
        deps.log(f"  [redaction] {len(existing)} region(s) already persisted — skip sweep")
        return existing
    frame_w = int(asset.width or 0)
    frame_h = int(asset.height or 0)
    if frame_w <= 0 or frame_h <= 0:
        deps.log("  [redaction] asset geometry unknown — skipping auto sweep")
        return []

    out_dir = deps.settings.output_dir / "pipeline" / project_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _sweep_cache_path(deps, project_id, asset_id, ranges_digest)
    payload_json: dict[str, Any]
    try:
        with _sweep_guard(out_dir / f"redact_sweep_{asset_id[:8]}.lock"):
            # lock re-check: the winner may have finished while we waited
            with deps.sessions() as s:
                row = s.get(PerceptionIndexRow, asset_id)
                won = (row.redaction_tracks_json or {}).get("regions") if row else None
                won_digest = (row.redaction_tracks_json or {}).get("ranges_digest") if row else None
            if won and won_digest == ranges_digest:
                deps.log(
                    f"  [redaction] co-running job already swept this asset — "
                    f"reusing {len(won)} region(s), no re-decode"
                )
                return won
            if cache_path.exists() and cache_path.stat().st_size > 0:
                cached = json.loads(cache_path.read_text())
                _persist_redaction_payload(deps, asset_id, cached)
                deps.log(
                    f"  [redaction] sweep cache hit ({ranges_digest}) — "
                    f"{len(cached.get('regions', []))} region(s) reused, no re-decode"
                )
                return cached.get("regions", [])

            tracked_extra: dict[str, Any] = {}  # per_frame plan + digest (tracked only)
            if backend == "tracked":
                try:
                    tracked_payload = _tracked_redaction_payload(
                        source,
                        project_id,
                        deps,
                        frame_w,
                        frame_h,
                        max_regions,
                        asset_id=asset_id,
                        keep_ranges=keep_ranges,
                    )
                    regions = tracked_payload["regions"]
                    source_tag = tracked_payload["source"]
                    stats = tracked_payload["stats"]
                    tracked_extra = {
                        "per_frame": tracked_payload["per_frame"],
                        "plan_digest": tracked_payload["plan_digest"],
                    }
                except Exception as e:  # model missing, sweep failure — degrade loudly
                    deps.log(
                        f"  [redaction] tracked 3-pass unavailable ({e}) — falling back to YuNet"
                    )
                    backend = "yunet"

            if backend == "modal":
                from director_os.perception.modal_redact import (
                    ModalRedactError,
                    parse_tracks_payload,
                    redact_video_modal,
                    tracks_to_regions,
                )

                try:
                    deps.log("  [redaction] modal lane sweeps the FULL source (GPU fleet)")
                    payload = redact_video_modal(str(source))
                    tracks = parse_tracks_payload(payload)
                    regions = tracks_to_regions(tracks, frame_w, frame_h)
                    geo = payload.get("video", {})
                    source_tag = "modal_yolo_bytetrack"
                    stats = {"tracks": len(tracks), "gpu_video": geo}
                except ModalRedactError as e:
                    deps.log(f"  [redaction] Modal fleet unavailable ({e}) — falling back to YuNet")
                    backend = "yunet"

            if backend == "yunet":
                from director_os.perception.faces import cluster_to_regions, detect_faces

                sweep = detect_faces(source, sample_every_s=1.0, ranges=keep_ranges)
                regions = cluster_to_regions(
                    sweep, max_regions=max_regions, frame_w=frame_w, frame_h=frame_h
                )
                source_tag = f"yunet_autodetect:{sweep.stats.detector}"
                stats = {
                    "frames_sampled": sweep.stats.frames_sampled,
                    "detections": sweep.stats.detections,
                    "kept_ranges": keep_ranges,
                }

            payload_json = {
                "regions": [r.model_dump(mode="json") for r in regions],
                "source": source_tag,
                "stats": stats,
                "ranges_digest": ranges_digest,
                **tracked_extra,
            }
            cache_path.write_text(json.dumps(payload_json))
            _persist_redaction_payload(deps, asset_id, payload_json)
    except TimeoutError as e:
        deps.log(
            f"  [redaction] sweep lock timeout ({e}) — stage skipped, canvas boxes still apply"
        )
        return existing or []
    deps.log(
        f"  [redaction] {len(regions)} region(s) via {source_tag} "
        f"({', '.join(sorted({r.track_id or 'box' for r in regions}))[:4]}...)"
    )
    return payload_json["regions"]


def _resolve_keep_track_ids(deps: PipelineDeps, asset_id: str) -> tuple[set[str], str]:
    """Per-track redaction opt-out (Stage 2/9 contract).

    Sources (union):
    - env `REDACT_KEEP_TRACKS` — comma-separated track ids ("person_1")
    - persisted `redaction_tracks_json["keep_track_ids"]` (Director Studio
      per-track toggles ride the same field)

    (the keep_primary POLICY is applied by the caller, which has the live
    detections)

    PLATES ARE NEVER KEEPABLE: `plate_*` ids are stripped from the keep set —
    a license plate is PII even when its owner is the subject of the video.
    Returns (keep_ids, policy_tag).
    """
    keep: set[str] = set()
    for tok in (os.getenv("REDACT_KEEP_TRACKS") or "").split(","):
        tok = tok.strip()
        if tok:
            keep.add(tok)
    policy = (os.getenv("REDACT_TRACK_POLICY") or "").strip().lower() or "all"
    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
    if row is not None:
        keep |= set((row.redaction_tracks_json or {}).get("keep_track_ids") or [])
    keep = {k for k in keep if not k.startswith("plate_")}
    return keep, policy


def _cfr_range(source: Path, out_dir: Path, start: float, end: float) -> Path:
    """Exact-CFR clip for ONE kept span — cached, video-only (detection input).

    `fps=30` re-encode gives the strict frame cadence the tracker expects;
    only kept% of the source is ever decoded. The name carries the span, so a
    moved cut gets a fresh clip and an unchanged cut reuses its bytes.
    """
    dst = out_dir / f"_sweep_cfr_{int(start * 1000):07d}_{int((end - start) * 1000):06d}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "fps=30",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ],
        f"CFR sweep clip [{start:.1f}s .. {end:.1f}s]",
    )
    return dst


def _sweep_kept_ranges(
    backend: Any,
    source: Path,
    out_dir: Path,
    keep_ranges: list[tuple[float, float]],
) -> tuple[list, list, int]:
    """Detect faces+plates ONLY inside the kept spans; hits re-indexed to
    SOURCE time (t_s += span start) so the per-frame plan keeps the exact
    source-time contract the render bake slices by (`_clip_plan`).

    Each span is detected independently — tracker state resets at a cut,
    because continuity across removed footage is fiction. Returns
    (face_tracks, plate_tracks, n_frames_spanned).
    """
    face_tracks: list = []
    plate_tracks: list = []
    max_end = 0.0
    for start, end in keep_ranges:
        cfr = _cfr_range(source, out_dir, start, end)
        span_faces = backend.detect_faces(cfr)
        span_plates = backend.detect_plates(cfr) if backend.capabilities().get("plates") else []
        for tr in [*span_faces, *span_plates]:
            tr = tr.model_copy(
                update={
                    "hits": [h.model_copy(update={"t_s": round(h.t_s + start, 3)}) for h in tr.hits]
                }
            )
            (plate_tracks if tr.track_id.startswith("plate_") else face_tracks).append(tr)
        max_end = max(max_end, end)
    return face_tracks, plate_tracks, int(round(max_end * 30))


def _tracked_redaction_payload(
    source: Path,
    project_id: str,
    deps: PipelineDeps,
    frame_w: int,
    frame_h: int,
    max_regions: int,
    asset_id: str = "",
    keep_ranges: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Pass 1+2 of the 3-pass pipeline — the plan the render stage bakes in.

    Persists BOTH contracts: the per-frame plan (`per_frame`, frame-indexed)
    for the render bake, and static padded regions (studio canvas). The plan
    digest rides in the payload so a re-sweep invalidates the baked source
    naturally (content-addressed, like the segment cache).

    Per-track opt-out: `keep_track_ids` (Studio/env) + `track_policy` are
    honored BEFORE Pass 2 — kept tracks never enter the trajectory plan, so
    the bake, the static regions and the zero-leak audit all agree on what is
    blurred. Plates are exempt from keeping (PII law).

    Post-plan scoping law: `keep_ranges` set -> detection decodes ONLY the
    kept spans (per-span exact-CFR clips, hits re-indexed to SOURCE time) —
    the per-frame plan keeps the same source-time contract the render bake
    slices by (`_clip_plan`).
    """
    import hashlib

    from director_os.vision_fx.compute.local_onnx import LocalOnnxCpuBackend
    from director_os.vision_fx.compute.tracker import (
        EmptySweepError,
        merge_redaction_tracks,
        tracks_to_regions,
    )
    from director_os.vision_fx.redact_track import normalize_cfr, video_n_frames
    from director_os.vision_fx.trajectory import repair_trajectories

    backend = LocalOnnxCpuBackend()
    caps = backend.capabilities()
    if not caps.get("yolo_model"):
        raise RuntimeError("yolov8n-face model not present (run scripts/fetch_yolov8n_face.py)")

    keep_ids, policy = _resolve_keep_track_ids(deps, asset_id)

    out_dir = deps.settings.output_dir / "pipeline" / project_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks: list = []
    plate_tracks: list = []
    n_frames = 0
    if keep_ranges:
        # post-plan scoping law: only kept spans are decoded; hits re-indexed
        # to SOURCE time so the render contract (`_clip_plan`) stays intact
        tracks, plate_tracks, n_frames = _sweep_kept_ranges(backend, source, out_dir, keep_ranges)
    else:
        cfr_path = normalize_cfr(source, out_dir / "source_cfr.mp4")
        legacy_n = video_n_frames(cfr_path)
        if not legacy_n:
            raise RuntimeError("CFR source frame count unknown — refusing to plan blind")
        n_frames = legacy_n

        tracks = backend.detect_faces(cfr_path)  # imgsz 960, conf 0.22, GMC + buffer 60
        if caps.get("plates"):
            # plates ride the same 3-pass law (issue #12): same sweep, same GMC
            # tracker, same Pass-2 repair. Absent model -> faces-only, never a blocker.
            plate_tracks = backend.detect_plates(cfr_path)

    # keep-primary heuristic BEFORE the merge (id collision handling intact):
    # the primary subject = the face track with the most sampled detections
    # (the person the camera follows — suspect or officer)
    if policy == "keep_primary" and tracks:
        primary = max(tracks, key=lambda t: len(t.hits))
        if not primary.track_id.startswith("plate_"):
            keep_ids.add(primary.track_id)
            deps.log(f"  [redaction] keep_primary policy: {primary.track_id} stays visible")

    if keep_ids:
        tracks = [t for t in tracks if t.track_id not in keep_ids]
        deps.log(
            f"  [redaction] per-track opt-out: {len(keep_ids)} track(s) kept visible "
            f"({sorted(keep_ids)}); plates always blurred"
        )

    try:
        merged = merge_redaction_tracks(tracks, plate_tracks)
    except EmptySweepError as e:
        # A sweep that sees NOTHING is a broken sweep, not an empty scene:
        # shipping it bakes zero redaction while the audit passes vacuously
        # (0 detections = 0 leaks). Fail the stage loud — privacy is not soft.
        # EXCEPT when the operator deliberately kept every detected identity.
        if not keep_ids:
            raise RuntimeError(str(e)) from e
        deps.log("  [redaction] all identities deliberately kept visible (opt-out policy)")
        merged = []

    plan = repair_trajectories(merged, fps=30, frame_w=frame_w, frame_h=frame_h, n_frames=n_frames)
    from director_os.vision_fx.redact_track import assert_zero_leaks

    assert_zero_leaks(plan, merged, 30.0)  # the zero-leak contract, enforced at plan time

    regions = tracks_to_regions(merged, frame_w=frame_w, frame_h=frame_h)[:max_regions]
    plan_json = plan.model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(plan_json, sort_keys=True).encode()).hexdigest()[:8]
    deps.log(
        f"  [redaction] tracked plan: {len(plan.trajectories)} track(s) "
        f"({len(tracks)} face + {len(plate_tracks)} plate), "
        f"{sum(t.n_windows for t in plan.trajectories)} window(s), digest {digest}"
    )
    return {
        "regions": [r.model_dump(mode="json") for r in regions],
        "per_frame": plan_json,
        "plan_digest": digest,
        "source": "tracked_local_gmc_3pass",
        "keep_track_ids": sorted(keep_ids),
        "track_policy": policy,
        "stats": {
            "tracks": len(tracks),
            "plate_tracks": len(plate_tracks),
            "n_frames": n_frames,
            "detector": getattr(backend, "_last_detector_name", "unknown"),
            "plate_detector": getattr(backend, "_last_plate_detector_name", None),
        },
    }


def _tracked_redacted_source(
    deps: PipelineDeps,
    project_id: str,
    source: Path,
    asset_id: str,
    out_dir: Path,
) -> Path | None:
    """Pass 3 (whole-source variant) — bake the plan into ONE redacted file.

    LEGACY fallback kept for full-source renders (REDACT_BAKE=full). The
    default path is per-clip baking (`_tracked_redacted_clip`): only KEPT
    clips get re-encoded, cut-out footage never pays the blur tax.
    """
    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        payload = (row.redaction_tracks_json or {}) if row else {}
    if deps.redact_backend != "tracked":
        return None
    from director_os.vision_fx.redact_track import (
        render_tracked_redaction,
    )

    with_ops = _render_plan_with_ops(deps, project_id, payload, asset_id)
    if with_ops is None:
        return None
    plan, digest = with_ops
    dst = out_dir / f"redacted_source_{digest}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        deps.log(f"  [render] baked redacted source reuse ({dst.name})")
        return dst
    cfr_path = _cfr_source(source, out_dir)
    deps.log(f"  [render] baking tracked redaction source ({digest})...")
    render_tracked_redaction(
        cfr_path,
        plan,
        dst,
        mode="gauss",
        n_frames_expected=plan.n_frames,
    )
    return dst


def _cfr_source(source: Path, out_dir: Path) -> Path:
    """CFR-normalized source, cached (a re-run never re-transcodes the raw)."""
    from director_os.vision_fx.redact_track import normalize_cfr

    dst = out_dir / "source_cfr.mp4"
    if not (dst.exists() and dst.stat().st_size > 0):
        normalize_cfr(source, dst)
    else:
        from director_os.vision_fx.redact_track import probe_cfr

        exact, _fps, n = probe_cfr(dst)
        if not exact or n <= 0:
            normalize_cfr(source, dst)
    return dst


def _render_plan_with_ops(
    deps: PipelineDeps,
    project_id: str,
    payload: dict,
    asset_id: str = "",
) -> tuple[PerFrameRedaction, str] | None:
    """The plan the render bakes: sweep plan + tracked op paths (step 4).

    The sweep's per-frame plan is content-addressed by its digest; every
    ACTIVE tracked blur op rides on top (Pass-2 repair, one smoothing path).
    The merged digest mixes the sweep digest with the op set's identity, so
    adding/revoking an op invalidates exactly the baked clips that cached the
    old program — preview is overlay-only, Export is the only render, and a
    re-render after an op edit re-bakes ONLY the dirty segments (cache law).
    """
    import hashlib

    from director_os.core.db import TimelineOpRow
    from director_os.vision_fx.trajectory import merge_op_params
    from sqlalchemy import select

    per_frame = payload.get("per_frame")
    if not per_frame:
        return None
    plan = PerFrameRedaction.model_validate(per_frame)
    digest = str(payload.get("plan_digest") or "unpinned")

    with deps.sessions() as s:
        rows = (
            s.execute(
                select(TimelineOpRow)
                .where(
                    TimelineOpRow.project_id == project_id,
                    TimelineOpRow.kind == "blur",
                    TimelineOpRow.revoked_at.is_(None),
                )
                .order_by(TimelineOpRow.rev)
            )
            .scalars()
            .all()
        )
        op_rows = [(str(r.id), dict(r.params or {})) for r in rows]
        # (the gate cursor lives in _stamp_export_gate now — it computes the
        # same max op rev itself, so the stamp also happens in default mode)

    merged, changed = merge_op_params(plan, [params for _id, params in op_rows])

    # export-gate stamp: THIS render consumed the op state up to max_rev AND
    # the active timeline version — the Studio's Export button reads both
    # (dirty = fold rev > stamped rev OR active version > stamped version).
    # Version-level edits (trim_clip / trim_audio / split / move / remove)
    # never touch the op fold: without the version stamp a post-export trim
    # would leave the gate "clean" while the disk carries the OLD cut.
    _stamp_export_gate(deps, project_id, asset_id)

    if not changed:
        return plan, digest
    op_fingerprint = hashlib.sha256(
        json.dumps(op_rows, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    merged_digest = f"{digest}+ops{op_fingerprint}"[:24]
    deps.log(
        f"  [render] merged {len(merged.trajectories)} trajectory(ies) "
        f"in the render plan (ops baked, digest {merged_digest})"
    )
    return merged, merged_digest


def _stamp_export_gate(deps: PipelineDeps, project_id: str, asset_id: str | None) -> None:
    """Stamp the export-gate cursor for ANY successful master render.

    Reads (max op rev, active timeline version) at call time — i.e. the exact
    program the render just baked to disk — into the PerceptionIndexRow the
    API gate reads. Without this, a project with NO tracked redaction (the
    default) never stamps and the Export button claims "dirty" forever (CR-002).
    Tracked renders additionally call this from _render_plan_with_ops so the
    stamp lands with the bake; calling again after stage_render is idempotent.
    """
    if not asset_id:
        return
    from director_os.core.db import ProjectRow, TimelineOpRow
    from sqlalchemy import func, select

    with deps.sessions() as s:
        max_rev = int(
            s.execute(
                select(func.max(TimelineOpRow.rev)).where(TimelineOpRow.project_id == project_id)
            ).scalar_one()
            or 0
        )
        project = s.get(ProjectRow, project_id)
        baked_version = (
            int(project.active_otio_version_id)
            if project is not None and project.active_otio_version_id is not None
            else None
        )
        prow = s.get(PerceptionIndexRow, asset_id)
        if prow is None:
            prow = PerceptionIndexRow(asset_id=asset_id)
            s.add(prow)
        prow.redaction_tracks_json = {
            **(prow.redaction_tracks_json or {}),
            "baked_ops_rev": max_rev,
            "baked_timeline_version": baked_version,
        }
        s.commit()


def _clip_plan(plan, start_s: float, dur_s: float):
    """Per-clip PerFrameRedaction: boxes of [start, start+dur) re-indexed to clip-local frames."""
    from director_os.vision_fx.trajectory import (
        PerFrameRedaction,
        RedactionTrajectory,
        TrackedBox,
    )

    fps = float(plan.fps)
    f0 = int(round(start_s * fps))
    f1 = int(round((start_s + dur_s) * fps))
    trajectories: list[RedactionTrajectory] = []
    for tr in plan.trajectories:
        boxes = [
            TrackedBox(
                frame_idx=b.frame_idx - f0,
                t_s=round(b.frame_idx / fps - start_s, 3),
                x=b.x,
                y=b.y,
                w=b.w,
                h=b.h,
                bridged=b.bridged,
                lead=b.lead,
            )
            for b in tr.boxes
            if f0 <= b.frame_idx < f1
        ]
        if boxes:
            trajectories.append(RedactionTrajectory(track_id=tr.track_id, boxes=boxes))
    return PerFrameRedaction(
        fps=plan.fps,
        n_frames=max(1, f1 - f0),
        frame_w=plan.frame_w,
        frame_h=plan.frame_h,
        trajectories=trajectories,
    )


def _tracked_redacted_clip(
    deps: PipelineDeps,
    source: Path,
    out_dir: Path,
    plan,
    plan_digest: str,
    start_s: float,
    dur_s: float,
) -> Path | None:
    """Pass 3 PER KEPT CLIP — blur baked into a clip-local file, cache-addressed.

    The expensive property: a 60-180 min FOIA packet where Stage 6 keeps 15%
    pays blur-bake for 15%, never 100%. Cache name carries the plan digest +
    the exact clip range, so a re-swept plan or a moved cut invalidates only
    the affected clips.
    """
    from director_os.core.ffmpeg_args import exact_slice_args
    from director_os.vision_fx.redact_track import render_tracked_redaction, video_n_frames

    clip_plan = _clip_plan(plan, start_s, dur_s)
    dst = (
        out_dir
        / f"redacted_clip_{plan_digest}_{int(start_s * 1000):07d}_{int(dur_s * 1000):06d}.mp4"
    )
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    cfr_path = _cfr_source(source, out_dir)
    # exact clip extraction off the CFR source (re-encode: stream copy would
    # cut on keyframes and break the frame-index contract)
    clip_src = out_dir / f"_clip_{int(start_s * 1000):07d}_{int(dur_s * 1000):06d}.mp4"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *exact_slice_args(start_s, start_s + dur_s),
            "-i",
            str(cfr_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(clip_src),
        ],
        what="kept-clip extraction",
    )
    n = video_n_frames(clip_src)
    if not n:
        clip_src.unlink(missing_ok=True)
        raise RuntimeError(f"clip extraction frame count unknown at {start_s:.2f}s")
    render_tracked_redaction(
        clip_src,
        clip_plan,
        dst,
        mode="gauss",
        n_frames_expected=min(n, clip_plan.n_frames),
    )
    clip_src.unlink(missing_ok=True)
    return dst


def _run_ffmpeg(cmd: list[str], what: str) -> None:
    import subprocess as _sp

    r = _sp.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed: {r.stderr[-400:]}")


def _baked_clip_sources(
    deps: PipelineDeps,
    project_id: str,
    source: Path,
    asset_id: str,
    timeline,
    out_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Per-clip baked sources + digests for every KEPT raw clip (Pass 3, lazy).

    Returns ({clip_key: path}, {clip_key: source_digest}) for compile_timeline.
    A bake failure degrades that clip to the raw source — loudly.
    """
    from director_os.otio_engine.contract import RawClipMeta, parse_clip_meta
    from director_os.otio_engine.timeline import (
        MASTER_TRACK,
        is_gap,
        item_duration_s,
        track_by_name,
    )

    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        payload = (row.redaction_tracks_json or {}) if row else {}
    if deps.redact_backend != "tracked":
        return {}, {}
    with_ops = _render_plan_with_ops(deps, project_id, payload, asset_id)
    if with_ops is None:
        return {}, {}
    plan, digest = with_ops

    master = track_by_name(timeline, MASTER_TRACK)
    if master is None:
        return {}, {}
    sources: dict[str, str] = {}
    digests: dict[str, str] = {}
    for item in master:
        dur = item_duration_s(item)
        if dur <= 0 or is_gap(item):
            continue
        try:
            meta = parse_clip_meta(item)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(meta, RawClipMeta) or meta.asset_id != asset_id:
            continue
        start_s = item.source_range.start_time.to_seconds() if item.source_range else 0.0
        clip_key = f"{asset_id}@{start_s:.3f}+{dur:.3f}"
        try:
            baked = _tracked_redacted_clip(deps, source, out_dir, plan, digest, start_s, dur)
        except Exception as e:  # noqa: BLE001 — one bad clip never kills the render
            deps.log(f"  [render] clip bake failed at {start_s:.2f}s ({e}) — raw fallback")
            continue
        sources[clip_key] = str(baked)
        digests[clip_key] = f"{digest}@{start_s:.3f}+{dur:.3f}"
    if sources:
        deps.log(
            f"  [render] baked {len(sources)} kept clip(s) with tracked redaction (digest {digest})"
        )
    return sources, digests


def stage_vision_enrichment(asset_id: str, source: Path, deps: PipelineDeps) -> None:
    """M4 vision stack enrichment — every product optional, every failure soft.

    Persists into the existing free-form JSON columns (no migration):
    - `shots_json["frame_quality"]`    — CPU blur/darkness pre-filter (always on)
    - `shots_json["audio_events"]`     — YAMNet CPU event spans (siren/gunshot/...)
    - `shots_json["visual_timeline"]`  — lite-tier contact-sheet indexing (needs key)
    - `redaction_tracks_json["timestamp_ocr"]` — AXON overlay OCR (needs key/fleet)
    """
    from dataclasses import asdict

    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        if row is None:
            row = PerceptionIndexRow(asset_id=asset_id)
            s.add(row)
        shots = dict(row.shots_json or {})
        red = dict(row.redaction_tracks_json or {})

    if "frame_quality" not in shots:
        try:
            from director_os.perception.frame_quality import score_frame_quality

            report = score_frame_quality(str(source), sample_every_s=2.0)
            shots["frame_quality"] = {
                "sample_every_s": report.sample_every_s,
                "usable_ratio": round(report.usable_ratio, 3),
                "unusable_spans": [[round(a, 2), round(b, 2)] for a, b in report.unusable_spans],
                "frames": [asdict(f) for f in report.frames[:600]],
            }
            deps.log(
                f"  [vision] frame quality: {report.usable_ratio:.0%} usable, "
                f"{len(report.unusable_spans)} bad span(s)"
            )
        except Exception as e:  # noqa: BLE001 — enrichment never kills the pipeline
            deps.log(f"  [vision] frame quality skipped: {e}")

    if "audio_events" not in shots:
        # YAMNet on CPU — local, $0, no key. Sirens/gunshots/shouting/glass as
        # explicit spans (issue #12): the speech-gated meter refuses to treat
        # them as program loudness, retention can anchor on them.
        try:
            from director_os.perception.audio_events import make_audio_event_backend

            sweep = make_audio_event_backend().detect_events(source)
            shots["audio_events"] = sweep.model_dump(mode="json")
            kinds = sweep.kinds()
            deps.log(
                f"  [vision] audio events: {len(sweep.events)} span(s)"
                + (f" — {', '.join(sorted(kinds))}" if kinds else " — quiet scene")
            )
        except Exception as e:  # noqa: BLE001
            deps.log(f"  [vision] audio events skipped: {e}")

    if deps.settings.vision_api_key and "visual_timeline" not in shots:
        try:
            from director_os.perception.vision_index import (
                build_contact_sheets,
                index_timeline,
                make_vision_clients,
            )

            sheets_dir = deps.settings.output_dir / "pipeline" / asset_id / "sheets"
            sheets = build_contact_sheets(str(source), sheets_dir, window_s=30.0)
            index_adapter, _ = make_vision_clients(deps.settings.vision_api_key)
            insights = index_timeline(index_adapter, sheets, window_s=30.0)
            shots["visual_timeline"] = {
                "model": index_adapter._client.model  # type: ignore[attr-defined]
                if hasattr(index_adapter._client, "model")
                else "fake",
                "sheets": len(sheets),
                "insights": [i.model_dump(mode="json") for i in insights],
            }
            deps.log(
                f"  [vision] timeline indexed: {len(sheets)} sheet(s) -> {len(insights)} insight(s)"
            )
        except Exception as e:  # noqa: BLE001
            deps.log(f"  [vision] contact-sheet indexing skipped: {e}")

    # Tier 2 — Frames-on-Demand QA: the FLAGSHIP model verifies the risky
    # moments the lite index flagged (full-res frames around the timestamp,
    # overlay/zoom safety check). Capped so the spend stays bounded; every
    # failure is soft — the pipeline never depends on the tier-2 pass.
    if deps.settings.vision_api_key and "fod_qa" not in shots:
        try:
            from director_os.perception.vision import SceneInsight
            from director_os.perception.vision_index import (
                frames_on_demand,
                make_vision_clients,
                qa_placement,
                suggest_fod_timestamps,
            )

            tl = shots.get("visual_timeline") or {}
            insights = [SceneInsight.model_validate(i) for i in tl.get("insights", [])]
            max_fod = int(os.getenv("VISION_FOD_MAX", "3"))
            picks = suggest_fod_timestamps(insights)[:max_fod]
            if picks:
                _, qa_adapter = make_vision_clients(deps.settings.vision_api_key)
                fod_dir = deps.settings.output_dir / "pipeline" / asset_id / "fod"
                runs: list[dict[str, Any]] = []
                for t in picks:
                    frames = frames_on_demand(str(source), t, out_dir=fod_dir)
                    found = qa_placement(
                        qa_adapter, frames, "overlay/zoom placement safety check", t
                    )
                    runs.append({"t_s": t, "insights": [i.model_dump(mode="json") for i in found]})
                shots["fod_qa"] = {
                    "model": qa_adapter._client.model  # type: ignore[attr-defined]
                    if hasattr(qa_adapter._client, "model")
                    else "fake",
                    "runs": runs,
                }
                deps.log(
                    f"  [vision] frames-on-demand QA: {len(runs)} run(s) at "
                    f"{[r['t_s'] for r in runs]}"
                )
            else:
                deps.log("  [vision] frames-on-demand: no risky moments picked")
        except Exception as e:  # noqa: BLE001
            deps.log(f"  [vision] frames-on-demand QA skipped: {e}")

    # Overlay OCR runs on EVERY asset: the LOCAL free PaddleOCR lane first
    # (cost law — CPU, $0, no key), Gemini only as the fallback when paddleocr
    # is not installed and the vision key exists. No stamp is a legit answer.
    if "timestamp_ocr" not in red:
        try:
            from director_os.perception.timestamp_ocr import (
                paddleocr_available,
                read_bodycam_timestamp,
            )

            client = None
            backend = "auto"
            if paddleocr_available():
                backend = "paddle"
            elif deps.settings.vision_api_key:
                from director_os.perception.vision import GeminiVisionClient

                client = GeminiVisionClient(deps.settings.vision_api_key)
                backend = "gemini"
            stamp = read_bodycam_timestamp(str(source), backend=backend, client=client)
            if stamp is not None:
                red["timestamp_ocr"] = {
                    "engine": backend,
                    "raw_text": stamp.raw_text,
                    "iso": stamp.iso,
                    "device": stamp.device,
                    "tz": stamp.tz,
                    "location_text": stamp.location_text,
                    "confidence": stamp.confidence,
                }
                location = f" @ {stamp.location_text}" if stamp.location_text else ""
                deps.log(
                    f"  [vision] overlay OCR ({backend}): {stamp.iso} ({stamp.device}){location}"
                )
            else:
                red["timestamp_ocr"] = {"found": False, "engine": backend}
                deps.log(f"  [vision] overlay OCR ({backend}): no timestamp found")
        except Exception as e:  # noqa: BLE001
            deps.log(f"  [vision] timestamp OCR skipped: {e}")

    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        if row is None:
            row = PerceptionIndexRow(asset_id=asset_id)
            s.add(row)
        row.shots_json = shots
        row.redaction_tracks_json = red
        s.commit()


def _persist_timeline_version(
    timeline, project_id: str, deps: PipelineDeps, *, created_by: str, note: str
) -> int:
    """Write a new timeline_versions row + advance the active pointer."""
    with deps.sessions() as s:
        last = s.execute(
            select(TimelineVersionRow.version_num)
            .where(TimelineVersionRow.project_id == project_id)
            .order_by(TimelineVersionRow.version_num.desc())
            .limit(1)
        ).scalar_one_or_none()
        version = (last or 0) + 1
        s.add(
            TimelineVersionRow(
                project_id=project_id,
                version_num=version,
                otio_json=to_json_dict(timeline),
                created_by=created_by,
                checkpoint_note=note[:500],
            )
        )
        project = s.get(ProjectRow, project_id)
        project.active_otio_version_id = version
        _set_status(s, project_id, ProjectStatus.planned)
        s.commit()
    if deps.tracker is not None:
        deps.tracker.timeline(version, note)  # the Studio NLE refetches on this
    return version


def _load_dossier(project_id: str, deps: PipelineDeps) -> dict | None:
    with deps.sessions() as s:
        row = s.get(CaseDossierRow, project_id)
        return dict(row.dossier_json) if row and row.dossier_json else None


def stage_dossier(project_id: str, deps: PipelineDeps) -> dict | None:
    """Extract every pending/failed case source and re-synthesize the dossier.

    Auto-discovery: a `source_page` source (the video's own page, e.g.
    Wikimedia/COPA) contributes Hop-1 links; each discovered pdf/url becomes a
    fresh `pending` source extracted in the same pass (Hop 2). The
    synthesized CaseDossier is persisted to `case_dossiers` for both the
    Director prompt and the Studio inspector. Adding a source later and
    calling this again refreshes the dossier WITHOUT re-running ASR.
    """
    from director_os.core.models import CaseSource
    from director_os.perception.case_dossier import extract_case_source, synthesize_dossier

    settings = deps.settings
    fetch = deps.fetch
    complete_json = deps.complete_json
    if complete_json is None:
        try:
            from director_os.agent_runtime.llm import build_backend_chain
            from director_os.agent_runtime.registry import ToolRegistry

            chain = build_backend_chain(ToolRegistry())
            complete_json = lambda sys_p, user_p: chain.complete_json(sys_p, user_p)  # noqa: E731
        except Exception:
            complete_json = None  # heuristic synthesis (no LLM rung reachable)

    for _hop in range(2):  # hop1: attached sources; hop2: discovered links
        with deps.sessions() as s:
            rows = (
                s.execute(
                    select(CaseSourceRow)
                    .where(CaseSourceRow.project_id == project_id)
                    .where(CaseSourceRow.status.in_(["pending", "failed", "extracting"]))
                    .order_by(CaseSourceRow.created_at)
                )
                .scalars()
                .all()
            )
        if not rows:
            break
        for row in rows:
            with deps.sessions() as s:
                row.status = "extracting"
                s.commit()
            source = CaseSource(
                id=row.id,
                project_id=project_id,
                kind=row.kind,  # type: ignore[arg-type]
                label=row.label or "",
                uri_or_storage_key=row.uri_or_storage_key or "",
                text=row.text or "",
                extracted_text="",
            )
            try:
                kwargs: dict = {"log": deps.log}
                if fetch is not None:
                    kwargs["fetch"] = fetch
                if settings.vision_api_key:
                    kwargs["vision_api_key"] = settings.vision_api_key
                    kwargs["vision_model"] = settings.vision_model or "gemini-3.5-flash-lite"
                text, discovered = extract_case_source(source, **kwargs)
                with deps.sessions() as s:
                    row = s.get(CaseSourceRow, row.id)
                    row.extracted_text = text[:200_000]
                    row.status = "ready"
                    row.error = None
                    s.commit()
                deps.log(f"  [dossier] {row.kind} '{row.label[:40]}' ready ({len(text)} chars)")
                for d in discovered:
                    with deps.sessions() as s:
                        dup = s.execute(
                            select(CaseSourceRow)
                            .where(CaseSourceRow.project_id == project_id)
                            .where(CaseSourceRow.uri_or_storage_key == d["url"])
                        ).scalar_one_or_none()
                        if dup is None:
                            s.add(
                                CaseSourceRow(
                                    project_id=project_id,
                                    kind=d["kind"],
                                    label=d["label"],
                                    uri_or_storage_key=d["url"],
                                    status="pending",
                                )
                            )
                            s.commit()
                            deps.log(f"  [dossier] auto-discovered {d['kind']}: {d['url'][:80]}")
            except Exception as e:
                with deps.sessions() as s:
                    row = s.get(CaseSourceRow, row.id)
                    row.status = "failed"
                    row.error = str(e)[:1000]
                    s.commit()
                deps.log(f"  [dossier] {row.kind} '{row.label[:40]}' FAILED: {e}")

    with deps.sessions() as s:
        rows = (
            s.execute(
                select(CaseSourceRow)
                .where(CaseSourceRow.project_id == project_id)
                .order_by(CaseSourceRow.created_at)
            )
            .scalars()
            .all()
        )
        sources = [
            CaseSource(
                id=r.id,
                project_id=project_id,
                kind=r.kind,  # type: ignore[arg-type]
                label=r.label or "",
                uri_or_storage_key=r.uri_or_storage_key or "",
                text=r.text or "",
                extracted_text=r.extracted_text or "",
                status=r.status,  # type: ignore[arg-type]
            )
            for r in rows
        ]
    if not sources:
        return _load_dossier(project_id, deps)

    dossier = synthesize_dossier(sources, None, complete_json=complete_json)
    payload = dossier.model_dump(mode="json")
    with deps.sessions() as s:
        row = s.get(CaseDossierRow, project_id)
        if row is None:
            row = CaseDossierRow(project_id=project_id, dossier_json=payload)
            s.add(row)
        else:
            row.dossier_json = payload
        s.commit()
    deps.log(
        f"  [dossier] synthesized: subject={dossier.subject_name} "
        f"facts={len(dossier.key_facts)} charges={len(dossier.charges)} "
        f"verdict={'yes' if dossier.verdict_outcome else 'none'}"
    )
    return payload


def _apply_audio_mode(timeline, deps: PipelineDeps) -> None:
    """Rewrite raw-clip audio_mode to the configured law (2-bus etc.)."""
    from director_os.otio_engine.timeline import MASTER_TRACK, is_gap, track_by_name

    mode_map = {
        "loudnorm_48k": "loudnorm_48k",
        "isolate_2bus": "isolate_2bus",
        "silence": "silence",
    }
    mode = mode_map.get(deps.audio_mode, "loudnorm_48k")
    master = track_by_name(timeline, MASTER_TRACK)
    if master is None:
        return
    for item in master:
        if is_gap(item):
            continue
        meta = item.metadata.get("director_os") or {}
        if meta.get("kind") == "raw":
            meta["audio_mode"] = mode


def _downgrade_2bus_audio_mode(timeline) -> int:
    """isolate_2bus -> loudnorm_48k on every raw master clip; returns the count.

    The renderer hard-refuses an isolate_2bus clip without its enhanced master
    WAV (the no-silent-fallback law) — a machine with no neural enhancer would
    die at audio_prep and never render anything. This downgrade keeps the
    render HONEST instead of dead: leveled raw audio, named in the receipt,
    and re-running after `scripts/fetch_deep_filter.py` restores 2-bus without
    redoing finished stages. In-memory only — the persisted version keeps the
    operator's audio law for the next, upgraded run.
    """
    from director_os.otio_engine.timeline import MASTER_TRACK, is_gap, track_by_name

    master = track_by_name(timeline, MASTER_TRACK)
    if master is None:
        return 0
    n = 0
    for item in master:
        if is_gap(item):
            continue
        meta = dict(item.metadata.get("director_os") or {})
        if meta.get("audio_mode") == "isolate_2bus":
            meta["audio_mode"] = "loudnorm_48k"
            item.metadata.update({"director_os": meta})
            n += 1
    return n


def _documentary_scaffold(
    asset_id: str, dur: float, fps: int, w: int, h: int, audio_mode: str
) -> Any:
    """Full-length raw clip + overlay gap track — the Phase B canvas."""
    spec = TimelineSpec(
        name=f"doc_scaffold_{asset_id[:8]}",
        fps=fps,
        width=w,
        height=h,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw_scaffold",
                        source_start_s=0.0,
                        dur_s=dur,
                        meta=RawClipMeta(asset_id=asset_id, audio_mode=audio_mode),  # type: ignore[arg-type]
                    )
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=dur)]),
        ],
    )
    return build_timeline(spec)


def _director_backend(deps: PipelineDeps):
    """Live routing chain from env, or the injected test factory; None = fallback."""
    if deps.director_backend_factory is not None:
        return deps.director_backend_factory()
    try:
        from director_os.agent_runtime.llm import build_backend_chain
        from director_os.agent_runtime.registry import ToolRegistry

        return build_backend_chain(ToolRegistry())
    except Exception as e:
        deps.log(f"  [planning] no LLM rung ({e}) — deterministic 6-act fallback")
        return None


def _plan_documentary(
    asset_id: str,
    transcript: dict,
    project_id: str,
    deps: PipelineDeps,
    *,
    dur: float,
    fps: int,
    card_fields: dict,
) -> tuple[Any, int]:
    """LLM 6-act story-map path with deterministic fallback (>= 15 min sources)."""
    from director_os.agent_runtime.checkpoint import CheckpointStore
    from director_os.agent_runtime.director_session import DirectorSession
    from director_os.agent_runtime.editing_tools import DirectorToolbox, build_director_registry
    from director_os.agent_runtime.fallback_cut import build_fallback_documentary
    from director_os.agent_runtime.plan_invariants import audit_retention
    from director_os.agent_runtime.story_map import build_story_map

    dossier = _load_dossier(project_id, deps) or stage_dossier(project_id, deps)
    buckets = build_story_map(transcript, source_duration_s=dur, shots=[], audio_events=[])
    deps.log(
        f"  [planning] story map: {len(buckets)} minute-buckets, "
        f"{sum(1 for b in buckets if b.has_signal)} with signal"
    )

    mode_map = {
        "loudnorm_48k": "loudnorm_48k",
        "isolate_2bus": "isolate_2bus",
        "silence": "silence",
    }
    audio_mode = mode_map.get(deps.audio_mode, "loudnorm_48k")

    backend = _director_backend(deps)
    if backend is not None:
        store = CheckpointStore(deps.sessions)
        scaffold = _documentary_scaffold(asset_id, dur, fps, deps.video.w, deps.video.h, audio_mode)
        toolbox = DirectorToolbox(scaffold, project_id, (deps.video.w, deps.video.h))
        registry = build_director_registry(toolbox)
        session = DirectorSession(
            registry, store, project_id, backend, source_duration_s=dur, transcript=transcript
        )
        plan, plan_err = session.plan_macro(buckets, dossier)
        if plan is None:
            deps.log(f"  [planning] Phase A failed ({plan_err}) — deterministic fallback")
        else:
            label = getattr(backend, "model", getattr(backend, "provider", "llm"))
            deps.log(
                f"  [planning] Phase A ok via {label}: {len(plan.acts)} acts, "
                f"{plan.raw_count} raw, {plan.narration_count} narration, "
                f"{plan.total_dur_s:.0f}s"
            )
            timeline, report = session.execute(plan, scaffold)
            retention = audit_retention(timeline, dur)
            if retention.ok and report.steps_failed == 0:
                _apply_audio_mode(timeline, deps)
                comp = report.composition
                total_s = comp.final_duration_s if comp else 0.0
                version = _persist_timeline_version(
                    timeline,
                    project_id,
                    deps,
                    created_by="director_agent",
                    note=(
                        f"LLM 6-act doc ({label}): {total_s:.0f}s, "
                        f"turns={report.turns}, steps={report.steps_ok}"
                    ),
                )
                deps.log(
                    f"  [planning] Director timeline v{version}: "
                    f"{(comp.final_duration_s if comp else 0):.1f}s, "
                    f"{(comp.raw_segment_count if comp else 0)} raw, "
                    f"{(comp.narrator_segment_count if comp else 0)} narration"
                )
                return timeline, version
            deps.log(
                f"  [planning] LLM plan rejected (steps_failed={report.steps_failed}, "
                f"retention_ok={retention.ok}) — deterministic fallback"
            )

    spec = build_fallback_documentary(
        asset_id,
        dur,
        transcript,
        fps=fps,
        width=deps.video.w,
        height=deps.video.h,
        project_id=project_id,
    )
    timeline = build_timeline(spec)
    _apply_audio_mode(timeline, deps)
    retention = audit_retention(timeline, dur)
    if not retention.ok:
        raise PipelineError(
            f"deterministic fallback failed retention: {[c.detail for c in retention.failures]}"
        )
    version = _persist_timeline_version(
        timeline,
        project_id,
        deps,
        created_by="assembler_agent",
        note=f"deterministic 6-act fallback ({dur / 60:.0f}min source)",
    )
    deps.log(f"  [planning] fallback timeline v{version} (retention ok)")
    return timeline, version


def stage_planning(
    asset_id: str,
    transcript: dict,
    project_id: str,
    deps: PipelineDeps,
    *,
    force_replan: bool = False,
) -> tuple[Any, int]:
    """Stage 6 — auto-cut v1 for short sources; the 2-phase story-map
    documentary for >= 15 min sources (Phase A macro plan over the FULL
    story index + dossier, Phase B act-by-act tool expansion + refinement
    loop, deterministic 6-act fallback when no LLM rung is reachable).

    `force_replan=True` invalidates the active-version shortcut (a refreshed
    CaseDossier re-runs Stage 6 without touching ingest or ASR).
    """
    with deps.sessions() as s:
        project = s.get(ProjectRow, project_id)
        if project.active_otio_version_id is not None and not force_replan:
            version = project.active_otio_version_id
            row = s.execute(
                select(TimelineVersionRow).where(
                    TimelineVersionRow.project_id == project_id,
                    TimelineVersionRow.version_num == version,
                )
            ).scalar_one()
            deps.log(f"  [planning] reusing timeline v{version}")
            return _timeline_from_json(row.otio_json), version

    asset = None
    with deps.sessions() as s:
        asset = s.get(MediaAssetRow, asset_id)
    dur = asset.duration_s
    fps = int(round(asset.fps))

    # M4: the AXON overlay OCR populates the intro card when the vision stack
    # read one (e.g. "AXON BODY 3 — 2022-07-04T18:11:55-04:00"); a street-sign
    # read rides along for the location line; otherwise the deterministic v1
    # subtitle stays.
    card_fields = {"title": "CASE FILE", "subtitle": "DirectorOS auto-cut v1"}
    with deps.sessions() as s:
        prow = s.get(PerceptionIndexRow, asset_id)
        ocr = (prow.redaction_tracks_json or {}).get("timestamp_ocr") if prow else None
        if ocr and ocr.get("iso"):
            device = ocr.get("device") or "BODYCAM"
            subtitle = f"{device} — {ocr['iso']}"
            if ocr.get("location_text"):
                subtitle += f" — {ocr['location_text']}"
            card_fields["subtitle"] = subtitle

    # >= 15 min sources: the 6-act documentary owns the edit — never a
    # 1-minute slice (Eustis postmortem), never an 8s-grained full dump
    if dur >= DOCUMENTARY_MIN_SOURCE_S:
        return _plan_documentary(
            asset_id, transcript, project_id, deps, dur=dur, fps=fps, card_fields=card_fields
        )

    n_segments = max(1, math.ceil(dur / AUTO_CUT_MAX_SEGMENT_S))
    seg_len = dur / n_segments

    _AUDIO_MODE_BY_ENV: dict[str, Any] = {
        "loudnorm_48k": "loudnorm_48k",
        "isolate_2bus": "isolate_2bus",
        "silence": "silence",
    }
    master_items: list[ClipSpec | GapSpec] = [
        ClipSpec(
            name=f"raw_{i + 1}",
            source_start_s=i * seg_len,
            dur_s=seg_len,
            meta=RawClipMeta(
                asset_id=asset_id,
                # unknown env values fall back to the legacy loudnorm law
                audio_mode=_AUDIO_MODE_BY_ENV.get(deps.audio_mode, "loudnorm_48k"),
            ),
        )
        for i in range(n_segments)
    ]
    overlay_items: list[ClipSpec | GapSpec] = [GapSpec(dur_s=1.0)] if dur > 1.0 else []
    if overlay_items:
        overlay_items.append(
            ClipSpec(
                name="card_location",
                source_start_s=0.0,
                dur_s=3.0,
                meta=CardMeta(template="location_intro", fields=card_fields),
            )
        )

    spec = TimelineSpec(
        name=f"autocut_{project_id[:8]}",
        fps=fps,
        width=deps.video.w,
        height=deps.video.h,
        tracks=[
            TrackSpec(name="V1_master", items=master_items),
            TrackSpec(name="V2_overlay_cards", items=overlay_items or []),
        ],
    )
    timeline = build_timeline(spec)

    with deps.sessions() as s:
        last = s.execute(
            select(TimelineVersionRow.version_num)
            .where(TimelineVersionRow.project_id == project_id)
            .order_by(TimelineVersionRow.version_num.desc())
            .limit(1)
        ).scalar_one_or_none()
        version = (last or 0) + 1
        s.add(
            TimelineVersionRow(
                project_id=project_id,
                version_num=version,
                otio_json=to_json_dict(timeline),
                created_by="assembler_agent",
                checkpoint_note="auto-cut v1 (deterministic; Director LLM lands M2 close-out)",
            )
        )
        project = s.get(ProjectRow, project_id)
        project.active_otio_version_id = version
        _set_status(s, project_id, ProjectStatus.planned)
        s.commit()
    deps.log(f"  [planning] timeline v{version}: {n_segments} segments, {dur:.1f}s")
    return timeline, version


def _timeline_from_json(otio_json: dict):
    import json

    import opentimelineio.adapters as ot_adapters

    return ot_adapters.read_from_string(json.dumps(otio_json), "otio_json")


def stage_subtitles(timeline, transcript: dict, project_id: str, deps: PipelineDeps) -> Path:
    """Persist the MASTER-CLOCK cue store — the v6 "caption truth" law.

    The cue list is built on the EDITED TIMELINE (what the viewer sees), never
    on the raw source clock: narrator cues from the TTS word sidecars rebased
    per narration block, raw-speech cues remapped through each kept raw clip's
    source->master window, and narration b-roll blocks NEVER carry raw captions
    (narrator-on-broll isolation). Building cues off the 43-minute source and
    burning them on a ~10-minute master was the v5 regression the v6 commit
    fixed at the root — the whole store now shares ONE builder with post-fx
    (`build_master_caption_cues`), so the Studio grid, the program monitor and
    the final burn all read the same truth.

    Word source: the PROOFREAD product when present (jargon-corrected,
    Officer/Suspect role-tagged — stage_proofread), else the raw ASR words.
    Output lands next to the timeline: output/pipeline/<project>/subtitles.{json,ass,srt}.
    The JSON cue store is what the Director Studio subtitle grid reads/edits.
    """
    import json as _json

    from director_os.subtitles.master_cues import build_master_caption_cues, narration_word_stamps
    from director_os.subtitles.store import save_cues

    sub_dir = deps.settings.output_dir / "pipeline" / project_id

    # narration word stamps ride the stage_narration manifest (persisted BEFORE
    # this stage runs — the pipeline order guarantees narration + retime first)
    manifest = sub_dir / "narration" / "clips.json"
    narration_clips: list[dict] = []
    if manifest.exists():
        try:
            narration_clips = list(_json.loads(manifest.read_text()).get("clips") or [])
        except (ValueError, OSError):
            narration_clips = []

    narration_words = narration_word_stamps(narration_clips)
    beats, counts = build_master_caption_cues(timeline, transcript, narration_words)
    save_cues(sub_dir, beats)
    out = sub_dir / "subtitles.ass"
    deps.log(
        f"  [subtitles] {counts['total']} master-clock cue(s) "
        f"({counts['narration']} narration / {counts['raw']} raw) -> subtitles.ass + srt + json"
    )
    return out


def _project_redaction_regions(sessions: Any, asset_id: str) -> list[BlurRegion]:
    """Studio-drawn blur boxes (perception_index.redaction_tracks_json) -> specs.

    Tolerant reader: the studio API persists the flat studio shape
    ({x, y, w, h, sigma, id, ...}); the canonical BlurRegion shape
    ({bbox: {x, y, w, h}, scope, mode}) is accepted too, including the tracked
    (t_start_s/t_end_s) windows the Modal ByteTrack sweep writes. A region that
    fails validation is skipped, never fatal to the render.
    """
    from director_os.core.models import BBox

    with sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
    regions: list[BlurRegion] = []
    for item in (row.redaction_tracks_json or {}).get("regions", []) if row else []:
        bbox = item.get("bbox") or {k: item.get(k, 0) for k in ("x", "y", "w", "h")}
        try:
            regions.append(
                BlurRegion(
                    scope=item.get("scope", "global"),
                    bbox=BBox(
                        x=int(bbox.get("x", 0)),
                        y=int(bbox.get("y", 0)),
                        w=int(bbox.get("w", 0)),
                        h=int(bbox.get("h", 0)),
                    ),
                    mode=item.get("mode", "static"),
                    sigma=int(item.get("sigma", 24)),
                    track_id=item.get("id") or item.get("track_id"),
                    t_start_s=item.get("t_start_s"),
                    t_end_s=item.get("t_end_s"),
                )
            )
        except (TypeError, ValueError):
            continue  # a corrupt region must never kill the render
    return regions


def stage_render(
    timeline,
    project_id: str,
    source: Path,
    asset_id: str,
    deps: PipelineDeps,
    enhanced_audio: dict[str, Path] | None = None,
    narration_audio: dict[str, Path] | None = None,
) -> tuple[RenderReport, Path]:
    """Compile + segment-cached render (content-addressed idempotency).

    REDACT_MODE=tracked + a persisted per-frame plan -> each KEPT clip is cut
    from its own baked redacted file (Pass 3 per-clip, content-addressed by
    plan digest + range) so cut-out footage never pays the blur tax; a clip
    bake failure degrades that clip to the raw source, loudly. Static blur
    regions still ride the filtergraph for studio-drawn boxes.
    `enhanced_audio` feeds the 2-bus speech-isolation masters
    (audio_mode=isolate_2bus); `narration_audio` feeds the persisted TTS
    voiceover clips (stage_narration) onto the narration segments.
    """
    out_dir = deps.settings.output_dir / "pipeline" / project_id
    save_otio(timeline, out_dir / "timeline.otio")
    cache = SegmentCache(deps.settings.segments_dir)
    with deps.sessions() as s:
        asset = s.get(MediaAssetRow, asset_id)
    geometry = (
        {asset_id: (asset.width, asset.height)}
        if asset and asset.width > 0 and asset.height > 0
        else None
    )
    clip_sources: dict[str, str] = {}
    clip_digests: dict[str, str] = {}
    baked_source: Path | None = None
    if os.getenv("REDACT_BAKE", "").strip().lower() != "full":
        try:
            clip_sources, clip_digests = _baked_clip_sources(
                deps, project_id, source, asset_id, timeline, out_dir
            )
        except Exception as e:  # a bake failure degrades to raw + static regions, loudly
            deps.log(f"  [render] per-clip tracked bake failed ({e}) — raw source fallback")
    if not clip_sources:
        try:
            baked_source = _tracked_redacted_source(deps, project_id, source, asset_id, out_dir)
        except Exception as e:  # a bake failure degrades to static regions, loudly
            deps.log(f"  [render] tracked redaction bake failed ({e}) — using raw source")
    dag = compile_timeline(
        timeline,
        assets={asset_id: str(baked_source or source)},
        cache=cache,
        video=deps.video,
        audio=deps.audio,
        out_dir=out_dir,
        audio_target_lufs=deps.settings.target_lufs,
        global_blur_regions=_project_redaction_regions(deps.sessions, asset_id),
        asset_geometry=geometry,
        enhanced_audio=enhanced_audio or None,
        clip_sources=clip_sources or None,
        clip_source_digests=clip_digests or None,
        narration_audio=narration_audio or None,
    )
    deps.log(f"  [render] {len(dag.segments)} segment(s), {dag.total_dur_s:.1f}s master")
    report = render_dag(dag, cache, log=deps.log, max_workers=deps.max_workers)
    return report, out_dir / "timeline.otio"


def stage_qa(
    final_path: Path,
    expected_dur_s: float,
    job_id: str,
    project_id: str,
    deps,
    transcript: dict | None = None,
    enhance_info: Any | None = None,
    raw_source: Path | None = None,
    *,
    timeline=None,
    source_duration_s: float = 0.0,
) -> AuditReport:
    """Deterministic QA + the 2-bus acoustic verification (SNR/clipping/engine).

    Full-arc retention gate (M5 postmortem): for >= 20-minute raw sources the
    master must hold 600s+/6+ narration/8+ raw clips/0.70+ source span. A
    failed master is QUARANTINED (renamed `<stem>.qa_failed<ext>`) — a
    truncated `eustis_llm_directed_master.mp4` can never ship.
    """
    audio_verification = None
    speech_intervals = None
    if enhance_info is not None:
        from director_os.core.qa_probe import AudioVerification

        # the -12..-6 speech window + SNR/clipping gates are the 2-BUS law —
        # legacy loudnorm_48k renders keep their integrated-loudness law
        speech_intervals = list(enhance_info.speech_intervals)
        audio_verification = AudioVerification(
            raw_audio_path=str(enhance_info.raw_reference_wav),
            enhancer_engine=enhance_info.engine,
            enhancer_neural=bool(enhance_info.neural),
            atten_lim_db=enhance_info.atten_lim_db,
        )
    audit = audit_technical(
        final_path,
        expected_dur_s=expected_dur_s,
        speech_intervals=speech_intervals,
        audio_verification=audio_verification,
    )
    retention_failed = False
    if timeline is not None and source_duration_s >= RETENTION_QA_MIN_SOURCE_S:
        from director_os.agent_runtime.plan_invariants import audit_retention

        retention = audit_retention(timeline, source_duration_s)
        retention_failed = not retention.ok
        audit = AuditReport(
            path=audit.path,
            ok=audit.ok and retention.ok,
            checks=[*audit.checks, *retention.checks],
        )
    if enhance_info is not None:
        # ear evidence next to the metrics: 15 s raw-vs-clean pair
        from director_os.worker.audio_enhance import export_enhance_ab

        export_enhance_ab(deps, project_id, enhance_info, video_path=raw_source)
    with deps.sessions() as s:
        if audit.ok:
            _update_job(deps.sessions, job_id, JobStage.human_gate, 100.0)
            _set_status(s, project_id, ProjectStatus.awaiting_review)
        else:
            failures = "; ".join(f"{c.name}: {c.detail}" for c in audit.failures)
            _update_job(deps.sessions, job_id, JobStage.qa, 90.0, error=f"qa_failed: {failures}")
            _set_status(s, project_id, ProjectStatus.qa_failed)
            if retention_failed and final_path.exists():
                quarantine = final_path.with_name(
                    final_path.stem + ".qa_failed" + final_path.suffix
                )
                final_path.rename(quarantine)
    return audit


def kept_windows_from_timeline(timeline, asset_id: str) -> list[tuple[float, float]]:
    """Master-track raw clips of one asset -> merged keep ranges (source time).

    This is the lazy-processing contract: everything the Director kept is
    enhanced + baked; everything cut is never touched again.
    """
    from director_os.otio_engine.contract import RawClipMeta, parse_clip_meta
    from director_os.otio_engine.timeline import (
        MASTER_TRACK,
        is_gap,
        item_duration_s,
        track_by_name,
    )

    master = track_by_name(timeline, MASTER_TRACK)
    if master is None:
        return []
    windows: list[tuple[float, float]] = []
    for item in master:
        dur = item_duration_s(item)
        if dur <= 0 or is_gap(item):
            continue
        try:
            meta = parse_clip_meta(item)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(meta, RawClipMeta) or meta.asset_id != asset_id:
            continue
        start_s = item.source_range.start_time.to_seconds() if item.source_range else 0.0
        windows.append((start_s, start_s + dur))
    return windows


# --------------------------------------------------------------------- runner
def run_pipeline(
    source: Path | str,
    project_id: str,
    deps: PipelineDeps,
    job_id: str | None = None,
    *,
    force_replan: bool = False,
    custom_headers: dict[str, str] | None = None,
) -> PipelineResult:
    """Full autonomous run; safe to re-invoke after a crash at any stage.

    `source` may be a local path or an http(s) URL — URLs are downloaded
    through the guardrail UA layer (`core.http.download`) INSIDE the tracked
    failure scope below, so a dead download records WHERE the job died
    instead of leaving a zombie running job. `custom_headers` rides on top
    of the mandatory identity headers (cookies / referer for FOIA portals).

    `force_replan=True` invalidates the planning shortcut (a refreshed
    CaseDossier re-runs Stage 6+ without re-downloading the video or
    re-running Stage 3 ASR — the transcript short-circuit stays intact).
    """
    source_is_url = isinstance(source, str) and source.startswith(("http://", "https://"))
    if not source_is_url:
        source = Path(source).resolve()
        if not source.exists():
            raise PipelineError(f"source missing: {source}")

    with deps.sessions() as s:
        if s.get(ProjectRow, project_id) is None:
            raise PipelineError(f"unknown project {project_id}")
        if job_id is None:
            job = JobRow(project_id=project_id, current_stage=JobStage.ingest.value)
            s.add(job)
            s.commit()
            job_id = job.id
        elif s.get(JobRow, job_id) is None:
            raise PipelineError(f"unknown job {job_id}")

    deps.log(f"[pipeline] project={project_id} job={job_id}")

    # Live observability: one tracker per run — every stage becomes a step with
    # its own progress bar, every deps.log line streams to the sandbox console.
    tracker = deps.tracker or RunTracker(deps.sessions, job_id, project_id)
    deps.tracker = tracker
    if deps.log is not tracker.log:
        deps.log = tracker.log

    try:
        if source_is_url:
            # The download lives INSIDE the failure contract: a dead URL used
            # to raise before the tracker existed and the job row never
            # learned about it (zombie "running" job, Run clicks said nothing).
            # As a tracked step it also streams byte progress to the console.
            name = Path(str(source).split("?")[0]).name or "ingest.bin"
            dest = deps.settings.ingest_dir / name
            deps.log(f"[pipeline] downloading source with UA guardrail -> {dest.name}")
            last_mib = {"v": -8}

            with tracker.step("ingest", f"Download source — {name}") as st:

                def _dl_progress(received: int, total: int | None) -> None:
                    mib = received >> 20
                    if total:
                        st.pct(
                            5.0 + 85.0 * (received / total),
                            f"downloading {mib} / {total >> 20} MiB",
                        )
                    elif mib >= last_mib["v"] + 8:
                        last_mib["v"] = mib
                        st.pct(5.0, f"downloading {mib} MiB")

                source = download(
                    str(source),
                    dest,
                    extra_headers=custom_headers or None,
                    progress_cb=_dl_progress,
                )
                st.pct(95, "download complete")
        # one unconditional rebind: after the tracked download (URL case) or
        # the resolve above (local-file case) the source is ALWAYS a local
        # file — mypy cannot narrow through the source_is_url flag, and the
        # eight stage calls below need the plain Path
        source = Path(source).resolve()
        if not source.exists():
            raise PipelineError(f"source missing: {source}")

        tracker.stage(JobStage.ingest.value, 5.0)
        with tracker.step("ingest", "Ingest source — probe + 480p proxy bundle") as st:
            st.pct(15, "probing source")
            asset_id = stage_ingest(source, project_id, deps)
            st.pct(92, "proxy bundle ready")

        tracker.stage(JobStage.perception.value, 22.0)
        _check_stop(deps, project_id)
        with tracker.step("perception", "Transcribe audio (ASR)") as st:
            st.pct(20, "audio in flight")
            transcript = stage_perception(asset_id, source, project_id, deps)
            st.pct(90, "transcript persisted")

        tracker.stage(JobStage.proofread.value, 26.0)
        with tracker.step("proofread", "Proofread captions — jargon + Officer/Suspect roles") as st:
            try:
                proof = stage_proofread(asset_id, transcript, project_id, deps)
                transcript = {**transcript, "proofread": proof}
            except PipelineError:
                raise
            except Exception as e:  # noqa: BLE001 — captions survive on raw words
                st.skip(f"skipped ({e}) — subtitles fall back to raw ASR")
                deps.log(f"  [proofread] skipped ({e}) — subtitles fall back to raw ASR")

        if deps.vision_enrich:
            with tracker.step("vision", "Vision index — frames + VLM timeline + OCR"):
                stage_vision_enrichment(asset_id, source, deps)

        tracker.stage(JobStage.planning.value, 40.0)
        _check_stop(deps, project_id)
        with tracker.step("planning", "LLM Director — cut the story"):
            timeline, version = stage_planning(
                asset_id, transcript, project_id, deps, force_replan=force_replan
            )

        # Stage 6.5 — the legacy Director autodrive re-cuts the v1 timeline
        # when enabled. The M5 story-map session already owns >= 15 min
        # sources end-to-end, so the autodrive only runs for short sources.
        documentary_planned = False
        with deps.sessions() as s:
            _asset = s.get(MediaAssetRow, asset_id)
            documentary_planned = bool(_asset and _asset.duration_s >= DOCUMENTARY_MIN_SOURCE_S)
        if deps.llm_direct and not documentary_planned:
            tracker.stage(JobStage.planning.value, 42.0)
            from director_os.worker.director import stage_director

            with tracker.step("director", "Director autodrive re-cut") as st:
                try:
                    timeline, version, _director_report = stage_director(
                        timeline, asset_id, transcript, project_id, deps
                    )
                except Exception as e:  # noqa: BLE001 — the LLM is a director, not a dependency
                    st.skip(f"failed ({e}) — deterministic v1 cut rides on")
                    deps.log(f"  [director] stage failed ({e}) — deterministic v1 cut rides on")

        # Stage 6.7 — redaction AFTER the plan (post-plan scoping law, the
        # operator mandate): detection pays only for footage the edit kept.
        # A 43-min source keeping 8 min sweeps 8 min — not 77k raw frames —
        # and two queued jobs share one sweep via the ranges cache + lock.
        if deps.redact_backend:
            tracker.stage(JobStage.redaction.value, 44.0)
            _check_stop(deps, project_id)
            with tracker.step("redaction", "Privacy redaction sweep — kept windows only"):
                stage_redaction(asset_id, source, project_id, deps, timeline=timeline)

        tracker.stage(JobStage.narration.value, 46.0)
        _check_stop(deps, project_id)
        narration_audio: dict[str, Path] | None = None
        narration_clips: list[dict] = []
        with tracker.step("narration", "Narration TTS") as st:
            try:
                narration_clips = stage_narration(timeline, project_id, deps)
            except Exception as e:  # noqa: BLE001 — a TTS outage must not kill the cut
                st.skip(f"failed ({e}) — narration segments render silent")
                deps.log(f"  [narration] skipped ({e}) — narration segments render silent")
                narration_clips = []
        if narration_clips:
            from director_os.otio_engine.contract import NarratorClipMeta, parse_clip_meta
            from director_os.otio_engine.timeline import MASTER_TRACK, is_gap, track_by_name

            # RETIME: narration segments follow the REAL TTS durations (no
            # trailing dead air, no clipped final words); cards re-anchor to
            # the measured durations so location_intro stays on its block
            pre_spans = master_spans(timeline)
            n_retimed = retime_narration_to_tts(timeline, narration_clips)
            if n_retimed:
                n_cards = retime_cards_after_narration(timeline, pre_spans)
                deps.log(
                    f"  [narration] retimed {n_retimed} segment(s) to TTS durations, "
                    f"re-anchored {n_cards} card(s)"
                )

            master = track_by_name(timeline, MASTER_TRACK)
            narration_audio = {}
            if master is not None:
                for item in master:
                    if is_gap(item):
                        continue
                    try:
                        meta = parse_clip_meta(item)
                    except Exception:  # noqa: BLE001
                        continue
                    if not isinstance(meta, NarratorClipMeta):
                        continue
                    match = next(
                        (
                            c
                            for c in narration_clips
                            if c.get("item") == (getattr(item, "name", "") or "")
                        ),
                        None,
                    )
                    if match and Path(match["clip"]).exists():
                        narration_audio[meta.script_text] = Path(match["clip"])

        # Stage 4 -> 5 seam: YAMNet music spans route the Demucs vocals stem
        # into the enhancer (Content-ID guardrail) BEFORE any heavy processing
        music_spans: list[tuple[float, float]] = []
        if deps.audio_mode == "isolate_2bus":
            with deps.sessions() as s:
                prow = s.get(PerceptionIndexRow, asset_id)
            shots = (prow.shots_json or {}) if prow else {}
            from director_os.worker.music_guard import resolve_music_routing

            music_spans, _resolver_active = resolve_music_routing(asset_id, shots, deps)

        # LAZY heavy processing: 2-bus enhancement runs AFTER planning, per
        # KEPT window — the cut-out 85% of a long FOIA packet is never sent
        # through MossFormer2. Compiled clips (audio_mode=isolate_2bus) find
        # their master WAV at render time, exactly as before.
        enhance_info = None
        if deps.audio_mode == "isolate_2bus":
            tracker.stage(JobStage.audio_prep.value, 52.0)
            _check_stop(deps, project_id)
            from director_os.audio_engine.enhance import EnhancerUnavailableError, resolve_enhancer
            from director_os.worker.audio_enhance import stage_audio_enhance

            kept = kept_windows_from_timeline(timeline, asset_id)
            # THE UNINSTALLED-ENHANCER LAW: a machine without deep-filter must
            # still get its cut. The step is SKIPPED — loudly, never silently —
            # the raw clips downgrade isolate_2bus -> loudnorm_48k so render
            # proceeds with leveled raw audio, and installing deep-filter later
            # (scripts/fetch_deep_filter.py) + Run upgrades the audio without
            # redoing finished stages. The published-QA law is untouched: the
            # receipt still names the engine actually used.
            enhancer_ready = True
            if deps.speech_enhancer is None:
                try:
                    resolve_enhancer()
                except EnhancerUnavailableError:
                    enhancer_ready = False
            if not enhancer_ready:
                with tracker.step("audio_prep", "2-bus speech enhancement — kept windows") as st:
                    st.skip(
                        "no neural enhancer installed (deep-filter) — loudnorm fallback; "
                        "install per scripts/fetch_deep_filter.py and re-run to upgrade"
                    )
                n_downgraded = _downgrade_2bus_audio_mode(timeline)
                deps.log(
                    "  [audio-prep] SKIPPED — no neural speech enhancer (deep-filter) on this "
                    f"machine; {n_downgraded} raw clip(s) render with loudnorm raw audio. "
                    "Install it (python scripts/fetch_deep_filter.py) and press Run again — "
                    "finished stages are never redone, only the audio is upgraded."
                )
            else:
                try:
                    with tracker.step("audio_prep", "2-bus speech enhancement — kept windows"):
                        enhance_info = stage_audio_enhance(
                            asset_id,
                            source,
                            transcript,
                            project_id,
                            deps,
                            enhancer=deps.speech_enhancer,
                            kept_windows=kept or None,
                            music_flagged_spans=music_spans,
                        )
                except EnhancerUnavailableError as e:
                    # binary vanished between probe and use — same honest skip
                    deps.log(
                        f"  [audio-prep] SKIPPED — enhancer unavailable mid-run ({e}); "
                        "loudnorm fallback. Install deep-filter and re-run to upgrade."
                    )
                    _downgrade_2bus_audio_mode(timeline)

        tracker.stage(JobStage.render.value, 60.0)
        _check_stop(deps, project_id)
        with tracker.step("render", "Render master — segment cache, misses only") as st:
            report, otio_path = stage_render(
                timeline,
                project_id,
                source,
                asset_id,
                deps,
                enhanced_audio=enhance_info.enhanced_audio_map() if enhance_info else None,
                narration_audio=narration_audio if narration_clips else None,
            )
            st.pct(96, f"{report.rendered} rendered / {report.cached} cached")
        # the disk now carries THIS program (ops@max_rev + active version) —
        # stamp the gate in EVERY redact mode, not only the tracked path,
        # or the Export button lies "dirty" forever on default projects
        _stamp_export_gate(deps, project_id, asset_id)

        tracker.stage(JobStage.subtitles.value, 85.0)
        _check_stop(deps, project_id)
        with tracker.step("subtitles", "Caption build — 2-row karaoke .ass + .srt"):
            stage_subtitles(timeline, transcript, project_id, deps)

        # POST-FX (owner rule): blur + enhance + captions run ONLY now, on the
        # assembled deliverable — never on the 43-min source. QA audits
        # THIS file.
        post_fx_report = None
        final_for_qa = report.final_path
        if deps.post_fx:
            tracker.stage(JobStage.qa.value, 88.0)
            with tracker.step(
                "post_fx", "Post-FX on the deliverable — tracked blur + burn-in"
            ) as st:
                try:
                    from director_os.worker.post_fx import stage_post_fx

                    post_fx_report = stage_post_fx(
                        report.final_path,
                        timeline,
                        transcript,
                        project_id,
                        deps,
                        narration_clips=narration_clips,
                    )
                    final_for_qa = post_fx_report.final_path or report.final_path
                except Exception as e:  # noqa: BLE001
                    st.skip(f"FAILED ({e}) — QA audits the un-post-fx'd master")
                    deps.log(f"  [post-fx] FAILED ({e}) — QA audits the un-post-fx'd master")

        tracker.stage(JobStage.qa.value, 90.0)
        _check_stop(deps, project_id)
        # expected duration = the DAG's frame-quantized total (what ffmpeg truly
        # renders), NOT the raw OTIO sum — unquantized sums accumulate per-segment
        # frame rounding and fail QA on long cuts (live: 4.92 s over 329 segments)
        with tracker.step("qa", "Technical QA audit — loudness, duration, retention"):
            audit = stage_qa(
                final_for_qa,
                report.total_dur_s,
                job_id,
                project_id,
                deps,
                transcript=transcript,
                enhance_info=enhance_info,
                raw_source=source,
                timeline=timeline,
                source_duration_s=_source_duration(deps, asset_id),
            )
            if not audit.ok:
                raise PipelineError(f"QA failed: {[c.detail for c in audit.failures]}")

        # Stage 10 — Creator Ship Pack (YouTube upload is disabled; this folder
        # is what the human ships). Soft-fail: a pack glitch never unpublishes
        # a QA-passed master. The JOB stays parked at human_gate — the pack is
        # a sub-product of the gate, not a stage beyond it.
        tracker.stage(JobStage.ship_pack.value, 97.0)
        with tracker.step("ship_pack", "Creator ship pack — chapters, SEO, thumbnails") as st:
            try:
                from director_os.worker.ship_pack import build_ship_pack

                with deps.sessions() as s:
                    prow = s.get(PerceptionIndexRow, asset_id)
                shots = (prow.shots_json or {}) if prow else {}
                ocr = (
                    ((prow.redaction_tracks_json or {}).get("timestamp_ocr") or {}) if prow else {}
                )
                card_sub = f"CASE FILE — {ocr.get('iso')}" if ocr.get("iso") else None
                # The pack MUST carry the post-fx deliverable (captions burned,
                # tracked blur baked, audio enhanced) — the file the human ships.
                # Passing report.final_path here shipped the PRE-fx master and
                # every upload from the pack silently lost captions/blur (CR-001).
                pack = build_ship_pack(
                    project_id,
                    deps,
                    timeline,
                    final_for_qa,
                    transcript=transcript,
                    insights=(shots.get("visual_timeline") or {}).get("insights", []),
                    card_subtitle=card_sub,
                    enhance_info=enhance_info,
                    duration_s=report.total_dur_s,
                )
                deps.log(
                    f"  [ship-pack] {pack.pack_dir} — chapters {len(pack.chapters)}, "
                    f"thumbnails {len(pack.seo.get('keywords', []))} kw, "
                    f"clean ends: {pack.verification.get('clean_first_last_frames')}"
                )
            except Exception as e:  # noqa: BLE001 — pack failure is never a QA failure
                st.skip(f"skipped ({e})")
                deps.log(f"  [ship-pack] skipped ({e})")
        tracker.stage(JobStage.human_gate.value, 100.0)
    except Exception as e:
        # soft-failure persistence: the job row records WHERE it died, so the
        # operator (or the retry task) resumes instead of guessing. A HARD kill
        # (SIGKILL) persists nothing — resume works because every stage's
        # product lives in its domain table, never only in memory.
        stage, progress = _current_stage(deps.sessions, job_id)
        error = str(e)[:2000]
        if deps.tracker is not None:
            deps.tracker.stage(stage.value, progress, error=error)
            deps.tracker.error(f"pipeline failed at {stage.value}: {error}")
        else:
            _update_job(deps.sessions, job_id, stage, progress, error=error)
        raise

    shipped = post_fx_report.final_path if post_fx_report else report.final_path
    if shipped is None:
        raise PipelineError("render produced no final deliverable path")
    deps.log(f"[pipeline] DONE {shipped} (audit ok)")
    return PipelineResult(
        job_id=job_id,
        project_id=project_id,
        asset_id=asset_id,
        otio_version=version,
        final_path=shipped,
        otio_path=otio_path,
        render=report,
        audit=audit,
        timeline=timeline,
        source_duration_s=_source_duration(deps, asset_id),
        dossier=_load_dossier(project_id, deps),
        post_fx=post_fx_report,
    )


def _source_duration(deps: PipelineDeps, asset_id: str) -> float:
    with deps.sessions() as s:
        asset = s.get(MediaAssetRow, asset_id)
        return float(asset.duration_s) if asset else 0.0


def _current_stage(sessions, job_id: str) -> tuple[JobStage, float]:
    with sessions() as s:
        job = s.get(JobRow, job_id)
        try:
            return JobStage(job.current_stage), job.progress_pct or 0.0
        except ValueError:
            return JobStage.ingest, 0.0


def dag_total_dur(timeline) -> float:
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.timeline import track_duration_s

    return track_duration_s(master_track(timeline))


def make_deps(
    database_url: str,
    transcribe: Callable[[Path, str], Any] | None = None,
    settings: Settings | None = None,
    **overrides: Any,
) -> PipelineDeps:
    engine = make_engine(database_url)
    init_models(engine)
    if settings is None:
        # Production path: Studio-saved credentials (Settings UI) override
        # .env — refresh the overlay per job so key changes never need a
        # worker restart. Tests inject explicit settings and skip this.
        from director_os.core.credentials import apply_credentials_to_settings

        apply_credentials_to_settings(engine)
        resolved = get_settings()
    else:
        resolved = settings
    # REDACT_MODE env default ("tracked" | "yunet" | "modal") — explicit
    # overrides always win; unset env keeps the hermetic CPU default
    overrides.setdefault("redact_backend", os.getenv("REDACT_MODE", "").strip().lower() or None)
    # AUDIO_MODE env default ("loudnorm_48k" | "isolate_2bus") — production
    # bodycam runs set AUDIO_MODE=isolate_2bus + AUDIO_ENHANCER=deepfilter
    overrides.setdefault(
        "audio_mode", os.getenv("AUDIO_MODE", "").strip().lower() or "loudnorm_48k"
    )
    # DIRECTOR_AUTODRIVE env default — the LLM Director re-cuts the v1 timeline
    overrides.setdefault(
        "llm_direct", os.getenv("DIRECTOR_AUTODRIVE", "").strip().lower() in {"1", "true", "yes"}
    )
    # POST_FX env default — the owner rule is now the standing law for
    # documentary runs; explicit overrides always win; tests stay hermetic
    overrides.setdefault(
        "post_fx", os.getenv("POST_FX", "").strip().lower() in {"1", "true", "yes"}
    )
    return PipelineDeps(
        sessions=make_session_factory(engine),
        settings=resolved,
        transcribe=transcribe,
        **overrides,
    )


def production_transcribe(
    settings: Settings | None = None,
    sessions: Any = None,
    log: Callable[[str], None] | None = None,
) -> Callable[[Path, str], Any] | None:
    """Gladia transcriber when GLADIA_API_KEY is configured, else None.

    The fleet entrypoint (`advance_job`) uses this so a real worker run
    transcribes instead of dying at perception with "no transcribe callable".
    Tests inject their own stub; production env provides the key.

    With `sessions` (the pipeline's own sessionmaker) the client is wrapped in
    the content-addressed ASR cache: identical media is NEVER re-uploaded —
    a retry adopts the transcription Gladia already has, or serves the stored
    payload outright. `log` streams upload/poll progress into the Studio
    console (the old silent 15-minute transcription read as a hang).
    """
    resolved = settings or get_settings()
    if not resolved.gladia_api_key:
        return None
    from director_os.perception.gladia import GladiaClient

    client = GladiaClient(resolved.gladia_api_key, log=log)
    if sessions is None:
        return client.transcribe
    from director_os.perception.asr_cache import CachedTranscriber

    return CachedTranscriber(client, sessions, log=log).transcribe


def advance_job(
    job_id: str | None,
    project_id: str,
    source: str,
    database_url: str,
    *,
    force_replan: bool = False,
    custom_headers: dict[str, str] | None = None,
) -> dict:
    """Fleet entrypoint: resume/advance one pipeline job (crash-safe by design).

    Registered as a Celery task in `director_os.worker.tasks`; kept as a plain
    function here so tests and a future Temporal workflow call it unchanged.
    """
    # deps FIRST: make_deps applies the Studio credential overlay (a fresh
    # worker process must see vault keys BEFORE any provider client resolves —
    # the old `make_deps(..., transcribe=production_transcribe())` evaluated
    # the transcriber before the overlay landed, so a cold worker's first job
    # died at perception with "Gladia not configured" even with a saved key).
    deps = make_deps(database_url)
    if deps.transcribe is None:
        # lazy lookup on purpose: run_pipeline rebinds deps.log to the job's
        # RunTracker after this point — the lambda must resolve the attribute
        # per call or the Gladia progress lines bypass job_logs + SSE
        deps.transcribe = production_transcribe(
            deps.settings, deps.sessions, log=lambda msg: deps.log(msg)
        )
    # source stays a str: run_pipeline detects http(s) URLs and downloads
    # through the guardrail UA layer — Path()-wrapping here would turn a URL
    # into a bogus relative path and kill URL ingest on the fleet path.
    result = run_pipeline(
        source,
        project_id,
        deps,
        job_id=job_id,
        force_replan=force_replan,
        custom_headers=custom_headers,
    )
    return {
        "job_id": result.job_id,
        "final": str(result.final_path),
        "audit_ok": result.audit.ok,
        "rendered": result.render.rendered,
        "cached": result.render.cached,
    }
