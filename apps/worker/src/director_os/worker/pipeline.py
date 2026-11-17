"""Durable staged pipeline — PLAN §7 Stage 1-10 state machine (M2 core).

Each stage persists its product in a DOMAIN table before advancing the job
row, so a worker that dies mid-pipeline resumes without redoing paid work:

- ingest     -> `media_assets` (probe facts)          — idempotent per project+source
- perception -> `perception_index` (ASR transcript)   — idempotent per asset: the
               crash-resume guarantee "never re-run ASR/TTS" comes from here
- planning   -> `timeline_versions` + `projects.active_otio_version_id`
- render     -> `segment_cache` (content-addressed)   — misses only, ever
- qa         -> `audit_technical` report; failure parks the job in `qa_failed`

The ASR callable is injected (`deps.transcribe`): Gladia in production, a stub
in tests. Nothing here talks to the network.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from director_os.core.config import Settings, get_settings
from director_os.core.db import (
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
from sqlalchemy import select

AUTO_CUT_MAX_SEGMENT_S = 8.0


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
    # --- M4 vision stack (all optional; None/False keeps the CPU-only pipeline)
    redact_backend: str | None = None  # "yunet" (CPU, offline) | "modal" (GPU fleet)
    vision_enrich: bool = False  # frame-quality score + VLM timeline + timestamp OCR
    log: Callable[[str], None] = lambda _msg: None


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


def _update_job(s, job_id: str, stage: JobStage, progress: float, error: str | None = None) -> None:
    job = s.get(JobRow, job_id)
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
    transcript = deps.transcribe(source, asset_id)
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


def stage_redaction(
    asset_id: str, source: Path, project_id: str, deps: PipelineDeps, max_regions: int = 8
) -> list:
    """Auto-detect redaction regions — idempotent, pluggable backend (M4).

    backend "modal": YOLOv8-face + plate + ByteTrack on the serverless GPU fleet
    (persistent Person/Plate IDs, time-binned tracked regions). backend "yunet":
    the zero-download CPU sweep used since the first real-footage run. Both write
    the same `redaction_tracks_json.regions` contract the studio canvas and the
    renderer already consume, so a region the backend fails to produce is a
    skipped stage, never a broken pipeline.
    """
    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        existing = (row.redaction_tracks_json or {}).get("regions") if row else None
        if existing:
            deps.log(f"  [redaction] {len(existing)} region(s) already persisted — skip sweep")
            return existing
        asset = s.get(MediaAssetRow, asset_id)
    frame_w = int(asset.width or 0)
    frame_h = int(asset.height or 0)
    if frame_w <= 0 or frame_h <= 0:
        deps.log("  [redaction] asset geometry unknown — skipping auto sweep")
        return []

    backend = deps.redact_backend or "yunet"
    if backend == "modal":
        from director_os.perception.modal_redact import (
            ModalRedactError,
            parse_tracks_payload,
            redact_video_modal,
            tracks_to_regions,
        )

        try:
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

        sweep = detect_faces(str(source), sample_every_s=1.0)
        regions = cluster_to_regions(
            sweep, max_regions=max_regions, frame_w=frame_w, frame_h=frame_h
        )
        source_tag = f"yunet_autodetect:{sweep.stats.detector}"
        stats = {
            "frames_sampled": sweep.stats.frames_sampled,
            "detections": sweep.stats.detections,
        }

    payload_json = {
        "regions": [r.model_dump(mode="json") for r in regions],
        "source": source_tag,
        "stats": stats,
    }
    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        if row is None:
            row = PerceptionIndexRow(asset_id=asset_id)
            s.add(row)
        # fresh rows hold None until flush (SQL column default applies at INSERT)
        row.redaction_tracks_json = {**(row.redaction_tracks_json or {}), **payload_json}
        s.commit()
    deps.log(
        f"  [redaction] {len(regions)} region(s) via {source_tag} "
        f"({', '.join(sorted({r.track_id or 'box' for r in regions}))[:4]}...)"
    )
    return payload_json["regions"]


def stage_vision_enrichment(asset_id: str, source: Path, deps: PipelineDeps) -> None:
    """M4 vision stack enrichment — every product optional, every failure soft.

    Persists into the existing free-form JSON columns (no migration):
    - `shots_json["frame_quality"]`    — CPU blur/darkness pre-filter (always on)
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
                "unusable_spans": [
                    [round(a, 2), round(b, 2)] for a, b in report.unusable_spans
                ],
                "frames": [asdict(f) for f in report.frames[:600]],
            }
            deps.log(
                f"  [vision] frame quality: {report.usable_ratio:.0%} usable, "
                f"{len(report.unusable_spans)} bad span(s)"
            )
        except Exception as e:  # noqa: BLE001 — enrichment never kills the pipeline
            deps.log(f"  [vision] frame quality skipped: {e}")

    if deps.settings.vision_api_key:
        if "visual_timeline" not in shots:
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
                    f"  [vision] timeline indexed: {len(sheets)} sheet(s) -> "
                    f"{len(insights)} insight(s)"
                )
            except Exception as e:  # noqa: BLE001
                deps.log(f"  [vision] contact-sheet indexing skipped: {e}")

        if "timestamp_ocr" not in red:
            try:
                from director_os.perception.timestamp_ocr import read_bodycam_timestamp
                from director_os.perception.vision import GeminiVisionClient

                client = GeminiVisionClient(deps.settings.vision_api_key)
                stamp = read_bodycam_timestamp(str(source), backend="gemini", client=client)
                if stamp is not None:
                    red["timestamp_ocr"] = {
                        "raw_text": stamp.raw_text,
                        "iso": stamp.iso,
                        "device": stamp.device,
                        "tz": stamp.tz,
                        "confidence": stamp.confidence,
                    }
                    deps.log(f"  [vision] overlay OCR: {stamp.iso} ({stamp.device})")
                else:
                    red["timestamp_ocr"] = {"found": False}
                    deps.log("  [vision] overlay OCR: no timestamp found")
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


def stage_planning(
    asset_id: str, transcript: dict, project_id: str, deps: PipelineDeps
) -> tuple[Any, int]:
    """Deterministic v1 auto-cut -> `timeline_versions`; idempotent via the project pointer.

    v1 rule (documented, no LLM yet): cut the raw asset into <=8s segments and
    open with a location card. The Director agent (M2 close-out) replaces this
    with LLM planning on top of the same timeline_versions durability.
    """
    with deps.sessions() as s:
        project = s.get(ProjectRow, project_id)
        if project.active_otio_version_id is not None:
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
    n_segments = max(1, math.ceil(dur / AUTO_CUT_MAX_SEGMENT_S))
    seg_len = dur / n_segments

    # M4: the AXON overlay OCR populates the intro card when the vision stack
    # read one (e.g. "AXON BODY 3 — 2022-07-04T18:11:55-04:00"); otherwise the
    # deterministic v1 subtitle stays.
    card_fields = {"title": "CASE FILE", "subtitle": "DirectorOS auto-cut v1"}
    with deps.sessions() as s:
        prow = s.get(PerceptionIndexRow, asset_id)
        ocr = (prow.redaction_tracks_json or {}).get("timestamp_ocr") if prow else None
        if ocr and ocr.get("iso"):
            device = ocr.get("device") or "BODYCAM"
            card_fields["subtitle"] = f"{device} — {ocr['iso']}"

    master_items: list[ClipSpec | GapSpec] = [
        ClipSpec(
            name=f"raw_{i + 1}",
            source_start_s=i * seg_len,
            dur_s=seg_len,
            meta=RawClipMeta(asset_id=asset_id, audio_mode="loudnorm_48k"),
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
    """Build the karaoke .ass sidecar from the transcript (deterministic, free).

    No paid API and a pure function of (timeline rate, transcript), so crash-
    resume is satisfied by recomputation — the stage never blocks recovery.
    Output lands next to the timeline: output/pipeline/<project>/subtitles.{json,ass,srt}.
    The JSON cue store is what the Director Studio subtitle grid reads/edits.
    """
    from director_os.subtitles.ass_builder import CaptionBeat, WordTiming
    from director_os.subtitles.store import save_cues

    words = [w for w in transcript.get("words", []) if w.get("text")]
    sub_dir = deps.settings.output_dir / "pipeline" / project_id
    if not words:
        deps.log("  [subtitles] transcript empty — skipping ass build")
        save_cues(sub_dir, [])
        return sub_dir / "subtitles.ass"

    beats: list[CaptionBeat] = []
    MAX_WORDS, MAX_S = 6, 3.5
    run: list[dict] = []

    def _flush() -> None:
        if not run:
            return
        beats.append(
            CaptionBeat(
                text=" ".join(w["text"] for w in run),
                start_ms=int(run[0]["start_s"] * 1000),
                end_ms=int(run[-1]["end_s"] * 1000),
                speaker=run[0].get("speaker") or "Unknown",
                words=[
                    WordTiming(
                        word=w["text"],
                        start_ms=int(w["start_s"] * 1000),
                        end_ms=int(w["end_s"] * 1000),
                    )
                    for w in run
                ],
            )
        )
        run.clear()

    for w in words:
        if run:
            same_speaker = (run[0].get("speaker") or "Unknown") == (w.get("speaker") or "Unknown")
            too_long = w["end_s"] * 1000 - run[0]["start_s"] * 1000 > MAX_S * 1000
            too_many = len(run) >= MAX_WORDS
            boundary = w["start_s"] - run[-1]["end_s"] > 1.0  # >1s hole: new cue
            if too_many or too_long or boundary or not same_speaker:
                _flush()
        run.append(w)
    _flush()

    save_cues(sub_dir, beats)
    out = sub_dir / "subtitles.ass"
    deps.log(f"  [subtitles] {len(beats)} cue(s) -> subtitles.ass + srt + json")
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
) -> tuple[RenderReport, Path]:
    """Compile + segment-cached render (content-addressed idempotency)."""
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
    dag = compile_timeline(
        timeline,
        assets={asset_id: str(source)},
        cache=cache,
        video=deps.video,
        audio=deps.audio,
        out_dir=out_dir,
        audio_target_lufs=deps.settings.target_lufs,
        global_blur_regions=_project_redaction_regions(deps.sessions, asset_id),
        asset_geometry=geometry,
    )
    deps.log(f"  [render] {len(dag.segments)} segment(s), {dag.total_dur_s:.1f}s master")
    report = render_dag(dag, cache, log=deps.log, max_workers=deps.max_workers)
    return report, out_dir / "timeline.otio"


def stage_qa(
    final_path: Path, expected_dur_s: float, job_id: str, project_id: str, deps
) -> AuditReport:
    audit = audit_technical(final_path, expected_dur_s=expected_dur_s)
    with deps.sessions() as s:
        if audit.ok:
            _update_job(s, job_id, JobStage.human_gate, 100.0)
            _set_status(s, project_id, ProjectStatus.awaiting_review)
        else:
            failures = "; ".join(f"{c.name}: {c.detail}" for c in audit.failures)
            _update_job(s, job_id, JobStage.qa, 90.0, error=f"qa_failed: {failures}")
            _set_status(s, project_id, ProjectStatus.qa_failed)
    return audit


# --------------------------------------------------------------------- runner
def run_pipeline(
    source: Path | str,
    project_id: str,
    deps: PipelineDeps,
    job_id: str | None = None,
) -> PipelineResult:
    """Full autonomous run; safe to re-invoke after a crash at any stage.

    `source` may be a local path or an http(s) URL — URLs are downloaded first
    through the guardrail UA layer (`core.http.download`), and the pipeline
    proceeds against the local copy exactly as with a file.
    """
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        name = Path(source.split("?")[0]).name or "ingest.bin"
        dest = deps.settings.ingest_dir / name
        deps.log(f"[pipeline] downloading source with UA guardrail -> {dest.name}")
        source = download(source, dest)
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

    try:
        _update_job(deps.sessions(), job_id, JobStage.ingest, 5.0)
        asset_id = stage_ingest(source, project_id, deps)

        if deps.redact_backend:
            _update_job(deps.sessions(), job_id, JobStage.redaction, 15.0)
            stage_redaction(asset_id, source, project_id, deps)

        _update_job(deps.sessions(), job_id, JobStage.perception, 25.0)
        transcript = stage_perception(asset_id, source, project_id, deps)

        if deps.vision_enrich:
            stage_vision_enrichment(asset_id, source, deps)

        _update_job(deps.sessions(), job_id, JobStage.planning, 45.0)
        timeline, version = stage_planning(asset_id, transcript, project_id, deps)

        _update_job(deps.sessions(), job_id, JobStage.render, 60.0)
        report, otio_path = stage_render(timeline, project_id, source, asset_id, deps)

        _update_job(deps.sessions(), job_id, JobStage.subtitles, 85.0)
        stage_subtitles(timeline, transcript, project_id, deps)

        _update_job(deps.sessions(), job_id, JobStage.qa, 90.0)
        audit = stage_qa(report.final_path, dag_total_dur(timeline), job_id, project_id, deps)
        if not audit.ok:
            raise PipelineError(f"QA failed: {[c.detail for c in audit.failures]}")
    except Exception as e:
        # soft-failure persistence: the job row records WHERE it died, so the
        # operator (or the retry task) resumes instead of guessing. A HARD kill
        # (SIGKILL) persists nothing — resume works because every stage's
        # product lives in its domain table, never only in memory.
        stage, progress = _current_stage(deps.sessions, job_id)
        _update_job(deps.sessions(), job_id, stage, progress, error=str(e)[:2000])
        raise

    deps.log(f"[pipeline] DONE {report.final_path} (audit ok)")
    return PipelineResult(
        job_id=job_id,
        project_id=project_id,
        asset_id=asset_id,
        otio_version=version,
        final_path=report.final_path,
        otio_path=otio_path,
        render=report,
        audit=audit,
    )


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
    resolved = settings or get_settings()
    return PipelineDeps(
        sessions=make_session_factory(engine),
        settings=resolved,
        transcribe=transcribe,
        **overrides,
    )


def advance_job(job_id: str, project_id: str, source: str, database_url: str) -> dict:
    """Fleet entrypoint: resume/advance one pipeline job (crash-safe by design).

    Registered as a Celery task in `director_os.worker.tasks`; kept as a plain
    function here so tests and a future Temporal workflow call it unchanged.
    """
    deps = make_deps(database_url)
    result = run_pipeline(Path(source), project_id, deps, job_id=job_id)
    return {
        "job_id": result.job_id,
        "final": str(result.final_path),
        "audit_ok": result.audit.ok,
        "rendered": result.render.rendered,
        "cached": result.render.cached,
    }
