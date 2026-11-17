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
import os
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
    # --- audio law: "loudnorm_48k" (legacy) | "isolate_2bus" (broadcast mix)
    audio_mode: str = "loudnorm_48k"
    speech_enhancer: Any | None = None  # SpeechEnhancer override (tests/fakes)
    vocals_stem_resolver: Callable[[str], Path | None] | None = None  # Demucs pre-split hook
    # --- M4 vision stack (all optional; None/False keeps the CPU-only pipeline)
    redact_backend: str | None = None  # "tracked" (3-pass, issue #14) | "yunet" | "modal"
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

    backend "tracked" (issue #14, the default when the YOLO model is present):
    the deterministic 3-pass pipeline — CFR normalize, imgsz-960/conf-0.22 YOLO
    + GMC tracking, One-Euro trajectory repair — persisted as a per-frame plan
    PLUS the static regions the studio canvas already draws. backend "modal":
    YOLOv8-face + plate + ByteTrack on the serverless GPU fleet (persistent
    Person/Plate IDs, time-binned tracked regions). backend "yunet": the
    zero-download CPU sweep used since the first real-footage run. A backend
    that cannot run is a skipped stage, never a broken pipeline.
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
    tracked_extra: dict[str, Any] = {}  # per_frame plan + digest (tracked backend only)
    if backend == "tracked":
        try:
            tracked_payload = _tracked_redaction_payload(
                source, project_id, deps, frame_w, frame_h, max_regions
            )
            regions = tracked_payload["regions"]
            source_tag = tracked_payload["source"]
            stats = tracked_payload["stats"]
            tracked_extra = {
                "per_frame": tracked_payload["per_frame"],
                "plan_digest": tracked_payload["plan_digest"],
            }
        except Exception as e:  # model missing, sweep failure — degrade loudly
            deps.log(f"  [redaction] tracked 3-pass unavailable ({e}) — falling back to YuNet")
            backend = "yunet"

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

        sweep = detect_faces(source, sample_every_s=1.0)
        regions = cluster_to_regions(
            sweep, max_regions=max_regions, frame_w=frame_w, frame_h=frame_h
        )
        source_tag = f"yunet_autodetect:{sweep.stats.detector}"
        stats = {
            "frames_sampled": sweep.stats.frames_sampled,
            "detections": sweep.stats.detections,
        }

    payload_json: dict[str, Any] = {
        "regions": [r.model_dump(mode="json") for r in regions],
        "source": source_tag,
        "stats": stats,
        **tracked_extra,
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


def _tracked_redaction_payload(
    source: Path,
    project_id: str,
    deps: PipelineDeps,
    frame_w: int,
    frame_h: int,
    max_regions: int,
) -> dict[str, Any]:
    """Pass 1+2 of the 3-pass pipeline — the plan the render stage bakes in.

    Persists BOTH contracts: the per-frame plan (`per_frame`, frame-indexed)
    for the render bake, and static padded regions (studio canvas). The plan
    digest rides in the payload so a re-sweep invalidates the baked source
    naturally (content-addressed, like the segment cache).
    """
    import hashlib
    import json as _json

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

    out_dir = deps.settings.output_dir / "pipeline" / project_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cfr_path = normalize_cfr(source, out_dir / "source_cfr.mp4")
    n_frames = video_n_frames(cfr_path)
    if not n_frames:
        raise RuntimeError("CFR source frame count unknown — refusing to plan blind")

    tracks = backend.detect_faces(cfr_path)  # imgsz 960, conf 0.22, GMC + buffer 60
    plate_tracks: list = []
    if caps.get("plates"):
        # plates ride the same 3-pass law (issue #12): same sweep, same GMC
        # tracker, same Pass-2 repair. Absent model -> faces-only, never a blocker.
        plate_tracks = backend.detect_plates(cfr_path)
    try:
        merged = merge_redaction_tracks(tracks, plate_tracks)
    except EmptySweepError as e:
        # A sweep that sees NOTHING is a broken sweep, not an empty scene:
        # shipping it bakes zero redaction while the audit passes vacuously
        # (0 detections = 0 leaks). Fail the stage loud — privacy is not soft.
        raise RuntimeError(str(e)) from e
    plan = repair_trajectories(merged, fps=30, frame_w=frame_w, frame_h=frame_h, n_frames=n_frames)
    from director_os.vision_fx.redact_track import assert_zero_leaks

    assert_zero_leaks(plan, merged, 30.0)  # the zero-leak contract, enforced at plan time

    regions = tracks_to_regions(merged, frame_w=frame_w, frame_h=frame_h)[:max_regions]
    plan_json = plan.model_dump(mode="json")
    digest = hashlib.sha256(_json.dumps(plan_json, sort_keys=True).encode()).hexdigest()[:8]
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
    """Pass 3 — bake the per-frame plan into a redacted source (once, cached).

    When the persisted plan exists (REDACT_MODE=tracked) the render DAG cuts
    its segments from THIS file instead of the raw source, so every segment —
    cached or fresh — inherits zero-leak redaction. Content-addressed by the
    plan digest: a changed plan is a different file, a stale artifact never
    gets reused.
    """
    with deps.sessions() as s:
        row = s.get(PerceptionIndexRow, asset_id)
        payload = (row.redaction_tracks_json or {}) if row else {}
    per_frame = payload.get("per_frame")
    if not per_frame or deps.redact_backend != "tracked":
        return None
    from director_os.vision_fx.redact_track import (
        normalize_cfr,
        render_tracked_redaction,
    )
    from director_os.vision_fx.trajectory import PerFrameRedaction

    plan = PerFrameRedaction.model_validate(per_frame)
    digest = payload.get("plan_digest") or "unpinned"
    dst = out_dir / f"redacted_source_{digest}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        deps.log(f"  [render] baked redacted source reuse ({dst.name})")
        return dst
    cfr_path = normalize_cfr(source, out_dir / "source_cfr.mp4")
    deps.log(f"  [render] baking tracked redaction source ({digest})...")
    render_tracked_redaction(
        cfr_path,
        plan,
        dst,
        mode="gauss",
        n_frames_expected=plan.n_frames,
    )
    return dst


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
    enhanced_audio: dict[str, Path] | None = None,
) -> tuple[RenderReport, Path]:
    """Compile + segment-cached render (content-addressed idempotency).

    REDACT_MODE=tracked + a persisted per-frame plan -> segments are cut from
    the baked redacted source (Pass 3), so the privacy redaction is frame-
    exact regardless of segment cache hits; the static blur regions still
    ride the filtergraph for studio-drawn boxes. `enhanced_audio` feeds the
    2-bus speech-isolation masters (audio_mode=isolate_2bus).
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
    baked_source: Path | None = None
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
) -> AuditReport:
    """Deterministic QA + the 2-bus acoustic verification (SNR/clipping/engine)."""
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
    if enhance_info is not None:
        # ear evidence next to the metrics: 15 s raw-vs-clean pair
        from director_os.worker.audio_enhance import export_enhance_ab

        export_enhance_ab(deps, project_id, enhance_info, video_path=raw_source)
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

        # 2-bus speech isolation runs BEFORE planning so the compiled clips
        # (audio_mode=isolate_2bus) always find their master WAV at render time
        enhance_info = None
        if deps.audio_mode == "isolate_2bus":
            _update_job(deps.sessions(), job_id, JobStage.audio_prep, 38.0)
            from director_os.worker.audio_enhance import stage_audio_enhance

            enhance_info = stage_audio_enhance(
                asset_id,
                source,
                transcript,
                project_id,
                deps,
                enhancer=deps.speech_enhancer,
            )

        _update_job(deps.sessions(), job_id, JobStage.planning, 45.0)
        timeline, version = stage_planning(asset_id, transcript, project_id, deps)

        _update_job(deps.sessions(), job_id, JobStage.render, 60.0)
        report, otio_path = stage_render(
            timeline,
            project_id,
            source,
            asset_id,
            deps,
            enhanced_audio=enhance_info.enhanced_audio_map() if enhance_info else None,
        )

        _update_job(deps.sessions(), job_id, JobStage.subtitles, 85.0)
        stage_subtitles(timeline, transcript, project_id, deps)

        _update_job(deps.sessions(), job_id, JobStage.qa, 90.0)
        audit = stage_qa(
            report.final_path,
            dag_total_dur(timeline),
            job_id,
            project_id,
            deps,
            transcript=transcript,
            enhance_info=enhance_info,
            raw_source=source,
        )
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
    # REDACT_MODE env default ("tracked" | "yunet" | "modal") — explicit
    # overrides always win; unset env keeps the hermetic CPU default
    overrides.setdefault("redact_backend", os.getenv("REDACT_MODE", "").strip().lower() or None)
    # AUDIO_MODE env default ("loudnorm_48k" | "isolate_2bus") — production
    # bodycam runs set AUDIO_MODE=isolate_2bus + AUDIO_ENHANCER=deepfilter
    overrides.setdefault(
        "audio_mode", os.getenv("AUDIO_MODE", "").strip().lower() or "loudnorm_48k"
    )
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
