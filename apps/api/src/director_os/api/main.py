"""FastAPI gateway skeleton (PLAN §2 layer 2).

M1 exposes health + project creation against the core persistence layer.
M2 adds SSE job progress, agent tool dispatch endpoints and the Human Review Gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from director_os.agent_runtime.agents import AgentRole, PlannerExecutor, ToolStep
from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.agent_runtime.editing_tools import DirectorToolbox, build_director_registry
from director_os.agent_runtime.llm import LLMBackendError, build_backend
from director_os.api.channels import ChannelPublisher, MultiChannelPublisher
from director_os.api.events import DEFAULT_MAX_S, DEFAULT_POLL_S, job_progress_stream, require_job
from director_os.api.youtube import (
    Chapter,
    DraftMetadata,
    InMemoryYouTubePublisher,
    YouTubePublisher,
)
from director_os.core.config import get_settings
from director_os.core.db import (
    AgentToolLogRow,
    JobRow,
    MediaAssetRow,
    PerceptionIndexRow,
    ProjectRow,
    init_models,
    make_engine,
    make_session_factory,
    new_id,
)
from director_os.core.http import download
from director_os.core.models import ProjectStatus
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

app = FastAPI(
    title="DirectorOS API",
    version="0.2.0",
    description="Autonomous multimodal video editing platform — gateway",
)

# Director Studio (M3) talks to this gateway from the browser in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None
_session_factory = None

# Publish transport is injected; tests swap in InMemoryYouTubePublisher.
_publisher: YouTubePublisher = InMemoryYouTubePublisher()


def set_publisher(publisher: YouTubePublisher) -> None:
    global _publisher
    _publisher = publisher


def session() -> Session:
    global _engine, _session_factory
    if _engine is None:
        _engine = make_engine()
        init_models(_engine)
        _session_factory = make_session_factory(_engine)
    return _session_factory()


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    niche_skill: str = "bodycam_documentary"


class ProjectOut(BaseModel):
    id: str
    title: str
    niche_skill: str
    status: str


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "directoros-api",
            "version": app.version,
            "time": datetime.now(UTC).isoformat(),
        }
    )


@app.post("/projects", status_code=201, response_model=ProjectOut)
def create_project(body: ProjectCreate) -> ProjectOut:
    s = session()
    row = ProjectRow(
        title=body.title, niche_skill=body.niche_skill, status=ProjectStatus.created.value
    )
    s.add(row)
    s.commit()
    return ProjectOut(id=row.id, title=row.title, niche_skill=row.niche_skill, status=row.status)


@app.get("/projects", response_model=list[ProjectOut])
def list_projects() -> list[ProjectOut]:
    """All projects, newest first (Director Studio project list)."""
    s = session()
    rows = s.execute(select(ProjectRow).order_by(ProjectRow.created_at.desc()).limit(200)).scalars()
    return [
        ProjectOut(id=r.id, title=r.title, niche_skill=r.niche_skill, status=r.status) for r in rows
    ]


@app.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: str) -> ProjectOut:
    s = session()
    row = s.execute(select(ProjectRow).where(ProjectRow.id == project_id)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="project not found")
    return ProjectOut(id=row.id, title=row.title, niche_skill=row.niche_skill, status=row.status)


# ------------------------------------------------------------------ read models for the studio
class JobOut(BaseModel):
    id: str
    stage: str
    progress_pct: float
    error: str | None = None


class TimelineVersionOut(BaseModel):
    version_num: int
    created_by: str
    checkpoint_note: str = ""
    created_at: str | None = None


class TimelineStateOut(BaseModel):
    active_version: int | None
    versions: list[TimelineVersionOut]


@app.get("/projects/{project_id}/jobs", response_model=list[JobOut])
def list_jobs(project_id: str) -> list[JobOut]:
    s = session()
    if s.get(ProjectRow, project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    rows = s.execute(
        select(JobRow).where(JobRow.project_id == project_id).order_by(JobRow.created_at)
    ).scalars()
    return [
        JobOut(id=r.id, stage=r.current_stage, progress_pct=r.progress_pct, error=r.error_message)
        for r in rows
    ]


@app.get("/projects/{project_id}/timeline", response_model=TimelineStateOut)
def timeline_state(project_id: str) -> TimelineStateOut:
    """Version history + active pointer (Director Studio timeline panel)."""
    s = session()
    row = s.get(ProjectRow, project_id)
    if row is None:
        raise HTTPException(status_code=404, detail="project not found")
    active = row.active_otio_version_id
    s.close()
    store = CheckpointStore(_session_factory)  # type: ignore[arg-type]
    versions = store.list_versions(project_id)
    return TimelineStateOut(
        active_version=active, versions=[TimelineVersionOut(**v) for v in versions]
    )


# ------------------------------------------------------------------ SSE progress
@app.get("/projects/{project_id}/jobs/{job_id}/events")
def job_events(
    project_id: str,
    job_id: str,
    poll_s: float = DEFAULT_POLL_S,
    max_seconds: float = DEFAULT_MAX_S,
) -> StreamingResponse:
    """Server-Sent Events stream of pipeline job progress (M2)."""
    require_job(session(), project_id, job_id)  # 404 before the stream opens

    async def gen():
        async for frame in job_progress_stream(
            job_id, _session_factory, poll_s=poll_s, max_seconds=max_seconds
        ):
            yield frame

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------ agent dispatch
class AgentDispatchRequest(BaseModel):
    """One agent tool call, executed checkpoint-first against the active timeline."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    checkpoint_note: str = ""


class AgentDispatchResponse(BaseModel):
    ok: bool
    agent: str
    tool: str
    version_before: int | None = None
    version: int | None = None
    rolled_back_to: int | None = None
    detail: str = ""


@app.post("/projects/{project_id}/agent/dispatch", response_model=AgentDispatchResponse)
def agent_dispatch(project_id: str, body: AgentDispatchRequest) -> AgentDispatchResponse:
    """Checkpoint-before-execute dispatch of one director tool (PLAN §3.3).

    Loads the project's active timeline version, binds the DirectorToolbox,
    runs the step through the typed registry inside a PlannerExecutor, and
    advances the active-version pointer to whatever state survives (the
    post-step state, or the rolled-back state on failure).
    """
    s = session()
    try:
        project = s.get(ProjectRow, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        active = project.active_otio_version_id
    finally:
        s.close()
    if active is None:
        raise HTTPException(
            status_code=422, detail="project has no timeline yet — run the pipeline first"
        )

    factory = _session_factory
    if factory is None:  # session() above always initializes it; belt and braces
        raise HTTPException(status_code=500, detail="session factory not initialized")

    store = CheckpointStore(factory)
    timeline = store.load_version(project_id, active)
    toolbox = DirectorToolbox(timeline, project_id)
    registry = build_director_registry(toolbox)
    executor = PlannerExecutor(registry, store, project_id, agent="director_agent", role="director")
    run = executor.execute_plan(
        [ToolStep(tool=body.tool, args=body.args, checkpoint_note=body.checkpoint_note)],
        timeline,
    )

    step = run.steps[-1] if run.steps else None
    detail = step.detail if step else "no steps executed"

    # advance the pointer to the surviving state (post-step OR rolled-back)
    if run.final_version is not None:
        with factory() as s2:
            p = s2.get(ProjectRow, project_id)
            p.active_otio_version_id = run.final_version
            s2.commit()

    return AgentDispatchResponse(
        ok=run.ok,
        agent=run.agent,
        tool=body.tool,
        version_before=step.version_before if step else None,
        version=run.final_version,
        rolled_back_to=run.rolled_back_to,
        detail=detail,
    )


# ------------------------------------------------------------------ LLM plan loop
class AgentPlanRequest(BaseModel):
    """Full LLM loop trigger: propose steps via the configured provider, execute."""

    role: AgentRole = "director"
    niche: str = "bodycam_documentary"


class AgentPlanStepOut(BaseModel):
    tool: str
    ok: bool
    detail: str = ""
    version_before: int | None = None


class AgentPlanResponse(BaseModel):
    ok: bool
    agent: str
    provider: str
    model: str
    steps: list[AgentPlanStepOut] = Field(default_factory=list)
    steps_ok: int = 0
    final_version: int | None = None
    rolled_back_to: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    detail: str = ""


@app.post("/projects/{project_id}/agent/plan", response_model=AgentPlanResponse)
def agent_plan(project_id: str, body: AgentPlanRequest | None = None) -> AgentPlanResponse:
    """Real provider loop (PLAN §3.1): propose -> checkpoint-first execute.

    `build_backend()` picks the provider from env: native Google AI Studio
    (GeminiRestBackend) when LLM_BASE_URL is blank, an OpenAI-compatible
    gateway when set, always against the DirectorToolbox registry. Tool results
    are replayed into the conversation when the backend supports multi-turn
    (`record_results` — Gemini 3.x thoughtSignature turns). 503 when no
    provider is configured; tests inject ScriptedBackend at the port instead.
    """
    body = body or AgentPlanRequest()
    s = session()
    try:
        project = s.get(ProjectRow, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        active = project.active_otio_version_id
    finally:
        s.close()
    if active is None:
        raise HTTPException(
            status_code=422, detail="project has no timeline yet — run the pipeline first"
        )
    factory = _session_factory
    if factory is None:
        raise HTTPException(status_code=500, detail="session factory not initialized")

    store = CheckpointStore(factory)
    timeline = store.load_version(project_id, active)
    toolbox = DirectorToolbox(timeline, project_id)
    registry = build_director_registry(toolbox)
    try:
        backend = build_backend(registry, body.niche)
    except LLMBackendError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    executor = PlannerExecutor(
        registry, store, project_id, agent=f"{body.role}_agent", role=body.role
    )
    run = executor.run(backend, timeline)

    if run.final_version is not None:
        with factory() as s2:
            p = s2.get(ProjectRow, project_id)
            p.active_otio_version_id = run.final_version
            s2.commit()

    usage = getattr(backend, "last_usage", None)
    usage = usage if isinstance(usage, dict) else {}
    return AgentPlanResponse(
        ok=run.ok,
        agent=run.agent,
        provider=str(getattr(backend, "provider", "scripted")),
        model=str(getattr(backend, "model", "")),
        steps=[
            AgentPlanStepOut(
                tool=st.tool, ok=st.ok, detail=st.detail, version_before=st.version_before
            )
            for st in run.steps
        ],
        steps_ok=run.steps_ok,
        final_version=run.final_version,
        rolled_back_to=run.rolled_back_to,
        tokens_in=usage.get("tokens_in"),
        tokens_out=usage.get("tokens_out"),
        detail="plan executed" if run.ok else f"rolled back to version {run.rolled_back_to}",
    )


# ------------------------------------------------------------------ cost dashboard
class CostReport(BaseModel):
    project_id: str
    tool_calls: int
    total_duration_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    token_cost_usd: float = 0.0
    recent: list[dict[str, Any]] = Field(default_factory=list)


@app.get("/projects/{project_id}/costs", response_model=CostReport)
def project_costs(project_id: str, limit: int = 20) -> CostReport:
    """M5 cost dashboard feed: durations + token spend from `agent_tool_logs`."""
    s = session()
    if s.get(ProjectRow, project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    row = s.execute(
        select(
            func.count(AgentToolLogRow.id),
            func.coalesce(func.sum(AgentToolLogRow.duration_ms), 0),
            func.coalesce(func.sum(AgentToolLogRow.tokens_in), 0),
            func.coalesce(func.sum(AgentToolLogRow.tokens_out), 0),
            func.coalesce(func.sum(AgentToolLogRow.token_cost_usd), 0.0),
        ).where(AgentToolLogRow.project_id == project_id)
    ).one()
    s.close()
    store = CheckpointStore(_session_factory)  # type: ignore[arg-type]
    return CostReport(
        project_id=project_id,
        tool_calls=row[0],
        total_duration_ms=int(row[1] or 0),
        tokens_in=int(row[2] or 0) or None,
        tokens_out=int(row[3] or 0) or None,
        token_cost_usd=float(row[4] or 0.0),
        recent=store.recent_logs(project_id, limit=limit),
    )


# ------------------------------------------------------------------ multi-channel publish
class ChannelPublishRequest(BaseModel):
    """Fan the approved master out to additional platforms (M5)."""

    video_path: str
    title: str = Field(min_length=1, max_length=100)
    description: str = ""


class ChannelPublishResponse(BaseModel):
    project_id: str
    channels: list[dict[str, Any]]


_multichannel = MultiChannelPublisher()


def set_channel_publisher(publisher: ChannelPublisher) -> None:
    """Register/replace one platform adapter (tests + production wiring)."""
    _multichannel.register(publisher)


@app.post("/projects/{project_id}/publish-channels", response_model=ChannelPublishResponse)
def publish_channels(project_id: str, body: ChannelPublishRequest) -> ChannelPublishResponse:
    """Report per-channel publish status — unconfigured channels never fake it."""
    s = session()
    try:
        if s.get(ProjectRow, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
    finally:
        s.close()
    if not Path(body.video_path).exists():
        raise HTTPException(status_code=422, detail=f"video not found: {body.video_path}")
    results = _multichannel.publish(Path(body.video_path), body.title, body.description)
    return ChannelPublishResponse(
        project_id=project_id,
        channels=[
            {
                "channel": r.channel.value,
                "status": r.status,
                "external_id": r.external_id,
                "detail": r.detail,
            }
            for r in results
        ],
    )


# ------------------------------------------------------------------ publish gate
class PublishDraftRequest(BaseModel):
    """Human Review Gate output: approve the cut as a YouTube Private Draft."""

    video_path: str
    title: str = Field(min_length=1, max_length=100)
    base_description: str = ""
    srt_path: str | None = None
    chapter_titles: list[str] = Field(default_factory=list)
    chapter_starts_s: list[float] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class PublishDraftResponse(BaseModel):
    project_id: str
    video_id: str
    privacy_status: str
    caption_attached: bool


@app.post("/projects/{project_id}/publish-draft", response_model=PublishDraftResponse)
def publish_draft(project_id: str, body: PublishDraftRequest) -> PublishDraftResponse:
    s = session()
    row = s.execute(select(ProjectRow).where(ProjectRow.id == project_id)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="project not found")

    if len(body.chapter_titles) != len(body.chapter_starts_s):
        raise HTTPException(status_code=422, detail="chapter titles/stamps length mismatch")
    chapters = [
        Chapter(title=t, start_s=st)
        for t, st in zip(body.chapter_titles, body.chapter_starts_s, strict=True)
    ]
    try:
        metadata = DraftMetadata.build(
            title=body.title,
            base_description=body.base_description,
            chapters=chapters,
            tags=body.tags,
            footer="Rendered by DirectorOS — private review draft.",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        result = _publisher.upload_draft(
            Path(body.video_path), metadata, Path(body.srt_path) if body.srt_path else None
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"publisher failed: {e}") from e

    row.youtube_draft_id = result.video_id
    row.status = ProjectStatus.published_draft.value
    s.commit()
    return PublishDraftResponse(
        project_id=project_id,
        video_id=result.video_id,
        privacy_status=result.privacy_status,
        caption_attached=result.caption_attached,
    )


# ------------------------------------------------------------------ studio media (M3 close-out)
class AssetOut(BaseModel):
    id: str
    role: str
    duration_s: float
    width: int
    height: int
    has_proxy: bool


def _primary_asset(s: Session, project_id: str) -> MediaAssetRow:
    if s.get(ProjectRow, project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    row = s.execute(
        select(MediaAssetRow)
        .where(MediaAssetRow.project_id == project_id)
        .order_by(MediaAssetRow.id)
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="project has no ingested asset yet")
    return row


def _proxy_file(key: str | None, name: str, sibling: str | None = None) -> Path:
    """Resolve a stored proxy/audio key to a local path (local backend serves direct)."""
    if not key:
        raise HTTPException(status_code=404, detail=f"no {name} for this asset — run the pipeline")
    p = Path(key)
    if sibling is not None:
        p = p.with_name(sibling)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{name} missing on disk: {p.name}")
    return p


@app.get("/projects/{project_id}/assets", response_model=list[AssetOut])
def list_assets(project_id: str) -> list[AssetOut]:
    s = session()
    if s.get(ProjectRow, project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    rows = s.execute(
        select(MediaAssetRow)
        .where(MediaAssetRow.project_id == project_id)
        .order_by(MediaAssetRow.id)
    ).scalars()
    return [
        AssetOut(
            id=r.id,
            role=r.role,
            duration_s=r.duration_s,
            width=r.width,
            height=r.height,
            has_proxy=bool(r.r2_proxy_key and Path(r.r2_proxy_key).exists()),
        )
        for r in rows
    ]


@app.get("/assets/{asset_id}/proxy")
def asset_proxy(asset_id: str) -> FileResponse:
    """480p web proxy — the scrub target for the Director Studio player."""
    s = session()
    row = s.get(MediaAssetRow, asset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="asset not found")
    s.close()
    return FileResponse(_proxy_file(row.r2_proxy_key, "proxy"), media_type="video/mp4")


@app.get("/assets/{asset_id}/scrubsheet")
def asset_scrubsheet(asset_id: str) -> FileResponse:
    """Thumbnail sprite strip rendered beside proxy.mp4 (deterministic sibling)."""
    s = session()
    row = s.get(MediaAssetRow, asset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="asset not found")
    s.close()
    return FileResponse(
        _proxy_file(row.r2_proxy_key, "scrubsheet", sibling="scrubsheet.jpg"),
        media_type="image/jpeg",
    )


# ------------------------------------------------------------------ subtitle grid (M3 close-out)
class CueOut(BaseModel):
    index: int
    text: str
    start_ms: int
    end_ms: int
    speaker: str


class CueEditRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    speaker: str | None = Field(default=None, min_length=1, max_length=80)


def _subtitle_dir(project_id: str) -> Path:
    settings = get_settings()
    return settings.output_dir / "pipeline" / project_id


@app.get("/projects/{project_id}/subtitles", response_model=list[CueOut])
def get_subtitles(project_id: str) -> list[CueOut]:
    """Structured cue list for the studio subtitle grid."""
    from director_os.subtitles.store import SubtitleStoreError, load_cues

    s = session()
    if s.get(ProjectRow, project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    s.close()
    try:
        cues = load_cues(_subtitle_dir(project_id))
    except SubtitleStoreError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return [
        CueOut(index=i, text=c.text, start_ms=c.start_ms, end_ms=c.end_ms, speaker=c.speaker)
        for i, c in enumerate(cues)
    ]


@app.patch("/projects/{project_id}/subtitles/{cue_index}", response_model=CueOut)
def edit_subtitle(project_id: str, cue_index: int, body: CueEditRequest) -> CueOut:
    """Proofread one cue — difflib word re-alignment (guardrail 5), artifacts rebuilt."""
    from director_os.subtitles.store import SubtitleStoreError, edit_cue

    s = session()
    if s.get(ProjectRow, project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    s.close()
    try:
        beat = edit_cue(_subtitle_dir(project_id), cue_index, body.text, speaker=body.speaker)
    except SubtitleStoreError as e:
        detail = str(e)
        raise HTTPException(
            status_code=404 if "out of range" in detail else 422, detail=detail
        ) from e
    return CueOut(
        index=cue_index,
        text=beat.text,
        start_ms=beat.start_ms,
        end_ms=beat.end_ms,
        speaker=beat.speaker,
    )


@app.get("/projects/{project_id}/subtitles/srt")
def subtitles_srt(project_id: str) -> FileResponse:
    s = session()
    if s.get(ProjectRow, project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    s.close()
    srt = _subtitle_dir(project_id) / "subtitles.srt"
    if not srt.exists():
        raise HTTPException(status_code=404, detail="no subtitles yet — run the pipeline")
    return FileResponse(srt, media_type="application/x-subrip", filename="subtitles.srt")


# ------------------------------------------------------------------ blur canvas (M3 close-out)
class RedactionIn(BaseModel):
    """One studio-drawn blur box, in RAW-FOOTAGE pixel coordinates."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    sigma: int = Field(default=24, ge=1, le=200)
    label: str = Field(default="", max_length=120)


class RedactionOut(RedactionIn):
    id: str
    created_by: str
    created_at: str


def _redaction_regions(s: Session, asset_id: str) -> list[dict[str, Any]]:
    row = s.get(PerceptionIndexRow, asset_id)
    if row is None:
        s.add(PerceptionIndexRow(asset_id=asset_id, redaction_tracks_json={"regions": []}))
        s.commit()
        return []
    return list((row.redaction_tracks_json or {}).get("regions", []))


def _save_redaction_regions(s: Session, asset_id: str, regions: list[dict[str, Any]]) -> None:
    row = s.get(PerceptionIndexRow, asset_id)
    assert row is not None  # created by _redaction_regions above
    row.redaction_tracks_json = {"regions": regions}
    s.commit()


@app.get("/projects/{project_id}/redactions", response_model=list[RedactionOut])
def list_redactions(project_id: str) -> list[RedactionOut]:
    s = session()
    try:
        asset = _primary_asset(s, project_id)
        regions = _redaction_regions(s, asset.id)
    finally:
        s.close()
    return [RedactionOut(**r) for r in regions]


@app.post("/projects/{project_id}/redactions", response_model=RedactionOut, status_code=201)
def add_redaction(project_id: str, body: RedactionIn) -> RedactionOut:
    """Draw a blur box. Pixel coords are clamped/validated against the raw dims."""
    from director_os.core.models import BBox, BlurRegion
    from director_os.vision_fx.redact import validate_regions

    s = session()
    try:
        asset = _primary_asset(s, project_id)
        frame_w = asset.width or 0
        frame_h = asset.height or 0
        if frame_w <= 0 or frame_h <= 0:
            raise HTTPException(status_code=409, detail="asset dimensions unknown — re-ingest")
        probe_region = BlurRegion(
            scope="global", bbox=BBox(x=body.x, y=body.y, w=body.w, h=body.h), sigma=body.sigma
        )
        if validate_regions([probe_region], frame_w, frame_h) == 0:
            raise HTTPException(
                status_code=422,
                detail=f"box fully outside frame ({frame_w}x{frame_h}) — check coordinates",
            )
        regions = _redaction_regions(s, asset.id)
        region = {
            "id": new_id(),
            "x": body.x,
            "y": body.y,
            "w": body.w,
            "h": body.h,
            "sigma": body.sigma,
            "label": body.label,
            "created_by": "human_edit",
            "created_at": datetime.now(UTC).isoformat(),
        }
        regions.append(region)
        _save_redaction_regions(s, asset.id, regions)
    finally:
        s.close()
    return RedactionOut.model_validate(region)


@app.delete("/projects/{project_id}/redactions/{region_id}", status_code=204)
def delete_redaction(project_id: str, region_id: str) -> Response:
    s = session()
    try:
        asset = _primary_asset(s, project_id)
        regions = _redaction_regions(s, asset.id)
        kept = [r for r in regions if r.get("id") != region_id]
        if len(kept) == len(regions):
            raise HTTPException(status_code=404, detail="redaction region not found")
        _save_redaction_regions(s, asset.id, kept)
    finally:
        s.close()
    return Response(status_code=204)


# ------------------------------------------------------------------ ingest by URL (M3)
class IngestRequest(BaseModel):
    """M3 done-when: paste a media URL in the browser — no terminal needed.

    `custom_headers` ride on top of the mandatory DirectorOS identity headers
    (cookies / referer for FOIA portals that gate downloads)."""

    url: str
    custom_headers: dict[str, str] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    job_id: str
    source_path: str
    enqueued: bool
    detail: str


def _source_filename(url: str) -> str:
    """Deterministic, filesystem-safe name for a downloaded media URL."""
    from urllib.parse import urlparse

    path = urlparse(url).path
    base = Path(path).name.strip()
    if not base:
        base = "media.mp4"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)
    if not safe.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi", ".mp3", ".wav")):
        safe += ".mp4"
    return safe


@app.post("/projects/{project_id}/ingest", response_model=IngestResponse, status_code=202)
def ingest_url(project_id: str, body: IngestRequest) -> IngestResponse:
    """Paste-a-URL ingest: download server-side (kill-safe .part rename),
    register a queued pipeline job, hand it to the worker fleet.

    The browser watches the returned `job_id` over SSE; the worker advances
    ingest -> perception -> render (crash-safe stages)."""
    s = session()
    try:
        if s.get(ProjectRow, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
    finally:
        s.close()
    if not body.url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="url must be http(s)")

    dest_dir = get_settings().output_dir / "pipeline" / project_id / "sources"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _source_filename(body.url)
    try:
        source = download(body.url, dest, extra_headers=body.custom_headers or None)
    except Exception as e:  # httpx errors, bad status codes, disk failures
        raise HTTPException(status_code=502, detail=f"media download failed: {e}") from e

    s = session()
    try:
        job = JobRow(project_id=project_id, current_stage="ingest", progress_pct=0.0)
        s.add(job)
        s.commit()
        job_id = job.id
    finally:
        s.close()

    enqueued, detail = False, "job queued — worker fleet will advance it"
    try:
        from director_os.worker.celery_app import celery_app

        celery_app.send_task(
            "directoros.advance_job",
            args=[job_id, project_id, str(source), get_settings().database_url],
            retry=False,
        )
        enqueued = True
        detail = "queued on the worker fleet"
    except Exception as e:  # broker down / worker package absent: stays queued
        detail = (
            f"job queued — fleet enqueue failed ({type(e).__name__}); "
            "start the worker fleet to advance"
        )
    return IngestResponse(job_id=job_id, source_path=str(source), enqueued=enqueued, detail=detail)
