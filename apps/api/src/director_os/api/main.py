"""FastAPI gateway skeleton (PLAN §2 layer 2).

M1 exposes health + project creation against the core persistence layer.
M2 adds SSE job progress, agent tool dispatch endpoints and the Human Review Gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from director_os.agent_runtime.agents import PlannerExecutor, ToolStep
from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.agent_runtime.editing_tools import DirectorToolbox, build_director_registry
from director_os.api.events import DEFAULT_MAX_S, DEFAULT_POLL_S, job_progress_stream, require_job
from director_os.api.youtube import (
    Chapter,
    DraftMetadata,
    InMemoryYouTubePublisher,
    YouTubePublisher,
)
from director_os.core.db import JobRow, ProjectRow, init_models, make_engine, make_session_factory
from director_os.core.models import ProjectStatus
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
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
