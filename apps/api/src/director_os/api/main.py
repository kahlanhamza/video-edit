"""FastAPI gateway skeleton (PLAN §2 layer 2).

M1 exposes health + project creation against the core persistence layer.
M2 adds SSE job progress, agent tool dispatch endpoints and the Human Review Gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from director_os.agent_runtime.agents import AgentRole, PlannerExecutor, ToolStep
from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.agent_runtime.editing_tools import DirectorToolbox, build_director_registry
from director_os.agent_runtime.llm import LLMBackendError, build_backend_chain
from director_os.api.channels import ChannelPublisher, MultiChannelPublisher
from director_os.api.events import (
    DEFAULT_LIVE_MAX_S,
    DEFAULT_LIVE_POLL_S,
    DEFAULT_MAX_S,
    DEFAULT_POLL_S,
    job_live_stream,
    job_progress_stream,
    require_job,
)
from director_os.api.youtube import (
    Chapter,
    DraftMetadata,
    InMemoryYouTubePublisher,
    YouTubePublisher,
)
from director_os.core.config import get_settings
from director_os.core.credentials import (
    CredentialRow,
    apply_credentials_to_settings,
    delete_credential,
    encryption_active,
    mask_secret,
    upsert_credentials,
)
from director_os.core.db import (
    AgentToolLogRow,
    CaseDossierRow,
    CaseSourceRow,
    JobEventRow,
    JobLogRow,
    JobRow,
    JobStepRow,
    MediaAssetRow,
    PerceptionIndexRow,
    ProjectRow,
    TimelineVersionRow,
    init_models,
    make_engine,
    make_session_factory,
    new_id,
)
from director_os.core.http import download
from director_os.core.models import ProjectStatus
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete as sa_delete
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
        # Studio-saved credentials (Settings UI) win over .env from here on.
        apply_credentials_to_settings(_engine)
    return _session_factory()


class SourceIn(BaseModel):
    """One attached case source at project creation."""

    kind: Literal["url", "text", "source_page"]
    label: str = Field(default="", max_length=300)
    url: str = Field(default="", max_length=2048)
    text: str = Field(default="", max_length=200_000)


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    niche_skill: str = "bodycam_documentary"
    video_urls: list[str] = Field(default_factory=list, max_length=8)
    auto_discover: bool = True
    sources: list[SourceIn] = Field(default_factory=list, max_length=32)


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


def _add_source_rows(project_id: str, body: ProjectCreate) -> int:
    """Persist creation-time sources + auto-discover source_page rows."""
    s = session()
    try:
        existing_urls = {
            u
            for (u,) in s.execute(
                select(CaseSourceRow.uri_or_storage_key).where(
                    CaseSourceRow.project_id == project_id
                )
            )
        }
        added = 0

        def _add(kind: str, label: str, uri: str = "", text: str = "") -> None:
            nonlocal added
            if uri and uri in existing_urls:
                return
            if uri:
                existing_urls.add(uri)
            s.add(
                CaseSourceRow(
                    project_id=project_id,
                    kind=kind,
                    label=label,
                    uri_or_storage_key=uri,
                    text=text,
                    status="pending",
                )
            )
            added += 1

        for src in body.sources:
            if src.kind == "text":
                if src.text.strip():
                    _add("text", src.label or "pasted notes", text=src.text)
            elif src.url.strip():
                _add(
                    "url" if src.kind == "url" else "source_page",
                    src.label or src.url[:120],
                    uri=src.url.strip(),
                )
        if body.auto_discover:
            for vurl in body.video_urls:
                if vurl.strip():
                    _add("source_page", f"video page: {vurl[:120]}", uri=vurl.strip())
        s.commit()
        return added
    finally:
        s.close()


# ------------------------------------------------------- job dispatch (one path for every trigger)
_in_flight: set[str] = set()
_in_flight_lock = __import__("threading").Lock()


def _dispatch_advance(
    job_id: str | None, project_id: str, source: str, *, force_replan: bool = False
) -> tuple[bool, str]:
    """Celery when the broker accepts the task, inline gateway thread otherwise.

    The inline path is what makes `make api` alone enough for the Studio: the
    API process runs the exact `advance_job` entrypoint the worker fleet runs,
    with identical resume semantics. Double-dispatch of the same job is
    guarded (the pipeline is resumable, not re-entrant). Returns
    (dispatched, detail).
    """
    args = [job_id, project_id, source, get_settings().database_url]
    if force_replan:
        args = [project_id, source, get_settings().database_url]
    try:
        from director_os.worker.celery_app import celery_app

        task = "directoros.replan_job" if force_replan else "directoros.advance_job"
        celery_app.send_task(task, args=args, retry=False)
        return True, "queued on the worker fleet"
    except Exception:  # noqa: BLE001 — no broker must never 500 the Studio
        pass

    if force_replan:
        _thread_replan(project_id, source)
        return True, "no broker — replan running inline in the gateway process"

    if job_id is not None:
        with _in_flight_lock:
            if job_id in _in_flight:
                return True, "job already running — watching progress"
            _in_flight.add(job_id)

    def _work() -> None:
        from director_os.worker.pipeline import advance_job

        try:
            advance_job(job_id, project_id, source, get_settings().database_url)
        except Exception as e:  # noqa: BLE001 — run_pipeline persisted the failure
            print(f"[inline pipeline] job {str(job_id)[:8]} failed: {e}", flush=True)
        finally:
            if job_id is not None:
                with _in_flight_lock:
                    _in_flight.discard(job_id)

    import threading

    threading.Thread(target=_work, daemon=True, name=f"pipeline-{str(job_id)[:8]}").start()
    return True, "no broker — pipeline running inline in the gateway process"


def _thread_replan(project_id: str, source: str) -> None:
    def _work() -> None:
        from director_os.worker.pipeline import advance_job

        try:
            advance_job(None, project_id, source, get_settings().database_url, force_replan=True)
        except Exception as e:  # noqa: BLE001
            print(f"[inline replan] {project_id[:8]} failed: {e}", flush=True)

    import threading

    threading.Thread(target=_work, daemon=True, name=f"replan-{project_id[:8]}").start()


def _validate_http_url(url: str, what: str) -> str:
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=422, detail=f"{what} must be an http(s) URL, got: {url[:120]}"
        )
    return url


@app.post("/projects", status_code=201, response_model=ProjectOut)
def create_project(body: ProjectCreate) -> ProjectOut:
    s = session()
    row = ProjectRow(
        title=body.title, niche_skill=body.niche_skill, status=ProjectStatus.created.value
    )
    s.add(row)
    s.commit()
    pid = row.id
    s.close()
    _add_source_rows(pid, body)
    s = session()
    fresh = s.get(ProjectRow, pid)
    assert fresh is not None
    out = ProjectOut(
        id=fresh.id, title=fresh.title, niche_skill=fresh.niche_skill, status=fresh.status
    )
    s.close()
    return out


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


# ------------------------------------------------------------------ case sources (M5)
class CaseSourceIn(BaseModel):
    """Attach one more source to an existing case."""

    kind: Literal["url", "text", "source_page"]
    label: str = Field(default="", max_length=300)
    url: str = Field(default="", max_length=2048)
    text: str = Field(default="", max_length=200_000)


class CaseSourceOut(BaseModel):
    id: str
    kind: str
    label: str
    status: str
    error: str | None = None
    created_at: str


def _source_out(r: CaseSourceRow) -> CaseSourceOut:
    return CaseSourceOut(
        id=r.id,
        kind=r.kind,
        label=r.label or "",
        status=r.status,
        error=r.error,
        created_at=r.created_at.isoformat() if r.created_at else "",
    )


@app.get("/projects/{project_id}/sources", response_model=list[CaseSourceOut])
def list_sources(project_id: str) -> list[CaseSourceOut]:
    s = session()
    try:
        if s.get(ProjectRow, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        rows = s.execute(
            select(CaseSourceRow)
            .where(CaseSourceRow.project_id == project_id)
            .order_by(CaseSourceRow.created_at)
        ).scalars()
        return [_source_out(r) for r in rows]
    finally:
        s.close()


@app.post("/projects/{project_id}/sources", status_code=201, response_model=CaseSourceOut)
def add_source(project_id: str, body: CaseSourceIn) -> CaseSourceOut:
    """Attach a URL / source page / pasted text to an existing case."""
    s = session()
    try:
        if s.get(ProjectRow, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        if body.kind == "text":
            if not body.text.strip():
                raise HTTPException(status_code=422, detail="text must not be empty")
            row = CaseSourceRow(
                project_id=project_id,
                kind="text",
                label=body.label or "pasted notes",
                text=body.text,
                status="pending",
            )
        else:
            if not body.url.strip():
                raise HTTPException(status_code=422, detail=f"{body.kind} source requires url")
            dup = s.execute(
                select(CaseSourceRow)
                .where(CaseSourceRow.project_id == project_id)
                .where(CaseSourceRow.uri_or_storage_key == body.url.strip())
            ).scalar_one_or_none()
            if dup is not None:
                return _source_out(dup)
            row = CaseSourceRow(
                project_id=project_id,
                kind="url" if body.kind == "url" else "source_page",
                label=body.label or body.url[:120],
                uri_or_storage_key=body.url.strip(),
                status="pending",
            )
        s.add(row)
        s.commit()
        return _source_out(row)
    finally:
        s.close()


UPLOAD_EXTENSIONS = {".pdf": "pdf", ".txt": "text", ".md": "text"}


@app.post("/projects/{project_id}/sources/upload", status_code=201, response_model=CaseSourceOut)
async def upload_source(
    project_id: str, file: UploadFile = File(...), label: str = ""
) -> CaseSourceOut:
    """Upload a .pdf / .txt / .md case source (stored under the project dir)."""
    s = session()
    try:
        if s.get(ProjectRow, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        name = Path(file.filename or "source").name
        ext = Path(name).suffix.lower()
        if ext not in UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=422, detail=f"unsupported file type {ext} (pdf/txt/md)")
        kind = UPLOAD_EXTENSIONS[ext]
        row = CaseSourceRow(project_id=project_id, kind=kind, label=label or name, status="pending")
        s.add(row)
        s.flush()
        dest_dir = get_settings().output_dir / "pipeline" / project_id / "sources"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{row.id[:8]}_{name}"
        content = await file.read()
        dest.write_bytes(content)
        if kind == "text":
            row.text = content.decode("utf-8", errors="ignore")[:200_000]
        else:
            row.uri_or_storage_key = str(dest)
        s.commit()
        return _source_out(row)
    finally:
        s.close()


@app.get("/projects/{project_id}/dossier")
def get_dossier(project_id: str) -> JSONResponse:
    s = session()
    try:
        if s.get(ProjectRow, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        row = s.get(CaseDossierRow, project_id)
        if row is None or not row.dossier_json:
            raise HTTPException(
                status_code=404, detail="no dossier yet — attach sources and refresh"
            )
        return JSONResponse(row.dossier_json)
    finally:
        s.close()


class RefreshResponse(BaseModel):
    refreshed: int
    dossier: dict[str, Any] | None = None
    replan: str = ""


@app.post("/projects/{project_id}/sources/refresh", response_model=RefreshResponse)
def refresh_sources(project_id: str) -> RefreshResponse:
    """Re-extract pending/failed sources, rebuild the dossier, re-run Stage 6+.

    The video is never re-downloaded and Stage 3 ASR never re-runs — the
    worker's replan task short-circuits on the persisted domain rows. Adding
    a source only refreshes the CaseDossier and re-plans the documentary.
    """
    from director_os.worker.pipeline import PipelineDeps, make_deps, stage_dossier

    s = session()
    try:
        if s.get(ProjectRow, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
    finally:
        s.close()

    deps: PipelineDeps = make_deps(get_settings().database_url)
    n_before = _count_ready_sources(project_id)
    dossier = stage_dossier(project_id, deps)
    n_after = _count_ready_sources(project_id)

    replan_note = ""
    if _has_ingested_asset(project_id):
        replan_note = _dispatch_replan(project_id)
    return RefreshResponse(
        refreshed=max(0, n_after - n_before), dossier=dossier, replan=replan_note
    )


def _count_ready_sources(project_id: str) -> int:
    s = session()
    try:
        return int(
            s.execute(
                select(func.count(CaseSourceRow.id)).where(
                    CaseSourceRow.project_id == project_id,
                    CaseSourceRow.status == "ready",
                )
            ).scalar_one()
        )
    finally:
        s.close()


def _has_ingested_asset(project_id: str) -> bool:
    s = session()
    try:
        row = s.execute(
            select(MediaAssetRow.id).where(MediaAssetRow.project_id == project_id).limit(1)
        ).scalar_one_or_none()
        return row is not None
    finally:
        s.close()


def _dispatch_replan(project_id: str) -> str:
    """Best-effort worker dispatch; never fails the refresh request."""
    s = session()
    try:
        asset = s.execute(
            select(MediaAssetRow)
            .where(MediaAssetRow.project_id == project_id)
            .order_by(MediaAssetRow.id)
            .limit(1)
        ).scalar_one()
    finally:
        s.close()
    enqueued, detail = _dispatch_advance(
        None, project_id, asset.source_url or "", force_replan=True
    )
    return f"replan dispatched ({detail}) — Stage 6+ re-runs, ASR never re-runs"


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


# ------------------------------------------------- live console (full-visibility Studio)
@app.get("/projects/{project_id}/jobs/{job_id}/snapshot")
def job_live_snapshot(project_id: str, job_id: str) -> JSONResponse:
    """Initial console render: job state + every step + the log tail."""
    require_job(session(), project_id, job_id)  # 404-safe
    s = session()
    try:
        steps = s.execute(
            select(JobStepRow).where(JobStepRow.job_id == job_id).order_by(JobStepRow.id)
        ).scalars()
        step_rows = [
            {
                "id": r.id,
                "key": r.key,
                "label": r.label,
                "status": r.status,
                "pct": r.pct,
                "detail": r.detail,
                "order": r.order,
                "duration_ms": r.duration_ms,
            }
            for r in steps
        ]
        logs = s.execute(
            select(JobLogRow)
            .where(JobLogRow.job_id == job_id)
            .order_by(JobLogRow.id.desc())
            .limit(400)
        ).scalars()
        log_rows = [
            {"id": r.id, "level": r.level, "message": r.message} for r in reversed(list(logs))
        ]
    finally:
        s.close()
    return JSONResponse({"steps": step_rows, "logs": log_rows})


@app.get("/projects/{project_id}/jobs/{job_id}/live")
def job_live(
    project_id: str,
    job_id: str,
    poll_s: float = DEFAULT_LIVE_POLL_S,
    max_seconds: float = DEFAULT_LIVE_MAX_S,
) -> StreamingResponse:
    """SSE stream of EVERY pipeline event — per-step progress bars, console
    log lines, timeline revisions. Terminal `done` closes the stream."""
    require_job(session(), project_id, job_id)  # 404 before the stream opens

    async def gen():
        async for frame in job_live_stream(
            job_id, _session_factory, poll_s=poll_s, max_seconds=max_seconds
        ):
            yield frame

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/projects/{project_id}/timeline/nle")
def timeline_nle(project_id: str) -> JSONResponse:
    """The multi-track NLE snapshot: video clips, card overlays, narration
    audio, captions, redaction windows — everything the timeline renders."""
    from director_os.api.nle import ProjectNotFound, build_nle_snapshot

    factory = _session_factory
    if factory is None:  # session() above always initializes it; belt and braces
        raise HTTPException(status_code=500, detail="session factory not initialized")
    try:
        snapshot = build_nle_snapshot(factory, project_id, get_settings().output_dir)
    except ProjectNotFound as e:
        raise HTTPException(status_code=404, detail="project not found") from e
    return JSONResponse(snapshot)


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
        backend = build_backend_chain(registry, body.niche)
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


@app.get("/assets/{asset_id}/frame")
def asset_frame(
    asset_id: str,
    t: float = Query(default=0.0, ge=0.0),
    w: int = Query(default=640, ge=64, le=1920),
    src: Literal["auto", "proxy", "full"] = "auto",
) -> FileResponse:
    """Frame-accurate JPEG of ANY frame — the frame-by-frame primitive.

    The owner law (2026-11-23): the editor (human scrubber or agent edit)
    addresses individual frames; nothing re-renders until Export. Serves the
    480p proxy by default (cheap, scrub-grade); `src=full` decodes the
    original media at up to its native width. Every (t, width) is LRU-cached
    per asset, so stepping back and forth over the same frames is free.
    """
    from director_os.core.frames import FrameGrabError, grab_frame
    from director_os.core.proxy import PROXY_HEIGHT

    s = session()
    row = s.get(MediaAssetRow, asset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="asset not found")
    s.close()

    settings = get_settings()
    t_clamped = min(t, max(0.0, float(row.duration_s) - 0.001))
    cache_dir = settings.output_dir / "pipeline" / row.project_id / "assets" / asset_id / "frames"

    media: Path
    width: int | None
    if src == "full":
        if not row.source_url or not Path(row.source_url).exists():
            raise HTTPException(status_code=404, detail="full-res source not on disk")
        media, width = Path(row.source_url), min(w, row.width or w)
    else:
        media = _proxy_file(row.r2_proxy_key, "proxy")
        # never upscale the 480p proxy: decode native when the request covers it
        proxy_w = (
            int(round((row.width or 854) * PROXY_HEIGHT / (row.height or 480)))
            if row.height
            else 854
        )
        width = w if w < proxy_w else None

    try:
        frame = grab_frame(media, t_clamped, cache_dir=cache_dir, width=width)
    except FrameGrabError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return FileResponse(
        frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
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
    """Merge-save regions — the sweep payload (per_frame plan, plan_digest,
    keep_track_ids, ranges_digest) rides the SAME column and must survive
    every canvas edit. The old overwrite silently destroyed the tracked
    redaction plan on the first studio-drawn box."""
    row = s.get(PerceptionIndexRow, asset_id)
    if row is None:
        row = PerceptionIndexRow(asset_id=asset_id)
        s.add(row)
    row.redaction_tracks_json = {**(row.redaction_tracks_json or {}), "regions": regions}
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

    enqueued, detail = _dispatch_advance(job_id, project_id, str(source))
    return IngestResponse(job_id=job_id, source_path=str(source), enqueued=enqueued, detail=detail)


# ------------------------------------------------------- run pipeline (Studio one-button)
class RunRequest(BaseModel):
    """Start (or RESUME) the autonomous pipeline for a project.

    With a `video_url` the gateway downloads it server-side (kill-safe .part
    rename) exactly like /ingest, then dispatches. Without one, the project's
    registered media asset (or its newest unfinished job) is resumed — a
    failed step continues WHERE IT STOPPED; finished stages short-circuit
    (ASR never re-runs, cached segments never re-encode, fetched sources
    never re-scrape).
    """

    video_url: str | None = None
    custom_headers: dict[str, str] | None = None


class RunResponse(BaseModel):
    project_id: str
    job_id: str
    reused: bool
    dispatched: bool
    detail: str
    stage: str
    progress_pct: float


@app.post("/projects/{project_id}/run", response_model=RunResponse, status_code=202)
def run_project(project_id: str, body: RunRequest | None = None) -> RunResponse:
    """One button: scrape sources -> ingest video -> ASR -> cut -> render -> QA.

    Resume contract: the project's newest unfinished job is reused when no
    fresh media URL is given; every stage persists its product, so a retry
    continues from the failed stage instead of restarting from scratch.
    """
    body = body or RunRequest()

    if body.video_url is not None and body.video_url.strip():
        # fresh media: same kill-safe download path as /ingest
        vurl = _validate_http_url(body.video_url, "video_url")
        dest_dir = get_settings().output_dir / "pipeline" / project_id / "sources"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / _source_filename(vurl)
        try:
            source = download(vurl, dest, extra_headers=body.custom_headers or None)
        except Exception as e:  # httpx errors, bad status codes, disk failures
            raise HTTPException(status_code=502, detail=f"media download failed: {e}") from e
        source_str = str(source)

        s = session()
        try:
            if s.get(ProjectRow, project_id) is None:
                raise HTTPException(status_code=404, detail="project not found")
            job = JobRow(project_id=project_id, current_stage="ingest", progress_pct=0.0)
            s.add(job)
            s.commit()
            job_id = job.id
        finally:
            s.close()
        reused = False
    else:
        # resume path: newest unfinished job + its registered media asset
        s = session()
        try:
            if s.get(ProjectRow, project_id) is None:
                raise HTTPException(status_code=404, detail="project not found")
            last = s.execute(
                select(JobRow)
                .where(JobRow.project_id == project_id)
                .order_by(JobRow.created_at.desc(), JobRow.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            asset = s.execute(
                select(MediaAssetRow)
                .where(MediaAssetRow.project_id == project_id)
                .order_by(MediaAssetRow.id)
                .limit(1)
            ).scalar_one_or_none()
            # resume path: newest unfinished job + its registered media asset
            # (inline `last is not None` keeps mypy's narrowing flowing)
            if (
                last is not None
                and asset is not None
                and asset.source_url
                and (last.error_message is not None or (last.progress_pct or 0.0) < 100.0)
            ):
                job_id, source_str, reused = last.id, asset.source_url, True
                if last.error_message is not None:
                    # Resume = fresh attempt: clear the terminal error BEFORE
                    # dispatch so (a) the SSE live stream does not insta-close
                    # on the stale error and (b) the Studio shows the run as
                    # live again. The worker's RunTracker takes over from here.
                    last.error_message = None
                    s.add(
                        JobEventRow(
                            job_id=last.id,
                            project_id=project_id,
                            kind="job",
                            payload={
                                "stage": last.current_stage,
                                "progress_pct": round(float(last.progress_pct or 0.0), 1),
                                "error": None,
                                "resumed": True,
                            },
                        )
                    )
                    s.commit()
            elif asset is not None and asset.source_url:
                # finished pipeline but a finished job exists: re-advance is
                # harmless (every stage short-circuits) — but a fresh job is
                # cleaner for SSE tracking
                job = JobRow(project_id=project_id, current_stage="ingest", progress_pct=0.0)
                s.add(job)
                s.commit()
                job_id, source_str, reused = job.id, asset.source_url, False
            else:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "nothing to run yet — provide a video_url (or ingest one first) "
                        "so the pipeline has media to process"
                    ),
                )
        finally:
            s.close()

    # Run = explicit resume intent: clear any operator STOP flag so the
    # pipeline advances instead of parking at the first stage boundary
    stop_flag = get_settings().output_dir / "pipeline" / project_id / "STOP"
    stop_flag.unlink(missing_ok=True)

    dispatched, dispatch_detail = _dispatch_advance(job_id, project_id, source_str)
    s2 = session()
    try:
        job_row = s2.get(JobRow, job_id)
        stage = job_row.current_stage if job_row else "ingest"
        pct = job_row.progress_pct if job_row else 0.0
    finally:
        s2.close()
    return RunResponse(
        project_id=project_id,
        job_id=job_id,
        reused=reused,
        dispatched=dispatched,
        detail=dispatch_detail,
        stage=stage,
        progress_pct=pct,
    )


# ------------------------------------------------------- approve & render gate
class ApproveResponse(BaseModel):
    ok: bool
    job_id: str
    status: str
    version: int | None = None
    enqueued: bool
    detail: str


@app.post("/projects/{project_id}/stop", status_code=202)
def stop_project(project_id: str) -> dict[str, Any]:
    """Cooperative stop: parks the running job at the NEXT stage boundary.

    The checkpoint law keeps every completed stage — nothing restarts. Run
    (resume) clears the flag and continues from the parked stage. An ffmpeg
    already in flight finishes; the next boundary stops.
    """
    s = session()
    try:
        if s.get(ProjectRow, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
    finally:
        s.close()
    flag = get_settings().output_dir / "pipeline" / project_id / "STOP"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(datetime.now(UTC).isoformat())
    return {
        "stopped": True,
        "note": "job parks at the next stage boundary — completed stages are kept; Run resumes",
    }


@app.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str) -> Response:
    """Full teardown: every DB row (jobs, assets, perception index, timeline
    versions, dossier, sources, agent logs) + the pipeline output tree.
    The content-addressed segment cache is shared across projects and stays.
    """
    import shutil

    s = session()
    try:
        project = s.get(ProjectRow, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        asset_ids = list(
            s.execute(
                select(MediaAssetRow.id).where(MediaAssetRow.project_id == project_id)
            ).scalars()
        )
        if asset_ids:
            s.execute(
                sa_delete(PerceptionIndexRow).where(PerceptionIndexRow.asset_id.in_(asset_ids))
            )
        # observability tables reference jobs (FK) — purge before the jobs row
        s.execute(sa_delete(JobEventRow).where(JobEventRow.project_id == project_id))
        s.execute(sa_delete(JobLogRow).where(JobLogRow.project_id == project_id))
        s.execute(sa_delete(JobStepRow).where(JobStepRow.project_id == project_id))
        s.execute(sa_delete(JobRow).where(JobRow.project_id == project_id))
        s.execute(sa_delete(MediaAssetRow).where(MediaAssetRow.project_id == project_id))
        s.execute(sa_delete(TimelineVersionRow).where(TimelineVersionRow.project_id == project_id))
        s.execute(sa_delete(CaseDossierRow).where(CaseDossierRow.project_id == project_id))
        s.execute(sa_delete(CaseSourceRow).where(CaseSourceRow.project_id == project_id))
        s.execute(sa_delete(AgentToolLogRow).where(AgentToolLogRow.project_id == project_id))
        s.delete(project)
        s.commit()
    finally:
        s.close()
    tree = get_settings().output_dir / "pipeline" / project_id
    if tree.exists():
        shutil.rmtree(tree, ignore_errors=True)
    return Response(status_code=204)


@app.post("/projects/{project_id}/approve", response_model=ApproveResponse, status_code=202)
def approve_project(project_id: str) -> ApproveResponse:
    """Human gate (M3): approve the reviewed cut -> re-render ONLY dirty segments.

    Requires `awaiting_review` (QA passed, studio edits done through
    agent/dispatch + PATCH subtitles + drawn redactions, which advance the
    ACTIVE timeline version). The approve job re-runs the idempotent pipeline:
    planning reuses the active version pointer and the content-addressed
    segment cache serves every unchanged segment — cache misses (the dirty
    segments) are the only render work paid for. The browser watches the
    returned `job_id` over SSE exactly like an ingest job.
    """
    s = session()
    try:
        project = s.get(ProjectRow, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        if project.active_otio_version_id is None:
            raise HTTPException(
                status_code=422, detail="project has no timeline yet — run the pipeline first"
            )
        if project.status != ProjectStatus.awaiting_review.value:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"project status is '{project.status}' — the approve gate opens at "
                    "'awaiting_review' (run the pipeline to QA-pass first)"
                ),
            )
        asset = _primary_asset(s, project_id)
        source = asset.source_url
        if not source:
            raise HTTPException(status_code=422, detail="asset has no source path")
        project.status = ProjectStatus.approved.value
        job = JobRow(project_id=project_id, current_stage="render", progress_pct=0.0)
        s.add(job)
        s.commit()
        job_id = job.id
        version = project.active_otio_version_id
    finally:
        s.close()

    enqueued, dispatch_detail = _dispatch_advance(job_id, project_id, source)
    detail = (
        "approved — re-render dispatched (cache hits are free)"
        if enqueued
        else "approve dispatch failed"
    ) + f" ({dispatch_detail})"
    return ApproveResponse(
        ok=True, job_id=job_id, status="approved", version=version, enqueued=enqueued, detail=detail
    )


# ---------------------------------------------------------------------------
# Settings UI — provider credentials vault + live connection probes (ADR-0012)
# ---------------------------------------------------------------------------

_PROVIDER_SPECS: list[dict[str, Any]] = [
    {
        "id": "llm",
        "label": "LLM — the AI editor's brain",
        "probe": "llm",
        "hint": (
            "Any OpenAI-compatible endpoint works: Meta Muse (https://api.meta.ai/v1), "
            "OpenRouter, Groq, DeepSeek, Together, local Ollama (http://localhost:11434/v1) or "
            "LM Studio (http://localhost:1234/v1). "
            "Leave the Base URL BLANK for native Google AI Studio."
        ),
        "fields": [
            {"key": "LLM_API_KEY", "kind": "secret", "label": "API key"},
            {
                "key": "LLM_BASE_URL",
                "kind": "text",
                "label": "Base URL (blank = Google AI Studio)",
                "placeholder": "https://api.meta.ai/v1",
            },
            {"key": "LLM_MODEL", "kind": "text", "label": "Model", "placeholder": "muse-ember-1.0"},
            {
                "key": "LLM_FALLBACK_MODELS",
                "kind": "text",
                "label": "Fallback models (comma-separated)",
                "placeholder": "optional",
            },
            {
                "key": "LLM_BUDGET_USD",
                "kind": "text",
                "label": "Spend cap (USD)",
                "placeholder": "15.0",
            },
        ],
    },
    {
        "id": "gladia",
        "label": "Gladia — ASR + diarization (captions)",
        "probe": "gladia",
        "hint": (
            "Free tier at app.gladia.io. Without it the pipeline runs but skips "
            "transcription/captions."
        ),
        "fields": [{"key": "GLADIA_API_KEY", "kind": "secret", "label": "API key"}],
    },
    {
        "id": "vision",
        "label": "Vision LLM — frame understanding",
        "probe": "vision",
        "hint": (
            "Contact-sheet indexing + Frames-on-Demand QA. Native Gemini (blank "
            "Base URL, AI Studio key) or ANY OpenAI-compatible endpoint — set "
            "the Base URL (e.g. https://openrouter.ai/api/v1, Groq, DeepSeek, "
            "local Ollama/LM Studio) and a vision-capable model."
        ),
        "fields": [
            {"key": "VISION_API_KEY", "kind": "secret", "label": "API key"},
            {
                "key": "VISION_BASE_URL",
                "kind": "text",
                "label": "Base URL (blank = native Gemini)",
                "placeholder": "https://api.openai-compatible.example/v1",
            },
            {
                "key": "VISION_MODEL",
                "kind": "text",
                "label": "Model",
                "placeholder": "gemini-3.5-flash-lite",
            },
        ],
    },
    {
        "id": "openrouter",
        "label": "OpenRouter — secondary LLM tier (:free models)",
        "probe": "openrouter",
        "hint": (
            "Automatic capacity failover tier when the primary LLM is rate-limited. "
            "Model ids look like vendor/model:free."
        ),
        "fields": [
            {"key": "OPENROUTER_API_KEY", "kind": "secret", "label": "API key"},
            {
                "key": "OPENROUTER_BASE_URL",
                "kind": "text",
                "label": "Base URL",
                "placeholder": "https://openrouter.ai/api/v1",
            },
            {
                "key": "OPENROUTER_MODEL",
                "kind": "text",
                "label": "Model",
                "placeholder": "vendor/model:free",
            },
        ],
    },
    {
        "id": "elevenlabs",
        "label": "ElevenLabs — studio-grade narration TTS",
        "probe": "elevenlabs",
        "hint": "Optional: edge-tts is the $0 default narrator.",
        "fields": [
            {"key": "ELEVENLABS_API_KEY", "kind": "secret", "label": "API key"},
            {
                "key": "ELEVENLABS_VOICE_ID",
                "kind": "text",
                "label": "Voice ID",
                "placeholder": "optional",
            },
        ],
    },
    {
        "id": "kaggle",
        "label": "Kaggle — free GPU compute lane",
        "probe": "kaggle",
        "hint": (
            "KGAT_ token from kaggle.com/settings → API → Create New Token. "
            "Set COMPUTE_BACKEND=kaggle to use it."
        ),
        "fields": [
            {"key": "KAGGLE_USERNAME", "kind": "text", "label": "Username"},
            {"key": "KAGGLE_KEY", "kind": "secret", "label": "API token (KGAT_...)"},
            {
                "key": "COMPUTE_BACKEND",
                "kind": "text",
                "label": "Compute backend",
                "placeholder": "local|kaggle|modal",
            },
        ],
    },
    {
        "id": "modal",
        "label": "Modal — serverless GPU (needs a payment method)",
        "probe": "modal",
        "hint": (
            "Dormant by design: their card verification charges a fee. "
            "Leave empty unless that changes."
        ),
        "fields": [
            {"key": "MODAL_TOKEN_ID", "kind": "secret", "label": "Token ID"},
            {"key": "MODAL_TOKEN_SECRET", "kind": "secret", "label": "Token secret"},
        ],
    },
    {
        "id": "youtube",
        "label": "YouTube — private draft publishing",
        "probe": "youtube",
        "hint": (
            "One-time interactive OAuth: run scripts/gen_youtube_refresh_token.py "
            "on any machine with a browser."
        ),
        "fields": [
            {"key": "YOUTUBE_CLIENT_ID", "kind": "secret", "label": "OAuth client ID"},
            {"key": "YOUTUBE_CLIENT_SECRET", "kind": "secret", "label": "OAuth client secret"},
            {"key": "YOUTUBE_REFRESH_TOKEN", "kind": "secret", "label": "Refresh token"},
        ],
    },
    {
        "id": "ingest",
        "label": "Ingest identity (guardrail UA contact)",
        "probe": None,
        "hint": "Rides the descriptive User-Agent on every cloud fetch (PLAN guardrail 3).",
        "fields": [{"key": "CONTACT_EMAIL", "kind": "text", "label": "Contact email"}],
    },
]


class CredentialEntry(BaseModel):
    key: str = Field(min_length=1, max_length=100)
    value: str = Field(max_length=4096)


class CredentialsIn(BaseModel):
    entries: list[CredentialEntry] = Field(default_factory=list, max_length=40)


class CredentialsOut(BaseModel):
    ok: bool
    version: int
    encrypted_at_rest: bool
    providers: list[dict[str, Any]]


class ProbeIn(BaseModel):
    provider: str = Field(min_length=1, max_length=40)
    params: dict[str, str] = Field(default_factory=dict)


class ProbeOut(BaseModel):
    ok: bool
    latency_ms: int
    detail: str
    extra: dict[str, str] = Field(default_factory=dict)


def _settings_providers_payload() -> list[dict[str, Any]]:
    """Catalog with per-field value hints — secrets masked, never returned raw."""
    s = get_settings()
    with session() as sess:
        studio_keys = {row.key for row in sess.query(CredentialRow).all()}
    providers: list[dict[str, Any]] = []
    for spec in _PROVIDER_SPECS:
        fields: list[dict[str, Any]] = []
        for f in spec["fields"]:
            key = f["key"]
            current = str(getattr(s, key.lower(), "") or "")
            source = "studio" if key in studio_keys else ("env" if current else None)
            fields.append(
                {
                    "key": key,
                    "kind": f["kind"],
                    "label": f["label"],
                    "placeholder": f.get("placeholder", ""),
                    "configured": bool(current),
                    "source": source,
                    # secrets travel as masked hints only — never raw
                    "value": mask_secret(current) if f["kind"] == "secret" else current,
                }
            )
        providers.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "hint": spec["hint"],
                "probe": spec["probe"],
                "fields": fields,
            }
        )
    return providers


@app.get("/settings/providers", response_model=CredentialsOut)
def list_settings_providers() -> CredentialsOut:
    return CredentialsOut(
        ok=True,
        version=0,
        encrypted_at_rest=encryption_active(),
        providers=_settings_providers_payload(),
    )


@app.put("/settings/credentials", response_model=CredentialsOut)
def save_settings_credentials(body: CredentialsIn) -> CredentialsOut:
    entries = {e.key: e.value for e in body.entries}
    try:
        with session() as sess:
            version = upsert_credentials(sess, entries)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    apply_credentials_to_settings(_engine)
    payload = _settings_providers_payload()
    return CredentialsOut(
        ok=True, version=version, encrypted_at_rest=encryption_active(), providers=payload
    )


@app.delete("/settings/credentials/{key}", response_model=CredentialsOut)
def delete_settings_credential(key: str) -> CredentialsOut:
    try:
        with session() as sess:
            version = delete_credential(sess, key)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    apply_credentials_to_settings(_engine)
    payload = _settings_providers_payload()
    return CredentialsOut(
        ok=True, version=version, encrypted_at_rest=encryption_active(), providers=payload
    )


@app.post("/settings/test", response_model=ProbeOut)
def test_settings_provider(body: ProbeIn) -> ProbeOut:
    from director_os.core.probes import run_probe

    try:
        result = run_probe(body.provider, body.params, get_settings())
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # a probe must never 500 on bad credentials
        return ProbeOut(ok=False, latency_ms=0, detail=f"{type(e).__name__}: {e}", extra={})
    return ProbeOut(**result.as_dict())
