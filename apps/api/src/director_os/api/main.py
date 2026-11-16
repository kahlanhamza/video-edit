"""FastAPI gateway skeleton (PLAN §2 layer 2).

M1 exposes health + project creation against the core persistence layer.
M2 adds SSE job progress, agent tool dispatch endpoints and the Human Review Gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from director_os.api.youtube import (
    Chapter,
    DraftMetadata,
    InMemoryYouTubePublisher,
    YouTubePublisher,
)
from director_os.core.db import ProjectRow, init_models, make_engine, make_session_factory
from director_os.core.models import ProjectStatus
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

app = FastAPI(
    title="DirectorOS API",
    version="0.2.0",
    description="Autonomous multimodal video editing platform — gateway",
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


@app.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: str) -> ProjectOut:
    s = session()
    row = s.execute(select(ProjectRow).where(ProjectRow.id == project_id)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="project not found")
    return ProjectOut(id=row.id, title=row.title, niche_skill=row.niche_skill, status=row.status)


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
