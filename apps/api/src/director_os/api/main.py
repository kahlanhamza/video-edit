"""FastAPI gateway skeleton (PLAN §2 layer 2).

M1 exposes health + project creation against the core persistence layer.
M2 adds SSE job progress, agent tool dispatch endpoints and the Human Review Gate.
"""

from __future__ import annotations

from datetime import UTC, datetime

from director_os.core.db import ProjectRow, init_models, make_engine, make_session_factory
from director_os.core.models import ProjectStatus
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

app = FastAPI(
    title="DirectorOS API",
    version="0.1.0",
    description="Autonomous multimodal video editing platform — gateway",
)

_engine = None
_session_factory = None


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
