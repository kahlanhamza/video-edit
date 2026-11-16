"""SQLAlchemy 2.0 persistence layer (PLAN §5.1).

SQLite-runnable in CI; Postgres 16 + pgvector in docker-compose / production
(ADR-0008: `scene_embeddings_json` becomes a pgvector column in M2).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid.uuid4())


class ProjectRow(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(300))
    niche_skill: Mapped[str] = mapped_column(String(100), default="bodycam_documentary")
    status: Mapped[str] = mapped_column(String(40), default="created")
    active_otio_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    youtube_draft_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class MediaAssetRow(Base):
    __tablename__ = "media_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    role: Mapped[str] = mapped_column(String(40), default="raw_bwc")
    r2_raw_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    r2_proxy_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    r2_audio_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    duration_s: Mapped[float] = mapped_column(Float)
    fps: Mapped[float] = mapped_column(Float)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    sync_offset_ms: Mapped[int] = mapped_column(Integer, default=0)


class PerceptionIndexRow(Base):
    __tablename__ = "perception_index"

    asset_id: Mapped[str] = mapped_column(ForeignKey("media_assets.id"), primary_key=True)
    transcript_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    shots_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    audio_events_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    redaction_tracks_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # ADR-0008: JSON stand-in until the pgvector migration in M2.
    scene_embeddings_json: Mapped[list[Any]] = mapped_column(JSON, default=list)


class TimelineVersionRow(Base):
    __tablename__ = "timeline_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    version_num: Mapped[int] = mapped_column(Integer)
    otio_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(String(40), default="director_agent")
    checkpoint_note: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SegmentCacheRow(Base):
    __tablename__ = "segment_cache"

    segment_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    r2_chunk_key: Mapped[str] = mapped_column(String(1024))
    duration_s: Mapped[float] = mapped_column(Float)
    codec_profile: Mapped[str] = mapped_column(String(120))
    last_used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AgentToolLogRow(Base):
    __tablename__ = "agent_tool_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("projects.id"), nullable=True, index=True
    )
    agent_name: Mapped[str] = mapped_column(String(60))
    tool_name: Mapped[str] = mapped_column(String(120))
    args_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_summary: Mapped[str] = mapped_column(String(1000), default="")
    token_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    temporal_workflow_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    current_stage: Mapped[str] = mapped_column(String(40), default="ingest")
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


ALL_TABLES = (
    ProjectRow,
    MediaAssetRow,
    PerceptionIndexRow,
    TimelineVersionRow,
    SegmentCacheRow,
    AgentToolLogRow,
    JobRow,
)


def make_engine(database_url: str | None = None):
    from director_os.core.config import get_settings

    url = database_url or get_settings().database_url
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, future=True, **kwargs)


def make_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_models(engine) -> None:
    """Dev/CI convenience: create all tables. Production uses Alembic migrations."""
    Base.metadata.create_all(engine)
