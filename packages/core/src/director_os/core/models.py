"""Pydantic v2 data contracts — the canonical domain models (PLAN §5.1).

These are the API/agent-facing contracts. SQLAlchemy rows (see `db.py`) persist them;
`to_orm`/`from_orm`-style helpers live next to each model where needed in M2.
"""

from __future__ import annotations

import enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ----------------------------------------------------------------- audio law (ADR-0005)
SUPPORTED_SAMPLE_RATE = 48_000
SUPPORTED_CHANNELS = 2


class AudioContract(BaseModel):
    """The audio law as an executable contract: 48 kHz stereo everywhere, always."""

    sample_rate: int = SUPPORTED_SAMPLE_RATE
    channels: int = SUPPORTED_CHANNELS
    target_lufs: float = -16.0

    @model_validator(mode="after")
    def _enforce_law(self) -> AudioContract:
        if self.sample_rate != SUPPORTED_SAMPLE_RATE:
            raise ValueError(
                f"audio law violated: sample_rate must be {SUPPORTED_SAMPLE_RATE}, "
                f"got {self.sample_rate} (mixed rates silently mute downstream narration)"
            )
        if self.channels != SUPPORTED_CHANNELS:
            raise ValueError(
                f"audio law violated: channels must be {SUPPORTED_CHANNELS}, got {self.channels}"
            )
        return self


# ----------------------------------------------------------------- enums
class AssetRole(enum.StrEnum):
    raw_bwc = "raw_bwc"
    dashcam = "dashcam"
    radio_911 = "911"
    pdf_report = "pdf_report"
    music = "music"


class ProjectStatus(enum.StrEnum):
    created = "created"
    ingested = "ingested"
    perceived = "perceived"
    planned = "planned"
    qa_failed = "qa_failed"
    awaiting_review = "awaiting_review"
    approved = "approved"
    published_draft = "published_draft"


class JobStage(enum.StrEnum):
    ingest = "ingest"
    perception = "perception"
    proofread = "proofread"
    planning = "planning"
    redaction = "redaction"
    audio_prep = "audio_prep"
    narration = "narration"
    subtitles = "subtitles"
    qa = "qa"
    ship_pack = "ship_pack"
    human_gate = "human_gate"
    render = "render"
    publish = "publish"


# ----------------------------------------------------------------- geometry / fx primitives
class BBox(BaseModel):
    """Pixel box on the render canvas: [x, y, w, h]."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    def intersects(self, other: BBox) -> bool:
        return not (
            self.x + self.w <= other.x
            or other.x + other.w <= self.x
            or self.y + self.h <= other.y
            or other.y + other.h <= self.y
        )


class BlurRegion(BaseModel):
    """Privacy/watermark redaction spec (consumed by vision_fx filter builder).

    Optional `t_start_s`/`t_end_s` bound the region to a SOURCE-footage time
    window — tracked ByteTrack outputs produce one region per time bin. None
    on both = active for the whole timeline (static studio/detector boxes).
    """

    scope: Literal["global", "clip", "range"] = "global"
    bbox: BBox
    mode: Literal["static", "tracked"] = "static"
    sigma: int = Field(default=24, ge=1, le=200)
    track_id: str | None = None
    t_start_s: float | None = Field(default=None, ge=0.0)
    t_end_s: float | None = Field(default=None, ge=0.0)


class ZoomKeyframe(BaseModel):
    t_s: float = Field(ge=0.0)
    zoom_factor: float = Field(gt=1.0, le=3.0)


# ----------------------------------------------------------------- §5.1 domain entities
class Project(BaseModel):
    id: str
    title: str
    niche_skill: str = "bodycam_documentary"
    status: ProjectStatus = ProjectStatus.created
    active_otio_version_id: int | None = None
    youtube_draft_id: str | None = None


class MediaAsset(BaseModel):
    id: str
    project_id: str
    source_url: str | None = None
    role: AssetRole = AssetRole.raw_bwc
    r2_raw_key: str | None = None
    r2_proxy_key: str | None = None
    r2_audio_key: str | None = None
    duration_s: float = Field(gt=0.0)
    fps: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sync_offset_ms: int = 0

    @field_validator("fps")
    @classmethod
    def _plausible_fps(cls, v: float) -> float:
        if v not in {23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0}:
            raise ValueError(f"implausible fps: {v}")
        return v


class PerceptionIndex(BaseModel):
    """Unified Multimodal Timeline Index for one asset (filled by Perceiver, M2+)."""

    asset_id: str
    transcript_json: dict[str, Any] = Field(default_factory=dict)
    shots_json: dict[str, Any] = Field(default_factory=dict)
    audio_events_json: dict[str, Any] = Field(default_factory=dict)
    redaction_tracks_json: dict[str, Any] = Field(default_factory=dict)
    scene_embeddings_json: list[float] = Field(default_factory=list)


class TimelineVersion(BaseModel):
    id: int | None = None
    project_id: str
    version_num: int
    otio_json: dict[str, Any]
    created_by: Literal["director_agent", "human_edit", "assembler_agent"] = "director_agent"
    checkpoint_note: str = ""


class SegmentCacheEntry(BaseModel):
    segment_hash: str
    r2_chunk_key: str
    duration_s: float
    codec_profile: str


class AgentToolLog(BaseModel):
    project_id: str | None = None
    agent_name: str
    tool_name: str
    args_json: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    token_cost_usd: float = 0.0
    duration_ms: int = 0


class Job(BaseModel):
    id: str
    project_id: str
    workflow_id: str | None = None
    current_stage: JobStage = JobStage.ingest
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    error_message: str | None = None
