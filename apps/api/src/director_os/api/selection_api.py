"""Selection API — promptable object tracking on any frame (roadmap step 3).

`POST /assets/{id}/selection`: a drawn box (or click) on frame X -> per-frame
tracked boxes across a bounded span (CPU template tracker by default). The
result can become a tracked blur op immediately (create_blur_op=true): the op
carries `params.tracked` (the per-frame path step 4 bakes) and DUAL-WRITES a
static union region so the shipped render path covers the span TODAY — a
tracked op never silently loses privacy coverage on export.

`GET /ops/{op_id}`: one op record in full (the Studio fetches the tracked
path of a selected effect without pulling the whole log).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from director_os.core.db import MediaAssetRow, TimelineOpRow
from director_os.core.ops import OpCreate, OpRecord, OpsStore, validate_op_window
from director_os.vision_fx.selection import (
    MAX_SPAN_S,
    SelectionError,
    SelectionNotAvailable,
    SelectionPrompt,
    TrackedSelection,
    available_selection_backends,
    run_selection,
)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

router = APIRouter()

_factory: Any = None


def bind_session_factory(factory: Any) -> None:
    global _factory
    _factory = factory


def _session() -> Any:
    if _factory is None:
        from director_os.api.main import session

        return session()
    return _factory()


class SelectionIn(BaseModel):
    """Body for a promptable selection (coordinates in RAW SOURCE pixels)."""

    kind: Literal["box", "point"] = "box"
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(default=0, ge=0)
    h: int = Field(default=0, ge=0)
    anchor_frame: int = Field(ge=0)
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    backend: str = Field(default="", max_length=40)
    create_blur_op: bool = True
    sigma: int = Field(default=24, ge=0, le=60)
    label: str = Field(default="", max_length=80)


class SelectionOut(BaseModel):
    backend: str
    anchor_frame: int
    frame_start: int
    frame_end: int
    n_hits: int
    n_interpolated: int
    mean_score: float
    hits: list[dict[str, Any]]
    union_box: dict[str, int]
    op_id: str | None = None
    detail: str


class OpOut(BaseModel):
    id: str
    rev: int
    kind: str
    asset_id: str
    frame_start: int | None
    frame_end: int | None
    track: str
    params: dict[str, Any]
    origin: str
    evidence: dict[str, Any]
    note: str
    revoked: bool


def _resolve_media(row: MediaAssetRow) -> str:
    """Decode target: the 480p proxy when it exists (cheap), else the source.

    The prompt/hit canvas is ALWAYS the raw source frame (asset row width x
    height); proxy hits are scaled back at run_selection's boundary.
    """
    from director_os.api.main import _proxy_file

    if row.r2_proxy_key:
        try:
            return str(_proxy_file(row.r2_proxy_key, "proxy"))
        except HTTPException:
            pass  # proxy not built yet — the source is still decodable
    if row.source_url and Path(row.source_url).exists():
        return str(row.source_url)
    raise HTTPException(status_code=404, detail="no decodable media for this asset")


def _dual_write_tracked_region(asset_id: str, rec: OpRecord, sel: TrackedSelection) -> None:
    """Mirror the tracked selection into the shipped render path's regions.

    The union box gives the CURRENT renderer full-span coverage (identical to
    the pipeline's tracks_to_regions law); the tracked path rides along in the
    region dict (`mode: tracked`) so the step-4 bake reads it without a
    schema change. A renderer that cannot read `tracked` yet still blurs the
    union — privacy coverage never regresses.
    """
    from director_os.api.main import _redaction_regions, _save_redaction_regions

    box = sel.union_box()
    with _session() as s:
        regions = _redaction_regions(s, asset_id)
        if any(str(r.get("id", "")) == rec.id for r in regions):
            return
        regions.append(
            {
                "id": rec.id,
                **box,
                "sigma": 24,
                "label": "tracked selection",
                "mode": "tracked",
                "backend": sel.backend,
                "tracked": [h.model_dump() for h in sel.hits],
                "created_by": "agent_edit" if rec.origin == "agent" else "human_edit",
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        _save_redaction_regions(s, asset_id, regions)


@router.post("/assets/{asset_id}/selection", response_model=SelectionOut, status_code=201)
def create_selection(asset_id: str, body: SelectionIn) -> SelectionOut:
    """Track a prompt across [frame_start, frame_end] on frame-addressed media."""
    s = _session()
    try:
        row = s.get(MediaAssetRow, asset_id)
        if row is None:
            raise HTTPException(status_code=404, detail="asset not found")
        # copy everything the request needs BEFORE the session closes
        duration_s, fps = float(row.duration_s), float(row.fps)
        src_w, src_h = int(row.width), int(row.height)
        project_id = str(row.project_id)
        media = _resolve_media(row)
    finally:
        s.close()

    span_s = (body.frame_end + 1 - body.frame_start) / fps if fps > 0 else 0.0
    if fps <= 0:
        raise HTTPException(status_code=422, detail="asset has no fps — run the pipeline")
    if span_s <= 0:
        raise HTTPException(status_code=422, detail="frame_end must be >= frame_start")
    if span_s > MAX_SPAN_S:
        raise HTTPException(
            status_code=422,
            detail=f"span {span_s:.1f}s exceeds the interactive cap ({MAX_SPAN_S:.0f}s) — "
            "split the selection into spans",
        )
    last_frame = max(0, int(duration_s * fps))
    frame_end = min(body.frame_end, last_frame)
    if body.anchor_frame > last_frame:
        raise HTTPException(status_code=422, detail="anchor frame beyond tape end")
    if body.kind == "box" and (body.w <= 0 or body.h <= 0):
        raise HTTPException(status_code=422, detail="box prompt needs w and h")
    if body.x >= src_w or body.y >= src_h:
        raise HTTPException(status_code=422, detail="prompt outside the frame")

    try:
        sel = run_selection(
            media,
            SelectionPrompt(
                kind=body.kind,
                x=body.x,
                y=body.y,
                w=body.w,
                h=body.h,
                anchor_frame=body.anchor_frame,
            ),
            frame_start=body.frame_start,
            frame_end=frame_end,
            fps=fps,
            frame_w=src_w,
            frame_h=src_h,
            backend=body.backend,
        )
    except SelectionNotAvailable as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except SelectionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    op_id: str | None = None
    if body.create_blur_op:
        params = sel.to_params()
        params["sigma"] = body.sigma
        params["label"] = body.label or "tracked selection"
        validate_op_window(body.frame_start, frame_end, duration_s, fps)
        store = OpsStore(_session, project_id)
        rec = store.create(
            OpCreate(
                kind="blur",
                asset_id=asset_id,
                frame_start=body.frame_start,
                frame_end=frame_end,
                params=params,
                origin="human",
                evidence={
                    "anchor_frame": body.anchor_frame,
                    "prompt": sel.prompt,
                    "backend": sel.backend,
                    "mean_score": sel.mean_score,
                },
                note=f"tracked selection ({sel.backend}, {len(sel.hits)} frames)",
            )
        )
        op_id = rec.id
        _dual_write_tracked_region(asset_id, rec, sel)

    return SelectionOut(
        backend=sel.backend,
        anchor_frame=sel.anchor_frame,
        frame_start=body.frame_start,
        frame_end=frame_end,
        n_hits=len(sel.hits),
        n_interpolated=sel.n_interpolated,
        mean_score=sel.mean_score,
        hits=[h.model_dump() for h in sel.hits],
        union_box=sel.union_box(),
        op_id=op_id,
        detail=f"{sel.backend} tracked {len(sel.hits)} frames "
        f"({sel.n_interpolated} interpolated, mean score {sel.mean_score:.2f})"
        + (f" -> blur op {op_id[:8]}" if op_id else ""),
    )


@router.get("/ops/{op_id}", response_model=OpOut)
def get_op(op_id: str) -> OpOut:
    """One op in full — the Studio reads a single effect's tracked path."""
    s = _session()
    try:
        row = s.execute(select(TimelineOpRow).where(TimelineOpRow.id == op_id)).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="op not found")
        return OpOut(
            id=str(row.id),
            rev=int(row.rev),
            kind=str(row.kind),
            asset_id=str(getattr(row, "asset_id", "") or ""),
            frame_start=row.frame_start,
            frame_end=row.frame_end,
            track=str(row.track or "V1_master"),
            params=dict(row.params or {}),
            origin=str(row.origin or "human"),
            evidence=dict(row.evidence or {}),
            note=str(row.note or ""),
            revoked=row.revoked_at is not None,
        )
    finally:
        s.close()


@router.get("/selection/backends")
def selection_backends() -> dict[str, bool]:
    """Which selection backends can run right now (sam2 needs its extra)."""
    return available_selection_backends()
