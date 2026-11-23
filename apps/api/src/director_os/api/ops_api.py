"""Op log API — CRUD + fold + SSE op stream (op log v1, roadmap step 2).

The op log is the atom of the frame-level editor: every human canvas edit and
agent proposal lands here as an `EditOp` row; undo = revoke (never delete);
`GET .../ops/stream` pushes op events by polling the monotonic `rev` cursor —
the durable table IS the feed (same law as `job_events`).

Blur ops dual-write the legacy `redaction_tracks_json["regions"]` entry so
the shipped render path keeps consuming them while the op log becomes the
source of truth for history/UI. `POST .../ops/import-legacy` converts
existing regions into ops once (idempotent via `params.legacy_region_id`).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from director_os.core.db import MediaAssetRow, ProjectRow
from director_os.core.ops import (
    EFFECT_KINDS,
    OpCreate,
    OpRecord,
    OpsFold,
    OpsStore,
    fold_ops,
    regions_to_op_params,
    validate_op_window,
)
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

router = APIRouter(prefix="/projects/{project_id}")

DEFAULT_OPS_POLL_S = 0.4
DEFAULT_OPS_MAX_S = 300.0

_factory_override = None


def bind_session_factory(factory: Any) -> None:
    """Tests/alt-wiring inject the factory; prod resolves the gateway lazily."""
    global _factory_override
    _factory_override = factory


def _factory() -> Any:
    if _factory_override is not None:
        return _factory_override
    from director_os.api import main as gateway  # deferred: no import cycle

    gateway.session()  # ensure engine + models init
    return gateway._session_factory


def _store(project_id: str) -> OpsStore:
    return OpsStore(_factory(), project_id)


def _require_project(project_id: str) -> None:
    factory = _factory()
    with factory() as s:
        if s.get(ProjectRow, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")


def _primary_asset(project_id: str) -> MediaAssetRow | None:
    factory = _factory()
    with factory() as s:
        return s.execute(
            select(MediaAssetRow)
            .where(MediaAssetRow.project_id == project_id)
            .order_by(MediaAssetRow.id)
            .limit(1)
        ).scalar_one_or_none()


def _legacy_regions_for(asset_id: str | None) -> list[dict[str, Any]]:
    if not asset_id:
        return []
    from director_os.api.main import _redaction_regions  # deferred: shared helper

    factory = _factory()
    with factory() as s:
        return list(_redaction_regions(s, asset_id))


def _dual_write_blur_region(asset_id: str, rec: OpRecord) -> None:
    """Blur ops keep the shipped render path alive: mirror the box into
    `redaction_tracks_json["regions"]` (op id is the region id)."""
    from director_os.api.main import _redaction_regions, _save_redaction_regions

    factory = _factory()
    with factory() as s:
        regions = _redaction_regions(s, asset_id)
        if any(str(r.get("id", "")) == rec.id for r in regions):
            return
        p = rec.params or {}
        regions.append(
            {
                "id": rec.id,
                "x": int(p.get("x", 0)),
                "y": int(p.get("y", 0)),
                "w": int(p.get("w", 0)),
                "h": int(p.get("h", 0)),
                "sigma": int(p.get("sigma", 24)),
                "label": str(p.get("label", "")),
                "created_by": "agent_edit" if rec.origin == "agent" else "human_edit",
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        _save_redaction_regions(s, asset_id, regions)


def _drop_blur_region(asset_id: str, op_id: str) -> None:
    from director_os.api.main import _redaction_regions, _save_redaction_regions

    factory = _factory()
    with factory() as s:
        regions = _redaction_regions(s, asset_id)
        kept = [r for r in regions if str(r.get("id", "")) != op_id]
        if len(kept) != len(regions):
            _save_redaction_regions(s, asset_id, kept)


# ------------------------------------------------------------------ models
class OpOut(BaseModel):
    rev: int
    id: str
    project_id: str
    asset_id: str = ""
    kind: str
    frame_start: int | None = None
    frame_end: int | None = None
    track: str = "V1_master"
    params: dict[str, Any] = Field(default_factory=dict)
    origin: str = "human"
    evidence: dict[str, Any] = Field(default_factory=dict)
    session_id: str = ""
    revokes_op_id: str | None = None
    revoked: bool = False
    note: str = ""
    created_at: str = ""


class OpsStateOut(BaseModel):
    project_id: str
    rev: int
    active_ops: int
    revoked_ops: int
    ops: list[OpOut]
    fold: OpsFold


class OpIn(BaseModel):
    """Create one op. `evidence` is required for agent-origin ops
    (evidence-before-action law): name the frame or transcript window that
    justifies the edit."""

    kind: str = Field(min_length=1, max_length=40)
    asset_id: str = Field(default="", max_length=64)
    frame_start: int | None = Field(default=None, ge=0)
    frame_end: int | None = Field(default=None, ge=0)
    track: str = Field(default="V1_master", max_length=60)
    params: dict[str, Any] = Field(default_factory=dict)
    origin: str = Field(default="human", pattern="^(agent|human)$")
    evidence: dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field(default="", max_length=64)
    note: str = Field(default="", max_length=1000)


class RevokeIn(BaseModel):
    note: str = Field(default="", max_length=1000)


class AdjustIn(BaseModel):
    """Manual geometry correction for one active blur/inpaint op (the owner
    mandate: agent ops stay manually adjustable afterwards). Adjust is NOT an
    in-place mutation — the append-only law holds: a NEW op carries the
    corrected box (`params.adjusts_op_id` chains it), the old op is revoked,
    every row and every hop stays on the audit trail."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    sigma: int | None = Field(default=None, ge=0, le=64)
    frame_start: int | None = Field(default=None, ge=0)
    frame_end: int | None = Field(default=None, ge=0)
    note: str = Field(default="", max_length=1000)


class ImportLegacyOut(BaseModel):
    imported: int
    skipped: int
    total_regions: int


def _out(rec: OpRecord) -> OpOut:
    return OpOut(**rec.model_dump())


def _validate_kind(kind: str) -> None:
    from director_os.core.ops import OP_KINDS

    if kind not in OP_KINDS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown op kind '{kind}' (known: {', '.join(OP_KINDS)})",
        )


# ------------------------------------------------------------------ endpoints
@router.get("/ops", response_model=OpsStateOut)
def list_ops(project_id: str, include_revoked: bool = True) -> OpsStateOut:
    _require_project(project_id)
    store = _store(project_id)
    asset = _primary_asset(project_id)
    legacy = _legacy_regions_for(asset.id if asset else None)
    records = store.list_ops(include_revoked=include_revoked)
    fold = fold_ops(records, legacy)
    return OpsStateOut(
        project_id=project_id,
        rev=store.max_rev(),
        active_ops=fold.active_ops,
        revoked_ops=fold.revoked_ops,
        ops=[_out(r) for r in records],
        fold=fold,
    )


def fold_with_legacy(records: list[OpRecord], legacy: list[dict[str, Any]]) -> OpsFold:
    """Fold ops over the legacy state (kept as a named seam for tests)."""
    return fold_ops(records, legacy)


@router.post("/ops", status_code=201, response_model=OpOut)
def create_op(project_id: str, body: OpIn) -> OpOut:
    """Append one edit op. Blur ops are frame-validated against the tape and
    dual-write the legacy region the render path consumes."""
    _require_project(project_id)
    _validate_kind(body.kind)
    if body.origin == "agent" and not body.evidence:
        raise HTTPException(
            status_code=422,
            detail="agent ops require evidence (frame ids / transcript windows) — "
            "evidence-before-action is a law, not a suggestion",
        )
    asset = _primary_asset(project_id)
    if body.frame_start is not None or body.frame_end is not None:
        try:
            validate_op_window(
                body.frame_start,
                body.frame_end,
                float(asset.duration_s) if asset else 0.0,
                float(asset.fps) if asset else 0.0,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    asset_id = body.asset_id or (asset.id if asset else "")
    store = _store(project_id)
    rec = store.create(
        OpCreate(
            kind=body.kind,
            asset_id=asset_id,
            frame_start=body.frame_start,
            frame_end=body.frame_end,
            track=body.track,
            params=body.params,
            origin=body.origin,  # type: ignore[arg-type]
            evidence=body.evidence,
            session_id=body.session_id,
            note=body.note,
        )
    )
    if body.kind in ("blur", "inpaint") and asset_id:
        _dual_write_blur_region(asset_id, rec)
    return _out(rec)


@router.post("/ops/{op_id}/revoke", response_model=OpOut)
def revoke_op(project_id: str, op_id: str, body: RevokeIn | None = None) -> OpOut:
    """Undo = revoke: the row stays (audit trail), the fold drops it, and a
    tombstone event rides the feed so every client sees the undo live."""
    _require_project(project_id)
    store = _store(project_id)
    try:
        rec = store.revoke(op_id, note=(body.note if body is not None else ""))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    if rec.kind in ("blur", "inpaint") and rec.asset_id:
        _drop_blur_region(rec.asset_id, rec.id)
    return _out(rec)


@router.post("/ops/{op_id}/adjust", response_model=OpOut)
def adjust_op(project_id: str, op_id: str, body: AdjustIn) -> OpOut:
    """Manual adjust of one active blur/inpaint box — the human half of the
    dual-modality law (agent proposes, human refines, both on one log).

    Semantics (append-only, never an in-place row edit):
    1. a NEW op is appended with the corrected geometry + `adjusts_op_id`;
    2. its region dual-writes the render path FIRST;
    3. only then the superseded op is revoked and its region dropped —
       privacy coverage never has a gap, not even across a crash.
    Tracked ops honestly refuse: their geometry is the per-frame path, a
    silent conversion to a static box would look like coverage but leak.
    """
    _require_project(project_id)
    store = _store(project_id)
    rec = store.get(op_id)
    if rec is None or rec.project_id != project_id:
        raise HTTPException(status_code=404, detail=f"op not found: {op_id}")
    if rec.revoked:
        raise HTTPException(status_code=409, detail=f"op already revoked: {op_id}")
    if rec.kind not in EFFECT_KINDS:
        raise HTTPException(
            status_code=422,
            detail=f"kind '{rec.kind}' has no box geometry to adjust "
            f"(adjustable: {', '.join(EFFECT_KINDS)})",
        )
    if str((rec.params or {}).get("mode", "static")) == "tracked":
        raise HTTPException(
            status_code=409,
            detail="tracked op: its geometry is the per-frame path — revoke it "
            "and track again from the corrected box (never a silent static conversion)",
        )
    asset = _primary_asset(project_id)
    fs = body.frame_start if body.frame_start is not None else rec.frame_start
    fe = body.frame_end if body.frame_end is not None else rec.frame_end
    if fs is not None or fe is not None:
        try:
            validate_op_window(
                fs,
                fe,
                float(asset.duration_s) if asset else 0.0,
                float(asset.fps) if asset else 0.0,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    params = dict(rec.params or {})
    params.update({"x": int(body.x), "y": int(body.y), "w": int(body.w), "h": int(body.h)})
    if body.sigma is not None:
        params["sigma"] = int(body.sigma)
    params["adjusts_op_id"] = rec.id
    note = body.note or f"adjusted from op {rec.id[:8]}"
    # PRIVACY ORDER (see docstring): new region lands BEFORE the old drops.
    new = store.create(
        OpCreate(
            kind=rec.kind,
            asset_id=rec.asset_id,
            frame_start=fs,
            frame_end=fe,
            track=rec.track,
            params=params,
            origin="human",
            evidence={**(rec.evidence or {}), "adjusted_from": rec.id},
            session_id=rec.session_id,
            note=note,
        )
    )
    if rec.asset_id:
        _dual_write_blur_region(rec.asset_id, new)
    store.revoke(rec.id, note=note)
    if rec.asset_id:
        _drop_blur_region(rec.asset_id, rec.id)
    return _out(new)


@router.post("/ops/import-legacy", response_model=ImportLegacyOut)
def import_legacy_regions(project_id: str) -> ImportLegacyOut:
    """One-time, idempotent: every legacy static blur region becomes a real
    blur op (`params.legacy_region_id` marks it; the fold then prefers ops)."""
    _require_project(project_id)
    asset = _primary_asset(project_id)
    if asset is None:
        return ImportLegacyOut(imported=0, skipped=0, total_regions=0)
    regions = _legacy_regions_for(asset.id)
    store = _store(project_id)
    owned = {
        str((r.params or {}).get("legacy_region_id", "") or "")
        for r in store.list_ops(include_revoked=True)
    }
    imported = 0
    for region in regions:
        rid = str(region.get("id", ""))
        if not rid or rid in owned:
            continue
        store.create(
            OpCreate(
                kind="blur",
                asset_id=asset.id,
                track="V1_master",
                params=regions_to_op_params(region),
                origin="human",
                evidence={"imported_from": "redaction_tracks_json"},
                note="imported legacy region",
            )
        )
        imported += 1
    return ImportLegacyOut(
        imported=imported, skipped=len(regions) - imported, total_regions=len(regions)
    )


@router.get("/ops/stream")
async def ops_stream(
    project_id: str,
    since: int = 0,
    poll_s: float = DEFAULT_OPS_POLL_S,
    max_seconds: float = DEFAULT_OPS_MAX_S,
) -> StreamingResponse:
    """SSE op events: replay every row past the `since` cursor, then follow
    the feed live; terminal `done` at the cap. The durable table IS the feed
    (same law as the job-event stream) so reconnects resume gap-free."""
    _require_project(project_id)

    async def gen() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        store = _store(project_id)
        cursor = since
        yield _sse("hello", {"rev": store.max_rev(), "since": cursor, "project_id": project_id})
        deadline = time.monotonic() + max_seconds
        while time.monotonic() < deadline:
            events = await loop.run_in_executor(None, store.events_after, cursor)
            for ev in events:
                cursor = ev.rev
                yield _sse("op", {"rev": ev.rev, "op": ev.model_dump()})
            if not events:
                await asyncio.sleep(poll_s)
        yield _sse("done", {"rev": cursor, "reason": "timeout"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
