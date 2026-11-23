"""The op log — everything-is-an-op (agent-frame-editor roadmap step 2).

An `EditOp` is the atom of editing (docs/research/agent-frame-editor.md §3.1,
law 1): `kind`, `frame_start`, `frame_end`, `track`, `params`, `origin`
(agent | human), `evidence` (frames + transcript windows that justify it).
The program at frame N is `fold(ops active at N)`; undo = revoke the op (the
row stays — the audit trail the bodycam niche legally requires); nothing is
deleted. Expensive work is scoped to the op's frame window (law 3), and the
export gate is the only full render (law 4).

The reducer is pure and hermetic: it folds rows/records (already revoked
filtered by the caller OR filtered here) into the views the Studio renders
(effect boxes, caption overrides) plus a generic passthrough for kinds whose
bake lives in the OTIO/segment layer (trim/split/zoom/...). Legacy state
(pre-op static blur regions, caption PATCH edits) is merged, and the importer
converts it into real ops once (idempotent via `params.legacy_region_id`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

OpKind = Literal[
    "blur",
    "inpaint",
    "caption",
    "zoom",
    "freeze",
    "card",
    "narration",
    "audio",
    "trim",
    "split",
    "note",
]
OpOrigin = Literal["agent", "human"]

OP_KINDS: tuple[str, ...] = (
    "blur",
    "inpaint",
    "caption",
    "zoom",
    "freeze",
    "card",
    "narration",
    "audio",
    "trim",
    "split",
    "note",
)

# kinds the fold renders as overlay effect boxes on the program monitor
EFFECT_KINDS: tuple[str, ...] = ("blur", "inpaint")


class OpCreate(BaseModel):
    """Payload for one new op (human canvas edit or agent proposal)."""

    kind: str = Field(min_length=1, max_length=40)
    asset_id: str = Field(default="", max_length=64)
    frame_start: int | None = Field(default=None, ge=0)
    frame_end: int | None = Field(default=None, ge=0)
    track: str = Field(default="V1_master", max_length=60)
    params: dict[str, Any] = Field(default_factory=dict)
    origin: OpOrigin = "human"
    evidence: dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field(default="", max_length=64)
    note: str = Field(default="", max_length=1000)


class OpRecord(BaseModel):
    """One row of the op log, as returned by the API / store."""

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


class OpsFold(BaseModel):
    """The program: what the active (non-revoked) ops mean for the Studio.

    - `effects`: overlay boxes (blur/inpaint) the program monitor + timeline
      render — legacy static regions merged, op-backed entries win.
    - `caption_overrides`: cue-indexed edits recorded as ops (text/speaker/
      timing), applied over the cue store read-only; the cue store on disk
      stays the baked truth.
    - `other`: passthrough digest of kinds whose bake lives in the OTIO layer.
    """

    rev: int = 0
    active_ops: int = 0
    revoked_ops: int = 0
    effects: list[dict[str, Any]] = Field(default_factory=list)
    caption_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    other: list[dict[str, Any]] = Field(default_factory=list)


def _row_to_record(row: Any) -> OpRecord:
    return OpRecord(
        rev=int(row.rev),
        id=str(row.id),
        project_id=str(row.project_id),
        asset_id=str(getattr(row, "asset_id", "") or ""),
        kind=str(row.kind),
        frame_start=row.frame_start,
        frame_end=row.frame_end,
        track=str(row.track or "V1_master"),
        params=dict(row.params or {}),
        origin=str(row.origin or "human"),
        evidence=dict(row.evidence or {}),
        session_id=str(getattr(row, "session_id", "") or ""),
        revokes_op_id=getattr(row, "revokes_op_id", None),
        revoked=row.revoked_at is not None,
        note=str(getattr(row, "note", "") or ""),
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


def records_from_rows(rows: Iterable[Any]) -> list[OpRecord]:
    return [_row_to_record(r) for r in rows]


def validate_op_window(
    frame_start: int | None,
    frame_end: int | None,
    duration_s: float,
    fps: float,
) -> None:
    """Frame window sanity: ordered, inside the tape, fps-consistent.

    `None` bounds are legal (whole-tape / not-frame-scoped ops such as a
    global static blur). A window that names no frames is a note, not a
    scoped edit.
    """
    if (frame_start is None) != (frame_end is None):
        raise ValueError("frame_start and frame_end must be set together (or both omitted)")
    if frame_start is None or frame_end is None:
        return
    if frame_end < frame_start:
        raise ValueError(f"frame_end {frame_end} < frame_start {frame_start}")
    if fps <= 0:
        return  # fps unknown (e.g. note-only op) — no tape to violate
    last_frame = max(0, int(duration_s * fps))
    if frame_start > last_frame:
        raise ValueError(
            f"frame_start {frame_start} beyond tape end (frame {last_frame} @ {fps}fps)"
        )
    if frame_end > last_frame:
        raise ValueError(f"frame_end {frame_end} beyond tape end (frame {last_frame} @ {fps}fps)")


def _effect_from_op(rec: OpRecord) -> dict[str, Any]:
    p = rec.params or {}
    return {
        "id": rec.id,
        "op_id": rec.id,
        "op_rev": rec.rev,
        "kind": rec.kind,
        "x": int(p.get("x", 0)),
        "y": int(p.get("y", 0)),
        "w": int(p.get("w", 0)),
        "h": int(p.get("h", 0)),
        "sigma": int(p.get("sigma", 24)),
        "label": str(p.get("label", "")),
        "mode": str(p.get("mode", "static")),
        "t_start_s": p.get("t_start_s"),
        "t_end_s": p.get("t_end_s"),
        "frame_start": rec.frame_start,
        "frame_end": rec.frame_end,
        "created_by": f"op:{rec.origin}",
        "origin": rec.origin,
        "evidence": rec.evidence,
    }


def _legacy_region_effect(region: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(region.get("id", "")),
        "op_id": None,
        "op_rev": None,
        "kind": "blur",
        "x": int(region.get("x", 0)),
        "y": int(region.get("y", 0)),
        "w": int(region.get("w", 0)),
        "h": int(region.get("h", 0)),
        "sigma": int(region.get("sigma", 24)),
        "label": str(region.get("label", "")),
        "mode": str(region.get("mode", "static")),
        "t_start_s": region.get("t_start_s"),
        "t_end_s": region.get("t_end_s"),
        "frame_start": None,
        "frame_end": None,
        "created_by": str(region.get("created_by", "legacy_region")),
        "origin": "human" if region.get("created_by") == "human_edit" else "legacy",
        "evidence": {},
    }


def fold_ops(
    ops: Sequence[OpRecord],
    legacy_regions: Sequence[Mapping[str, Any]] = (),
) -> OpsFold:
    """Fold the op log into the program the Studio renders.

    Order = `rev` ascending (append-only log; later ops observe earlier
    ones). Revoked ops are skipped (undo = revoke — the row remains for the
    audit trail). `kind="revoke"` tombstone rows are pure feed events — the
    durable announcement of an undo — never counted, never rendered. A
    blur/inpaint op whose params carry `legacy_region_id` supersedes the
    legacy region of that id, so importing the legacy state never doubles an
    effect.
    """
    fold = OpsFold()
    superseded_legacy_ids: set[str] = set()
    active: list[OpRecord] = []
    for rec in sorted(ops, key=lambda r: r.rev):
        if rec.kind == "revoke":
            continue  # tombstone event, not an op
        if rec.revoked:
            fold.revoked_ops += 1
            continue
        fold.active_ops += 1
        fold.rev = max(fold.rev, rec.rev)
        # A blur op owns the legacy region it imported OR the region the
        # dual-write mirrored under the op's own id — never double-render.
        legacy_key = str((rec.params or {}).get("legacy_region_id", "") or "")
        if rec.kind in EFFECT_KINDS and (legacy_key or rec.id):
            superseded_legacy_ids.add(legacy_key or rec.id)
        active.append(rec)

    for rec in active:
        if rec.kind in EFFECT_KINDS:
            fold.effects.append(_effect_from_op(rec))
        elif rec.kind == "caption":
            p = rec.params or {}
            idx = p.get("cue_index")
            if idx is None:
                continue
            entry = fold.caption_overrides.setdefault(str(idx), {})
            for key in ("text", "speaker", "start_ms", "end_ms"):
                if key in p:
                    entry[key] = p[key]
        else:
            fold.other.append(
                {
                    "op_id": rec.id,
                    "rev": rec.rev,
                    "kind": rec.kind,
                    "track": rec.track,
                    "frame_start": rec.frame_start,
                    "frame_end": rec.frame_end,
                    "origin": rec.origin,
                    "summary": str((rec.params or {}).get("summary", "") or rec.note),
                }
            )

    for region in legacy_regions:
        rid = str(region.get("id", ""))
        if rid and rid in superseded_legacy_ids:
            continue  # an op already owns this region — never double it
        fold.effects.append(_legacy_region_effect(region))

    fold.effects.sort(key=lambda e: (e.get("op_rev") is None, e.get("op_rev") or 0))
    return fold


def apply_caption_overrides(
    cues: Sequence[dict[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply caption ops over the cue store (read-only base, copy-on-write).

    Cue text timing fields that an op did not touch pass through unchanged.
    Unknown indices are ignored (the fold never invents cues).
    """
    out: list[dict[str, Any]] = [dict(c) for c in cues]
    for key, ov in overrides.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(out):
            continue
        for field, value in ov.items():
            if field in ("text", "speaker"):
                out[idx][field] = str(value)
            elif field in ("start_ms", "end_ms"):
                out[idx][field] = int(value)
    return out


def regions_to_op_params(region: Mapping[str, Any]) -> dict[str, Any]:
    """One legacy static region -> params for its blur op (import law)."""
    return {
        "x": int(region.get("x", 0)),
        "y": int(region.get("y", 0)),
        "w": int(region.get("w", 0)),
        "h": int(region.get("h", 0)),
        "sigma": int(region.get("sigma", 24)),
        "label": str(region.get("label", "")),
        "mode": str(region.get("mode", "static")),
        "legacy_region_id": str(region.get("id", "")),
    }


class OpsStore:
    """Session-factory-backed store for the op log (API + agent tools share it)."""

    def __init__(self, session_factory: Any, project_id: str) -> None:
        self._factory = session_factory
        self._project_id = project_id

    def create(
        self,
        op: OpCreate,
        *,
        agent_name: str = "",
    ) -> OpRecord:
        from director_os.core.db import TimelineOpRow

        with self._factory() as s:
            row = TimelineOpRow(
                project_id=self._project_id,
                asset_id=op.asset_id or None,
                kind=op.kind,
                frame_start=op.frame_start,
                frame_end=op.frame_end,
                track=op.track or "V1_master",
                params=dict(op.params),
                origin=op.origin,
                evidence=dict(op.evidence),
                session_id=op.session_id or agent_name or None,
                note=op.note,
            )
            s.add(row)
            s.commit()
            return _row_to_record(row)

    def list_ops(
        self,
        *,
        include_revoked: bool = False,
        limit: int = 1000,
    ) -> list[OpRecord]:
        """Op rows (feed-event tombstones excluded — they are not ops)."""
        from director_os.core.db import TimelineOpRow
        from sqlalchemy import select

        with self._factory() as s:
            q = select(TimelineOpRow).where(TimelineOpRow.project_id == self._project_id)
            q = q.where(TimelineOpRow.kind != "revoke")
            if not include_revoked:
                q = q.where(TimelineOpRow.revoked_at.is_(None))
            rows = s.execute(q.order_by(TimelineOpRow.rev).limit(limit)).scalars()
            return records_from_rows(rows)

    def get(self, op_id: str) -> OpRecord | None:
        from director_os.core.db import TimelineOpRow
        from sqlalchemy import select

        with self._factory() as s:
            row = s.execute(
                select(TimelineOpRow).where(
                    TimelineOpRow.project_id == self._project_id,
                    TimelineOpRow.id == op_id,
                )
            ).scalar_one_or_none()
            return _row_to_record(row) if row is not None else None

    def revoke(self, op_id: str, note: str = "") -> OpRecord:
        """Undo = revoke (never delete).

        Two durable writes: the target row gets `revoked_at`, and a tombstone
        event row (kind="revoke", `params.revokes_op_id`) rides the feed so
        every connected client sees the undo — an in-place edit would never
        advance the `rev` cursor. Raises LookupError when unknown, ValueError
        when already revoked.
        """
        from datetime import UTC, datetime

        from director_os.core.db import TimelineOpRow
        from sqlalchemy import select

        with self._factory() as s:
            row = s.execute(
                select(TimelineOpRow).where(
                    TimelineOpRow.project_id == self._project_id,
                    TimelineOpRow.id == op_id,
                )
            ).scalar_one_or_none()
            if row is None:
                raise LookupError(f"op not found: {op_id}")
            if row.revoked_at is not None:
                raise ValueError(f"op already revoked: {op_id}")
            row.revoked_at = datetime.now(UTC)
            tomb = TimelineOpRow(
                project_id=self._project_id,
                asset_id=row.asset_id,
                kind="revoke",
                track=row.track,
                params={"revokes_op_id": op_id},
                origin=row.origin,
                evidence={},
                session_id=row.session_id,
                note=note,
            )
            s.add(tomb)
            s.commit()
            return _row_to_record(row)

    def max_rev(self) -> int:
        from director_os.core.db import TimelineOpRow
        from sqlalchemy import func, select

        with self._factory() as s:
            return int(
                s.execute(
                    select(func.max(TimelineOpRow.rev)).where(
                        TimelineOpRow.project_id == self._project_id
                    )
                ).scalar_one()
                or 0
            )

    def fold(self, legacy_regions: Sequence[Mapping[str, Any]] = ()) -> OpsFold:
        return fold_ops(self.list_ops(include_revoked=True), legacy_regions)

    def events_after(self, rev: int, limit: int = 200) -> list[OpRecord]:
        """Op rows past a cursor (SSE push law: durable table IS the feed)."""
        from director_os.core.db import TimelineOpRow
        from sqlalchemy import select

        with self._factory() as s:
            rows = s.execute(
                select(TimelineOpRow)
                .where(TimelineOpRow.project_id == self._project_id)
                .where(TimelineOpRow.rev > rev)
                .order_by(TimelineOpRow.rev)
                .limit(limit)
            ).scalars()
            return records_from_rows(rows)
