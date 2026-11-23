"""Agent op tools — the agent's surface onto the op log (op log v1).

Three typed tools registered when an `OpsStore` is available (API dispatch
paths bind one; hermetic tests bind an in-memory factory):

- `ops.list`     — read the op log + the folded program (the agent perceives
                   the timeline state before touching it — timeline-aware law).
- `ops.propose`  — append one edit op; `origin=agent`; evidence REQUIRED
                   (evidence-before-action: name the frame or transcript
                   window that justifies the edit).
- `ops.revoke`   — undo by op id (revoke, never delete).

Subagents are roles, not services (ADR-0011): these tools ride the same
registry as every other director tool.
"""

from __future__ import annotations

from typing import Any

from director_os.agent_runtime.editing_tools import ToolOperationError
from director_os.agent_runtime.registry import ToolRegistry
from director_os.core.ops import OP_KINDS, OpCreate, OpsStore
from pydantic import BaseModel, Field


class _OpsListInput(BaseModel):
    include_revoked: bool = False
    limit: int = Field(default=200, ge=1, le=1000)


class _OpsProposeInput(BaseModel):
    kind: str
    frame_start: int | None = Field(default=None, ge=0)
    frame_end: int | None = Field(default=None, ge=0)
    track: str = Field(default="V1_master", max_length=60)
    params: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    note: str = Field(default="", max_length=1000)


class _OpsRevokeInput(BaseModel):
    op_id: str = Field(min_length=8, max_length=64)
    note: str = Field(default="", max_length=1000)


def register_ops_tools(
    registry: ToolRegistry,
    ops_store: OpsStore,
    project_id: str,
    agent_name: str = "director_agent",
) -> None:
    """Register the op-log tools for one session (never share a registry)."""

    def ops_list(**kw: Any) -> dict[str, Any]:
        records = ops_store.list_ops(
            include_revoked=bool(kw.get("include_revoked", False)),
            limit=int(kw.get("limit", 200)),
        )
        from director_os.core.ops import fold_ops

        fold = fold_ops(records)
        return {
            "ok": True,
            "active_ops": fold.active_ops,
            "revoked_ops": fold.revoked_ops,
            "rev": fold.rev,
            "effects": fold.effects,
            "caption_overrides": fold.caption_overrides,
            "other": fold.other,
            "detail": f"{fold.active_ops} active ops (rev {fold.rev})",
        }

    def ops_propose(**kw: Any) -> dict[str, Any]:
        kind = str(kw.get("kind", ""))
        if kind not in OP_KINDS:
            raise ToolOperationError(f"unknown op kind '{kind}' (known: {', '.join(OP_KINDS)})")
        evidence = dict(kw.get("evidence") or {})
        if not evidence:
            raise ToolOperationError(
                "evidence is required for agent ops — cite the frame you grabbed "
                "(frame.grab) or the transcript window that justifies this edit "
                "(evidence-before-action law)"
            )
        rec = ops_store.create(
            OpCreate(
                kind=kind,
                frame_start=kw.get("frame_start"),
                frame_end=kw.get("frame_end"),
                track=str(kw.get("track") or "V1_master"),
                params=dict(kw.get("params") or {}),
                origin="agent",
                evidence=evidence,
                session_id=agent_name,
                note=str(kw.get("note") or ""),
            ),
            agent_name=agent_name,
        )
        return {
            "ok": True,
            "op_id": rec.id,
            "rev": rec.rev,
            "kind": rec.kind,
            "detail": f"op {rec.id[:8]} ({rec.kind}) appended at rev {rec.rev}",
        }

    def ops_revoke(**kw: Any) -> dict[str, Any]:
        op_id = str(kw.get("op_id", ""))
        try:
            rec = ops_store.revoke(op_id, note=str(kw.get("note") or ""))
        except LookupError as e:
            raise ToolOperationError(str(e)) from e
        except ValueError as e:
            raise ToolOperationError(str(e)) from e
        return {
            "ok": True,
            "op_id": rec.id,
            "kind": rec.kind,
            "detail": f"op {rec.id[:8]} ({rec.kind}) revoked — undo is a revoke, row kept",
        }

    registry.register(
        "ops.list",
        "Read the project op log and the folded program: effect boxes, caption "
        "overrides, op counts, current rev. Perceive the timeline state BEFORE "
        "proposing edits (timeline-aware law).",
        _OpsListInput,
        ops_list,
    )
    registry.register(
        "ops.propose",
        "Append ONE edit op to the op log (blur/inpaint/caption/zoom/freeze/card/"
        "narration/audio/trim/split/note). Frame-addressed: name frame_start/"
        "frame_end when the op scopes frames; params carry kind specifics "
        "(bbox x/y/w/h + sigma for blur, cue_index/text for caption). Evidence "
        "REQUIRED: cite the frame or transcript window that justifies it.",
        _OpsProposeInput,
        ops_propose,
    )
    registry.register(
        "ops.revoke",
        "Undo one op by id — revoke, never delete; the audit trail stays.",
        _OpsRevokeInput,
        ops_revoke,
    )
