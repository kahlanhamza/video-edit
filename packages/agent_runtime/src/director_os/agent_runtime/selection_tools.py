"""Agent selection tool — promptable object tracking (roadmap step 3).

`selection.track` (the agent's eye-hand link): point at an object ON a frame
of the working master (a drawn box or a click) and get its per-frame path
across a bounded span. The tool enforces the house laws end-to-end:

- evidence-before-action: an `evidence` dict (the frame.grab call that showed
  the object) is REQUIRED — a tracked blur on an unseen object is a blind op;
- scoped expensive work: the span is bounded (MAX_SPAN_S) and decoded once;
- everything-is-an-op: with `create_op` (default true) the selection lands as
  a `blur` op with `params.tracked` — undoable by ops.revoke, visible on the
  SSE feed, and never baked outside the Export gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from director_os.agent_runtime.editing_tools import ToolOperationError
from director_os.agent_runtime.registry import ToolRegistry
from director_os.core.ops import OpCreate, OpsStore, validate_op_window
from director_os.vision_fx.selection import (
    MAX_SPAN_S,
    SelectionError,
    SelectionNotAvailable,
    SelectionPrompt,
    probe_video,
    run_selection,
)
from pydantic import BaseModel, Field, model_validator


class _SelectionTrack(BaseModel):
    """Track one object from an anchor frame across a span of the master."""

    anchor_t_s: float = Field(ge=0.0, le=24 * 3600.0)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(default=0, ge=0)
    h: int = Field(default=0, ge=0)
    span_s: float = Field(default=6.0, gt=0.0, le=MAX_SPAN_S)
    backend: str = Field(default="", max_length=40)
    create_op: bool = True
    sigma: int = Field(default=24, ge=0, le=60)
    label: str = Field(default="", max_length=80)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _box_shape(self) -> _SelectionTrack:
        if self.w == 0 and self.h == 0:
            return self  # point prompt at (x, y)
        if self.w <= 0 or self.h <= 0:
            raise ValueError("box prompt needs both w and h (or neither for a point)")
        return self


def _resolve_master(work_dir: Path) -> Path:
    master = work_dir / "master_postfx_v4.mp4"
    if master.exists():
        return master
    candidates = sorted(work_dir.glob("master*.mp4"))
    if not candidates:
        raise ToolOperationError(f"no master video in work dir to track on ({work_dir})")
    return candidates[0]


def register_selection_tools(
    registry: ToolRegistry,
    ops_store: OpsStore,
    work_dir: Path,
    agent_name: str = "director_agent",
) -> None:
    """Register the selection surface for one session (never share a registry)."""

    def selection_track(**kw: Any) -> dict[str, Any]:
        evidence = dict(kw.get("evidence") or {})
        if not evidence:
            raise ToolOperationError(
                "evidence is required — grab the anchor frame first (frame.grab) "
                "and cite it; a tracked blur on an unseen object is a blind op "
                "(evidence-before-action law)"
            )
        master = _resolve_master(work_dir)
        try:
            duration_s, fps, frame_w, frame_h = probe_video(master)
        except SelectionError as e:
            raise ToolOperationError(str(e)) from e

        anchor_t = float(kw["anchor_t_s"])
        if anchor_t > duration_s:
            raise ToolOperationError(
                f"anchor_t_s {anchor_t:.2f}s beyond the master ({duration_s:.2f}s)"
            )
        anchor_frame = int(anchor_t * fps)
        half = float(kw.get("span_s", 6.0)) / 2.0
        t_start = max(0.0, anchor_t - half)
        t_end = min(duration_s, t_start + float(kw.get("span_s", 6.0)))
        frame_start = int(t_start * fps)
        frame_end = min(int(duration_s * fps) - 1, int(t_end * fps))

        prompt = SelectionPrompt(
            kind="box" if kw.get("w", 0) else "point",
            x=int(kw["x"]),
            y=int(kw["y"]),
            w=int(kw.get("w", 0) or 0),
            h=int(kw.get("h", 0) or 0),
            anchor_frame=anchor_frame,
        )
        try:
            sel = run_selection(
                str(master),
                prompt,
                frame_start=frame_start,
                frame_end=frame_end,
                fps=fps,
                frame_w=frame_w,
                frame_h=frame_h,
                backend=str(kw.get("backend", "") or ""),
            )
        except SelectionNotAvailable as e:
            raise ToolOperationError(str(e)) from e
        except SelectionError as e:
            raise ToolOperationError(str(e)) from e

        out: dict[str, Any] = {
            "ok": True,
            "backend": sel.backend,
            "anchor_frame": sel.anchor_frame,
            "n_hits": len(sel.hits),
            "n_interpolated": sel.n_interpolated,
            "mean_score": sel.mean_score,
            "union_box": sel.union_box(),
            "first_box": sel.hits[0].model_dump() if sel.hits else None,
            "last_box": sel.hits[-1].model_dump() if sel.hits else None,
            "detail": f"{sel.backend} tracked {len(sel.hits)} frames "
            f"({sel.n_interpolated} interpolated, mean score {sel.mean_score:.2f})",
        }

        if kw.get("create_op", True):
            validate_op_window(frame_start, frame_end, duration_s, fps)
            params = sel.to_params()
            params["sigma"] = int(kw.get("sigma", 24))
            params["label"] = str(kw.get("label", "") or "tracked selection")
            full_evidence = {
                **evidence,
                "anchor_frame": anchor_frame,
                "anchor_t_s": round(anchor_t, 3),
                "prompt": sel.prompt,
                "backend": sel.backend,
                "mean_score": sel.mean_score,
            }
            try:
                rec = ops_store.create(
                    OpCreate(
                        kind="blur",
                        frame_start=frame_start,
                        frame_end=frame_end,
                        params=params,
                        origin="agent",
                        evidence=full_evidence,
                        session_id=agent_name,
                        note=f"tracked selection ({sel.backend}, {len(sel.hits)} frames)",
                    ),
                    agent_name=agent_name,
                )
            except ValueError as e:
                raise ToolOperationError(str(e)) from e
            out["op_id"] = rec.id
            out["rev"] = rec.rev
            out["detail"] += f" -> blur op {rec.id[:8]} (rev {rec.rev}, undoable)"
        return out

    registry.register(
        "selection.track",
        "Track ONE object across a time span: point at it on a frame (box x/y/"
        "w/h, or a bare x/y click) at anchor_t_s and get its per-frame path. "
        "Creates a tracked blur op (mode=tracked, undoable via ops.revoke). "
        "Evidence REQUIRED: grab the anchor frame first (frame.grab) and cite "
        "it. CPU template tracker by default; backend='detector-assist' "
        "prefers pipeline face tracks; backend='sam2' needs the optional extra.",
        _SelectionTrack,
        selection_track,
    )
