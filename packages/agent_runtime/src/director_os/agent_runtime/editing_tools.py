"""Director editing tools — the native function-calling surface (PLAN §3.3B/C).

Every timeline mutation an agent can perform is a strongly-typed tool here.
Arguments are JSON-native (indices, seconds, string bboxes) so a hallucinated
argument is caught by the registry's Pydantic validation BEFORE the timeline
is touched; semantic validation (index in range, cut points sane) happens
inside the tool and raises `ToolOperationError` — also a failed dispatch from
the agent's point of view.

Tools mutate the bound OpenTimelineIO timeline in place. Durability comes from
`CheckpointStore` (agents checkpoint BEFORE each mutating step) — see
`director_os.agent_runtime.checkpoint`.
"""

from __future__ import annotations

import copy
from typing import Any

import opentimelineio.opentime as ot_time
import opentimelineio.schema as ot_schema
from director_os.agent_runtime.registry import ToolRegistry
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    OVERLAY_TRACK,
    item_duration_s,
    track_by_name,
)
from pydantic import BaseModel, Field, ValidationError


class ToolOperationError(RuntimeError):
    """Semantic rejection: args parse but the operation is impossible."""

    def as_dict(self) -> dict[str, Any]:
        return {"error": "tool_operation_failed", "detail": str(self)}


def _master(timeline) -> ot_schema.Track:
    track = track_by_name(timeline, MASTER_TRACK)
    if track is None:
        raise ToolOperationError(f"timeline has no '{MASTER_TRACK}' track")
    return track


def _overlay(timeline) -> ot_schema.Track:
    track = track_by_name(timeline, OVERLAY_TRACK)
    if track is None:
        raise ToolOperationError(f"timeline has no '{OVERLAY_TRACK}' track")
    return track


def _master_clip(timeline, item_index: int) -> ot_schema.Clip:
    items = list(_master(timeline))
    if not 0 <= item_index < len(items):
        raise ToolOperationError(
            f"item_index {item_index} out of range (master has {len(items)} items)"
        )
    item = items[item_index]
    if isinstance(item, ot_schema.Gap):
        raise ToolOperationError(f"item {item_index} is a gap, not a clip")
    return item


class DirectorToolbox:
    """Bound to one project's active OTIO timeline; registered into a ToolRegistry."""

    def __init__(self, timeline, project_id: str, resolution: tuple[int, int] = (1920, 1080)):
        self.timeline = timeline
        self.project_id = project_id
        self.w, self.h = resolution

    # ------------------------------------------------------------ master cut
    def trim_clip(self, item_index: int, new_start_s: float, new_dur_s: float) -> dict[str, Any]:
        """Move the clip's in-point and set a new duration (source seconds)."""
        clip = _master_clip(self.timeline, item_index)
        if new_dur_s <= 0:
            raise ToolOperationError("new_dur_s must be > 0")
        if new_start_s < 0:
            raise ToolOperationError("new_start_s must be >= 0")
        sr = clip.source_range
        rate = sr.start_time.rate
        clip.source_range = ot_time.TimeRange(
            start_time=ot_time.RationalTime(new_start_s * rate, rate),
            duration=ot_time.RationalTime(new_dur_s * rate, rate),
        )
        return {
            "ok": True,
            "detail": f"trimmed clip {item_index} to start={new_start_s:.2f}s dur={new_dur_s:.2f}s",
        }

    def split_clip(self, item_index: int, at_local_s: float) -> dict[str, Any]:
        """Split the clip at a track-local offset; both halves inherit the metadata."""
        clip = _master_clip(self.timeline, item_index)
        dur = item_duration_s(clip)
        if not 0.0 < at_local_s < dur:
            raise ToolOperationError(
                f"at_local_s {at_local_s:.2f}s outside clip 0..{dur:.2f}s"
            )
        sr = clip.source_range
        rate = sr.start_time.rate
        left = ot_schema.Clip(name=f"{clip.name or 'clip'}.a")
        left.source_range = ot_time.TimeRange(
            start_time=sr.start_time,
            duration=ot_time.RationalTime(at_local_s * rate, rate),
        )
        right = ot_schema.Clip(name=f"{clip.name or 'clip'}.b")
        right.source_range = ot_time.TimeRange(
            start_time=sr.start_time + ot_time.RationalTime(at_local_s * rate, rate),
            duration=ot_time.RationalTime((dur - at_local_s) * rate, rate),
        )
        left.metadata.update(copy.deepcopy(clip.metadata))
        right.metadata.update(copy.deepcopy(clip.metadata))
        track = _master(self.timeline)
        idx = list(track).index(clip)
        track[idx] = left
        track.insert(idx + 1, right)
        return {
            "ok": True,
            "detail": f"split clip {item_index} at {at_local_s:.2f}s -> {idx},{idx + 1}",
        }

    def remove_clip(self, item_index: int) -> dict[str, Any]:
        """Delete the clip from the master cut."""
        clip = _master_clip(self.timeline, item_index)
        _master(self.timeline).remove(clip)
        return {"ok": True, "detail": f"removed clip {item_index} ({clip.name})"}

    def move_clip(self, from_index: int, to_index: int) -> dict[str, Any]:
        """Reorder master clips (positions refer to the current item list)."""
        track = _master(self.timeline)
        items = list(track)
        n = len([i for i in items if not isinstance(i, ot_schema.Gap)])
        if not 0 <= from_index < len(items) or not 0 <= to_index < len(items):
            raise ToolOperationError(f"index out of range (master has {len(items)} items)")
        if isinstance(items[from_index], ot_schema.Gap):
            raise ToolOperationError("master track has no gaps to move")
        clip = items[from_index]
        track.remove(clip)
        track.insert(to_index, clip)
        _ = n  # master is gap-free by contract; kept for symmetric validation
        return {"ok": True, "detail": f"moved clip {from_index} -> {to_index}"}

    # ------------------------------------------------------------ narration
    def set_narration_text(self, item_index: int, script_text: str) -> dict[str, Any]:
        """Rewrite the narrator script on a narration clip (re-hash re-renders it)."""
        clip = _master_clip(self.timeline, item_index)
        meta = clip.metadata.get("director_os")
        if not meta or meta.get("kind") != "narration":
            raise ToolOperationError(f"clip {item_index} is not a narration clip")
        if not script_text.strip():
            raise ToolOperationError("script_text must not be empty")
        meta["script_text"] = script_text
        return {"ok": True, "detail": f"narration {item_index} -> {script_text[:48]!r}"}

    # ------------------------------------------------------------ privacy redaction
    def add_blur_region(
        self, item_index: int, bbox: list[int], sigma: int = 24
    ) -> dict[str, Any]:
        """Add a privacy blur box (canvas pixels [x, y, w, h]) to a raw clip."""
        clip = _master_clip(self.timeline, item_index)
        meta = clip.metadata.get("director_os")
        if not meta or meta.get("kind") != "raw":
            raise ToolOperationError(f"clip {item_index} is not a raw clip")
        if sigma < 1 or sigma > 200:
            raise ToolOperationError("sigma must be 1..200")
        _validate_bbox(bbox, self.w, self.h)
        region = {
            "scope": "clip",
            "bbox": dict(zip(("x", "y", "w", "h"), bbox, strict=True)),
            "sigma": sigma,
        }
        meta.setdefault("blur_regions", []).append(region)
        return {"ok": True, "detail": f"blur added to clip {item_index}: {region}"}

    def remove_blur_region(self, item_index: int, region_index: int) -> dict[str, Any]:
        """Remove one blur box by index within the clip's blur_regions list."""
        clip = _master_clip(self.timeline, item_index)
        meta = clip.metadata.get("director_os")
        if not meta or meta.get("kind") != "raw":
            raise ToolOperationError(f"clip {item_index} is not a raw clip")
        # copy -> mutate -> reassign: holding OTIO AnyDictionary proxies across
        # mutation crashes (underlying C++ storage is destroyed on GC)
        regions = _plain(list(meta.get("blur_regions", [])))
        if not 0 <= region_index < len(regions):
            raise ToolOperationError(
                f"region_index {region_index} out of range (clip has {len(regions)})"
            )
        removed = regions.pop(region_index)
        meta["blur_regions"] = regions
        return {
            "ok": True,
            "detail": f"removed blur {region_index} from clip {item_index}: {removed}",
        }

    # ------------------------------------------------------------ overlay cards
    def add_card(
        self,
        template: str,
        title: str,
        start_s: float,
        dur_s: float,
        subtitle: str = "",
        fade_s: float = 0.5,
    ) -> dict[str, Any]:
        """Insert a graphic card at an absolute timeline position (splits gaps)."""
        if dur_s < 0.5:
            raise ToolOperationError("dur_s must be >= 0.5s")
        if start_s < 0:
            raise ToolOperationError("start_s must be >= 0")
        track = _overlay(self.timeline)
        t = 0.0
        for i, item in enumerate(track):
            dur = item_duration_s(item)
            covers = t <= start_s and start_s + dur_s <= t + dur + 1e-9
            if isinstance(item, ot_schema.Gap) and covers:
                return _insert_card(
                    track, i, t, item, template, title, subtitle, start_s, dur_s, fade_s
                )
            t += dur
        raise ToolOperationError(
            f"no free gap at {start_s:.1f}..{start_s + dur_s:.1f}s on '{OVERLAY_TRACK}'"
        )


def _insert_card(  # noqa: PLR0913 — positional glue for the gap-splitting helper
    track, gap_idx, gap_t0, gap, template, title, subtitle, start_s, dur_s, fade_s
) -> dict:
    rate = 30  # overlay timings ride the master rate; compiler normalizes on read
    pre = start_s - gap_t0
    post = gap_t0 + item_duration_s(gap) - (start_s + dur_s)
    meta = {
        "director_os": {
            "kind": "card",
            "template": template,
            "fields": {"title": title, **({"subtitle": subtitle} if subtitle else {})},
            "fade_s": fade_s,
        }
    }
    card = ot_schema.Clip(name=f"card_{template}")
    card.source_range = ot_time.TimeRange(
        start_time=ot_time.RationalTime(0, rate),
        duration=ot_time.RationalTime(dur_s * rate, rate),
    )
    card.metadata.update(meta)
    new_items: list[Any] = []
    if pre > 1e-9:
        g = ot_schema.Gap()
        g.source_range = ot_time.TimeRange(
            start_time=ot_time.RationalTime(0, rate),
            duration=ot_time.RationalTime(pre * rate, rate),
        )
        new_items.append(g)
    new_items.append(card)
    if post > 1e-9:
        g2 = ot_schema.Gap()
        g2.source_range = ot_time.TimeRange(
            start_time=ot_time.RationalTime(0, rate),
            duration=ot_time.RationalTime(post * rate, rate),
        )
        new_items.append(g2)
    track[gap_idx] = new_items[0]
    for offset, extra in enumerate(new_items[1:], start=1):
        track.insert(gap_idx + offset, extra)
    return {
        "ok": True,
        "detail": f"card '{template}' inserted at {start_s:.1f}s for {dur_s:.1f}s",
    }


def _plain(value: Any) -> Any:
    """Deep-convert OTIO AnyDictionary/AnyVector proxies to plain Python.

    OTIO proxies register as ``collections.abc.Mapping``/``Sequence`` but are
    NOT ``dict``/``list`` subclasses, and their C++ backing dies with the
    parent object — so every metadata read-back that must outlive the proxy
    goes through here.
    """
    from collections.abc import Mapping, Sequence

    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [_plain(v) for v in value]
    return value


def _validate_bbox(bbox: list[int], w: int, h: int) -> None:
    if len(bbox) != 4:
        raise ToolOperationError(f"bbox must be [x, y, w, h], got {len(bbox)} values")
    try:
        model = _BBoxInput.model_validate({"bbox": bbox})
    except ValidationError as e:
        raise ToolOperationError(f"invalid bbox {bbox}: {e}") from e
    x, y, bw, bh = model.bbox
    if x + bw > w or y + bh > h:
        raise ToolOperationError(f"bbox {bbox} exceeds canvas {w}x{h}")


class _BBoxInput(BaseModel):
    bbox: tuple[int, int, int, int] = Field(...)


def build_director_registry(toolbox: DirectorToolbox) -> ToolRegistry:
    """Register the director toolset into a fresh registry (agents never share one)."""
    reg = ToolRegistry()
    reg.register(
        "timeline.trim_clip",
        "Move a master clip's in-point and duration (source seconds).",
        _TrimInput,
        lambda **kw: toolbox.trim_clip(**kw),
    )
    reg.register(
        "timeline.split_clip",
        "Split a master clip at a track-local offset; both halves keep metadata.",
        _SplitInput,
        lambda **kw: toolbox.split_clip(**kw),
    )
    reg.register(
        "timeline.remove_clip",
        "Delete a master clip by item index.",
        _IndexInput,
        lambda **kw: toolbox.remove_clip(**kw),
    )
    reg.register(
        "timeline.move_clip",
        "Reorder a master clip from one item index to another.",
        _MoveInput,
        lambda **kw: toolbox.move_clip(**kw),
    )
    reg.register(
        "narration.set_text",
        "Rewrite a narration clip's script text (re-renders only that segment).",
        _NarrationInput,
        lambda **kw: toolbox.set_narration_text(**kw),
    )
    reg.register(
        "redaction.add_blur",
        "Add a privacy blur box [x, y, w, h] (canvas px) to a raw clip.",
        _BlurAddInput,
        lambda **kw: toolbox.add_blur_region(**kw),
    )
    reg.register(
        "redaction.remove_blur",
        "Remove a privacy blur box by region index.",
        _BlurRemoveInput,
        lambda **kw: toolbox.remove_blur_region(**kw),
    )
    reg.register(
        "cards.add",
        "Insert a graphic card at an absolute timeline position (splits gaps).",
        _CardInput,
        lambda **kw: toolbox.add_card(**kw),
    )
    return reg


class _IndexInput(BaseModel):
    item_index: int = Field(ge=0)


class _TrimInput(BaseModel):
    item_index: int = Field(ge=0)
    new_start_s: float = Field(ge=0.0)
    new_dur_s: float = Field(gt=0.0)


class _SplitInput(BaseModel):
    item_index: int = Field(ge=0)
    at_local_s: float = Field(gt=0.0)


class _MoveInput(BaseModel):
    from_index: int = Field(ge=0)
    to_index: int = Field(ge=0)


class _NarrationInput(BaseModel):
    item_index: int = Field(ge=0)
    script_text: str = Field(min_length=1, max_length=4000)


class _BlurAddInput(BaseModel):
    item_index: int = Field(ge=0)
    bbox: list[int]
    sigma: int = Field(default=24, ge=1, le=200)


class _BlurRemoveInput(BaseModel):
    item_index: int = Field(ge=0)
    region_index: int = Field(ge=0)


class _CardInput(BaseModel):
    template: str
    title: str = Field(min_length=1, max_length=120)
    start_s: float = Field(ge=0.0)
    dur_s: float = Field(ge=0.5, le=60.0)
    subtitle: str = Field(default="", max_length=200)
    fade_s: float = Field(default=0.5, ge=0.0, le=5.0)
