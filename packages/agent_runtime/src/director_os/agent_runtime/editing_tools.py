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
from director_os.otio_engine.contract import (
    NarratorClipMeta,
    RawClipMeta,
    attach,
)
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
    def _raw_asset_id(self) -> str:
        """Asset id of the first raw clip on the master track."""
        for item in _master(self.timeline):
            meta = item.metadata.get("director_os") if not isinstance(item, ot_schema.Gap) else None
            if meta and meta.get("kind") == "raw":
                return str(meta.get("asset_id", ""))
        raise ToolOperationError("master track has no raw clip to inherit asset_id from")

    def _new_raw_clip(
        self, name: str, asset_id: str, source_start_s: float, dur_s: float, act: str | None = None
    ):
        rate = 30.0
        clip = ot_schema.Clip(name=name)
        clip.source_range = ot_time.TimeRange(
            start_time=ot_time.RationalTime(source_start_s * rate, rate),
            duration=ot_time.RationalTime(dur_s * rate, rate),
        )
        meta: dict[str, Any] = {
            "kind": "raw",
            "asset_id": asset_id,
            "audio_mode": "loudnorm_48k",
            "trim_silence": True,
            "zoom_keyframes": [],
            "blur_regions": [],
            "freeze_callout": None,
        }
        if act:
            meta["act_name"] = act
        clip.metadata.update({"director_os": meta})
        return clip

    def insert_raw_clip(
        self,
        source_start_s: float,
        dur_s: float,
        at_index: int | None = None,
        asset_id: str = "",
        act_name: str = "",
    ) -> dict[str, Any]:
        """Insert a raw clip from ANY point of the source (cold-open hooks).

        The Colab-proven cold open needs climax clips pulled from LATER in the
        video; trim-only tools can never reach them. `at_index=None` appends.
        """
        if dur_s <= 0:
            raise ToolOperationError("dur_s must be > 0")
        if source_start_s < 0:
            raise ToolOperationError("source_start_s must be >= 0")
        asset = asset_id.strip() or self._raw_asset_id()
        track = _master(self.timeline)
        items = list(track)
        if at_index is None:
            at_index = len(items)
        if not 0 <= at_index <= len(items):
            raise ToolOperationError(
                f"at_index {at_index} out of range (master has {len(items)} items)"
            )
        clip = self._new_raw_clip(f"raw_{at_index}", asset, source_start_s, dur_s, act_name or None)
        track.insert(at_index, clip)
        return {
            "ok": True,
            "detail": (
                f"raw clip inserted at {at_index}: src {source_start_s:.2f}s "
                f"+{dur_s:.2f}s (asset {asset[:8]})"
            ),
        }

    def insert_narration_block(
        self,
        script_text: str,
        dur_s: float,
        at_index: int | None = None,
        broll_source_start_s: float | None = None,
        act_name: str = "",
    ) -> dict[str, Any]:
        """Add a narrator block (TTS over muted, darkened B-roll).

        B-roll defaults to the source range of the preceding raw clip (a quiet
        moment of the same scene the narration describes). Volume is 0 by the
        audio law for narration bridges (speech rides the TTS bus).
        """
        if not script_text.strip():
            raise ToolOperationError("script_text must not be empty")
        if not 1.0 <= dur_s <= 120.0:
            raise ToolOperationError("dur_s must be 1..120s")
        track = _master(self.timeline)
        items = list(track)
        if at_index is None:
            at_index = len(items)
        if not 0 <= at_index <= len(items):
            raise ToolOperationError(
                f"at_index {at_index} out of range (master has {len(items)} items)"
            )

        asset = self._raw_asset_id()
        if broll_source_start_s is None:
            broll_source_start_s = 0.0
            for item in reversed(items[:at_index]):
                if isinstance(item, ot_schema.Gap):
                    continue
                prev_meta = item.metadata.get("director_os") or {}
                if prev_meta.get("kind") == "raw" and item.source_range is not None:
                    broll_source_start_s = item.source_range.start_time.to_seconds()
                    break
        rate = 30.0
        clip = ot_schema.Clip(name=f"narration_{at_index}")
        clip.source_range = ot_time.TimeRange(
            start_time=ot_time.RationalTime(broll_source_start_s * rate, rate),
            duration=ot_time.RationalTime(dur_s * rate, rate),
        )
        narr_meta: dict[str, Any] = {
            "kind": "narration",
            "tts_provider": "edge",
            "voice_id": "en-US-BrianMultilingualNeural",
            "script_text": script_text,
            "broll_asset_id": asset,
            "broll_range": [broll_source_start_s, broll_source_start_s + dur_s],
            "broll_audio_volume": 0.0,
            "jl_cut": {"j_s": 0.4},
        }
        if act_name:
            narr_meta["act_name"] = act_name
        clip.metadata.update({"director_os": narr_meta})
        track.insert(at_index, clip)
        return {
            "ok": True,
            "detail": (
                f"narration at {at_index}: {dur_s:.1f}s over broll "
                f"{broll_source_start_s:.1f}s — {script_text[:60]!r}"
            ),
        }

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
            raise ToolOperationError(f"at_local_s {at_local_s:.2f}s outside clip 0..{dur:.2f}s")
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

    def add_narration(
        self, item_index: int, script_text: str, broll_audio_volume: float = 0.0
    ) -> dict[str, Any]:
        """Convert a raw master clip into a narration clip (M2 close-out).

        The voiceover script replaces the clip's source audio; the SAME source
        window keeps playing as b-roll underneath (muted by default —
        broll_audio_volume 0..1 restores ambience). The trimmed in/out points
        are untouched, so the compiled segment still cuts exactly the action
        the director kept; only the audio track swaps to the TTS stem, which
        stage_narration synthesizes content-addressed before render.
        """
        clip = _master_clip(self.timeline, item_index)
        meta = clip.metadata.get("director_os")
        if not meta or meta.get("kind") != "raw":
            raise ToolOperationError(
                f"clip {item_index} is a {meta.get('kind') if meta else 'bare'} clip, "
                "not a raw clip — narrate over raw footage"
            )
        raw = RawClipMeta.model_validate(meta)
        text = script_text.strip()
        if not text:
            raise ToolOperationError("script_text must not be empty")
        start_s = clip.source_range.start_time.to_seconds() if clip.source_range else 0.0
        dur = item_duration_s(clip)
        narr = NarratorClipMeta(
            script_text=text,
            broll_asset_id=raw.asset_id,
            broll_range=(round(start_s, 3), round(start_s + dur, 3)),
            broll_audio_volume=max(0.0, min(1.0, broll_audio_volume)),
        )
        attach(clip, narr)
        return {
            "ok": True,
            "detail": f"narration {item_index}: {text[:48]!r} over "
            f"{raw.asset_id[:8]} {start_s:.1f}+{dur:.1f}s "
            f"(src audio {narr.broll_audio_volume:.2f})",
        }

    # ------------------------------------------------------------ privacy redaction
    def add_blur_region(self, item_index: int, bbox: list[int], sigma: int = 24) -> dict[str, Any]:
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

    # ------------------------------------------------------------ multi-cam
    def switch_camera(
        self,
        item_index: int,
        at_local_s: float,
        dur_s: float,
        angle_asset_id: str,
        angle_offset_s: float = 0.0,
        angle_drift_ppm: float = 0.0,
        current_offset_s: float = 0.0,
    ) -> dict[str, Any]:
        """Show another synced angle for `dur_s` starting at a clip-local time.

        Three-way splits the raw clip (like two `split_clip` calls) and rewires
        the middle piece to the angle's asset, mapping the switch point from
        reference wall time into the angle's source clock via the
        `perception.sync_multicam` offsets: angle_src = wall - offset - drift
        ppm compensation. The angle piece starts CLEAN — blur boxes, zoom
        keyframes and freeze callouts are raw-geometry artifacts of the OLD
        angle and would land wrong on different footage; redact the new angle
        with its own boxes (or a face sweep) after switching.
        """
        clip = _master_clip(self.timeline, item_index)
        meta = _plain(clip.metadata.get("director_os"))
        if not meta or meta.get("kind") != "raw":
            raise ToolOperationError(f"clip {item_index} is not a raw clip")
        if not angle_asset_id.strip():
            raise ToolOperationError("angle_asset_id must not be empty")
        if angle_asset_id == meta.get("asset_id"):
            raise ToolOperationError(
                f"angle_asset_id '{angle_asset_id}' is the clip's current asset"
            )
        dur = item_duration_s(clip)
        if not 0.0 <= at_local_s < dur:
            raise ToolOperationError(f"at_local_s {at_local_s:.2f}s outside clip 0..{dur:.2f}s")
        end = at_local_s + dur_s
        if end > dur + 1e-9:
            raise ToolOperationError(
                f"switch range {at_local_s:.2f}..{end:.2f}s exceeds clip ({dur:.2f}s)"
            )

        sr = clip.source_range
        rate = sr.start_time.rate
        src0 = sr.start_time.to_seconds()
        ref_wall = src0 + at_local_s + current_offset_s
        angle_src = ref_wall - angle_offset_s - angle_drift_ppm * ref_wall / 1e6
        if angle_src < 0:
            raise ToolOperationError(
                f"switch maps before the angle's first frame ({angle_src:.2f}s)"
            )

        left_dur = at_local_s
        right_dur = dur - end
        name = clip.name or "clip"

        def _piece(name_suffix: str, start_s: float, piece_dur: float) -> ot_schema.Clip:
            piece = ot_schema.Clip(name=f"{name}.{name_suffix}")
            piece.source_range = ot_time.TimeRange(
                start_time=ot_time.RationalTime(start_s * rate, rate),
                duration=ot_time.RationalTime(piece_dur * rate, rate),
            )
            piece.metadata.update(copy.deepcopy(clip.metadata))
            return piece

        angle_meta = dict(meta)
        angle_meta["asset_id"] = angle_asset_id
        angle_meta["angle_id"] = angle_asset_id
        angle_meta["trim_silence"] = False  # mid-event switch: never re-trim
        angle_meta["blur_regions"] = []
        angle_meta["zoom_keyframes"] = []
        angle_meta["freeze_callout"] = None

        mid = _piece(f"cam_{angle_asset_id}", angle_src, dur_s)
        mid.metadata.update({"director_os": angle_meta})

        track = _master(self.timeline)
        idx = list(track).index(clip)
        inserts: list[ot_schema.Clip] = []
        if left_dur > 1e-9:
            track[idx] = _piece("a", src0, left_dur)
            inserts.append(mid)
        else:
            track[idx] = mid
        if right_dur > 1e-9:
            inserts.append(_piece("b", src0 + end, right_dur))
        for offset, piece in enumerate(inserts, start=1):
            track.insert(idx + offset, piece)
        n_after = len(inserts) + (1 if left_dur > 1e-9 else 0)
        return {
            "ok": True,
            "detail": (
                f"switched clip {item_index} to '{angle_asset_id}' at wall "
                f"{ref_wall:.2f}s -> angle_src {angle_src:.2f}s for {dur_s:.2f}s "
                f"({n_after} piece(s) from {left_dur:.2f}/{dur_s:.2f}/{right_dur:.2f}s)"
            ),
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
        "timeline.insert_raw_clip",
        "Insert a raw clip from any source time (e.g. a cold-open climax hook "
        "pulled from later in the video) at a master track position.",
        _InsertRawInput,
        lambda **kw: toolbox.insert_raw_clip(**kw),
    )
    reg.register(
        "timeline.add_narration",
        "Insert a NEW narrator block (TTS script over muted, darkened B-roll "
        "from any source moment) at a master track position.",
        _AddNarrationInput,
        lambda **kw: toolbox.insert_narration_block(**kw),
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
        "timeline.switch_camera",
        "Switch the visible camera to another synced angle for a time range "
        "(offsets/drift from perception.sync_multicam).",
        _SwitchCameraInput,
        lambda **kw: toolbox.switch_camera(**kw),
    )
    reg.register(
        "narration.set_text",
        "Rewrite a narration clip's script text (re-renders only that segment).",
        _NarrationInput,
        lambda **kw: toolbox.set_narration_text(**kw),
    )
    reg.register(
        "narration.add",
        "Convert a raw master clip into a NARRATION clip: your voiceover script "
        "replaces the clip's source audio while the same window plays as muted "
        "b-roll (broll_audio_volume 0..1 restores ambience). Use on clips whose "
        "raw audio carries no key dialogue; the TTS clip is synthesized and "
        "cached before render.",
        _NarrationAddInput,
        lambda **kw: toolbox.add_narration(**kw),
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


class _InsertRawInput(BaseModel):
    source_start_s: float = Field(ge=0.0)
    dur_s: float = Field(gt=0.0, le=600.0)
    at_index: int | None = Field(default=None, ge=0)
    asset_id: str = Field(default="", max_length=120)
    act_name: str = Field(default="", max_length=120)


class _AddNarrationInput(BaseModel):
    script_text: str = Field(min_length=1, max_length=4000)
    dur_s: float = Field(ge=1.0, le=120.0)
    at_index: int | None = Field(default=None, ge=0)
    broll_source_start_s: float | None = Field(default=None, ge=0.0)
    act_name: str = Field(default="", max_length=120)


class _SplitInput(BaseModel):
    item_index: int = Field(ge=0)
    at_local_s: float = Field(gt=0.0)


class _MoveInput(BaseModel):
    from_index: int = Field(ge=0)
    to_index: int = Field(ge=0)


class _NarrationInput(BaseModel):
    item_index: int = Field(ge=0)
    script_text: str = Field(min_length=1, max_length=4000)


class _NarrationAddInput(BaseModel):
    item_index: int = Field(ge=0)
    script_text: str = Field(min_length=1, max_length=4000)
    broll_audio_volume: float = Field(default=0.0, ge=0.0, le=1.0)


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


class _SwitchCameraInput(BaseModel):
    item_index: int = Field(ge=0)
    at_local_s: float = Field(ge=0.0)
    dur_s: float = Field(gt=0.0)
    angle_asset_id: str = Field(min_length=1, max_length=120)
    angle_offset_s: float = 0.0
    angle_drift_ppm: float = 0.0
    current_offset_s: float = 0.0
