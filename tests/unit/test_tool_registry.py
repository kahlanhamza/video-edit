"""Typed tool registry — native function-calling engine (PLAN §3.1)."""

from __future__ import annotations

import pytest
from director_os.agent_runtime.registry import ToolDispatchError, ToolRegistry
from pydantic import BaseModel, Field


def test_decorator_registration_and_dispatch():
    reg = ToolRegistry()

    @reg.tool("timeline.add_clip", description="Add a clip to the OTIO timeline")
    def add_clip(track: str, asset_id: str, start_s: float, dur_s: float = 5.0) -> dict:
        return {"track": track, "asset_id": asset_id, "start": start_s, "dur": dur_s}

    result = reg.dispatch(
        "timeline.add_clip", {"track": "V1_master", "asset_id": "bwc_1", "start_s": 12.5}
    )
    assert result == {"track": "V1_master", "asset_id": "bwc_1", "start": 12.5, "dur": 5.0}


def test_hallucinated_args_rejected_before_execution():
    reg = ToolRegistry()
    calls = []

    @reg.tool("perception.inspect_frames")
    def inspect(timestamps: list[float]) -> int:
        calls.append(timestamps)
        return len(timestamps)

    with pytest.raises(ToolDispatchError):
        reg.dispatch("perception.inspect_frames", {"timestamps": "now"})
    assert calls == [], "validator must reject before the tool body runs"


def test_unknown_tool_and_schema_generation():
    reg = ToolRegistry()

    @reg.tool("audio.mix_ducking")
    def mix(duck_db: float = -18.0) -> float:
        return duck_db

    with pytest.raises(ToolDispatchError):
        reg.dispatch("does_not_exist", {})
    schema = reg.openai_schema("audio.mix_ducking")
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "audio.mix_ducking"
    assert "properties" in schema["function"]["parameters"]
    assert [s["function"]["name"] for s in reg.openai_schemas()] == ["audio.mix_ducking"]


def test_explicit_model_registration():
    class BlurArgs(BaseModel):
        x: int = Field(ge=0)
        y: int = Field(ge=0)

    reg = ToolRegistry()
    reg.register("redact.add_blur_region", "Add blur", BlurArgs, lambda x, y: {"at": (x, y)})
    assert reg.dispatch("redact.add_blur_region", {"x": 10, "y": 20}) == {"at": (10, 20)}
    with pytest.raises(ToolDispatchError):
        reg.dispatch("redact.add_blur_region", {"x": -3, "y": 0})
