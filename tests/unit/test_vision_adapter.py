"""Vision LLM adapter — hermetic contract tests (provider pending API key)."""

from __future__ import annotations

import pytest
from director_os.perception.vision import (
    FakeVisionClient,
    FrameRef,
    SceneInsight,
    VisionAdapter,
    VisionError,
)


@pytest.fixture()
def frame(tmp_path):
    p = tmp_path / "frame_0.jpg"
    p.write_bytes(b"\xff\xd8fake-jpeg")
    return FrameRef(timestamp_s=12.5, image_path=p)


def test_understanding_passes_through_canned_insights(frame):
    canned = [
        SceneInsight(
            timestamp_s=12.5,
            kind="redaction_candidate",
            description="plate visible",
            confidence=0.9,
        ),
        SceneInsight(
            timestamp_s=12.5, kind="observation", description="traffic stop", confidence=0.6
        ),
    ]
    client = FakeVisionClient(canned)
    adapter = VisionAdapter(client)
    out = adapter.understand("Where are privacy risks?", [frame])
    assert len(out) == 2
    assert client.calls == [("Where are privacy risks?", 1)]


def test_confidence_filter(frame):
    canned = [
        SceneInsight(timestamp_s=1.0, description="sure", confidence=0.9),
        SceneInsight(timestamp_s=2.0, description="unsure", confidence=0.3),
    ]
    out = VisionAdapter(FakeVisionClient(canned)).understand("q", [frame], min_confidence=0.5)
    assert [i.description for i in out] == ["sure"]


def test_missing_frame_rejected(tmp_path):
    ghost = FrameRef(timestamp_s=0.0, image_path=tmp_path / "missing.jpg")
    with pytest.raises(VisionError, match="missing"):
        VisionAdapter(FakeVisionClient()).understand("q", [ghost])


def test_empty_question_and_empty_frames_rejected(frame):
    adapter = VisionAdapter(FakeVisionClient())
    with pytest.raises(VisionError, match="empty"):
        adapter.understand("   ", [frame])
    with pytest.raises(VisionError, match="no frames"):
        adapter.understand("q", [])
