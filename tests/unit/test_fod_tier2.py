"""Frames-on-Demand (Tier 2) wiring — flagship model QA on flagged moments."""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.perception.vision import FakeVisionClient, SceneInsight, VisionAdapter
from director_os.perception.vision_index import (
    frames_on_demand,
    make_vision_clients,
    qa_placement,
    suggest_fod_timestamps,
)


def test_suggest_fod_picks_risky_moments():
    insights = [
        SceneInsight(timestamp_s=12.0, kind="risk", description="obscured lens", confidence=0.8),
        SceneInsight(timestamp_s=13.5, kind="risk", description="obscured lens 2", confidence=0.8),
        SceneInsight(timestamp_s=40.0, kind="event", description="gunshot", confidence=0.7),
        SceneInsight(timestamp_s=60.0, kind="observation", description="calm", confidence=0.9),
        SceneInsight(timestamp_s=90.0, kind="event", description="too unsure", confidence=0.2),
    ]
    picks = suggest_fod_timestamps(insights)
    assert 12.0 in picks
    assert 40.0 in picks
    assert 60.0 not in picks  # observation kind
    assert 90.0 not in picks  # below confidence floor
    assert all(abs(a - b) > 5.0 for a, b in zip(picks, picks[1:], strict=False)), "5s dedupe"


def test_make_vision_clients_blank_key_fakes():
    idx, qa = make_vision_clients("")
    assert isinstance(idx._client, FakeVisionClient)
    assert isinstance(qa._client, FakeVisionClient)


def test_models_default_to_pipeline_tiers(monkeypatch):
    monkeypatch.delenv("VISION_INDEX_MODEL", raising=False)
    monkeypatch.delenv("VISION_QA_MODEL", raising=False)
    from director_os.perception.vision_index import index_model, qa_model

    assert index_model() == "gemini-3.5-flash-lite"
    assert qa_model() == "gemini-3.8-flash"


def test_env_overrides_model_tier(monkeypatch):
    monkeypatch.setenv("VISION_INDEX_MODEL", "gemini-9.9-flash")
    monkeypatch.setenv("VISION_QA_MODEL", "gemini-9.8-pro")
    from director_os.perception.vision_index import index_model, qa_model

    assert index_model() == "gemini-9.9-flash"
    assert qa_model() == "gemini-9.8-pro"


def test_qa_placement_contract(tmp_path):
    from PIL import Image

    frames = []
    for i in range(3):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 40, 30, 30)).save(p)
        frames.append(p)

    class Frame:
        def __init__(self, ts, path):
            self.timestamp_s = ts
            self.image_path = path

        @property
        def exists(self):
            return self.image_path.exists()

    from director_os.perception.vision import FrameRef

    refs = [FrameRef(timestamp_s=float(i), image_path=frames[i]) for i in range(3)]
    fake = FakeVisionClient(
        canned=[
            SceneInsight(timestamp_s=1.0, kind="risk", description="covers plate", confidence=0.9)
        ]
    )
    found = qa_placement(VisionAdapter(fake), refs, "place card at corner", 1.0)
    assert len(found) == 1 and found[0].kind == "risk"
    # the prompt carried the proposal + timestamp
    assert "QA-critic" in fake.calls[0][0]
    assert "1.00" in fake.calls[0][0]


def test_frames_on_demand_span_math():
    pytest.importorskip("subprocess")
    # frames_on_demand needs a real video: contract check on the span math
    # via a synthetic 2s clip
    import subprocess

    out_dir = Path("/tmp/dos_fod_test")
    src = out_dir / "tiny.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=30:duration=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    refs = frames_on_demand(str(src), 1.0, n=3, span_s=1.0, out_dir=out_dir, tile_w=64)
    assert len(refs) == 3
    assert all(0.4 <= r.timestamp_s <= 1.6 for r in refs)
