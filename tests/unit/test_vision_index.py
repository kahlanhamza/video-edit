"""Two-tier Gemini vision stack — contact sheets, frames-on-demand, model tiers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from director_os.perception.vision import (
    FakeVisionClient,
    GeminiVisionClient,
    SceneInsight,
    VisionAdapter,
)
from director_os.perception.vision_index import (
    build_contact_sheets,
    frames_on_demand,
    index_model,
    index_timeline,
    make_vision_clients,
    qa_model,
    qa_placement,
    suggest_fod_timestamps,
)


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    out = tmp_path / "tiny.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=4",
        "-frames:v", "40", "-y", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return out


def test_contact_sheets_one_per_window(tiny_video, tmp_path):
    sheets = build_contact_sheets(str(tiny_video), tmp_path / "sheets", window_s=2.0)
    assert len(sheets) == 2
    assert (sheets[0].t_start_s, sheets[0].t_end_s) == (0.0, 2.0)
    assert (sheets[1].t_start_s, sheets[1].t_end_s) == (2.0, 4.0)
    assert all(len(s.frame_times) == 9 for s in sheets)
    for s in sheets:
        assert s.path.exists() and s.path.stat().st_size > 2_000


def test_contact_sheet_frame_times_are_absolute_and_sorted(tiny_video, tmp_path):
    sheets = build_contact_sheets(str(tiny_video), tmp_path / "sheets", window_s=2.0)
    second = sheets[1]
    assert all(2.0 <= t <= 4.0 for t in second.frame_times)
    assert second.frame_times == tuple(sorted(second.frame_times))


def test_index_timeline_keeps_absolute_and_clamps_stray_answers(tiny_video, tmp_path):
    sheets = build_contact_sheets(str(tiny_video), tmp_path / "sheets", window_s=2.0)
    canned = [
        SceneInsight(timestamp_s=2.5, kind="event", description="door opens", confidence=0.8),
        SceneInsight(timestamp_s=99.0, kind="risk", description="hallucinated", confidence=0.9),
    ]
    adapter = VisionAdapter(FakeVisionClient(canned))
    insights = index_timeline(adapter, sheets, window_s=2.0)
    assert len(insights) == 4  # 2 canned answers x 2 sheets
    clamped = [i for i in insights if i.description == "hallucinated"]
    assert {i.timestamp_s for i in clamped} == {0.0, 2.0}  # clamped into sheet windows


def test_frames_on_demand_pulls_fullres_frames_around_t(tiny_video, tmp_path):
    refs = frames_on_demand(str(tiny_video), t_s=2.0, n=3, span_s=2.0, out_dir=tmp_path / "fod")
    assert len(refs) == 3
    for ref in refs:
        assert ref.exists and ref.image_path.stat().st_size > 1_000
        assert 1.0 <= ref.timestamp_s <= 3.0
    assert [r.timestamp_s for r in refs] == sorted(r.timestamp_s for r in refs)


def test_qa_placement_routes_through_flagship_port(tiny_video, tmp_path):
    refs = frames_on_demand(str(tiny_video), t_s=2.0, n=2, span_s=1.0, out_dir=tmp_path / "fod")
    fake = FakeVisionClient(
        [
            SceneInsight(
                timestamp_s=2.0, kind="risk", description="card would cover face", confidence=0.7
            )
        ]
    )
    out = qa_placement(VisionAdapter(fake), refs, "punch-in zoom 1.4x on the driver", t_s=2.0)
    assert fake.calls[0][0].startswith("You are QA-critic")
    assert fake.calls[0][1] == len(refs)
    assert out[0].description == "card would cover face"


def test_suggest_fod_timestamps_dedupes_within_5s():
    insights = [
        SceneInsight(timestamp_s=10.0, kind="risk", description="a", confidence=0.8),
        SceneInsight(timestamp_s=12.0, kind="risk", description="b", confidence=0.8),  # too close
        # not in kinds
        SceneInsight(timestamp_s=30.0, kind="observation", description="c", confidence=0.9),
        SceneInsight(timestamp_s=40.0, kind="event", description="d", confidence=0.3),  # low conf
    ]
    assert suggest_fod_timestamps(insights) == [10.0]


def test_model_tiers_and_env_overrides(monkeypatch):
    monkeypatch.delenv("VISION_INDEX_MODEL", raising=False)
    monkeypatch.delenv("VISION_QA_MODEL", raising=False)
    assert index_model() == "gemini-3.5-flash-lite"
    assert qa_model() == "gemini-3.8-flash"

    index_adapter, qa_adapter = make_vision_clients("key")
    assert isinstance(index_adapter._client, GeminiVisionClient)
    assert index_adapter._client.model == "gemini-3.5-flash-lite"
    assert qa_adapter._client.model == "gemini-3.8-flash"

    blank_index, blank_qa = make_vision_clients("")
    assert isinstance(blank_index._client, FakeVisionClient)
    assert isinstance(blank_qa._client, FakeVisionClient)

    monkeypatch.setenv("VISION_INDEX_MODEL", "my-lite")
    assert index_model() == "my-lite"
