"""Cost dashboard + multi-channel publish endpoints (M5)."""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api_m5.db")
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


def test_cost_endpoint_aggregates_tool_logs(api_client, tmp_path):
    pid = api_client.post("/projects", json={"title": "Costs"}).json()["id"]

    import director_os.api.main as api_main
    from director_os.agent_runtime.checkpoint import CheckpointStore

    store = CheckpointStore(api_main._session_factory)
    store.log_tool("director_agent", "timeline.trim_clip", {}, duration_ms=120, project_id=pid)
    store.log_tool(
        "director_agent",
        "llm.propose",
        {"role": "director"},
        duration_ms=800,
        project_id=pid,
        tokens_in=1500,
        tokens_out=350,
    )

    r = api_client.get(f"/projects/{pid}/costs")
    assert r.status_code == 200
    body = r.json()
    assert body["tool_calls"] == 2
    assert body["total_duration_ms"] == 920
    assert body["tokens_in"] == 1500 and body["tokens_out"] == 350
    assert len(body["recent"]) == 2
    assert any(e["tool"] == "llm.propose" for e in body["recent"])
    assert api_client.get("/projects/nope/costs").status_code == 404


def test_multichannel_reports_unconfigured_channels(api_client, tmp_path):
    pid = api_client.post("/projects", json={"title": "Channels"}).json()["id"]
    video = tmp_path / "master.mp4"
    video.write_bytes(b"0" * 128)

    r = api_client.post(
        f"/projects/{pid}/publish-channels",
        json={"video_path": str(video), "title": "Approved cut"},
    )
    assert r.status_code == 200
    channels = {c["channel"]: c["status"] for c in r.json()["channels"]}
    # every PLAN channel is reported; none pretend to be published without creds
    assert set(channels) == {"youtube_draft", "tiktok", "youtube_shorts", "instagram_reels"}
    assert all(s == "not_configured" for s in channels.values())

    # registered adapters publish honestly
    import director_os.api.main as api_main
    from director_os.api.channels import Channel, ChannelPublisher, ChannelUpload

    class StubShorts(ChannelPublisher):
        channel = Channel.youtube_shorts

        def upload(self, video_path, title, description):
            return ChannelUpload(channel=self.channel, status="published", external_id="abc123")

    api_main.set_channel_publisher(StubShorts())
    r2 = api_client.post(
        f"/projects/{pid}/publish-channels",
        json={"video_path": str(video), "title": "Approved cut"},
    )
    statuses = {c["channel"]: c for c in r2.json()["channels"]}
    assert statuses["youtube_shorts"]["status"] == "published"
    assert statuses["youtube_shorts"]["external_id"] == "abc123"
    assert statuses["tiktok"]["status"] == "not_configured"

    assert (
        api_client.post(
            f"/projects/{pid}/publish-channels",
            json={"video_path": str(tmp_path / "gone.mp4"), "title": "x"},
        ).status_code
        == 422
    )
    assert (
        api_client.post(
            "/projects/nope/publish-channels", json={"video_path": str(video), "title": "x"}
        ).status_code
        == 404
    )
