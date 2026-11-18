"""LLM plan-loop endpoint (`POST /projects/{id}/agent/plan`) — real provider wiring.

Covers the env-driven provider choice end to end over HTTP:
- 503 when no LLM provider is configured (build_backend refuses),
- 404 on unknown projects,
- full loop with the native Gemini wire format via MockTransport
  (parametersJsonSchema declarations, thoughtSignature replay, usage report),
- pointer advance to the surviving timeline version.
"""

from __future__ import annotations

import json

import httpx
import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api_plan.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    # plan endpoint must see a clean provider config unless a test sets one
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


@pytest.fixture()
def project_with_timeline(api_client):
    """A project whose active timeline has one 5s raw clip on the master track."""
    import director_os.api.main as api_main
    from director_os.agent_runtime.checkpoint import CheckpointStore
    from director_os.core.db import ProjectRow
    from director_os.otio_engine.contract import RawClipMeta
    from director_os.otio_engine.timeline import (
        ClipSpec,
        GapSpec,
        TimelineSpec,
        TrackSpec,
        build_timeline,
    )

    pid = api_client.post("/projects", json={"title": "Plan case"}).json()["id"]
    timeline = build_timeline(
        TimelineSpec(
            name="plan_case",
            fps=30,
            width=1920,
            height=1080,
            tracks=[
                TrackSpec(
                    name="V1_master",
                    items=[
                        ClipSpec(
                            name="raw_1",
                            source_start_s=0.0,
                            dur_s=5.0,
                            meta=RawClipMeta(asset_id="asset_x"),
                        )
                    ],
                ),
                TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=5.0)]),
            ],
        )
    )
    store = CheckpointStore(api_main._session_factory)
    version = store.checkpoint(timeline, pid, note="plan test seed", created_by="test")
    factory = api_main._session_factory
    assert factory is not None
    with factory() as s:
        p = s.get(ProjectRow, pid)
        p.active_otio_version_id = version
        s.commit()
    return pid, version


GEMINI_PLAN_OK = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "thoughtSignature": "sig-plan-1",
                        "functionCall": {
                            "id": "call_9",
                            "name": "timeline.split_clip",
                            "args": {"item_index": 0, "at_local_s": 1.5},
                        },
                    }
                ]
            }
        }
    ],
    "usageMetadata": {"promptTokenCount": 210, "candidatesTokenCount": 40},
}


def test_agent_plan_unknown_project_404(api_client):
    r = api_client.post("/projects/nope/agent/plan", json={})
    assert r.status_code == 404


def test_agent_plan_503_without_provider(api_client, project_with_timeline):
    pid, _ = project_with_timeline
    r = api_client.post(f"/projects/{pid}/agent/plan", json={})
    assert r.status_code == 503
    assert "LLM not configured" in r.json()["detail"]


def test_agent_plan_422_without_timeline(api_client):
    pid = api_client.post("/projects", json={"title": "empty"}).json()["id"]
    r = api_client.post(f"/projects/{pid}/agent/plan", json={})
    assert r.status_code == 422


def test_agent_plan_full_loop_native_gemini(api_client, project_with_timeline, monkeypatch):
    """Real GeminiRestBackend behind MockTransport: declarations ride in
    parametersJsonSchema, the proposal executes checkpoint-first, token spend
    is reported and the active version advances."""
    import director_os.api.main as api_main
    from director_os.agent_runtime.llm import GeminiRestBackend

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=GEMINI_PLAN_OK)

    def fake_build_chain(registry, niche="bodycam_documentary", **_kw):
        return GeminiRestBackend(
            "k", "gemini-3.8-flash", registry, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(api_main, "build_backend_chain", fake_build_chain)
    pid, v0 = project_with_timeline

    r = api_client.post(f"/projects/{pid}/agent/plan", json={"role": "director"})
    assert r.status_code == 200, r.text
    body = r.json()

    # wire format: declarations via parametersJsonSchema (Gemini 3.x trap #2)
    decl = bodies[0]["tools"][0]["functionDeclarations"][0]
    assert "parametersJsonSchema" in decl and "parameters" not in decl

    assert body["provider"] == "gemini"
    assert body["model"] == "gemini-3.8-flash"
    assert body["ok"] is True
    assert body["steps_ok"] == 1
    assert body["steps"][0]["tool"] == "timeline.split_clip"
    assert body["tokens_in"] == 210 and body["tokens_out"] == 40
    assert body["final_version"] is not None and body["final_version"] > v0

    from director_os.core.db import ProjectRow

    factory = api_main._session_factory
    assert factory is not None
    with factory() as s:
        assert s.get(ProjectRow, pid).active_otio_version_id == body["final_version"]
