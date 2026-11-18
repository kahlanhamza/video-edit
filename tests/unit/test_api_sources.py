"""Multi-Source Ingest API — sources CRUD, upload, dossier, refresh (M5)."""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api_sources.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    for var in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "OPENROUTER_API_KEY", "VISION_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    api_main._engine = None
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


def test_create_project_with_sources_and_auto_discover(api_client):
    r = api_client.post(
        "/projects",
        json={
            "title": "Eustis FOIA case",
            "video_urls": ["https://commons.example/bodycam.webm"],
            "auto_discover": True,
            "sources": [
                {"kind": "url", "label": "news", "url": "https://news.example/eustis"},
                {"kind": "text", "label": "my notes", "text": "guilty plea in 2023"},
            ],
        },
    )
    assert r.status_code == 201
    pid = r.json()["id"]

    sources = api_client.get(f"/projects/{pid}/sources").json()
    kinds = sorted(s["kind"] for s in sources)
    # news url + pasted text + the video page itself (auto-discover ON)
    assert kinds == ["source_page", "text", "url"]
    assert all(s["status"] == "pending" for s in sources)


def test_create_project_auto_discover_off(api_client):
    pid = api_client.post(
        "/projects",
        json={"title": "t", "video_urls": ["https://x.example/v.mp4"], "auto_discover": False},
    ).json()["id"]
    assert api_client.get(f"/projects/{pid}/sources").json() == []


def test_add_source_later_and_dedupe(api_client):
    pid = api_client.post("/projects", json={"title": "case"}).json()["id"]
    r1 = api_client.post(
        f"/projects/{pid}/sources", json={"kind": "url", "url": "https://court.example/doc"}
    )
    assert r1.status_code == 201
    r2 = api_client.post(
        f"/projects/{pid}/sources", json={"kind": "url", "url": "https://court.example/doc"}
    )
    assert r2.json()["id"] == r1.json()["id"], "duplicate url is idempotent"

    r3 = api_client.post(f"/projects/{pid}/sources", json={"kind": "url", "url": ""})
    assert r3.status_code == 422
    r4 = api_client.post(f"/projects/{pid}/sources", json={"kind": "text", "text": "  "})
    assert r4.status_code == 422


def test_upload_txt_and_pdf_sources(api_client):
    pid = api_client.post("/projects", json={"title": "case"}).json()["id"]
    txt = api_client.post(
        f"/projects/{pid}/sources/upload",
        files={"file": ("notes.txt", b"officer resigned after the arrest", "text/plain")},
        data={"label": "my notes"},
    )
    assert txt.status_code == 201
    assert txt.json()["kind"] == "text"

    pdf = api_client.post(
        f"/projects/{pid}/sources/upload",
        files={"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert pdf.status_code == 201
    assert pdf.json()["kind"] == "pdf"

    bad = api_client.post(
        f"/projects/{pid}/sources/upload",
        files={"file": ("x.exe", b"MZ", "application/octet-stream")},
    )
    assert bad.status_code == 422

    sources = api_client.get(f"/projects/{pid}/sources").json()
    assert len(sources) == 2  # the .exe was rejected before any row was written


def test_dossier_404_then_refresh_builds_heuristically(api_client):
    pid = api_client.post("/projects", json={"title": "case"}).json()["id"]
    assert api_client.get(f"/projects/{pid}/dossier").status_code == 404

    api_client.post(
        f"/projects/{pid}/sources",
        json={
            "kind": "text",
            "label": "sentencing update",
            "text": "Update: he pleaded guilty. Sentenced to 12 months probation.",
        },
    )
    r = api_client.post(f"/projects/{pid}/sources/refresh")
    assert r.status_code == 200
    body = r.json()
    assert body["refreshed"] == 1
    assert body["dossier"]["verdict_outcome"] is not None
    assert "12 months probation" in body["dossier"]["verdict_outcome"]

    got = api_client.get(f"/projects/{pid}/dossier")
    assert got.status_code == 200
    assert got.json()["verdict_outcome"] == body["dossier"]["verdict_outcome"]


def test_refresh_marks_failing_source_with_error(api_client, monkeypatch):
    import httpx

    def dead_fetch(url):
        raise httpx.ConnectError("no route to host (hermetic test)")

    monkeypatch.setattr("director_os.perception.case_dossier.default_fetch", dead_fetch)
    pid = api_client.post("/projects", json={"title": "case"}).json()["id"]
    api_client.post(
        f"/projects/{pid}/sources",
        json={"kind": "url", "url": "https://news.example/missing"},
    )
    r = api_client.post(f"/projects/{pid}/sources/refresh")
    assert r.status_code == 200
    sources = api_client.get(f"/projects/{pid}/sources").json()
    url_row = next(s for s in sources if s["kind"] == "url")
    assert url_row["status"] == "failed"
    assert url_row["error"]
