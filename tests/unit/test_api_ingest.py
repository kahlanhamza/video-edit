"""M3 paste-a-URL ingest — hermetic API tests (no network, no broker).

The done-when is "paste a URL in the browser and watch SSE progress without
touching a terminal": the download is faked, the broker enqueue is recorded,
and only the wire contract + job-row semantics are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    get_settings.cache_clear()
    api_main._engine = None  # force re-bind to the tmp database
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()


@pytest.fixture()
def fake_download(monkeypatch, tmp_path):
    """Replace the network download with a local file writer; record calls."""
    calls: list[tuple[str, Path]] = []

    def _fake(url: str, dest: Path, extra_headers=None, **kwargs) -> Path:
        calls.append((url, Path(dest)))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"fake mp4 bytes")
        return Path(dest)

    import director_os.api.main as api_main

    monkeypatch.setattr(api_main, "download", _fake)
    return calls


@pytest.fixture()
def recorded_enqueue(monkeypatch):
    sent: list[dict] = []

    class _FakeCelery:
        def send_task(self, name, args=None, **kwargs):
            sent.append({"name": name, "args": args, "kwargs": kwargs})
            return None

    import sys

    fake_mod = type(sys)("director_os.worker.celery_app")
    fake_mod.celery_app = _FakeCelery()
    monkeypatch.setitem(sys.modules, "director_os.worker.celery_app", fake_mod)
    return sent


def _make_project(api_client) -> str:
    return api_client.post("/projects", json={"title": "Ingest case"}).json()["id"]


def test_ingest_url_downloads_registers_job_and_enqueues(
    api_client, fake_download, recorded_enqueue
):
    pid = _make_project(api_client)
    r = api_client.post(
        f"/projects/{pid}/ingest",
        json={
            "url": "https://foia.example/eustis-bodycam.mp4",
            "custom_headers": {"Referer": "https://portal.example/case"},
        },
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["enqueued"] is True
    assert body["source_path"].endswith("eustis-bodycam.mp4")
    # download hit the right URL and landed in the project sources dir
    assert fake_download[0][0] == "https://foia.example/eustis-bodycam.mp4"
    assert f"/{pid}/sources/" in str(fake_download[0][1])
    # the fleet got the advance task with the job id + downloaded source
    assert recorded_enqueue[0]["name"] == "directoros.advance_job"
    job_id, proj, src, _db = recorded_enqueue[0]["args"]
    assert proj == pid and src == body["source_path"] == str(fake_download[0][1])
    # a queued job row exists and shows in /jobs (the SSE feed reads the same row)
    jobs = api_client.get(f"/projects/{pid}/jobs").json()
    assert [j["id"] for j in jobs] == [job_id]
    assert jobs[0]["stage"] == "ingest" and jobs[0]["progress_pct"] == 0.0


def test_ingest_url_rejects_non_http_scheme(api_client, fake_download, recorded_enqueue):
    pid = _make_project(api_client)
    r = api_client.post(f"/projects/{pid}/ingest", json={"url": "ftp://x/y.mp4"})
    assert r.status_code == 422
    assert recorded_enqueue == [] and fake_download == []


def test_ingest_url_404_unknown_project(api_client, fake_download, recorded_enqueue):
    r = api_client.post("/projects/nope/ingest", json={"url": "https://x.example/v.mp4"})
    assert r.status_code == 404
    assert recorded_enqueue == []


def test_ingest_url_502_on_download_failure(api_client, monkeypatch, recorded_enqueue):
    import director_os.api.main as api_main

    def _boom(url: str, dest: Path, extra_headers=None, **kwargs) -> Path:
        raise httpx.HTTPStatusError(
            "404", request=httpx.Request("GET", url), response=httpx.Response(404)
        )

    monkeypatch.setattr(api_main, "download", _boom)
    pid = _make_project(api_client)
    r = api_client.post(f"/projects/{pid}/ingest", json={"url": "https://x.example/gone.mp4"})
    assert r.status_code == 502
    assert "download failed" in r.json()["detail"]
    assert recorded_enqueue == []
