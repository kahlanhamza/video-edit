"""core.kaggle — hermetic wire-contract tests (MockTransport, no live API).

The live contract (bearer KGAT auth, camelCase push body, query-param status,
output log harvest) is additionally pinned by a manual smoke run documented
in the module docstring; these tests keep every byte of it from drifting.
"""

from __future__ import annotations

import json

import httpx
import pytest
from director_os.core.kaggle import (
    KaggleClient,
    KaggleError,
    KaggleNotConfigured,
    kaggle_client_from_env,
)


def _client(handler) -> KaggleClient:
    # production always pins KAGGLE_USERNAME once discovered (hamzakhl); the
    # dedicated discovery test below covers the empty-username KGAT path
    return KaggleClient(
        username="hamzakhl", key="KGAT_testtoken123", transport=httpx.MockTransport(handler)
    )


def test_kgat_token_discovers_username_from_hello():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = request.headers.get("Authorization", "")
        return httpx.Response(200, json={"userName": "hamzakhl"})

    c = KaggleClient(username="", key="KGAT_testtoken123", transport=httpx.MockTransport(handler))
    assert c.whoami() == "hamzakhl"
    assert seen["/api/v1/hello"] == "Bearer KGAT_testtoken123"


def test_bearer_auth_header_used_for_kgat_tokens():
    """Every request the client makes rides the KGAT bearer header."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = request.headers.get("Authorization", "")
        if request.url.path == "/api/v1/kernels/push":
            return httpx.Response(200, json={"ref": "/code/hamzakhl/s", "url": "u", "error": ""})
        return httpx.Response(200, json={"userName": "hamzakhl"})

    c = _client(handler)
    c.push_kernel("my-slug", "Valid Title Length", "print(1)")
    assert seen["/api/v1/kernels/push"] == "Bearer KGAT_testtoken123"


def test_legacy_key_without_username_fails_loud():
    with pytest.raises(KaggleNotConfigured, match="KAGGLE_USERNAME"):
        KaggleClient(username="", key="abc123def456abc123def456abc12345")


def test_push_body_camelcase_wire_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.read())
        return httpx.Response(
            200,
            json={
                "ref": "/code/hamzakhl/directoros-gpu-smoke",
                "url": "u",
                "versionNumber": 1,
                "error": "",
            },
        )

    c = _client(handler)
    res = c.push_kernel(
        "directoros-gpu-smoke", "DirectorOS GPU Smoke", "print(1)", gpu=True, internet=False
    )
    assert seen["path"] == "/api/v1/kernels/push"
    body = seen["body"]
    assert body["slug"] == "hamzakhl/directoros-gpu-smoke"
    assert body["newTitle"] == "DirectorOS GPU Smoke"
    assert body["kernelType"] == "script"
    assert body["language"] == "python"
    assert body["enableGpu"] is True
    assert body["enableInternet"] is False
    assert body["isPrivate"] is True
    assert res["versionNumber"] == 1


def test_push_rejects_error_field_and_short_titles():
    c = _client(lambda req: httpx.Response(200, json={"error": "invalid kernel type"}))
    with pytest.raises(KaggleError, match="invalid kernel type"):
        c.push_kernel("s", "Valid Title", "x")
    with pytest.raises(KaggleError, match="five characters"):
        c.push_kernel("s", "tiny", "x")


def test_status_query_params_and_terminal_statuses():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"status": "complete"})

    c = _client(handler)
    assert c.kernel_status("hamzakhl/my-slug") == "complete"
    assert seen["params"] == {"userName": "hamzakhl", "kernelSlug": "my-slug"}


def test_run_kernel_end_to_end_harvest(monkeypatch):
    """push -> poll(running -> complete) -> output(log parse + file download)."""
    calls = {"status": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/kernels/push":
            return httpx.Response(200, json={"ref": "/code/u/s", "url": "u", "error": ""})
        if path == "/api/v1/kernels/status":
            calls["status"] += 1
            st = "running" if calls["status"] == 1 else "complete"
            return httpx.Response(200, json={"status": st})
        if path == "/api/v1/kernels/output":
            log = json.dumps(
                [
                    {"stream_name": "stdout", "data": 'DIRECTOROS_RESULT={"cuda": true}'},
                ]
            )
            return httpx.Response(200, json={"logNullable": log, "files": ["result.json"]})
        if path == "/api/v1/kernels/output/download/u/s/result.json":
            return httpx.Response(200, content=b'{"ok": true}')
        return httpx.Response(404, text="nope")

    c = _client(handler)
    monkeypatch.setattr("director_os.core.kaggle.time.sleep", lambda *_: None)
    import tempfile as _tf
    from pathlib import Path

    out_dir = Path(_tf.mkdtemp())
    res = c.run_kernel("s", "Valid Title", "code", out_dir=out_dir, poll_interval_s=0.0)
    assert res.status == "complete"
    assert res.result_json() == {"cuda": True}
    assert res.files == ["result.json"]
    assert (out_dir / "result.json").read_text() == '{"ok": true}'


def test_run_kernel_error_status_surfaces_log_tail(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/kernels/push":
            return httpx.Response(200, json={"ref": "/code/u/s", "error": ""})
        if path == "/api/v1/kernels/status":
            return httpx.Response(200, json={"status": "error"})
        if path == "/api/v1/kernels/output":
            log = json.dumps([{"stream_name": "stderr", "data": "Traceback: OOM at frame 4211"}])
            return httpx.Response(200, json={"logNullable": log, "files": []})
        return httpx.Response(404)

    c = _client(handler)
    monkeypatch.setattr("director_os.core.kaggle.time.sleep", lambda *_: None)
    with pytest.raises(KaggleError, match="OOM at frame 4211"):
        c.run_kernel("s", "Valid Title", "code", poll_interval_s=0.0)


def test_kaggle_client_from_env_requires_key(monkeypatch):
    from director_os.core.config import get_settings

    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    get_settings.cache_clear()
    with pytest.raises(KaggleNotConfigured):
        kaggle_client_from_env()
    get_settings.cache_clear()
