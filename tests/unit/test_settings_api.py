"""Settings API surface (ADR-0012) — hermetic wire-contract tests.

Pins: catalog shape with masked secrets, save/load/delete round-trips
through the vault, probe endpoint contract (monkeypatched — no network),
and 400-on-unknown-key fail-closed behavior. Raw secret values must NEVER
appear in any response body.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    import director_os.api.main as api_main
    from director_os.core.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.delenv("GLADIA_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    get_settings.cache_clear()
    api_main._engine = None  # force re-bind to the tmp database
    with TestClient(api_main.app) as client:
        yield client
    get_settings.cache_clear()
    api_main._engine = None


def _providers(client):
    r = client.get("/settings/providers")
    assert r.status_code == 200
    return r.json()


def test_providers_catalog_shape_and_masking(api_client):
    data = _providers(api_client)
    assert data["ok"] is True
    ids = [p["id"] for p in data["providers"]]
    assert {
        "llm",
        "gladia",
        "vision",
        "openrouter",
        "kaggle",
        "modal",
        "elevenlabs",
        "youtube",
    } <= set(ids)
    llm = next(p for p in data["providers"] if p["id"] == "llm")
    keys = [f["key"] for f in llm["fields"]]
    assert {"LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"} <= set(keys)
    for p in data["providers"]:
        for f in p["fields"]:
            assert f["kind"] in ("secret", "text")
            assert f["source"] in ("studio", "env", None)
            if f["kind"] == "secret":
                # masked hints only — a raw key never leaves the API
                assert f["value"] == "" or f["value"].startswith("••••")


def test_save_load_delete_roundtrip(api_client):
    body = {
        "entries": [
            {"key": "GLADIA_API_KEY", "value": "glad_live_abc123"},
            {"key": "LLM_MODEL", "value": "studio-model"},
        ]
    }
    r = api_client.put("/settings/credentials", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True and data["version"] >= 1

    gladia = next(p for p in data["providers"] if p["id"] == "gladia")
    field = gladia["fields"][0]
    assert field["source"] == "studio"
    assert field["configured"] is True
    assert field["value"] == "••••c123"
    assert "glad_live_abc123" not in r.text  # raw secret never echoed back

    # the overlay is live: a fresh catalog read still shows studio source
    data2 = _providers(api_client)
    gladia2 = next(p for p in data2["providers"] if p["id"] == "gladia")
    assert gladia2["fields"][0]["source"] == "studio"

    # delete -> falls back (env unset here) -> source None
    r3 = api_client.delete("/settings/credentials/GLADIA_API_KEY")
    assert r3.status_code == 200
    gladia3 = next(p for p in r3.json()["providers"] if p["id"] == "gladia")
    assert gladia3["fields"][0]["source"] is None
    assert gladia3["fields"][0]["configured"] is False


def test_unknown_key_rejected_400(api_client):
    r = api_client.put(
        "/settings/credentials", json={"entries": [{"key": "DATABASE_URL", "value": "x"}]}
    )
    assert r.status_code == 400
    r2 = api_client.delete("/settings/credentials/DATABASE_URL")
    assert r2.status_code == 400


def test_empty_value_ignored(api_client):
    r = api_client.put(
        "/settings/credentials", json={"entries": [{"key": "GLADIA_API_KEY", "value": "   "}]}
    )
    assert r.status_code == 200
    gladia = next(p for p in r.json()["providers"] if p["id"] == "gladia")
    assert gladia["fields"][0]["configured"] is False


def test_probe_endpoint_contract(api_client, monkeypatch):
    """The probe endpoint dispatches to core.probes with resolved params and
    turns bad providers into 400, probe errors into ok=False (never a 500)."""
    from director_os.core.probes import ProbeResult

    seen: dict = {}

    _KNOWN_PROVIDERS = {"gladia", "llm", "llm_gemini"}

    def _fake_run_probe(provider, params, settings, transport=None):
        if provider not in _KNOWN_PROVIDERS:
            raise KeyError(f"unknown probe provider: {provider}")
        seen["provider"] = provider
        seen["params"] = dict(params)
        return ProbeResult(True, 321, "fake probe ok", {"served_model": "muse-ember-1.0"})

    import director_os.core.probes as probes_mod

    monkeypatch.setattr(probes_mod, "run_probe", _fake_run_probe)

    r = api_client.post(
        "/settings/test", json={"provider": "gladia", "params": {"GLADIA_API_KEY": "typed-key"}}
    )
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "ok": True,
        "latency_ms": 321,
        "detail": "fake probe ok",
        "extra": {"served_model": "muse-ember-1.0"},
    }
    assert seen["params"]["GLADIA_API_KEY"] == "typed-key"

    r2 = api_client.post("/settings/test", json={"provider": "does-not-exist", "params": {}})
    assert r2.status_code == 400


def test_probe_endpoint_never_500s_on_bad_credentials(api_client, monkeypatch):
    import director_os.core.probes as probes_mod

    def _boom(provider, params, settings, transport=None):
        raise RuntimeError("connection exploded")

    monkeypatch.setattr(probes_mod, "run_probe", _boom)
    r = api_client.post("/settings/test", json={"provider": "gladia", "params": {}})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert "exploded" in r.json()["detail"]


def test_probe_uses_saved_credential_when_no_params(api_client, monkeypatch):
    """Params empty -> the probe resolves from the overlay (saved key)."""
    import director_os.core.probes as probes_mod
    from director_os.core.probes import ProbeResult

    seen: dict = {}

    def _fake_run_probe(provider, params, settings, transport=None):
        seen["params"] = dict(params)
        seen["settings_gladia"] = settings.gladia_api_key
        return ProbeResult(True, 10, "ok")

    monkeypatch.setattr(probes_mod, "run_probe", _fake_run_probe)

    api_client.put(
        "/settings/credentials", json={"entries": [{"key": "GLADIA_API_KEY", "value": "saved-key"}]}
    )
    r = api_client.post("/settings/test", json={"provider": "gladia", "params": {}})
    assert r.status_code == 200
    assert seen["settings_gladia"] == "saved-key"
