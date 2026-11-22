"""Provider connection probes — hermetic wire tests (httpx MockTransport).

Pins the cheapest-authenticated-endpoint law: a probe never starts a paid
job, 401/403 = auth-rejected, 404 = authenticated (gladia semantics), 429 =
reachable+authenticated (capacity is not a credential failure), and the
dispatcher resolution order params > saved credentials > env.
"""

from __future__ import annotations

import httpx
import pytest
from director_os.core.probes import (
    GLADIA_DEFAULT_BASE,
    ProbeResult,
    probe_elevenlabs,
    probe_gemini_rest,
    probe_gladia,
    probe_kaggle,
    probe_openai_compat,
    probe_openrouter,
    probe_youtube,
    run_probe,
)


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_openai_compat_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer sk-test"
        body = request.read()
        assert b'"max_tokens": 1' in body or b'"max_tokens":1' in body
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "pong"}}],
                "model": "muse-ember-1.0",
                "usage": {"total_tokens": 3},
            },
        )

    r = probe_openai_compat(
        "sk-test", "https://api.meta.ai/v1", "muse-ember-1.0", transport=_transport(handler)
    )
    assert r.ok and r.extra["served_model"] == "muse-ember-1.0"


def test_openai_compat_auth_rejected():
    r = probe_openai_compat(
        "bad",
        "https://api.groq.com/openai/v1",
        "m",
        transport=_transport(lambda req: httpx.Response(401, json={"e": "no"})),
    )
    assert not r.ok and "401" in r.detail


def test_openai_compat_429_is_reachable_not_broken():
    r = probe_openai_compat(
        "k",
        "https://api.meta.ai/v1",
        "m",
        transport=_transport(lambda req: httpx.Response(429, json={})),
    )
    assert r.ok and "rate" in r.detail.lower()


def test_openai_compat_missing_inputs_fail_fast():
    assert not probe_openai_compat("", "https://x/v1", "m").ok
    assert not probe_openai_compat("k", "", "m").ok
    assert not probe_openai_compat("k", "https://x/v1", "").ok


def test_gemini_rest_ok_and_invalid():
    ok_transport = _transport(lambda req: httpx.Response(200, json={"models": [{"name": "m1"}]}))
    r = probe_gemini_rest("good", transport=ok_transport)
    assert r.ok

    bad_transport = _transport(
        lambda req: httpx.Response(400, json={"error": {"message": "API_KEY_INVALID"}})
    )
    r2 = probe_gemini_rest("bad", transport=bad_transport)
    assert not r2.ok and "API_KEY_INVALID" in r2.detail


def test_gladia_404_means_authenticated():
    """The zero-cost law: nonexistent job + valid key = 404; invalid = 401."""
    r = probe_gladia(
        "glad_key",
        transport=_transport(lambda req: httpx.Response(404, json={"error": "not found"})),
    )
    assert r.ok and "authenticated" in r.detail

    r2 = probe_gladia("wrong", transport=_transport(lambda req: httpx.Response(401, json={})))
    assert not r2.ok

    # the probe must GET a job path, never POST an upload/transcription
    seen: list[httpx.Request] = []

    def recorder(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(404, json={})

    probe_gladia("glad_key", transport=_transport(recorder))
    assert seen[0].method == "GET"
    assert seen[0].url.path.startswith("/v2/transcription/")
    assert str(seen[0].url).startswith(GLADIA_DEFAULT_BASE)


def test_openrouter_key_metadata():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["Authorization"] == "Bearer or-key"
        assert req.url.path.endswith("/key")
        return httpx.Response(200, json={"data": {"usage": 0.42, "limit": None}})

    r = probe_openrouter("or-key", transport=_transport(handler))
    assert r.ok and r.extra["limit"] == "unlimited" and r.extra["usage"] == "0.42"


def test_kaggle_kgat_bearer_and_basic():
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"userName": "casey"})

    r = probe_kaggle("ignored", "KGAT_abc123", transport=_transport(handler))
    assert r.ok and r.extra["user"] == "casey"
    assert seen[0].headers["Authorization"] == "Bearer KGAT_abc123"

    seen.clear()
    r2 = probe_kaggle("casey", "legacy-pass", transport=_transport(handler))
    assert r2.ok
    assert seen[0].headers["Authorization"].startswith("Basic ")


def test_elevenlabs_user_endpoint():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["xi-api-key"] == "el-key"
        assert req.url.path.endswith("/v1/user")
        return httpx.Response(200, json={"subscription": {"tier": "creator"}})

    r = probe_elevenlabs("el-key", transport=_transport(handler))
    assert r.ok and r.extra["tier"] == "creator"


def test_youtube_refresh_roundtrip():
    def ok_handler(req: httpx.Request) -> httpx.Response:
        form = req.read().decode()
        assert "grant_type=refresh_token" in form
        return httpx.Response(200, json={"access_token": "ya29.x", "expires_in": 3600})

    r = probe_youtube("cid", "csecret", "1//refresh", transport=_transport(ok_handler))
    assert r.ok

    def bad_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": "invalid_grant", "error_description": "Token has been expired."}
        )

    r2 = probe_youtube("cid", "csecret", "dead", transport=_transport(bad_handler))
    assert not r2.ok and "expired" in r2.detail
    assert not probe_youtube("", "", "").ok


def test_run_probe_dispatch_and_resolution(tmp_path, monkeypatch):
    """Resolution law: explicit params beat settings (env/overlay); unknown
    provider is a KeyError."""
    from director_os.core.config import get_settings

    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.meta.ai/v1")
    monkeypatch.setenv("LLM_MODEL", "muse-ember-1.0")
    get_settings.cache_clear()

    seen: dict = {}

    def fake_compat(api_key, base_url, model, transport=None, timeout_s=20.0):
        seen.update(api_key=api_key, base_url=base_url, model=model)
        return ProbeResult(True, 1, "ok")

    import director_os.core.probes as probes_mod

    monkeypatch.setattr(probes_mod, "probe_openai_compat", fake_compat)

    # no params -> env values flow in
    run_probe("llm", {}, get_settings())
    assert seen == {
        "api_key": "env-key",
        "base_url": "https://api.meta.ai/v1",
        "model": "muse-ember-1.0",
    }

    # typed params win
    run_probe(
        "llm",
        {
            "LLM_API_KEY": "typed",
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_MODEL": "llama3.1",
        },
        get_settings(),
    )
    assert seen == {
        "api_key": "typed",
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
    }

    # the gemini alias resolves the same primary LLM surface
    monkeypatch.setattr(
        probes_mod,
        "probe_gemini_rest",
        lambda key, base_url=None, transport=None, timeout_s=15.0: ProbeResult(True, 1, "ok"),
    )
    assert run_probe("llm_gemini", {"LLM_API_KEY": "g-key"}, get_settings()).ok

    with pytest.raises(KeyError):
        run_probe("nope", {}, get_settings())
    get_settings.cache_clear()


def test_probe_result_shape():
    assert ProbeResult(True, 5, "d").as_dict() == {
        "ok": True,
        "latency_ms": 5,
        "detail": "d",
        "extra": {},
    }
