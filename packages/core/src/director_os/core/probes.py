"""Provider connection probes — the Settings UI "Test connection" buttons.

Each probe hits the provider's CHEAPEST authenticated endpoint (never starts
a paid job, never uploads audio) and returns a ProbeResult(ok, latency,
detail). All HTTP probes accept an httpx transport so tests are hermetic
(MockTransport — same pattern as the Gladia client). Params resolution
order: explicit probe params (what the user just typed) > saved Studio
credentials > .env — so a button tests exactly what will be used.

Probe semantics per provider:
- llm_openai_compat  POST /chat/completions max_tokens=1 (Meta Muse, OpenRouter,
                     Groq, DeepSeek, Together, Ollama, LM Studio, vLLM, ...)
- llm_gemini         GET /v1beta/models — native Google AI Studio
- gladia             GET /v2/transcription/{probe-id}: 404 == authenticated
                     (invalid key would 401) — zero-cost key validation
- openrouter         GET /auth/key — returns key metadata (usage/limit)
- kaggle             GET /api/v1/hello — Bearer KGAT or basic auth
- elevenlabs         GET /v1/user — account + subscription info
- youtube            POST oauth2.googleapis.com/token refresh — proves the
                     OAuth client + refresh token without touching the API
- modal              SDK auth round-trip via modal.App.lookup (real SDK, no
                     transport injection — classify by exception kind)
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com"
OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
ELEVENLABS_DEFAULT_BASE = "https://api.elevenlabs.io"
GLADIA_DEFAULT_BASE = "https://api.gladia.io"
KAGGLE_DEFAULT_BASE = "https://www.kaggle.com"
META_DEFAULT_BASE = "https://api.meta.ai/v1"

_PROBE_JOB_ID = "directoros-probe-nonexistent-job"


@dataclass
class ProbeResult:
    ok: bool
    latency_ms: int
    detail: str
    extra: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
            "extra": self.extra,
        }


class _ProbeError(Exception):
    pass


def _timed() -> float:
    return time.monotonic()


# --- LLM probes -------------------------------------------------------------


def probe_openai_compat(
    api_key: str,
    base_url: str,
    model: str,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = 20.0,
) -> ProbeResult:
    """Any OpenAI-compatible /chat/completions gateway (incl. Meta Muse)."""
    t0 = _timed()
    base = base_url.rstrip("/")
    if not api_key:
        return ProbeResult(False, 0, "API key is empty")
    if not base:
        return ProbeResult(
            False, 0, "Base URL is empty (blank = native Gemini, use the gemini probe)"
        )
    if not model:
        return ProbeResult(False, 0, "Model id is empty (e.g. muse-ember-1.0, gpt-4o-mini)")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            r = client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
    except httpx.HTTPError as e:
        return ProbeResult(False, int((time.monotonic() - t0) * 1000), f"unreachable: {e}")
    latency = int((time.monotonic() - t0) * 1000)
    if r.status_code == 200:
        body = r.json()
        served = str(body.get("model") or model)
        usage = body.get("usage") or {}
        extra = {"served_model": served}
        if usage.get("total_tokens") is not None:
            extra["total_tokens"] = str(usage["total_tokens"])
        return ProbeResult(True, latency, f"chat/completions answered ({served})", extra)
    if r.status_code in (401, 403):
        return ProbeResult(False, latency, f"auth rejected ({r.status_code}): {r.text[:200]}")
    if r.status_code == 404:
        return ProbeResult(
            False,
            latency,
            f"model or endpoint not found (404) — check base URL and model id: {r.text[:200]}",
        )
    if r.status_code == 429:
        return ProbeResult(
            True, latency, "reachable + authenticated (rate limited right now)", {"status": "429"}
        )
    return ProbeResult(False, latency, f"HTTP {r.status_code}: {r.text[:200]}")


def probe_gemini_rest(
    api_key: str,
    base_url: str = GEMINI_DEFAULT_BASE,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = 15.0,
) -> ProbeResult:
    """Native Google AI Studio (Generative Language REST)."""
    t0 = _timed()
    if not api_key:
        return ProbeResult(False, 0, "API key is empty")
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            r = client.get(
                f"{base_url.rstrip('/')}/v1beta/models", params={"key": api_key, "pageSize": 1}
            )
    except httpx.HTTPError as e:
        return ProbeResult(False, int((time.monotonic() - t0) * 1000), f"unreachable: {e}")
    latency = int((time.monotonic() - t0) * 1000)
    if r.status_code == 200:
        models = r.json().get("models", [])
        return ProbeResult(
            True,
            latency,
            f"authenticated — {len(models) + 0}+ models visible",
            {"count": str(len(models))},
        )
    detail = r.text[:200]
    if "API_KEY_INVALID" in detail or r.status_code in (400, 401, 403):
        return ProbeResult(False, latency, f"auth rejected ({r.status_code}): {detail}")
    if "location" in detail.lower() or r.status_code == 403:
        return ProbeResult(
            False, latency, f"region-blocked or forbidden ({r.status_code}): {detail}"
        )
    return ProbeResult(False, latency, f"HTTP {r.status_code}: {detail}")


def probe_gladia(
    api_key: str,
    base_url: str = GLADIA_DEFAULT_BASE,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = 15.0,
) -> ProbeResult:
    """Zero-cost auth check: a GET for a nonexistent job 404s with a valid
    key (401 with an invalid one). No upload, no job created."""
    t0 = _timed()
    if not api_key:
        return ProbeResult(False, 0, "API key is empty")
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            r = client.get(
                f"{base_url.rstrip('/')}/v2/transcription/{_PROBE_JOB_ID}",
                headers={"x-gladia-key": api_key},
            )
    except httpx.HTTPError as e:
        return ProbeResult(False, int((time.monotonic() - t0) * 1000), f"unreachable: {e}")
    latency = int((time.monotonic() - t0) * 1000)
    if r.status_code == 404:
        return ProbeResult(
            True, latency, "authenticated (probe job 404 = key valid)", {"signal": "404"}
        )
    if r.status_code in (401, 403):
        return ProbeResult(False, latency, f"auth rejected ({r.status_code}): {r.text[:200]}")
    return ProbeResult(False, latency, f"HTTP {r.status_code}: {r.text[:200]}")


def probe_openrouter(
    api_key: str,
    base_url: str = OPENROUTER_DEFAULT_BASE,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = 15.0,
) -> ProbeResult:
    t0 = _timed()
    if not api_key:
        return ProbeResult(False, 0, "API key is empty")
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            r = client.get(
                f"{base_url.rstrip('/')}/key", headers={"Authorization": f"Bearer {api_key}"}
            )
    except httpx.HTTPError as e:
        return ProbeResult(False, int((time.monotonic() - t0) * 1000), f"unreachable: {e}")
    latency = int((time.monotonic() - t0) * 1000)
    if r.status_code == 200:
        data = r.json().get("data", {})
        usage = data.get("usage")
        limit = data.get("limit")
        extra = {
            "usage": str(usage) if usage is not None else "?",
            "limit": str(limit) if limit is not None else "unlimited",
        }
        return ProbeResult(True, latency, "key valid — metadata returned", extra)
    if r.status_code in (401, 403):
        return ProbeResult(False, latency, f"auth rejected ({r.status_code}): {r.text[:200]}")
    return ProbeResult(False, latency, f"HTTP {r.status_code}: {r.text[:200]}")


def probe_kaggle(
    username: str,
    key: str,
    base_url: str = KAGGLE_DEFAULT_BASE,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = 15.0,
) -> ProbeResult:
    t0 = _timed()
    if not key:
        return ProbeResult(False, 0, "KAGGLE_KEY is empty")
    if key.startswith("KGAT_"):
        headers = {"Authorization": f"Bearer {key}"}
    else:  # legacy username:password basic auth
        encoded = base64.b64encode(f"{username or ''}:{key}".encode()).decode()
        headers = {"Authorization": f"Basic {encoded}"}
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            r = client.get(f"{base_url.rstrip('/')}/api/v1/hello", headers=headers)
    except httpx.HTTPError as e:
        return ProbeResult(False, int((time.monotonic() - t0) * 1000), f"unreachable: {e}")
    latency = int((time.monotonic() - t0) * 1000)
    if r.status_code == 200:
        who = str(r.json().get("userName") or username or "?")
        return ProbeResult(True, latency, f"authenticated as {who}", {"user": who})
    if r.status_code in (401, 403):
        return ProbeResult(
            False, latency, f"auth rejected ({r.status_code}) — key must be a KGAT_ token"
        )
    return ProbeResult(False, latency, f"HTTP {r.status_code}: {r.text[:200]}")


def probe_elevenlabs(
    api_key: str,
    base_url: str = ELEVENLABS_DEFAULT_BASE,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = 15.0,
) -> ProbeResult:
    t0 = _timed()
    if not api_key:
        return ProbeResult(False, 0, "API key is empty")
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            r = client.get(f"{base_url.rstrip('/')}/v1/user", headers={"xi-api-key": api_key})
    except httpx.HTTPError as e:
        return ProbeResult(False, int((time.monotonic() - t0) * 1000), f"unreachable: {e}")
    latency = int((time.monotonic() - t0) * 1000)
    if r.status_code == 200:
        sub = r.json().get("subscription") or {}
        tier = str(sub.get("tier") or "?")
        return ProbeResult(True, latency, f"authenticated ({tier} tier)", {"tier": tier})
    if r.status_code in (401, 403):
        return ProbeResult(False, latency, f"auth rejected ({r.status_code}): {r.text[:200]}")
    return ProbeResult(False, latency, f"HTTP {r.status_code}: {r.text[:200]}")


def probe_youtube(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = 15.0,
) -> ProbeResult:
    """Refresh-token round-trip: proves client + token without the Data API."""
    t0 = _timed()
    if not (client_id and client_secret and refresh_token):
        return ProbeResult(False, 0, "client_id, client_secret and refresh_token are all required")
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            r = client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
    except httpx.HTTPError as e:
        return ProbeResult(False, int((time.monotonic() - t0) * 1000), f"unreachable: {e}")
    latency = int((time.monotonic() - t0) * 1000)
    if r.status_code == 200 and r.json().get("access_token"):
        return ProbeResult(True, latency, "refresh token valid — access token minted")
    detail = r.json().get("error_description") or r.text[:200]
    return ProbeResult(False, latency, f"token refresh failed ({r.status_code}): {detail}")


def probe_modal(token_id: str, token_secret: str) -> ProbeResult:
    """Real SDK auth round-trip (modal.App.lookup). NotFound on a nonexistent
    app AFTER successful auth == credentials valid."""
    t0 = _timed()
    if not (token_id and token_secret):
        return ProbeResult(False, 0, "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are required")
    try:
        import modal
        import modal.exception
    except ImportError:
        return ProbeResult(False, 0, "modal package not installed in this environment")
    os.environ["MODAL_TOKEN_ID"] = token_id
    os.environ["MODAL_TOKEN_SECRET"] = token_secret
    try:
        modal.App.lookup("directoros-credentials-probe", create_if_missing=False)
    except modal.exception.NotFoundError:
        return ProbeResult(
            True,
            int((time.monotonic() - t0) * 1000),
            "authenticated (probe app not found = auth ok)",
        )
    except modal.exception.AuthError as e:
        return ProbeResult(False, int((time.monotonic() - t0) * 1000), f"auth rejected: {e}")
    except Exception as e:  # network/other — classify honestly
        return ProbeResult(
            False, int((time.monotonic() - t0) * 1000), f"lookup failed: {type(e).__name__}: {e}"
        )
    return ProbeResult(True, int((time.monotonic() - t0) * 1000), "authenticated")


# --- catalog + dispatcher ----------------------------------------------------

PROBE_TIMEOUT_S = 20.0


def run_probe(
    provider: str,
    params: dict[str, str],
    settings: Any,
    transport: httpx.BaseTransport | None = None,
) -> ProbeResult:
    """Dispatch with resolution: params > saved credentials (in settings) > env.

    `settings` is the effective Settings object (overlay already applied).
    """

    def p(key: str) -> str:
        v = (params.get(key) or "").strip()
        if v:
            return v
        return str(getattr(settings, key.lower(), "") or "")

    if provider in ("llm", "llm_openai_compat", "meta", "openai_compat"):
        base = p("LLM_BASE_URL")
        model = p("LLM_MODEL") or "muse-ember-1.0"
        return probe_openai_compat(p("LLM_API_KEY"), base, model, transport=transport)
    if provider in ("llm_gemini", "gemini"):
        return probe_gemini_rest(p("LLM_API_KEY"), transport=transport)
    if provider in ("vision", "vision_gemini"):
        # OpenAI-compatible vision endpoint when a Base URL is set (Groq,
        # DeepSeek, Ollama, LM Studio, any /chat/completions gateway); blank
        # Base URL = native Gemini generateContent.
        base = p("VISION_BASE_URL")
        if base:
            return probe_openai_compat(
                p("VISION_API_KEY"),
                base,
                p("VISION_MODEL") or "gemini-3.5-flash-lite",
                transport=transport,
            )
        return probe_gemini_rest(p("VISION_API_KEY"), transport=transport)
    if provider == "gladia":
        return probe_gladia(p("GLADIA_API_KEY"), transport=transport)
    if provider == "openrouter":
        return probe_openrouter(
            p("OPENROUTER_API_KEY"),
            p("OPENROUTER_BASE_URL") or OPENROUTER_DEFAULT_BASE,
            transport=transport,
        )
    if provider == "kaggle":
        return probe_kaggle(p("KAGGLE_USERNAME"), p("KAGGLE_KEY"), transport=transport)
    if provider == "elevenlabs":
        return probe_elevenlabs(p("ELEVENLABS_API_KEY"), transport=transport)
    if provider == "youtube":
        return probe_youtube(
            p("YOUTUBE_CLIENT_ID"),
            p("YOUTUBE_CLIENT_SECRET"),
            p("YOUTUBE_REFRESH_TOKEN"),
            transport=transport,
        )
    if provider == "modal":
        return probe_modal(p("MODAL_TOKEN_ID"), p("MODAL_TOKEN_SECRET"))
    raise KeyError(f"unknown probe provider: {provider}")
