"""Live smoke test for the native Gemini REST provider (VISION_API_KEY / LLM_API_KEY).

Verifies the exact thing the pipeline needs from Stage 4 (vision) and the
Director (text): the key is valid, which models the project can see, and that
`gemini-3.5-flash-lite` (scene indexing) + `gemini-3.8-flash` (Frames-on-Demand)
answer. Cost is a handful of tokens (well below $0.001).

Also diagnoses the two known failure modes:
- 400 FAILED_PRECONDITION "User location is not supported" — datacenter region
  geo-blocked by the Generative Language API. The pipeline's GeminiVisionClient
  already retries through the Modal fleet (`vlm_call`, US/EU egress) when
  MODAL_TOKEN_ID/SECRET exist and the fleet is deployed.
- 401/403 — wrong key, wrong project, or API not enabled for the project.

Run from the repo root:
    uv run python scripts/gemini_smoke_test.py
Reads .env (gitignored): LLM_API_KEY / VISION_API_KEY, MODAL_TOKEN_ID/SECRET.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://generativelanguage.googleapis.com"
PIPELINE_MODELS = (
    "gemini-3.5-flash-lite",  # Tier 1 — scene indexing (contact sheets)
    "gemini-3.8-flash",  # Tier 2 — Frames-on-Demand QA + Director text LLM
)


def load_env() -> None:
    """Parse .env into os.environ without overriding existing vars."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def mask(key: str) -> str:
    return f"{key[:4]}...{key[-4:]} ({len(key)} chars)" if key else "(empty)"


def main() -> int:
    load_env()
    key = os.getenv("VISION_API_KEY") or os.getenv("LLM_API_KEY") or ""
    print(f"key: {mask(key)}")
    if not key:
        print("FAIL: no VISION_API_KEY / LLM_API_KEY in .env")
        return 1

    headers = {"x-goog-api-key": key}
    ok = True

    # 1) model listing — what the project is entitled to
    with httpx.Client(timeout=30) as client:
        r = client.get(f"{BASE}/v1beta/models", headers=headers)
    print(f"\n[1] GET /v1beta/models -> {r.status_code}")
    visible: set[str] = set()
    if r.status_code == 200:
        for m in r.json().get("models", []):
            name = str(m.get("name", "")).removeprefix("models/")
            visible.add(name)
        print(f"    {len(visible)} model(s) visible")
    else:
        ok = False
        print(f"    body: {r.text[:300]}")

    for model in PIPELINE_MODELS:
        present = any(v == model or v.startswith(model) for v in visible)
        print(f"    {'OK ' if present else 'MISSING'} {model}")

    # 2) tiny real generation on each pipeline model
    for model in PIPELINE_MODELS:
        body = {
            "contents": [{"role": "user", "parts": [{"text": "Reply with exactly: OK"}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8},
        }
        with httpx.Client(timeout=60) as client:
            r = client.post(
                f"{BASE}/v1beta/models/{model}:generateContent", headers=headers, json=body
            )
        print(f"\n[2] {model}:generateContent -> {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            usage = data.get("usageMetadata", {})
            text = "".join(
                p.get("text", "")
                for p in data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            ).strip()
            print(
                f"    reply: {text!r}  tokens in/out: "
                f"{usage.get('promptTokenCount')}/{usage.get('candidatesTokenCount')}"
            )
        elif "location is not supported" in r.text:
            ok = False
            print("    GEO-BLOCKED: this region cannot call the Generative Language API.")
            modal = bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))
            print(f"    MODAL_TOKEN_ID/SECRET present: {modal}")
            print("    Pipeline path: GeminiVisionClient auto-retries via Modal fleet egress")
            print("    (deploy/modal/vlm_proxy.py must be deployed as 'directoros-vlm').")
        elif r.status_code in (401, 403):
            ok = False
            print(f"    AUTH: {r.text[:300]}")
        elif r.status_code == 404:
            ok = False
            print(f"    MODEL NOT FOUND: {r.text[:200]}")
        else:
            ok = False
            print(f"    body: {r.text[:300]}")

    print("\nVERDICT:", "GO — pipeline can use native Gemini now" if ok else "see diagnosis above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
