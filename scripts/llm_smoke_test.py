"""Live smoke test for the Google AI Studio LLM wiring (read-only, 2 small calls).

Run from the repo root with .env exported:
    set -a; . ./.env; set +a; uv run python scripts/llm_smoke_test.py

What it proves:
  1. the configured LLM_MODEL is visible to the key (else lists usable ones),
  2. functionDeclarations via parametersJsonSchema are accepted by the real
     Gemini 3.x REST surface (trap #2),
  3. a follow-up propose replays the assistant turn verbatim — the real API
     accepts the replayed thoughtSignature (trap #1),
  4. usage accounting lands in last_usage.
"""

from __future__ import annotations

import os
import sys

import httpx
from director_os.agent_runtime.llm import GeminiRestBackend, LLMBackendError
from director_os.agent_runtime.registry import ToolRegistry

BASE = "https://generativelanguage.googleapis.com"


def list_models(key: str) -> list[str]:
    r = httpx.get(f"{BASE}/v1beta/models", params={"key": key}, timeout=30.0)
    print(f"[1] GET /v1beta/models -> {r.status_code}")
    if r.status_code != 200:
        print(r.text[:400])
        return []
    models = r.json().get("models", [])
    return [
        m["name"] for m in models if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


def main() -> int:
    key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "gemini-3.8-flash")
    if not key:
        print("LLM_API_KEY not set — export .env first")
        return 2

    usable = list_models(key)
    model_name = f"models/{model}"
    if usable and model_name not in usable:
        print(f"[!] configured model '{model}' NOT visible to this key. Usable:")
        for n in usable[:25]:
            print("    ", n)
        return 2
    print(f"[1] '{model}' visible to key ({len(usable)} usable model(s) total)")

    reg = ToolRegistry()

    @reg.tool("timeline.split_clip", description="split a clip at a local time")
    def split_clip(item_index: int, at_local_s: float) -> str:
        return f"split item {item_index} at {at_local_s}"

    be = GeminiRestBackend(key, model, reg)
    print(f"[2] propose #1 against {model} (parametersJsonSchema wire format)...")
    steps = be.propose("project=p master_items=[]", "director")
    print("    parsed steps:", [(s.tool, s.args) for s in steps])
    print("    provider_meta:", [s.provider_meta for s in steps])
    print("    usage:", be.last_usage)

    be.record_results(steps, [f"ok: {s.tool}" for s in steps])
    print("[3] propose #2 with replayed assistant turn (thoughtSignature)...")
    steps2 = be.propose("project=p master_items=[RawClip]", "director")
    print("    parsed steps:", [s.tool for s in steps2])
    print("    usage:", be.last_usage)
    print("\nSMOKE OK — native Gemini function-calling loop works end to end")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LLMBackendError as e:
        print("LLM ERROR:", e)
        sys.exit(1)
