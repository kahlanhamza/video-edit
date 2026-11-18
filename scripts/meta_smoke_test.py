"""Live smoke test for the Meta Model API (Muse) provider.

Cheapest plan: muse-ember-1.0 + reasoning_effort minimal, three tiny calls
(far below $0.001 at Ember pricing: $0.10/M in, $0.50/M out, $0.004/M cached).
Prints usage, prompt-cache hits, per-call USD cost and the cumulative spend
ledger total. Appends to the same ledger the budget guard reads.

Run from the repo root:
    set -a; . ./.env; set +a; uv run python scripts/meta_smoke_test.py
Requires .env (gitignored): LLM_API_KEY (LLM_...), optionally LLM_BASE_URL /
LLM_MODEL / LLM_BUDGET_USD.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from director_os.agent_runtime.meta import META_DEFAULT_BASE, SpendLedger, estimate_cost_usd

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """Parse .env into os.environ without overriding existing vars."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def post(client: httpx.Client, base: str, key: str, payload: dict) -> dict:
    r = client.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
    return r.json()


def report(tag: str, data: dict, model: str, ledger: SpendLedger) -> float:
    usage = data.get("usage", {})
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    tin = int(usage.get("prompt_tokens", 0))
    tout = int(usage.get("completion_tokens", 0))
    cost = estimate_cost_usd(model, tin, cached, tout)
    ledger.record(
        {
            "provider": "meta",
            "model": model,
            "purpose": f"smoke.{tag}",
            "tokens_in": tin,
            "tokens_cached": cached,
            "tokens_out": tout,
            "cost_usd": round(cost, 8),
        }
    )
    msg = data.get("choices", [{}])[0].get("message", {})
    tool_calls = [c.get("function", {}).get("name") for c in msg.get("tool_calls") or []]
    print(
        f"  {tag}: in={tin} cached={cached} out={tout} "
        f"cost=${cost:.6f} content={str(msg.get('content'))[:40]!r} tools={tool_calls}"
    )
    return cost


def main() -> int:
    load_env()
    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        print("LLM_API_KEY missing (.env)")
        return 1
    base = (os.environ.get("LLM_BASE_URL") or META_DEFAULT_BASE).rstrip("/")
    model = os.environ.get("LLM_MODEL", "muse-ember-1.0")
    ledger = SpendLedger(ROOT / "output" / "llm_spend.jsonl")
    total = 0.0

    try:
        with httpx.Client() as client:
            print(f"base={base} model={model}")
            models = client.get(
                f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=60
            )
            listing = models.json().get("data", []) if models.status_code == 200 else []
            print(f"  /models -> {models.status_code}, {len(listing)} model(s) for this key")
            if not listing:
                print(
                    "  WARNING: key sees no models. In dev.meta.ai (team/project dashboard) "
                    "enable Model API models + attach pay-as-you-go billing, then re-run."
                )

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Return the current wall-clock time.",
                        "parameters": {
                            "type": "object",
                            "properties": {"utc": {"type": "boolean"}},
                            "required": [],
                        },
                    },
                }
            ]
            hello = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a terse pipeline component."},
                    {"role": "user", "content": "Say OK and nothing else."},
                ],
                "reasoning_effort": "minimal",
            }

            # 1) vanilla chat; 2) repeat to observe prompt-cache hits;
            # 3) tool-calling round.
            data = post(client, base, key, hello)
            total += report("chat.minimal", data, model, ledger)
            second = post(client, base, key, hello)
            total += report("chat.repeat", second, model, ledger)
            data_tools = post(client, base, key, {**hello, "tools": tools, "tool_choice": "auto"})
            total += report("chat.tools", data_tools, model, ledger)
    except httpx.HTTPStatusError as e:
        print(f"  FAILED: {e.response.status_code}: {e.response.text[:200]}")
        print(
            "  Fix in the dev.meta.ai dashboard: enable the Model API models for this "
            "team/project and attach the pay-as-you-go billing / credits, then re-run."
        )
        return 1

    print(f"cumulative ledger total: ${ledger.total_usd():.6f} (this run ${total:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
