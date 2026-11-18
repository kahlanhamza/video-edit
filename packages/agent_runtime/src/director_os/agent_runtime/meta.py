"""Meta Model API (Muse) backend — cost-aware default LLM provider.

Replaces Google AI Studio (region-blocked from the sandbox: "User location is
not supported"). Docs: https://dev.meta.ai/docs.

Wiring:
- base ``https://api.meta.ai/v1`` (``META_DEFAULT_BASE``), ``Authorization:
  Bearer`` with an ``LLM_...`` key (the console docs call it ``MODEL_API_KEY``;
  this repo keeps it in ``LLM_API_KEY`` for one config surface).
- Tool calling on ``/chat/completions`` is native OpenAI shape
  (``tools[].function``, ``tool_calls[].function``) — no translation needed.
- Usage: ``usage.prompt_tokens`` / ``usage.completion_tokens``; when the
  automatic prompt cache hits, ``usage.prompt_tokens_details.cached_tokens``.

Cost law (per 1M tokens, https://dev.meta.ai/docs/pricing-rate-limits):

    muse-wonder-1.0              in 4.00 / cached 0.20  / out 20.00
    muse-spark-1.{1,2,3}         in 1.25 / cached 0.15  / out  4.25
    muse-spark-*-contributor     in 0.10 / cached 0.002 / out  0.20  (TRAINS ON CONTENT)
    muse-ember-1.0  in <=200K    in 0.10 / cached 0.004 / out  0.50
    muse-ember-1.0  in >200K     in 0.30 / cached 0.012 / out  1.50

Default routing: **muse-ember-1.0** for the tool loop (Meta's own guidance:
"sub-agent tasks, tool calls, tight tool loops, high-volume requests") with
``reasoning_effort="minimal"`` — reasoning tokens bill as output, and Ember
defaults to "max". Harder proposals can be switched via ``LLM_MODEL=
muse-spark-1.3``. Contributor variants are 5-12x cheaper but Meta trains on
the content you send — unacceptable for bodycam footage (PII), so they are
rejected unless ``LLM_ALLOW_TRAINING_ON_CONTENT=1``.

Budget guard: every call is checked against a persistent JSONL spend ledger
(``output/llm_spend.jsonl``) BEFORE the HTTP request; ``LLM_BUDGET_USD``
(default 15.0, headroom under the ~20 USD credit balance) stops the run with
``BudgetExceededError`` instead of silently draining the account.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from director_os.agent_runtime.agents import AgentRole
from director_os.agent_runtime.llm import LLMBackendError, OpenAICompatBackend
from director_os.agent_runtime.registry import ToolRegistry

META_DEFAULT_BASE = "https://api.meta.ai/v1"

# muse-ember-1.0 switches to the >200K-input tier (docs: Pricing and rate limits).
EMBER_LARGE_INPUT_THRESHOLD = 200_000


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens."""

    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float


MODEL_PRICES: dict[str, ModelPrice] = {
    "muse-wonder-1.0": ModelPrice(4.00, 0.20, 20.00),
    "muse-spark-1.3": ModelPrice(1.25, 0.15, 4.25),
    "muse-spark-1.2": ModelPrice(1.25, 0.15, 4.25),
    "muse-spark-1.1": ModelPrice(1.25, 0.15, 4.25),
    # Contributor mode: Meta may use the content to train future models.
    "muse-spark-1.3-contributor": ModelPrice(0.10, 0.002, 0.20),
    "muse-spark-1.2-contributor": ModelPrice(0.10, 0.002, 0.20),
    "muse-ember-1.0": ModelPrice(0.10, 0.004, 0.50),
}

EMBER_LARGE_TIER = ModelPrice(0.30, 0.012, 1.50)
CONTRIBUTOR_FALLBACK = MODEL_PRICES["muse-spark-1.3-contributor"]
# Unknown ids are over-estimated with standard Spark pricing so the budget
# guard never under-charges while Meta ships new model versions.
UNKNOWN_FALLBACK = MODEL_PRICES["muse-spark-1.3"]


def price_for(model: str, tokens_in: int = 0) -> ModelPrice:
    """Price row for a model id (Ember tier by total input tokens)."""
    if model == "muse-ember-1.0" and tokens_in > EMBER_LARGE_INPUT_THRESHOLD:
        return EMBER_LARGE_TIER
    if model in MODEL_PRICES:
        return MODEL_PRICES[model]
    if "contributor" in model:
        return CONTRIBUTOR_FALLBACK
    return UNKNOWN_FALLBACK


def estimate_cost_usd(
    model: str, tokens_in: int, tokens_cached: int = 0, tokens_out: int = 0
) -> float:
    """USD cost of one call; cached tokens are a subset of the input tokens."""
    cached = min(max(tokens_cached, 0), max(tokens_in, 0))
    uncached_in = max(tokens_in - cached, 0)
    p = price_for(model, tokens_in)
    return (
        uncached_in * p.input_per_mtok / 1e6
        + cached * p.cached_input_per_mtok / 1e6
        + tokens_out * p.output_per_mtok / 1e6
    )


class BudgetExceededError(LLMBackendError):
    """Raised BEFORE the HTTP call when the persistent spend ledger hits the cap."""


class SpendLedger:
    """Append-only JSONL ledger of LLM spend, shared across processes/restarts."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def record(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def total_usd(self) -> float:
        if not self.path.exists():
            return 0.0
        total = 0.0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    total += float(json.loads(line).get("cost_usd") or 0.0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue  # a torn line from a killed process never blocks a run
        return total


class MetaModelBackend(OpenAICompatBackend):
    """Meta Model API chat-completions backend with cost accounting.

    Adds on top of ``OpenAICompatBackend``: cached-token parsing, per-call
    USD estimation, ledger recording, a pre-call budget cap, and
    ``reasoning_effort`` control (reasoning tokens bill as output).
    """

    provider = "meta"

    DEFAULT_REASONING_EFFORT = "minimal"

    def __init__(
        self,
        api_key: str,
        model: str,
        registry: ToolRegistry,
        niche: str = "bodycam_documentary",
        base_url: str = META_DEFAULT_BASE,
        transport: Any = None,
        *,
        ledger: SpendLedger | None = None,
        budget_usd: float | None = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ):
        super().__init__(api_key, model, registry, niche, base_url=base_url, transport=transport)
        self.ledger = ledger
        self.budget_usd = budget_usd
        self.reasoning_effort = reasoning_effort
        self.last_cost_usd: float | None = None
        self.last_tokens_cached = 0

    # ------------------------------------------------------------- budget
    def spend_usd(self) -> float:
        return self.ledger.total_usd() if self.ledger else 0.0

    def _check_budget(self) -> None:
        if self.budget_usd is None or self.ledger is None:
            return
        total = self.ledger.total_usd()
        if total >= self.budget_usd:
            raise BudgetExceededError(
                f"Meta LLM budget exhausted: ${total:.4f} >= cap ${self.budget_usd:.2f} "
                "(ledger: " + str(self.ledger.path) + "; raise LLM_BUDGET_USD to continue)"
            )

    # ------------------------------------------------------- propose hooks
    def _extra_payload(self) -> dict[str, Any]:
        if self.reasoning_effort:
            return {"reasoning_effort": self.reasoning_effort}
        return {}

    def _absorb_usage(self, usage: dict[str, Any]) -> None:
        super()._absorb_usage(usage)
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        self.last_tokens_cached = cached
        u = self.last_usage or {}
        cost = estimate_cost_usd(
            self.model, int(u.get("tokens_in", 0)), cached, int(u.get("tokens_out", 0))
        )
        self.last_cost_usd = round(cost, 8)
        if self.ledger is not None:
            self.ledger.record(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "purpose": "agent.propose",
                    "tokens_in": int(u.get("tokens_in", 0)),
                    "tokens_cached": cached,
                    "tokens_out": int(u.get("tokens_out", 0)),
                    "cost_usd": round(cost, 8),
                }
            )

    def propose(self, state_summary: str, role: AgentRole) -> list:
        self._check_budget()
        return super().propose(state_summary, role)
