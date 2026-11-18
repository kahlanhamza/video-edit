"""Meta Model API (Muse) vision adapter — Muse Spark media understanding (M4).

Unblocks the `VISION_API_KEY` blocker (Google AI Studio 403 PERMISSION_DENIED /
region-blocked from this sandbox) by routing Frames-on-Demand through the same
Meta Model API key that powers the agent LLM (`LLM_API_KEY` + base
``api.meta.ai``). Wire law (docs: dev.meta.ai/docs/media-understanding):

- ``/chat/completions`` OpenAI content parts: ``{"type": "text", ...}`` and
  ``{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}``
  — frames ride inline as data URLs (no hosting, no Files API round-trip).
- Media understanding runs on **Muse Spark** (``muse-spark-1.3`` default);
  hard limit **50 images per request** (HTTP 400 beyond) — this client chunks
  at 40 and merges insights.
- Usage: ``usage.prompt_tokens`` / ``completion_tokens`` with
  ``prompt_tokens_details.cached_tokens`` when the prompt cache hits.

Cost law (per 1M tokens, mirrors ``agent_runtime.meta.MODEL_PRICES``):
Spark standard in 1.25 / cached 0.15 / out 4.25; Wonder 4.00 / 0.20 / 20.00;
Ember small tier 0.10 / 0.004 / 0.50 (vision batches stay far below the 200K
tier switch). Unknown future ids are over-estimated at standard Spark pricing
so the budget guard never under-charges. Contributor variants are 5-12x
cheaper but Meta trains on the content — bodycam frames are exactly the PII
that must never leak into training, so they are rejected unless
``allow_contributor=True`` (same law as the text backend).

Budget guard: every call is checked against the shared JSONL spend ledger
BEFORE the HTTP request; ``JsonlSpendRecorder`` writes the same
``cost_usd`` rows to the same file as ``agent_runtime.meta.SpendLedger``
(default ``output/llm_spend.jsonl``), so text + vision spend add up to one
unified cap (``LLM_BUDGET_USD``).
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from director_os.perception.vision import (
    INSIGHT_INSTRUCTION,
    FrameRef,
    SceneInsight,
    VisionError,
)

if TYPE_CHECKING:
    import httpx

META_VISION_BASE_URL = "https://api.meta.ai/v1"

# (input, cached_input, output) USD per 1M tokens — mirror of
# agent_runtime.meta.MODEL_PRICES; dev.meta.ai/docs/pricing-rate-limits.
VISION_MODEL_PRICES: dict[str, tuple[float, float, float]] = {
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
    "muse-spark-1.1": (1.25, 0.15, 4.25),
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
    "muse-ember-1.0": (0.10, 0.004, 0.50),
}
VISION_UNKNOWN_FALLBACK = VISION_MODEL_PRICES["muse-spark-1.3"]


def vision_price_for(model: str) -> tuple[float, float, float]:
    """Price row for a model id (unknown ids -> conservative Spark standard)."""
    if model in VISION_MODEL_PRICES:
        return VISION_MODEL_PRICES[model]
    return VISION_UNKNOWN_FALLBACK


def estimate_vision_cost_usd(
    model: str, tokens_in: int, tokens_cached: int = 0, tokens_out: int = 0
) -> float:
    """USD cost of one call; cached tokens are a subset of the input tokens."""
    cached = min(max(tokens_cached, 0), max(tokens_in, 0))
    uncached_in = max(tokens_in - cached, 0)
    p_in, p_cached, p_out = vision_price_for(model)
    return uncached_in * p_in / 1e6 + cached * p_cached / 1e6 + tokens_out * p_out / 1e6


class SpendRecorder(Protocol):
    """Duck-typed spend sink — satisfied by agent_runtime.meta.SpendLedger."""

    def record(self, entry: dict[str, Any]) -> None: ...

    def total_usd(self) -> float: ...


class VisionBudgetExceededError(VisionError):
    """Raised BEFORE the HTTP call when the shared spend ledger hits the cap."""


class JsonlSpendRecorder:
    """Same JSONL file/format as ``agent_runtime.meta.SpendLedger``.

    Kept dependency-free on purpose: perception does not import the agent
    runtime, yet both recorders must read/write ONE ledger file so the $20
    credit budget covers text + vision spend together.
    """

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


class MetaVisionClient:
    """VisionClient port over Meta Model API media understanding (Spark)."""

    provider = "meta"

    DEFAULT_MODEL = "muse-spark-1.3"
    MAX_IMAGES_PER_REQUEST = 50  # docs hard limit; HTTP 400 beyond
    CHUNK = 40  # safety margin under the doc limit

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = META_VISION_BASE_URL,
        transport: httpx.BaseTransport | None = None,
        ledger: SpendRecorder | None = None,
        budget_usd: float | None = None,
        reasoning_effort: str = "minimal",
        allow_contributor: bool = False,
        timeout: float = 120.0,
    ):
        if not api_key:
            raise VisionError("MetaVisionClient requires the Model API key (LLM_API_KEY, LLM_...)")
        if "contributor" in model and not allow_contributor:
            raise VisionError(
                f"{model} trains on the content you send (Contributor mode) — "
                "rejected for bodycam PII unless allow_contributor=True"
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self.ledger = ledger
        self.budget_usd = budget_usd
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.last_usage: dict[str, int] | None = None
        self.last_tokens_cached = 0
        self.last_cost_usd: float | None = None

    # ------------------------------------------------------------- budget
    def spend_usd(self) -> float:
        return self.ledger.total_usd() if self.ledger else 0.0

    def _check_budget(self) -> None:
        if self.budget_usd is None or self.ledger is None:
            return
        total = self.ledger.total_usd()
        if total >= self.budget_usd:
            raise VisionBudgetExceededError(
                f"Meta vision budget exhausted: ${total:.4f} >= cap "
                f"${self.budget_usd:.2f} (ledger: {self.ledger.path})"
            )

    # ------------------------------------------------------------- analyze
    def analyze(self, question: str, frames: Sequence[FrameRef]) -> list[SceneInsight]:
        if not question.strip():
            raise VisionError("question must not be empty")
        if not frames:
            raise VisionError("no frames provided")
        self._check_budget()
        insights: list[SceneInsight] = []
        for start in range(0, len(frames), self.CHUNK):
            chunk = list(frames[start : start + self.CHUNK])
            data = self._post(self._payload(question, chunk))
            insights.extend(self._parse(data, chunk))
        return insights

    def _payload(self, question: str, frames: list[FrameRef]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": INSIGHT_INSTRUCTION + question}]
        for f in frames:
            mime = "image/png" if f.image_path.suffix.lower() == ".png" else "image/jpeg"
            data = base64.b64encode(f.image_path.read_bytes()).decode()
            content.append({"type": "text", "text": f"frame at {f.timestamp_s}s:"})
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        with httpx.Client(transport=self._transport, timeout=self.timeout) as client:
            r = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        if r.status_code != 200:
            detail = r.text[:200]
            hint = ""
            if r.status_code == 404 and "model_not_found" in r.text:
                hint = (
                    " — key valid but its team has zero Model API model grants; "
                    "mint the key inside the credited team/project and re-run"
                )
            raise VisionError(f"vision provider {r.status_code}: {detail}{hint}")
        data: dict[str, Any] = r.json()
        self._absorb_usage(data)
        return data

    def _absorb_usage(self, data: dict[str, Any]) -> None:
        usage = data.get("usage", {}) or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        tin = int(usage.get("prompt_tokens", 0))
        tout = int(usage.get("completion_tokens", 0))
        cost = estimate_vision_cost_usd(self.model, tin, cached, tout)
        prev = self.last_usage or {"tokens_in": 0, "tokens_out": 0}
        self.last_usage = {
            "tokens_in": prev["tokens_in"] + tin,
            "tokens_out": prev["tokens_out"] + tout,
        }
        self.last_tokens_cached = self.last_tokens_cached + cached
        self.last_cost_usd = round((self.last_cost_usd or 0.0) + cost, 8)
        if self.ledger is not None:
            self.ledger.record(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "purpose": "vision.analyze",
                    "tokens_in": tin,
                    "tokens_cached": cached,
                    "tokens_out": tout,
                    "cost_usd": round(cost, 8),
                }
            )

    def _parse(self, data: dict[str, Any], frames: list[FrameRef]) -> list[SceneInsight]:
        text = str((data.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
        try:
            if text.startswith("```"):
                text = text.strip("`").removeprefix("json").strip()
            raw = json.loads(text)
            fallback_ts = frames[0].timestamp_s if frames else 0.0
            return [
                SceneInsight(
                    timestamp_s=float(i.get("timestamp_s", fallback_ts)),
                    kind=str(i.get("kind", "observation")),
                    description=str(i.get("description", ""))[:500],
                    confidence=float(i.get("confidence", 0.5)),
                )
                for i in raw
            ]
        except (ValueError, TypeError, KeyError):
            return [
                SceneInsight(
                    timestamp_s=frames[0].timestamp_s,
                    kind="observation",
                    description=text[:500] or "(empty vision response)",
                    confidence=0.3,
                )
            ]


# Shared prompt law with GeminiVisionClient (defined there, imported here to
# keep both providers' answer contract identical).
