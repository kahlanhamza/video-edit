"""Meta Model API (Muse) vision adapter — hermetic wire/cost/budget/routing tests.

Pins (docs: dev.meta.ai/docs/media-understanding):
- /chat/completions OpenAI content parts with inline base64 data-URL frames,
- 50-image hard limit handled by chunking (client chunks at 40) with merged
  insights and summed usage,
- prompt-cache usage parsing + unified spend ledger rows (same JSONL file the
  text-LLM backend writes) + pre-call budget cap,
- contributor-mode frames rejected (Meta trains on the content — bodycam PII),
- 404 model_not_found surfaces the entitlement hint,
- routing: VISION_PROVIDER=meta | gemini | fake, auto = Gemini key wins, else
  Meta key on an api.meta.ai base, else hermetic fake.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
from director_os.core.config import get_settings
from director_os.perception.vision import (
    FakeVisionClient,
    FrameRef,
    GeminiVisionClient,
    VisionError,
    build_vision_client,
)
from director_os.perception.vision_meta import (
    JsonlSpendRecorder,
    MetaVisionClient,
    VisionBudgetExceededError,
    estimate_vision_cost_usd,
)

META_VISION_OK = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": (
                    '```json\n[{"timestamp_s": 12.5, "kind": "redaction_candidate", '
                    '"description": "plate visible", "confidence": 0.9}]\n```'
                ),
            }
        }
    ],
    "usage": {
        "prompt_tokens": 1100,
        "completion_tokens": 60,
        "prompt_tokens_details": {"cached_tokens": 800},
    },
}


def make_frames(tmp_path: Path, n: int, start_ts: float = 0.0):
    frames = []
    for i in range(n):
        p = tmp_path / f"frame_{i}.jpg"
        p.write_bytes(b"\xff\xd8fake-jpeg-" + str(i).encode())
        frames.append(FrameRef(timestamp_s=start_ts + i, image_path=p))
    return frames


def make_client(
    tmp_path,
    handler,
    *,
    model: str = "muse-spark-1.3",
    budget: float | None = 15.0,
    ledger: JsonlSpendRecorder | None = None,
    **kw,
) -> MetaVisionClient:
    return MetaVisionClient(
        "LLM_k",
        model=model,
        transport=httpx.MockTransport(handler),
        ledger=ledger if ledger is not None else JsonlSpendRecorder(tmp_path / "spend.jsonl"),
        budget_usd=budget,
        **kw,
    )


# ------------------------------------------------------------------ pricing
def test_vision_cost_math_with_cached_discount():
    # 300 uncached in + 800 cached in + 60 out on Spark standard
    cost = estimate_vision_cost_usd("muse-spark-1.3", 1100, 800, 60)
    expected = 300 * 1.25 / 1e6 + 800 * 0.15 / 1e6 + 60 * 4.25 / 1e6
    assert abs(cost - expected) < 1e-12


def test_vision_unknown_model_priced_conservatively():
    assert estimate_vision_cost_usd("muse-spark-9.9", 1_000_000, 0, 0) == 1.25
    assert estimate_vision_cost_usd("muse-spark-9.9", 1_000_000, 1_000_000, 0) == 0.15


# ------------------------------------------------------------ wire + parse
def test_wire_shape_data_url_frames_and_ledger(tmp_path):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=META_VISION_OK)

    frames = make_frames(tmp_path, 1)
    ledger_path = tmp_path / "spend.jsonl"
    client = make_client(tmp_path, handler, ledger=JsonlSpendRecorder(ledger_path))
    out = client.analyze("Where are privacy risks?", frames)

    assert seen["auth"] == "Bearer LLM_k"
    assert "/chat/completions" in seen["url"]
    assert seen["body"]["model"] == "muse-spark-1.3"
    assert seen["body"]["reasoning_effort"] == "minimal"
    content = seen["body"]["messages"][0]["content"]
    assert content[0]["type"] == "text" and "JSON array" in content[0]["text"]
    img_part = content[2]
    assert img_part["type"] == "image_url"
    url = img_part["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == frames[0].image_path.read_bytes()
    assert [i.kind for i in out] == ["redaction_candidate"]
    assert client.last_usage == {"tokens_in": 1100, "tokens_out": 60}
    assert client.last_tokens_cached == 800
    expected = 300 * 1.25 / 1e6 + 800 * 0.15 / 1e6 + 60 * 4.25 / 1e6
    assert client.last_cost_usd == pytest.approx(expected, abs=1e-9)
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert rows[0]["purpose"] == "vision.analyze"
    assert rows[0]["tokens_cached"] == 800


def test_chunking_over_doc_limit_merges_insights(tmp_path):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        bodies.append(body)
        n_frames = sum(1 for p in body["messages"][0]["content"] if p.get("type") == "image_url")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                [
                                    {
                                        "timestamp_s": float(len(bodies)),
                                        "kind": "observation",
                                        "description": f"chunk {len(bodies)} saw {n_frames}",
                                        "confidence": 0.8,
                                    }
                                ]
                            ),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    frames = make_frames(tmp_path, 45)  # > client.CHUNK (40), <= doc limit 50
    client = make_client(tmp_path, handler)
    out = client.analyze("q", frames)

    assert len(bodies) == 2  # 40 + 5
    counts = [
        sum(1 for p in b["messages"][0]["content"] if p.get("type") == "image_url") for b in bodies
    ]
    assert counts == [40, 5]
    assert [i.description for i in out] == ["chunk 1 saw 40", "chunk 2 saw 5"]
    assert client.last_usage == {"tokens_in": 200, "tokens_out": 20}


def test_malformed_output_falls_back_to_low_confidence(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "I see a traffic stop."}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 8},
            },
        )

    frames = make_frames(tmp_path, 1, start_ts=7.0)
    client = make_client(tmp_path, handler)
    out = client.analyze("q", frames)
    assert len(out) == 1
    assert out[0].confidence == 0.3
    assert out[0].timestamp_s == 7.0
    assert "traffic stop" in out[0].description


# ------------------------------------------------------------------ budget
def test_budget_cap_blocks_vision_call_before_http(tmp_path):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=META_VISION_OK)

    ledger = JsonlSpendRecorder(tmp_path / "spend.jsonl")
    ledger.record({"cost_usd": 15.0, "model": "muse-spark-1.3"})
    client = make_client(tmp_path, handler, ledger=ledger, budget=15.0)
    with pytest.raises(VisionBudgetExceededError, match="budget exhausted"):
        client.analyze("q", make_frames(tmp_path, 1))
    assert calls == []  # no HTTP request was paid for


# ---------------------------------------------------------- contributor law
def test_contributor_frames_rejected_unless_opt_in(tmp_path):
    with pytest.raises(VisionError, match="trains on the content"):
        MetaVisionClient("LLM_k", model="muse-spark-1.3-contributor")
    client = MetaVisionClient(
        "LLM_k",
        model="muse-spark-1.3-contributor",
        allow_contributor=True,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=META_VISION_OK)),
    )
    assert client.analyze("q", make_frames(tmp_path, 1))


# ----------------------------------------------------------------- errors
def test_http_404_entitlement_hint(tmp_path):
    client = make_client(
        tmp_path,
        lambda r: httpx.Response(
            404, json={"error": {"code": "model_not_found", "message": "no access"}}
        ),
    )
    with pytest.raises(VisionError, match="zero Model API model grants"):
        client.analyze("q", make_frames(tmp_path, 1))


def test_empty_question_and_frames_rejected(tmp_path):
    client = make_client(tmp_path, lambda r: httpx.Response(200, json=META_VISION_OK))
    with pytest.raises(VisionError, match="empty"):
        client.analyze("   ", make_frames(tmp_path, 1))
    with pytest.raises(VisionError, match="no frames"):
        client.analyze("q", [])


def test_missing_api_key_rejected():
    with pytest.raises(VisionError, match="requires the Model API key"):
        MetaVisionClient("")


# ---------------------------------------------------------------- routing
def test_routing_explicit_provider_meta(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_PROVIDER", "meta")
    monkeypatch.setenv("LLM_API_KEY", "LLM_k")
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.meta.ai/v1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        client = build_vision_client()
        assert isinstance(client, MetaVisionClient)
        assert client.ledger is not None
        assert client.ledger.path == tmp_path / "llm_spend.jsonl"
        assert client.budget_usd == 15.0
    finally:
        get_settings.cache_clear()


def test_routing_auto_meta_via_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_PROVIDER", "")
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "LLM_k")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.meta.ai/v1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        assert isinstance(build_vision_client(), MetaVisionClient)
    finally:
        get_settings.cache_clear()


def test_routing_auto_gemini_key_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_PROVIDER", "")
    monkeypatch.setenv("VISION_API_KEY", "AIza_k")
    monkeypatch.setenv("LLM_API_KEY", "LLM_k")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.meta.ai/v1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        assert isinstance(build_vision_client(), GeminiVisionClient)
    finally:
        get_settings.cache_clear()


def test_routing_fake_and_no_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_PROVIDER", "fake")
    monkeypatch.setenv("VISION_API_KEY", "AIza_k")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        assert isinstance(build_vision_client(), FakeVisionClient)
    finally:
        get_settings.cache_clear()

    monkeypatch.setenv("VISION_PROVIDER", "")
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "")
    get_settings.cache_clear()
    try:
        assert isinstance(build_vision_client(), FakeVisionClient)
    finally:
        get_settings.cache_clear()
