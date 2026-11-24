"""The live-run failure classes the REAL e2e exposed (2026-11-24, Eustis bodycam):

1. the critic's vision channel shredded the model's ACTION JSON —
   `analyze()` decodes insight arrays, so a correct ReAct reply died as
   "2 consecutive unusable rounds" before any verdict;
2. Gemini `thought` parts rode inside joined completions and broke strict
   JSON consumers (Phase A died on every retry, supervisor batches 503'd);
3. the garbage failout lost the round details — an operator log that says
   "unusable" without WHY is a maze;
4. LLM JSON with stacked values and no comma died on every strict parse
   despite retries.

Each pin: the class must stay dead.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from director_os.agent_runtime.llm import (
    GeminiRestBackend,
    extract_json_payload,
    repair_common_llm_json,
)
from director_os.agent_runtime.registry import ToolRegistry
from director_os.agent_runtime.review import (
    CriticTurnError,
    run_critic,
    vision_critic_adapter,
)
from director_os.perception.vision import FrameRef, GeminiVisionClient


def _gemini_body(text: str, thought: str | None = None) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if thought:
        parts.append({"text": thought, "thought": True})
    parts.append({"text": text})
    return {"candidates": [{"content": {"parts": parts}}]}


def make_gemini_vision(tmp_path, reply: str, thought: str | None = None, seen: dict | None = None):
    body = _gemini_body(reply, thought)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen["payload"] = json.loads(request.content.decode())
        return httpx.Response(200, json=body)

    frame = tmp_path / "f0.jpg"
    frame.write_bytes(b"\xff\xd8fake-jpeg")
    client = GeminiVisionClient(
        "v-key",
        model="gemini-3.5-flash-lite",
        transport=httpx.MockTransport(handler),
        modal_egress=False,
    )
    frames = [FrameRef(timestamp_s=10.0, image_path=frame)]
    return client, frames


def test_raw_analysis_returns_verbatim_action_json_thought_free(tmp_path):
    seen: dict = {}
    reply = '{"action": "inspect", "times": [42.5], "why": "check the holster"}'
    client, frames = make_gemini_vision(
        tmp_path, reply, thought="I should check the holster area first.", seen=seen
    )
    text = client.raw_analysis("look at the master", frames)
    assert text == reply
    assert '"thought"' not in str(seen)
    payload = seen["payload"]
    assert payload["contents"][0]["parts"][0]["text"] == "look at the master"


def test_analyze_drops_thought_parts_in_degraded_parse(tmp_path):
    reply = "I am only prose, no JSON array here."
    client, frames = make_gemini_vision(tmp_path, reply, thought="hmm let me think.")
    insights = client.analyze("look", frames)
    assert len(insights) == 1
    assert insights[0].description == reply


def test_critic_adapter_prefers_raw_channel(tmp_path):
    reply = '{"action": "inspect", "times": [7.0]}'
    client, frames = make_gemini_vision(tmp_path, reply)
    adapter = vision_critic_adapter(client)
    assert adapter is not None
    assert adapter.ask("judge this", [(f.timestamp_s, f.image_path) for f in frames]) == reply


def test_critic_adapter_structural_failure_is_loud(tmp_path):
    class Boom:
        provider = "gemini"

        def raw_analysis(self, question: str, frames: Any) -> str:
            raise AttributeError("'str' object has no attribute 'get'")

    adapter = vision_critic_adapter(Boom())
    assert adapter is not None
    with pytest.raises(CriticTurnError, match="vision adapter failed"):
        adapter.ask("judge", [])


def test_garbage_failout_names_what_it_saw():
    class Garbage:
        provider = "scripted"

        def ask(self, prompt: str, frames: Any) -> str:
            return "I looked at the video and everything seems fine."

    with pytest.raises(CriticTurnError, match="no JSON object in reply"):
        run_critic(
            Garbage(),
            base_prompt="judge",
            initial_frames=[(0.0, "f.jpg")],
            grab_frames=lambda times: [],
            duration_s=30.0,
        )


def test_run_critic_accepts_action_json_from_raw_vision_channel(tmp_path):
    reply = json.dumps({"action": "verdict", "verdict": "ok", "issues": [], "repairs": []})
    client, frames = make_gemini_vision(tmp_path, reply)
    adapter = vision_critic_adapter(client)
    assert adapter is not None
    verdict, trace = run_critic(
        adapter,
        base_prompt="judge",
        initial_frames=[(f.timestamp_s, f.image_path) for f in frames],
        grab_frames=lambda times: [],
        duration_s=30.0,
    )
    assert verdict.verdict == "ok"
    assert trace[-1]["action"] == "verdict"


def test_repair_common_llm_json_missing_comma_between_keys():
    broken = '{\n  "title": "The Disarming of Officer Oscar"\n  "acts": [\n    1\n  ]\n}'
    assert json.loads(repair_common_llm_json(broken))["acts"] == [1]


def test_repair_common_llm_json_missing_comma_in_array():
    assert json.loads(repair_common_llm_json('["a"\n"b"]')) == ["a", "b"]


def test_repair_common_llm_json_smart_quotes_and_trailing_comma():
    fixed = repair_common_llm_json("{\u201ckind\u201d: 1}")
    assert json.loads(fixed) == {"kind": 1}
    assert json.loads(repair_common_llm_json('{"kind": [1, 2,]}')) == {"kind": [1, 2]}


def test_extract_json_payload_repairs_then_validates():
    broken = '{"title": "x"\n  "acts": [{"raw": 1}]}'
    assert extract_json_payload(broken, expect="object") == {"title": "x", "acts": [{"raw": 1}]}


def test_extract_json_payload_unrepairable_raises_original_error():
    with pytest.raises(ValueError, match="Expecting value"):
        extract_json_payload('{"a": b c "x"}', expect="object")


def test_gemini_complete_json_drops_thought_parts():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_gemini_body('{"title": "x"}', thought="reasoning noise"))

    backend = GeminiRestBackend(
        "k",
        "gemini-3.5-flash",
        ToolRegistry(),
        transport=httpx.MockTransport(handler),
    )
    text = backend.complete_json("system", "user")
    assert text == '{"title": "x"}'
    assert "generateContent" in seen["url"]


def test_frame_bytes_ride_inline(tmp_path):
    seen: dict = {}
    client, frames = make_gemini_vision(tmp_path, "{}", seen=seen)
    client.raw_analysis("look", frames)
    parts = seen["payload"]["contents"][0]["parts"]
    inline = [p for p in parts if "inline_data" in p]
    assert len(inline) == 1
    assert base64.b64decode(inline[0]["inline_data"]["data"]) == b"\xff\xd8fake-jpeg"
