"""Lenient JSON extraction — the Phase A "Expecting value: char 0" failure class.

Reasoning models wrap payloads in <think> blocks; proxies prepend prose or
wrap in fences. The manual sandbox workflow's extract_json handles all of it;
Phase A must never die on char 0 again.
"""

from __future__ import annotations

import json

import pytest
from director_os.agent_runtime.llm import extract_json_payload


def test_plain_object():
    assert extract_json_payload('{"a": 1}') == {"a": 1}


def test_think_wrapped_payload():
    text = '<think>I should reason about the plan first...</think>{"title": "x", "acts": []}'
    assert extract_json_payload(text)["title"] == "x"


def test_code_fence_and_prose_wrapped():
    text = "Here is the plan:\n```json\n" + json.dumps({"ok": True}) + "\n```\nHope that helps!"
    assert extract_json_payload(text) == {"ok": True}


def test_array_expect():
    text = "```json\n" + json.dumps([{"id": 1, "clean_text": "10-4"}]) + "\n```"
    assert extract_json_payload(text, expect="array")[0]["id"] == 1


def test_empty_response_raises_with_quoted_head():
    with pytest.raises(ValueError, match="empty response"):
        extract_json_payload("")
    with pytest.raises(ValueError, match="502"):
        extract_json_payload("<html>502 Bad Gateway</html>")


def test_parse_macro_plan_accepts_think_wrapped_plan():
    from director_os.agent_runtime.director_session import MacroPlan, PlanAct, parse_macro_plan

    plan = MacroPlan(
        acts=[PlanAct(name="Cold Open", blocks=[{"type": "raw", "source_start_s": 0, "dur_s": 4}])]
    )
    raw = "<think>drafting</think>\n```json\n" + plan.model_dump_json() + "\n```"
    parsed = parse_macro_plan(raw)
    assert parsed.acts[0].name == "Cold Open"
