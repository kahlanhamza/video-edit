"""Chat agent v2 — the orchestrator turn loop (roadmap step 5).

Scripted-LLM tests of the house laws: perceive-first (context pack carries
the op digest + tool catalog), fail-closed-with-feedback (unknown tool becomes
a result the next round sees), maker-checker (critic runs + ONE repair round),
never-an-empty-reply, checkpoint-first on a REAL OTIO timeline.
"""

from __future__ import annotations

import json

from director_os.agent_runtime.agents import PlannerExecutor
from director_os.agent_runtime.chat_v2 import (
    ChatTurnResult,
    _extract_json,
    build_context_pack,
    run_chat_turn,
)
from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.agent_runtime.ops_tools import register_ops_tools
from director_os.agent_runtime.registry import ToolRegistry
from director_os.core.db import init_models, make_engine, make_session_factory
from director_os.core.ops import OpCreate, OpsStore
from director_os.otio_engine.contract import RawClipMeta
from director_os.otio_engine.timeline import (
    MASTER_TRACK,
    ClipSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)
from pydantic import BaseModel


class ScriptedChatLLM:
    """complete_json-shaped scripted backend: pops one JSON response per call."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
        self.prompts.append(user)
        return json.dumps(self._responses.pop(0))


class _Empty(BaseModel):
    pass


def _timeline():
    """A real OTIO master (checkpoint serialization needs a serializable timeline)."""
    return build_timeline(
        TimelineSpec(
            name="chat_case",
            fps=30,
            width=1920,
            height=1080,
            tracks=[
                TrackSpec(
                    name=MASTER_TRACK,
                    items=[
                        ClipSpec(
                            name="raw_1",
                            source_start_s=0.0,
                            dur_s=40.0,
                            meta=RawClipMeta(asset_id="a1"),
                        ),
                    ],
                )
            ],
        )
    )


def _env(tmp_path, name: str):
    engine = make_engine(f"sqlite:///{tmp_path}/{name}.db")
    init_models(engine)
    factory = make_session_factory(engine)
    store = OpsStore(factory, "p1")
    reg = ToolRegistry()
    register_ops_tools(reg, store, "p1", agent_name="test")
    executor = PlannerExecutor(reg, CheckpointStore(factory), "p1", agent="chat_agent")
    tl = _timeline()
    return store, reg, executor, tl


def test_extract_json_balanced_and_fenced():
    assert _extract_json('{"a": {"b": "}"}}') == {"a": {"b": "}"}}
    assert _extract_json('```json\n{"x": 1}\n```') == {"x": 1}
    assert _extract_json('noise {"y": 2} tail') == {"y": 2}
    assert _extract_json("no json here") is None


def test_context_pack_carries_op_digest_and_catalog(tmp_path):
    store, reg, _, _ = _env(tmp_path, "pack")
    store.create(
        OpCreate(
            kind="blur",
            params={"x": 1, "y": 2, "w": 3, "h": 4, "mode": "tracked"},
            evidence={"frames": [1]},
        )
    )
    pack = build_context_pack(store, reg, timeline_summary="master: 40s")
    assert "OP LOG: rev" in pack
    assert "tracked" in pack  # the op's mode is visible
    assert "ops.list" in pack  # the catalog lists real tools
    assert "ops.propose" in pack
    assert "master: 40s" in pack


def test_context_pack_survives_a_broken_store(tmp_path):
    store, reg, _, _ = _env(tmp_path, "broken")
    reg.register("ops.list", "Read the op log", _Empty, lambda **kw: {})

    class _Broken:
        def fold(self):
            raise RuntimeError("db closed")

    pack = build_context_pack(_Broken(), reg, timeline_summary="")
    assert "OP LOG: unavailable" in pack  # context, not a crash
    assert "ops.list" in pack


def test_turn_executes_calls_and_replies(tmp_path):
    store, reg, executor, tl = _env(tmp_path, "turn")
    llm = ScriptedChatLLM(
        [
            {
                "reply": None,
                "done": False,
                "calls": [
                    {
                        "tool": "ops.propose",
                        "args": {
                            "kind": "blur",
                            "params": {"x": 4, "y": 4, "w": 40, "h": 40, "sigma": 12},
                            "evidence": {"frame": 12},
                        },
                        "note": "watermark top-left",
                    }
                ],
            },
            {"reply": "Blurred the watermark (op created, undoable).", "calls": [], "done": True},
        ]
    )
    result = run_chat_turn(
        llm,
        reg,
        executor,
        store,
        user_message="blur the watermark",
        context_pack="OP LOG: rev 0",
        timeline=tl,
    )
    assert isinstance(result, ChatTurnResult)
    assert result.rounds == 2
    assert len(result.calls) == 1 and result.calls[0]["ok"] is True
    assert "op" in result.reply.lower()
    assert result.critic is None  # no qa.self_audit registered here
    assert store.list_ops() and store.list_ops()[0].kind == "blur"


def test_unknown_tool_is_feedback_not_silence(tmp_path):
    store, reg, executor, tl = _env(tmp_path, "unk")
    llm = ScriptedChatLLM(
        [
            {"reply": None, "calls": [{"tool": "ops.blurt", "args": {}}]},
            {
                "reply": "That tool does not exist; ops.propose is the edit surface.",
                "calls": [],
                "done": True,
            },
        ]
    )
    result = run_chat_turn(
        llm, reg, executor, store, user_message="blur", context_pack="OP LOG: rev 0", timeline=tl
    )
    assert result.calls[0]["ok"] is False
    assert "unknown tool" in result.calls[0]["detail"]
    assert result.reply  # never empty


def test_failed_step_reported_and_reply_still_lands(tmp_path):
    store, reg, executor, tl = _env(tmp_path, "fail")
    llm = ScriptedChatLLM(
        [
            {
                "reply": None,
                "calls": [{"tool": "ops.propose", "args": {"kind": "blur"}}],  # no evidence
            },
            {
                "reply": "The op was rejected: evidence is required. Please name the frame.",
                "calls": [],
                "done": True,
            },
        ]
    )
    result = run_chat_turn(
        llm, reg, executor, store, user_message="blur", context_pack="OP LOG", timeline=tl
    )
    assert result.calls[0]["ok"] is False
    assert "evidence" in result.calls[0]["detail"]
    assert result.reply


def test_critic_gate_and_one_repair_round(tmp_path):
    store, reg, executor, tl = _env(tmp_path, "critic")
    verdicts = [{"ok": False, "failures": ["blur box off-target"]}, {"ok": True}]
    reg.register("qa.self_audit", "deterministic critic", _Empty, lambda **kw: verdicts.pop(0))

    llm = ScriptedChatLLM(
        [
            {
                "reply": "Blur proposed.",
                "calls": [
                    {
                        "tool": "ops.propose",
                        "args": {
                            "kind": "blur",
                            "params": {"x": 0, "y": 0, "w": 10, "h": 10},
                            "evidence": {"frame": 3},
                        },
                    }
                ],
                "done": True,
            },
            {
                "reply": "Repaired and verified: the critic now passes.",
                "calls": [],
                "done": True,
            },
        ]
    )
    result = run_chat_turn(
        llm, reg, executor, store, user_message="blur something", context_pack="OP LOG", timeline=tl
    )
    assert result.critic is not None and result.critic["ok"] is True
    assert "Repaired" in result.reply
    assert len(result.calls) == 1  # the repair round only answered


def test_no_reply_gets_deterministic_fallback(tmp_path):
    store, reg, executor, tl = _env(tmp_path, "fb")
    llm = ScriptedChatLLM(
        [
            {"reply": None, "calls": [], "done": False},  # round 1: nothing
            {"reply": None, "calls": [], "done": True},  # round 2: still nothing
        ]
    )
    result = run_chat_turn(
        llm, reg, executor, store, user_message="?", context_pack="OP LOG", timeline=tl
    )
    assert result.reply == "I read the timeline but found nothing to change for that request."


def test_round_cap_is_respected(tmp_path):
    """The model keeps proposing calls — the loop still stops (cap law)."""
    store, reg, executor, tl = _env(tmp_path, "cap")
    endless = {"reply": None, "calls": [{"tool": "ops.list", "args": {}}], "done": False}
    llm = ScriptedChatLLM([dict(endless) for _ in range(6)])
    result = run_chat_turn(
        llm,
        reg,
        executor,
        store,
        user_message="loop",
        context_pack="OP LOG",
        timeline=tl,
        max_tool_rounds=3,
    )
    assert result.rounds == 3
    assert len(llm.prompts) == 3
    assert result.reply  # fallback, never empty
