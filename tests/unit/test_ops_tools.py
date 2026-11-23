"""Agent op tools — ops.list / ops.propose / ops.revoke via the typed registry."""

from __future__ import annotations

import pytest
from director_os.agent_runtime.editing_tools import ToolOperationError
from director_os.agent_runtime.ops_tools import register_ops_tools
from director_os.agent_runtime.registry import ToolRegistry
from director_os.core.db import init_models, make_engine, make_session_factory
from director_os.core.ops import OpCreate, OpsStore


@pytest.fixture()
def store(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/ops_tools.db")
    init_models(engine)
    return OpsStore(make_session_factory(engine), "p1")


@pytest.fixture()
def registry(store):
    reg = ToolRegistry()
    register_ops_tools(reg, store, "p1", agent_name="test_agent")
    return reg


def test_three_ops_tools_registered(registry):
    assert {"ops.list", "ops.propose", "ops.revoke"} <= set(registry.names())


def test_ops_list_reports_fold_digest(registry, store):
    store.create(
        OpCreate(
            kind="blur",
            params={"x": 5, "y": 5, "w": 5, "h": 5, "sigma": 10},
            origin="agent",
            evidence={"frames": [1]},
        )
    )
    store.create(OpCreate(kind="caption", params={"cue_index": 2, "text": "fixed"}))
    out = registry.dispatch("ops.list", {})
    assert out["active_ops"] == 2
    assert len(out["effects"]) == 1 and out["effects"][0]["sigma"] == 10
    assert out["caption_overrides"]["2"]["text"] == "fixed"
    assert "2 active ops" in out["detail"]


def test_ops_propose_requires_evidence(registry):
    with pytest.raises(ToolOperationError, match="evidence"):
        registry.dispatch("ops.propose", {"kind": "blur", "params": {"x": 1}})


def test_ops_propose_unknown_kind_rejected(registry):
    with pytest.raises(ToolOperationError, match="unknown op kind"):
        registry.dispatch("ops.propose", {"kind": "warp", "evidence": {"frames": [0]}})


def test_ops_propose_appends_agent_op(registry, store):
    out = registry.dispatch(
        "ops.propose",
        {
            "kind": "blur",
            "params": {"x": 1, "y": 2, "w": 3, "h": 4, "sigma": 20},
            "evidence": {"frames": [4201], "grab": "assets/frame?t=140.03"},
            "note": "face leak at the traffic stop",
        },
    )
    assert out["ok"] is True
    rec = store.get(out["op_id"])
    assert rec is not None
    assert rec.origin == "agent" and rec.session_id == "test_agent"
    assert rec.evidence["frames"] == [4201]


def test_ops_revoke_roundtrip(registry, store):
    rec = store.create(OpCreate(kind="zoom", params={"keyframes": [[1.0, 1.2]]}))
    out = registry.dispatch("ops.revoke", {"op_id": rec.id, "note": "too strong"})
    assert out["ok"] is True
    assert store.get(rec.id).revoked is True
    with pytest.raises(ToolOperationError, match="already revoked"):
        registry.dispatch("ops.revoke", {"op_id": rec.id})
    with pytest.raises(ToolOperationError, match="not found"):
        registry.dispatch("ops.revoke", {"op_id": "does-not-exist"})


# ------------------------------------------------- caption write-through law
@pytest.fixture()
def cue_store(tmp_path, monkeypatch):
    """Point OUTPUT_DIR at a tmp tree with two seeded cues."""
    from director_os.core.config import get_settings
    from director_os.subtitles.ass_builder import CaptionBeat, WordTiming
    from director_os.subtitles.store import save_cues

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    get_settings.cache_clear()
    base = tmp_path / "out" / "pipeline" / "p1"
    save_cues(
        base,
        [
            CaptionBeat(
                text="unit one contact",
                start_ms=0,
                end_ms=1800,
                speaker="OFFICER_1",
                words=[WordTiming(word="unit", start_ms=0, end_ms=1800)],
            ),
            CaptionBeat(text="roger that", start_ms=2000, end_ms=3000, speaker="OFFICER_2"),
        ],
    )
    yield base
    get_settings.cache_clear()


def test_ops_propose_caption_writes_through_to_the_cue_store(registry, store, cue_store):
    from director_os.subtitles.store import load_cues

    out = registry.dispatch(
        "ops.propose",
        {
            "kind": "caption",
            "params": {"cue_index": 1, "text": "roger, moving up"},
            "evidence": {"transcript": "00:02:00-00:02:03 words unclear"},
        },
    )
    assert out["ok"] is True and "cue store" in out["detail"]
    cues = load_cues(cue_store)
    assert cues[1].text == "roger, moving up", "the baked truth carries the agent edit"
    rec = store.get(out["op_id"])
    assert rec is not None and rec.params["text"] == "roger, moving up"
    assert rec.params["start_ms"] == 2000, "op records the final window"


def test_ops_propose_caption_retime_moves_words(registry, store, cue_store):
    from director_os.subtitles.store import load_cues

    out = registry.dispatch(
        "ops.propose",
        {
            "kind": "caption",
            "params": {"cue_index": 0, "start_ms": 500, "end_ms": 1500},
            "evidence": {"frames": [15, 16]},
        },
    )
    assert out["ok"] is True
    cues = load_cues(cue_store)
    assert cues[0].start_ms == 500 and cues[0].end_ms == 1500
    assert cues[0].words[0].start_ms >= 500, "words ride into the new window"


def test_ops_propose_caption_failures_create_no_op(registry, store, cue_store):
    from director_os.subtitles.store import load_cues

    before = store.list_ops(include_revoked=True)
    with pytest.raises(ToolOperationError, match="cue_index"):
        registry.dispatch(
            "ops.propose",
            {"kind": "caption", "params": {"text": "no cue"}, "evidence": {"frames": [0]}},
        )
    with pytest.raises(ToolOperationError, match="out of range"):
        registry.dispatch(
            "ops.propose",
            {
                "kind": "caption",
                "params": {"cue_index": 99, "text": "ghost"},
                "evidence": {"frames": [0]},
            },
        )
    with pytest.raises(ToolOperationError, match="together"):
        registry.dispatch(
            "ops.propose",
            {
                "kind": "caption",
                "params": {"cue_index": 0, "start_ms": 100},
                "evidence": {"frames": [0]},
            },
        )
    assert store.list_ops(include_revoked=True) == before, "fail-closed: no op rows"
    cues = load_cues(cue_store)
    assert cues[0].text == "unit one contact", "and no store writes"
