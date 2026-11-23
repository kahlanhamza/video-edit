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
