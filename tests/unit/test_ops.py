"""Op log core — the everything-is-an-op atom + reducer fold (roadmap step 2)."""

from __future__ import annotations

import pytest
from director_os.core.ops import (
    OpCreate,
    OpRecord,
    OpsStore,
    apply_caption_overrides,
    fold_ops,
    regions_to_op_params,
    validate_op_window,
)


def _op(rev: int, kind: str = "blur", *, revoked: bool = False, **kw) -> OpRecord:
    return OpRecord(
        rev=rev,
        id=kw.pop("id", f"op-{rev}"),
        project_id=kw.pop("project_id", "p1"),
        kind=kind,
        params=kw.pop("params", {"x": 10, "y": 20, "w": 30, "h": 40, "sigma": 24}),
        origin=kw.pop("origin", "agent"),
        revoked=revoked,
        **kw,
    )


# ---------------------------------------------------------------- fold laws
def test_fold_blur_op_becomes_effect_box():
    fold = fold_ops([_op(1)])
    assert fold.active_ops == 1
    assert len(fold.effects) == 1
    fx = fold.effects[0]
    assert fx["x"] == 10 and fx["w"] == 30 and fx["sigma"] == 24
    assert fx["origin"] == "agent"
    assert fx["created_by"] == "op:agent"
    assert fx["op_id"] == "op-1"


def test_fold_revoked_ops_are_dropped_not_deleted():
    fold = fold_ops([_op(1), _op(2, revoked=True), _op(3)])
    assert fold.active_ops == 2
    assert fold.revoked_ops == 1
    assert len(fold.effects) == 2  # the revoked blur never renders


def test_fold_revoke_tombstones_are_pure_feed_events():
    rows = [
        _op(1),
        OpRecord(rev=2, id="t1", project_id="p1", kind="revoke", params={"revokes_op_id": "op-1"}),
        _op(3, id="op-3"),
    ]
    fold = fold_ops(rows)
    assert fold.active_ops == 2, "tombstones are events, not ops"
    assert fold.revoked_ops == 0
    assert [e["op_id"] for e in fold.effects] == ["op-1", "op-3"]


def test_fold_orders_by_rev_asc():
    fold = fold_ops([_op(3, id="later"), _op(1, id="earlier")])
    assert [e["op_id"] for e in fold.effects] == ["earlier", "later"]
    assert fold.rev == 3


def test_fold_caption_ops_group_by_cue_index():
    recs = [
        _op(1, "caption", params={"cue_index": 4, "text": "fixed"}),
        _op(2, "caption", params={"cue_index": 4, "speaker": "OFFICER"}),
        _op(3, "caption", params={"cue_index": 7, "start_ms": 1200}),
        _op(4, "caption", params={"note": "no index — ignored"}),
    ]
    fold = fold_ops(recs)
    assert fold.caption_overrides["4"] == {"text": "fixed", "speaker": "OFFICER"}
    assert fold.caption_overrides["7"] == {"start_ms": 1200}
    assert "3" not in fold.caption_overrides


def test_fold_other_kinds_pass_through():
    fold = fold_ops(
        [_op(1, "trim", params={"summary": "cut dead air"}, frame_start=90, frame_end=300)]
    )
    assert fold.effects == []
    assert len(fold.other) == 1
    o = fold.other[0]
    assert o["kind"] == "trim" and o["frame_start"] == 90 and "cut dead air" in o["summary"]


def test_fold_op_supersedes_legacy_region_by_id():
    legacy = [{"id": "legacy-1", "x": 1, "y": 2, "w": 3, "h": 4, "sigma": 8, "label": "old"}]
    recs = [_op(1, params={"x": 5, "y": 5, "w": 5, "h": 5, "legacy_region_id": "legacy-1"})]
    fold = fold_ops(recs, legacy)
    assert len(fold.effects) == 1, "imported-over region must not double-render"
    assert fold.effects[0]["x"] == 5
    assert fold.effects[0]["origin"] == "agent"


def test_fold_keeps_unowned_legacy_regions():
    legacy = [
        {"id": "r1", "x": 1, "y": 2, "w": 3, "h": 4, "created_by": "human_edit"},
        {"id": "r2", "x": 5, "y": 6, "w": 7, "h": 8, "created_by": "auto_sweep"},
    ]
    fold = fold_ops([], legacy)
    assert len(fold.effects) == 2
    assert fold.effects[0]["origin"] == "human"
    assert fold.effects[1]["origin"] == "legacy"


def test_fold_effect_order_ops_then_legacy():
    legacy = [{"id": "r1", "x": 1, "y": 2, "w": 3, "h": 4}]
    recs = [_op(2), _op(1)]
    fold = fold_ops(recs, legacy)
    assert [e["op_id"] for e in fold.effects[:2]] == ["op-1", "op-2"]
    assert fold.effects[-1]["id"] == "r1"


# ---------------------------------------------------------------- overrides
def test_apply_caption_overrides_copy_on_write():
    cues = [
        {"index": 0, "text": "original", "speaker": "", "start_ms": 0, "end_ms": 900},
        {"index": 1, "text": "keep me", "speaker": "NARRATOR", "start_ms": 1000, "end_ms": 2000},
    ]
    out = apply_caption_overrides(cues, {"0": {"text": "corrected", "speaker": "OFFICER"}})
    assert out[0]["text"] == "corrected" and out[0]["speaker"] == "OFFICER"
    assert out[1] == cues[1], "untouched cues pass through unchanged"
    assert cues[0]["text"] == "original", "base store is read-only"


def test_apply_caption_overrides_ignores_unknown_index():
    cues = [{"text": "a", "start_ms": 0, "end_ms": 100}]
    out = apply_caption_overrides(cues, {"9": {"text": "ghost"}, "x": {"text": "nan"}})
    assert out[0]["text"] == "a"


# ---------------------------------------------------------------- windows
def test_validate_window_both_none_is_legal():
    validate_op_window(None, None, 100.0, 30.0)


def test_validate_window_mismatch_raises():
    with pytest.raises(ValueError, match="together"):
        validate_op_window(10, None, 100.0, 30.0)


def test_validate_window_inverted_raises():
    with pytest.raises(ValueError, match="< frame_start"):
        validate_op_window(50, 10, 100.0, 30.0)


def test_validate_window_beyond_tape_raises():
    with pytest.raises(ValueError, match="beyond tape"):
        validate_op_window(3001, 3100, 100.0, 30.0)  # frame 3000 is the last


def test_validate_window_end_beyond_tape_raises():
    # the END bound lies just as loudly as the start — a window past the tape
    # would bake coverage (or an effect) into frames that do not exist
    with pytest.raises(ValueError, match="frame_end.*beyond tape"):
        validate_op_window(0, 3001, 100.0, 30.0)


def test_validate_window_edge_ok():
    validate_op_window(0, 3000, 100.0, 30.0)


# ---------------------------------------------------------------- store
@pytest.fixture()
def store(tmp_path):
    from director_os.core.db import init_models, make_engine, make_session_factory

    engine = make_engine(f"sqlite:///{tmp_path}/ops.db")
    init_models(engine)
    return OpsStore(make_session_factory(engine), "p1")


def test_store_create_list_revoke_roundtrip(store):
    rec = store.create(
        OpCreate(kind="blur", params={"x": 1}, origin="agent", evidence={"frame": 42})
    )
    assert rec.rev == 1
    assert rec.evidence == {"frame": 42}
    listed = store.list_ops()
    assert len(listed) == 1
    revoked = store.revoke(rec.id)
    assert revoked.revoked is True
    assert store.list_ops() == []  # active view hides it (and the tombstone)
    assert len(store.list_ops(include_revoked=True)) == 1  # audit trail keeps the row
    assert store.max_rev() == 2  # the tombstone advanced the feed cursor


def test_store_revoke_errors(store):
    rec = store.create(OpCreate(kind="note"))
    with pytest.raises(LookupError):
        store.revoke("nope")
    store.revoke(rec.id)
    with pytest.raises(ValueError, match="already revoked"):
        store.revoke(rec.id)


def test_store_events_after_cursor(store):
    r1 = store.create(OpCreate(kind="blur"))
    store.create(OpCreate(kind="caption", params={"cue_index": 0, "text": "x"}))
    store.revoke(r1.id)
    events = store.events_after(0)
    assert [e.rev for e in events] == [1, 2, 3], "revoke rides the feed as a tombstone"
    assert events[2].kind == "revoke" and events[2].params["revokes_op_id"] == r1.id
    assert store.events_after(3) == []


def test_store_get(store):
    rec = store.create(OpCreate(kind="zoom", params={"keyframes": [[1.0, 1.15]]}))
    got = store.get(rec.id)
    assert got is not None and got.kind == "zoom"
    assert store.get("missing") is None


def test_store_scoped_to_project(tmp_path):
    from director_os.core.db import init_models, make_engine, make_session_factory

    engine = make_engine(f"sqlite:///{tmp_path}/scoped.db")
    init_models(engine)
    factory = make_session_factory(engine)
    a = OpsStore(factory, "pa")
    b = OpsStore(factory, "pb")
    a.create(OpCreate(kind="blur"))
    b.create(OpCreate(kind="note"))
    assert [r.project_id for r in a.list_ops()] == ["pa"]
    assert len(b.list_ops()) == 1


def test_regions_to_op_params_marks_legacy():
    params = regions_to_op_params(
        {"id": "r9", "x": 1, "y": 2, "w": 3, "h": 4, "sigma": 12, "label": "face"}
    )
    assert params["legacy_region_id"] == "r9"
    assert params["sigma"] == 12 and params["label"] == "face"
