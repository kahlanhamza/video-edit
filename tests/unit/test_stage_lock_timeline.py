"""Stage-lock timeline mechanics — hermetic unit test.

Covers the timeline (video-editor metaphor) end-to-end on a tmp_path:
lock -> edit (sig changes) -> rerun (archives to attic, history grows) ->
rollback (restores previous output, current archived) -> restore-forward.

Also covers the regression that shipped before this test existed: hardlinks
share an inode, so an owner that overwrites its output in place silently
corrupted the attic archive. The archive path must use full byte copies.

Hermetic: no network, no media, no git history (owner scripts live in
tmp_path so the git probe falls through to content hashing).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from director_os.core.stage_lock import InputSpec, StageLock, StageSpec


class Sandbox:
    """Test double: the StageLock plus the owner-regeneration helper."""

    def __init__(self, lock: StageLock, write_owner: Callable[[str], None]):
        self.lock = lock
        self.write_owner = write_owner


@pytest.fixture()
def sandbox(tmp_path: Path) -> Sandbox:
    """A sandbox graph whose owner script writes one small text file."""
    root = tmp_path
    run_dir = root / "pipeline_run"
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)

    owner = scripts_dir / "test_owner.py"
    output = run_dir / "test_output.txt"

    def write_owner(marker: str) -> None:
        owner.write_text(
            "# synthetic owner — writes one marker line\n"
            "from pathlib import Path\n"
            f"out = Path(r'{output}')\n"
            "out.parent.mkdir(parents=True, exist_ok=True)\n"
            f"out.write_text('MARKER={marker}\\n')\n"
            f"print('owner wrote {marker}')\n"
        )

    write_owner("v1")

    graph = {
        "test_stage": StageSpec(
            stage_id="test_stage",
            owner=owner,
            outputs=[output],
            inputs=[InputSpec(path=owner, produced_by=None)],
            params={"test": True},
            rules=["synthetic"],
        )
    }
    return Sandbox(StageLock(graph, root=root, run_dir=run_dir), write_owner)


def test_lock_bootstrap(sandbox: Sandbox):
    output = sandbox.lock.graph["test_stage"].outputs[0]
    output.write_text("MARKER=initial\n")
    res = sandbox.lock.lock_stage("test_stage")
    assert res.state == "locked"
    assert sandbox.lock.status_of("test_stage").state == "locked"


def test_owner_edit_changes_signature(sandbox: Sandbox):
    output = sandbox.lock.graph["test_stage"].outputs[0]
    output.write_text("MARKER=v1\n")
    sandbox.lock.lock_stage("test_stage")
    sig1 = sandbox.lock.compute_signature("test_stage")
    sandbox.write_owner("v1b")  # owner content edit
    sig2 = sandbox.lock.compute_signature("test_stage")
    assert sig1 != sig2
    assert sandbox.lock.status_of("test_stage").state == "stale"


def test_touch_does_not_flip_signature(sandbox: Sandbox):
    """mtime-only churn (sandbox recycle, scp -p) must NOT invalidate."""
    owner = sandbox.lock.graph["test_stage"].owner
    output = sandbox.lock.graph["test_stage"].outputs[0]
    output.write_text("MARKER=v1\n")
    sandbox.lock.lock_stage("test_stage")
    sig1 = sandbox.lock.compute_signature("test_stage")
    os.utime(owner, (0, 0))
    os.utime(owner, (12345, 12345))
    sig2 = sandbox.lock.compute_signature("test_stage")
    assert sig1 == sig2


def test_rerun_archives_and_records_history(sandbox: Sandbox):
    output = sandbox.lock.graph["test_stage"].outputs[0]
    output.write_text("MARKER=initial\n")
    sandbox.lock.lock_stage("test_stage")
    sig1 = sandbox.lock.compute_signature("test_stage")

    sandbox.write_owner("v2")
    res = sandbox.lock.run_stage("test_stage", force=True)
    assert res.state == "locked"
    sig2 = sandbox.lock.compute_signature("test_stage")
    assert output.read_text().strip() == "MARKER=v2"

    # v1 archived under its own signature, with ORIGINAL content
    attic_v1 = sandbox.lock.attic_root / "test_stage" / sig1 / output.name
    assert attic_v1.exists()
    assert attic_v1.read_text().strip() == "MARKER=initial"

    # history recorded
    entry = sandbox.lock.load_manifest()["test_stage"]
    hist = entry.get("history") or []
    assert len(hist) == 1
    assert hist[0]["signature"] == sig1
    assert sig2 == entry["signature"]


def test_rollback_restores_previous_content(sandbox: Sandbox):
    output = sandbox.lock.graph["test_stage"].outputs[0]
    output.write_text("MARKER=initial\n")
    sandbox.lock.lock_stage("test_stage")
    sandbox.write_owner("v2")
    sandbox.lock.run_stage("test_stage", force=True)
    assert output.read_text().strip() == "MARKER=v2"

    res = sandbox.lock.rollback_stage("test_stage")
    # outputs are v1; owner code is still v2 -> deliberately RESTORED, not stale
    assert res.state == "restored"
    assert output.read_text().strip() == "MARKER=initial"

    entry = sandbox.lock.load_manifest()["test_stage"]
    attic_v1 = sandbox.lock.attic_root / "test_stage" / entry["signature"]
    assert (attic_v1 / output.name).exists()


def test_restore_forward_jumps_to_specific_run(sandbox: Sandbox):
    output = sandbox.lock.graph["test_stage"].outputs[0]
    output.write_text("MARKER=initial\n")
    sandbox.lock.lock_stage("test_stage")
    sig1 = sandbox.lock.compute_signature("test_stage")
    sandbox.write_owner("v2")
    sandbox.lock.run_stage("test_stage", force=True)
    sig2 = sandbox.lock.compute_signature("test_stage")

    sandbox.lock.rollback_stage("test_stage")  # now on v1 outputs
    res = sandbox.lock.restore_outputs("test_stage", sig2)  # jump forward to v2
    assert res.state == "locked"  # sig2 == current owner -> genuinely locked
    assert output.read_text().strip() == "MARKER=v2"
    entry = sandbox.lock.load_manifest()["test_stage"]
    assert entry["signature"] == sig2
    assert sig1 != sig2


def test_archive_survives_inplace_overwrite_regression(sandbox: Sandbox):
    """Regression: the first archive implementation used hardlinks, which
    share an inode — an owner that truncates its output in place corrupted
    the attic copy too. The archive must be a full byte copy."""
    output = sandbox.lock.graph["test_stage"].outputs[0]
    output.write_text("MARKER=v1\n")
    sandbox.lock.lock_stage("test_stage")
    sig1 = sandbox.lock.compute_signature("test_stage")

    sandbox.write_owner("v2")
    sandbox.lock.run_stage("test_stage", force=True)

    attic_v1 = sandbox.lock.attic_root / "test_stage" / sig1 / output.name
    assert attic_v1.exists()
    # THE assertion: attic content is the ORIGINAL v1, not the overwritten v2
    assert attic_v1.read_text().strip() == "MARKER=v1"
    assert output.read_text().strip() == "MARKER=v2"
    # and the two are distinct inodes (a copy, not a shared inode)
    assert os.stat(attic_v1).st_ino != os.stat(output).st_ino


def test_status_distinguishes_restored_from_stale(sandbox: Sandbox):
    output = sandbox.lock.graph["test_stage"].outputs[0]
    output.write_text("MARKER=initial\n")
    sandbox.lock.lock_stage("test_stage")
    sandbox.write_owner("v2")
    sandbox.lock.run_stage("test_stage", force=True)
    sandbox.lock.rollback_stage("test_stage")
    # after rollback: outputs=v1, code=v2 -> "restored" (deliberate)
    assert sandbox.lock.status_of("test_stage").state == "restored"
    # further input drift keeps the restored marker (it is historical fact:
    # the outputs on disk ARE from a restored run) — but the reason string
    # still surfaces the current-input signature so the operator sees both.
    sandbox.write_owner("v3")
    st = sandbox.lock.status_of("test_stage")
    assert st.state == "restored"
    assert "current inputs would produce" in st.reason

    # a stage with input drift and NO restore marker reads "stale"
    sandbox.write_owner("v4")
    sandbox.lock.run_stage("test_stage", force=True)  # fresh run clears the marker
    assert sandbox.lock.status_of("test_stage").state == "locked"
    sandbox.write_owner("v5")
    assert sandbox.lock.status_of("test_stage").state == "stale"


def test_failed_owner_never_locks(sandbox: Sandbox):
    """A crashing owner must NOT update the manifest — a failed run can
    never be mistaken for a locked one on the next pass."""
    output = sandbox.lock.graph["test_stage"].outputs[0]
    output.write_text("MARKER=v1\n")
    sandbox.lock.lock_stage("test_stage")
    sandbox.lock.graph["test_stage"].owner.write_text("import sys; sys.exit(3)\n")
    res = sandbox.lock.run_stage("test_stage", force=True)
    assert res.state != "locked"
    entry = sandbox.lock.load_manifest().get("test_stage")
    assert entry is not None and entry.get("locked_without_rerun") is True
    # the manifest still points at the bootstrapped signature and the
    # previous outputs are untouched — nothing was faked.
    assert output.read_text().strip() == "MARKER=v1"


def test_run_from_only_touches_downstream(sandbox: Sandbox):
    """run_from executes the target + downstream stages, never upstream."""
    lk = sandbox.lock
    up_owner = lk.root / "scripts" / "up_owner.py"
    up_out = lk.root / "pipeline_run" / "up_output.txt"
    up_owner.write_text(
        "from pathlib import Path\n"
        f"out = Path(r'{up_out}')\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text('UP=v1\\n')\n"
    )
    lk.graph["up_stage"] = StageSpec(
        stage_id="up_stage",
        owner=up_owner,
        outputs=[up_out],
        inputs=[InputSpec(path=up_owner, produced_by=None)],
    )
    dn_owner = lk.root / "scripts" / "dn_owner.py"
    dn_out = lk.root / "pipeline_run" / "dn_output.txt"
    dn_owner.write_text(
        "from pathlib import Path\n"
        f"out = Path(r'{dn_out}')\n"
        f"inp = Path(r'{up_out}')\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text('DN=' + inp.read_text().strip().split('=')[1] + '\\n')\n"
    )
    lk.graph["dn_stage"] = StageSpec(
        stage_id="dn_stage",
        owner=dn_owner,
        outputs=[dn_out],
        depends_on=["up_stage"],
        inputs=[InputSpec(path=up_out, produced_by="up_stage")],
    )
    assert lk.downstream_of("up_stage") == ["dn_stage"]
    order = lk.topological_order()
    assert order.index("up_stage") < order.index("dn_stage")

    res = lk.run_from("up_stage")
    assert [r.stage_id for r in res] == ["up_stage", "dn_stage"]
    assert all(r.state == "locked" for r in res)
    assert up_out.read_text().strip() == "UP=v1"
    assert dn_out.read_text().strip() == "DN=v1"

    # downstream rerun consumes the upstream artifact but never rewrites it
    up_out.write_text("UP=v2\n")
    lk.run_from("dn_stage")
    assert up_out.read_text().strip() == "UP=v2"
    assert dn_out.read_text().strip() == "DN=v2"
