"""CR-007 regression: concurrent checkpoints must never duplicate a version.

Two dispatches racing SELECT-max/INSERT used to both claim the same
version_num — the rollback chain and the export gate's version comparison
then lie. The (project_id, version_num) unique constraint + the store's
retry-on-conflict turn the race into an orderly contention.
"""

from __future__ import annotations

import threading

import pytest
from director_os.agent_runtime.checkpoint import CheckpointStore
from director_os.core.db import TimelineVersionRow, init_models, make_engine, make_session_factory
from director_os.otio_engine.timeline import (
    TimelineSpec,
    TrackSpec,
    build_timeline,
)
from sqlalchemy import select


@pytest.fixture()
def store(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/race.db")
    init_models(engine)
    return CheckpointStore(make_session_factory(engine)), make_session_factory(engine)


def _blank():
    """A tiny valid timeline — checkpoint payload shape is what matters."""
    return build_timeline(
        TimelineSpec(
            name="race_case",
            fps=30,
            width=640,
            height=360,
            tracks=[TrackSpec(name="V1_master", items=[])],
        )
    )


def test_concurrent_checkpoints_get_unique_versions(store):
    ops_store, sessions_factory = store
    versions: list[int] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(n_writes: int) -> None:
        try:
            for _ in range(n_writes):
                v = ops_store.checkpoint(_blank(), "proj-race", note="race")
                with lock:
                    versions.append(v)
        except Exception as e:  # noqa: BLE001 — collected below
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(10,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == []
    assert len(versions) == 40
    # strictly unique AND gap-free 1..40 — every write won exactly one slot
    assert sorted(versions) == list(range(1, 41))

    with sessions_factory() as s:
        rows = (
            s.execute(
                select(TimelineVersionRow.version_num).where(
                    TimelineVersionRow.project_id == "proj-race"
                )
            )
            .scalars()
            .all()
        )
    assert sorted(rows) == list(range(1, 41))
