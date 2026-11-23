"""Phase A provider resilience — the planning stage must survive the flaky
LLM layer that real operators hit (empty 200s, gateway 5xx pages, dead
sockets) instead of cascading into a pipeline failure with no reason.
"""

from __future__ import annotations

import pytest
from director_os.agent_runtime.director_session import DirectorSession
from director_os.agent_runtime.editing_tools import DirectorToolbox, build_director_registry
from director_os.agent_runtime.llm import LLMBackendError


def _scaffold():
    from director_os.otio_engine.contract import RawClipMeta
    from director_os.otio_engine.timeline import (
        ClipSpec,
        GapSpec,
        TimelineSpec,
        TrackSpec,
        build_timeline,
    )

    return build_timeline(
        TimelineSpec(
            name="scaffold",
            fps=30,
            width=320,
            height=240,
            tracks=[
                TrackSpec(
                    name="V1_master",
                    items=[
                        ClipSpec(
                            name="raw_scaffold",
                            source_start_s=0.0,
                            dur_s=2627.0,
                            meta=RawClipMeta(asset_id="asset_x"),
                        )
                    ],
                ),
                TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=2627.0)]),
            ],
        )
    )


class FlakyChain:
    """ScriptedChain's evil twin: fails N times, then answers."""

    provider = "flaky"
    model = "flaky:plan"

    def __init__(self, failures: list[str | Exception], then: str):
        self._failures = list(failures)
        self._then = then
        self.calls = 0

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
        i = self.calls
        self.calls += 1
        if i < len(self._failures):
            f = self._failures[i]
            if isinstance(f, Exception):
                raise f
            return f
        return self._then

    def propose(self, state_summary: str, role: str):
        return []

    def record_results(self, steps, results) -> None:
        pass

    def reset_history(self) -> None:
        pass


def _session(backend, tmp_path):
    from director_os.agent_runtime.checkpoint import CheckpointStore
    from director_os.core.db import init_models, make_engine, make_session_factory

    engine = make_engine(f"sqlite:///{tmp_path}/s.db")
    init_models(engine)
    store = CheckpointStore(make_session_factory(engine))
    return DirectorSession(
        build_director_registry(DirectorToolbox(_scaffold(), "p1")),
        store,
        "p1",
        backend,
        source_duration_s=2627.0,
        phase_a_retries=3,
        quote_refine=False,
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Backoff is correct behavior; sleeping through it in tests is not."""
    import director_os.agent_runtime.director_session as ds

    monkeypatch.setattr(ds.time, "sleep", lambda s: None)


def test_provider_error_is_retried_not_fatal(tmp_path):
    """A 502/empty-body provider error on attempt 1 recovers on attempt 2."""
    from test_director_session import _good_plan_json

    backend = FlakyChain(
        [LLMBackendError("openai-compat 502: upstream unavailable")], _good_plan_json()
    )
    session = _session(backend, tmp_path)
    plan, err = session.plan_macro([], None)
    assert plan is not None, err
    assert backend.calls == 2, "exactly one retry"


def test_garbage_body_names_itself_in_the_error(tmp_path):
    """An HTML error page or empty 200 surfaces its head in last_err — the
    operator log must say WHY the parse failed, not just char 0."""
    backend = FlakyChain(
        ["", "not json", ""],  # empty 200, prose, empty 200 — then the gateway page
        "<html>502 Bad Gateway</html>",
    )
    session = _session(backend, tmp_path)
    plan, err = session.plan_macro([], None)
    assert plan is None
    assert backend.calls == 4  # every attempt was spent
    assert "got: '<html>502 Bad Gateway</html>'" in err  # the LAST body is quoted


def test_persistent_provider_error_falls_back_cleanly(tmp_path):
    """Every attempt raises -> (None, provider error) — planning falls back to
    the deterministic cut, the pipeline does not die at planning."""
    backend = FlakyChain([LLMBackendError("connection reset by peer")] * 4, "")
    session = _session(backend, tmp_path)
    plan, err = session.plan_macro([], None)
    assert plan is None
    assert "provider error" in err
    assert backend.calls == 4  # phase_a_retries(3) + 1
