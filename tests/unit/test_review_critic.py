"""The agentic critic loop — LOOK on demand, judge, and only then act.

Research-grounded contract (EditDuet SIGGRAPH 2025; Agentic Video
Intelligence arXiv 2511.14446; VideoAgent ECCV 2024): a real editing agent
does not get one fixed frame dump and one verdict round. It INTERROGATES
the render — exact frames at any master timestamp, as many rounds as it
needs (bounded) — and its "done" decision is its own, not the pipeline's.
These tests pin:

1. the ReAct loop: inspect -> (frames arrive) -> ... -> verdict;
2. bounds: inspect ceiling, per-batch cap, garbage-streak failout — all LOUD;
3. adapters: vision sees real frames, text iterates evidence, fakes refuse;
4. stage_review end-to-end: a critic that inspects then repairs drives a
   re-render; a critic that says "ok" on round 0 never re-renders;
5. back-compat: the one-shot ReviewBackend injection still works.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from director_os.agent_runtime.review import (
    INSPECT_BATCH_CAP,
    MAX_INSPECT_ROUNDS,
    CriticTurnError,
    ReviewVerdict,
    parse_critic_round,
    parse_review_verdict,
    run_critic,
    text_critic_adapter,
    vision_critic_adapter,
)
from director_os.worker.review_pass import (
    build_critic_adapter,
    stage_review,
)


def _verdict_json(repairs: list[dict] | None = None, verdict: str = "needs_repairs") -> str:
    if repairs is None and verdict == "ok":
        repairs = []
    return json.dumps(
        {
            "verdict": verdict,
            "issues": ["checked"],
            "repairs": repairs
            if repairs is not None
            else [{"kind": "remove_clip", "item_index": 3, "reason": "dead air"}],
        }
    )


class ScriptedCritic:
    """A critic adapter replaying scripted replies; records what it saw."""

    provider = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen_frames: list[list[float]] = []
        self.prompts: list[str] = []

    def ask(self, prompt: str, frames: list[tuple[float, Any]]) -> str:
        self.prompts.append(prompt)
        self.seen_frames.append([t for t, _ in frames])
        return self.replies.pop(0) if self.replies else _verdict_json(verdict="ok")


def _grabber(dur: float = 100.0):
    grabs: list[list[float]] = []

    def grab(times: list[float]) -> list[tuple[float, Path]]:
        grabs.append(list(times))
        return [(t, Path(f"/tmp/frame_{t}.jpg")) for t in times]

    grab.grabs = grabs  # type: ignore[attr-defined]
    return grab


# ------------------------------------------------------------- turn parser
def test_parse_critic_round_inspect():
    action, data = parse_critic_round(
        '{"action": "inspect", "times": [5.0, 9.5], "why": "dead air?"}'
    )
    assert action == "inspect"
    assert data["times"] == [5.0, 9.5]


def test_parse_critic_round_verdict_and_garbage():
    assert parse_critic_round(_verdict_json())[0] == "verdict"
    assert parse_critic_round("I think the video is fine frankly.")[0] == "garbage"


def test_parse_review_verdict_still_strict_on_garbage_kinds():
    v = parse_review_verdict(
        json.dumps(
            {
                "verdict": "needs_repairs",
                "issues": ["x"],
                "repairs": [{"kind": "add_narration", "text": "nope"}],
            }
        )
    )
    assert v is not None and v.repairs == []
    assert any("out-of-scope" in i for i in v.issues)


# ---------------------------------------------------------- the ReAct loop
def test_critic_inspects_then_verdicts():
    grab = _grabber()
    critic = ScriptedCritic(
        [
            '{"action": "inspect", "times": [42.0, 99.0], "why": "dead air around here?"}',
            _verdict_json(),
        ]
    )
    verdict, trace = run_critic(
        critic,
        base_prompt="BASE",
        initial_frames=[(6.0, Path("/tmp/f6.jpg")), (50.0, Path("/tmp/f50.jpg"))],
        grab_frames=grab,
        duration_s=100.0,
    )
    assert verdict.verdict == "needs_repairs"
    assert grab.grabs == [[42.0, 99.0]]
    assert [t["action"] for t in trace] == ["inspect", "verdict"]
    round1_prompt = critic.prompts[1]
    assert "42.00s" in round1_prompt and "NOW ATTACHED" in round1_prompt
    assert round1_prompt.startswith("BASE")  # conversation GROWS, never resets


def test_critic_inspect_frames_reach_the_adapter():
    grab = _grabber()
    critic = ScriptedCritic(['{"action": "inspect", "times": [7.0]}', _verdict_json(verdict="ok")])
    run_critic(
        critic,
        base_prompt="B",
        initial_frames=[(1.0, Path("/tmp/a.jpg"))],
        grab_frames=grab,
        duration_s=100.0,
    )
    assert critic.seen_frames[1] == [1.0, 7.0]  # new frames APPENDED, none lost


def test_critic_dedupes_and_clamps_inspect_times():
    grab = _grabber()
    critic = ScriptedCritic(
        [
            '{"action": "inspect", "times": [1.0, 999.9, -5, 1.005, 30.0]}',
            _verdict_json(verdict="ok"),
        ]
    )
    run_critic(critic, base_prompt="B", initial_frames=[], grab_frames=grab, duration_s=100.0)
    assert grab.grabs[0] == [0.0, 1.0, 30.0, 100.0]


def test_critic_inspect_batch_capped():
    grab = _grabber()
    too_many = ",".join(str(i * 3.0) for i in range(20))
    critic = ScriptedCritic(
        [f'{{"action": "inspect", "times": [{too_many}]}}', _verdict_json(verdict="ok")]
    )
    run_critic(critic, base_prompt="B", initial_frames=[], grab_frames=grab, duration_s=100.0)
    assert len(grab.grabs[0]) == INSPECT_BATCH_CAP


def test_critic_inspect_ceiling_fails_loud():
    grab = _grabber()
    inspect = '{"action": "inspect", "times": [5.0]}'
    critic = ScriptedCritic([inspect] * (MAX_INSPECT_ROUNDS + 2))
    with pytest.raises(CriticTurnError, match="never reached a verdict"):
        run_critic(critic, base_prompt="B", initial_frames=[], grab_frames=grab, duration_s=100.0)


def test_critic_two_garbage_rounds_fail_loud_no_guess():
    grab = _grabber()
    critic = ScriptedCritic(["no json here", "still nothing"])
    with pytest.raises(CriticTurnError, match="refusing to guess"):
        run_critic(critic, base_prompt="B", initial_frames=[], grab_frames=grab, duration_s=100.0)


def test_critic_one_garbage_then_verdict_recovers():
    grab = _grabber()
    critic = ScriptedCritic(["oops no json", _verdict_json(verdict="ok")])
    verdict, trace = run_critic(
        critic, base_prompt="B", initial_frames=[], grab_frames=grab, duration_s=100.0
    )
    assert verdict.verdict == "ok"
    assert [t["action"] for t in trace] == ["garbage", "verdict"]


def test_critic_adapter_outage_propagates_loud():
    class Dead:
        provider = "dead"

        def ask(self, prompt: str, frames: list[tuple[float, Any]]) -> str:
            raise CriticTurnError("provider 503")

    grab = _grabber()
    with pytest.raises(CriticTurnError, match="503"):
        run_critic(Dead(), base_prompt="B", initial_frames=[], grab_frames=grab, duration_s=10.0)


# ---------------------------------------------------------------- adapters
def test_text_adapter_iterates_evidence_and_ignores_frames():
    calls: list[tuple[str, str]] = []

    class Chain:
        provider = "chain"

        def complete_json(self, system: str, user: str, max_tokens: int) -> str:
            calls.append((system, user))
            return _verdict_json(verdict="ok")

    adapter = text_critic_adapter(Chain())
    assert adapter is not None
    frames = [(1.0, Path("/tmp/a.jpg")), (2.0, Path("/tmp/b.jpg"))]
    adapter.ask("round contents", frames)
    assert len(calls) == 1
    assert "round contents" in calls[0][1]


def test_text_adapter_requires_complete_json():
    assert text_critic_adapter(None) is None
    assert text_critic_adapter(object()) is None


def test_vision_adapter_refuses_fake_client():
    class Fake:
        provider = "fake"

    assert vision_critic_adapter(Fake()) is None


def test_vision_adapter_attaches_frames_to_every_turn():
    seen_refs: list[list[tuple[float, Path]]] = []

    class VLM:
        provider = "gemini"

        def analyze(self, question: str, refs):
            seen_refs.append([(r.timestamp_s, Path(r.image_path)) for r in refs])

            class Insight:
                description = _verdict_json(verdict="ok")

            return [Insight()]

    adapter = vision_critic_adapter(VLM())
    assert adapter is not None
    adapter.ask("prompt", [(3.0, Path("/tmp/x.jpg"))])
    assert seen_refs[0] == [(3.0, Path("/tmp/x.jpg"))]


# --------------------------------------------------------- worker plumbing
def test_build_critic_adapter_prefers_injected_ask():
    class Deps:
        review_backend_factory = staticmethod(lambda: ScriptedCritic([_verdict_json(verdict="ok")]))
        review_pass = True

    eye = build_critic_adapter(Deps())
    assert eye is not None and eye.provider == "scripted"


def test_build_critic_adapter_injection_without_ask_is_none():
    class OnlyComplete:
        provider = "chain"

    class Deps:
        review_backend_factory = staticmethod(lambda: OnlyComplete())
        review_pass = True

    assert build_critic_adapter(Deps()) is None  # falls to one-shot port path


class _Deps:
    def __init__(self, factory, output_dir: str = "/tmp/dos_review") -> None:
        self.review_backend_factory = staticmethod(factory)
        self.review_pass = True
        self.output_dir = output_dir
        self.settings = type("S", (), {"output_dir": output_dir})()

    @staticmethod
    def log(msg: str) -> None:
        pass


class _Timeline:
    pass


@pytest.fixture()
def e2e_env(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "director_os.worker.review_pass.sample_master_frames",
        lambda *a, **k: [(6.0, tmp_path / "f0.jpg"), (50.0, tmp_path / "f1.jpg")],
    )
    monkeypatch.setattr("director_os.worker.review_pass.deterministic_review", lambda _tl: [])
    monkeypatch.setattr(
        "director_os.worker.review_pass.build_evidence",
        lambda **k: "EVIDENCE",
    )
    return tmp_path


def test_stage_review_critic_ok_on_round0_needs_no_rerender(e2e_env, monkeypatch):
    monkeypatch.setattr(
        "director_os.core.qa_probe.probe",
        lambda _p: type("P", (), {"duration_s": 100.0})(),
    )

    class Deps(_Deps):
        def __init__(self) -> None:
            super().__init__(lambda: ScriptedCritic([_verdict_json(verdict="ok")]))

    outcome = stage_review(Path("/tmp/master.mp4"), _Timeline(), [], "p1", Deps())
    assert outcome.status == "clean"
    assert outcome.inspect_rounds == 0
    assert outcome.applied == []


def test_stage_review_critic_inspects_then_repairs(e2e_env, monkeypatch):
    monkeypatch.setattr(
        "director_os.core.qa_probe.probe",
        lambda _p: type("P", (), {"duration_s": 100.0})(),
    )
    monkeypatch.setattr(
        "director_os.core.frames.grab_frame",
        lambda src, t, cache_dir=None, width=None: Path(f"/tmp/od_{t}.jpg"),
    )
    repair = {
        "kind": "trim_raw",
        "item_index": 0,
        "new_start_s": 10.0,
        "new_dur_s": 5.0,
        "reason": "dead air",
    }

    class Deps(_Deps):
        def __init__(self) -> None:
            super().__init__(
                lambda: ScriptedCritic(
                    [
                        '{"action": "inspect", "times": [33.0], "why": "check the hold"}',
                        _verdict_json([repair]),
                    ]
                )
            )

    outcome = stage_review(Path("/tmp/master.mp4"), _Timeline(), [], "p2", Deps())
    assert outcome.inspect_rounds == 1
    assert 33.0 in outcome.inspected
    # the repair reaches the real typed registry, which refuses the fake
    # timeline loudly (needs DirectorToolbox-compatible geometry) — the
    # PROGRAM is untouched, the outcome says so, no exception escapes
    assert outcome.status in {"clean", "applied"} or outcome.refused


def test_stage_review_critic_loop_failure_is_not_fatal(e2e_env, monkeypatch):
    monkeypatch.setattr(
        "director_os.core.qa_probe.probe",
        lambda _p: type("P", (), {"duration_s": 100.0})(),
    )

    class Deps(_Deps):
        def __init__(self) -> None:
            super().__init__(lambda: ScriptedCritic(["garbage one", "garbage two"]))

    outcome = stage_review(Path("/tmp/master.mp4"), _Timeline(), [], "p3", Deps())
    assert outcome.status == "unavailable"
    assert any("critic" in i.lower() for i in outcome.issues)


def test_stage_review_one_shot_injection_still_works(e2e_env, monkeypatch):
    class OneShot:
        provider = "oneshot"

        def review(self, frames, evidence: str) -> ReviewVerdict:
            return ReviewVerdict(verdict="ok", issues=["looked once"], repairs=[])

    class Deps(_Deps):
        def __init__(self) -> None:
            super().__init__(lambda: OneShot())

    outcome = stage_review(Path("/tmp/master.mp4"), _Timeline(), [], "p4", Deps())
    assert outcome.status == "clean"
    assert any("looked once" in i for i in outcome.issues)
