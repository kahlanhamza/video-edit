"""The Legal & Documentary QA Critic pass (the notebook's second LLM stage).

The first-draft plan hallucinates (an ASR misheard word becomes a narcotics
accusation) or skips high-drama beats. The critic audits facts over the FULL
transcript — and ANY failure keeps the accepted plan (the notebook's
"keeping previous plan safely" law). The critic plan must re-pass the Phase A
pre-flight; a critic that inflates acts or breaks pacing is rejected.
"""

from __future__ import annotations

from director_os.agent_runtime.director_session import (
    MacroPlan,
    PlanAct,
    critic_refine_plan,
)


def _plan():
    return MacroPlan(
        title="t",
        acts=[
            PlanAct(
                name="Cold Open",
                blocks=[
                    {
                        "type": "narration",
                        "script": "One stop on a July morning turns into a career-ending "
                        "investigation for a young officer.",
                    },
                    {
                        "type": "raw",
                        "source_start_s": 10.0,
                        "dur_s": 30.0,
                        "note": "contact",
                    },
                ],
            ),
            PlanAct(
                name="Outro",
                blocks=[
                    {
                        "type": "narration",
                        "script": "The certified breath test records the number that ends "
                        "the story, and the badge comes off.",
                    },
                ],
            ),
        ],
    )


def _transcript():
    words = []
    line = ["officer", "asks", "the", "driver", "when", "he", "got", "off", "work", "today"]
    for i, w in enumerate(line):
        words.append(
            {"text": w, "start_s": 12.0 + i * 0.3, "end_s": 12.0 + i * 0.3 + 0.25, "speaker": "1"}
        )
    return {"words": words}


class FakeBackend:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete_json(self, system, user, max_tokens=4096):
        self.calls += 1
        assert "QA Critic" in user or "Critic" in system
        return self.reply


def test_critic_hallucination_fix_is_accepted():
    revised = _plan().model_copy(deep=True)
    revised.acts[1].blocks[0].script = (
        "The certified breath test records the number that ends the story, and the "
        "badge comes off — the arrest played out exactly as the transcript shows."
    )
    backend = FakeBackend(revised.model_dump_json())
    out, note = critic_refine_plan(_plan(), _transcript(), backend, 2600.0)
    assert backend.calls == 1
    assert "transcript shows" in out.acts[1].blocks[0].script
    assert note == ""


def test_critic_garbage_keeps_the_accepted_plan():
    backend = FakeBackend("I am sorry, I cannot comply.")
    out, note = critic_refine_plan(_plan(), _transcript(), backend, 2600.0)
    assert out.model_dump() == _plan().model_dump()
    assert "kept accepted plan" in note


def test_critic_backend_failure_keeps_the_plan():
    class DeadBackend:
        provider = "dead"

        def complete_json(self, system, user, max_tokens=4096):
            raise RuntimeError("socket dead")

    out, note = critic_refine_plan(_plan(), _transcript(), DeadBackend(), 2600.0)
    assert out.model_dump() == _plan().model_dump()
    assert "kept accepted plan" in note


def test_critic_inflating_act_count_is_rejected():
    inflated = _plan().model_copy(deep=True)
    for i in range(12):
        inflated.acts.append(
            PlanAct(
                name=f"Filler {i}",
                blocks=[{"type": "raw", "source_start_s": 20.0 + i, "dur_s": 5.0}],
            )
        )
    backend = FakeBackend(inflated.model_dump_json())
    out, note = critic_refine_plan(_plan(), _transcript(), backend, 2600.0)
    assert len(out.acts) == 2
    assert "act count" in note


def test_critic_with_no_transcript_is_a_noop():
    out, note = critic_refine_plan(_plan(), None, FakeBackend("{}"), 2600.0)
    assert out.model_dump() == _plan().model_dump()
    assert note == ""


def test_critic_prompt_carries_full_transcript_and_draft():
    seen: dict[str, str] = {}

    class Recorder(FakeBackend):
        def complete_json(self, system, user, max_tokens=4096):
            seen["user"] = user
            return _plan().model_dump_json()

    critic_refine_plan(_plan(), _transcript(), Recorder(None), 2600.0)
    assert "DRAFT PLAN" in seen["user"]
    assert "FULL DIALOGUE TRANSCRIPT" in seen["user"]
    assert "HALLUCINATION" in seen["user"]
