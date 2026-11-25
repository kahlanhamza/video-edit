"""Foreign narration detection + the planner's replacement law."""

from __future__ import annotations

from director_os.agent_runtime.fallback_cut import plan_fallback_blocks
from director_os.agent_runtime.foreign_media import (
    detect_narrator_speaker,
    foreign_narration_ranges,
)


def _words(*utterances: tuple[float, str]) -> dict:
    words = []
    for t, text in utterances:
        words.append({"text": text, "start_s": t, "end_s": t + len(text) * 0.05})
    return {"words": words}


# ------------------------------------------------------------- marker phrases
def test_channel_greeting_and_cta_are_foreign_narration():
    t = _words(
        (5.0, "officer says step back please"),
        (30.0, "welcome back to the channel"),
        (120.0, "do not forget to subscribe and ring the bell"),
        (300.0, "he refused to leave the store"),
    )
    ranges = foreign_narration_ranges(t)
    assert len(ranges) == 2
    assert any(a < 30.0 < b for a, b, _ in ranges)
    assert any(a < 120.0 < b for a, b, _ in ranges)


def test_video_framing_and_comment_prompts_are_foreign_narration():
    t = _words(
        (10.0, "in today's video we look at the parking lot"),
        (200.0, "let me know in the comments what you think"),
        (400.0, "the auditor started filming at the station"),
    )
    ranges = foreign_narration_ranges(t)
    assert len(ranges) == 2


def test_clean_bodycam_dialogue_has_no_foreign_narration():
    t = _words(
        (8.0, "can I help you with something today"),
        (40.0, "I am filming in public that is allowed"),
        (90.0, "you need to step behind the line please"),
        (150.0, "am I being detained or am I free to go"),
        (220.0, "you are free to go but stay out of the lot"),
    )
    assert foreign_narration_ranges(t) == []


def test_ranges_are_padded_and_merged():
    t = _words(
        (10.0, "welcome back to the channel"),
        (13.0, "let's get right into it"),
    )
    ranges = foreign_narration_ranges(t, pad_s=1.0)
    # the two marker utterances sit 3s apart -> one merged window
    assert len(ranges) == 1
    a, b, _label = ranges[0]
    assert a <= 9.0 and b >= 14.5


# --------------------------------------------------------- narrator speaker
def _dialogue_plus_narrator_segments() -> dict:
    """Officers/subject converse in short alternating turns; a narrator
    speaker runs long solo blocks over the scene audio."""
    segs: list[dict] = []
    t = 0.0
    for _window_start in (0.0, 110.0, 240.0):
        for _i in range(8):
            segs.append(
                {"start_s": t, "end_s": t + 2.4, "speaker": 0, "text": "step back behind the line"}
            )
            t += 3.0
            segs.append(
                {
                    "start_s": t,
                    "end_s": t + 2.4,
                    "speaker": 1,
                    "text": "I am standing in a public lot",
                }
            )
            t += 3.0
        t += 7.0  # gap before the narrator block
    # three long narrator monologues (diarized as speaker 9)
    for a in (60.0, 170.0, 300.0):
        segs.append(
            {
                "start_s": a,
                "end_s": a + 35.0,
                "speaker": 9,
                "text": "before we get into the footage make sure you subscribe to the channel",
            }
        )
    return {"segments": segs}


def test_narrator_speaker_is_detected_from_monologue_shape():
    t = _dialogue_plus_narrator_segments()
    assert detect_narrator_speaker(t) == "9"


def test_no_narrator_in_pure_dialogue():
    segs = []
    t = 0.0
    for _i in range(40):
        segs.append(
            {"start_s": t, "end_s": t + 2.5, "speaker": 0, "text": "sir I need you to calm down"}
        )
        t += 3.0
        segs.append({"start_s": t, "end_s": t + 2.5, "speaker": 1, "text": "I am calm officer"})
        t += 3.0
    assert detect_narrator_speaker({"segments": segs}) is None


def test_narrator_speaker_windows_are_all_excluded():
    t = _dialogue_plus_narrator_segments()
    ranges = foreign_narration_ranges(t)
    assert len(ranges) >= 3
    for a, b, label in ranges:
        assert "narrator" in label
        assert b > a


# ------------------------------------------------- planner replacement law
AUDIT_FACTS = "first amendment audit at the police station parking lot"


def test_planner_cuts_no_raw_clip_from_foreign_narration_window():
    t = _words(
        (60.0, "officers make contact at the parking lot"),
        (200.0, "welcome back to the channel — in today's video the audit"),
        (2600.0, "do not forget to subscribe for more breakdowns"),
        (900.0, "the lieutenant walked toward the camera"),
        (1500.0, "the auditor kept filming in public"),
        (2100.0, "the confrontation escalates in the parking lot"),
    )
    blocks = plan_fallback_blocks(2700.0, t, case_facts=AUDIT_FACTS)
    raws = [(b.source_start_s, b.source_start_s + b.dur_s) for b in blocks if b.kind == "raw"]
    assert raws, "an on-menu case must still produce raw clips"
    for a, b in raws:
        assert not (a < 208.0 and b > 192.0), f"raw clip {a:.1f}-{b:.1f}s rides foreign narration"
        assert not (a < 2615.0 and b > 2585.0), f"raw clip {a:.1f}-{b:.1f}s rides the outro CTA"


def test_planner_replacement_keeps_full_plan_when_narration_is_excluded():
    # the foreign windows remove candidate speech — the plan must still hit
    # its invariants (blocks exist, narration carries the story) rather than
    # collapsing to an empty or raw-only timeline
    t = _words(
        (60.0, "officers make contact at the parking lot"),
        (200.0, "welcome back to the channel"),
        (900.0, "the lieutenant blocked the camera path"),
        (1500.0, "the auditor kept filming in public"),
        (2100.0, "the internal investigation was announced"),
        (2600.0, "thanks for watching this episode"),
    )
    blocks = plan_fallback_blocks(2700.0, t, case_facts=AUDIT_FACTS)
    kinds = [b.kind for b in blocks]
    assert kinds.count("narration") >= 9  # hook + bridges + transitions + outro
    assert kinds.count("raw") >= 4
