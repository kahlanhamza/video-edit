"""Full-dialogue transcript turns — the notebook law for Phase A quality.

The owner's weak-narration report traced to the prompt: the 1-minute story
index caps speech at 360 chars/bucket, so the Director LLM wrote generic
commentary. The manual sandbox workflow feeds the WHOLE condensed transcript
(speaker turns, gap <= 1.8s merged, <= 240 chars, timestamped) — that is what
makes contradiction-quoting narration possible.
"""

from __future__ import annotations

from director_os.agent_runtime.story_map import (
    TURN_MAX_CHARS,
    build_transcript_turns,
    render_full_transcript,
)


def _words():
    return [
        {"text": "What", "start_s": 125.0, "end_s": 125.4, "speaker": "1"},
        {"text": "time", "start_s": 125.5, "end_s": 125.8, "speaker": "1"},
        {"text": "did", "start_s": 125.9, "end_s": 126.1, "speaker": "1"},
        {"text": "you", "start_s": 126.2, "end_s": 126.4, "speaker": "1"},
        {"text": "get", "start_s": 126.5, "end_s": 126.7, "speaker": "1"},
        {"text": "off?", "start_s": 126.8, "end_s": 127.2, "speaker": "1"},
        {"text": "Six", "start_s": 129.5, "end_s": 129.7, "speaker": "0"},
        {"text": "thirty.", "start_s": 129.8, "end_s": 130.1, "speaker": "0"},
        {"text": "You", "start_s": 133.0, "end_s": 133.3, "speaker": "0"},
        {"text": "said", "start_s": 133.4, "end_s": 133.7, "speaker": "0"},
        {"text": "eight.", "start_s": 133.8, "end_s": 134.1, "speaker": "0"},
    ]


def test_turns_merge_same_speaker_within_gap():
    turns = build_transcript_turns({"words": _words()})
    assert len(turns) == 3  # officer question, answer, contradiction
    assert turns[0]["text"] == "What time did you get off?"
    assert turns[0]["spk"] == "S1"
    assert turns[1]["text"] == "Six thirty."


def test_turn_render_is_timestamped_and_readable():
    text = render_full_transcript({"words": _words()})
    assert "[02:05-02:07]" in text
    assert "(125.0-127.2s) S1: What time did you get off?" in text
    assert "S0: Six thirty." in text


def test_oversized_turns_split_at_the_cap():
    words = []
    for i in range(200):
        words.append(
            {"text": f"word{i}", "start_s": i * 0.2, "end_s": i * 0.2 + 0.15, "speaker": "0"}
        )
    turns = build_transcript_turns({"words": words})
    assert turns, "turns must exist"
    assert all(len(t["text"]) <= TURN_MAX_CHARS for t in turns)
    assert len(turns) > 1


def test_segments_path_preferred_when_diarized():
    transcript = {
        "segments": [
            {"text": "Step out of the vehicle.", "start_s": 10.0, "end_s": 12.5, "speaker": "1"},
            {"text": "Why?", "start_s": 14.0, "end_s": 14.5, "speaker": "0"},
        ]
    }
    turns = build_transcript_turns(transcript)
    assert len(turns) == 2
    assert turns[0]["text"] == "Step out of the vehicle."


def test_budget_truncation_never_crashes():
    words = [
        {"text": f"w{i}", "start_s": i * 3.0, "end_s": i * 3.0 + 0.5, "speaker": "0"}
        for i in range(2000)
    ]
    text = render_full_transcript({"words": words}, char_budget=5000)
    assert len(text) < 6000
    assert "truncated" in text


def test_empty_transcript_renders_empty():
    assert render_full_transcript({}) == ""
