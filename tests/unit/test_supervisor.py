"""LLM proofread supervisor — the manual sandbox workflow, locked as tests.

The owner's workflow (Kaggle notebook + v6 decisions pass): readable speaker
turns go to the LLM in batches of 70 with the case facts; it fixes real ASR
mishearings, assigns TRUE Officer/Suspect roles, masks profanity, drops garble
— and every correction is WORD-LOCKED back onto the original ASR timestamps.
"""

from __future__ import annotations

import json

from director_os.subtitles.supervisor import (
    build_turns,
    supervise_transcript,
)


def _words():
    return [
        {"text": "I", "start_s": 0.0, "end_s": 0.1, "speaker": "0"},
        {"text": "got", "start_s": 0.2, "end_s": 0.3, "speaker": "0"},
        {"text": "a", "start_s": 0.4, "end_s": 0.45, "speaker": "0"},
        {"text": "super", "start_s": 0.5, "end_s": 0.7, "speaker": "0"},
        {"text": "suit", "start_s": 0.8, "end_s": 1.0, "speaker": "0"},
        {"text": "do", "start_s": 3.0, "end_s": 3.1, "speaker": "1"},
        {"text": "you", "start_s": 3.2, "end_s": 3.3, "speaker": "1"},
        {"text": "need", "start_s": 3.4, "end_s": 3.5, "speaker": "1"},
        {"text": "EMS", "start_s": 3.6, "end_s": 3.8, "speaker": "1"},
        {"text": "or", "start_s": 3.9, "end_s": 4.0, "speaker": "1"},
        {"text": "no", "start_s": 4.1, "end_s": 4.3, "speaker": "1"},
    ]


def test_build_turns_merges_same_speaker_runs():
    lines = build_turns({"words": _words()})
    # speaker "0" words (gap 0.1-2.9 > 1.8 to the next) -> one line; speaker "1" -> another
    assert len(lines) == 2
    assert lines[0].text == "I got a super suit"
    assert lines[0].speaker == "0"
    assert lines[1].text == "do you need EMS or no"
    assert lines[1].words[0]["start_s"] == 3.0


def test_supervisor_corrects_word_locked_and_roles_majority():
    decisions = [
        {"id": 0, "role": "Suspect", "clean_text": "I got off work"},
        {"id": 1, "role": "Officer", "clean_text": "Do you need EMS or no?"},
    ]

    def complete_fn(system, user):
        # the prompt must carry the batch payload + case facts law
        assert "Officer" in user and "Suspect" in user
        payload = user.split("Lines:\n", 1)[1]
        ids = [item["id"] for item in json.loads(payload)]
        return json.dumps([{**d, "id": ids[i] % len(decisions)} for i, d in enumerate(decisions)])

    proof = supervise_transcript({"words": _words()}, complete_fn, case_facts={"subject": "O.M."})
    assert proof is not None
    assert proof["supervised"] is True
    texts = [w["text"] for w in proof["words"] if w["text"]]
    # the mishearing is fixed ("super suit" -> "off work") — word-locked onto
    # the ORIGINAL timings (first word still starts at 0.0s)
    assert "off" in texts and "super" not in texts
    assert proof["words"][0]["start_s"] == 0.0
    assert proof["words"][0]["speaker"] == "Suspect"
    officer_words = [w for w in proof["words"] if w["speaker"] == "Officer"]
    assert officer_words and officer_words[0]["text"] == "Do"
    assert any(c["before"] == "I got a super suit" for c in proof["corrections"])
    assert proof["speaker_roles"] == {"0": "Suspect", "1": "Officer"}


def test_supervisor_drops_garble_but_keeps_timings():
    def complete_fn(system, user):
        return json.dumps([{"id": 0, "role": "Suspect", "clean_text": ""}])

    proof = supervise_transcript({"words": _words()}, complete_fn)
    assert proof is not None
    garbled = [w for w in proof["words"][:5]]
    assert all(w["text"] == "" for w in garbled)
    # timings survive (the cue builders skip empty text; time never moves)
    assert garbled[0]["start_s"] == 0.0 and garbled[-1]["end_s"] == 1.0
    assert proof["stats"]["garbled_lines"] == 1


def test_supervisor_lenient_parse_fences_and_think():
    def complete_fn(system, user):
        return (
            "<think>hmm let me reason about codes</think>```json\n"
            + json.dumps([{"id": 0, "role": "Suspect", "clean_text": "I got off work"}])
            + "\n```"
        )

    proof = supervise_transcript({"words": _words()}, complete_fn)
    assert proof is not None
    assert "off" in " ".join(w["text"] for w in proof["words"])


def test_supervisor_batch_failure_keeps_the_floor():
    calls = {"n": 0}

    def complete_fn(system, user):
        calls["n"] += 1
        return "not json at all"

    proof = supervise_transcript({"words": _words()}, complete_fn)
    # both retries burned -> no usable decisions -> deterministic floor (None)
    assert proof is None
    assert calls["n"] >= 2


def test_supervisor_no_llm_returns_none():
    assert supervise_transcript({"words": _words()}, None) is None


def test_supervisor_batches_of_70():
    # 140 DISTINCT turns (2.5s gaps break the turn-merge law) -> 2 batches of 70
    words = []
    for i in range(140):
        words.append({"text": f"w{i}", "start_s": i * 3.0, "end_s": i * 3.0 + 0.5, "speaker": "0"})
    seen_sizes: list[int] = []

    def complete_fn(system, user):
        payload = json.loads(user.split("Lines:\n", 1)[1])
        seen_sizes.append(len(payload))
        return json.dumps(
            [{"id": item["id"], "role": "Officer", "clean_text": item["text"]} for item in payload]
        )

    proof = supervise_transcript({"words": words}, complete_fn)
    assert seen_sizes == [70, 70]
    assert proof is not None and proof["stats"]["lines"] == 140
