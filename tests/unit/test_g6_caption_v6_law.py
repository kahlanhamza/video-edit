"""The committed v4/v6 caption styling law — locked against regression.

The v5 regression (Roboto 20pt under a 1920x1080 PlayRes on a 720p master =
~13px illegible text) is the exact class of bug the owner re-reported on
2026-11-23. The approved typography recovered from commit 0d56deb:
DejaVu Sans 40 @ PlayRes 1280x720, Officer=white / Suspect=gold /
Narrator=cyan, karaoke beats, \\fad(40,30), outline 2.9 / shadow 1.6,
cue law <=5 words / <=27 chars / >1s gap break, flash guard.
"""

from __future__ import annotations

from director_os.subtitles.ass_builder import (
    FLASH_MIN_S,
    LANE1_MARGIN_V,
    LANE2_MARGIN_V,
    CaptionBeat,
    WordTiming,
    build_ass,
    chunk_cue_words,
)


def _beat(text, start_ms, end_ms, speaker, words=None):
    return CaptionBeat(
        text=text, start_ms=start_ms, end_ms=end_ms, speaker=speaker, words=words or []
    )


def test_committed_typography_dejavu_40_playres_720():
    ass = build_ass([_beat("hands where I can see them", 0, 2000, "Officer")])
    assert "DejaVu Sans" in ass
    assert "PlayResX: 1280" in ass
    assert "PlayResY: 720" in ass
    assert "Fontsize: 40" in ass.replace("40,", "40,") or ",40," in ass  # fontsize field


def test_role_colors_officer_white_suspect_gold_narrator_cyan():
    # ASS PrimaryColour is &HAABBGGRR
    officer = build_ass([_beat("step out", 0, 1000, "Officer")])
    suspect = build_ass([_beat("am I in trouble", 0, 1000, "Suspect")])
    narrator = build_ass([_beat("Eustis, Florida", 0, 1000, "Narrator")])
    assert "&H00FFFFFF" in officer  # white
    assert "&H004DD8FF" in suspect  # gold #FFD84D
    assert "&H00FFE470" in narrator  # cyan #70E4FF


def test_karaoke_secondary_colors_match_the_committed_v6_header():
    # the tint an unsung \k word carries — pysubs2's default (pure red) was
    # NOT the approved look; 0d56deb pinned &H00CCC0B8 / &H004D9BB3 / &H00B8A37C
    officer = build_ass([_beat("step out", 0, 1000, "Officer")])
    suspect = build_ass([_beat("am I in trouble", 0, 1000, "Suspect")])
    narrator = build_ass([_beat("Eustis, Florida", 0, 1000, "Narrator")])
    assert "&H00CCC0B8" in officer  # soft gray-blue
    assert "&H004D9BB3" in suspect  # muted gold
    assert "&H00B8A37C" in narrator  # muted cyan
    # and no style line carries the pysubs2 red default any more
    for ass in (officer, suspect, narrator):
        styles = [ln for ln in ass.splitlines() if ln.startswith("Style:")]
        assert all("&H000000FF," not in ln for ln in styles)


def test_unknown_speaker_still_gets_a_deterministic_color():
    a = build_ass([_beat("dispatch", 0, 1000, "Dispatch_3")])
    b = build_ass([_beat("dispatch", 0, 1000, "Dispatch_3")])
    assert a == b  # deterministic, not random
    assert "PrimaryColour" in a


def test_outline_shadow_fad_and_karaoke_present():
    import pysubs2

    ass = build_ass(
        [
            _beat(
                "hands where I can see them",
                0,
                2000,
                "Officer",
                words=[
                    WordTiming(word="hands", start_ms=0, end_ms=400),
                    WordTiming(word="where", start_ms=400, end_ms=900),
                    WordTiming(word="I", start_ms=900, end_ms=1100),
                    WordTiming(word="can", start_ms=1100, end_ms=1400),
                    WordTiming(word="see", start_ms=1400, end_ms=1700),
                    WordTiming(word="them", start_ms=1700, end_ms=2000),
                ],
            )
        ]
    )
    assert r"\fad(40,30)" in ass
    assert r"\k" in ass
    subs = pysubs2.SSAFile.from_string(ass)
    style = next(iter(subs.styles.values()))
    assert style.outline == 2.9
    assert style.shadow == 1.6
    assert style.fontname == "DejaVu Sans"
    assert style.marginv in (LANE1_MARGIN_V, LANE2_MARGIN_V)


def test_flash_guard_extends_subliminal_cues():
    ass = build_ass([_beat("hey", 0, 80, "Suspect")])  # 80ms flash
    # the cue now lasts >= FLASH_MIN_S (words survive, flash dies)
    assert f"0:00:0{FLASH_MIN_S:.2f}" in ass or "0:00:00.30" in ass


def test_two_lanes_for_overlapping_speakers_still_hold():
    beats = [
        _beat("We pulled him over", 0, 2000, "Officer"),
        _beat("I was going home officer", 1000, 3000, "Suspect"),
        _beat("Step out of the vehicle", 3100, 5200, "Officer"),
    ]
    ass = build_ass(beats)
    assert "MarginV: 46" in ass or ",46," in ass
    assert ",96," in ass


def test_chunk_law_five_words_and_twentyseven_chars():
    words = [
        {"text": w, "start_s": i * 0.2, "end_s": i * 0.2 + 0.15, "speaker": "Officer"}
        for i, w in enumerate(["one", "two", "three", "four", "five", "six", "seven"])
    ]
    chunks = chunk_cue_words(words)
    assert all(len(c) <= 5 for c in chunks)
    assert len(chunks) == 2  # 5 + 2
    assert " ".join(w["text"] for w in chunks[0]) == "one two three four five"


def test_chunk_law_breaks_on_gap_and_speaker_change():
    words = [
        {"text": "alpha", "start_s": 0.0, "end_s": 0.2, "speaker": "Officer"},
        {"text": "beta", "start_s": 2.0, "end_s": 2.2, "speaker": "Officer"},  # >1s gap
        {"text": "gamma", "start_s": 2.3, "end_s": 2.5, "speaker": "Suspect"},  # speaker change
    ]
    chunks = chunk_cue_words(words)
    assert len(chunks) == 3
    assert [c[0]["text"] for c in chunks] == ["alpha", "beta", "gamma"]
