"""Master-clock caption cues — the v6 "caption truth on the master clock" law.

The failure this locks out (the owner's 2026-11-23 report): captions built on
the 43-minute source clock, so raw dialogue captions landed over narration
b-roll blocks and every cue drifted once the edit compressed the timeline.
Law: narrator cues ride TTS stamps rebased per narration block, raw cues remap
through each kept raw clip's window, and narration blocks NEVER carry raw
captions (narrator-on-broll isolation).
"""

from __future__ import annotations

from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)
from director_os.subtitles.master_cues import (
    _even_word_distribution,
    build_master_caption_cues,
    narration_word_stamps,
)


def _timeline():
    spec = TimelineSpec(
        name="cues",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw1",
                        source_start_s=100.0,
                        dur_s=10.0,
                        meta=RawClipMeta(asset_id="a"),
                    ),
                    ClipSpec(
                        name="narr1",
                        source_start_s=500.0,
                        dur_s=8.0,
                        meta=NarratorClipMeta(script_text="the narrator explains the case here"),
                    ),
                    ClipSpec(
                        name="raw2",
                        source_start_s=300.0,
                        dur_s=6.0,
                        meta=RawClipMeta(asset_id="a"),
                    ),
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=24.0)]),
        ],
    )
    return build_timeline(spec)


TRANSCRIPT = {
    "words": [
        {"text": "stop", "start_s": 102.0, "end_s": 102.5, "speaker": "Officer"},
        {"text": "the", "start_s": 102.6, "end_s": 102.8, "speaker": "Officer"},
        {"text": "vehicle", "start_s": 103.0, "end_s": 103.6, "speaker": "Officer"},
        # speech INSIDE the b-roll source window (500-508s) — must NEVER caption
        {"text": "background", "start_s": 502.0, "end_s": 502.6, "speaker": "Suspect"},
        {"text": "chatter", "start_s": 503.0, "end_s": 503.6, "speaker": "Suspect"},
        # speech in the second raw clip (300-306s -> master 18-24s)
        {"text": "I", "start_s": 301.0, "end_s": 301.2, "speaker": "Suspect"},
        {"text": "was", "start_s": 301.3, "end_s": 301.5, "speaker": "Suspect"},
        {"text": "going", "start_s": 301.6, "end_s": 301.9, "speaker": "Suspect"},
        {"text": "home", "start_s": 302.0, "end_s": 302.4, "speaker": "Suspect"},
    ]
}

NARRATION_WORDS = {
    "the narrator explains the case here": [
        {"word": w, "start_s": 0.4 + i * 0.5, "end_s": 0.8 + i * 0.5}
        for i, w in enumerate(["the", "narrator", "explains", "the", "case", "here"])
    ]
}


def test_narration_cues_ride_master_clock_after_the_raw_clip():
    beats, counts = build_master_caption_cues(_timeline(), TRANSCRIPT, NARRATION_WORDS)
    narr = [b for b in beats if b.speaker == "Narrator"]
    assert narr and counts["narration"] >= 1
    # the narration block starts at master 10s (after the 10s raw clip)
    assert narr[0].start_ms >= 10_000
    assert narr[0].start_ms < 18_000  # and ends before the second raw clip


def test_raw_cues_remap_through_source_windows():
    beats, _counts = build_master_caption_cues(_timeline(), TRANSCRIPT, NARRATION_WORDS)
    officer = [b for b in beats if b.speaker == "Officer"]
    suspect = [b for b in beats if b.speaker == "Suspect"]
    # raw clip 1 at source 102s -> master 2s
    assert officer and 1_000 <= officer[0].start_ms < 5_000
    # raw clip 2 at source 301s -> master 18+1=19s
    assert suspect and 18_000 <= suspect[0].start_ms < 24_000


def test_broll_blocks_never_carry_raw_captions():
    """THE narrator-on-broll isolation gate — the owner's bug report."""
    beats, counts = build_master_caption_cues(_timeline(), TRANSCRIPT, NARRATION_WORDS)
    # the b-roll source window (500-508s) had Suspect speech in the transcript;
    # no cue may carry that text at ANY master time
    texts = " ".join(b.text for b in beats)
    assert "background" not in texts and "chatter" not in texts
    # every Suspect cue comes from the second raw clip's window only
    suspect = [b for b in beats if b.speaker == "Suspect"]
    assert suspect and all(b.start_ms >= 18_000 for b in suspect)


def test_proofread_words_win_over_raw_and_roles_flow_through():
    proofed = {
        "proofread": {
            "words": [
                {"text": "Stop", "start_s": 102.0, "end_s": 102.5, "speaker": "Officer"},
                {"text": "the", "start_s": 102.6, "end_s": 102.8, "speaker": "Officer"},
                {"text": "vehicle", "start_s": 103.0, "end_s": 103.6, "speaker": "Officer"},
            ]
        }
    }
    beats, _ = build_master_caption_cues(_timeline(), proofed, NARRATION_WORDS)
    officer = [b for b in beats if b.speaker == "Officer"]
    assert officer and officer[0].text.startswith("Stop")


def test_sidecar_stamps_read_from_the_manifest():
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        sidecar = Path(td) / "clip.json"
        sidecar.write_text(
            json.dumps({"words": NARRATION_WORDS["the narrator explains the case here"]})
        )
        stamps = narration_word_stamps(
            [
                {
                    "script_text": "the narrator explains the case here",
                    "sidecar": str(sidecar),
                    "duration_s": 4.0,
                }
            ]
        )
        assert "the narrator explains the case here" in stamps
        assert len(stamps["the narrator explains the case here"]) == 6

        # missing sidecar -> even distribution keeps narration captions alive
        stamps2 = narration_word_stamps(
            [
                {
                    "script_text": "seven words of narration here now ok",
                    "sidecar": "",
                    "duration_s": 7.0,
                }
            ]
        )
        assert len(stamps2["seven words of narration here now ok"]) == 7


def test_even_word_distribution_spans_the_block():
    words = _even_word_distribution("one two three four", 4.0)
    assert len(words) == 4
    assert words[0]["start_s"] == 0.0
    assert abs(words[-1]["end_s"] - 4.0) < 1e-6


def test_a1_trimmed_block_shifts_narration_stamps_by_vo_in():
    """A trimmed VO (vo_in=1.5) plays the stem from 1.5s — captions shift too."""
    from director_os.otio_engine.timeline import build_timeline as _bt

    spec = TimelineSpec(
        name="trim",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="narr_trim",
                        source_start_s=0.0,
                        dur_s=5.0,
                        meta=NarratorClipMeta(
                            script_text="the narrator explains the case here",
                            vo_in_s=1.5,
                            vo_out_s=6.5,
                        ),
                    )
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=5.0)]),
        ],
    )
    tl = _bt(spec)
    beats, counts = build_master_caption_cues(tl, {}, NARRATION_WORDS)
    narr = [b for b in beats if b.speaker == "Narrator"]
    assert narr and counts["narration"] >= 1
    # first word starts at 0.4s on the full stem -> 0.4-1.5 < 0 -> first VISIBLE
    # word is the one at 0.9s stem (0.9-1.5 <0 clipped) — the 1.4s word lands at
    # -0.1 (clipped), 1.9s lands at 0.4s master-block time. No cue may start
    # before the block start, and words before vo_in never caption.
    assert all(b.start_ms >= 0 for b in narr)
    first_word_ms = narr[0].words[0].start_ms
    assert 0 <= first_word_ms < 1_000  # shifted into the block, not at stem 1.9s->400ms


def test_proofread_garble_mask_renders_inaudible_tag():
    """A supervisor-masked word renders *inaudible* — tagged, never dropped.

    Measured live: 19 masked words in the Eustis proofread vanished silently
    (the words_src text filter), so the runbook §2.1 caption-truth law had no
    representation in the pipeline burn.
    """
    from director_os.otio_engine.timeline import build_timeline as _bt

    spec = TimelineSpec(
        name="garble",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw1",
                        source_start_s=100.0,
                        dur_s=10.0,
                        meta=RawClipMeta(asset_id="a"),
                    ),
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=10.0)]),
        ],
    )
    tl = _bt(spec)
    transcript = {
        "proofread": {
            "words": [
                {"text": "get", "start_s": 101.0, "end_s": 101.4, "speaker": "Officer"},
                {"text": "", "start_s": 101.5, "end_s": 102.2, "speaker": "Officer"},
                {"text": "", "start_s": 102.3, "end_s": 103.0, "speaker": "Officer"},
                {"text": "on the ground", "start_s": 103.1, "end_s": 103.8, "speaker": "Officer"},
            ],
            "supervised": True,
        }
    }
    beats, stats = build_master_caption_cues(tl, transcript, {})
    joined = " ".join(b.text for b in beats)
    assert "*inaudible*" in joined
    assert "on the ground" in joined
    assert stats["raw"] >= 2
    # masked words keep their real timings (the supervisor's word-lock law)
    tag_beat = next(b for b in beats if "*inaudible*" in b.text)
    tag_word = next(w for w in tag_beat.words if w.word == "*inaudible*")
    assert tag_word.start_ms == 1500
    assert tag_word.end_ms == 2200


def test_raw_asr_fallback_never_grows_tags():
    """Without a proofread pass the raw-ASR fallback path is untouched."""
    from director_os.otio_engine.timeline import build_timeline as _bt

    spec = TimelineSpec(
        name="rawonly",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(
                        name="raw1",
                        source_start_s=100.0,
                        dur_s=10.0,
                        meta=RawClipMeta(asset_id="a"),
                    ),
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=10.0)]),
        ],
    )
    tl = _bt(spec)
    transcript = {
        "words": [
            {"text": "visible speech", "start_s": 101.0, "end_s": 101.8, "speaker": "Officer"},
        ]
    }
    beats, _ = build_master_caption_cues(tl, transcript, {})
    assert [b.text for b in beats] == ["visible speech"]
    assert not any("inaudible" in b.text for b in beats)
