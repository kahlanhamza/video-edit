"""The length-adaptive + post-truth laws (owner rule 2026-11-19).

- doc_scale: ANY video length plans correctly (the 43-min acceptance numbers
  are preserved exactly; shorter/longer tapes scale).
- snap_plan_to_speech: punchlines complete (pad_after), dead air >1.8s splits
  out, silent action windows ride through untouched.
- word-count law: narration durations are computed from words — an inflated
  LLM dur_s buys nothing.
- retime_cards_after_narration: cards re-anchor to MEASURED TTS durations.
- multi-b-roll renderer: 2-3 sequential muted windows under ONE TTS block.
"""

from __future__ import annotations

import json

from director_os.agent_runtime.director_session import (
    MacroPlan,
    PlanNarration,
    PlanRawClip,
    expand_macro_plan,
    narr_est_s,
    narr_words,
    parse_macro_plan,
    snap_plan_to_speech,
)
from director_os.agent_runtime.doc_scale import doc_scale
from director_os.agent_runtime.plan_invariants import audit_invariants, invariants_ok
from director_os.otio_engine.contract import CardMeta, NarratorClipMeta, RawClipMeta
from director_os.otio_engine.renderer import build_segment_cmd
from director_os.otio_engine.segment_cache import AudioRenderSpec, SegmentRenderSpec, VideoRenderSpec
from director_os.otio_engine.timeline import (
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)


# ------------------------------------------------------------------ doc scale
def test_doc_scale_preserves_the_eustis_law_exactly():
    s = doc_scale(2627.0)
    assert s.target_lo == 2627.0 * 0.28  # proportional band, uncapped
    assert s.target_hi == 900.0
    assert s.min_final_s == 600.0
    assert s.narr_lo == 10 and s.narr_hi == 14
    assert s.raw_min == 8
    assert s.min_coverage_span == 0.70
    assert s.hook_law and s.is_documentary


def test_doc_scale_adapts_to_any_length():
    short = doc_scale(900.0)  # 15-min tape: ~4-5 min doc, NOT the 10-min law
    assert 240 <= short.target_lo <= 270
    assert short.min_final_s == 225.0
    assert short.narr_lo == 4 and short.raw_min == 3
    assert short.hook_law  # the hook is not a long-video rule

    tiny = doc_scale(480.0)  # 8-min tape still gets the documentary treatment
    assert tiny.is_documentary and tiny.target_lo >= 120

    v_short = doc_scale(300.0)  # below 8 min: auto-cut law, generous floors
    assert not v_short.is_documentary and not v_short.hook_law

    huge = doc_scale(5400.0)  # 90-min tape: same master clamps as Eustis
    assert huge.target_lo == 900.0 and huge.min_final_s == 600.0

    # planner and QA share ONE scale (cannot disagree)
    from director_os.agent_runtime.plan_invariants import _scale_for

    assert _scale_for(900.0) == short


def test_invariants_scale_down_for_a_15_minute_tape():
    # sting + hook + 5 narration blocks + 5 raws spread across a 15-min tape
    raws = [ClipSpec(
        name=f"raw{i}",
        source_start_s=60.0 + i * 140.0,
        dur_s=25.0,
        meta=RawClipMeta(asset_id="a"),
    ) for i in range(5)]
    narrs = [ClipSpec(
        name=f"narr{i}",
        source_start_s=120.0 + i * 140.0,
        dur_s=22.0,
        meta=NarratorClipMeta(script_text="y" * 200),
    ) for i in range(5)]
    master = [
        ClipSpec(name="sting", source_start_s=0.0, dur_s=3.5, meta=RawClipMeta(asset_id="a")),
        ClipSpec(name="hook", source_start_s=30.0, dur_s=20.0,
                 meta=NarratorClipMeta(script_text="x" * 200)),
    ]
    for n, r in zip(narrs, raws, strict=True):
        master.extend([n, r])
    spec = TimelineSpec(
        name="short-doc",
        fps=30,
        width=1280,
        height=720,
        tracks=[
            TrackSpec(name="V1_master", items=master),
            TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=263.5)]),
        ],
    )
    timeline = build_timeline(spec)
    ok, failures = invariants_ok(timeline, 900.0)
    # a ~4.4-min doc from a 15-min tape with 6 narration blocks + 5 raws is a
    # VALID short documentary — the 43-min floors must not apply
    assert ok, [f.detail for f in failures]
    # ... and the SAME shape against a 43-min tape MUST fail (floors scale up)
    ok2, failures2 = invariants_ok(timeline, 2627.0)
    assert not ok2 and any(f.code == "final_duration_floor" for f in failures2)


# ------------------------------------------------------------- word-count law
def test_narration_duration_is_computed_from_words():
    assert narr_words("one two three four") == 4
    # 60 words at 2.6 w/s + tail ~= 23.9s regardless of the LLM's claim
    script = " ".join(f"w{i}" for i in range(60))
    est = narr_est_s(script)
    assert 23.0 <= est <= 24.5
    # floor: a 5-word stub still plans at >= 8s (never a 2s blip)
    assert narr_est_s("too short script") >= 8.0


def test_expand_macro_plan_ignores_inflated_llm_dur_s():
    plan = MacroPlan(
        acts=[
            {
                "name": "Cold Open",
                "blocks": [
                    {
                        "type": "narration",
                        "script": " ".join(f"w{i}" for i in range(40)),
                        "dur_s": 99.0,  # the LLM's inflated guess
                    }
                ],
            }
        ],
        cards=[],
    )
    steps = expand_macro_plan(plan, 1000.0)
    narr = next(s for s in steps if s.tool == "timeline.add_narration")
    assert abs(narr.args["dur_s"] - narr_est_s(" ".join(f"w{i}" for i in range(40)))) < 0.1
    assert narr.args["dur_s"] < 30.0  # NOT the 99s guess


# -------------------------------------------------------- ASR-boundary snap
def _transcript_40s():
    return {
        "segments": [
            {"speaker": "Officer", "start_s": float(t), "end_s": float(t) + 40.0,
             "text": f"utterance {t}"}
            for t in range(0, 2600, 60)
        ]
    }


def test_snap_completes_punchlines_and_splits_dead_air():
    plan = parse_macro_plan(
        json.dumps(
            {
                "acts": [
                    {
                        "name": "Act 1",
                        "blocks": [
                            # ends mid-utterance: snap extends to the true end + pad
                            {"type": "raw", "source_start_s": 70.0, "dur_s": 20.0},
                            # spans three utterances with 19s holes: SPLIT
                            {"type": "raw", "source_start_s": 60.0, "dur_s": 130.0},
                        ],
                    }
                ]
            }
        )
    )
    snapped, pieces = snap_plan_to_speech(plan, _transcript_40s(), 2700.0)
    blocks = [b for a in snapped.acts for b in a.blocks if isinstance(b, PlanRawClip)]
    # clip 1: 70..90 starts mid-utterance (60..100): start pulls back max 8s
    # (to 62.0) and the END always completes the line: 100 + 0.6 = 100.6
    first = blocks[0]
    assert abs(first.source_start_s - 62.0) < 0.01
    assert abs((first.source_start_s + first.dur_s) - 100.6) < 0.01
    # clip 2: 60..190 -> runs 59.5..100.6, 119.5..160.6, 179.5..220.6 — the
    # dead-air holes are split out and every punchline completes
    spans = [(b.source_start_s, b.source_start_s + b.dur_s) for b in blocks[1:]]
    assert len(spans) == 3
    assert spans[0][1] < spans[1][0] < spans[1][1] < spans[2][0]
    assert abs(spans[2][1] - 220.6) < 0.01


def test_snap_keeps_silent_action_windows_and_no_transcript_is_a_noop():
    plan = parse_macro_plan(
        json.dumps(
            {
                "acts": [
                    {"name": "Act 1", "blocks": [{"type": "raw", "source_start_s": 2050.0, "dur_s": 10.0}]}
                ]
            }
        )
    )
    # 2050..2060 sits in the 2020..2060/2060..2100 boundary? 2060 > 2060 ->
    # overlapping utterance 2020..2060 -> snapped; use a TRUE silent gap 2045
    snapped, _ = snap_plan_to_speech(plan, _transcript_40s(), 2700.0)
    blk = snapped.acts[0].blocks[0]
    assert isinstance(blk, PlanRawClip)  # survives (snapped to the real edges)

    same = parse_macro_plan(
        json.dumps(
            {"acts": [{"name": "A", "blocks": [{"type": "raw", "source_start_s": 5.0, "dur_s": 9.0}]}]}
        )
    )
    untouched, pieces = snap_plan_to_speech(same, None, 2700.0)
    assert pieces == 1  # no transcript -> plan rides through unchanged


# --------------------------------------------------------- card re-anchoring
def test_cards_reanchor_to_measured_tts_durations():
    import opentimelineio.opentime as ot_time

    from director_os.worker.pipeline import master_spans, retime_cards_after_narration

    spec = TimelineSpec(
        name="retime",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(name="sting", source_start_s=0.0, dur_s=4.0, meta=RawClipMeta(asset_id="a")),
                    ClipSpec(
                        name="hook",
                        source_start_s=30.0,
                        dur_s=20.0,  # planned; the real TTS is 12s
                        meta=NarratorClipMeta(script_text="hook line"),
                    ),
                    ClipSpec(name="raw2", source_start_s=100.0, dur_s=30.0, meta=RawClipMeta(asset_id="a")),
                ],
            ),
            TrackSpec(
                name="V2_overlay_cards",
                items=[
                    GapSpec(dur_s=26.0),
                    ClipSpec(
                        name="card_location_intro",
                        source_start_s=0.0,
                        dur_s=4.0,
                        meta=CardMeta(template="location_intro", fields={"title": "CASE FILE"}),
                    ),
                    GapSpec(dur_s=24.0),
                ],
            ),
        ],
    )
    timeline = build_timeline(spec)
    pre = master_spans(timeline)  # [(0,4), (4,20), (24,30)]
    assert abs(pre[2][0] - 24.0) < 0.01

    # the TTS clip for the hook measures 12s (8s shorter than planned)
    clips = [{"script_text": "hook line", "duration_s": 12.0, "clip": "/tmp/x.mp3"}]
    from director_os.worker.pipeline import retime_narration_to_tts

    assert retime_narration_to_tts(timeline, clips) == 1
    # the card sits 2s into the RAW item after the narration (pos 26.0) — the
    # narration shrink moves that raw item's start 24.0 -> 16.35, so the card
    # re-anchors to 16.35 + 2 = 18.35 instead of drifting with the old clock
    moved = retime_cards_after_narration(timeline, pre)
    assert moved == 1

    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.timeline import item_duration_s

    master = master_track(timeline)
    durs = [item_duration_s(i) for i in master]
    assert abs(durs[1] - 12.35) < 0.01  # 12s + 0.35 pad
    from director_os.otio_engine.timeline import track_by_name

    overlay = track_by_name(timeline, "V2_overlay_cards")
    t = 0.0
    card_pos = None
    for item in overlay:
        meta = (item.metadata.get("director_os") or {})
        if meta.get("kind") == "card":
            card_pos = t
        t += item_duration_s(item)
    assert card_pos is not None and abs(card_pos - 18.35) < 0.6


# ------------------------------------------------------- multi-b-roll render
def test_multi_broll_narration_renders_concat_windows():
    spec = SegmentRenderSpec(
        kind="narration",
        asset_id="a",
        source_start_s=10.0,
        source_dur_s=20.0,
        video=VideoRenderSpec(w=640, h=360, fps=30, preset="ultrafast", crf=40),
        audio=AudioRenderSpec(),
        narration_audio_path=None,
        broll_windows=[(10.0, 20.0), (40.0, 50.0), (80.0, 88.0)],
    )
    cmd = build_segment_cmd(spec, "/tmp/fake_source.mp4", "/tmp/out.mp4")
    joined = " ".join(cmd)
    # 3 windows -> base slice + 2 extra slice inputs + concat graph cycling to 20s
    assert joined.count("-ss") == 3
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=3" in graph and "[vbase]" in graph
    assert "-filter_complex" in joined  # the plain -vf path cannot concat


def test_single_broll_narration_keeps_the_plain_path():
    spec = SegmentRenderSpec(
        kind="narration",
        asset_id="a",
        source_start_s=10.0,
        source_dur_s=20.0,
        video=VideoRenderSpec(w=640, h=360, fps=30, preset="ultrafast", crf=40),
        audio=AudioRenderSpec(),
        broll_windows=None,
    )
    cmd = build_segment_cmd(spec, "/tmp/fake_source.mp4", "/tmp/out.mp4")
    assert "-filter_complex" not in cmd  # plain -vf chain, unchanged law


# ------------------------------------------------- post-fx captions + blur law
def test_master_caption_cues_cover_narration_and_raw_speech():
    from director_os.worker.post_fx import build_master_caption_cues

    spec = TimelineSpec(
        name="caps",
        fps=30,
        width=640,
        height=360,
        tracks=[
            TrackSpec(
                name="V1_master",
                items=[
                    ClipSpec(name="raw1", source_start_s=100.0, dur_s=10.0,
                             meta=RawClipMeta(asset_id="a")),
                    ClipSpec(name="narr1", source_start_s=30.0, dur_s=8.0,
                             meta=NarratorClipMeta(script_text="the narrator explains the case")),
                ],
            ),
            TrackSpec(name="V2_overlay_cards", items=[GapSpec(dur_s=18.0)]),
        ],
    )
    timeline = build_timeline(spec)
    transcript = {
        "words": [
            {"text": "officer", "start_s": 102.0, "end_s": 102.6, "speaker": "Officer"},
            {"text": "speak", "start_s": 103.0, "end_s": 103.8, "speaker": "Officer"},
            {"text": "here", "start_s": 104.0, "end_s": 104.4, "speaker": "Officer"},
        ]
    }
    narration_words = {
        "the narrator explains the case": [
            {"word": w, "start_s": 0.5 + i * 0.4, "end_s": 0.8 + i * 0.4}
            for i, w in enumerate(["the", "narrator", "explains", "the", "case"])
        ]
    }
    beats, counts = build_master_caption_cues(timeline, transcript, narration_words)
    assert counts["narration"] >= 1 and counts["raw"] >= 1 and counts["total"] == len(beats)
    narr_beats = [b for b in beats if b.speaker == "NARRATOR"]
    raw_beats = [b for b in beats if b.speaker == "Officer"]
    assert narr_beats and raw_beats
    # narration cues ride the MASTER clock (after the 10s raw clip)
    assert narr_beats[0].start_ms >= 10_000
    # raw speech remaps through the source->master window (102s -> 2s)
    assert raw_beats[0].start_ms >= 1_000 and raw_beats[0].start_ms < 5_000


def test_primary_track_selection_and_watermark_band():
    from director_os.vision_fx.compute.base import FaceTrack, TrackHit
    from director_os.worker.post_fx import _primary_track_id, _watermark_trajectory

    main = FaceTrack(track_id="face_1", hits=[
        TrackHit(t_s=t, x=400, y=200, w=220, h=260, score=0.9) for t in range(0, 60)
    ])
    bystander = FaceTrack(track_id="face_2", hits=[
        TrackHit(t_s=t, x=20, y=300, w=60, h=60, score=0.6) for t in range(0, 5)
    ])
    assert _primary_track_id([bystander, main]) == "face_1"
    assert _primary_track_id([]) is None

    wm = _watermark_trajectory(150, 30.0, 1280, 720)
    assert wm.track_id == "watermark_static"
    assert len(wm.boxes) == 150  # dense: every frame gets a box
    b = wm.boxes[0]
    assert b.x + b.w <= 1280 and b.y + b.h <= 720  # inside the canvas
    assert b.y > 720 * 0.7  # bottom band
