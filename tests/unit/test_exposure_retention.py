"""Exposure matching (issue #16) + retention scoring (issue #19)."""

import subprocess
from pathlib import Path

import pytest
from director_os.agent_runtime.retention import RetentionReport, SectionSpan, score_retention
from director_os.vision_fx.exposure import (
    ExposureError,
    luma_stats,
    match_exposure_filters,
    night_rescue_filters,
    set_median_of,
)

# ------------------------------------------------------------- exposure


def _flat_clip(tmp_path: Path, name: str, luma_hex: str, seconds: float = 2.0) -> Path:
    out = tmp_path / f"{name}.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x{luma_hex}:size=320x240:rate=30:duration={seconds},noise=alls=6:allf=t",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def test_dark_clip_gets_subtle_lift(tmp_path: Path):
    clip = _flat_clip(tmp_path, "night", "181818")  # ~24 luma = dark interior
    stats = luma_stats(clip, sample_every_s=0.5)
    assert stats.median < 64
    filters = night_rescue_filters(stats)
    assert filters, "dark clip must get a lift proposal"
    assert all(f.startswith(("eq=gamma", "curves=")) for f in filters)
    g = float(filters[0].split("gamma=")[1])
    # eq=gamma maps v -> v^g: lift means g BELOW 1 (never below 0.8 — realism)
    assert 0.8 <= g < 1.0


def test_in_range_clip_untouched(tmp_path: Path):
    clip = _flat_clip(tmp_path, "day", "808080")  # ~128 luma = fine
    stats = luma_stats(clip, sample_every_s=0.5)
    assert night_rescue_filters(stats) == []


def test_bright_clip_protected(tmp_path: Path):
    clip = _flat_clip(tmp_path, "blown", "E8E8E8")
    stats = luma_stats(clip, sample_every_s=0.5)
    filters = night_rescue_filters(stats)
    assert filters and float(filters[0].split("gamma=")[1]) > 1.0


def test_match_exposure_converges_medians(tmp_path: Path):
    """Dashcam (bright) vs bodycam (dark): corrections pull both toward the set median."""
    dash = luma_stats(_flat_clip(tmp_path, "dash", "909090"), sample_every_s=0.5)
    body = luma_stats(_flat_clip(tmp_path, "body", "303030"), sample_every_s=0.5)
    target = set_median_of([dash, body])
    f_dash = match_exposure_filters(dash, target)
    f_body = match_exposure_filters(body, target)
    assert f_dash and f_body, "both clips differ from the set median"
    g_dash = float(f_dash[0].split("gamma=")[1])
    g_body = float(f_body[0].split("gamma=")[1])
    # bright pulled DOWN (g > 1), dark lifted UP (g < 1)
    assert g_dash > 1.0 > g_body
    assert 0.8 <= g_dash <= 1.45 and 0.8 <= g_body <= 1.45


def test_close_medians_emit_nothing(tmp_path: Path):
    a = luma_stats(_flat_clip(tmp_path, "a", "808080"), sample_every_s=0.5)
    b = luma_stats(_flat_clip(tmp_path, "b", "828282"), sample_every_s=0.5)
    target = set_median_of([a, b])
    assert match_exposure_filters(a, target) == []
    assert match_exposure_filters(b, target) == []


def test_no_frames_fails_loud(tmp_path: Path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("not a video")
    with pytest.raises(ExposureError):
        luma_stats(bogus)


# ------------------------------------------------------------- retention


def _acts() -> list[SectionSpan]:
    return [
        SectionSpan(start_s=0, end_s=7, kind="hook", title="Cold open: taser deploy"),
        SectionSpan(start_s=7, end_s=35, kind="location_intro", title="Eustis FL stop"),
        SectionSpan(start_s=35, end_s=90, kind="evidence_fact", title="0.14% BAC", tease_next=True),
        SectionSpan(start_s=90, end_s=140, kind="interview", title="Field sobriety"),
        SectionSpan(start_s=140, end_s=175, kind="case_outcome", title="Charges filed"),
    ]


def test_three_act_timeline_scores_high():
    rep = score_retention(_acts())
    assert rep.score >= 70
    assert rep.verdict == "shippable"
    assert any("hook" in r for r in rep.reasons)
    assert any("open loop" in r for r in rep.reasons)


def test_no_hook_scores_low():
    secs = [
        SectionSpan(start_s=0, end_s=60, kind="dashcam", title="Raw footage"),
        SectionSpan(start_s=60, end_s=120, kind="interview", title="Interview"),
        SectionSpan(start_s=120, end_s=180, kind="case_outcome", title="Outcome"),
    ]
    rep = score_retention(secs)
    assert rep.score < 70
    assert any("no hook" in p for p in rep.penalties)


def test_section_hog_penalized():
    secs = [
        SectionSpan(start_s=0, end_s=5, kind="hook", title="Hook"),
        SectionSpan(start_s=5, end_s=95, kind="interview", title="One long block"),  # 90/95 = 94%
        SectionSpan(start_s=95, end_s=100, kind="case_outcome", title="End"),
    ]
    rep = score_retention(secs)
    assert any("owns" in p and "momentum" in p for p in rep.penalties)
    assert rep.score < 70


def test_dead_span_penalized():
    secs = [
        SectionSpan(start_s=0, end_s=5, kind="hook", title="Hook"),
        SectionSpan(start_s=5, end_s=20, kind="intro", title="Setup"),
        SectionSpan(start_s=60, end_s=80, kind="case_outcome", title="End"),  # 40s gap
    ]
    rep = score_retention(secs)
    assert any("dead span" in p for p in rep.penalties)


def test_empty_timeline_zero():
    rep = score_retention([])
    assert rep.score == 0
    assert isinstance(rep, RetentionReport)


def test_tease_next_counts_as_open_loop():
    secs = [
        SectionSpan(start_s=0, end_s=6, kind="hook", title="Hook"),
        SectionSpan(start_s=6, end_s=40, kind="setup", title="Setup"),
        SectionSpan(start_s=40, end_s=80, kind="interview", title="Q&A", tease_next=True),
        SectionSpan(start_s=80, end_s=110, kind="case_outcome", title="End"),
    ]
    rep = score_retention(secs)
    assert any("open loop" in r for r in rep.reasons)
