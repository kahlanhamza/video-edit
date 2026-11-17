"""evidence_fact card (issue #17) + YouTube safe-zone & clean out-frames (issue #18)."""

import subprocess
from pathlib import Path

import pytest
from director_os.core.qa_probe import (
    audit_layout_and_policy,
    audit_technical,
    clean_out_frames_check,
)
from director_os.vision_fx.cards import CardError, content_digest, render_card
from director_os.vision_fx.safe_zone import check_safe_zone, safe_zone_rect

# ------------------------------------------------------- evidence_fact card


@pytest.fixture()
def fact_card(tmp_path: Path) -> Path:
    return render_card(
        "evidence_fact",
        {
            "fact_value": "0.14% BAC",
            "fact_label": "Breath test at 22:41",
            "source": "TOX-2024-1187, p.3",
            "badge": "FACT",
        },
        tmp_path / "fact.png",
    )


def test_evidence_fact_renders(fact_card: Path):
    assert fact_card.exists()
    assert fact_card.stat().st_size > 2000  # real content, not a blank canvas
    assert content_digest(fact_card)  # cache key works


def test_evidence_fact_is_distinct_per_content(tmp_path: Path):
    a = render_card(
        "evidence_fact",
        {"fact_value": "0.14% BAC", "fact_label": "l", "source": "s"},
        tmp_path / "a.png",
    )
    b = render_card(
        "evidence_fact",
        {"fact_value": "1.2x limit", "fact_label": "l", "source": "s"},
        tmp_path / "b.png",
    )
    assert content_digest(a) != content_digest(b)


def test_unknown_template_still_rejected():
    with pytest.raises(CardError):
        render_card("nope", {}, "/tmp/nope.png")


# ------------------------------------------------------------- safe zones


def test_safe_zone_rect_geometry():
    r = safe_zone_rect(1920, 1080)
    assert r.x0 == int(1920 * 0.06) and r.y0 == int(1080 * 0.135)
    assert r.x1 == int(1920 * 0.94) and r.y1 == int(1080 * 0.88)


def test_card_inside_zone_passes():
    ok, detail = check_safe_zone((200, 200, 640, 180), 1920, 1080)
    assert ok, detail


def test_card_in_title_band_fails():
    ok, detail = check_safe_zone((200, 20, 640, 180), 1920, 1080)
    assert not ok and "top" in detail


def test_card_on_progress_bar_fails():
    ok, detail = check_safe_zone((200, 950, 640, 100), 1920, 1080)
    assert not ok and "bottom" in detail


def test_audit_layout_card_safe_zone_check():
    good = {
        "template": "evidence_fact",
        "fields": {"title": "BAC"},
        "bbox": {"x": 200, "y": 300, "w": 640, "h": 180},
    }
    bad = {
        "template": "evidence_fact",
        "fields": {"title": "BAC"},
        "bbox": {"x": 200, "y": 5, "w": 640, "h": 180},
    }
    rep_good = audit_layout_and_policy(cards=[good], frame=(1920, 1080))
    rep_bad = audit_layout_and_policy(cards=[bad], frame=(1920, 1080))
    assert all(c.ok for c in rep_good.checks if c.name == "card_safe_zone")
    assert any(not c.ok for c in rep_bad.checks if c.name == "card_safe_zone")


# --------------------------------------------------- clean out-frame audit


def _render_src(
    seconds: float, extra_vf: str | None = None, tail_pad: float = 0.0, tmp: Path | None = None
) -> Path:
    out = (tmp or Path("/tmp")) / f"qa_{abs(hash((seconds, extra_vf, tail_pad))) % 10**9}.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=320x240:rate=30:duration={seconds}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={seconds + tail_pad}",
        "-map",
        "0:v",
        "-map",
        "1:a",
    ]
    if extra_vf:
        cmd += ["-vf", extra_vf]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def test_clean_fixture_passes(tmp_path: Path):
    src = _render_src(3.0, tmp=tmp_path)
    check = clean_out_frames_check(src)
    assert check.ok, check.detail


def test_trailing_black_fails(tmp_path: Path):
    """The failure mode from real renders: black tail after content ends."""
    content = _render_src(2.0, tmp=tmp_path)
    out = tmp_path / "with_tail.mp4"
    r = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(content),
            "-f",
            "lavfi",
            "-i",
            "color=black:size=320x240:rate=30:duration=1.5",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    check = clean_out_frames_check(out)
    assert not check.ok and "tail_black" in check.detail


def test_audit_technical_includes_clean_out_frames(tmp_path: Path):
    src = _render_src(2.5, tmp=tmp_path)
    report = audit_technical(src)
    names = [c.name for c in report.checks]
    assert "clean_out_frames" in names
