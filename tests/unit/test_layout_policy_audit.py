"""qa.audit_layout_and_policy — the layout/policy QA gates (PLAN §3.3G)."""

from __future__ import annotations

from director_os.core.qa_probe import audit_layout_and_policy
from director_os.subtitles.ass_builder import CaptionBeat, WordTiming, build_ass


def _beats() -> list[CaptionBeat]:
    """Two cues per speaker, overlapping so both lanes get used."""

    def beat(text: str, start_ms: int, end_ms: int, speaker: str) -> CaptionBeat:
        words = text.split()
        span = (end_ms - start_ms) // max(len(words), 1)
        return CaptionBeat(
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
            speaker=speaker,
            words=[
                WordTiming(word=w, start_ms=start_ms + i * span, end_ms=start_ms + (i + 1) * span)
                for i, w in enumerate(words)
            ],
        )

    # overlapping time ranges -> builder must stack into lane 1 + lane 2
    return [
        beat("units requesting backup now", 0, 3000, "OFFICER"),
        beat("copy that en route", 1000, 4000, "DISPATCH"),
        beat("suspect vehicle heading west", 5000, 8000, "OFFICER"),
        beat("air support is six minutes out", 6000, 9000, "DISPATCH"),
    ]


def test_good_layout_passes(tmp_path):
    ass = tmp_path / "subs.ass"
    ass.write_text(build_ass(_beats()), encoding="utf-8")
    report = audit_layout_and_policy(
        ass_path=ass,
        cards=[{"template": "location_intro", "fields": {"title": "CASE FILE"}}],
        blurs=[{"bbox": {"x": 100, "y": 100, "w": 200, "h": 100}, "sigma": 24}],
    )
    assert report.ok, [c.detail for c in report.failures]
    names = [c.name for c in report.checks]
    assert "subtitle_lanes" in names and "card_policy" in names and "blur_policy" in names


def test_margin_violation_fails(tmp_path):
    import pysubs2

    ass = tmp_path / "bad.ass"
    subs = pysubs2.SSAFile.from_string(build_ass(_beats()))
    lane1_name, lane1 = next((n, s) for n, s in subs.styles.items() if n.startswith("Lane1_"))
    lane1.marginv = 30  # break the 46 law
    ass.write_text(subs.to_string("ass"), encoding="utf-8")
    report = audit_layout_and_policy(ass_path=ass)
    assert not report.ok
    detail = next(c for c in report.checks if c.name == "subtitle_lanes").detail
    assert "MarginV" in detail


def test_italic_style_fails(tmp_path):
    import pysubs2

    ass = tmp_path / "italic.ass"
    subs = pysubs2.SSAFile.from_string(build_ass(_beats()))
    style_name, style = next(iter(subs.styles.items()))
    style.italic = True  # law: upright bold only
    ass.write_text(subs.to_string("ass"), encoding="utf-8")
    report = audit_layout_and_policy(ass_path=ass)
    assert not report.ok
    assert "italic" in next(c for c in report.checks if c.name == "subtitle_lanes").detail


def test_blur_outside_frame_fails():
    report = audit_layout_and_policy(
        blurs=[{"bbox": {"x": 1800, "y": 900, "w": 400, "h": 400}, "sigma": 24}]
    )
    assert not report.ok
    assert "EXCEEDS" in next(c for c in report.checks if c.name == "blur_policy").detail


def test_bad_sigma_fails():
    report = audit_layout_and_policy(
        blurs=[{"bbox": {"x": 0, "y": 0, "w": 100, "h": 100}, "sigma": 500}]
    )
    assert not report.ok


def test_empty_card_title_fails():
    report = audit_layout_and_policy(
        cards=[{"template": "location_intro", "fields": {"title": "  "}}]
    )
    assert not report.ok
    assert any(c.name == "card_policy" and not c.ok for c in report.checks)


def test_no_inputs_fails():
    report = audit_layout_and_policy()
    assert not report.ok
    assert report.checks[0].name == "inputs"
