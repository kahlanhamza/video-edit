"""Guardrail 5 (PLAN §9.5): proofread transcripts re-align via difflib against
raw ASR word timestamps — never linear interpolation of the whole track."""

from __future__ import annotations

import pytest
from director_os.subtitles.align import AlignedWord, RawWord, align_words


def raw() -> list[RawWord]:
    return [
        RawWord(word="We", start_ms=0, end_ms=200),
        RawWord(word="pulled", start_ms=200, end_ms=600),
        RawWord(word="him", start_ms=600, end_ms=900),
        RawWord(word="over", start_ms=900, end_ms=1200),
    ]


def test_insertion_keeps_neighbor_timings():
    out = align_words("We pulled the driver over", raw())
    assert [w.word for w in out] == ["We", "pulled", "the", "driver", "over"]
    assert out[0].start_ms == 0 and out[-1].end_ms == 1200
    assert all(w.end_ms >= w.start_ms for w in out)
    # strictly monotonic
    for a, b in zip(out, out[1:], strict=False):
        assert b.start_ms >= a.end_ms


def test_replacement_distributes_raw_span():
    out = align_words("We pulled the suspect over", raw())
    replaced = [w for w in out if w.word in {"the", "suspect"}]
    assert len(replaced) == 2
    # the replacement must live inside the original "him" span (600-900)
    assert replaced[0].start_ms >= 600 - 1
    assert replaced[-1].end_ms <= 900 + 1


def test_deletion_drops_words_without_breaking_alignment():
    out = align_words("We pulled over", raw())
    assert [w.word for w in out] == ["We", "pulled", "over"]
    assert out[2].start_ms == 900  # "over" keeps its exact raw timing


def test_police_code_correction_survives():
    raw_words = [
        RawWord(word="Code", start_ms=0, end_ms=300),
        RawWord(word="108", start_ms=300, end_ms=700),
        RawWord(word="clear", start_ms=700, end_ms=1100),
    ]
    out = align_words("Signal 10-8, clear", raw_words)
    assert [w.word for w in out] == ["Signal", "10-8,", "clear"]
    assert out[-1].start_ms == 700


def test_empty_raw_raises():
    with pytest.raises(ValueError):
        align_words("anything", [])
    assert align_words("We pulled over", raw()) != AlignedWord.__init__  # sanity
