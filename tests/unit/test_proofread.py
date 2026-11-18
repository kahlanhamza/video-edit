"""Proofread stage — police-jargon corrections + role tagging (PLAN guardrail 5)."""

from __future__ import annotations

import pytest
from director_os.subtitles.proofread import (
    ROLE_OFFICER,
    ROLE_SUSPECT,
    ProofreadError,
    prepare_transcript,
    proofread_words,
    tag_speakers,
)


def _words(texts: list[tuple[str, float, float, str | None]]) -> list[dict]:
    return [{"text": t, "start_s": s, "end_s": e, "speaker": spk} for t, s, e, spk in texts]


def test_ten_code_phrase_correction_retimed():
    # "ten four" (2 raw words) -> "10-4" (1 corrected word): the raw span is
    # redistributed, timings stay monotonic and inside the raw window
    raw = _words([("Copy", 0.0, 0.3, "S1"), ("ten", 0.3, 0.6, "S1"), ("four.", 0.6, 0.9, "S1")])
    out = proofread_words(raw)
    texts = [w["text"] for w in out["words"]]
    assert "10-4" in " ".join(texts)
    assert any(c["after"] == "10-4" for c in out["corrections"])
    first = next(w for w in out["words"] if w["text"].startswith("10-4"))
    assert 0.2 <= first["start_s"] <= 0.4
    assert first["end_s"] <= 0.9 + 1e-6
    # monotonic
    spans = [(w["start_s"], w["end_s"]) for w in out["words"]]
    assert all(a[1] <= b[0] + 1e-3 for a, b in zip(spans, spans[1:], strict=False))


def test_single_token_expansions():
    raw = _words([("Check", 0.0, 0.4, None), ("DL", 0.4, 0.8, None), ("vin", 0.8, 1.2, None)])
    out = proofread_words(raw)
    joined = " ".join(w["text"] for w in out["words"])
    assert "driver's license" in joined
    assert "VIN" in joined


def test_punctuation_and_case_survive():
    raw = _words([("Ten,", 0.0, 0.3, None), ("four?", 0.3, 0.6, None)])
    out = proofread_words(raw)
    assert any(w["text"].startswith("10-4") and w["text"].endswith("?") for w in out["words"])


def test_already_canonical_is_stable():
    raw = _words([("10-4", 0.0, 0.4, None), ("copy", 0.4, 0.8, None)])
    out = proofread_words(raw)
    assert out["words"][0]["text"] == "10-4"
    assert out["corrections"] == []


def test_empty_words_raise():
    with pytest.raises(ProofreadError, match="no timed words"):
        proofread_words([])


def test_tag_speakers_ranking():
    words = _words(
        [
            ("a", 0, 1, "S1"),
            ("b", 1, 2, "S1"),
            ("c", 2, 3, "S1"),
            ("d", 3, 4, "S2"),
            ("e", 4, 5, "S3"),
        ]
    )
    roles = tag_speakers(words)
    assert roles["S1"] == ROLE_OFFICER
    assert roles["S2"] == ROLE_SUSPECT
    assert roles["S3"] not in (ROLE_OFFICER, ROLE_SUSPECT)


def test_tag_speakers_never_narrator():
    words = _words([("a", 0, 1, "S1")])
    roles = tag_speakers(words)
    assert roles["S1"] == ROLE_OFFICER
    assert all(r != "Narrator" for r in roles.values())


def test_prepare_transcript_end_to_end():
    raw = _words(
        [
            ("ten", 0.0, 0.3, "S1"),
            ("four", 0.3, 0.6, "S1"),
            ("he", 0.6, 0.9, "S2"),
            ("has", 0.9, 1.2, "S2"),
            ("no", 1.2, 1.5, "S2"),
            ("DL", 1.5, 1.8, "S2"),
        ]
    )
    out = prepare_transcript({"words": raw})
    # rank = word count: S2 speaks 4 words (Officer), S1 speaks 2 (Suspect)
    assert out["speaker_roles"] == {"S2": ROLE_OFFICER, "S1": ROLE_SUSPECT}
    joined = " ".join(w["text"] for w in out["words"])
    assert "10-4" in joined
    assert "driver's license" in joined
    assert out["version"] == "v1"
