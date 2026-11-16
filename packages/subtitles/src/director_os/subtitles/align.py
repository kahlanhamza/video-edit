"""Word-level alignment of proofread transcripts — PLAN guardrail 5.

Never linearly interpolate text when words are edited; align via
`difflib.SequenceMatcher` against the raw ASR word timestamps so millisecond
precision survives police-code corrections ("10-8" -> "Signal 10-8, back in service").
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

_PUNCT = str.maketrans("", "", string.punctuation)


class RawWord(BaseModel):
    word: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class AlignedWord(BaseModel):
    word: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


def _norm(token: str) -> str:
    return token.translate(_PUNCT).lower()


def _tokens(text: str) -> list[str]:
    return text.split()


def align_words(corrected_text: str, raw_words: Sequence[RawWord]) -> list[AlignedWord]:
    """Map corrected text back onto raw timings.

    - equal blocks map 1:1 to raw word timings
    - replaced blocks distribute the raw time span proportionally by word length
    - inserted blocks interpolate between their neighbors (midpoint slicing)
    - deleted raw words are dropped
    Output is monotonic: start_ms <= end_ms <= next.start_ms.
    """
    corrected = _tokens(corrected_text)
    raw = [(w.word, w.start_ms, w.end_ms) for w in raw_words]
    if not corrected:
        return []
    if not raw:
        raise ValueError("raw_words is empty: no ASR timings to align against")

    a = [_norm(t) for t in corrected]
    b = [_norm(w) for w, _, _ in raw]
    matcher = SequenceMatcher(None, a, b, autojunk=False)

    out: list[AlignedWord] = []
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(a2 - a1):
                word, s, e = raw[b1 + k]
                out.append(AlignedWord(word=corrected[a1 + k], start_ms=s, end_ms=e))
        elif tag == "replace":
            raw_span_s = raw[b1][1]
            raw_span_e = raw[b2 - 1][2]
            span = max(1, raw_span_e - raw_span_s)
            weights = [max(1, len(_norm(corrected[i]))) for i in range(a1, a2)]
            total_w = sum(weights)
            cursor = raw_span_s
            for k, i in enumerate(range(a1, a2)):
                share = int(span * weights[k] / total_w)
                s = cursor
                e = raw_span_e if k == a2 - a1 - 1 else min(raw_span_e, s + share)
                cursor = e
                out.append(AlignedWord(word=corrected[i], start_ms=s, end_ms=e))
        elif tag == "insert":
            prev_end = raw[b1 - 1][2] if b1 > 0 else raw[0][1]
            next_start = raw[b1][1] if b1 < len(raw) else raw[-1][2]
            gap = max(1, next_start - prev_end)
            weights = [max(1, len(_norm(corrected[i]))) for i in range(a1, a2)]
            total_w = sum(weights)
            cursor = prev_end
            for k, i in enumerate(range(a1, a2)):
                share = int(gap * weights[k] / total_w)
                s = cursor
                e = next_start if k == a2 - a1 - 1 else min(next_start, s + share)
                cursor = e
                out.append(AlignedWord(word=corrected[i], start_ms=s, end_ms=e))
        # tag == "delete": raw words dropped by the proofread — skip

    # monotonicity repair (rounding can create 1ms overlaps)
    for prev, cur in zip(out, out[1:], strict=False):
        if cur.start_ms < prev.end_ms:
            cur.start_ms = prev.end_ms
            if cur.end_ms < cur.start_ms:
                cur.end_ms = cur.start_ms
    return out
