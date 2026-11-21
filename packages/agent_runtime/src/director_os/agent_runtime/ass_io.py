"""Minimal .ass subtitle I/O for the caption workbench.

Parses and writes ONLY what the caption law needs (styles + dialogue events)
with byte-stable round-tripping for untouched lines, so a work copy differs
from its source exclusively where an edit was applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DIALOGUE_RE = re.compile(
    r"^Dialogue:\s*([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),(.*)$"
)
STYLE_RE = re.compile(r"^Style:\s*(\w+),")

TAG_STYLE_NAME = "Tag"
#: gray italic, no speaker colour — the caption law for unreliable audio
TAG_STYLE_LINE = (
    "Style: Tag,Roboto,42,&H00A0A0A0,&H00CCC0B8,&H00000000,&H90000000,"
    "-1,0,-1,0,100,100,0.4,0,1,2.4,1.2,2,80,80,58,1"
)


@dataclass
class AssCue:
    index: int  # position among Dialogue lines (0-based, stable within a file)
    style: str
    start_s: float
    end_s: float
    text: str
    raw: str
    tagged: bool = False
    meta: dict = field(default_factory=dict)


def ts_to_s(ts: str) -> float:
    h, m, rest = ts.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def s_to_ts(s: float) -> str:
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}:{m:02d}:{s % 60:05.2f}"


def parse_cues(text: str) -> list[AssCue]:
    cues: list[AssCue] = []
    for line in text.splitlines():
        m = DIALOGUE_RE.match(line)
        if m:
            cues.append(
                AssCue(
                    index=len(cues),
                    style=m.group(4).strip(),
                    start_s=ts_to_s(m.group(2)),
                    end_s=ts_to_s(m.group(3)),
                    text=m.group(10),
                    raw=line,
                )
            )
    return cues


def has_style(text: str, name: str) -> bool:
    for line in text.splitlines():
        m = STYLE_RE.match(line)
        if m and m.group(1) == name:
            return True
    return False


def ensure_tag_style(text: str) -> str:
    """Insert the gray-italic Tag style after the last Style line (idempotent)."""
    if has_style(text, TAG_STYLE_NAME):
        return text
    lines = text.splitlines()
    last_style = max((i for i, ln in enumerate(lines) if ln.startswith("Style:")), default=None)
    if last_style is None:
        lines.append(TAG_STYLE_LINE)
    else:
        lines.insert(last_style + 1, TAG_STYLE_LINE)
    return "\n".join(lines) + ("\n" if text.endswith("\n") or not text else "")


def rewrite_cue(original: str, style: str, start_s: float, end_s: float, text: str) -> str:
    """Replace style/timing/text of one Dialogue line, preserving the rest."""
    m = DIALOGUE_RE.match(original)
    if not m:
        return original
    fields = [
        m.group(1),
        s_to_ts(start_s),
        s_to_ts(end_s),
        style,
        m.group(5),
        m.group(6),
        m.group(7),
        m.group(8),
        m.group(9),
        text,
    ]
    return "Dialogue: " + ",".join(fields)
