#!/usr/bin/env python3
"""tts_norm — TTS text normalization via the editor AI agent (LLM), NOT regex.

LAW (owner directive 2026-11-21): normalization must NOT be a pile of
per-case regex rules. director-os is an editor AI agent: it must handle ANY
future content — units, ratios, currency, chemical formulas, URLs, dates,
math, radio codes, any unit system — by UNDERSTANDING the line, not by
pattern-matching one case at a time. So the law here is:

  1. The LLM rewrites the line into exactly what a human narrator would SAY.
     The prompt teaches the PRINCIPLE with generic cross-domain examples —
     no case-specific patterns. "0.391g/210L" -> "0.391 grams per 210
     liters" is an OUTCOME of the agent, true for any unit system.
  2. A validator keeps the agent honest: no meta prose, no paraphrase-length
     drift, and NUMBERS are invariant (a normalization that changes digits
     changes facts). Failures fall back to a tiny content-agnostic symbol
     pass that never rewrites words (% -> percent, $ amounts, &, <, >).
  3. Results are cached per input hash — the same line never re-bills.

Shared by the narration planner (pipe_03), the package TTS engine
(audio_engine.tts) and the voice patch (postfx_v3), so every future video
inherits the fix regardless of which entry point synthesized the voice.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

_CACHE_PATH = Path("/home/z/my-project/pipeline_run/tts_norm_cache.json")
_BRIDGE_DIR = "/home/z/my-project/scripts"

_PROMPT = """You are the text-normalization stage of a documentary narration pipeline (a text-to-speech front-end).

Task: rewrite the narration line into EXACTLY what a professional narrator would SAY aloud, so the TTS voice never reads tokens letter-by-letter (like "gee slash two ten el").

Transform only what a human would pronounce differently from how it is written, for ANY domain and ANY unit system — do not limit yourself to these illustrative cases:
- unit ratios: "0.391g/210L" -> "0.391 grams per 210 liters"; "3.2g/5L" -> "3.2 grams per 5 liters"; likewise mg/dL, km/h, mph, lb/ft, µg/m³, any combination
- units: "45 mph" -> "45 miles per hour"; "12 km" -> "12 kilometers"; "98.6°F" -> "98.6 degrees Fahrenheit"
- percent / currency: "0.08%" -> "0.08 percent"; "$1,250" -> "1,250 dollars"; "€5" -> "5 euros"
- math symbols: "&" -> "and"; "+" -> "plus"; "=" -> "equals"; "<" -> "less than"; ">" -> "more than"
- formulas read letter-by-letter: "H2O" -> "H two O"; "CO2" -> "C O two"
- acronyms that ARE spoken as letters (BAC, DUI, SFST, DNA) stay as-is
- serial-ish identifiers ("Unit 12", "Case No. 4471", "I-4") stay natural — TTS reads digits and simple hyphens fine

Hard rules:
- Do NOT paraphrase, summarize, translate, add commentary, or drop any information.
- Keep every word you do not transform untouched, in the same order.
- NUMBERS are invariant: digits stay digits, same values, same order ("$1,250" -> "1,250 dollars" keeps 1250; never spell numbers as words).
- Keep the input's language.
- Output ONLY the rewritten line — no quotes, no explanations, no markdown.
"""

# emergency-only, content-agnostic symbol pass (never rewrites words).
# There are deliberately NO unit tables here: units are the agent's job.
_FALLBACK: list[tuple[str, object]] = [
    (r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)", r"\1 dollars"),
    (r"([0-9](?:\.[0-9]+)?)\s*%", r"\1 percent"),
    (r"\s*&\s*", " and "),
    (r"\s*<\s*", " less than "),
    (r"\s*>\s*", " more than "),
]

_META_HINTS = ("here is", "rewritten", "normalized:", "output:", "sure,", "as an ai")

# Generic DETECTOR (never a replacer): did the agent leave an unspeakable
# unit-ratio-shaped token behind ("391g/210L", "mg/dL", "km/h")? A letter-run
# glued to a slash-bound token with digits. Used only to trigger one retry —
# the LLM still does 100% of the solving.
_UNSPOKEN_RATIO = re.compile(
    r"[a-zA-Z]{1,5}\s*/\s*(?:[a-zA-Z]{1,5}|\d+(?:\.\d+)?[a-zA-Z]{1,5})"
)


def _missed_ratios(src: str, out: str) -> bool:
    """True when the SOURCE had ratio-shaped tokens the output still carries."""
    src_ratios = set(_UNSPOKEN_RATIO.findall(src))
    if not src_ratios:
        return False
    return any(r in out for r in src_ratios)


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, indent=0))
    except Exception:  # noqa: BLE001
        pass


def _clean(text: str) -> str:
    """Strip fences / quotes / meta prose; keep the first content line."""
    t = text.strip()
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t).strip()
    line = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
    line = line.strip().strip('"').strip("'").strip()
    return line


def _digit_groups(s: str) -> list[str]:
    """Standalone numbers only: digit-runs NOT immediately preceded by a
    letter. Unit-glued numbers ("0.391g", "45mph") are enforced; digits
    inside formula blends ("H2O" -> "H two O") may legitimately become
    words, so they are exempt. Commas stripped ("1,250" == "1250")."""
    s = s.replace(",", "")
    return sorted(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", s))


def _validate(src: str, out: str) -> bool:
    """Reject agent drift: meta prose, empty, length blowup, digit change,
    script switch. Words may change ONLY where the principle demands it."""
    if not out or len(out) < 2:
        return False
    low = out.lower()
    if any(h in low for h in _META_HINTS) or "```" in out:
        return False
    s_len, o_len = len(src.strip()), len(out.strip())
    if o_len > s_len * 2.6 or o_len < s_len * 0.45:
        return False
    if _digit_groups(src) != _digit_groups(out):
        return False  # a normalization that changes digits changes FACTS
    # script switch guard: ASCII-only source must not come back CJK/Cyrillic
    if src.isascii() and not out.isascii():
        return False
    return True


def _agent_normalize(text: str, context: str | None) -> str | None:
    sys.path.insert(0, _BRIDGE_DIR)
    try:
        from llm_bridge import llm_chat
    except Exception:  # noqa: BLE001
        return None
    prompt = _PROMPT
    if context:
        prompt += f"\nContext (never mention it): {context}\n"
    prompt += f"\nLine:\n{text}"
    best: str | None = None
    for attempt in range(3):
        try:
            txt = llm_chat(prompt, timeout=120)
        except Exception:  # noqa: BLE001
            continue
        cand = _clean(txt)
        if not cand or not _validate(text, cand):
            continue
        best = cand
        if not _missed_ratios(text, cand):
            return cand
        # the agent left unit ratios unspeakable — one strict retry
        prompt = (_PROMPT + "\nYour previous answer still contains unspeakable "
                  "slash unit tokens. Expand EVERY unit ratio into spoken words "
                  f"(any unit system). Line:\n{text}")
    return best


def _fallback(text: str) -> str:
    out = text
    for pat, rep in _FALLBACK:
        out = re.sub(pat, rep, out)
    return re.sub(r"\s{2,}", " ", out).strip()


def normalize(text: str, context: str | None = None) -> str:
    """Narration line -> speakable line. Agent-first, validator-guarded,
    cached; tiny generic fallback when the agent is unavailable."""
    if not text or not text.strip():
        return text
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    cache = _load_cache()
    hit = cache.get(key)
    if hit:
        return hit["out"]
    out = _agent_normalize(text, context)
    via = "agent"
    if out is None:
        out = _fallback(text)
        via = "fallback"
    cache[key] = {"out": out, "via": via}
    _save_cache(cache)
    return out


if __name__ == "__main__":
    samples = [
        "His blood alcohol content was later confirmed at a staggering 0.391g/210L, "
        "nearly five times the legal limit.",
        "The IV bag held 3.2g/5L of saline. He blew 0.08% BAC and was going 45 mph "
        "in a 30 km/h zone, fined $1,250, rated 5x over the limit.",
        "The lab report listed 12 mg/dL and a compound close to H2O, filed as Case No. 4471.",
        "Officers cleared the scene at 10-8, and the dashboard read 98.6°F.",
    ]
    for s in samples:
        print("IN :", s)
        print("OUT:", normalize(s, context="documentary narration"))
        print()
