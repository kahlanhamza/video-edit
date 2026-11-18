"""Police-jargon transcript proofreading + speaker role tagging (PLAN guardrail 5).

Stage: `proofread` — runs ONCE after perception (ASR), before subtitles,
narration and Director planning. Idempotent: the product persists in
`perception_index.transcript_json["proofread"]`, so a crash-resume never
re-derives (and never re-bills) the pass.

1. `proofread_words(words)` — canonicalize the police jargon ASR mangles
   ("ten four" -> "10-4", "DL" -> "driver's license", "code six" -> "Code 6").
   Corrections change word counts, so the corrected text is re-timed through
   `align_words` (difflib, PLAN guardrail 5): equal blocks keep raw timings,
   replacements split the raw span proportionally, inserts interpolate.
   Millisecond word stamps survive the edit — never linear interpolation.
2. `tag_speakers(words)` — raw diarization labels ("S1"/"1"/"OFFICER_1"/...)
   become documentary ROLES: the speaker with the most words is `Officer`,
   the runner-up `Suspect`, everyone else `Unknown`. `Narrator` is RESERVED
   for TTS narration cues (the voiceover never comes from ASR).

The output shape is transcript-shaped (`words` with text/start_s/end_s/
speaker), so every downstream consumer — subtitle beats, speech-gated duck,
Director planning — reads it unchanged.
"""

from __future__ import annotations

import string
from collections import Counter

from director_os.subtitles.align import AlignedWord, RawWord, align_words

PROOFREAD_VERSION = "v1"

_PUNCT = str.maketrans("", "", string.punctuation)

# Multi-token ASR sequences -> canonical caption text (greedy longest-match).
JARGON_PHRASES: dict[tuple[str, ...], str] = {
    ("ten", "four"): "10-4",
    ("ten", "eight"): "10-8",
    ("ten", "fifteen"): "10-15",
    ("ten", "twenty"): "10-20",
    ("ten", "twentytwo"): "10-22",
    ("ten", "fifty", "six"): "10-56",
    ("code", "four"): "Code 4",
    ("code", "six"): "Code 6",
    ("code", "ninetynine"): "Code 99",
    ("shots", "fired"): "shots fired",
    ("pursuit", "in", "progress"): "pursuit in progress",
    ("bo", "lo"): "BOLO",
}

# Single-token ASR outputs -> canonical caption text.
JARGON_SINGLE: dict[str, str] = {
    "dl": "driver's license",
    "vin": "VIN",
    "bolo": "BOLO",
    "apb": "APB",
    "ncic": "NCIC",
    "edp": "EDP",
    "dwai": "DWAI",
    "dui": "DUI",
    "miranda": "Miranda",
    "bodycam": "bodycam",
    "bodycam?": "bodycam",
    "axon": "AXON",
    "cpr": "CPR",
    "narcan": "Narcan",
    "taser": "TASER",
    "vic": "victim",
    "rp": "reporting party",
    "911": "911",
    "104": "10-4",  # normalized keys: _norm strips punctuation ("10-4" -> "104")
    "108": "10-8",
    "1020": "10-20",
}

# Roles (documentary contract — consumers may switch on these exact strings)
ROLE_OFFICER = "Officer"
ROLE_SUSPECT = "Suspect"
ROLE_UNKNOWN = "Unknown"
ROLE_NARRATOR = "Narrator"  # reserved for TTS cues; never assigned from ASR


class ProofreadError(RuntimeError):
    pass


def _norm(token: str) -> str:
    return token.translate(_PUNCT).lower()


def _correct_tokens(tokens: list[str]) -> tuple[list[str], list[dict]]:
    """Greedy longest-match jargon pass over the raw token stream.

    Matching is case/punctuation-insensitive (`Ten, four` matches); the
    replacement re-inserts the ORIGINAL punctuation trail so sentence ends
    survive ("...ten four." -> "...10-4.").
    """
    out: list[str] = []
    corrections: list[dict] = []
    i = 0
    keys = sorted(JARGON_PHRASES, key=len, reverse=True)
    while i < len(tokens):
        trail = _punct_trail(tokens[i])
        core = tokens[i][: len(tokens[i]) - len(trail)] if trail else tokens[i]
        matched = False
        for phrase in keys:
            n = len(phrase)
            if i + n <= len(tokens) and all(_norm(tokens[i + k]) == phrase[k] for k in range(n)):
                # the punctuation trail belongs to the LAST token of the phrase
                last = tokens[i + n - 1]
                last_trail = _punct_trail(last)
                canonical = JARGON_PHRASES[phrase]
                out.append(canonical + last_trail)
                before = " ".join(tokens[i : i + n])
                if _norm(before) != _norm(canonical):
                    corrections.append({"before": before, "after": canonical, "at_token": i})
                i += n
                matched = True
                break
        if matched:
            continue
        key = _norm(core)
        single = JARGON_SINGLE.get(key) if key else None
        if single:
            out.append(single + trail)
            if single.lower() != (core + trail).lower() and _norm(core) != _norm(single):
                corrections.append({"before": core, "after": single, "at_token": i})
        else:
            out.append(core + trail)
        i += 1
    return out, corrections


def _punct_trail(token: str) -> str:
    """Trailing punctuation of a token ("four?" -> "?", "10-4" -> "")."""
    trail = ""
    while token and token[-1] in string.punctuation:
        trail = token[-1] + trail
        token = token[:-1]
    return trail


def proofread_words(words: list[dict]) -> dict:
    """Raw transcript words -> jargon-corrected words with aligned timings.

    Input words are the transcript shape ({text, start_s, end_s, speaker?}).
    Output: {"words": [...], "corrections": [...], "version": PROOFREAD_VERSION}.
    Raises ProofreadError when there are no raw timings to align against.
    """
    raw = [w for w in words if w.get("text") and w.get("end_s", 0) > w.get("start_s", 0)]
    if not raw:
        raise ProofreadError("no timed words to proofread (empty ASR?)")

    corrected, corrections = _correct_tokens([w["text"] for w in raw])
    corrected_text = " ".join(corrected)
    raw_words = [
        RawWord(word=w["text"], start_ms=int(w["start_s"] * 1000), end_ms=int(w["end_s"] * 1000))
        for w in raw
    ]
    aligned: list[AlignedWord] = align_words(corrected_text, raw_words)

    out_words: list[dict] = []
    for src, aw in zip(raw, aligned, strict=False):
        out_words.append(
            {
                "text": aw.word,
                "start_s": round(aw.start_ms / 1000, 3),
                "end_s": round(aw.end_ms / 1000, 3),
                "speaker": src.get("speaker"),
                "confidence": src.get("confidence", 1.0),
            }
        )
    # align_words can end with fewer/more entries than raw when edits shift
    # boundaries — trust the aligned list (zip(strict) guards only exactness);
    # carry any surplus raw speakers onto the tail
    while len(out_words) < len(aligned) and raw:
        aw = aligned[len(out_words)]
        out_words.append(
            {
                "text": aw.word,
                "start_s": round(aw.start_ms / 1000, 3),
                "end_s": round(aw.end_ms / 1000, 3),
                "speaker": raw[-1].get("speaker"),
                "confidence": 1.0,
            }
        )
    return {"words": out_words, "corrections": corrections, "version": PROOFREAD_VERSION}


def tag_speakers(words: list[dict]) -> dict[str, str]:
    """Raw diarization labels -> documentary roles (Officer / Suspect / Unknown).

    Rank = word count: most words -> Officer, runner-up -> Suspect. `Narrator`
    is NEVER assigned here (TTS voiceover only). Words without a speaker label
    stay unlabeled (the subtitle builder's "Unknown" default applies).
    """
    counts: Counter[str] = Counter(
        str(w["speaker"]) for w in words if w.get("speaker") not in (None, "")
    )
    if not counts:
        return {}
    ranked = [s for s, _ in counts.most_common()]
    roles: dict[str, str] = {}
    for idx, speaker in enumerate(ranked):
        if idx == 0:
            roles[speaker] = ROLE_OFFICER
        elif idx == 1:
            roles[speaker] = ROLE_SUSPECT
        else:
            roles[speaker] = ROLE_UNKNOWN
    return roles


def prepare_transcript(transcript: dict) -> dict:
    """Proofread + role-tag in one pass — the persisted `proofread` payload."""
    words = [w for w in transcript.get("words", []) if w.get("text")]
    proof = proofread_words(words)
    roles = tag_speakers(proof["words"])
    tagged: list[dict] = []
    for w in proof["words"]:
        spk = w.get("speaker")
        w = dict(w)
        if spk and spk in roles:
            w["speaker"] = roles[spk]
        tagged.append(w)
    return {
        "words": tagged,
        "corrections": proof["corrections"],
        "speaker_roles": roles,
        "version": PROOFREAD_VERSION,
    }
