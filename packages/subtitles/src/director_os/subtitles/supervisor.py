"""LLM proofread supervisor — the sandbox workflow the owner ran by hand.

The deterministic proofread (jargon map + word-count roles) is the always-
available floor. This module adds the LLM pass the manual Kaggle workflow used
(notebook cell "Proofreading kept bodycam lines", and the v6 committed
"LLM operator supervisor pass (decisions_v6.json)"):

1. the transcript is grouped into readable LINES (speaker turns, gap <= 1.8s
   merged, each capped at 240 chars — the notebook's turn law),
2. lines go to the LLM in batches of 70 with the case facts, and it returns
   `[{"id": <int>, "role": "Officer"|"Suspect", "clean_text": "<text>"}]`:
   - fix obvious ASR mishearings from context (police codes, names, agencies),
   - set `clean_text` to "" for pure garble / unverified hallucination,
   - mask harsh profanity for YouTube monetization,
   - assign the TRUE role even when raw diarization swapped speakers,
3. every correction is WORD-LOCKED back onto the original ASR word timestamps
   through `align_words` (difflib) — sync cannot drift (the v6 word-lock law),
4. per-line roles vote into a per-speaker role map (raw label -> Officer/
   Suspect), replacing the word-count guess wherever LLM evidence exists.

Failure law: no LLM, a dead backend, or a malformed batch NEVER kills the
proofread — the deterministic floor rides on, loudly logged.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from director_os.subtitles.align import RawWord, align_words

SUPERVISOR_VERSION = "v2-supervised"
BATCH_LINES = 70
TURN_MERGE_GAP_S = 1.8
TURN_MAX_CHARS = 240
MAX_GARBLED_LINES = 4000  # hard ceiling — a transcript that is ALL garble still completes

SUPERVISOR_SYSTEM = (
    "You are a caption supervisor for a police bodycam documentary channel. "
    "You return ONLY valid JSON arrays — no prose, no markdown fences."
)


@dataclass
class TranscriptLine:
    """One readable speaker turn with its original word timings."""

    id: int
    start_s: float
    end_s: float
    speaker: str
    text: str
    words: list[dict] = field(default_factory=list)  # {text, start_s, end_s}


def build_turns(transcript: dict) -> list[TranscriptLine]:
    """Group the transcript into speaker turns (the notebook turn law).

    Segments (diarized, sentence-shaped) are preferred; word timings are the
    fallback. Utterances closer than TURN_MERGE_GAP_S from the same speaker
    merge; a turn never exceeds TURN_MAX_CHARS.
    """
    raw_words = [w for w in transcript.get("words", []) if w.get("text")]
    if not raw_words:
        return []

    # pseudo-utterances: same-speaker word runs broken at gaps > TURN_MERGE_GAP_S
    utterances: list[list[dict]] = []
    for w in raw_words:
        if utterances:
            prev = utterances[-1][-1]
            same = str(w.get("speaker") or "") == str(prev.get("speaker") or "")
            gap = float(w.get("start_s", 0.0)) - float(prev.get("end_s", 0.0))
            if same and gap <= TURN_MERGE_GAP_S:
                utterances[-1].append(w)
                continue
        utterances.append([w])

    lines: list[TranscriptLine] = []
    for utt in utterances:
        spk = str(utt[0].get("speaker") or "Unknown")
        text = " ".join(str(w["text"]).strip() for w in utt).strip()
        if not text:
            continue
        # split oversized utterances into <= TURN_MAX_CHARS turns
        chunks: list[list[dict]] = []
        cur: list[dict] = []
        cur_chars = 0
        for w in utt:
            wl = len(str(w["text"])) + 1
            if cur and cur_chars + wl > TURN_MAX_CHARS:
                chunks.append(cur)
                cur, cur_chars = [], 0
            cur.append(w)
            cur_chars += wl
        if cur:
            chunks.append(cur)
        for chunk in chunks:
            chunk_text = " ".join(str(w["text"]).strip() for w in chunk).strip()
            if not chunk_text:
                continue
            lines.append(
                TranscriptLine(
                    id=len(lines),
                    start_s=float(chunk[0].get("start_s", 0.0)),
                    end_s=float(chunk[-1].get("end_s", 0.0)),
                    speaker=spk,
                    text=chunk_text,
                    words=[
                        {
                            "text": str(w["text"]).strip(),
                            "start_s": float(w.get("start_s", 0.0)),
                            "end_s": float(w.get("end_s", 0.0)),
                        }
                        for w in chunk
                    ],
                )
            )
    return lines


def _extract_json_array(text: str) -> list[dict]:
    """Lenient parse: strip <think> blocks and fences, take the JSON array."""
    import re

    cleaned = re.sub(r"<(?:think|thought)>[\s\S]*?</(?:think|thought)>", "", text or "")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip())
    a, b = cleaned.find("["), cleaned.rfind("]")
    if a == -1 or b == -1 or b <= a:
        raise ValueError("no JSON array in supervisor response")
    data = json.loads(cleaned[a : b + 1])
    if not isinstance(data, list):
        raise ValueError("supervisor payload is not a list")
    return [item for item in data if isinstance(item, dict)]


def _supervise_batch(
    lines: list[TranscriptLine],
    case_facts: dict | None,
    complete_fn,
) -> list[dict]:
    """One batch -> decision list (the notebook prompt, verbatim structure)."""
    payload = [
        {"id": ln.id, "t": round(ln.start_s, 1), "speaker": ln.speaker, "text": ln.text}
        for ln in lines
    ]
    facts = json.dumps(case_facts, ensure_ascii=False) if case_facts else "not available"
    user = f"""Proofread these raw ASR captions for a police bodycam documentary.
Case facts: {facts}

Return ONLY a valid JSON array matching every input `id` (no prose, no code fences):
[{{"id": <int>, "role": "Officer" or "Suspect", "clean_text": "<text>"}}]

Rules:
- Fix obvious ASR mishearings from context (police codes like 10-4, 10-8,
  10-15, Signal Zero, Standard Field Sobriety; names/cities/agencies in Case
  facts). Set clean_text to "" if pure garble or unverified hallucination.
- Assign the true "Officer" (asking questions, giving orders, dispatch, EMS)
  vs "Suspect" (the arrested/detained person) even if the raw speaker labels
  swapped.
- Mask harsh profanity (e.g. "f***", "s***") for YouTube monetization.
- Keep the spoken words; never invent content that is not in the line.

Lines:
{json.dumps(payload, ensure_ascii=False)}"""
    last_err: Exception | None = None
    for _attempt in range(2):
        try:
            raw = complete_fn(SUPERVISOR_SYSTEM, user)
            decisions = _extract_json_array(raw)
            by_id = {ln.id: ln for ln in lines}
            out: list[dict] = []
            for item in decisions:
                raw_id = item.get("id")
                if raw_id is None:
                    continue
                try:
                    lid = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if lid not in by_id:
                    continue
                role = str(item.get("role") or "").strip().capitalize()
                clean = str(item.get("clean_text") or "").strip()
                out.append({"id": lid, "role": role, "clean_text": clean})
            if out:
                return out
            last_err = ValueError("empty decision list")
        except Exception as e:  # noqa: BLE001 — one bad batch must never kill the pass
            last_err = e
    if last_err is not None:
        raise ValueError(f"supervisor batch failed after retry: {last_err}") from last_err
    return []


def supervise_transcript(
    transcript: dict,
    complete_fn,
    case_facts: dict | None = None,
    log=None,
) -> dict | None:
    """Run the LLM supervisor over the whole transcript; None = keep the floor.

    `complete_fn(system, user) -> str` is the routing-chain completion callable.
    Returns the proofread payload in the SAME shape as `prepare_transcript`
    (words / corrections / speaker_roles / version) so every downstream
    consumer reads it unchanged.
    """
    if complete_fn is None:
        return None
    lines = build_turns(transcript)
    if not lines:
        return None

    decisions: dict[int, dict] = {}
    failed_batches = 0
    for i in range(0, min(len(lines), MAX_GARBLED_LINES), BATCH_LINES):
        batch = lines[i : i + BATCH_LINES]
        try:
            for item in _supervise_batch(batch, case_facts, complete_fn):
                decisions[item["id"]] = item
        except ValueError as e:
            failed_batches += 1
            if log is not None:
                log(f"  [supervisor] batch {i // BATCH_LINES + 1} skipped ({e})")

    if not decisions and failed_batches >= 1:
        if log is not None:
            log("  [supervisor] no usable LLM decisions — deterministic floor rides on")
        return None

    by_id = {ln.id: ln for ln in lines}
    word_changes = 0
    garbled_lines = 0
    role_votes: Counter[tuple[str, str]] = Counter()  # (raw speaker, role) -> votes
    corrections: list[dict] = []

    # pass 1: word-level text/role application (word-locked)
    for lid, dec in sorted(decisions.items()):
        ln = by_id[lid]
        clean = dec.get("clean_text", "")
        role = dec.get("role") or ""
        if role in {"Officer", "Suspect", "Narrator"}:
            role_votes[(ln.speaker, role)] += 1
        if not clean:
            garbled_lines += 1
            continue
        if clean != ln.text:
            corrections.append(
                {"before": ln.text, "after": clean, "at_line": lid, "at_s": round(ln.start_s, 2)}
            )
        # word-lock: realign the corrected text onto the ORIGINAL word timings
        raw_words = [
            RawWord(
                word=str(w["text"]),
                start_ms=int(w["start_s"] * 1000),
                end_ms=int(w["end_s"] * 1000),
            )
            for w in ln.words
        ]
        if not raw_words:
            continue
        try:
            aligned = align_words(clean, raw_words)
        except ValueError:
            continue
        for w, aw in zip(ln.words, aligned, strict=False):
            if str(w["text"]) != aw.word:
                w["text"] = aw.word
                word_changes += 1

    # pass 2: write the word stream back (order preserved; garbled lines keep
    # their timings but drop their text so captions never show garble)
    out_words: list[dict] = []
    for ln in lines:
        dec2: dict | None = decisions.get(ln.id)
        garbled = bool(dec2 and not dec2.get("clean_text", ""))
        role = (dec2.get("role") or "") if dec2 else ""
        for w in ln.words:
            entry = {
                "text": "" if garbled else str(w["text"]),
                "start_s": round(float(w["start_s"]), 3),
                "end_s": round(float(w["end_s"]), 3),
                "speaker": role if role in {"Officer", "Suspect"} else ln.speaker,
                "confidence": 1.0,
            }
            out_words.append(entry)

    # pass 3: per-raw-speaker majority role map (replaces the word-count guess)
    votes_by_speaker: dict[str, Counter[str]] = {}
    for (spk, role), n in role_votes.items():
        votes_by_speaker.setdefault(spk, Counter())[role] += n
    speaker_roles: dict[str, str] = {}
    ranked = sorted(votes_by_speaker.items(), key=lambda kv: -sum(kv[1].values()))
    taken: set[str] = set()  # each role maps to exactly ONE speaker
    for spk, votes in ranked:
        for role, _n in votes.most_common():
            if role not in taken:
                speaker_roles[spk] = role
                taken.add(role)
                break
    for w in out_words:
        mapped = speaker_roles.get(str(w["speaker"]))
        if mapped and str(w["speaker"]) not in {"Officer", "Suspect"}:
            w["speaker"] = mapped

    payload: dict = {
        "words": out_words,
        "corrections": corrections,
        "speaker_roles": speaker_roles,
        "version": SUPERVISOR_VERSION,
        "supervised": True,
        "stats": {
            "lines": len(lines),
            "supervised_lines": len(decisions),
            "failed_batches": failed_batches,
            "word_changes": word_changes,
            "garbled_lines": garbled_lines,
        },
    }
    if log is not None:
        st = payload["stats"]
        assert isinstance(st, dict)
        log(
            f"  [supervisor] {st['supervised_lines']}/{st['lines']} line(s) supervised, "
            f"{len(corrections)} correction(s), {word_changes} word edit(s), "
            f"{garbled_lines} garbled line(s) dropped, roles: "
            f"{speaker_roles or '(single speaker)'}"
        )
    return payload
