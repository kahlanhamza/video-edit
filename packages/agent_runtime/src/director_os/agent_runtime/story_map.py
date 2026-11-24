"""Story Map — the whole raw footage, condensed for one LLM context window.

The Eustis failure mode this kills: the Director only ever saw the first
slice of the timeline (2 `trim_clip` calls, 59s master, story arc dropped).
Before any Director call, `build_story_map()` condenses the ENTIRE raw video
(`00:00` -> end) into a compact chronological index of 1-minute buckets that
merge, per bucket:

- speech highlights  — the highest-signal utterances (longest / most loaded),
  trimmed to a token budget, with speaker labels
- visual context     — shot density + optional VLM scene tags
- audio events       — non-speech sound events (sirens, shouts, doors)

A 45-minute video lands at ~2,500 tokens, so the LLM sees the ENTIRE arrest
from the opening stop to the booking room without blowing context or TPM
limits (Groq 8K TPM included).
"""

from __future__ import annotations

from dataclasses import dataclass, field

BUCKET_S = 60.0
MAX_HIGHLIGHT_WORDS = 28  # per bucket utterance cap (token budget)
MAX_HIGHLIGHTS_PER_BUCKET = 3
SPEECH_CHARS_PER_BUCKET = 360  # hard ceiling so 45min ~= 2.5k tokens
TAGS_PER_BUCKET = 3


@dataclass
class StoryBucket:
    """One minute of the raw footage, condensed."""

    index: int
    start_s: float
    end_s: float
    speech: list[str] = field(default_factory=list)  # "Speaker: text"
    speakers: list[str] = field(default_factory=list)
    audio_events: list[str] = field(default_factory=list)
    visual_tags: list[str] = field(default_factory=list)
    shots: int = 0

    @property
    def has_signal(self) -> bool:
        return bool(self.speech or self.audio_events or self.visual_tags or self.shots)

    def render(self) -> str:
        parts = [f"[{self.start_s / 60:02.0f}:{self.start_s % 60:02.0f}-{self.end_s % 60:02.0f}]"]
        if self.speakers:
            parts.append("(" + ",".join(self.speakers[:4]) + ")")
        if self.visual_tags:
            parts.append("<" + " ".join(self.visual_tags[:TAGS_PER_BUCKET]) + ">")
        if self.shots:
            parts.append(f"{self.shots}cuts")
        if self.audio_events:
            parts.append("~" + " ".join(self.audio_events[:3]))
        head = " ".join(parts)
        if not self.speech:
            return head
        return head + "\n  " + "\n  ".join(self.speech)


def _bucket_index(t_s: float) -> int:
    return max(0, int(t_s // BUCKET_S))


def _split_speakers(transcript: dict) -> dict[str, list]:
    """Group diarized segments by speaker -> [(start, end, text), ...]."""
    out: dict[str, list] = {}
    for seg in transcript.get("segments", []) or []:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        out.setdefault(str(seg.get("speaker") or "Unknown"), []).append(
            (float(seg.get("start_s", 0.0)), float(seg.get("end_s", 0.0)), text)
        )
    return out


def build_story_map(
    transcript: dict,
    source_duration_s: float,
    shots: list[tuple[float, float]] | None = None,
    audio_events: list[dict] | None = None,
    vision_insights: list[dict] | None = None,
    bucket_s: float = BUCKET_S,
) -> list[StoryBucket]:
    """Condense the full raw footage into chronological story buckets.

    All inputs are the persisted perception JSON shapes (transcript_json,
    shots_json.shots, audio_events_json.events, SceneInsight dicts) so the
    pipeline can call this straight off `perception_index` rows.
    """
    n_buckets = max(1, -(-source_duration_s // bucket_s))  # ceil: no spurious tail bucket
    n_buckets = int(n_buckets)
    buckets = [
        StoryBucket(index=i, start_s=i * bucket_s, end_s=(i + 1) * bucket_s)
        for i in range(n_buckets)
    ]

    # --- speech highlights: per bucket, keep the longest utterances (they
    # carry the narrative: admissions, contradictions, Miranda text)
    char_budget = [SPEECH_CHARS_PER_BUCKET] * n_buckets
    speaker_map = _split_speakers(transcript)
    per_bucket: list[list[tuple[float, str]]] = [[] for _ in range(n_buckets)]
    for speaker, utts in speaker_map.items():
        for start, _end, text in utts:
            if not text.strip():
                continue
            if len(text.split()) > MAX_HIGHLIGHT_WORDS:
                text = " ".join(text.split()[:MAX_HIGHLIGHT_WORDS]) + "…"
            idx = min(_bucket_index(start), n_buckets - 1)
            per_bucket[idx].append((start, f"{speaker}: {text}"))
            if speaker not in buckets[idx].speakers:
                buckets[idx].speakers.append(speaker)
    for idx, items in enumerate(per_bucket):
        items.sort(key=lambda p: -len(p[1]))  # longest first = most signal
        for _t, line in items[:MAX_HIGHLIGHTS_PER_BUCKET]:
            if char_budget[idx] - len(line) < 0:
                break
            buckets[idx].speech.append(line)
            char_budget[idx] -= len(line) + 2

    # --- visual: shot density + VLM tags
    for start, _end in shots or []:
        idx = min(_bucket_index(start), n_buckets - 1)
        buckets[idx].shots += 1
    for ins in vision_insights or []:
        t = float(ins.get("timestamp_s", 0.0))
        desc = str(ins.get("description", "")).strip()
        idx = min(_bucket_index(t), n_buckets - 1)
        if desc and len(buckets[idx].visual_tags) < TAGS_PER_BUCKET:
            tag = " ".join(desc.split()[:6])
            buckets[idx].visual_tags.append(tag)

    # --- audio events (non-speech sound)
    for ev in audio_events or []:
        start = float(ev.get("start_s", 0.0))
        kind = str(ev.get("kind", "sound")).strip() or "sound"
        idx = min(_bucket_index(start), n_buckets - 1)
        if len(buckets[idx].audio_events) < 3:
            buckets[idx].audio_events.append(kind)
    return buckets


def render_story_map(buckets: list[StoryBucket]) -> str:
    """The compact chronological index the Director LLM reads (Phase A input)."""
    lines: list[str] = []
    for b in buckets:
        if b.has_signal:
            lines.append(b.render())
    if not lines:
        return "(no perception signal — treat the whole runtime as one act)"
    return "\n".join(lines)


def story_map_prompt(buckets: list[StoryBucket], source_duration_s: float) -> str:
    """Framed story index for the Director prompt (Phase A)."""
    return (
        f"## Full-timeline Story Index ({source_duration_s / 60:.0f} min raw, "
        "1-minute buckets: [mm:ss] (speakers) <visual tags> cuts ~audio; speech lines below)\n"
        + render_story_map(buckets)
    )


# ---------------------------------------------------------------- full dialogue
TURN_MERGE_GAP_S = 1.8  # same-speaker utterances closer than this are one turn
TURN_MAX_CHARS = 240
FULL_TRANSCRIPT_CHAR_BUDGET = 60_000  # ~= 15k tokens — every word of a 45-min tape


def build_transcript_turns(transcript: dict) -> list[dict]:
    """The notebook's condensed FULL transcript — every speaker turn, timestamped.

    The 1-minute story index caps speech at 360 chars/bucket: the LLM sees
    HIGHLIGHTS, not the dialogue, and writes generic commentary. The manual
    sandbox workflow (the owner's notebook) feeds the WHOLE condensed
    transcript — utterances merged into turns (same speaker, gap <= 1.8s,
    <= 240 chars each) with `[mm:ss-mm:ss] (s.s-e.s) Speaker: text` lines —
    which is what makes specific, contradiction-quoting narration possible.

    Segments are preferred (diarized, sentence-shaped); words are the fallback.
    """
    turns: list[dict] = []
    if transcript.get("segments"):
        for seg in transcript.get("segments", []):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            spk = f"S{seg.get('speaker', '?')}" if seg.get("speaker") is not None else "S?"
            turns.append(
                {
                    "spk": spk,
                    "start": float(seg.get("start_s", 0.0)),
                    "end": float(seg.get("end_s", 0.0)),
                    "text": text,
                }
            )
    else:
        words = [w for w in transcript.get("words", []) if w.get("text")]
        for w in words:
            spk = f"S{w.get('speaker', '?')}" if w.get("speaker") is not None else "S?"
            if (
                turns
                and turns[-1]["spk"] == spk
                and float(w.get("start_s", 0.0)) - turns[-1]["end"] <= TURN_MERGE_GAP_S
                and len(turns[-1]["text"]) < TURN_MAX_CHARS
            ):
                turns[-1]["end"] = float(w.get("end_s", 0.0))
                turns[-1]["text"] += " " + str(w["text"]).strip()
            else:
                turns.append(
                    {
                        "spk": spk,
                        "start": float(w.get("start_s", 0.0)),
                        "end": float(w.get("end_s", 0.0)),
                        "text": str(w["text"]).strip(),
                    }
                )

    # split oversized turns (a 240-char cap keeps lines readable)
    out: list[dict] = []
    for t in turns:
        if len(t["text"]) <= TURN_MAX_CHARS:
            out.append(t)
            continue
        words = t["text"].split()
        cur: list[str] = []
        cur_chars = 0
        for tok in words:
            if cur and cur_chars + len(tok) + 1 > TURN_MAX_CHARS:
                out.append({**t, "text": " ".join(cur)})
                cur, cur_chars = [], 0
            cur.append(tok)
            cur_chars += len(tok) + 1
        if cur:
            out.append({**t, "text": " ".join(cur)})
    return out


def render_full_transcript(transcript: dict, char_budget: int = FULL_TRANSCRIPT_CHAR_BUDGET) -> str:
    """The timestamped full-dialogue block the Director prompt reads."""
    turns = build_transcript_turns(transcript)
    if not turns:
        return ""
    lines = []
    used = 0
    for t in turns:
        line = (
            f"[{t['start'] / 60:02.0f}:{t['start'] % 60:02.0f}-"
            f"{t['end'] / 60:02.0f}:{t['end'] % 60:02.0f}] "
            f"({t['start']:.1f}-{t['end']:.1f}s) {t['spk']}: {t['text']}"
        )
        if used + len(line) > char_budget:
            lines.append("…(older dialogue truncated at the token budget)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
