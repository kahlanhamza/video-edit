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
