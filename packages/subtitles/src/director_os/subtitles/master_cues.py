"""Master-clock caption cues — the committed v6 law "caption truth on the master clock".

Captions are timed on the EDITED TIMELINE (what the viewer sees), never on the
raw source clock. The v5 regression the v6 commit fixed was exactly this: a cue
list built off the 43-minute source pasted onto the ~10-minute master — raw
dialogue captions landing over narration b-roll because both were read at
source-time. The law, now in one shared implementation:

- NARRATOR cues come from the TTS word sidecars (the voiceover NEVER comes from
  ASR), rebased onto the master clock per narration block.
- RAW speech cues come from the proofread transcript, remapped through each kept
  raw clip's source->master window (source start + clip offset).
- Narration blocks (b-roll visuals, muted bed) NEVER carry raw captions —
  the narrator-on-broll isolation gate.
- Cue chunking rides the caption_v4 law (`chunk_cue_words`: <=5 words /
  <=27 chars / >1s gap breaks).

`post_fx` burns exactly these cues on the deliverable, and `stage_subtitles`
persists exactly these cues into the Studio cue store (subtitles.{json,ass,srt})
— one truth, three consumers (grid edits, program monitor, final burn).
"""

from __future__ import annotations

from director_os.subtitles.ass_builder import CaptionBeat, WordTiming, chunk_cue_words

MAX_CUE_SPAN_S = 6.0  # a raw-speech cue never rides longer than this


def narration_word_stamps(narration_clips: list[dict]) -> dict[str, list[dict]]:
    """script_text -> word stamps from the persisted TTS sidecars.

    Falls back to an even distribution over the clip duration when a sidecar is
    missing (TTS speech is near-constant rate) — narration captions must never
    silently vanish because a JSON file went missing.
    """
    import json as _json
    from pathlib import Path

    stamps: dict[str, list[dict]] = {}
    for clip in narration_clips or []:
        sidecar = Path(clip.get("sidecar") or "")
        script = str(clip.get("script_text") or "")
        words: list[dict] = []
        if sidecar.exists():
            try:
                words = list(_json.loads(sidecar.read_text()).get("words") or [])
            except (ValueError, OSError):
                words = []
        if not words and script:
            words = _even_word_distribution(script, float(clip.get("duration_s") or 0.0))
        if words and script:
            stamps[script] = words
    return stamps


def _even_word_distribution(script: str, duration_s: float) -> list[dict]:
    tokens = script.split()
    if not tokens or duration_s <= 0:
        return []
    per = duration_s / len(tokens)
    return [
        {
            "word": tok,
            "start_s": round(i * per, 3),
            "end_s": round(min(duration_s, (i + 1) * per), 3),
        }
        for i, tok in enumerate(tokens)
    ]


def build_master_caption_cues(
    timeline,
    transcript: dict,
    narration_words: dict[str, list[dict]],
) -> tuple[list[CaptionBeat], dict[str, int]]:
    """Master-clock caption beats: NARRATOR (TTS stamps) + raw speech (remapped).

    narration_words maps script_text -> [{"word","start_s","end_s"}] from the
    persisted TTS sidecars (already re-based onto the trimmed stem by
    generate_tts). Raw-speech words come from the transcript (source clock)
    and are remapped through each kept raw clip's window; narration blocks
    never receive raw cues (the bed is muted there — narrator-on-broll
    isolation).
    """
    from director_os.otio_engine.compiler import master_track
    from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta, parse_clip_meta
    from director_os.otio_engine.timeline import is_gap, item_duration_s

    # proofread words carry documentary roles + clean text; fall back to raw ASR.
    # Proofread-masked words (text="", the supervisor's garble decision) STAY in
    # the stream — they are the caption-truth tag signal and must render as
    # *inaudible* instead of vanishing (runbook §2.1: tagged, never dropped).
    proof = (transcript or {}).get("proofread") or {}
    proof_masked = bool(proof.get("words"))
    words_src = list(proof.get("words", [])) or [
        w for w in (transcript or {}).get("words", []) if w.get("text")
    ]
    transcript_words: list[dict] = []
    for w in words_src:
        item = dict(w)
        if not item.get("text") and item.get("word"):
            item["text"] = item["word"]
        transcript_words.append(item)

    beats: list[CaptionBeat] = []

    def _beats_from_chunks(chunks: list[list[dict]], speaker_default: str) -> int:
        n = 0
        for chunk in chunks:
            if not chunk:
                continue
            spk = str(chunk[0].get("speaker") or speaker_default)
            start_ms = int(float(chunk[0]["m_start"]) * 1000)
            end_ms = int(float(chunk[-1]["m_end"]) * 1000)
            if end_ms <= start_ms:
                continue
            beats.append(
                CaptionBeat(
                    text=" ".join(w["text"] for w in chunk),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker=spk,
                    words=[
                        WordTiming(
                            word=w["text"],
                            start_ms=int(float(w["m_start"]) * 1000),
                            end_ms=int(float(w["m_end"]) * 1000),
                        )
                        for w in chunk
                    ],
                )
            )
            n += 1
        return n

    narr_cues = raw_cues = 0

    track = master_track(timeline)
    t_master = 0.0
    for item in track:
        dur = item_duration_s(item)
        if dur <= 0:
            continue
        if is_gap(item):
            t_master += dur
            continue
        meta = parse_clip_meta(item)
        sr = getattr(item, "source_range", None)
        src0 = sr.start_time.to_seconds() if sr is not None else 0.0

        if isinstance(meta, NarratorClipMeta):
            # NARRATOR ONLY — the b-roll bed is muted; raw captions never land here.
            # An A1 audio trim rides atrim(vo_in:vo_out) on the stem (renderer law),
            # so the word stamps (full-stem clock) shift by vo_in inside the block.
            vo_offset = float(getattr(meta, "vo_in_s", 0.0) or 0.0)
            words = narration_words.get(meta.script_text, [])
            stampeds: list[dict] = []
            for w in words:
                w_start = float(w.get("start_s", 0.0))
                w_end = float(w.get("end_s", 0.0))
                if w_end <= w_start:
                    continue
                b_start = w_start - vo_offset
                b_end = w_end - vo_offset
                if b_end <= 0 or b_start >= dur:
                    continue
                stampeds.append(
                    {
                        "text": str(w.get("word", "")).strip(),
                        "speaker": "Narrator",
                        "m_start": t_master + max(0.0, b_start),
                        "m_end": min(t_master + dur, t_master + b_end),
                    }
                )
            stampeds = [w for w in stampeds if w["m_end"] > w["m_start"] and w["text"]]
            chunks = chunk_cue_words(stampeds)
            narr_cues += _beats_from_chunks(chunks, "Narrator")
        elif isinstance(meta, RawClipMeta):
            # remap proofread/ASR words inside [src0, src0+dur) onto the master clock
            stampeds2: list[dict] = []
            for w in transcript_words:
                text = str(w.get("text") or w.get("word") or "").strip()
                if not text:
                    if not proof_masked:
                        continue
                    text = "*inaudible*"
                ws, we = float(w.get("start_s", 0.0)), float(w.get("end_s", 0.0))
                if we <= src0 or ws >= src0 + dur:
                    continue
                m_start = t_master + max(0.0, ws - src0)
                m_end = t_master + min(dur, we - src0)
                if m_end - m_start < 0.05:
                    continue
                stampeds2.append(
                    {
                        "text": text,
                        "speaker": str(w.get("speaker") or "Unknown"),
                        "m_start": m_start,
                        "m_end": m_end,
                    }
                )
            chunks = chunk_cue_words(stampeds2)
            raw_cues += _beats_from_chunks(chunks, "Unknown")
        t_master += dur

    beats.sort(key=lambda b: b.start_ms)
    return beats, {"narration": narr_cues, "raw": raw_cues, "total": len(beats)}
