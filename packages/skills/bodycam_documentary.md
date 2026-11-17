# Skill: bodycam_documentary (v0.1 draft)

Dynamic skill injection target for the Director Agent (PLAN §3.1.4). Loaded as system
instructions when a project's `niche_skill == "bodycam_documentary"`.

## Editorial pacing

- Hook in the first 12 seconds: the most dramatic *moving* moment, raw audio on.
- Dead air is cut with speech + audio-event + motion intervals (guardrail 4), leaving
  `pad_before_s=0.6` / `pad_after_s=0.8` breathing room, `min_gap_s=1.8`.
- Narration rides muted slow-mo or freeze-callouts over the exact semantic moment it
  describes — never over unrelated b-roll.

## Audio law (enforced by tools)

- Levels are audited on **spoken-word intervals only** (ASR anchors) — sirens, door
  slams and cruisers skew whole-program loudnorm. Dialogue rides −12 dB … −6 dB RMS;
  master ceiling −1.5 dBTP (`speech_master_chain`).
- Bystander music flagged by Demucs stem separation → duck, mute or cut (copyright).

## Narration voice

- Veteran true-crime podcaster talking to one viewer: conversational, contractions,
  present tense, short sentences, dry wit, zero melodrama.
- Never editorialize guilt or innocence; charges are allegations only.
- Never mention timestamps, timecodes, "seconds" or "minutes".
- TTS: one voice, `trim_pauses=True` (silenceremove both ends), target ≤ 12–20 min.

## Retention standard

- Every timeline segment is scored by the Critic Agent (Beginning → Middle → End
  momentum + open-loop tension) before render; a segment that loses the thread gets
  re-cut or dropped. Story first, footage second.

## Legal & compliance (hard rules, enforced by tools, not vibes)

- Raw public-record footage only; every clip license-verified and vision-screened.
- Mandatory disclaimer in description; "presumed innocent" end card.
- Bystander faces, minors, plates, IDs → `redact.add_blur_region` before render.
- No advertiser-hostile content policy in skill scope: mask profanity in captions
  (`profanity_mode="mask"`), flag graphic moments for the Human Review Gate.

## Caption & card style

- 2-row speaker-colored `.ass`, 3–5 word beats, karaoke active-word highlight.
- Officer → Lane 1 (`MarginV=46`), Suspect/Other → Lane 2 (`MarginV=96`).
- Cards: `location_intro` (agency | location), `case_outcome` (dispositions),
  `evidence_quote`, `evidence_fact` (mid-frame fact card: BAC values, policy
  violations, dispatch quotes with the timeline anchor it cites) — collision-free
  corner placement, 30–50 ms fades, never inside the YouTube 16:9 player safe zone.
