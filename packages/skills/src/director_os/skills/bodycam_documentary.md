# Skill: bodycam_documentary (v0.2 — Colab-proven editorial rules)

Dynamic skill injection target for the Director Agent (PLAN §3.1.4). Loaded as system
instructions when a project's `niche_skill == "bodycam_documentary"`. These rules were
proven in the Colab full-arc edit of the Eustis FL bodycam case — they are requirements,
not suggestions.

## Structure: the 6-Act documentary (never a slice, never a dump)

- Target runtime **720s-900s (12-15 min)** when the raw source is >= 25 minutes; for
  shorter sources use a proportional ~50-65% of raw. A 40-min source must NEVER produce
  a sub-10-minute edit (hard QA floor: 600s).
- Acts, in order (acts[0] is the Cold Open):
  1. **Cold Open Hook (25-45s)**: 1-2 climax raw clips PULLED FROM LATER IN THE VIDEO
     (raw audio on) + a 15-20s Narrator hook setting date/location/mystery from the
     case dossier only.
  2. **Act 1: Initial Contact / Disarming / First Contradictions**
  3. **Act 2: Evidence Discovery & Excuses**
  4. **Act 3: Escalation / Key Admissions**
  5. **Act 4: Miranda / Policy Violations / Interrogation**
  6. **Act 5: Arrest / Detention / Witness Interaction**
  7. **Act 6: Booking / Final Testing + Outro (verified outcome)**
- Coverage law: clip starts must span the WHOLE file — earliest inside the first 15%,
  latest ending inside the final 25%. Dropping the arrest/booking ending is a QA failure.

## Pacing & segment sizing

- Keep raw clips focused: **15s to 75s** per uninterrupted stretch. If a compelling
  scene runs >90s, split it cleanly and insert a 10-15s Narrator bridge calling out the
  lie, the contradiction, or the legal rule at stake.
- Cut dead air (>1.8s silence) AND irrelevant logistics (personal key/phone details,
  routine radio spelling) — while KEEPING ironic, emotional, or incriminating exchanges
  intact. Never cut an admission to save time.
- Hook in the first 12 seconds: the most dramatic *moving* moment, raw audio on.
- Leave `pad_before_s=0.6` / `pad_after_s=0.8` breathing room around speech.

## Narration (8-12 blocks, 25-35% of runtime)

- Each block rides **muted (`volume=0`), slightly darkened B-roll** from a quiet,
  non-spoiling moment (never over the climax it precedes, never over unrelated footage).
- Voice: veteran true-crime podcaster talking to one viewer — conversational,
  contractions, present tense, short sentences, dry wit, zero melodrama.
- Never editorialize guilt or innocence; charges are allegations only.
- Never mention timestamps, timecodes, "seconds" or "minutes".
- Never invent case facts, charges, names, test results, or verdicts: narration facts
  come from the CaseDossier (each claim cites a source); if the dossier lacks a fact,
  the narration hints at mystery instead of filling it in.
- TTS: one voice, `trim_pauses=True`, 3-6s J-cut lead into the following raw clip,
  target ≤ 12-20 min.

## Audio law (enforced by tools)

- Levels are audited on **spoken-word intervals only** (ASR anchors) — sirens, door
  slams and cruisers skew whole-program loudnorm. Dialogue rides −12 dB … −6 dB RMS;
  master ceiling −1.5 dBTP (`speech_master_chain`).
- Bystander music flagged by Demucs stem separation → duck, mute or cut (copyright).

## Retention standard

- Every timeline segment is scored by the Critic Agent (Beginning → Middle → End
  momentum + open-loop tension) before render; a segment that loses the thread gets
  re-cut or dropped. Story first, footage second.
- The Phase B loop does NOT end while the timeline is under target: >= 600s total,
  >= 6 narration blocks, >= 8 raw clips, source span >= 0.70 — add material from the
  story index until the invariants read ALL MET.

## Cards & captions

- `location_intro` card on Act 1's Narrator intro; **1-3 `evidence_fact`** cards
  mid-documentary on concrete contradictions/BAC/policy facts; `case_outcome` card on
  the final Outro — strictly using verified case facts/transcript, never invented
  charges or sentences.
- 2-row speaker-colored `.ass`, 3-5 word beats, karaoke active-word highlight.
  Officer -> Lane 1 (`MarginV=46`), Suspect/Other -> Lane 2 (`MarginV=96`).
- `evidence_quote` / `evidence_fact` carry the timeline anchor they cite; cards sit
  collision-free, 30-50 ms fades, never inside the YouTube 16:9 player safe zone.

## Legal & compliance (hard rules, enforced by tools, not vibes)

- Raw public-record footage only; every clip license-verified and vision-screened.
- Mandatory disclaimer in description; "presumed innocent" end card.
- Bystander faces, minors, plates, IDs -> `redact.add_blur` before render.
- Profanity masked in captions (`profanity_mode="mask"`); graphic moments flagged for
  the Human Review Gate.

## Tool discipline (Phase B execution loop)

- Work act-by-act: build the whole arc with `timeline.insert_raw_clip` /
  `timeline.add_narration` / `cards.add`, then refine (`timeline.trim_clip`,
  `timeline.split_clip`, `narration.set_text`, `narration.add` over an existing raw
  window, `redaction.add_blur`).
- The loop does NOT terminate while the timeline is under target (see Retention
  standard). Source seconds come from the story index `[mm:ss]` markers (mm:ss ->
  seconds math).
