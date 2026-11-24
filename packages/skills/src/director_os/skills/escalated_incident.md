
Dynamic skill injection target for the Director Agent — the GENERIC playbook.
Loaded when a project's `niche_skill` names an escalated-incident niche the
platform has no dedicated playbook for (and as the fallback when an unknown
niche string reaches the loader). ANY narratable encounter/escalation footage:
audits, shoplifting, trespassing, confrontations, interviews gone wrong.
Requirements, not suggestions.

## Structure: Cold Open + 5-6 Act documentary (general escalation arc)

- Target runtime follows `doc_scale` (the code owns the floors). A 40-min
  source must NEVER produce a sub-10-minute edit.
- Acts, in order (acts[0] is the Cold Open):
  1. **Cold Open Hook (25-45s)**: 1-2 climax raw clips PULLED FROM LATER IN
     THE VIDEO (raw audio on) + the hook narration (date/place/who/mystery
     from the case dossier only) over muted climax b-roll.
  2. **Act 1: Arrival & Contact** — the encounter begins; first oddities.
  3. **Act 2: Friction & Questions** — demands, explanations, excuses; the
     gap between what is said and what the camera shows starts to show.
  4. **Act 3: Escalation** — the moment the encounter changes: voices,
     postures, threats, or the arrival of new parties.
  5. **Act 4: Peak** — detention/citation/confrontation/endgame; whatever
     the peak is, play it in full.
  6. **Act 5: Aftermath & Outcome** — resolution, paperwork, dispersal —
     include ONLY when present in the footage.
- **Highlights over mandates**: whatever the footage's own peak beats are
  (admissions, refusals, contradictions, unexpected turns), surface them —
  never force a template beat into footage that doesn't contain it.
- Coverage law: clip starts span the WHOLE file — earliest inside the first
  15%, latest ending inside the final 25%. Dropping the outcome ending is a
  QA failure.

## Pacing & segment sizing

- Raw clips: 12-18 clips totalling ~55-70% of target runtime, mostly 20-55s
  per stretch; split anything >65s with a commentary block between halves.
- Cut dead air (>1.8s silence) and idle logistics — KEEP ironic, emotional
  and incriminating exchanges COMPLETE. Never cut an admission to save time.
- Hook in the first 12 seconds: the most dramatic *moving* moment, raw
  audio on.
- Leave `pad_before_s=0.6` / `pad_after_s=0.8` breathing room around speech.

## Narration (the word-count law lives in doc_scale; 30-52% of runtime)

The narrator carries the story. Author BY WORD COUNT (2.6 words/sec):

- Cold open: 45-60 words. Mid-video blocks: 50-80 words (never >95). Outro:
  75-100 words.
- Tone: calm, observant documentary voice (zero melodrama, zero snark);
  highlight contradictions factually rather than editorializing.
- Never invent case facts: every claim comes from the CaseDossier or the
  transcript; absent facts, hint at the mystery instead of filling it in.
- **J/L-cut law**: 0.3-0.6s lead into silent padding ONLY — never let
  Narrator speech overlap active dialogue.
- **Visual movement**: 2-3 sequential muted b-roll cuts under longer blocks
  (consecutive blocks, different `broll_source_start_s`), slow zoom,
  `evidence_fact` / `location_intro` cards.
- **A1 audio trim (`timeline.trim_audio`)**: cuts INSIDE the synthesized TTS
  stem (vo window `[vo_in_s, vo_out_s]`), never the b-roll window — shave
  leading dead air (~0.3-0.6s) or a trailing tail; the 0.5s floor is a law:
  never trim below one intelligible phrase. Raw clips keep their source
  audio — trim their SOURCE windows with `timeline.trim_clip` instead.

## Legal & tone precision

- The story is an ALLEGATION until the dossier says otherwise: pending
  charges get "allegations only" / "presumed innocent" framing; a verified
  outcome is stated cleanly — never a contradictory "presumed innocent"
  card bolted onto it.
- Name only verified facts (dossier citations); no invented names, charges,
  or verdicts.
- Raw publicly-recordable, consent-first footage only; license-verified
  clips.
- Bystander faces, minors, plates, IDs -> `redact.add_blur` before render.
  Profanity masked in captions; graphic moments flagged for the Human
  Review Gate.

## Audio law (enforced by tools)

- Levels are audited on **spoken-word intervals only** (ASR anchors) —
  sirens, applause, background noise skew whole-program loudnorm. Dialogue
  rides −12 … −6 dB RMS; master ceiling −1.5 dBTP. Shouting IS dialogue.
- Background music flagged by Demucs -> duck, mute or cut (copyright).

## Cards & captions

- `location_intro` on Act 1's narration; 1-3 `evidence_fact` mid-doc on
  concrete episodes/contradictions; `case_outcome` on the Outro — verified
  facts only.
- 2-row speaker-colored `.ass`, 3-5 word beats, karaoke highlight — the two
  dialogue speakers stay distinct by color, Lane 1 for single speakers, Lane
  2 ONLY for simultaneous speech.

## Tool discipline (Phase B execution loop)

Build the arc with `timeline.insert_raw_clip` / `timeline.add_narration` /
`cards.add`; refine with `timeline.trim_clip`, `timeline.split_clip`,
`timeline.remove_clip`, `timeline.move_clip`, `narration.set_text`,
`narration.add`, `redaction.add_blur`, `redaction.remove_blur`,
`timeline.switch_camera`. Meet the `doc_scale` invariants before finishing.
Source seconds come from the story index `[mm:ss]` markers (mm:ss ->
seconds).

The shared craft lawbook (caption craft, dead-air cuts, punch-ins, audio
chain, provenance rules) is appended by the skill loader — it binds with the
same force as this file.
