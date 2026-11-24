
Dynamic skill injection target for the Director Agent. Loaded when a project's
`niche_skill == "retail_theft"` — shoplifting / loss-prevention / retail
encounter footage and its commentary breakdowns. Requirements, not suggestions.

## Structure: Cold Open + 5-6 Act documentary (retail incident arc)

- Target runtime follows `doc_scale` (the code owns the floors). A 40-min
  source must NEVER produce a sub-10-minute edit.
- Acts, in order (acts[0] is the Cold Open):
  1. **Cold Open Hook (25-45s)**: 1-2 climax raw clips PULLED FROM LATER IN
     THE VIDEO (raw audio on) + the hook narration (date/store/mystery from
     the case dossier only) over muted climax b-roll.
  2. **Act 1: Arrival & Observation** — the suspected entry, LPO/employee
     observation, the first radio or register call.
  3. **Act 2: Watching & Build-up** — the suspected concealment, recovery
     attempts, the stop decision being made.
  4. **Act 3: The Stop** — the approach, the position of every party, the
     first demands. This is the content's spine.
  5. **Act 4: Peak Confrontation / Recovery** — merchandise recovery, the
     struggle or the compliance, police arrival — whatever the peak is,
     play it in full.
  6. **Act 5: Aftermath & Outcome** — citation, trespass notice, police
     transport, LPO paperwork — include ONLY when present.
- **Highlight beats, not mandatory beats**: LPO stop scripting, entrance
  blocking, merchandise recovery, police handoff, "I didn't pay" or "my kid
  did it" admissions — surface WHENEVER present, never force them.
- Coverage law: clip starts span the WHOLE file — earliest inside the first
  15%, latest ending inside the final 25%. Dropping the outcome ending is a
  QA failure.

## Pacing & segment sizing

- Raw clips: 12-18 clips totalling ~55-70% of target runtime, mostly 20-55s
  per stretch; split anything >65s with a commentary block between halves.
- Cut dead air (>1.8s silence) and empty surveillance stretches — KEEP every
  demand/recovery/admission exchange COMPLETE. Never cut an admission to
  save time.
- Hook in the first 12 seconds: the most dramatic *moving* moment, raw
  audio on.
- Leave `pad_before_s=0.6` / `pad_after_s=0.8` breathing room around speech.

## Narration (the word-count law lives in doc_scale; 30-52% of runtime)

The narrator carries the story. Author BY WORD COUNT (2.6 words/sec):

- Cold open: 45-60 words. Mid-video blocks: 50-80 words (never >95). Outro:
  75-100 words.
- Tone: calm, observant retail-crime documentary voice (zero melodrama, zero
  snark); highlight contradictions factually, never editorialize about guilt
  beyond the dossier.
- Never invent case facts: every claim comes from the CaseDossier with its
  citation; absent facts, hint at the mystery instead of filling it in.
  Suspects are CHARGED or ACCUSED people, not certified thieves — the
  verified-outcome framing law governs the end card.
- **J/L-cut law**: 0.3-0.6s lead into silent padding ONLY — never let
  Narrator speech overlap active dialogue.
- **Visual movement**: 2-3 sequential muted b-roll cuts under longer blocks,
  slow zoom, or `evidence_fact` / `location_intro` cards.
- **A1 audio trim (`timeline.trim_audio`)**: cuts INSIDE the synthesized TTS
  stem (vo window `[vo_in_s, vo_out_s]`), never the b-roll window — shave
  leading dead air (~0.3-0.6s) or a trailing tail; the 0.5s floor is a law:
  never trim below one intelligible phrase. Raw clips keep their source
  audio — trim their SOURCE windows with `timeline.trim_clip` instead.

## Legal & tone precision

- Store-detention law varies by state (merchant statutes); explain the
  general rule without case-specific legal advice.
- Name only verified facts (dossier citations). Pending charges: "allegations
  only" framing; a verified conviction/plea is stated cleanly — never a
  contradictory "presumed innocent" card bolted onto it.
- Raw consent-first/store footage only; license-verified clips.
- Bystander faces, minors, plates, IDs -> `redact.add_blur` before render.
  Profanity masked in captions; graphic moments flagged for the Human
  Review Gate.

## Audio law (enforced by tools)

- Levels are audited on **spoken-word intervals only** (ASR anchors) —
  beeping registers, store music and PA systems skew whole-program loudnorm.
  Dialogue rides −12 … −6 dB RMS; master ceiling −1.5 dBTP.
- Store/background music flagged by Demucs -> duck, mute or cut (copyright).

## Cards & captions

- `location_intro` on Act 1's narration; 1-3 `evidence_fact` mid-doc on
  concrete contradictions (receipt vs merchandise, timestamp gaps);
  `case_outcome` on the Outro — verified facts only.
- 2-row speaker-colored `.ass`, 3-5 word beats, karaoke highlight — the two
  dialogue speakers stay distinct by color, Lane 1 for single speakers, Lane
  2 ONLY for simultaneous speech.

## Tool discipline (Phase B execution loop)

Build the arc with `timeline.insert_raw_clip` / `timeline.add_narration` /
`cards.add`; refine with `timeline.trim_clip`, `timeline.split_clip`,
`timeline.remove_clip`, `timeline.move_clip`, `narration.set_text`,
`narration.add`, `redaction.add_blur`, `redaction.remove_blur`,
`timeline.switch_camera`. Meet the `doc_scale` invariants before finishing.
Source seconds come from the story index `[mm:ss]` markers (mm:ss -> seconds).

The shared craft lawbook (caption craft, dead-air cuts, punch-ins, audio
chain, provenance rules) is appended by the skill loader — it binds with the
same force as this file.
