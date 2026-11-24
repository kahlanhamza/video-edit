
Dynamic skill injection target for the Director Agent. Loaded when a project's
`niche_skill == "first_amendment_audit"` — recording police/public officials in
public, consensual encounter recordings, audits and cop-watch footage.
Requirements, not suggestions.

## Structure: Cold Open + 5-6 Act documentary (public-recording arc)

- Target runtime follows `doc_scale` (the code owns the floors). A 40-min source
  must NEVER produce a sub-10-minute edit.
- Acts, in order (acts[0] is the Cold Open):
  1. **Cold Open Hook (25-45s)**: 1-2 climax raw clips PULLED FROM LATER IN THE
     VIDEO (raw audio on) + the hook narration (date/location/mystery from the
     case dossier only) over muted climax b-roll.
  2. **Act 1: Arrival & Setup** — arriving on scene, camera introduction, the
     first "am I being detained" or "is there a problem" exchange.
  3. **Act 2: Questions & Friction** — ID demands, justification requests,
     refusals, back-up arriving. The legal chess match IS the content.
  4. **Act 3: Escalation** — orders to stop filming, interference, threats of
     arrest, the moment the encounter turns.
  5. **Act 4: Peak Confrontation / Detention** — detention attempt, hands-on,
     or officer back-down. Whatever the peak is, play it in full.
  6. **Act 5: Aftermath & Outcome** — release, supervisor complaint, ticket,
     or dispersal — include ONLY when present in the footage.
- **Highlight beats, not mandatory beats**: "am I free to go" / "am I being
  detained" exchanges, badge/refusal-to-name moments, supervisor arrivals,
  consent-vs-order confusions, plain-view recording policies — surface WHENEVER
  present, never force them.
- Coverage law: clip starts span the WHOLE file — earliest inside the first
  15%, latest ending inside the final 25%. Dropping the outcome ending is a
  QA failure.

## Pacing & segment sizing

- Raw clips: 12-18 clips totalling ~55-70% of target runtime, mostly 20-55s per
  stretch; split anything >65s with a commentary block between the halves.
- Cut dead air (>1.8s silence) and idle walking footage without speech or
  motion — KEEP every demand/refusal exchange COMPLETE. Never cut an exchange
  to save time; the legal tension lives in the exact words.
- Hook in the first 12 seconds: the most dramatic *moving* moment, raw audio on.
- Leave `pad_before_s=0.6` / `pad_after_s=0.8` breathing room around speech.

## Narration (the word-count law lives in doc_scale; 30-52% of runtime)

The narrator carries the legal context. Author BY WORD COUNT; Brian at +5%
speed speaks ~2.6 words/sec, so set `dur_s` = words / 2.6:

- Cold open: 45-60 words. Mid-video blocks: 50-80 words (never >95). Outro:
  75-100 words.
- Tone: calm, precise, civics-teacher voice — explain the rights question at
  play WITHOUT giving legal advice ("courts have generally...", "policy
  varies by department" is honest; "he had a right to..." as a settled
  conclusion is not — the dossier decides).
- Never invent case facts: every claim comes from the CaseDossier with its
  citation; absent facts, hint at the mystery instead of filling it in.
- **J/L-cut law**: 0.3-0.6s lead into silent padding ONLY — never overlap
  narrator speech with an active exchange; the exact words ARE the evidence.
- **Visual movement**: 2-3 sequential muted b-roll cuts under longer blocks
  (consecutive blocks with different `broll_source_start_s`), slow zoom, or
  `evidence_fact` / `location_intro` cards.
- **A1 audio trim (`timeline.trim_audio`)**: cuts INSIDE the synthesized TTS
  stem (vo window `[vo_in_s, vo_out_s]`), never the b-roll window — shave
  leading dead air (~0.3-0.6s) or a trailing tail; the 0.5s floor is a law:
  never trim below one intelligible phrase. Raw clips keep their source audio
  — trim their SOURCE windows with `timeline.trim_clip` instead.

## Legal & tone precision

- Filming-in-public rights vary by country and circuit; state the general rule
  from the dossier, never as case-specific legal advice.
- State only verified facts (dossier citations). Pending complaints/cases:
  "allegations only" framing; a verified outcome is stated cleanly — never a
  contradictory "presumed innocent" card bolted onto it.
- Raw public-record or consent-first footage only; license-verified clips.
- Third-party bystander faces, minors, plates, IDs -> `redact.add_blur`
  before render. Profanity masked in captions; graphic moments flagged for
  the Human Review Gate.

## Audio law (enforced by tools)

- Levels are audited on **spoken-word intervals only** (ASR anchors) —
  street noise, wind and sirens skew whole-program loudnorm. Dialogue rides
  −12 … −6 dB RMS; master ceiling −1.5 dBTP. Shouting IS dialogue — never
  let the limiter chase peaks with the speech gate open.
- Band/or urban background music flagged by Demucs -> duck, mute or cut
  (copyright).

## Cards & captions

- `location_intro` on Act 1's narration; 1-3 `evidence_fact` mid-doc on
  concrete episodes (an ID statute, a policy number, a contradiction);
  `case_outcome` on the Outro — verified facts only.
- 2-row speaker-colored `.ass`, 3-5 word beats, karaoke highlight — the two
  dialogue speakers stay distinct by color, Lane 1 for single speakers, Lane
  2 ONLY for simultaneous speech.

## Tool discipline (Phase B execution loop)

Build the arc with `timeline.insert_raw_clip` / `timeline.add_narration` /
`cards.add`; refine with `timeline.trim_clip`, `timeline.split_clip`,
`timeline.remove_clip`, `timeline.move_clip`, `narration.set_text`,
`narration.add`, `redaction.add_blur`, `redaction.remove_blur`,
`timeline.switch_camera`. The loop does NOT end while the timeline is under
target: meet the `doc_scale` invariants (total, narration blocks, ratio, raw
clips, source span) — add material from the story index until ALL MET. Source
seconds come from the story index `[mm:ss]` markers (mm:ss -> seconds).

The shared craft lawbook (caption craft, dead-air cuts, punch-ins, audio
chain, provenance rules) is appended by the skill loader — it binds with the
same force as this file.
