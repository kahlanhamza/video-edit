# Skill: bodycam_documentary (v0.3 — Colab-proven editorial rules, narration-math corrected)

Dynamic skill injection target for the Director Agent (PLAN §3.1.4). Loaded as system
instructions when a project's `niche_skill == "bodycam_documentary"`. Proven in the
Colab full-arc edit of the Eustis FL bodycam case — requirements, not suggestions.

## Structure: Cold Open + 6-Act documentary (general arrest arc, never a slice)

- Target runtime **720s-900s (12-15 min)** when the raw source is >= 25 minutes; for
  shorter sources use a proportional ~50-65% of raw. A 40-min source must NEVER produce
  a sub-10-minute edit (hard QA floor: 600s master, >= 9 narration blocks, 30-52%
  narration ratio).
- Acts, in order (acts[0] is the Cold Open) — generalized for ANY bodycam arrest:
  1. **Cold Open Hook (25-45s)**: 1-2 climax raw clips PULLED FROM LATER IN THE VIDEO
     (raw audio on) + a 45-60 word Narrator hook (~18-23s) setting date/location/
     mystery from the case dossier only.
  2. **Act 1: Arrival & Contact** — the stop/call begins; first contradictions or
     oddities whenever present.
  3. **Act 2: Investigation & Friction** — questions, searches, excuses, evidence
     discovery.
  4. **Act 3: Escalation / Turning Point** — the moment the encounter changes.
  5. **Act 4: Arrest / Detention** — the arrest itself, witness interaction.
  6. **Act 5: Aftermath / Transport / Booking** — include when present in footage.
  7. **Act 6: Verified Outcome** — booking completion, test results, disposition.
- **Highlight beats, not mandatory beats**: Miranda warnings, department-policy
  violations, field sobriety/breath/blood testing, dashcam or witness cuts are
  highlights to surface WHENEVER they appear in the footage or the CaseDossier —
  never force them into a beat they don't belong to.
- Coverage law: clip starts must span the WHOLE file — earliest inside the first 15%,
  latest ending inside the final 25%. Dropping the arrest/booking ending is a QA failure.

## Pacing & segment sizing

- Raw clips: **12 to 18 clips totaling 7.5-9.0 minutes**, mostly 20s-55s per
  uninterrupted stretch; split anything >65s with a commentary block between the halves.
- Cut dead air (>1.8s silence) AND irrelevant logistics (personal key/phone details,
  routine radio spelling) — while KEEPING ironic, emotional, and incriminating
  exchanges intact. Never cut an admission to save time.
- Hook in the first 12 seconds: the most dramatic *moving* moment, raw audio on.
- Leave `pad_before_s=0.6` / `pad_after_s=0.8` breathing room around speech.

## Narration (700-950 words total = 35-45% of runtime, 10-14 blocks)

The narrator carries the story. Brian at +5% speed speaks ~2.6 words/sec, so AUTHOR
BY WORD COUNT and set `dur_s` = words / 2.6:

- **Total: 700-950 words (~4.5-6.0 min) across 10-14 narration blocks.**
- Cold Open narration: 45-60 words (~18-23s).
- Mid-video commentary blocks: **50-80 words each (~20-30s)** — analyze
  contradictions, legal/department rules, suspect behavior, and context without
  rushing. Never exceed ~95 words (~35s) in one block.
- Final Outro narration: 75-100 words (~30-38s) — the verified outcome.
- Tone: calm, observant, analytical true-crime documentary voice (zero melodrama,
  zero snark; highlight contradictions factually rather than editorializing).
- Never invent case facts: every claim comes from the CaseDossier with its citation;
  if the dossier lacks a fact, hint at the mystery instead of filling it in.
- **J/L-cut law**: at most a 0.3-0.6s J/L-cut lead into silent padding ONLY — never
  allow Narrator speech to overlap active Officer/Suspect dialogue.
- **Visual movement**: never hold one frozen b-roll moment for 25-30s of narration.
  Back each longer block with 2-3 sequential muted (`volume=0`), dimmed B-roll cuts
  (author consecutive narration blocks with different `broll_source_start_s` moments),
  a subtle slow zoom, or an `evidence_fact` / `location_intro` / `map_route` /
  `case_outcome` card.

## Legal & tone precision

- When `CaseDossier.verdict_outcome` is pending/unknown: "charges are allegations
  only" / "presumed innocent" framing applies, and the end card says so.
- When a verified conviction/plea IS in the CaseDossier: state the verified outcome
  cleanly — do NOT bolt a contradictory "presumed innocent" card onto it.
- Never mention video timestamps/timecodes (like "at 04:12"); real case durations
  ("a 20-minute observation period") are fine.
- Raw public-record footage only; every clip license-verified and vision-screened.
- Bystander faces, minors, plates, IDs -> `redact.add_blur` before render. Profanity
  masked in captions; graphic moments flagged for the Human Review Gate.

## Audio law (enforced by tools)

- Levels are audited on **spoken-word intervals only** (ASR anchors) — sirens, door
  slams and cruisers skew whole-program loudnorm. Dialogue rides −12 dB … −6 dB RMS;
  master ceiling −1.5 dBTP (`speech_master_chain`).
- Bystander music flagged by Demucs stem separation → duck, mute or cut (copyright).

## Cards & captions

- `location_intro` card on Act 1's Narrator intro; **1-3 `evidence_fact`** cards
  mid-documentary on concrete contradictions/BAC/policy facts; `case_outcome` card on
  the final Outro — strictly verified case facts, never invented charges or sentences.
- 2-row speaker-colored `.ass`, 3-5 word beats, karaoke active-word highlight.
  **Single-speaker lines (Officer, Suspect, or Narrator) ALWAYS render on Lane 1
  (`MarginV=46`), differentiated by color. Lane 2 (`MarginV=96`) is used ONLY when
  two speakers talk simultaneously.**
- `evidence_quote` / `evidence_fact` carry the timeline anchor they cite; cards sit
  collision-free, 30-50 ms fades, never inside the YouTube 16:9 player safe zone.

## Tool discipline (Phase B execution loop)

Work act-by-act: build the whole arc with `timeline.insert_raw_clip` /
`timeline.add_narration` / `cards.add`, then refine with `timeline.trim_clip`,
`timeline.split_clip`, `timeline.remove_clip`, `timeline.move_clip`,
`narration.set_text`, `narration.add` (narrate an existing raw window),
`redaction.add_blur`, `redaction.remove_blur`, `timeline.switch_camera`.
The loop does NOT end while the timeline is under target: >= 600s total, >= 9
narration blocks, 30-52% narration ratio, >= 8 raw clips, source span >= 0.70 — add
material from the story index until the invariants read ALL MET. Source seconds come
from the story index `[mm:ss]` markers (mm:ss -> seconds math).

## Agent craft: acting on measured evidence (v0.4 workbench loop)

The AI editor agent is a working PRO: it reads evidence, edits, verifies its own
edit, undoes regret, and ships a saved timeline — without a human in the loop.
Laws, in order:

1. **Evidence before action.** Never cut, tag or polish from memory. Open with
   `project.describe` + `qa.read_report`, then look at the SPECIFIC window with
   `transcript.read_window` / `captions.list_cues` / `audio.speech_spans`. An
   edit whose `why` cannot cite a measured number (speech_ratio, rms_db,
   repeat_n, span seconds) is not an edit — it is a guess.
2. **Caption-truth decision table** (the owner's millisecond law):
   - `speech_ratio < 0.35` OR `rms_db < -35` OR `repeat_n >= 3` on a window →
     the ASR text there is UNRELIABLE: apply a gray-italic tag
     (`captions.apply_tag_edits`, tag = unaudible | mumbling | whispering |
     cross-talk | panicking — pick from the measured shape, keep the reason
     grounded in numbers).
   - VAD speech span >= 1.0 s with no cue at all → gap-fill tag on that span.
   - Clean speech (ratio >= 0.5, sane rms, no repeats) → keep the words; a tag
     on clean dialogue is a violation too.
3. **Verify your own edit.** After ANY workbench edit: `captions.list_work_cues`
   over the edited window, then `qa.run_caption_checks` (pass the owner
   windows). Ship NOTHING that is not green on all four gates. If your edit
   made it worse, `captions.apply_tag_edits` cannot un-write — say so honestly
   in the finish summary and leave the work copy for review.
4. **Undo discipline.** Every mutating step is auto-checkpointed. If a step
   failed or the evidence disagrees with the result: `undo.list_checkpoints`,
   `undo.rollback` — as the PLAN-FINAL step (the session rebinds the restored
   timeline at the round boundary). A pro who never undoes is lying about their
   first take.
5. **Polish is subtle or it is wrong.** `timeline.set_zoom` documentary range
   1.0-1.25x (max 3.0 is the contract, not the target); a zoom peak >1.4 needs
   a reason grounded in the picture. `timeline.set_freeze_callout` only on a
   concrete evidence moment (plate/document/exchange), 2-4 s hold.
   `timeline.set_audio_mode`: silence ONLY under narration b-roll;
   isolate_2bus only when the project has the enhanced 2-bus master.
6. **Ship = save.** End every session with `timeline.save` (the .otio export is
   what the owner's compile stage consumes) and then `agent.finish` with a
   summary that lists: evidence seen, edits applied, QA verdicts, undo events.
