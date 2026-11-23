# Skill: bodycam_documentary (v0.5 — Colab-proven rules + industry-verified craft constants)

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
- **A1 audio trim (`timeline.trim_audio`)**: the audio track's own trim. It cuts
  INSIDE the synthesized TTS stem — a vo window `[vo_in_s, vo_out_s]` — and never
  moves the b-roll window. Use it to shave leading dead air (~0.3-0.6s) or a
  trailing tail after a re-authored script; the block re-sizes to the window +
  breathing pad on the next render. The 0.5s floor is a law: never trim a block
  below one intelligible phrase (one word ~ 0.3s at +5% speed). Raw clips keep
  their source audio — trim their SOURCE windows with `timeline.trim_clip` instead.

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
6. **Ship = save, LAST.** `timeline.save` after your FINAL mutation — if you
   undo or roll back AFTER a save, save AGAIN before finishing (session v2
   shipped a saved .otio of an already-undone trim). Then `agent.finish` with
   a summary that lists: evidence seen (with numbers), edits applied, QA
   verdicts (all four gates, truthfully), undo events.
7. **Answer to the critic before you claim done.** Run `qa.self_audit` after
   structural edits and before `timeline.save`: master gaps, zero-length
   items, empty narration scripts, zoom law, plan invariants, your own caption
   QA, the shipped master's gates. A check that never ran is "not
   established" — never report it as a pass. Fix every RED name first.
8. **Leave memory for the next session.** `session.plan` at session start
   (set the task ledger), `session.notes` whenever evidence teaches you
   something (facts learned, decisions, blockers), `session.plan mark_done`
   as tasks complete, `session.status` FIRST on a resumed session. The notes
   file IS the memory across process death — a fact only you discovered and
   did not write down is a fact the next session must re-derive.

## Craft constants (v0.5 — industry-verified, keep sources with numbers)

Caption craft (Netflix Timed Text style guide, via kmilo93sd/agent-video-editor):
- <= 42 chars/line, <= 2 lines, reading speed <= 20 chars/s, min duration
  ~0.83 s; break lines at punctuation, never split article-from-noun or a name.
- Close a cue on sentence end, or at 6 s max, or when a speech gap >= 0.6 s
  opens (cut-as-code chunking law). Gaps < 0.3 s between cues get closed.
- Never double-deliver: footage that burns captions must not also ship an
  external .srt with the same text (viewer sees both).

Silence/dead-air cuts (browser-use/video-use measured thresholds):
- Gaps >= 400 ms are the clean cut targets; 150-400 ms usable only with a
  frame check; < 150 ms is mid-phrase — never cut.
- Pad every cut edge 30-200 ms; speaker-handoff air 400-600 ms. ASR word
  ends run 50-150 ms early — extend the cut-out until the room tone is quiet
  (~6 dB over floor) before you cut (konradre two-band edge law: place
  edges from SOUND, never from word timestamps alone).

Punch-ins (social-media-skills + kmilo93sd):
- Scale 1.10-1.20x, never > 1.25x on this genre; switch at cuts that remove
  >= 1 s, else only after 4 s of the same framing; never across a cut;
  space them 5-7 s apart. (Matches the tool law: `timeline.set_zoom` 1.0-1.25.)

Audio (YouTube delivery, verified chain):
- Target -14 LUFS integrated, true peak <= -1 dBTP, two-pass loudnorm, verify
  with ebur128. Chain: gain -> highpass 80 Hz -> afftdn (skip if room noise
  < -60 dB) -> acompressor 3:1 from -24 dB (5 ms attack / 120 ms release)
  -> limiter -2 dB -> one clean gain to target (social-media-skills).
- Music beds sit 18-25 dB under dialogue; SFX 12-24 dB under; narration blocks
  TTS clips must never clip -1 dBTP (kmilo93sd sourced numbers).

Provenance lawbook for bodycam/news editing (0xsline/OpenChatCut news-rough-cut):
- ONLY the sources' raw sound may enter the cut. NO music beds, NO added VO
  over evidence audio, NO SFX on evidence moments — the genre's credibility
  IS the provenance. (Narration blocks ride muted, darkened b-roll — never
  over a speaker's evidence audio.)
- Speech is cut only at sentence ends / natural pauses / speaker turns;
  never splice two contexts into a fake causality; never present conjecture
  as fact; restrained dissolves only, no filters or stickers.
- The verified-outcome framing law overrides this only where the dossier
  carries a verdict (see Legal & tone precision above).
