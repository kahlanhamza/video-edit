# STATUS.md — the single source of truth (ADR-008)

Project: REGRET ON CAMERA autopilot — automated police-bodycam documentary
video production pipeline (sourcing → script → voice → semantic edit →
package → human-review queue). Every task appends a row before it is closed.

## Session log

| # | Date | Lane | What actually happened | Outcome | Commits |
|---|------|------|------------------------|---------|---------|
| 1 | 2026-11-15 | day-0 | Repo adopted (repo was rebuilt 2x in wiped sandboxes before this). Token stored 0600, double-excluded. | docs committed | day0 |
| 2 | 2026-11-15 | **war story: unpushed commits lost** | Sandbox recycled AFTER 5 lane commits existed locally but were never pushed (`git push` missing from the loop). Remote still at stub; issues' evidence comments survived (they go through the API). Everything re-derivable: code from context, footage via committed manifest pattern, artifacts re-generated + committed. **New standing rule: a lane is not done until `git push` returns success — push is part of the commit.** | recovery | day0b |
| 3 | 2026-11-15 | M1 sourcing | Commons deep search 87 candidates → VLM watermark screen (3 frames/clip): round-1 7/8 rejected ('POLICE ACTIVITY' baked in). Round-2 official-release phrasing: 6 clean. Policy: no fatal-shooting content. Pool: mcneil_jax 414s/720p, tiger_dui_2017 1767s/480p, okchief_dui 2852s/480p. | issue #1 closed w/ evidence | 2223b7e (lost) → recommit |
| 4 | 2026-11-15 | M4 anchors | Motion+audio energy → peaks → GLM-vision labels. Every anchor labeled. gibson_arrest dropped (VLM caught baked-in text narration = title cards, raw-only rule). Acts: mcneil (door→restrain→cuffs), tiger (roadside→tests→handcuffed escort), okchief (stop→patrol car→cuffed w/ phone). | issue #4 closed | a8b7eb8 (lost) → recommit |
| 5 | 2026-11-15 | M2 script | 14 beats, conversational, mode+cue per beat matched to anchor labels; gates guardrails+no_timestamp_speak GREEN. | issue #2 closed | 9968e02 (lost) → recommit |
| 6 | 2026-11-15 | M3 voice | kazi@1.05, continuous 140wpm, effective 102.4 wpm (pauses); gate 95-155 effective; 429 rate limit hit + survived (backoff 20-100s x6). | issue #3 closed | c66be3e (lost) → recommit |
| 7 | 2026-11-15 | M5 editor | freeze-and-explain timeline v2: 33 segments / 11.9 min (hook 3.5s w/ audio → muted 0.45x graded replay per beat → 1.0x audio resume from replay stop-point → outro replay + end overlay). Segment cache made the post-recycle resume free; 0 re-renders. Frame contact-sheet verified: no black cards, lower-thirds, end overlay correct. | issue #5 closed | ff8e662 |
| 8 | 2026-11-15 | M6 gates | all 8 gates GREEN on final artifacts: imports / guardrails (19 texts) / no_timestamp_speak / voice_pace 112.8 wpm (95-155) / compliance (disclaimer in description) / render (715.7s, streams ok, 6/6 frames non-black) / no_black_cards (0/33) / narration_sync. Self-test proven RED-capable (ADR-0007). Video #1 queued for owner review. | issue #6 closed | ff8e662 |
| 9 | 2026-11-15 | M7 release | v0.1.0 tagged on 5275aee; GitHub Release with case1_final.mp4 (236MB) + case1_thumb.png; both assets unauth-download verified HTTP 200; leak check clean (output/, secrets/, db excluded). Metadata: 13 tags, 3 chapters, per-act attribution, disclaimer (gate GREEN). Video #1 awaiting owner review — pipeline never uploads. | issue #7 closed; AC-1..AC-8 all satisfied | 5275aee + this commit |
