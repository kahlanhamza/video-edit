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
| 7 | 2026-11-15 | M5 editor | freeze-and-explain timeline v2 built; render interrupted by recycle; rebuilding now. | in progress | — |
