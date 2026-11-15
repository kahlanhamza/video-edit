# ADRs — ratified decisions (adapted from zero-to-shipped)

**ADR-0001 — The record is in-repo (STATUS.md), pushed every task.** Sandbox
recycles wiped the tree three times; amnesia is solved by the record, not by
chat memory.

**ADR-0002 — Raw footage only, vision-gated.** Every source clip passes a VLM
watermark/baked-VO/title-card screen before use; aggregator content is
rejected. Policy: no fatal-shooting footage (advertiser + ethics).

**ADR-0003 — Timeline-as-journal.** The edit is a serialized, resumable list of
segments with exact in/out/effect/offset (JSON in output/cache). Every segment
renders to its own cached file; a killed run resumes at segment granularity.
Verify-before-declare: the final mux is probed (duration, streams) before any
"rendered" claim.

**ADR-0004 — Freeze-and-explain editing model.** Narration only over MUTED
slow-motion replays of the semantic moment it describes; forward play with raw
audio between narrations resumes from the point the replay reached; freeze via
`tpad=stop_mode=clone`; flashback via desaturated grade; no full-screen text
cards. Hook opens raw with audio. End overlay on-frame, no black card.

**ADR-0005 — Semantic anchors over time windows.** Beats are matched to
*anchor moments* found by motion+audio energy analysis and labeled by a VLM
(z-ai vision). A beat's cue is matched to anchor labels; never to a bare
timestamp. Rationale: fixes narration↔footage drift.

**ADR-0006 — ffmpeg-exact over framework.** Primary renderer is explicit,
frame-exact ffmpeg filtergraphs built in Python (scenedetect/opencv + PIL +
ffmpeg). moviepy/Remotion parked as alternatives, not the engine.

**ADR-0007 — Gates are non-bypassable.** `pipeline/gates.py` enforces: imports,
guardrail regex, no-timestamp-speak, voice pace, compliance, render sanity
(duration/streams/non-black), no-black-cards, narration-sync. RED gate = stop.
Every gate has `--self-test`.

**ADR-0008 — Heavy artifacts live in Releases, not git.** >100MB files cannot
be pushed; final MP4s/thumbnails attach to GitHub Releases. Footage cache is
gitignored; the *manifest* (URLs, licenses, screening verdicts) IS committed —
a fresh clone can re-download everything.

**ADR-0009 — Secrets: local-only, 0600, double-excluded.** Token lives at
`data/secrets/github_token` (gitignored AND `.git/info/exclude`), never
printed, never committed. Tokens that appear in chat are burned; rotate.

**ADR-0010 — Push is part of the commit (owner mandate, 2026-11-15).** Five
lane commits were lost to a sandbox recycle because `git push` was treated as
a separate step. Effective immediately: every commit command chain ends with
`git push origin main` and the lane is not done until the remote SHA matches.
