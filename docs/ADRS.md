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

**ADR-0011 — Smooth motion, no still-scene slow-mo, tight gaps (v0.2.0 rework).**
Owner rejected v0.1.0 as "laggy, not how professional editing works". Measured:
replay sections played at 54-98% duplicate consecutive frames (naive
`setpts`+`fps` duplication), one replay carried a 15.2s freeze pad, the outro
was a 28s still image, 16 silent gaps averaged 23.2s, and a 474p source was
stretched 2.5x to a 1080p canvas at 2.77 Mbps. Decisions, all measurable:
(1) slowed content goes through `framerate` motion interpolation — measured
2% duplicates vs 25% naive on the same window; (2) narration replays run at
0.75x, deep 0.45x is reserved for the hook cold-open and flashbacks;
(3) any window whose measured motion energy < 0.08 plays real-time muted —
a still scene is never slowed; (4) frame-clone freeze is capped at 1.2s and
the speed ladder (0.75 down to 0.30) absorbs the rest; (5) silent forward
gaps are capped at 12s; (6) canvas is 1280x720 — 1:1 for 720p sources, 1.5x
worst-case upscale — CRF 18; (7) two new gates enforce it on the OUTPUT:
smooth-motion (<15% near-identical consecutive frames per replay segment)
and a bitrate floor. Rationale: gates checked structure but not feel; feel is
now also executable.

**ADR-0012 — Adopt proven open-source patterns; keep ffmpeg-direct engine.**
Deep research (docs/RESEARCH-opensource-video-engines.md) surveyed video-use,
NarratoAI, MoneyPrinterTurbo, ShortGPT, Practical-RIFE, auto-editor and
faster-whisper against our raw-footage commentary requirements. Adopted from
video-use (MIT): render→self-eval loop as hard gates (smooth-motion,
boundary-pop, bitrate on the OUTPUT), 30 ms audio fades at every segment cut,
and EDL-formalized timeline JSON. Adopted now from ffmpeg: `framerate` blend
interpolation for all slowed content (measured 2% duplicate frames vs 25%
naive); Practical-RIFE (MIT) is the phase-2 provider behind
`pipeline/interp.py` once onnxruntime CPU throughput is benchmarked — no
torch on this box. NOT adopted: MoneyPrinterTurbo/ShortGPT (stock-footage
sourcing model violates AC-1), NarratoAI engine (MoviePy slower than
frame-exact ffmpeg; no semantic-anchor model), Remotion/Manim overlay stacks
(over-engineering for static compliance overlays). Rationale: the wheel to
stop reinventing is the QC loop and the fade/interp craft, not the sourcing
and anchor model that differentiates the channel.
