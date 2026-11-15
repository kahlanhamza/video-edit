# SPEC — REGRET ON CAMERA autopilot (v2)

One command produces one documentary-style bodycam video per day and queues it
for the owner's manual review. The owner is the publisher; the pipeline never
uploads.

## Acceptance criteria (v0.1.0)

- **AC-1 Sourcing**: every clip used is (a) public record / public domain with
  license verified per file, and (b) vision-screened RAW — no third-party
  channel watermark, no baked-in voiceover or title cards. Screening evidence stored.
- **AC-2 Script**: conversational documentary narration; zero timestamp
  references (regex-gated); defamation guardrails applied (regex-gated);
  per-beat `mode` ∈ {replay, flashback} + semantic `cue` matched to VLM anchor labels.
- **AC-3 Voice**: one consistent voice (kazi), natural pace (measured, gated
  95-155 wpm effective), never robotic.
- **AC-4 Semantic edit**: narration always sits over a muted slow-mo replay of
  the semantic moment it describes (anchor-matched, not time-window-matched);
  after narration the footage resumes 1.0x WITH audio from the point the
  replay reached; freeze via frame-clone when replay is short; flashback
  desaturated when the beat references the past; raw footage otherwise
  untouched; NO full-screen text cards.
- **AC-5 Compliance**: mandatory disclaimer in description; presumption of
  innocence on the end overlay; no advertiser-hostile content (no fatal
  shooting footage selected by policy).
- **AC-6 Verification**: gates run on the committed tree + artifacts and fail
  loudly; final render verified frame-wise (non-black) and audio-wise.
- **AC-7 Release**: code in this repo; heavy artifacts (final MP4, thumbnail)
  attached to a GitHub Release (tag `v0.1.0`); nothing internal leaks.
- **AC-8 The push rule (owner mandate)**: every lane is committed AND pushed in
  the same step; the working tree and origin never diverge between lanes.

## Milestones

- M0 day-0 docs · M1 raw-footage · M2 scriptwriter v2 · M3 voice v2 ·
  M4 semantic anchors · M5 editor v2 · M6 gates · M7 packaging + release
