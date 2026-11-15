# TEARDOWN — clean-room evidence before writing code

Studied (evidence, not code): how successful bodycam commentary channels edit,
and how naive AI-agent attempts fail. Conclusions drive our ADRs.

## 1. Why naive "ffmpeg-cut" agents fail (observed failure modes)

| Failure mode | Evidence from our own v1 demo (2026-11-15) |
|---|---|
| Narration ↔ footage drift | Narrator described one thing while unrelated footage played (owner complaint). Cause: beats mapped to *time windows*, not to *semantic moments*. |
| Timestamp-speak | Script read like a robot ("at 180 seconds…"). Cause: LLM given timeline math. |
| Black text cards | Full-screen intro/chapter cards kill watch time (owner complaint). Real editors never cut to a dead screen mid-video. |
| Wrong pace | TTS at 87 wpm felt slow/robotic. Documentary VO is ~130-150 wpm. |
| Aggregator footage | v1 footage carried third-party channel watermarks ("POLICE ACTIVITY", "BODYCAM LOCKUP"). Owner: **raw footage only**. |
| A/V competition | Original audio fought the narrator. During VO the bed must be silent (muted replay), not ducked under. |
| Unpushed work | 5 lane commits sat local-only and died in a recycle (owner: "always push"). The skill's §2.7 exists for a reason. |

## 2. How the successful style actually works (decoded)

The "freeze-and-explain" model used by top bodycam channels:

1. **Hook**: a raw, loud, dramatic 3-4s moment plays untouched (with audio).
2. **Explain**: narration plays over a *muted slow-motion replay* of exactly
   the moment being described (zoom + grade = "analysis" look).
3. **Resume**: when the narration ends, footage continues *from the point the
   replay reached*, at 1.0x with original audio — the viewer "keeps watching".
4. **Freeze** when the narrator needs more time than the replay offers;
   **flashback** (desaturated) when referencing an earlier moment;
   raw untouched footage everywhere else.
5. NO black cards. Lower-thirds only, brief.

## 3. Tool survey (open source, free)

| Tool | Verdict |
|---|---|
| PySceneDetect | scene-cut tables; adopted |
| OpenCV | motion-energy analysis for anchor detection; adopted |
| z-ai vision (GLM VLM) | semantic frame labeling ("what is happening at t=X"); adopted — the scene *understanding* the owner demands |
| ffmpeg `tpad stop_mode=clone` | freeze frames; adopted |
| ffmpeg `setpts` / `minterpolate` | slow-motion (dup-frame vs interpolated); adopted (dup at 0.45x; minterpolate too slow for 1080p daily runs) |
| ffmpeg `eq`/`colorbalance` | analysis grade + flashback desaturation; adopted |
| moviepy 2.x | available as fallback; not primary (slower, less deterministic than explicit ffmpeg graphs) |
| Remotion | frame-exact programmatic video via React; powerful but heavy toolchain for a daily batch job; parked |
| Open-Higgsfield / generative studios | generation, not footage editing; out of scope for documentary (synthetic content breaks authenticity + monetization) |
| auto-editor / editly | silence-cutting and declarative assembly; patterns noted, not needed yet |
