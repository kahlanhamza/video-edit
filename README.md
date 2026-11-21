# DirectorOS

Autonomous multimodal video editing & documentary platform. Long-form (12–20 min)
documentary / commentary / bodycam post-production, scalable to Shorts and other niches.

**Master plan:** [PLAN.md](PLAN.md) — the single source of truth (Architecture RFC v1.0).
Everything we build maps to a milestone and an acceptance criterion in that document.

## Architecture in one line

Durable cloud state + multimodal perception + native agentic tool execution, all operating
on an industry-standard **OpenTimelineIO** timeline through a strongly-typed tool registry,
with a human-in-the-loop studio gate before anything is published.

```text
Director Studio (Next.js 16, M3)  →  FastAPI gateway + Celery/Redis orchestration
                                  →  Agent runtime (Perceiver → Director → Assembler → Critic)
                                  →  Media worker fleet (FFmpeg, libass, Pillow, OTIO)
                                  →  GPU services (Modal, M4: WhisperX, Demucs, YOLO blur)
                                  →  YouTube Private Draft (YouTube Data API v3)
```

## Monorepo layout

| Path | Purpose |
| --- | --- |
| `packages/core` | Config, Pydantic v2 contracts, SQLAlchemy models, storage, QA probe, fonts |
| `packages/otio_engine` | OTIO timeline builder, metadata contract, segment cache, render compiler |
| `packages/audio_engine` | FFmpeg audio chains (48 kHz lock, exact slicing, ducking, speech-aware trims) |
| `packages/subtitles` | difflib word alignment, 2-row speaker-colored `.ass` builder |
| `packages/vision_fx` | Glass graphic cards, blur/redaction filter chains |
| `packages/agent_runtime` | Typed tool registry (native function-calling engine): 15 timeline mutations + 9 evidence reads + 7 workshop actions — the AI editor's full 31-tool surface |
| `packages/perception`, `packages/skills` | M2/M4 — perception index + niche playbooks |
| `apps/api`, `apps/worker` | FastAPI gateway + Celery workers |
| `apps/cli` | `director` command line entrypoint |
| `apps/web` | M3 — Director Studio UI (stubbed) |
| `gpu_services` | M4 — Modal serverless GPU endpoints (stubbed) |

## Quickstart

```bash
uv sync                      # resolve + install the whole workspace
make lint type test          # ruff + mypy + pytest (unit + integration)
docker compose up -d         # postgres(pgvector) + redis + minio + api + worker
uv run director render tests/fixtures/fixture_30s.mp4 --max-seconds 12
```

## The hard-won guardrails (PLAN §9) are executable

Every R&D bug is a CI test — see `tests/unit/test_g1..g8`:

1. `-ar 48000 -ac 2` **after** `loudnorm` on every segment, or downstream narration dies.
2. Exact slicing: `-ss <start> -t <end-start>` — never `-ss` before `-i` with `-to` after.
3. Every cloud ingest sends a descriptive `User-Agent`.
4. Dead-air cuts come from speech + audio-event + motion intervals, never raw volume.
5. Proofread transcripts re-align via `difflib.SequenceMatcher`, never linear interpolation.
6. Simultaneous speakers ride two `.ass` lanes (`MarginV=46` / `MarginV=96`).
7. Fonts matched as exact upright filenames (`Roboto-Bold.ttf`, never `Roboto-BoldItalic.ttf`).
8. Licensing safety: no Remotion above small-team scale — FFmpeg + Pillow + Playwright.

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/e2e_real_video.py` | Full pipeline on REAL footage (resumable): ingest → YuNet face sweep → live Gladia ASR → render → subtitles → QA → burn-in |
| `scripts/download_real_video.sh` | Rate-limit-aware Wikimedia download with backoff + checksum validation |
| `scripts/demo_zoom_freeze.py` | Keyframed punch-in + freeze-callout demo over the real-footage master |
| `scripts/e2e_smoke.py` | Full pipeline on a synthetic *spoken* fixture — live Gladia ASR, real render, QA audit |
| `scripts/e2e_rerender.py` | Crash-resume semantics: reuse transcript/timeline, re-render only changed segments |
| `scripts/llm_smoke_test.py` | Live AI Studio probe: model visibility + function-calling wire format |
| `scripts/gen_youtube_refresh_token.py` | One-time interactive OAuth consent → `YOUTUBE_REFRESH_TOKEN` into `.env` |

## The AI editor agent — a working pro, not a script

The agent runtime gives one LLM loop a **complete editor's toolbox** (31 typed,
Pydantic-validated tools) plus the craft rules of the niche skill, and lets it
act on its own:

- **Read (9)** — `project.describe`, `qa.read_report`, `transcript.read_window`,
  `captions.list_cues`, `audio.speech_spans` (cached VAD compute),
  `timeline.list_items` / `get_item`, `captions.propose_tag_window`,
  `agent.finish`. Evidence laws are unenforceable without eyes.
- **Mutate (15)** — the master cut (`trim/split/remove/move/insert_raw`),
  narration (`add_narration`, `narration.add/set_text`), privacy
  (`redaction.add/remove_blur`), multi-cam (`switch_camera`), polish
  (`timeline.set_zoom` 1.0–1.25x punch-ins, `set_freeze_callout`,
  `set_audio_mode` loudnorm/isolate-2bus/silence), cards.
- **Act on the deliverable (7)** — the caption workbench:
  `captions.init_work_copy` → `apply_tag_edits` (gray-italic tags, gap-fill)
  → `list_work_cues` → `qa.run_caption_checks` (color-vs-speaker,
  owner-window tag law, loop-text, VAD gap coverage) → `undo.list_checkpoints`
  / `undo.rollback` (durable, plan-terminal) → `timeline.save` (.otio for
  compile).

Laws that make it safe to run unsupervised: every mutating step is
checkpointed BEFORE execution and auto-rolled-back on failure; workbench edits
land in a session workspace (shipped artifacts are never touched in place);
the agent must run its own QA gates before recommending a re-burn; and the
skill playbook (`packages/skills/bodycam_documentary.md` v0.4 "agent craft"
section) binds the decision table — evidence before action, measured numbers
in every `why`, subtle polish, ship = save.

Live proof: `agent_session_v1/v2` run the real PlannerExecutor loop with
native Gemini function calling on a working copy of the accepted Eustis cut —
evidence reads, caption-truth tag edits with self-QA, measured polish moves,
checkpointed timeline saves (see STATUS.md session log).

## Status

| Milestone | Scope | State |
| --- | --- | --- |
| M1 | Modular core engine + OTIO + segment caching | **closed** — 1080p demo (42s, 48kHz stereo), parallel segment fleet, ffprobe-validated cache (zero-change re-render: 4/4 hits, 0.1s) |
| M2 | Native agentic tool runtime + durable orchestration | **core landed** — 31 typed tools (cut/trim/split/move/narrate/blur/cards/multi-cam zoom·freeze·audio-mode mutations, evidence reads, caption-workbench actions with self-QA + undo + save), checkpoint/rollback store, planner-executor with auto-rollback, durable 5-stage pipeline (crash resume, ASR never re-run), Gladia ASR, shot boundaries + audio events, edge-tts `generate_tts`, SKILL.md prompt injection, SSE job progress, YouTube Private-Draft publisher + Human Review Gate |
| M3 | Director Studio Next.js UI | **closed + acceptance-verified** — cases, live SSE pipeline progress, timeline versions, checkpoint-first dispatch with rollback, **proxy scrub player (480p) + mouse-drawn blur canvas + editable subtitle grid**, Human Review Gate publish (Next.js 15, `apps/web`). M3 acceptance was executed live in a real browser: a blur box drawn over the frame corner survived to the next render (cache-invalidation A/B + pixel proof) and an edited cue re-aligned via difflib without drift (see `docs/m3_acceptance/` evidence) |
| M4 | Multimodal vision + Modal serverless GPU | **4-job vision stack landed, real-footage proven** — (1) CPU: YuNet face sweep + frame-quality pre-filter (dark/blur scoring); (2) VLM: 2-tier Gemini (`gemini-3.5-flash-lite` contact-sheet indexing / `gemini-3.8-flash` Frames-on-Demand QA) with **Modal-egress fallback for geo-blocked regions** (auto-retry, transient-503 backoff); (3) GPU fleet: `deploy/modal/redact_gpu.py` — YOLOv8-face + license-plate + ByteTrack persistent IDs → time-binned tracked `BlurRegion`s (modal free tier needs a payment method for T4; CPU-mode works today), YuNet fallback when fleet down; (4) AXON overlay OCR → intro card auto-population. Renderer: `enable='between(t,..)'` time-windowed blur (per-segment source→local translation) — pixel-verified on real footage (var 2.6 blurred vs 295.5 sharp). Full e2e on real Eustis FL bodycam: auto face blur + burned ASR subtitles delivered |
| M5 | Cloud deployment + multi-channel hardening | **scaffolding landed** — compose healthchecks/restart policies, cost dashboard endpoint (durations + tokens), multi-channel publish fan-out with honest `not_configured` states, `.env.example`, **kill-mid-encode crash-recovery drill executable** |

Live credentials (verified 2026-11-17): **Gladia cloud ASR is end-to-end green**
(real transcription + diarization through the v2 API, incl. the
`language_behaviour` removal and utterance-word flattening drift fixes).
**Full pipeline e2e-verified with live ASR**: TTS-built spoken fixture → ingest
(480p proxy + scrubsheet) → live Gladia transcription (25 words, diarized) →
auto-cut → segment-cached render → subtitle trio → QA audit all green
(`scripts/e2e_smoke.py`). The **LLM backends are live behind the existing
`LLMBackend` port**: **Meta Model API / Muse** (`agent_runtime/meta.py` — the
default: `LLM_BASE_URL=https://api.meta.ai/v1`, `LLM_MODEL=muse-ember-1.0`,
cost-engineered for the ~$20 credit balance: $0.10/M in / $0.004/M cached in /
$0.50/M out on Ember vs $1.25/$4.25 on Spark and $4/$20 on Wonder; every call
is priced from the real usage incl. prompt-cache hits, appended to a
persistent JSONL spend ledger, and a hard `LLM_BUDGET_USD` cap refuses calls
BEFORE the HTTP request; contributor variants — cheaper but Meta trains on
your content, a PII hazard for bodycam footage — are rejected unless
`LLM_ALLOW_TRAINING_ON_CONTENT=1`; per-proposal USD lands in
`agent_tool_logs.token_cost_usd` feeding the M5 cost dashboard), native
Gemini (`LLM_API_KEY`, REST function-calling with the three Gemini 3.x REST
traps handled — `parametersJsonSchema` full-JSON-Schema declarations,
verbatim `thoughtSignature` replay across multi-step tool turns, `?key=`
fallback on 401 `ACCESS_TOKEN_TYPE_UNSUPPORTED`) and any OpenAI-compatible
gateway (`LLM_BASE_URL`). `POST /projects/{id}/agent/plan` runs the real
provider loop. The sandbox IP is region-blocked by Google ("User location is
not supported"), which is why the default provider is Meta; the wire format
itself is MockTransport-pinned (`tests/unit/test_meta_backend.py`).
**Modal tokens validated** against the real API (`modal.App.lookup` auth round-
trip; `modal` added to deps). Remaining: YouTube refresh token
(`YOUTUBE_REFRESH_TOKEN` — run `scripts/gen_youtube_refresh_token.py` on any
machine with a browser; Google removed the OOB flow so this needs an
interactive consent screen), Modal GPU renders, and TikTok/Shorts/Reels API apps.

Legacy note: the previous v0.2 bodycam-autopilot prototype is archived on the
`legacy/v0.2-autopilot` branch. Do not build on it; PLAN.md supersedes it.
