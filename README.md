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
| `packages/agent_runtime` | Typed tool registry (native function-calling engine) |
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

## Status

| Milestone | Scope | State |
| --- | --- | --- |
| M1 | Modular core engine + OTIO + segment caching | **closed** — 1080p demo (42s, 48kHz stereo), parallel segment fleet, ffprobe-validated cache (zero-change re-render: 4/4 hits, 0.1s) |
| M2 | Native agentic tool runtime + durable orchestration | **core landed** — 8 typed director tools, checkpoint/rollback store, planner-executor with auto-rollback, durable 5-stage pipeline (crash resume, ASR never re-run), Gladia ASR, shot boundaries + audio events, edge-tts `generate_tts`, SKILL.md prompt injection, SSE job progress, YouTube Private-Draft publisher + Human Review Gate |
| M3 | Director Studio Next.js UI | **landed** — cases, live SSE pipeline progress, timeline version history, checkpoint-first tool dispatch with visible rollback, Human Review Gate publish (Next.js 15, `apps/web`) |
| M4 | Multimodal vision + Modal serverless GPU | **local surface landed** — `map_route` card (Playwright HTML→PNG + deterministic Pillow fallback), vision LLM adapter port (provider pending key), Modal GPU app scaffold |
| M5 | Cloud deployment + multi-channel hardening | **scaffolding landed** — compose healthchecks/restart policies, cost dashboard endpoint (durations + tokens), multi-channel publish fan-out with honest `not_configured` states, `.env.example` |

Remaining across milestones (all external-credential-gated):
Pydantic AI LLM wiring (needs `LLM_API_KEY` — the tool registry, checkpointing,
dispatch endpoint, loop and SKILL.md injection are live), real Gladia key for
cloud ASR (stub path already green), YouTube OAuth for real drafts (in-memory
publisher green), Modal tokens for GPU renders, and real TikTok/Shorts/Reels
API apps for multi-channel.

Legacy note: the previous v0.2 bodycam-autopilot prototype is archived on the
`legacy/v0.2-autopilot` branch. Do not build on it; PLAN.md supersedes it.
