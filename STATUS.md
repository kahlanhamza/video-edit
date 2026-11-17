# STATUS.md — the record (single source of truth, ADR-0008)

Every lane appends one row. Dense, factual, surprises included. Pushed every task.

## Session Log

| Date | Lane | What happened | Outcome | Commits |
|---|---|---|---|---|
| 2026-11-14 | day 0 | PLAN.md v1.0 ratified — DirectorOS blueprint, M1–M5, 10-stage pipeline | spec-first, zero code | a6d80e5 |
| 2026-11-14 | M1 scaffold | uv monorepo, docker-compose, CI, Makefile, ADRs | engine skeleton | 6477f97 |
| 2026-11-14 | M1 core | contracts, OTIO compile, segment cache, render path | M1 engine landed | f07ea47 |
| 2026-11-15 | apps | FastAPI gateway, Celery+Redis fleet, CLI | service layer | df5547e |
| 2026-11-15 | guardrails | all 8 PLAN §9 traps as executable CI tests | 485d4e7, 08222fd |
| 2026-11-15 | render perf | parallel segment encoding + ffprobe-validated cache admission | d6ef039, 7f7a348 |
| 2026-11-15 | M2 | director editing tools, durable checkpoint/rollback, staged crash-resume pipeline, Gladia v2 ASR, YouTube draft publisher, SSE progress, shot/audio-event perception | M2 core landed | 0061f54, d18a536, 6c20aa5, 8dc7fb4, a6fd112, 7fd8a15 |
| 2026-11-15 | M2 close | layout/policy audit, subtitle stage, URL ingest, pgvector | M2 v1 complete | 6efe206 |
| 2026-11-16 | M3 | Next.js 15 Director Studio (cases, SSE, timeline versions, publish gate) | studio surface | 491341b |
| 2026-11-16 | M4 | map_route Playwright card + vision adapter port | cards | c339007 |
| 2026-11-16 | M2 live | **real** Gemini REST + OpenAI-compatible LLMBackend; Gladia live e2e GREEN | live LLM verified | fd9a23f, 1a60379 |
| 2026-11-16 | M3 acceptance | studio e2e with live ASR, browser-verified | M3 accepted | b1de51f |
| 2026-11-17 | M4 faces | YuNet sweep -> clustered regions; raw->render blur coordinate mapping; **real Eustis FL bodycam e2e** | first real-footage run | ec912c1 |
| 2026-11-17 | ci+zoom | CI branch filter fixed; zoompan punch-in + freeze-callout | — | fcce0e6, 3a6566f, 6158e15, ea89635 |
| 2026-11-17 | M4 multicam | audio xcorr sync ±1ms, switch_camera tool, crash drill | — | 42c4ba8, 2e859cf, e268326, 7656c69 |
| 2026-11-17 | M4 vision | 4-job stack (CPU pre-filter, 2-tier Gemini + Modal egress, ByteTrack fleet, AXON OCR); ComputeBackend port (LocalOnnxCpu default, KaggleBatchGpu, Modal dormant) + ElevenLabs TTS | stack wired | 4c8f25d, 39b01e6 |
| 2026-11-17 | **CI RED** | runs 44112010637/44125589107 RED — wire facts: 12 stale-format files + 5 latent mypy errors; all gates GREEN locally, fixed, pushed | CI restored | d745dcb |
