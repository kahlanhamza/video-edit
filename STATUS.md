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
| 2026-11-18 | **redaction v2 (#14)** | User verdict: static union boxes are not tracking. Landed the deterministic 3-pass pipeline: CFR normalize; imgsz960/conf0.22 + GMC(sparseOptFlow)+track_buffer60 tracker (found+fixed a phantom-step birth bug via the whip-pan test); Pass2 repair (<=25f bridge +15%, >25f split, 5f lead-in/out, One-Euro, 1.35/1.40 envelope UNION raw); Pass3 per-frame feathered blur + debug_overlay; **zero-leak integration gate: 0 unblurred face frames pixel-verified on a fast-pan clip**. Worker seam REDACT_MODE=tracked bakes content-addressed redacted source. | 44 unit + 6 integration tests GREEN | c6dbd2f |
| 2026-11-18 | audio gating (#15) | speech-gated leveling: RMS strictly on ASR spans (siren fixture proves the meter is gated), gain clamp +-12dB, -1.5 dBTP alimiter AFTER gain; audit_technical speech_gated_loudness check. | 11 tests GREEN | 0871f4d |
| 2026-11-18 | qa+graphics (#17 #18) | evidence_fact middle-card (fact_value/label/SOURCE); safe_zone fractions -> card_safe_zone audit; clean_out_frames (blackdetect head/tail + freezedetect tail) in audit_technical — name-set contract test caught the new check (RED-first as designed). | 12 tests GREEN | 30dd40f |
| 2026-11-18 | exposure+retention (#16 #19) | night-rescue (g<1 lifts, g>1 darkens — direction nailed by tests; in-range clips untouched), match_exposure to set median; retention scoring (hook/acts/momentum ladder/open-loops/dead-spans). eq-gamma inversion bug caught pre-push. | 13 tests GREEN | 0e6bacd |
| 2026-11-18 | dev rules + record | SKILL.md (binding loop: bootstrap/wire-facts/issue-first/RED-GREEN/gates/push-always) + STATUS.md reconstructed from git history; CI RED triaged as format+5 latent mypy errors first. Issues #14-#19 filed with acceptance criteria; creds verified live (Kaggle 200, ElevenLabs 200), REDACT_MODE=tracked set. | rules now live in-repo | d745dcb, 0588544 |
| 2026-11-18 | nle shortcuts (#19) | Issue #19 was closed short — the shortcuts half never landed (no key handling anywhere in apps/web). Landed pure `nle_reducer` (J/K/L rungs 1→2→4, K stop re-arms 1x, in-drag/out-drag NLE convention, ms-quantized split dedupe, LIFO undo) + `useNLEShortcuts` hook (reverse shuttle = rAF scrub — HTML5 can't play backwards; typing guard; disabled during blur-box draw) wired in ProxyPlayer with in/out/split markers + key legend. Vitest+jsdom with a fake video element; CI gains a `web` job (tsc + vitest). Surprise: React 18 batches dispatches inside one `act()` — transport-effect assertions must flush per act block. | 11 reducer + 9 hook tests GREEN, tsc clean | (this commit) |
