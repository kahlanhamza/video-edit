# DirectorOS — Repository Index & Architecture Walkthrough

> Purpose of this file: a single orientation map for the whole repo — what every
> package does, how a run flows end-to-end, where the laws live, and where to
> change things. Read this before exploring file-by-file.
>
> Source of truth for **product/architecture**: `PLAN.md`. Source of truth for
> **what actually works / history**: `STATUS.md`. Source of truth for
> **engineering rules**: `SKILL.md`. Implementation decisions: `docs/ADRS.md`.

---

## 1. What this is

**DirectorOS** is an autonomous multimodal video-editing and documentary
platform. It takes long-form public-record footage (police bodycam, dashcam,
commentary), perceives it (ASR + diarization, shot boundaries, audio events,
faces/plates, VLM frame understanding, OCR), lets an LLM **direct** a cut using a
typed tool API, renders through FFmpeg with a content-addressed segment cache,
adds broadcast audio + captions + redaction, runs deterministic QA, and parks
the result at a human review gate before any publish.

One-line mental model:

```text
raw video ──► perception index ──► LLM Director (typed tools) ──► OTIO timeline
         ──► compiled render DAG ──► cached segments ──► concat master
         ──► post-FX (blur + audio + captions) ──► QA gates ──► human gate
```

Everything edit-decision-shaped lives on an **OpenTimelineIO** timeline; every
external capability sits behind a **port** (LLM, compute, TTS, publisher); every
paid/heavy stage persists its product before advancing so a crash resumes free.

---

## 2. Repository map

```text
video-edit/
├── PLAN.md                  Architecture RFC v1.0 — milestones M1–M5, 10-stage pipeline
├── STATUS.md                Dated session log — the real record of what shipped
├── SKILL.md                 Binding engineering rules (push, CI, redaction/audio/cost laws)
├── README.md                Product overview + quickstart + agent-tool summary
├── pyproject.toml           uv workspace root (Python 3.12, ruff/mypy/pytest config)
├── docker-compose.yml       postgres+pgvector, redis, minio, api, worker
├── Makefile                 setup / lint / type / test / up / api / worker / render-fixture
├── .env.example             Full config surface (LLM routing, vision, audio, TTS, Kaggle/Modal)
│
├── packages/                Pure Python libraries (no app framework)
│   ├── core/                Config, Pydantic contracts, SQLAlchemy models, storage,
│   │                        ffmpeg args, proxy builder, QA probe, Kaggle client
│   ├── otio_engine/         OTIO timeline builder + metadata contract + compiler +
│   │                        segment cache + renderer (FFmpeg DAG execution)
│   ├── audio_engine/        Audio law chains, 2-bus speech isolation, VAD, leveling,
│   │                        SNR measurement, TTS port, A/B export
│   ├── subtitles/           difflib proofread alignment, 2-row speaker-colored ASS,
│   │                        cue store
│   ├── vision_fx/           Cards, blur/redaction, tracked redaction (3-pass),
│   │                        trajectory repair, watermark/exposure/safe-zone,
│   │                        compute backends (local ONNX / Kaggle / Modal)
│   ├── perception/          Gladia ASR, shots, audio events, faces, vision index,
│   │                        timestamp OCR, multicam sync, CaseDossier
│   ├── agent_runtime/       Typed tool registry, editing/read/workshop/memory tools,
│   │                        planner-executor, checkpoints, critic, story map,
│   │                        Director session, fallback cut, LLM backends/routing
│   └── skills/              Niche playbooks (bodycam_documentary.md) + loader
│
├── apps/
│   ├── api/                 FastAPI gateway: projects, jobs/SSE, timeline, agent
│   │                        dispatch/plan, sources/dossier, subtitles, redactions,
│   │                        ingest, approve, costs, publish
│   ├── worker/              Celery app + durable pipeline stages + post-FX +
│   │                        ship pack + audio enhance + director autodrive
│   ├── cli/                 `director render <url|path>` (M1 one-command path)
│   └── web/                 Next.js 15 "Director Studio" (cases, SSE, player,
│                            subtitle grid, blur canvas, NLE shortcuts)
│
├── gpu_services/            Placeholder/README (Modal code actually lives in deploy/)
├── deploy/modal/            Serverless GPU: redact_gpu.py, vlm_proxy.py, gpu_render.py
├── scripts/                 Ops/acceptance/e2e probes (real-footage runs, evals,
│                            caption v4, post-FX v3, TTS normalization, etc.)
├── tests/unit/              89 hermetic test modules (contracts, laws, tools, backends)
├── tests/integration/       4 heavier tests (two-bus, fixture render, redact, deepfilter)
├── docs/                    ADRs, CRITIQUE.md (adversarial review + open items), research, M3 acceptance, this file
└── demo/                    demo_1080p.py — the M1 full-res demonstration
```

---

## 3. Layered architecture

| Layer | Code | Responsibility |
|---|---|---|
| UI | `apps/web` | Create a case, watch SSE progress, scrub proxy, edit captions, draw blur boxes, dispatch tools, approve, publish draft |
| Gateway | `apps/api/main.py` | REST + SSE, DB access, enqueue jobs, agent dispatch/plan, human gate |
| Orchestration | `apps/worker` (`pipeline.py`, `tasks.py`, `celery_app.py`) | Durable 10-stage state machine; Celery tasks for fan-out and resume |
| Agent runtime | `packages/agent_runtime` | Typed 35-tool registry, planner-executor, checkpoints/rollback, LLM routing, Director session, critic, memory |
| Domain engines | `packages/{otio_engine,audio_engine,subtitles,vision_fx,perception}` | Timeline, render, audio, captions, vision, redaction, perception |
| Contracts/persistence | `packages/core` | Pydantic domain models, SQLAlchemy tables, settings, storage, ffmpeg guards, QA probe |
| External ports | `agent_runtime.llm`, `vision_fx.compute`, `audio_engine.tts`, `api.youtube`, `api.channels` | LLM, compute, TTS, publishers — adapters + fakes |
| Serverless GPU | `deploy/modal` | Redaction fleet, VLM egress proxy, optional GPU render |

Dependency direction is one-way: `apps → packages`; `core` is the bottom;
`otio_engine` depends on `core`/`vision_fx`/`audio_engine`; `agent_runtime`
depends on `otio_engine`.

---

## 4. Persistence model (`packages/core/db.py`)

SQLite in CI/dev, Postgres 16 + pgvector in compose. Tables:

| Table | Holds | Resume role |
|---|---|---|
| `projects` | title, niche, status, `active_otio_version_id` | Current cut pointer |
| `media_assets` | source path/URL, role, duration/fps/size, proxy/audio keys | Ingest idempotency |
| `perception_index` | `transcript_json`, `shots_json`, `audio_events_json`, `redaction_tracks_json`, embeddings | ASR never re-runs; redaction plan lives here |
| `timeline_versions` | version_num, full `otio_json`, author, note | Checkpoint/undo history |
| `segment_cache` | segment hash → chunk key, duration, codec profile | Content-addressed render reuse |
| `agent_tool_logs` | tool, args, result, duration, tokens, USD | Cost dashboard + audit trail |
| `jobs` | current stage, progress %, error | Operator resume + SSE |
| `case_sources` | PDF/URL/text/source-page + extraction status | Multi-source dossier |
| `case_dossiers` | latest synthesized `dossier_json` | Director grounding |

Alembic migrations live in `packages/core/migrations/versions/` (0001 init …
0004 case sources). `init_models()` creates tables for dev/CI.

---

## 5. End-to-end run — actual code path

Entry points:

- CLI: `uv run director render <source>` → `apps/cli/.../main.py` (M1 path:
  ingest → probe → OTIO → compile → render → concat → QA).
- API: `POST /projects/{id}/ingest` downloads, creates a `jobs` row, enqueues
  `directoros.advance_job`.
- Worker: `advance_job` → `run_pipeline` (`apps/worker/.../pipeline.py`).

`run_pipeline` order (the real one):

1. **Ingest** — `stage_ingest`: probe, register `media_assets`, build 480p
   proxy + scrubsheet + 64k audio. Idempotent per project+source.
2. **Redaction sweep** — only if `REDACT_MODE`/`redact_backend` set:
   `stage_redaction` → `tracked` 3-pass, `modal` fleet, or `yunet` fallback.
   Persists regions + per-frame plan to `perception_index`.
3. **Perception / ASR** — `stage_perception`: Gladia (or injected fake),
   persisted transcript short-circuits every re-run.
4. **Proofread** — `stage_proofread`: police-jargon corrections + speaker roles
   via difflib-preserving alignment; persisted under `transcript_json["proofread"]`.
5. **Vision enrichment** — optional `stage_vision_enrichment`: frame quality,
   YAMNet audio events, 2-tier VLM index + Frames-on-Demand, AXON OCR.
6. **Planning** — `stage_planning`: reuses `active_otio_version_id` unless
   `force_replan`; otherwise short-source auto-cut **or** documentary path (see §6).
7. **Legacy Director autodrive** — optional, short sources only; soft-fails to
   the deterministic cut.
8. **Narration** — `stage_narration`: content-addressed TTS clips, then
   `retime_narration_to_tts` + `retime_cards_after_narration` so real speech
   durations drive the timeline.
9. **Audio prep** — only for `isolate_2bus`: YAMNet music routing → Demucs
   vocals stem when flagged → `stage_audio_enhance` **lazily per kept window**.
10. **Render** — `stage_render`: save `.otio`, optionally bake tracked redaction
    per kept clip, `compile_timeline` → `render_dag` (parallel misses, cache
    hits skipped, `concat -c copy`).
11. **Subtitles** — `stage_subtitles`: proofread words → cue beats (≤6 words /
    ≤3.5 s / speaker change / >1 s hole) → `.ass`, `.srt`, `.json`.
12. **Post-FX** — optional `stage_post_fx`: captions on the assembled master,
    tracked blur sweep + trajectory repair + zero-leak audit, audio enhance,
    final mux. Resumable via `.done` markers.
13. **QA** — `stage_qa`: `audit_technical` + retention audit for long sources;
    on failure status `qa_failed`, and a retention failure **quarantines** the
    master as `*.qa_failed.mp4`.
14. **Ship pack** — `build_ship_pack`: final MP4, chapters, `.srt`, SEO copy,
    thumbnails, verification hashes. Soft-fails.
15. **Human gate** — job parked at `human_gate`, project `awaiting_review`.
    `POST /approve` re-renders only dirty segments.

Every stage persists its product **before** advancing the job row (ADR-0009), so
re-invoking `run_pipeline` is the recovery mechanism. A hard kill loses only
in-memory state.

---

## 6. Two planning paths

### Short sources (`< 480 s`)

`stage_planning` slices the source into ≤8 s raw clips (`AUTO_CUT_MAX_SEGMENT_S`)
and adds a location card. Deterministic, free, no LLM required.

### Documentary path (`>= 480 s`, `DOCUMENTARY_MIN_SOURCE_S`)

`_plan_documentary`:

1. **Story map** — `story_map.build_story_map` condenses the *whole* tape into
   ~1-minute buckets (speech highlights, speakers, shot density, audio events).
2. **Dossier** — `_load_dossier` / `stage_dossier` gives verified case facts.
3. **Phase A (macro plan)** — `DirectorSession.plan_macro` sends story map +
   dossier in one JSON completion; returns `MacroPlan` (acts, raw clips,
   narration scripts, cards). Narration duration is **computed from word count**
   (~2.6 w/s), never trusted from the LLM.
4. **Phase A.5 (deterministic truth)** — snap clip edges to ASR utterance
   boundaries, split internal dead air >1.8 s, optionally one quote-grounded
   refinement round.
5. **Phase B (tool execution)** — `expand_macro_plan` → `timeline.insert_raw_clip`
   / `timeline.add_narration` / `cards.add` steps executed act-by-act through the
   checkpointed registry, then a refinement loop.
6. **Gate** — `plan_invariants.audit_retention` (length-scaled floors, source
   span, narration ratio). Failure → retry with feedback → deterministic
   `fallback_cut.build_fallback_documentary` (Cold Open + 6-act sampler). The
   session can never return a truncated timeline.

Length adaptation lives in `agent_runtime/doc_scale.py`: an 8-minute tape plans
a short doc, a 43-minute tape plans a 10–15-minute doc.

---

## 7. Agent runtime — the 35-tool editor

`packages/agent_runtime` is the differentiating layer. It is **not** MCP/A2A;
it is a typed in-process registry (ADR-0011).

### Tool surface (35)

| Group | Count | Tools |
|---|---|---|
| **Mutate** (`editing_tools.py`) | 15 | `timeline.trim_clip`, `insert_raw_clip`, `split_clip`, `remove_clip`, `move_clip`, `switch_camera`, `set_zoom`, `set_freeze_callout`, `set_audio_mode`; `narration.add`, `narration.set_text`; `redaction.add_blur`, `redaction.remove_blur`; `cards.add` |
| **Read** (`inspection_tools.py`) | 9 | `project.describe`, `qa.read_report`, `transcript.read_window`, `captions.list_cues`, `audio.speech_spans`, `captions.propose_tag_window`, `timeline.list_items`, `timeline.get_item`, `agent.finish` |
| **Workshop** (`workshop_tools.py`) | 7 | `captions.init_work_copy`, `apply_tag_edits`, `list_work_cues`, `qa.run_caption_checks`, `undo.list_checkpoints`, `undo.rollback`, `timeline.save` |
| **Memory** (`session_memory.py`) | 3 | `session.plan`, `session.notes`, `session.status` |
| **Critic** (`critic.py`) | 1 | `qa.self_audit` (honest-absence: never reports "passed" for a check that did not run) |

### Execution loop

- `ToolRegistry` (`registry.py`): decorator or explicit registration; input
  schema is a Pydantic model; `dispatch()` validates **before** the function runs;
  `openai_schemas()` emits function-calling JSON Schema.
- `PlannerExecutor` (`agents.py`): **checkpoint before every mutating step**,
  dispatch, log; on any validation/semantic failure → durable rollback to the
  pre-step version and stop. Rollback writes a new version (history is never
  erased). A terminal checkpoint makes the post-plan state durable.
- `CheckpointStore` (`checkpoint.py`): timeline versions + `agent_tool_logs`.
- `DirectorSession` (`director_session.py`): Phase A/B flow described in §6.
- `story_map.py`, `fallback_cut.py`, `plan_invariants.py`, `doc_scale.py`:
  documentary structure, deterministic fallback, length-scaled invariants.
- `prompting.py` + `packages/skills`: inject the niche playbook into the system
  prompt (`bodycam_documentary.md`).

### LLM routing (`llm.py`, `meta.py`)

Port: `LLMBackend.propose(state, role) -> list[ToolStep]`; optional
`record_results()` for multi-turn replay. Adapters:

- `GeminiRestBackend` — native AI Studio REST; handles `parametersJsonSchema`,
  verbatim `thoughtSignature` replay, `?key=` 401 fallback, geo-block → Modal
  egress proxy, capacity fallbacks.
- `OpenAICompatBackend` — any `/chat/completions` endpoint (Groq, OpenRouter,
  vLLM, NIM); sanitizes dots in function names for NIM.
- `MetaModelBackend` (`meta.py`) — Meta Muse; cached-token pricing, persistent
  JSONL spend ledger, hard `LLM_BUDGET_USD` cap **before** the HTTP call,
  refuses contributor models unless explicitly opted in.
- `FailoverBackend` + `build_backend_chain()` — capacity-only failover across
  model chains and tiers; 400 contract errors re-raise instead of masking bugs.
- `ScriptedBackend` — hermetic test fake.

---

## 8. Timeline & render engine

### OTIO contract (`otio_engine/contract.py`)

Tracks: `V1_master`, `V2_overlay_cards`, `A1_narration`. Every clip carries
`metadata["director_os"]`:

- `RawClipMeta`: asset_id, audio_mode (`loudnorm_48k` | `isolate_2bus` |
  `silence`), trim_silence, zoom_keyframes, blur_regions, freeze_callout,
  angle_id, act_name.
- `NarratorClipMeta`: TTS provider/voice, script_text, b-roll asset/range/windows,
  audio volume, J/L cut, act_name.
- `CardMeta`: template, fields, fade.

`parse_clip_meta()` is the only sanctioned reader; raw dict access is a contract
violation.

### Compiler (`otio_engine/compiler.py`)

Walks `V1_master`, resolves sources (including per-clip baked redacted sources),
maps blur boxes from raw footage pixels into letterboxed render space, translates
tracked blur source-time windows into segment-local time, resolves freeze
callouts + spotlight PNGs, places cards collision-free on owning segments, hashes
every segment, and validates uniform codec/audio params so `concat -c copy` is
safe.

### Segment cache (`otio_engine/segment_cache.py`)

Key = SHA-256 over the full render spec (source range, filters, blur regions,
card content, codec/audio settings). Changing one word or sigma changes exactly
one key — this is why "fix one word, re-render in seconds" is true (ADR-0003).
Chunks are admitted only after ffprobe validation, so truncated encodes never
poison the cache.

### Renderer (`otio_engine/renderer.py`)

Builds one FFmpeg command per segment with the audio law flags (`-ar 48000 -ac 2`
after filters), exact slicing (`-ss start -t dur`, never `-to`), multi-b-roll
concat for narration, card/spotlight overlay inputs, 2-bus or TTS audio routing,
then renders misses in a thread pool and concatenates in DAG order. Celery
fan-out equivalents live in `apps/worker/tasks.py`.

---

## 9. Audio law (`packages/audio_engine`)

- **Global law**: 48 kHz stereo everywhere; loudnorm; speech-gated leveling
  (−12…−6 dBFS RMS on spoken intervals only); −1.5 dBTP limiter.
- **Bus-1 speech isolation** (`isolate_2bus`): neural extractor (deep-filter
  default, ClearerVoice optional) → tone restoration (mud/clarity/presence/air)
  → compression → VAD+ASR speech-gated duck → gain refinement → limiter.
  No silent fallback to `afftdn`; a missing neural denoiser is a hard fail.
- **Verification**: `snr.py` measures clean SNR, noise-floor drop, timbre tilt,
  crest factor, clipping. `qa_probe.audit_technical` enforces
  `snr_verification`, `timbre_not_muffled`, `natural_dynamics`, `zero_clipping`,
  `speech_gated_loudness`.
- **TTS** (`tts.py`): `TtsBackend` port; Edge TTS default ($0), ElevenLabs
  optional with word timestamps.
- **VAD** (`vad.py`): Silero ONNX for speech-presence truth.
- **Music guard**: YAMNet music spans route a Demucs vocals stem into the
  enhancer before any heavy processing.

---

## 10. Subtitles (`packages/subtitles`)

- `proofread.py`: police-jargon fixes + speaker-role tagging.
- `align.py`: `difflib.SequenceMatcher` re-alignment after text edits — never
  linear interpolation (guardrail 5).
- `ass_builder.py`: 3–5 word karaoke beats; simultaneous speakers ride two lanes
  (`MarginV=46` / `MarginV=96`); exact upright font filenames; profanity masking.
- `store.py`: JSON cue store backing the Studio subtitle grid.
- Caption-truth v4 (scripts + agent workshop): gray-italic tags for
  unaudible/ASR-loop/owner windows via inline ASS override; 5 agent gates
  (color-vs-speaker, owner-window tag-only, loop text, VAD gap coverage,
  no-tags-on-clean-speech).

---

## 11. Vision & redaction (`packages/vision_fx`, `packages/perception`)

Redaction is a deterministic 3-pass pipeline, **not static boxes** (SKILL §3):

- **Pass 0** `normalize_cfr`: strict 30 fps so detection frame `i` == render
  frame `i`.
- **Pass 1** detect + track: YOLOv8n-face / plate ONNX (or Modal fleet),
  imgsz 960, conf ≈0.22, BoT-SORT-style tracker with global motion compensation
  and `track_buffer=60`.
- **Pass 2** `trajectory.py`: bridge gaps ≤25 frames (+15% inflation), split
  larger holes, 5-frame lead-in/out, One-Euro smoothing, envelope
  `scale_w=1.35` / `scale_h=1.40` that must cover the raw box.
- **Pass 3** `redact_track.py`: per-frame feathered elliptical blur/mosaic,
  optional debug overlay, zero-leak audit (`assert_zero_leaks`) — a fast-pan
  integration test proves 0 unblurred face frames.

Compute backends (`vision_fx/compute`): `LocalOnnxCpuBackend` default ($0),
`KaggleBatchGpuBackend` optional free T4, `ModalGpuBackend` dormant. One
`COMPUTE_BACKEND` env switch; ports + adapters, fakes in tests.

Other vision: frame-quality pre-filter, YAMNet audio events, 2-tier Gemini VLM
index + Frames-on-Demand, AXON timestamp OCR, multi-cam audio cross-correlation
sync, CaseDossier extraction/synthesis, watermark detection, exposure rescue,
safe-zone geometry, glass cards, map-route Playwright cards.

---

## 12. QA gates (`packages/core/qa_probe.py`)

`audit_technical` checks (enabled by inputs):

- `streams_present`, `audio_law_48k_stereo`, `black_frame_budget`,
  `duration_drift`, `stream_uniformity`, `clean_out_frames`
- Audio: `speech_gated_loudness`, `enhancer_engine_used`, `snr_verification`,
  `timbre_not_muffled`, `natural_dynamics`, `zero_clipping`
- Layout/policy: `card_policy`, `card_safe_zone`, `blur_policy`,
  `subtitle_lanes`
- Retention (long sources): length-scaled master duration, narration count/ratio,
  raw clip count, source coverage. Failure quarantines the master.

Agent-side critic (`qa.self_audit`) adds structural/zoom/plan/caption verdicts
with honest-absence semantics.

---

## 13. API surface (`apps/api/main.py`)

| Method & path | Purpose |
|---|---|
| `GET /health` | liveness |
| `POST/GET /projects`, `GET /projects/{id}` | create/list/read cases |
| `POST /projects/{id}/ingest` | URL download + enqueue pipeline job |
| `GET /projects/{id}/jobs`, `.../jobs/{job}/events` | job list + SSE progress |
| `GET /projects/{id}/timeline` | active version + version history |
| `POST /projects/{id}/agent/dispatch` | one checkpoint-first tool call |
| `POST /projects/{id}/agent/plan` | full LLM propose→execute loop |
| `GET/POST /projects/{id}/sources`, `.../sources/upload`, `.../sources/refresh` | multi-source case ingest |
| `GET /projects/{id}/dossier` | synthesized case brief |
| `GET /projects/{id}/assets`, `/assets/{id}/proxy`, `/assets/{id}/scrubsheet` | studio media |
| `GET/PATCH /projects/{id}/subtitles`, `.../subtitles/srt` | caption grid + export |
| `GET/POST/DELETE /projects/{id}/redactions` | blur canvas |
| `POST /projects/{id}/approve` | human gate → dirty-segment re-render |
| `POST /projects/{id}/publish-draft`, `.../publish-channels` | private YouTube draft / multi-channel |
| `GET /projects/{id}/costs` | durations + token spend |

Publishing is always a **private draft** (ADR-0010); `InMemoryYouTubePublisher`
is the test fake, `GoogleYouTubePublisher` the real adapter.

---

## 14. Web UI (`apps/web`)

Next.js 15 + React 19 + Tailwind 4, no component library.

- `src/app/page.tsx` — case creation: title, video URLs, attached sources
  (URL/text/PDF upload), auto-discover.
- `src/app/projects/[id]/page.tsx` — the studio: sources + dossier, ingest,
  SSE job progress, timeline versions, tool dispatch, proxy player, subtitle
  grid, redaction canvas, approve, publish.
- `src/components/ProxyPlayer.tsx` — 480p scrub player + blur-box drawing +
  NLE markers/legend.
- `src/components/SubtitleGrid.tsx` — inline cue editing.
- `src/hooks/useNLEShortcuts.ts` + `src/lib/nle_reducer.ts` — J/K/L shuttle
  rungs (1→2→4), in/out drag, ms-quantized split, LIFO undo, typing guard.
- `src/lib/api.ts` — typed fetch client for every gateway endpoint.
- Tests: Vitest + jsdom (reducer + hook), `tsc --noEmit` in CI.

---

## 15. Configuration (`.env.example` → `core/config.py`)

| Group | Keys |
|---|---|
| Core | `DATABASE_URL`, `REDIS_URL`, `OUTPUT_DIR`, `STORAGE_BACKEND` |
| Storage | `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` |
| Ingest identity | `CONTACT_EMAIL` (descriptive User-Agent) |
| ASR | `GLADIA_API_KEY` |
| LLM | `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_PROVIDER`, `LLM_BUDGET_USD`, `LLM_SPEND_LEDGER`, `LLM_ALLOW_TRAINING_ON_CONTENT`, `LLM_FALLBACK_MODELS`, `GEMINI_FALLBACK_MODELS`, `OPENROUTER_*` |
| Vision | `VISION_API_KEY`, `VISION_PROVIDER`, `VISION_MODEL`, `VISION_FALLBACK_MODELS` |
| Compute | `COMPUTE_BACKEND` (`local`/`kaggle`/`modal`), `KAGGLE_USERNAME`, `KAGGLE_KEY`, `MODAL_TOKEN_ID/SECRET` |
| TTS | `TTS_PROVIDER`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID` |
| Audio | `AUDIO_MODE`, `AUDIO_ENHANCER`, `DEEPFILTER_BIN`, `AUDIO_BUS2_VOLUME`, `DIRECTOR_OS_ALLOW_AFFTDN` |
| Redaction | `REDACT_MODE` (`tracked`/`yunet`/`modal`), `REDACT_BAKE`, `REDACT_KEEP_TRACKS`, `REDACT_TRACK_POLICY` |
| Pipeline | `DIRECTOR_AUTODRIVE`, `POST_FX`, `FPS`, `RES_W`, `RES_H`, `TARGET_LUFS` |
| Web | `NEXT_PUBLIC_API_BASE` |
| YouTube | `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN` |

Defaults are deliberately $0: SQLite, local storage, Edge TTS, local ONNX CPU,
no Modal card.

---

## 16. Tests & CI

- `tests/unit/` — 89 modules, hermetic (no network/GPU/model downloads).
  Guardrails G1–G8 are executable: 48k lock, exact slicing, ingest UA, speech
  trimming, difflib alignment, two-row ASS, upright fonts, licensing.
- `tests/integration/` — 4 heavier: fixture render, two-bus pipeline, tracked
  redaction, real DeepFilter (skipped without the binary).
- `scripts/agent_evals.py` — 6 hermetic agent scenarios (evidence-before-action,
  fail-closed feedback, tag law, critic damage detection, memory resume, undo),
  also bound into pytest.
- CI (`.github/workflows/ci.yml`): `ruff check` + `ruff format --check`, `mypy
  packages apps`, `pytest`; separate `web` job runs `tsc` + Vitest.
- Local gates: `make lint`, `make type`, `make test`. SKILL.md requires green
  before every push.

---

## 17. How to run

```bash
# Python workspace
uv sync
make lint type test

# Local services (Postgres+pgvector, Redis, MinIO, API, worker)
docker compose up -d

# API + worker without Docker
make api
make worker

# M1 one-command render
uv run director render tests/fixtures/fixture_30s.mp4 --max-seconds 12 --preset 360p

# Studio UI
cd apps/web && npm ci && npm run dev
```

Typical production env for real bodycam runs:
`AUDIO_MODE=isolate_2bus`, `AUDIO_ENHANCER=auto`, `REDACT_MODE=tracked`,
`POST_FX=1`, `DIRECTOR_AUTODRIVE=1` (short sources), LLM keys set.

---

## 18. Invariants and gotchas (read before changing code)

1. **Audio**: `-ar 48000 -ac 2` must come *after* `-af`/`loudnorm`; exact
   slicing is `-ss start -t dur`, never `-to`.
2. **Timeline**: all edit state is OTIO + `metadata["director_os"]`; go through
   `attach`/`parse_clip_meta`.
3. **Cache**: never hand-build a segment key; `segment_hash(spec)` is the only
   source. A spec change that doesn't change the key is a silent stale render.
4. **Redaction**: any PR touching it re-runs the zero-leak integration test.
   "Looked fine" is not evidence.
5. **Privacy**: plates are never keepable; an empty tracked plan must never be
   persisted as success.
6. **Agent mutations**: always checkpoint before dispatch; rollback writes a new
   version, never deletes history.
7. **Narration duration**: computed from word count (~2.6 w/s), never trusted
   from the LLM; real TTS retime wins after synthesis.
8. **Retention**: long-source masters are gated on length-scaled floors and
   quarantined on failure.
9. **Costs**: every LLM call priced from real usage into the ledger; the budget
   cap refuses before the HTTP request.
10. **Coordinates**: detections/studio canvases are raw-footage pixels; render
    effects are letterboxed render space; map via `scale_bbox`, never eyeball.
11. **Ports**: new external service = Protocol first, adapter second, fake third.
12. **Push discipline**: work that exists only in the working tree does not exist
    (SKILL.md §0).

---

## 19. Suggested reading order

1. `README.md` → `PLAN.md` §1–3, §7–9 → `STATUS.md` (newest rows first).
2. `packages/core/models.py` + `db.py` — the domain language.
3. `packages/otio_engine/{contract,timeline,compiler,renderer,segment_cache}.py`
   — how an edit becomes an MP4.
4. `apps/worker/pipeline.py` — `run_pipeline` and the stage functions.
5. `packages/agent_runtime/{registry,agents,editing_tools,inspection_tools,
   workshop_tools,director_session,llm}.py` — the AI editor.
6. `packages/audio_engine/two_bus.py` + `core/qa_probe.py` — the laws and proof.
7. `packages/vision_fx/{redact_track,trajectory}.py` + `compute/` — privacy.
8. `apps/api/main.py` → `apps/web/src/lib/api.ts` → studio pages.
9. `tests/unit/test_g1..g8` — the guardrails as executable specs.

---

## 20. Change map (where to touch what)

| Task | Start here |
|---|---|
| Add an agent tool | `agent_runtime/editing_tools.py` (mutate) or `inspection_tools.py` (read); register with a Pydantic input model; add a hermetic test |
| Change timeline metadata | `otio_engine/contract.py` + every consumer via `parse_clip_meta` |
| Change render behavior | `otio_engine/compiler.py` (spec) + `renderer.py` (FFmpeg cmd) + `segment_cache.py` (hashing) |
| Add a pipeline stage | `worker/pipeline.py` + a domain table to persist its product (ADR-0009) |
| Change documentary structure | `agent_runtime/{story_map,doc_scale,plan_invariants,director_session,fallback_cut}.py` |
| Change LLM provider/routing | `agent_runtime/llm.py`, `meta.py`; keep `LLMBackend` protocol |
| Change audio treatment | `audio_engine/{two_bus,enhance,leveling}.py` + QA gates in `qa_probe.py`; bump `ENHANCE_CACHE_VERSION` |
| Change captions | `subtitles/{proofread,align,ass_builder}.py` + `worker/post_fx.py` |
| Change redaction | `vision_fx/{redact_track,trajectory}.py` + `perception/faces.py`; re-run zero-leak test |
| Add an API endpoint | `apps/api/main.py` + `apps/web/src/lib/api.ts` + UI |
| Add a niche | `packages/skills/` playbook + `niche_skill` on the project |

---

*Generated from a full read of the repo at HEAD `516595b` (2026-11-21). If the
code and this document disagree, the code wins — update this file.*
