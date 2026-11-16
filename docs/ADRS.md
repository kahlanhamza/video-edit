# ADRs — DirectorOS ratified decisions

Format: short, numbered, immutable once ratified; supersede by writing a new ADR.
PLAN.md (§1–§9) is the product/architecture source of truth; ADRs record *implementation*
decisions taken while executing it.

**ADR-0001 — PLAN.md is the single source of truth.** The uploaded RFC v1.0 is committed
verbatim at the repo root as `PLAN.md`. Code, issues and milestones map to it; PRs that
contradict it need a new ADR first. Legacy prototype archived at `legacy/v0.2-autopilot`.

**ADR-0002 — OpenTimelineIO is the canonical timeline.** Every edit decision lives as an
`otio.schema.Timeline` with a `director_os` metadata block per clip (PLAN §5.2). Custom
JSON EDL from the legacy prototype is dead. Export adapters (FCPXML/EDL) ride on OTIO.

**ADR-0003 — Segment cache is content-hashed, never time-stamped.** A segment's cache key
is a stable SHA-256 over its full render specification (source range, filter chain, blur
regions, card content hashes, codec/audio law settings). Changing one narrator word or one
blur sigma changes exactly one key — that is how "fix one word, re-render <5s" is true.

**ADR-0004 — Orchestration: Celery 5 + Redis now, Temporal later.** Big-tech-shaped but
sandbox-runnable: stage functions are plain Python, exposed as Celery tasks. The task
signatures are written so a Temporal workflow can wrap them without changes (durable
HITL pauses arrive with Temporal in M2/M5 per PLAN §4.2).

**ADR-0005 — The audio law is a contract, not a convention.** 48 kHz / stereo / LUFS
target are validated in three places: the Pydantic `AudioContract` model, the FFmpeg
command builder (`-af` must precede `-ar`/`-ac`), and the QA probe on the rendered file.
Violations fail CI, not the upload.

**ADR-0006 — Trunk-based development, push is part of the commit.** Direct commits to
`main` behind green CI (inherited from the legacy ADR-0010 war story). A lane is not done
until `git push` returns success.

**ADR-0007 — `apps/cli` is added to the PLAN §6 tree.** The `director` entrypoint
(`uv run director render <url>`) lives in a small `apps/cli` workspace member. Deviation
from the PLAN folder listing; everything else follows §6 verbatim.

**ADR-0008 — pgvector arrives with M2, JSON stands in for M1.** `scene_embeddings` is a
JSON column in M1 (sqlite-runnable, CI-friendly); the pgvector column type lands with the
Postgres migration work in M2. Schema column is named for the swap.

## ADR-0009 — Pipeline idempotency via domain-table persistence (2026-11-16)

**Status:** Accepted (M2)

**Context:** PLAN §8 M2 done-when: a worker crash mid-render must not re-run
ASR/TTS. Celery retries and operator re-runs are facts of life.

**Decision:** Every pipeline stage persists its product in a *domain* table
(`media_assets`, `perception_index`, `timeline_versions`, `segment_cache`)
BEFORE the job row advances. Resume = re-run the pipeline; each stage checks
its domain table and skips completed work. Soft failures also record
stage + error on `jobs`; hard kills (SIGKILL) persist nothing and still
resume correctly because no stage ever holds its only copy of finished work
in memory.

**Alternatives rejected:** workflow-engine state snapshots (Temporal adopts
this naturally later; Celery-only for now per ADR-0004); idempotency keys on
job rows (duplicates state that domain tables already hold).

**Consequences:** stage logic is naturally restartable and testable without
a broker; adding a stage means picking its domain table first.

## ADR-0010 — Publish gate is a Private Draft, never auto-public (2026-11-16)

**Status:** Accepted (M2)

**Decision:** The Human Review Gate outcome is a YouTube upload with
`privacyStatus=private`, `notifySubscribers=False`. A human presses Publish.
Chapter metadata is validated against YouTube's real rules (0:00 first, >=3
chapters, >=10s apart) at the API boundary so the Data API never rejects a
payload. `GoogleYouTubePublisher` lazy-imports googleapiclient; tests and
dry runs use `InMemoryYouTubePublisher` behind the same protocol.

**Consequences:** no token-scope surprises in CI; publishing is auditable
(`projects.youtube_draft_id`) and reversible before the human step.
