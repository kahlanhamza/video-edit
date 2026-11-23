# CRITIQUE — full-project adversarial review (2026-11-23)

Owner directive: question every line of code, every UI affordance, every design
decision, the architecture, the UX — critic-loop until the flaws are found and
the blockers to **a publishable YouTube video this week** are removed.

Method: three parallel adversarial audits (UI/UX across all 28 web files;
security/ops across the whole repo; test-gap/reliability across the 987-test
suite), then **every CRITICAL/HIGH finding re-verified by hand against the
source before it was accepted** — two audit claims were corrected during
verification (one severity, one mechanism). Unverified claims do not appear
below. Fixes landed in the same session; each fix carries a regression test
named after its finding ID.

The niche law holds throughout: this is NOT a general video editor — it exists
for content that can be narrated or commented on (bodycam documentary et al.).
Every finding below was judged against that niche, not against generic-editor
feature parity.

---

## The one-sentence verdict

The engine is real and disciplined (987 tests, cache laws, checkpoint
rollback, honest refusals) — but the **last mile was fake**: the publish
button returned a simulated video id, the export gate could never go clean on
default projects, and the ship pack baked the wrong master. All three are now
fixed and pinned by tests.

## Findings

| ID | Sev | Area | Finding | Status |
|---|---|---|---|---|
| CR-001 | CRITICAL | pipeline | `build_ship_pack` received the PRE-post-fx master (`report.final_path`) while QA shipped the post-fx one — uploads from the pack silently lost burned captions, tracked blur, enhanced audio | **FIXED** — pack source is `final_for_qa`; `test_ship_pack_built_from_post_fx_master` pins the wiring with a marker-artifact post-fx stub |
| CR-002 | CRITICAL | pipeline/API | export-gate cursor only stamped inside the tracked-redaction path; with the default REDACT_MODE no render ever stamped → gate "dirty" forever, Export button untrustworthy, publish 409s a baked program | **FIXED** — `_stamp_export_gate` runs after EVERY master render; `test_gate_stamped_without_tracked_redaction` runs a real pipeline and asserts the served gate is clean |
| CR-003 | CRITICAL | API/UI | publish last mile lied twice: `_publisher` was the InMemory stub in production (fake `draft_…` id, project flipped to `published_draft` with nothing uploaded) and the publish form demanded a hand-typed server filesystem path | **FIXED** — vault-backed publisher resolution (YOUTUBE_* keys → real `GoogleYouTubePublisher`, else honest `dry_run: true` that persists NOTHING); `GET /deliverables` serves the real shippable files; the Studio publish view prefills from it and labels simulations as simulations |
| CR-004 | HIGH | UI | Export button was a disguised pipeline restart — both buttons called `runOrResume`, and a clean gate still rendered a clickable "Exported ✓" that re-ran everything | **FIXED** — own `doExport` handler; disabled when `exported_once && !dirty` |
| CR-005 | HIGH | UI | "Director tools" panel fired hardcoded destructive dispatches ("trim master clip 0" → 3 s) with no context, no confirmation — dev leftovers presented as product | **FIXED** — panel is now "Agent activity": the dispatch/rollback log + guidance; the blind buttons are gone |
| CR-006 | HIGH | UI/niche | multi-camera composer advertised 1..N bodycam videos but dispatched only URL[0]; naively looping ingest would launch a second full pipeline on the same project and CLOBBER the active timeline version | **FIXED (honest semantics)** — camera 1 is the primary cut; extra URLs are kept as case sources with an explicit note; true multi-angle sync stays an OPEN design item (below) |
| CR-007 | HIGH | data | checkpoint version race: SELECT-max/INSERT with no unique constraint — concurrent dispatches could duplicate `version_num` and corrupt the rollback chain + gate comparisons | **FIXED** — `UniqueConstraint(project_id, version_num)` + migration 0008 (dedupe-then-index, idempotent) + retry-on-IntegrityError in `CheckpointStore.checkpoint`; 4-thread race test asserts gap-free unique 1..40 |
| CR-008 | HIGH | security | vault encryption could never activate: `cryptography` missing from all deps → `SETTINGS_SECRET_KEY` silently degraded to plaintext (keys incl. YouTube refresh token readable in the DB); setup.sh never generated the key | **FIXED** — dependency added; setup.sh auto-generates the key; `test_real_encryption_activates_with_dependency` proves Fernet at rest (gAAAA… prefix, roundtrip) |
| CR-009 | HIGH | UI | `running` was welded to SSE liveness — a dropped stream re-enabled Run/Export mid-run and froze progress with no signal | **FIXED** — running = server truth first (newest job unfinished & not at gate); 5 s server poll while the stream is dead; "reconnecting…" chip |
| CR-010 | MEDIUM | UI | cases front door never refreshed — finished cases invisible without F5 | **FIXED** — 10 s visible-tab poll + visibilitychange refetch |
| CR-011 | MEDIUM | ops | every service published 0.0.0.0 with default creds (db/redis/minio/sam2) | **FIXED** — db/redis/minio/sam2 bound to 127.0.0.1; api/web stay 0.0.0.0 deliberately (owner tests the Studio from a phone on the LAN) |
| CR-012 | MEDIUM | UI | the A1/CC "mute" button called the same `setHidden` as the eye toggle — audio monitoring never existed, the label lied | **FIXED** — one honest control (eye = collapse) |
| CR-013 | MEDIUM | UI | track headers scrolled away (unlabeled tracks after any horizontal scroll) and ruler ticks went blank past the first viewport (no scroll listener) | **FIXED** — pinned header column overlay (inert during drags so trims can't be cancelled by a mouseleave) + rAF-throttled scroll-driven tick window |
| CR-014 | MEDIUM | UX | `.eustis_pid` sandbox junk was git-tracked | **FIXED** — untracked + ignored |
| CR-015 | MEDIUM | tests | `sandbox_stack_check.py` was orphaned (referenced nowhere) | documented below — still manual; wired into docs this pass (the lane row + CRITIQUE record the runbook: boot :8011 with sqlite env, run the script) |

Test count after this pass: **993 pytest** (all green) + 80 vitest + mypy 127
files + ruff clean + `next build` clean + the real-HTTP stack check ALL GREEN
+ a live curl drill proving `deliverables` honesty and the 409 dirty-gate law.

## What was deliberately NOT fixed (open, prioritized for the publish goal)

1. **Real `GoogleYouTubePublisher` has never talked to YouTube** (stub-tested
   only). First real upload = first execution. Mitigation for this week: the
   ship pack (`ship_pack/final_1080p.mp4` + chapters + captions) is the
   primary publish path — manual upload to YouTube Studio works TODAY even
   with the API path unexercised. Cheapest next test: mocked
   `googleapiclient.discovery` happy-path, then one manual draft upload drill.
2. **Nothing renders at 1080p in CI** (all real renders are 320×240/640×360).
   The ship pack's name promises 1080p. Cheapest test: one 1080p 4 s fixture
   render probing w/h/codec.
3. **POST-FX internals untested** (`stage_post_fx` 4-pass state machine: .done
   resume, speech-keep-ratio gate, mux drift) — its failures are soft-skipped
   by design, so a broken pass degrades quietly. The wiring is now pinned
   (CR-001); the internals need a tmp-dir drive with a fake enhancer.
4. **Timeline cannot move/delete clips from the UI** (trim/split only) and
   version history has no restore affordance — the reviewer's editorial power
   is thinner than the Export tooltip promises. Server tools exist
   (`timeline.move_clip/remove_clip`, checkpoint list); this is pure UI work.
5. **No auth on the gateway** (LAN-exposed by design this week) — a shared
   bearer-token middleware is a half-day job the moment anything leaves the
   LAN; `/agent/dispatch` also executes mutating tools without
   confirm-before-execute (prompt-injection surface when exposed).
6. **Ingest SSRF hardening**: URL fetches validate scheme only — no
   private-address block, no byte cap; uploads have no size limit. Acceptable
   for a single-user host this week; mandatory before any exposure.
7. **Subtitle editor ergonomics** (raw-ms inputs, no cue delete, CC track not
   refreshed after edits) — captions are make-or-break for the niche; the
   store/API laws are right, the grid is the weak surface.
8. **Monitoring depth**: keyboard legend promises S/I/O actions that are
   partially decorative; trim rejections are silent; no drag ghost; zoom has
   no anchor/follow-playhead. Polish, not blockers.

## Architecture verdict (the deep question)

The load-bearing design decisions survive the critique: op-fold + append-only
history, checkpoint-before-execute with rollback, segment-cache content
addressing, dual-axis export gate (ops rev + timeline version), the human
gate as a stage, and the niche skill as the contract. What the critique
exposed is a **trust boundary problem, not an architecture problem**: three
surfaces (Export button, publish response, ship pack content) claimed things
the disk did not carry. The fix pattern is the same everywhere and now
applied: **the response of every terminal action is derived from server truth
about the artifact, never from intent**. That law is the thing to keep
enforcing in review: any new button that speaks for the disk must read the
disk.
