# RUNBOOK — Tape Production (the laws, the failures, the procedure)

Written 2026-11-25 after tape-1 v1 (release `newlondon-tape1-v1`) was rejected
by the owner and rebuilt bug-by-bug. Purpose: the next production run starts
from this document — every failure below is already fixed and committed; do
not re-derive, do not improvise. Each fix cites its commit.

Commits in the chain: `11802a2` (duration band + short-doc threshold + geo
marker + driver standing laws), `e21c211` (16384-token planning budget),
`c6c1e9a` (even blur plane + raw-budget law), `b712ad8` (test parser fix).

---

## 1. What went wrong in tape-1 v1 (and the fix that landed)

| # | Symptom | Root cause | Fix (commit) |
|---|---------|-----------|--------------|
| 1 | 251s master from a 933s source ("4-minute stupid video") | `doc_scale` kept 28-35% of ANY source: 933s x 0.28 = 261s | `doc_scale.py`: sub-25-min sources keep **50-65% of raw** (the niche skill's editorial contract, owner-ratified); sources >= 25min keep the Eustis long-form band EXACTLY (2627s -> [735.6, 900]). Tape 1 now plans 466-606s. Pinned in `test_doc_scale_adapts_to_any_length` (`11802a2`) |
| 2 | Narration was generic Eustis-template text ("the next forty minutes", "tests that turn suspicion into numbers") | FOUR stacked causes, each sufficient alone: (a) `LLM_MODEL` missing from the env -> `build_backend_chain` raised -> `_director_backend` returned None -> deterministic fallback silently won; (b) the Gemini tier's geo-block error text carried no capacity marker -> `FailoverBackend` re-raised instead of walking to OpenRouter; (c) the OpenRouter tier envs were never configured; (d) `plan_macro` used the 4096-token default — the free-tier model burned the whole budget on prose (12058 chars of reasoning, ZERO braces) and every retry died "unparseable" | (a)+(c) env contract below; (b) `"geo-block"` joined `_CAPACITY_ERROR_MARKERS` + wire-fact test `test_gemini_geo_block_fails_over_to_openrouter` (`11802a2`); (d) `plan_macro` requests `max_tokens=16384` (the notebook's proven budget) + spy test (`e21c211`) |
| 3 | The model ignored the raw budget: 1336s of raw authored for a 933s tape (ratio 0.18 — physically impossible) | prompt law alone cannot be trusted on free tiers | `enforce_raw_budget` in `director_session.py`: the word-count law extended to raw windows — the Director's SELECTION is kept, lengths rescale proportionally to `doc_scale.target_hi * 0.7`. Tape-1: 1336s -> 425s raw, execute 27/27 steps, retention GREEN (`c6c1e9a`) |
| 4 | Static yunet boxes instead of the tracked blur | `POST_FX` defaults False and `REDACT_MODE` unset -> hermetic defaults | `produce_tape.py` setdefaults the production standing laws at startup: `POST_FX=1`, `AUDIO_MODE=isolate_2bus`, `AUDIO_ENHANCER=deepfilter`, `REDACT_MODE=tracked` (`11802a2`) |
| 5 | No audio enhancement / noise removal | post-fx stage (tracked blur + deep-filter + caption burn on the ASSEMBLED master, owner rule 2026-11-19) never ran because POST_FX=0 | same standing laws as #4; post-fx passes are `.done`-marker resumable |
| 6 | Badge bottom-right hid the captions | driver default | `--corner` default `top_left` (owner directive: the badge never hides captions) (`11802a2`) |
| 7 | Segment render died: `boxblur radius 21 must be < 21` | a 43px tracked region computed radius from the raw box; the yuv420p crop plane rounds DOWN to even (42) | `_safe_sigmas` derives from the even plane + `test_redact_blur_plane.py` sweeps every emitted radius (`c6c1e9a`) |
| 8 | 480s tapes would blow the narration ratio (0.58) | short-doc threshold (240s) rode the old 28% band | threshold raised to **400s** — short bridges engage below a ~400s master (`11802a2`) |

## 2. The env contract (`.secrets/produce_env.sh`, gitignored — names only here)

Required exports for a production run (values live in the secrets file):

- `GLADIA_API_KEY` — ASR
- `LLM_API_KEY` + `LLM_MODEL=gemini-3.5-flash-lite` + `GEMINI_FALLBACK_MODELS` — primary tier (geo-blocked in this sandbox: it FAILS OVER, do not remove it)
- `OPENROUTER_API_KEY` + `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1` + `OPENROUTER_MODEL=google/gemma-4-31b-it:free` + `OPENROUTER_FALLBACK_MODELS` — the tier that actually serves planning here (notebook-proven JSON model; nemotron burned its budget on prose)
- `DATABASE_URL` — the production sqlite
- The driver itself setdefaults `POST_FX / AUDIO_MODE / AUDIO_ENHANCER / REDACT_MODE` — never pass hermetic overrides in production.

Live-probe before committing to a long run (30s, saves hours):

```
source .secrets/produce_env.sh
.venv/bin/python -c "from director_os.agent_runtime.llm import build_backend_chain; \
 from director_os.agent_runtime.registry import ToolRegistry; \
 c = build_backend_chain(ToolRegistry()); print(c.tier_names); \
 print(c.complete_json('Reply with exactly {\"ok\":true}', 'JSON only.'))"
```

Expect: tiers `['gemini:...', 'openai_compat:...']` and `{"ok":true}`.

## 3. The production procedure (tape = one owner-approved pick)

1. **Source**: `bin/yt-dlp-nightly --cookies .secrets/yt_cookies.txt -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b" --merge-output-format mp4 -o footage/<slug>.mp4 <url>` (cookies + Deno in `bin/` defeat the bot wall; probe with ffprobe — record duration/geometry, they must match the dossier).
2. **Dossier**: verified case facts in `docs/cases/<case>.json` (9 facts, 2 citations for tape 1) — the Director authors narration from THIS, never invented.
3. **Run**: `scripts/produce_tape.py --source footage/<slug>.mp4 --title ... --slug ... --render 1920x1080 --dossier-json docs/cases/<case>.json --case-facts "..."` (fresh project; then `--project-id <id>` to resume).
4. **Stages, all resumable**: gate -> caption-guard -> ingest (proxy+scrubsheet) -> yunet sweep (8 regions) -> Gladia ASR (persisted per asset, never re-uploads) -> proofread (deterministic floor; LLM supervisor is an upgrade, not a gate) -> planning -> TTS (edge-tts, content-addressed cache) -> render (segment cache, per-segment `.json` keys) -> post-fx (captions -> CFR -> tracked blur chunks -> deep-filter -> mux, `.done` markers) -> QA audit -> finalize (v6 captions + badge in ONE re-encode) -> deliverables in `download/` + production report JSON.

## 4. The window laws (the sandbox kills processes at ~10 min)

- The tool window is ~555s. Long stages WILL be killed mid-flight. Everything is resumable EXCEPT planning-in-flight — so:
- **Planning is the only stage to pre-run standalone**: `/home/z/my-project/scripts/plan_standalone.py` runs `plan_macro` with a DISK CACHE (`scratch/plan_<pid>.json`), applies `enforce_raw_budget`, runs Phase B `execute` + retention, and commits the timeline version (`created_by=director_agent`). The driver's `active_otio_version_id` shortcut then skips planning on resume. If Phase A must re-generate, delete the cache file and re-run the script (it retries 4x internally with feedback).
- **Never judge progress by stdout alone**: run with `PYTHONUNBUFFERED=1`, but the truth is in the DB — `select level, message from job_logs where job_id=...` and `timeline_versions` (`created_by` says `director_agent` vs `assembler_agent`; the note says which plan won).
- Narration TTS clips land in `output/pipeline/<pid>/narration/*.json` — READ THEM before shipping: case-specific narration names the case (Kokoszka / Bergeson / the audit); generic bridges mention "forty minutes" and "the roadside". A generic narration = the fallback won = do NOT ship.
- Blur chunks: `post_fx/blur_chunk_NN.done` per chunk; deep-filter + mux follow automatically.

## 5. The QA gate (all lines must be GREEN on the FINAL file)

- `streams_present`, `audio_law_48k_stereo`, `stream_uniformity`
- `clean_out_frames` (head/tail black = 0.00, tail freeze = 0.00)
- `black_frame_budget` (0.00s, max 3.0s), `duration_drift` (tol 1.0s)
- `final_duration_floor` (25% of source), `narrator_floor`, `raw_floor`
- `narration_ratio` **0.30..0.52**, `coverage_span` (0.55; 0.70 for >= 1200s sources)
- caption-guard median < 0.012 (burned captions on the SOURCE tape)
- foreign-narration windows (if any) excluded from raw — the report lists every replaced window

Frame verification before release: sample frames at 8-12 points — badge TOP LEFT, v6 captions (dialogue white / suspect gold / narrator cyan), tracked blur on faces, authentic Axon overlays intact.

## 6. The release law (owner directive 2026-11-24, after the first master was lost)

**Render -> GitHub Release -> byte-verify -> only then everything else.**

1. Create the release (tag `newlondon-tape1-vN`) with the small assets (report, srt, ass).
2. Upload the master MP4 (`uploads.github.com`, Content-Type video/mp4).
3. Byte-verify: `state == "uploaded"`, size == local size, ranged GETs of head 2MB + tail match the local bytes (a plain HEAD parse lies — the asset URL 302s to S3).
4. STATUS.md row + push + CI green via the GitHub API.
5. The sandbox recycles: NOTHING in `download/` or `output/` survives between sessions. The release is the only durable home.

## 7. Sandbox drift checklist (every session start)

- `git config core.fileMode false` (the recycle flips every file to 100755 — fake dirt)
- `git fetch origin && git merge --ff-only origin/main` (local may lag the pushed tip)
- venv drift repair: `uv pip install -q --python .venv/bin/python python-multipart botocore onnxruntime cryptography PyMuPDF beautifulsoup4 trafilatura lxml boto3 python-docx`
- run pytest on the touched test files BEFORE the full suite (drift shows as collection errors)
- GitHub API list-releases occasionally 404s transiently — retry before debugging auth

## 8. The editorial laws (one line each, all enforced in code)

- Keep 50-65% of raw for sub-25-min sources; 720-900s band for >= 25min (`doc_scale.py`)
- Narration: 9-13 blocks for tape-1 scale, word bounds cold 40-65 / mid 48-85 / outro 70-105, ratio 0.30..0.52 (`doc_scale.py` + `plan_invariants.py`)
- Raw budget = `target_hi * 0.7`, enforced deterministically (`enforce_raw_budget`)
- Hook: first narration <= 8s, context <= 32s, sting <= 4s
- Coverage: clip starts span the whole file (first 15% / final 25%)
- Captions: ONE master-clock builder (TTS stamps for narration, remapped ASR for raw, narrator-on-broll isolation); v6 typography law; badge top-left
- Post-fx on the ASSEMBLED master only (tracked blur + deep-filter with the muted-speech rescue gate + caption burn); never on the raw source
- Content gate FIRST (strict allowlist: shoplifting / trespassing / first_amendment_audit; violence vetoed at plan time)
- The Director authors narration from the CaseDossier only; never invent facts
