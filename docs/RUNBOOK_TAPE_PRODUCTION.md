# PLAYBOOK — DirectorOS Production (every issue we hit, how we solved it, the laws that stand)

Written 2026-11-24 as the tape-production runbook; **expanded 2026-11-25 into the full playbook**
after the owner called out what it missed: the caption saga (v4 -> v5 -> v6), async captions,
caption tags, the LLM decision chain, and the rest of the 15-day failure history. Sources for
every line: the 55+ commits of 2026-11-11..2026-11-25, all 19 GitHub issues (#1-#19), CI runs
#108-#147, docs/CRITIQUE.md (CR-001..015), and docs/ADRS.md. The next production run starts
from this document. Do not re-derive, do not improvise. Each entry cites its commit.

## 0. The standing laws (violating any of these is how the failures below happened)

1. **REPO-FIRST LAW** (the owner's #1 rule): the repo has 7+ days of commits, issues and CI
   workflows that already solve these problems. Before writing ANY script or improvising a
   local fix, search the repo for the committed solution and use it. "We have solutions but you
   choose to improvise with local scripts instead of ours" — never again.
2. **RELEASE LAW** (owner directive 2026-11-24, after the first master was lost):
   **render -> GitHub Release -> byte-verify -> only then everything else.** Nothing in
   `download/` or `output/` survives a sandbox recycle. The release is the only durable home.
3. **CI LAW**: GitHub API check-runs are the only truth ("CI green" = check-runs `success` via
   the API, not local claims). Run `ruff check` + `ruff format --check` BEFORE every push —
   three pushes in this window died on the format gate alone (2026-11-21 twice, 2026-11-25
   once). RED is fixed forward, never bypassed (ADR-0007: gates are non-bypassable).
4. **WINDOW LAW**: the sandbox kills processes at ~10 min (~555s usable). Long stages must be
   resumable; planning must run standalone (section 5). Background processes do NOT survive
   across messages — run synchronously with `timeout 555`.
5. **NICHE LAW** (67faf48): this editor is NOT general-purpose. Narration/commentary bodycam
   niche only, laws pinned in code (`doc_scale.py`, `plan_invariants.py`, SKILL.md).
6. **DUAL-MODALITY LAW** (e9994f8): the agent acts live AND the human refines afterwards.
   Agent ops stay human-adjustable (`POST /ops/{id}/adjust`); a wrong box is adjusted, never
   delete+redraw (that would lose the provenance chain).
7. **NO-FAKE-DATA LAW**: every number in a report/release/STATUS row comes from a real run in
   this session. QA gates evaluate the OUTPUT FILE itself (the M5-R1 lesson), not planner intent.
8. **BADGE LAW** (owner directive 2026-11-25): the LENS & LAW channel watermark goes **top-left**
   — bottom-right hides captions. Default `top_left` in `scripts/produce_tape.py`.
9. **SPEAK LAW**: narration is authored from the CaseDossier only, in our voice — never a
   copy-paste of the source channel's (Eustis-style) narration template. Generic bridges
   ("the next forty minutes", "the roadside") = the deterministic fallback won = do NOT ship.

---

## 1. CAPTIONS & ASYNC CAPTIONS (the full v4 -> v5 -> v6 saga)

This is the most repeatedly-broken subsystem. The owner rejected captions THREE separate times
for THREE different root causes. All three are now pinned by tests.

### 1.1 The v4 rejection: wrong colors, hallucinated lines, silent speech (2026-11-21, a374d91)
Owner verdict on the Eustis master: officer speech rendered in YELLOW (the Suspect color);
609.4-628.9s was an ASR hallucination loop ("people want to stop" x4 over barely-audible
murmurs); and VAD-detected speech existed with ZERO captions on it.
**Fix — the VAD-evidence pipeline** (`scripts/caption_v4.py`, `scripts/vad_evidence.py`,
`scripts/caption_evidence.py`): webrtcvad 30ms @16k + voice-band evidence; a caption may only
exist where evidence says speech exists, role color comes from diarization evidence, and
hallucination loops are caught against the audio. QA grew to 11 gates, all green.

### 1.2 The v5 regression: ASYNC captions — source-clock cues on the master (2026-11-22, 0d56deb)
The owner's logs from the first full 43-min run showed captions synced to the 43-min RAW tape,
not the edited master. Root cause: a cue list built off the 43-minute SOURCE clock pasted onto
the ~10-minute master — raw dialogue captions landing over narration b-roll because BOTH were
read at source-time. (The owner also rejected the style: Roboto 20pt under PlayRes 1920x1080 on
a 720p master = ~13px illegible text.)
**Fix — the v6 law, one shared implementation** (`packages/subtitles/src/director_os/subtitles/master_cues.py`):
- Captions are timed on the **EDITED TIMELINE (the master clock)**, never the raw source clock.
- NARRATOR cues come from the **TTS word sidecars** (the voiceover NEVER comes from ASR),
  rebased onto the master clock per narration block.
- RAW speech cues come from the proofread transcript, **remapped through each kept raw clip's
  source->master window** (source start + clip offset).
- **Narrator-on-broll isolation gate**: narration blocks (b-roll visuals, muted bed) NEVER carry
  raw captions.
- Cue chunking rides the caption_v4 law: <=5 words / <=27 chars / >1s gap break; a raw-speech
  cue never rides longer than `MAX_CUE_SPAN_S = 6.0`.
- One truth, three consumers: `post_fx` burns exactly these cues, `stage_subtitles` persists
  exactly these cues into the Studio store ({json,ass,srt}), the grid edits exactly these cues.

### 1.3 The committed typography law (pinned in `tests/unit/test_g6_caption_v6_law.py`)
- DejaVu Sans 40 @ PlayRes 1280x720 (NOT Roboto 20 @ 1080).
- Role colors: Officer = white (&H00FFFFFF), Suspect = gold (&H004DD8FF, #FFD84D),
  Narrator = cyan (&H00FFE470, #70E4FF).
- Karaoke `\k` centisecond beats from aligned word timings; `\fad(40,30)`; outline 2.9 /
  shadow 1.6.
- **Karaoke SECONDARY colors pinned** (1660103, found live in the 480s e2e acceptance): the
  unsung-word tint pysubs2 defaults to (pure red) was NOT the approved look — committed values
  are officer &H00CCC0B8 / suspect &H004D9BB3 / narrator &H00B8A37C; a test asserts no style
  line carries the pysubs2 red default anymore.
- **Flash guard**: `FLASH_MIN_S = 0.30` — a cue shorter than that is a subliminal flash; it is
  extended to the floor, words survive, flashes die (`ass_builder.py`).

### 1.4 Proofread (the deterministic floor + the LLM supervisor)
- `subtitles.proofread` (jargon map + **difflib re-timing** + Officer/Suspect/Narrator roles) is
  a durable pipeline stage; narration TTS clips are content-addressed and persisted before render
  (a19284d).
- The LLM operator supervisor (`supervisor.py`, `decisions_v6.json`) audits roles line-by-line,
  splits mixed diarization at word level, rule-tags ASR ghosts — but it is an **upgrade, not a
  gate**: when the LLM returns nothing usable, "deterministic floor rides on" (see 3.4).
- The owner's manual sandbox workflow was the reference — d6b939e ported the notebook editorial
  workflow into the pipeline so the pipeline matches what the owner verified by hand.

### 1.5 Caption sync is EDITABLE, not just burnable (Studio surface, 2026-11-23)
- `b22fa07` — **cue retime**: the subtitle grid edits TIME, not just words (`start_ms`/`end_ms`
  set TOGETHER — half a window is a corrupt cue).
- `708e514` — **caption op write-through**: a dead data path was found — `apply_caption_overrides`
  was consumed NOWHERE outside tests; the grid, the CC track and the render path all read the raw
  cue store. Killed: the store IS the baked truth.
- CI note: the caption-guard test fixture must be **font-independent** (ad6f2d7) — the CI runner
  lacks Noto; the PIL fallback font scored 0.009 < 0.012 and turned run #140 RED for a phantom
  reason. Calibrate the fixture, not the threshold.

---

## 2. CAPTION TAGS (what the owner means by "caption tags")

Two tag systems exist and both are law:

### 2.1 The proofread-stage tags (postfx_v3, 8e1c5f8)
Owner review finding: unintelligible audio must be TAGGED, never filled with hallucinated text.
The LLM proofread decision prompt (verbatim contract):
`{"id": <int>, "clean_text": "<corrected text>", "tag": null or "inaudible"|"mumbling"|"cross-talk"}`
- A pure-garble / hallucinated / unintelligible line: `clean_text` -> `""` and the tag set
  ("mumbling" if they trail off, "cross-talk" if two people overlap).
- The builder renders the tags as `*inaudible*` / `*mumbling*` / `*cross-talk*` in the burned
  captions; the report counts them (`n inaudible-tagged`).

### 2.2 The agent workshop tags (workshop_tools.py)
- Vocabulary (5 tags, Pydantic-validated): `unaudible | mumbling | whispering | cross-talk |
  panicking` — each edit needs start_s/end_s ordered, non-overlapping, with a reason
  (min length enforced).
- `captions.apply_tag_edits` rewrites windows as **gray-italic `TAG_OVERRIDE` cues** on the WORK
  COPY; **gap-fill law**: speech without a cue still gets the gray tag (a tag is inserted even
  where the cue store had nothing).
- Owner law: tag edits are agent-op work on the work copy; **only the owner's re-burn stage
  touches the master**.

### 2.3 The tag QA gates (found by evals, refined by wire facts)
- `no_tags_on_clean_speech` (3f4be47 — the 5th caption-truth gate, FOUND BY THE EVAL HARNESS):
  fires when a tag cue rides a window where the speech evidence says clean speech.
- Refinement 52f24f1: it fires ONLY when a tag REPLACED real text — an evidence window that
  merely contains a tag alongside real text must not false-fire.
- `9452fbb` — the color-vs-role audit judges only FULLY-CONTAINED cues: a cue straddling a
  window boundary no longer false-fails the audit.

---

## 3. LLM DECISIONS (the decision chain — every way it silently failed and the fix)

The owner's tape-1 complaint "the narration is a copy-paste of the old Eustis style" was not a
prompting bug — FOUR stacked LLM-decision failures each sufficed alone to make the deterministic
fallback win. All four are now closed, and the whole chain is pinned by tests.

### 3.1 Tier chain and the capacity/contract split (`agent_runtime/llm.py`)
- Ordered provider tiers behind one `LLMBackend` port: Groq (openai-compat, model chain inside
  the tier) -> OpenRouter (:free) -> native Gemini. A tier that raises a CAPACITY error hands
  over to the next tier; a CONTRACT error (400 shape) raises — a contract bug must never be
  papered over by failover.
- `_CAPACITY_ERROR_MARKERS`: 429/500/503, dead-key 401/403, tier exhaustion, AND **"geo-block"**
  — the Google AI Studio 403 "geo-blocked from this region" is an AVAILABILITY condition, not a
  contract bug (11802a2 added it after the Gemini tier re-raised instead of walking to
  OpenRouter; wire-fact test `test_gemini_geo_block_fails_over_to_openrouter`).
- **Sticky egress**: on the first geo-block the chain transparently switches to the Modal fleet
  egress, STICKY for the session (`MODAL_TOKEN_ID/SECRET` required for that path).
- **Within-tier model fallback**: `complete_json` walks the tier's model fallbacks
  (`GEMINI_FALLBACK_MODELS` / `OPENROUTER_FALLBACK_MODELS`) — a geo/capacity failure on one model
  downgrades to the next BEFORE the tier ever raises. Only full exhaustion reaches the tier above.

### 3.2 The silent-fallback trap (tape-1 root cause #1)
`LLM_MODEL` missing from the env -> `build_backend_chain` raised -> `_director_backend` returned
None -> the deterministic fallback silently won -> generic Eustis-template narration. NOTHING in
the logs screamed. **Fix**: the env contract (section 6) + the 30s live-probe BEFORE committing
to a long run (expect tiers `['gemini:...', 'openai_compat:...']` and `{"ok":true}`).

### 3.3 The token-budget trap (tape-1 root cause #4, e21c211)
`plan_macro` used the 4096-token default — the free-tier model burned the entire budget on prose
(12058 chars of reasoning, ZERO braces) and every retry died "unparseable". **Fix**: `plan_macro`
requests `max_tokens=16384` (the notebook's proven budget) + a spy test asserts the request.

### 3.4 Prompt budget and supervisor decisions
- Groq free-tier TPM ceiling: the composed prompt stays inside ~6k chars; `LLM_PROMPT_CHAR_BUDGET`
  overrides (clamp applies to Groq only).
- The supervisor (`supervise_transcript`) batches transcript lines -> one decision list per batch
  -> `decisions_v6.json`; failed batches are tolerated; when NO usable decisions come back the
  deterministic floor rides on (supervisor = upgrade, not gate — never let an LLM outage block
  the pipeline).
- `bd97893` — **fail-closed WITH feedback**: unknown tool proposals are rejected with a
  corrective message back to the model (so it can retry correctly), never a bare veto.
- `5a38ecf` — v4 rate-limit-proof live runs: the tier chain proven end-to-end on a real session.
- **Operator-as-LLM** (2026-11-21, M6 live): when Gemini was rate-limited the session's LLM role
  was played by the operator assistant (provider=`operator-llm`, zero external LLM) — the full
  pipeline ran without skips. The LLM role is a role, not a vendor.
- Costs: `agent_tool_logs` token spend + `GET /projects/{id}/costs`; `LLM_BUDGET_USD` (default
  15) pre-call budget cap shared by LLM + vision; the credentials vault (d6905f6) serves ANY
  OpenAI-compatible provider with live connect-test probes and zero-restart rotation (ADR-0012).
- Durable session memory (ab68fb8): `session.plan / session.notes / session.status` — the plan
  and its decisions survive the session.

---

## 4. REDACTION & TRACKING (static boxes -> tracked blur, bug by bug)

- **Owner verdict #14 (2026-11-18): static union boxes are NOT tracking.** Landed the
  deterministic 3-pass pipeline: CFR normalize -> yolov8n-face imgsz960/conf0.22 +
  GMC(sparseOptFlow)+track_buffer60 -> Pass-2 repair (<=25f bridge +15%, >25f split, 5f
  hysteresis) -> per-frame tracked blur. Re-verified on real pixels: 7 tracks, conf 0.22-0.87,
  person_1 spans 11.6s with 104 real hits, audit 0 leaks.
- **The empty-plan bug (real footage)**: the lindevs yolov8n-face ONNX pins its input at
  640x640 but the sweep fed 960x960 -> onnxruntime INVALID_ARGUMENT on EVERY frame; the sweep's
  per-frame `except Exception` SWALLOWED it -> an empty redaction plan. Fixes: `resolve_input_size`
  (fixed exports win over the request), detector exceptions PROPAGATE (fail loud),
  `median_flow_shift` reshapes before median, and the worker **refuses to persist an empty plan**
  (an empty sweep is a broken sweep, not an empty scene).
- **The blur-plane law (c6c1e9a)**: segment render died `boxblur radius 21 must be < 21` — a
  43px tracked region computed its radius from the raw box, but the yuv420p crop plane rounds
  DOWN to even (42). `_safe_sigmas` derives from the even plane; `test_redact_blur_plane.py`
  sweeps every emitted radius (parser fixed in b712ad8 — regex, not colon-splitting).
- **Chunked-resumable blur render** (326880e): long blur bakes survive window kills via
  `post_fx/blur_chunk_NN.done` markers; deep-filter + mux follow automatically.
- **Raw->render coordinate mapping** (ec912c1): studio-canvas and detector boxes are authored in
  RAW-footage pixels; the renderer blurs AFTER scale+pad, so `scale_region` +
  `compile_timeline(asset_geometry=...)` map boxes letterbox-aware into render space. Before this
  fix, ANY render not at source resolution misplaced or silently dropped every redaction box.
- **Post-plan redaction scoping** (52cad6e — the operator mandate): what gets redacted is
  decided after planning, per kept window; per-track keep opt-outs (`REDACT_KEEP_TRACKS`,
  Studio `keep_track_ids`); plates are NEVER keepable.
- **SAM2 + per-object tracking**: local weight service (accebb6 — sha-pinned, volume-cached,
  >1MB stub rejection), promptable selections (a9309fc), frame-scoped op baking (3f88f82),
  manual box adjust (e9994f8), FX-window retime (0d4314c).
- **YuNet is the CPU sweep** (ec912c1): OpenCV 5.0 removed Haar cascades entirely;
  `cv2.FaceDetectorYN` (232 KB model bundled) is the supported local detector; YOLO+ByteTrack is
  the occlusion-hard upgrade lane.
- **Watermark law**: `brand_watermark.py` (LENS & LAW RGBA badge burn) is a DIFFERENT module from
  the pre-existing source-watermark detector (`watermark.py`, untouched). Badge corner default
  `top_left` (owner directive 2026-11-25: the badge never hides captions).

---

## 5. AUDIO (enhancement + leveling + the measured findings)

- **Tape-1 root cause**: post-fx never ran because `POST_FX=0` hermetic default. The driver now
  setdefaults the standing laws at startup: `POST_FX=1`, `AUDIO_MODE=isolate_2bus`,
  `AUDIO_ENHANCER=deepfilter`, `REDACT_MODE=tracked` (11802a2). Post-fx passes are
  `.done`-marker resumable. Post-fx runs on the ASSEMBLED master ONLY (owner rule 2026-11-19),
  never on the raw source.
- **Speech-gated leveling** (#15, 0871f4d): loudnorm alone lets sirens/door slams skew dialogue
  — RMS is measured STRICTLY on ASR spoken-word intervals (siren fixture proves the meter is
  gated); gain to -9 dB RMS target, clamped +/-12 dB; -1.5 dBTP alimiter AFTER gain;
  `audit_technical` gained `speech_gated_loudness` (window -12..-6 dB).
- **The retune that made the audit pass** (f55d7ae): master target -9 -> **-11 dBFS** and
  `MAX_GAIN_DB` 12 -> 32 (MossFormer2 crest ~19 dB + the -1.5 dBTP limiter eat the one-shot gain;
  the loudness audit failed at the old clamps); post-limiter gain refinement is a BOUNDED LOOP
  (8 passes — the limiter response is concave once engaged); new QA gate `natural_dynamics`
  (speech crest >= 6 dB); **bus-2 ambience is OPT-IN** (`AUDIO_BUS2_VOLUME`, default OFF — the
  default master is pure isolated voice).
- **Measured finding — do not re-litigate**: `scripts/tune_bus1_comp.py` proved acompressor
  threshold/ratio do NOT move speech crest on this material (17.9-19.5 dB at every candidate,
  incl. 2:1@-12). The perceived punchiness came from the bus2 bed + gain/limiter interplay.
- **Quantified on the final master** (window 26.6-41.6s): SNR 3.5 -> 28.1 dB; presence -11.5 dB
  (raw was -15.2 — not muffled); crest 10.5 dB.
- **TTS norm** (8e1c5f8): agent-authored TTS text is normalized before synthesis (`tts_norm.py`;
  3.2g/5L-type tokens read as words); codec-safe container pairing (mp3->libmp3lame,
  m4a->aac); 48 kHz stereo law pinned (audit check `audio_law_48k_stereo`).
- **A1 trim craft** (5e1781a/d2fd156): sample-exact `atrim` + `asetpts` + 10 ms anti-click fades
  (mp3 input-seek lands on ~26 ms frames — a mid-word start would click); `VO_MIN_WINDOW_S=0.5`
  (one word ~0.3s at +5% speed); a stale trim can never overrun a re-synthesized shorter stem;
  the vo window rides the segment cache hash (a trim re-renders exactly one block).

---

## 6. DURATION & STORY LAWS (every owner rejection on length, and the committed band)

- **v0.1.0 OWNER-REJECTED** (#8, M5-R1): 54-98% duplicate-frame stretches, 15.2s freeze pad,
  28s still-image outro, 23.2s mean silent gaps, 2.5x upscale mush at 2.77 Mbps. Fix (v0.2.0,
  research-first per owner direction): interpolation on ALL slowed content (measured <3% dups vs
  25% naive), slow-mo reserved for dynamic windows (motion-floor rule), freeze cap 1.2s, gaps
  <=12s, 720p/CRF18, fades at every cut. **Gates grew 8 -> 11** — smooth_motion / bitrate /
  boundary_pop evaluate the OUTPUT FILE itself; two planner bugs were caught by these gates
  during development.
- **59.8s master postmortem** (2026-11-19): the autodrive ran on a 90s window and shipped a
  59.8s "master" — no story arc, no narration law, no QA teeth. Fix: `story_map.py` 2-phase
  full-arc Director + hard retention QA (600s / 9-narration / 0.30-0.52 ratio / 0.70 coverage
  with master quarantine).
- **The 251s tape-1 rejection** (11802a2): `doc_scale` kept 28-35% of ANY source
  (933s x 0.28 = 261s). The law now: **sub-25-min sources keep 50-65% of raw** (the niche
  skill's editorial contract, owner-ratified — a 15.5-min audit release is a ~8-10 min video);
  sources >= 1500s keep the Eustis long-form band EXACTLY (2627s -> [735.6, 900]); floor 25%,
  master never above 900s, never below 75s. Tape 1 now plans 466-606s. Short-doc threshold
  raised to **400s** (a 480s tape would otherwise blow the narration ratio at 0.58).
- **The raw-budget law** (c6c1e9a): the model authored 1336s of raw for a 933s tape (ratio 0.18
  — physically impossible). Prompt law alone cannot be trusted on free tiers: `enforce_raw_budget`
  in `director_session.py` extends the word-count law to raw windows — the Director's SELECTION
  is kept, lengths rescale proportionally to `target_hi * 0.7` (tape 1: 1336 -> 425s raw,
  execute 27/27 steps, retention GREEN).
- **Narration word bounds** (pinned, pro TTS pace): cold 40-65 / mid 48-85 / outro 70-105 words;
  ratio 0.30..0.52; hook: first narration <= 8s (context <= 32s, sting <= 4s); coverage: clip
  starts span the whole file (first 15% / final 25%).
- **Foreign-narration law** (ADR-0014, 2408955): `foreign_media.py` detects the source channel's
  narration (channel markers + narrator-speaker monologue) from the transcript; the planner
  EXCLUDES those windows from every raw-audio decision; muted b-roll keeps their pixels, OUR
  narration replaces their voice. The report lists every replaced window.
- **Retention scoring** (#19, 0e6bacd): `score_retention` 0-100 — hook in first 8s,
  beginning->middle->end coverage, momentum ladder (owning >40% sags -15, >60% dies -25),
  open-loop tease bonus, dead-span (>10s gap) penalty.
- **Content gate FIRST** (8969e49, ADR-0013): strict allowlist (shoplifting / trespassing /
  first_amendment_audit); violence vetoed at plan time. The gate runs before anything else in
  the pipeline.

---

## 7. CI & PROCESS FAILURES (every red run in the window, and the process law)

- **2026-11-21, twice** (a374d91, 8e1c5f8): pushed WITHOUT running `make lint` first — 49 ruff
  errors (F401/E501) across the new scripts. Process law: `ruff check` + `ruff format --check`
  before EVERY push; the format gate is part of quality (4748411, 4a7cbda).
- **2026-11-25** (runs #145/#146/#147, c6c1e9a/b712ad8/2099aca): one signature in
  `director_session.py` drifted from ruff-format style — three pushes red on `ruff format
  --check` while `ruff check` passed locally. LESSON: run BOTH commands, not just the linter.
- **2026-11-17** (runs 44112010637/44125589107): 12 stale-format files + 5 latent mypy errors —
  all gates were GREEN locally on the author's machine; wire facts, not vibes, decide CI.
- **mypy None-narrowing** (62aac6a): an inline tiling condition had to preserve None-narrowing
  (line 305 index error) — don't refactor a narrowing expression into a helper.
- **tsc TS2556 under CI's stricter resolution** (958a0be): ops-stream test mock spread — CI's
  TypeScript resolves differently than the dev box; run `tsc --noEmit` + `vitest run` locally.
- **CI branch filter** fixed (fcce0e6) — workflows must run on the branches we actually push.
- **RED-first as designed**: the exact-check-name contract test caught `clean_out_frames` the
  moment it landed (30dd40f); the eval harness found `no_tags_on_clean_speech` (3f4be47).
  Gates that can't go RED are decoration (ADR-0007 self-test law).
- Session-start drift check: run pytest on the TOUCHED test files BEFORE the full suite — a
  recycled sandbox shows up as collection errors.

---

## 8. SANDBOX & INFRA FAILURES (what the environment does to us, and the countermeasures)

- **The recycle wipes**: (a) `.env` (ALL keys) and reverted the repo dir to a 2026-11-17
  snapshot (2026-11-21) — recovered by `git fetch` + hard reset to origin; (b) `download/`
  master + `footage/` mp4s (2026-11-24) — the produced master was LOST; only captions + report
  survived. That loss CREATED the release law. Assume every session starts from the repo +
  secrets dir only.
- **Fake dirt**: the recycle flips every file to 100755 — `git config core.fileMode false` first,
  always.
- **Background processes do NOT survive across messages.** Run long stages synchronously with
  `timeout 555` (or 560 for the outer wrapper). Gladia polling killed mid-flight is fine:
  resume with `--project-id <id>`.
- **Gladia never re-uploads** (17d5539, 0c320b8): ASR transcripts persist per asset; retry
  actually resumes (three stacked root causes found and fixed for the retry bug); ASR uploads
  COMPRESSED audio (a 300MB mp4 upload was the slow path).
- **Crash-recovery drill is a permanent CI test** (7656c69): segment A admitted -> worker dies
  leaving a half-written `B.tmp.mp4` corpse -> resume hits A from cache, re-encodes B OVER the
  partial tmp (corpses can never poison admission — ffprobe validation on `put()`); found+fixed
  while writing it: `SegmentCache.count()` globbed `*.mp4` and counted tmp corpses as entries.
  The drill already paid off in production: a sandbox-killed render resumed from cache and
  delivered.
- **minio removed from Docker Hub mid-2025** -> images moved to quay.io + auto-create bucket on
  first write so first render never fails on a missing bucket (9c5f038); compose unblocked
  (edc27f4); **boto3 must be declared** (567a8bd) — the s3 storage backend crashed
  ModuleNotFoundError on FIRST RENDER.
- **One-command run + env wiring** (60aefec, f8bd2d5): `make api` wires .env keys (uv
  --env-file, shell wins); setup.sh/.bat; docker-compose healthchecks on all five services.
- **venv drift repair** (fresh sandbox): `uv pip install -q --python .venv/bin/python
  python-multipart botocore onnxruntime cryptography PyMuPDF beautifulsoup4 trafilatura lxml
  boto3 python-docx`.
- **`produce_tape.py` runs with `.venv/bin/python`** — the system python3 has no `director_os`
  package. Every time.
- **GitHub API quirks**: list-releases occasionally 404s transiently — retry before debugging
  auth; verify a release via `releases/tags/<tag>`; a plain HEAD asset check LIES (the asset URL
  302s to S3) — byte-verify with ranged GETs (head 2MB + tail 2MB) against local bytes; and keep
  `curl -w` OUT of the payload pipe (its `HTTP:206` suffix once produced a false byte-mismatch).
- **Remote-URL token extraction**: parse with python `re`
  (`r'https://([^@]+)@github\.com/([^/]+)/([^\.]+)'`) — a sed one-liner once grabbed `//TOKEN`.
- **yt-dlp bot wall**: `bin/yt-dlp-nightly --cookies .secrets/yt_cookies.txt -f
  "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b" --merge-output-format mp4 -o footage/<slug>.mp4`
  (Deno 2.9.8 in `bin/`). Probe with ffprobe after download — duration/geometry must match the
  dossier. THE COOKIES ARE LIVE SESSION CREDENTIALS: if they ever appear in chat, sign that
  session out / rotate them.

---

## 9. UI HONESTY & PUBLISH (the CR-001..015 adversarial critique, 7b7e016)

Owner-ordered 3-audit critique found 15 issues; 14 fixed in the same pass. The headline ones
(full table lives in docs/CRITIQUE.md):

- **CR-001 CRITICAL**: the ship pack received the PRE-post-fx master — uploads silently lost
  burned captions, tracked blur, enhanced audio. Pack source is `final_for_qa`, pinned by test.
- **CR-002 CRITICAL**: the export-gate cursor was only stamped inside the tracked-redaction
  path — with the default REDACT_MODE the gate was "dirty" forever. `_stamp_export_gate` runs
  after EVERY master render.
- **CR-003 CRITICAL**: publish lied twice — the publisher was an InMemory stub (fake `draft_…`
  id, project flipped to `published_draft` with NOTHING uploaded) and the form demanded a
  hand-typed server path. Vault-backed publisher resolution; unconfigured = honest `dry_run:
  true`; `GET /deliverables` serves the real files. **Multi-channel publish never fakes an
  upload** (`published` / `not_configured` / `rejected` states per channel).
- **CR-007**: checkpoint version race — `UniqueConstraint(project_id, version_num)` + migration
  0008 + retry-on-IntegrityError; 4-thread race test asserts gap-free unique 1..40.
- **CR-008 CRITICAL security**: vault encryption could never activate (`cryptography` missing
  from deps -> `SETTINGS_SECRET_KEY` silently degraded to PLAINTEXT keys in the DB). Dependency
  added; setup.sh auto-generates the key; Fernet-at-rest proven by test.
- **CR-004/005/012**: the Export button was a disguised pipeline restart; the "Director tools"
  panel fired hardcoded destructive dispatches; the A1 "mute" button lied (it called setHidden —
  audio monitoring never existed). One honest control each.
- **CR-009**: `running` was welded to SSE liveness — a dropped stream re-enabled Run/Export
  mid-run. Server truth first: newest unfinished job = running; 5s poll while the stream is dead.
- **CR-011 ops security**: every service published 0.0.0.0 with default creds — db/redis/minio/
  sam2 bound to 127.0.0.1 (api/web stay 0.0.0.0 deliberately: the owner tests the Studio from a
  phone on the LAN).
- **The op-log constitution** (27291fa): everything-is-an-op — `timeline_ops` (kind / frame
  window / track / params / origin agent|human / evidence); undo = REVOKE (row kept, provenance
  trail for the bodycam niche); Ctrl+Z undoes the newest op (9b3f586 — the legend stopped
  lying); agent ops stay human-refinable via adjust (e9994f8).

---

## 10. THE ENV CONTRACT (`.secrets/produce_env.sh`, gitignored — names only here)

Required exports for a production run (values live in the secrets file):

- `GLADIA_API_KEY` — ASR
- `LLM_API_KEY` + `LLM_MODEL=gemini-3.5-flash-lite` + `GEMINI_FALLBACK_MODELS` — primary tier
  (geo-blocked in this sandbox: it FAILS OVER, do not remove it)
- `OPENROUTER_API_KEY` + `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1` +
  `OPENROUTER_MODEL=google/gemma-4-31b-it:free` + `OPENROUTER_FALLBACK_MODELS` — the tier that
  actually serves planning here (notebook-proven JSON model; nemotron burned its budget on prose)
- `DATABASE_URL` — the production sqlite
- The driver itself setdefaults `POST_FX / AUDIO_MODE / AUDIO_ENHANCER / REDACT_MODE` — never
  pass hermetic overrides in production.
- Optional: `MODAL_TOKEN_ID/SECRET` (sticky geo-block egress), `LLM_BUDGET_USD`,
  `LLM_PROMPT_CHAR_BUDGET`, `AUDIO_BUS2_VOLUME` (default OFF).

Live-probe before committing to a long run (30s, saves hours):

```
source .secrets/produce_env.sh
.venv/bin/python -c "from director_os.agent_runtime.llm import build_backend_chain; \
 from director_os.agent_runtime.registry import ToolRegistry; \
 c = build_backend_chain(ToolRegistry()); print(c.tier_names); \
 print(c.complete_json('Reply with exactly {\"ok\":true}', 'JSON only.'))"
```

Expect: tiers `['gemini:...', 'openai_compat:...']` and `{"ok":true}`.
Anything else = do not start the run; fix the chain first (section 3).

---

## 11. THE PRODUCTION PROCEDURE (tape = one owner-approved pick)

1. **Source**: `bin/yt-dlp-nightly --cookies .secrets/yt_cookies.txt -f
   "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b" --merge-output-format mp4 -o footage/<slug>.mp4
   <url>` (cookies + Deno in `bin/` defeat the bot wall; probe with ffprobe — record
   duration/geometry, they must match the dossier).
2. **Dossier**: verified case facts in `docs/cases/<case>.json` (9 facts, 2 citations for
   tape 1) — the Director authors narration from THIS, never invented.
3. **Run**: `scripts/produce_tape.py --source footage/<slug>.mp4 --title ... --slug ... --render
   1920x1080 --dossier-json docs/cases/<case>.json --case-facts "..."` (fresh project; then
   `--project-id <id>` to resume). Badge defaults `top_left` — leave it there.
4. **Stages, all resumable**: gate -> caption-guard -> ingest (proxy+scrubsheet) -> yunet sweep
   (8 regions) -> Gladia ASR (persisted per asset, never re-uploads) -> proofread (deterministic
   floor; LLM supervisor is an upgrade, not a gate) -> planning -> TTS (edge-tts,
   content-addressed cache) -> render (segment cache, per-segment `.json` keys) -> post-fx
   (captions -> CFR -> tracked blur chunks -> deep-filter -> mux, `.done` markers) -> QA audit ->
   finalize (v6 captions + top-left badge in ONE re-encode) -> deliverables + production report
   JSON.

---

## 12. THE WINDOW LAWS (the sandbox kills processes at ~10 min)

- The tool window is ~555s. Long stages WILL be killed mid-flight. Everything is resumable
  EXCEPT planning-in-flight — so:
- **Planning is the only stage to pre-run standalone**: `/home/z/my-project/scripts/plan_standalone.py`
  runs `plan_macro` with a DISK CACHE (`scratch/plan_<pid>.json`), applies `enforce_raw_budget`,
  runs Phase B `execute` + retention, and commits the timeline version
  (`created_by=director_agent`). The driver's `active_otio_version_id` shortcut then skips
  planning on resume. If Phase A must re-generate, delete the cache file and re-run the script
  (it retries 4x internally with feedback).
- **Never judge progress by stdout alone**: run with `PYTHONUNBUFFERED=1`, but the truth is in
  the DB — `select level, message from job_logs where job_id=...` and `timeline_versions`
  (`created_by` says `director_agent` vs `assembler_agent`; the note says which plan won).
- Narration TTS clips land in `output/pipeline/<pid>/narration/*.json` — READ THEM before
  shipping: case-specific narration names the case (Kokoszka / Bergeson / the audit); generic
  bridges mention "forty minutes" and "the roadside". A generic narration = the fallback won =
  do NOT ship (standing law 9).
- Blur chunks: `post_fx/blur_chunk_NN.done` per chunk; deep-filter + mux follow automatically.

---

## 13. THE QA GATE (all lines must be GREEN on the FINAL file)

- `streams_present`, `audio_law_48k_stereo`, `stream_uniformity`
- `clean_out_frames` (head/tail black = 0.00, tail freeze = 0.00)
- `black_frame_budget` (0.00s, max 3.0s), `duration_drift` (tol 1.0s)
- `final_duration_floor` (25% of source), `narrator_floor`, `raw_floor`
- `narration_ratio` **0.30..0.52**, `coverage_span` (0.55; 0.70 for >= 1200s sources)
- `speech_gated_loudness` (-12..-6 dB window), `natural_dynamics` (crest >= 6 dB)
- caption-guard median < 0.012 (burned captions on the SOURCE tape)
- foreign-narration windows (if any) excluded from raw — the report lists every replaced window

Frame verification before release: sample frames at 8-12 points — badge **TOP LEFT**, v6
captions (dialogue white / suspect gold / narrator cyan, karaoke tints per the committed law),
tracked blur on faces, authentic Axon overlays intact.

---

## 14. THE RELEASE LAW (owner directive 2026-11-24, after the first master was lost)

**Render -> GitHub Release -> byte-verify -> only then everything else.**

1. Create the release (tag `newlondon-tape1-vN`) with the small assets (report, srt, ass).
2. Upload the master MP4 (`uploads.github.com`, Content-Type video/mp4).
3. Byte-verify: `state == "uploaded"`, size == local size, ranged GETs of head 2MB + tail match
   the local bytes (a plain HEAD parse lies — the asset URL 302s to S3; keep `curl -w` out of
   the payload pipe).
4. STATUS.md row + push + CI green via the GitHub API.
5. The sandbox recycles: NOTHING in `download/` or `output/` survives between sessions. The
   release is the only durable home.

---

## 15. SANDBOX DRIFT CHECKLIST (every session start)

- `git config core.fileMode false` (the recycle flips every file to 100755 — fake dirt)
- `git fetch origin && git merge --ff-only origin/main` (local may lag the pushed tip)
- venv drift repair: `uv pip install -q --python .venv/bin/python python-multipart botocore
  onnxruntime cryptography PyMuPDF beautifulsoup4 trafilatura lxml boto3 python-docx`
- run pytest on the touched test files BEFORE the full suite (drift shows as collection errors)
- `ruff check .` AND `ruff format --check .` before any push (three red runs this window prove it)
- GitHub API list-releases occasionally 404s transiently — retry before debugging auth

---

## 16. THE EDITORIAL LAWS (one line each, all enforced in code)

- Keep 50-65% of raw for sub-25-min sources; 720-900s band for >= 1500s (`doc_scale.py`)
- Narration: 9-13 blocks for tape-1 scale, word bounds cold 40-65 / mid 48-85 / outro 70-105,
  ratio 0.30..0.52 (`doc_scale.py` + `plan_invariants.py`)
- Raw budget = `target_hi * 0.7`, enforced deterministically (`enforce_raw_budget`)
- Hook: first narration <= 8s, context <= 32s, sting <= 4s
- Coverage: clip starts span the whole file (first 15% / final 25%)
- Captions: ONE master-clock builder (TTS stamps for narration, remapped ASR for raw,
  narrator-on-broll isolation); v6 typography law with pinned karaoke secondaries; badge top-left
- Caption tags: proofread tags (inaudible/mumbling/cross-talk) + workshop 5-tag gray-italic
  system with gap-fill; `no_tags_on_clean_speech` fires only when a tag REPLACED real text
- Post-fx on the ASSEMBLED master only (tracked blur + deep-filter with the muted-speech rescue
  gate + caption burn); never on the raw source
- Content gate FIRST (strict allowlist: shoplifting / trespassing / first_amendment_audit;
  violence vetoed at plan time)
- The Director authors narration from the CaseDossier only; never invent facts
- LLM failover: capacity errors (incl. geo-block) walk DOWN the tier chain, contract errors
  raise; planning asks 16384 tokens; the deterministic floor never blocks on an LLM outage
