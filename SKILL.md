# SKILL.md — DirectorOS Development Rules

> **This file is LAW.** Every session MUST read it before writing code. These rules exist
> because we ship real video to real channels: a leaked face is a legal incident, a red CI
> is a broken release, an unclosed issue is invisible work. If a change violates this file,
> the change is wrong — not this file.

---

## 0. The three non-negotiables

1. **Always push.** The sandbox is recycled between sessions. Any milestone, fix or
   module that exists only in the working tree does not exist. Commit + push at the end
   of every completed unit of work (feature, fix, test block) — never batch "a day of
   work" into one mystery commit.
2. **CI must be green before push.** `make lint && make type && make test` locally, then
   push, then confirm the run on GitHub. Never push red, never "fix later". Failures are
   fixed forward in a follow-up commit within the same session.
3. **Issues are the scoreboard.** Milestone issues (#11 M3 Studio, #12 M4 Vision+GPU,
   #13 M5 Cloud) stay open until every acceptance criterion is demonstrably met, then
   close with evidence links (test names, artifact paths, commit range). Every commit
   that advances a milestone references it (`feat(m4): ... (#12)`). Never declare work
   done in chat without closing the loop in the repo.

---

## 1. Engineering standards (big-tech team rules)

- **Typed or it doesn't land.** `mypy packages apps` clean. No `# type: ignore` without
  a reason comment. pydantic v2 models are the domain language (`packages/core/models.py`).
- **Tested or it doesn't land.** Every module ships with pytest coverage in the same
  commit. Tests are hermetic: no network, no model downloads, no GPU in unit tests.
  Real-model paths are optional and skipped when weights are absent.
- **Ports and adapters, not singletons.** Every external capability sits behind a
  Protocol so tests inject fakes:
  - `agent_runtime.llm.LLMBackend` — Gemini/OpenAI-compatible chat
  - `vision_fx.compute.ComputeBackend` — vision compute (LocalOnnxCpu default,
    KaggleBatchGpu free cloud, ModalGpu dormant)
  - `audio_engine.tts.TtsBackend` — EdgeTTS default ($0), ElevenLabs optional
  New external service? It gets a port first, an adapter second, a fake third.
- **Zero heavy SDKs.** Pure `httpx` + REST against vendor APIs. No vendor SDK, no torch
  in the default path. ONNX Runtime (CPU) + OpenCV are the approved vision runtimes.
- **pydantic v2 traps.** `model_copy(update={...})` skips validation — pass model
  instances (`BBox(...)`), not dicts. Validators run on construction; construct with
  kwargs, not positional soup.
- **Conventional commits.** `feat|fix|test|docs|refactor|chore(scope): summary`. One
  logical change per commit. `Closes #N` in the body when a requirement is met.

---

## 2. Sandbox execution rules (how we run jobs here)

- Nothing survives between tool calls: **no nohup/setsid background jobs**. Long jobs
  run in the foreground under `timeout 570 <cmd>` (leave headroom under the 600 s tool
  limit) and MUST be **kill-safe + resumable**: segment caches, idempotent sweeps,
  `count()` that ignores `.tmp` corpses.
- Long stdout goes to a file: `cmd > /tmp/x.txt 2>&1; echo exit=$?; tail -40 /tmp/x.txt`.
- `gh` CLI is absent — use `curl` against the GitHub REST API with the token already
  embedded in the git remote URL. Never print the token.

---

## 3. Redaction law (face blur) — the standard we were failing

Static blur boxes are not tracking. The ONLY acceptable face redaction is the
deterministic multi-pass trajectory pipeline in `packages/vision_fx`:

- **Pass 0 — frame truth.** Exact frame indexing on CFR (30 fps) media; VFR sources are
  normalized first. Frame `i` in the trajectory table is frame `i` in the encoder.
- **Pass 1 — detect + track.** Strong detector (imgsz 960–1280, conf ≈ 0.22) feeding a
  **BoT-SORT-style tracker with Global Motion Compensation (sparse optical flow) and
  `track_buffer=60`**. Bare IoU association and short buffers are FORBIDDEN — a chest-cam
  whip-pan destroys track identity within 3 frames. CPU stride > 1 turns skipped frames
  into gaps for Pass 2; never carry stale un-interpolated boxes.
- **Pass 2 — trajectory repair.** Linear interpolation bridges gaps ≤ 25 frames with a
  +15% box expansion on bridged boxes; 5-frame lead-in/lead-out holds both ends; the
  One-Euro Filter smooths jitter (EMA is forbidden — it lags 5–10 frames); the envelope
  `scale_w=1.35, scale_h=1.40` pads every box and the padded box must fully cover the
  original detection.
- **Pass 3 — render + prove.** Feathered elliptical Gaussian blur/mosaic composited per
  frame at exact indices; `debug_overlay=True` draws green = raw detection, red = padded
  blur box, `track_id` labels; `excluded_track_ids` lets Studio toggle tracks. Ship with
  a **whip-pan zero-leak integration test**: fast-pan clip, every frame inside active
  track ranges must have zero un-blurred face pixels.

Any PR that touches redaction re-runs the zero-leak test. "It looked fine on the preview"
is not evidence; the test is.

---

## 4. Media & cost laws

- **Audio law (ADR-0005):** 48 kHz stereo AAC stems; loudnorm; speech-gated level audit
  targets −12 dB … −6 dB RMS on spoken-word intervals only (never whole-program — sirens
  and door slams skew loudnorm); hard limiter ceiling −1.5 dBTP.
- **Coordinates law:** detector sweeps and Studio canvases live in RAW-footage pixels;
  render effects live in letterboxed render space. Mapping goes through
  `redact.scale_bbox` (`s = min(tw/fw, th/fh)` + centered pad) — never eyeball it.
- **Cost law:** $0 burn by default. `LocalOnnxCpuBackend` is the default ComputeBackend;
  Kaggle free GPU (`KAGGLE_USERNAME`/`KAGGLE_KEY` via `/api/v1/kernels/push`) is the
  optional batch lane; **Modal stays dormant — we do NOT attach a payment method**
  (Modal requires verification fees). One `.env` key must be able to switch a provider;
  switching providers is config, not code.

---

## 5. Definition of done

A task is done when ALL of these hold:
1. Code + tests pushed, CI green on GitHub.
2. The relevant milestone issue's acceptance items are checked with evidence, or the
   issue is closed.
3. `PLAN.md` milestone table reflects reality (what works, what remains).
4. The feature was exercised on real footage (or the milestone's fixture) — not only on
   unit-test fakes.
