# Research: open-source video engines & editing frameworks (2026-11)

Trigger: owner verdict on v0.1.0 ("laggy, not how professional editing works")
+ owner direction to stop reinventing wheels and adopt proven open-source
projects ("video-use and a lot of other projects — do deep research").

Method: live survey of GitHub/READMEs (stars, licenses, last releases), plus
hands-on benchmarks of the relevant techniques inside THIS sandbox (2 cores,
no GPU, Python 3.12, 3 GB RAM).

---

## 1. The landscape, mapped to what this pipeline actually does

Our pipeline is a **raw-footage commentary editor**: it takes agency-released
bodycam files, finds semantic moments, writes narration, and assembles a
documentary cut with freeze-and-explain structure (ADR-0004). The candidate
projects fall into four buckets:

| Bucket | Projects | Fit for us |
|---|---|---|
| A. Agent-driven editing toolkits | **video-use** (browser-use, MIT, ~17k★) | **Adopt patterns** (QC loop, fades, EDL) — built for interactive editing of provided footage |
| B. Faceless-video generators (script → stock clips → TTS → render) | MoneyPrinterTurbo (85k★), ShortGPT | Wrong sourcing model (stock footage, not public-record raw); assembly ideas only |
| C. AI video-commentary editors (解说) | **NarratoAI** (MIT, 11.8k★, v0.8.7 2026-07) | Closest concept twin; validates our approach; uses MoviePy engine (slower than our ffmpeg-direct) |
| D. Low-level craft engines | **Practical-RIFE** (MIT, code+weights), auto-editor, faster-whisper, ffmpeg filters | **Direct adoption** into the renderer |

## 2. What we are adopting now (v0.2.0)

### 2.1 From video-use: render → SELF-EVAL loop, audio fades, EDL formality
video-use architecture: `Transcribe → Pack → LLM Reasons → EDL → Render →
Self-Eval (issue? fix + re-render, max 3)`. It self-evaluates the rendered
output at every cut boundary, catching **visual jumps, audio pops, hidden
subtitles** before a human ever sees it. Its editors report "30 ms audio
fades at every cut so you never hear a pop".

What v0.1.0 got wrong that this pattern would have caught: our concat had
**hard audio boundaries** (bed −6 dB → digital silence → bed) with no fades —
classic click/pop source — and no boundary QC at all.

Adopted:
- 30 ms `afade` in/out on every rendered segment (pop prevention).
- New gates that eval the OUTPUT like video-use's self-eval:
  `smooth_motion` (% near-identical consecutive frames per slowed segment,
  fails ≥15%), `bitrate_floor` (encoded density), `boundary_pop`
  (sample-level discontinuity scan at every segment boundary).
- Timeline formalized as an explicit EDL JSON (already our journal, now with
  per-segment provider metadata: speed, interpolation provider, motion score).

### 2.2 From Practical-RIFE (phase 2) + ffmpeg `framerate` (now): real interpolation
Root cause of "laggy": naive `setpts=PTS/0.45,fps=30` duplicates frames —
**measured 54–98 % near-identical consecutive frames** in replay sections of
v0.1.0, and the outro was a 28 s still image (100 % identical).

Benchmarks in this sandbox (854×474 source, 0.75×):
| Technique | Near-identical frames | Speed |
|---|---|---|
| naive setpts+fps (v0.1.0) | **25 %** | 1.5× realtime |
| ffmpeg `framerate` blend-interp | **2 %** | ~3× realtime |
| ffmpeg `minterpolate` mci/aobmc | — | >450 s for 8 s → unusable here |
| Practical-RIFE v4.25 | best quality, MIT code+weights | torch needs ≤3.11 + no GPU here; ONNX/onnxruntime route exists (RIFE_fp32) — phase 2 |

Decision: `framerate` interpolation ships in v0.2.0 behind a provider
interface (`pipeline/interp.py`) so RIFE-onnxruntime drops in as a config
flag without touching the renderer.

### 2.3 Motion-floor rule (never slow-motion a still scene)
v0.1.0 slowed `camera views empty rear seats` into a 25 s freeze. We already
compute per-second motion energy — now the timeline builder scores the actual
replay window and plays low-motion windows **real-time muted** instead.

### 2.4 Gap pacing + canvas/bitrate (ADR-0011)
16 silent gaps averaged 23.2 s (max 30 s) → capped at 12 s. 854×474 source
was stretched 2.5× onto a 1080p canvas at 2.77 Mbps → 720p canvas (1:1 for
720p sources), CRF 18, frame-clone freeze capped at 1.2 s, speed ladder
0.75→0.30 absorbs narration length instead of long pads.

## 3. What we deliberately do NOT adopt (and why)

| Project | Why not (now) |
|---|---|
| MoneyPrinterTurbo / ShortGPT | sourcing model = stock-footage b-roll from topic keywords; violates our raw-public-record requirement (AC-1). Their task-queue/WebUI layer is worth borrowing if a human-facing UI is ever needed. |
| NarratoAI as the engine | concept twin (LLM commentary + TTS + subs + auto-edit) but its MoviePy core is slower than frame-exact ffmpeg filtergraphs and it has no semantic-anchor model for raw multi-act footage; we adopt its *validated structure*, not its engine. |
| Remotion / Manim / HyperFrames (video-use overlay generators) | designed for motion-graphic overlays; our compliance overlay is a static drawtext — adding a JS/React runtime to render text would be over-engineering. Revisit if animated lower-thirds are ever wanted. |
| auto-editor | removes silence/filler via loudness; our gaps are *audible bed audio*, not silence. Its motion mode duplicates logic we already own (per-second motion arrays). |

## 4. Phase-2 backlog (research → issues)

1. **RIFE-onnx interpolation provider** (`pipeline/interp.py`): benchmark
   RIFE_fp32 ONNX via onnxruntime CPU on 480p/720p; adopt if ≥4 fps → deep
   slow-mo segments interpolate with optical flow instead of blend.
2. **faster-whisper burned captions** (MIT): we already hold narration text +
   per-unit WAVs; word timestamps give aligned captions over muted replays —
   standard for professional bodycam channels.
3. **video-use `timeline_view` filmstrip QC PNGs** attached to the review
   queue so the owner can inspect cut boundaries without scrubbing.
4. **Remotion-free animated end-card** via PIL frames only if needed.

## 5. Sources (opened, verified 2026-11-15)

- video-use — https://github.com/browser-use/video-use (MIT; pipeline + self-eval + fades)
- NarratoAI — https://github.com/linyqh/NarratoAI (MIT; 11.8k★; v0.8.7 2026-07)
- MoneyPrinterTurbo — https://github.com/harry0703/MoneyPrinterTurbo (README-en)
- ShortGPT — https://github.com/rayventura/shortgpt
- Practical-RIFE — https://github.com/hzwer/Practical-RIFE (MIT incl. weights; v4.25 recommended; ≤3.11)
- RIFE_fp32 ONNX — https://huggingface.co/FuryTMP/RIFE_fp32
- auto-editor — https://github.com/wyattblue/auto-editor
- faster-whisper — https://github.com/SYSTRAN/faster-whisper (MIT, CTranslate2)
- concat pop mechanics — https://superuser.com/questions/1867552
