# The Frame-Accurate Agent Editor — Research Dossier & Target Architecture

> Owner mandate (2026-11-23, condensed): a full chat AI-agent + subagents video
> editor. Frame-by-frame access — I select something on frame X, tell the agent
> to fix that specific thing, and ONLY that frame (or the smallest span that
> contains it) is processed. The timeline is frame-by-frame, not the baked
> video; nothing renders until I press Export. The agent must know exactly
> which frame and where to edit — never blindly create things — it must have
> full knowledge of the timeline, and access to all tools. Efficiency and
> speed are hard requirements: real-time live editing must not heavily consume
> the machine.

This dossier is the deep research the mandate asked for: what real editors and
real research systems do, which patterns we adopt, which we reject and why,
and the concrete architecture that takes DirectorOS from "pipeline that
produces a documentary" to "frame-addressable, agent-driven editor with an
export gate". What already exists in this repo is cited by file path; what
shipped in this push is marked **(shipped)**.

Companion research already in-repo: `docs/research/agent-research-dossier.md`
(agent memory/critic/fail-closed, primary-source verified) and
`docs/research/skills-mining.md` (128-link skill inventory with fit tags).

---

## 1. Prior art — how the real ones do it

### 1.1 Pull-based frame servers: MLT (Kdenlive, Shotcut)

MLT is the engine under Kdenlive and Shotcut, built for broadcast. Its core
architecture is **producers → tractor → consumers**: a `tractor` composes
tracks, and — the decisive detail — **frames are produced only when asked for**
(pull, not push). The Kdenlive developer docs put it plainly: "they produce
frames only when asked to do so, because MLT adheres to pulling frames; that's
why there are tractors and producers." ([MLT framework design][mlt], [Kdenlive
dev-docs: mlt-intro][kdenlive-mlt])

**Adopted:** the timeline is a *query*, not a render. Scrubbing to frame X
costs one frame's decode, not a re-encode of anything. Our frame server
**(shipped: `packages/core/src/director_os/core/frames.py` + `GET
/assets/{id}/frame`)** is exactly this law for the web/agent surface: one
accurate-seek ffmpeg decode per (frame, width), LRU-cached to disk, never a
passthrough re-encode.

### 1.2 Non-destructive EDL: the cut list is data

Professional editing is *non-destructive*: the source media is never modified;
the "edit" is a decision list (EDL). OpenTimelineIO is the modern,
editor-neutral form — "a modern Edit Decision List (EDL) that also includes an
API for reading, writing, and manipulating editorial data", focused on edit
information rather than media. ([OTIO docs][otio], [OTIO project][otio-gh])

**Already ours:** the whole pipeline is OTIO-native (`packages/otio_engine`),
every timeline is a versioned row (`TimelineVersionRow` in
`packages/core/src/director_os/core/db.py`), and agent + human edits both land
as new versions (`created_by: director_agent | human_edit`). Blur boxes, zoom
keyframes, caption cues — all metadata on clips, zero pixels touched.

### 1.3 Node graphs: Olive (and Blender) — effects as DAGs

Olive 0.2's headline feature is node-based compositing — "a form of visual
programming that gives you full control" where every effect is a node in a
directed graph and any frame is the evaluation of the graph at that point.
([Olive][olive])

**Adopted (as a mental model, not a UI):** a frame of our program = eval of
(clip → ops that overlap this time) at (asset, frame). This is what makes
"only that frame is processed" well-defined: an op list filtered by time range
IS the subgraph for frame X. We keep the linear NLE timeline the owner knows
(Premiere/CapCut-style) — the DAG lives underneath, in the op log.

### 1.4 Deterministic frame functions: Remotion

Remotion renders video *frame-by-frame* by mapping frame numbers to React
components with deterministic hooks (`useCurrentFrame()`); everything is a
pure function of (state, frame). ([Remotion][remotion], [community write-ups][remotion-article])

**Adopted:** frame N of our program must be a pure function of (timeline ops,
N). Purity is what makes caching sound (the LRU frame cache never serves a
stale frame after an edit — the cache key includes the op-set digest), and
what lets the export gate re-render only *dirty* segments.

### 1.5 Proxy-first workflows: Premiere / DaVinci / Kdenlive

Every professional NLE edits against a low-res **proxy** and only touches
full-res at export. We already have the full bundle: 480p `proxy.mp4` +
`scrubsheet.jpg` sprite + 64k audio (`packages/core/src/director_os/core/proxy.py`),
the ProgramMonitor plays the proxy, and **(shipped)** the frame endpoint serves
proxy frames by default and decodes full-res only on `src=full`, never
upscaling. Live editing therefore costs ~0 even on a 4 GB machine.

### 1.6 Promptable video segmentation: SAM 2 — "select things on frame X"

Meta's SAM 2 is the missing primitive for the owner's literal words: it takes
prompts (clicks, boxes, masks) on *one frame* and tracks/propagates the
segmentation across the whole video — "fast, precise selection of any object
in any video", interactive, frame by frame. ([Meta SAM 2][sam2], [SAM 2
paper][sam2-arxiv], [Ultralytics SAM 2 docs][sam2-ultra])

**Shipped as the selection service (roadmap step 3, 2026-11-23):** a box or
click on frame X returns the object's per-frame path (`vision_fx/selection.py`).
Backend ladder, strongest last and never silently downgraded:

- `detector-assist` — IoU-best pipeline face/plate track at the anchor (exact,
  zero re-decode — the sweep already ran);
- `ncc-cpu` (always available) — normalized cross-correlation template tracker:
  forward+backward from the anchor, template blending for drift, bounded-gap
  interpolation, honest refusals (texture-free prompt, object lost, < 2 real
  frames — a tracker that hallucinates across occlusions is a privacy leak);
- `sam2` — wired, env-gated, LOUD when unavailable (never a silent swap).

**EdgeTAM** (Meta, Jan 2025, Apache-2.0) is the on-device SAM 2 variant: 22x
faster than SAM 2 (15.7 FPS iPhone 15 Pro Max, ~20 MB of CoreML encoders) —
the chosen shape for the local SAM 2 service when weights ship locally
([EdgeTAM][edgetam]). Premiere's Object Masking (2026 docs) and DaVinci's
Magic Mask are the product bar this ladder is climbing toward: one click,
isolate, track ([premiere-object-mask][premiere-om]).

### 1.7 Agent-assisted editing: LAVE (ACM IUI 2024)

LAVE is the closest published blueprint to the owner's vision: an LLM agent
that *plans and executes editing actions* over footage it can reason about via
language descriptions, while the user can edit "through either the agent or
direct UI manipulation, providing flexibility and enabling manual refinement
of agent actions." ([LAVE, U of Toronto / CMU][lave-page], [arXiv:2402.10294][lave-arxiv])

**Adopted wholeheartedly — and already half-built:** our dual-modality law
(agent ops and human ops both land in the same versioned timeline; the Studio
shows every agent step live via `RunTracker` SSE) is LAVE's core finding,
operationalized. Our improvement over LAVE: grounding is not
language-descriptions-only — the agent reads the measured perception index
(transcript words with speakers, VAD spans, audio events, shot density, vision
insights) and **(shipped)** can pull the actual frame pixels via `frame.grab`.

### 1.8 Agent orchestration: what the 2025-2026 field converged on

Four sources shaped the chat-orchestrator design (roadmap step 5):

- **EditDuet** (arXiv:2509.10761) — two LLM agents over an NLE: an Editor that
  mutates the timeline through tools and a Critic that evaluates the draft
  against the user request and either gives natural-language feedback or calls
  RENDER. Human preference 86.9% vs 13.1% for the best single-agent baseline.
  Blueprint for our editor/critic roles + the Export gate as the critic's
  terminal action ([editduet][editduet]).
- **Azure Architecture Center** orchestration patterns: maker-checker loops
  need iteration caps AND fallback behavior; group chat ≤ 3 agents to stay
  controllable; every pattern needs explicit completion criteria ([azure-patterns][azure-patterns]).
- **Anthropic's multi-agent research system**: orchestrator-worker with clear
  task boundaries, parallel subagents, effort scaled to complexity, and —
  matching our pipeline's step-resume law — agents must RESUME from where
  errors occurred, not restart; end-state evaluation beats step-by-step
  evaluation for state-mutating agents ([anthropic-multiagent][anthropic-multiagent]).
- **LAVE** (§1.7) already gave us the dual-modality law; these three add the
  loop discipline (caps, criteria, critic roles, resume-not-restart).

### 1.9 Efficient decode engineering: TorchCodec's performance notes

Meta's TorchCodec performance guide codifies the decode facts we rely on:
exact vs approximate seeking cost, batch APIs, multithreading, optional CUDA.
([TorchCodec performance tips][torchcodec]) Our frame server uses ffmpeg's
input-side `-ss` (keyframe seek + decode-forward = exact frame, ~one decode
per grab); the same document justifies the roadmap's optional GPU decode path
only when a user opts in.

---

## 2. What we reject (and why)

| Pattern | Verdict | Reason |
| --- | --- | --- |
| Bake-on-edit (render after every change) | **Never** | The exact waste the owner banned: "only after everything is done then we click export". MLT/Remotion purity + our segment cache make it unnecessary. |
| Push-based preview streams | **Never** | Decode-on-demand (pull) is the entire performance story of MLT-class editors. |
| Full-tape detection/recognition by default | **Already killed** | The post-plan scoping law (commit 52cad6e): expensive stages run only on kept windows. Frame-scoped ops are the next tightening of the same law. |
| A2A/multi-agent protocol plumbing | **Rejected earlier** | ADR-0011 (single-team orchestrator; in-process subagents). Reaffirmed: subagents are roles in one session, not network peers. |
| Pixel-streaming the whole UI | **Rejected** | The proxy + frame-JPEG + overlay-composite path is 100x cheaper on the owner's hardware. |

---

## 3. Target architecture

### 3.1 The four laws

1. **Everything is an op.** An `EditOp` is the atom: `kind` (blur / inpaint /
   trim / caption / zoom / narration / card / audio), `frame_start`,
   `frame_end`, `track`, `params`, `origin` (agent | human), `evidence`
   (frame ids + transcript windows that justify it). The program at frame N is
   `fold(ops active at N)`. Undo = revoke the op. The SSE stream already
   carries timeline revision events — ops ride the same feed.
2. **Frames are addressed, not rendered.** `GET /assets/{id}/frame?t&w&src`
   **(shipped)** is the single frame primitive for UI and agent alike. One
   accurate decode, LRU disk cache, proxy-first.
3. **Expensive work is scoped, never global.** Post-plan scoping shipped for
   detection; op-scoping extends it to every mutation: an op touching frames
   [X, X+30] processes ≤ 30 frames (SAM 2 mask propagation may need a few
   frames of context on each side — still O(span), never O(tape)).
4. **Export is the only full render.** Compile ops → segment graph → render
   only dirty segments (content-addressed `SegmentCache` already reuses every
   clean chunk) → lossless concat. The Export button is the gate the owner
   asked for.

### 3.2 Component map — today's reality

| Capability | Where it lives today |
| --- | --- |
| Versioned OTIO timeline (EDL) | `packages/core/.../db.py` `TimelineVersionRow`; `packages/otio_engine/` |
| NLE web timeline + transport | `apps/web/.../studio/Timeline.tsx`, `ProgramMonitor.tsx` (frame-step, J/K/L, in/out, box-draw selection), `useNLEShortcuts.ts` |
| Frame server | **(shipped)** `core/frames.py`, `GET /assets/{id}/frame`, agent tool `frame.grab` (`inspection_tools.py`) |
| Proxy bundle (480p + scrub sheet + audio) | `core/proxy.py`, built at ingest |
| Perception index (grounding) | `perception_index` row: transcript words/speakers, shots, audio events, vision insights, redaction tracks |
| Agent tool registry (32→35 typed tools: ops.list / ops.propose / ops.revoke) | `agent_runtime/registry.py` + `editing_tools` / `inspection_tools` / `workshop_tools` / `ops_tools` / `critic` |
| Plan-and-execute session + checkpoints | `agent_runtime/director_session.py`, `checkpoint.py`, `session_memory.py` |
| Live observability (per-step bars, console) | `worker/progress.py` `RunTracker`, `api/events.py` SSE, `useLiveJob` |
| Segment cache (content-addressed partial renders) | `otio_engine/segment_cache.py` |
| ASR content cache (never re-upload) | **(shipped)** `perception/asr_cache.py`, `asr_cache` table (migration 0006) |
| Failure-resume (stage checkpoints) | every stage persists its product; Run resumes the failed stage — **(shipped)** the run-again path is now actually wired: worker resolves vault keys before the transcriber builds (pipeline `advance_job`), the API clears the terminal error before dispatch, the SSE stream reopens on Run |

### 3.3 The agent loop v2 (chat + subagents)

```
        ┌──────────────────────── Studio chat (owner) ─────────────────────┐
        │  "fix the face leaking at frame 4201 and brighten that shot"    │
        └────────────────────────────────┬─────────────────────────────────┘
                                       v
                            ┌── Orchestrator (LLM) ──┐
                            │  grounded context pack: │
                            │  - timeline op digest   │
                            │  - transcript window    │
                            │  - perception index hits│
                            │  - frame(s) via grab    │
                            └──┬─────────┬─────────┬──┘
              ┌────────────────┘         │         └────────────────┐
              v                          v                          v
      Selector subagent          Editor subagent            Critic subagent
      (vision: which frames,     (ops: SAM2 mask/blur,      (qa.self_audit +
       where exactly — frame-     caption fix, trim, zoom;   owner laws on the
       grounded, evidence-first)  MINIMAL span only)         exact op result)
              │                          │                          │
              └─────────────── ops + evidence citations ─────────────┘
                                       v
                        op log  ──SSE──▶  live timeline (human can adjust)
                                       v
                    Export button ─▶ compile ─▶ dirty segments only ─▶ master
```

Design rules (all inherited from verified research + our own live sessions):

- **Evidence-before-action** (skill law since v0.4): an edit proposal that
  names a frame must cite the frame — now enforceable with `frame.grab`
  pixels, not just timestamps.
- **Subagents are roles, not services** (ADR-0011): same session, same
  registry, scoped tool subsets + scoped prompts. The `critic` already exists
  (`agent_runtime/critic.py`); the selector is the new vision role.
- **Fail-closed with feedback**: unknown tools rejected with the known list,
  errors fed back to the next round (shipped earlier, kept).
- **Every op carries provenance**: `origin`, `evidence`, and the session id —
  the audit trail the bodycam-documentary niche legally requires.

### 3.4 Efficiency budget (the "don't melt my machine" contract)

| Phase | Cost ceiling | Mechanism |
| --- | --- | --- |
| Scrub / frame-step | 1 decode per new frame, 0 for cached | frame server LRU (600 JPEGs/asset, tranche eviction) **(shipped)** |
| Op preview (blur box, caption, zoom) | ~0 CPU | overlay composite of the cached frame + op params in canvas — never a server render |
| Agent grounding | O(evidence window) | perception index queries + single-frame grabs; vision enrichment already ran once per content hash |
| ASR | 0 on retry / re-ingest of same media | content-addressed ASR cache — adopt-or-hit, never re-upload **(shipped)** |
| Detection / SAM 2 | O(op span) | post-plan + op scoping; CPU-default models, GPU opt-in |
| Export | O(dirty segments) only | content-addressed segment cache + `-c copy` concat |

---

## 4. Roadmap (ordered, each step independently shippable)

1. **Frame server + agent eye** — *done in this push.*
2. **Op log v1** — *shipped:* `timeline_ops` table (migration 0007) + CRUD
   API + reducer fold (`core/ops.py`); blur regions dual-write/import onto it
   and caption PATCH edits record caption ops; SSE `ops/stream` pushes every
   row (undo rides the feed as a revoke tombstone); the NLE snapshot renders
   effects from the fold (ops supersede legacy by id — never double-render);
   Studio draws boxes as ops and shows the agent origin badge live.
3. **Selection service** — *shipped:* `vision_fx/selection.py` (detector-assist
   → ncc-cpu ladder + env-gated sam2 adapter, honest refusals); `POST
   /assets/{id}/selection` + `GET /ops/{op_id}` + `GET /selection/backends`;
   agent tool `selection.track` (evidence-before-action enforced) writes blur
   ops with `params.tracked` (per-frame path step 4 bakes) and dual-writes the
   union region so the shipped render path never loses privacy coverage; the
   ProgramMonitor draws the box FOLLOWING the object at the playhead and
   offers one-click "track from playhead" (seed op superseded, history kept).
4. **Frame-scoped render ops**: per-op bake of [frame_start, frame_end] into
   the segment graph (the `_tracked_redacted_clip` machinery generalized);
   preview = overlay; nothing baked until Export.
5. **Chat agent v2**: orchestrator + selector/editor/critic roles over the
   registry, context pack = op digest + evidence; Studio chat panel (the
   console + SSE rails already exist).
6. **Export gate UI**: one button, dirty-segment compile, progress per
   segment (RunTracker steps already give per-step bars).

---

## 5. References

- [MLT Framework — design docs][mlt] — pluggable producer/tractor/consumer architecture for broadcast.
- [Kdenlive dev-docs — MLT intro][kdenlive-mlt] — pull-based frame production ("frames only when asked").
- [OpenTimelineIO documentation][otio] — the modern EDL: interchange + API for editorial data.
- [OpenTimelineIO GitHub (Academy Software Foundation)][otio-gh]
- [Olive Video Editor][olive] — node-based compositing (0.2).
- [Meta AI — SAM 2 research page][sam2]; [SAM 2 paper (arXiv:2408.00714)][sam2-arxiv]; [Ultralytics SAM 2 docs][sam2-ultra] — promptable per-frame selection + video tracking.
- [LAVE project page (U of Toronto DGP)][lave-page]; [LAVE paper (arXiv:2402.10294, ACM IUI 2024)][lave-arxiv] — LLM agent + dual-modality editing, the closest published blueprint.
- [Remotion][remotion] — deterministic frame-by-frame programmatic rendering.
- [TorchCodec performance tips (Meta-PyTorch)][torchcodec] — exact vs approximate seek, batch decode, threading, CUDA.
- [EdgeTAM (Meta AI, GitHub)][edgetam] — on-device SAM 2 variant: 22x faster, Apache-2.0, the local SAM 2 shape.
- [Adobe Premiere — Object Masking][premiere-om] — 2026 product bar for one-click subject isolation + tracking.
- [EditDuet: A Multi-Agent System for Video NLE (arXiv:2509.10761)][editduet] — Editor+Critic loop over an NLE, RENDER as the terminal action.
- [Azure Architecture Center — AI agent design patterns][azure-patterns] — maker-checker caps, completion criteria, pattern tradeoffs.
- [Anthropic — How we built our multi-agent research system][anthropic-multiagent] — orchestrator-worker, resume-not-restart, end-state evaluation.
- Prior in-repo research: `docs/research/agent-research-dossier.md`, `docs/research/skills-mining.md`.

[mlt]: https://www.mltframework.org/docs/framework
[kdenlive-mlt]: https://invent.kde.org/multimedia/kdenlive/-/blob/master/dev-docs/mlt-intro.md
[otio]: https://opentimelineio.readthedocs.io/en/latest/
[otio-gh]: https://github.com/AcademySoftwareFoundation/OpenTimelineIO
[olive]: https://www.olivevideoeditor.org
[sam2]: https://ai.meta.com/research/sam2
[sam2-arxiv]: https://arxiv.org/abs/2408.00714
[sam2-ultra]: https://docs.ultralytics.com/models/sam-2
[lave-page]: https://www.dgp.toronto.edu/~bryanw/lave
[lave-arxiv]: https://arxiv.org/abs/2402.10294
[remotion]: https://www.remotion.dev
[remotion-article]: https://nkaushik.in/writing/programmatic-video-with-remotion-react-and-ai
[edgetam]: https://github.com/facebookresearch/EdgeTAM
[premiere-om]: https://helpx.adobe.com/premiere/desktop/add-video-effects/work-with-masks/object-masking.html
[editduet]: https://arxiv.org/abs/2509.10761
[azure-patterns]: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns
[anthropic-multiagent]: https://www.anthropic.com/engineering/built-multi-agent-research-system
[torchcodec]: https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/performance_tips.html
