# Skills Mining — the owner's 128-link inventory, evaluated for DirectorOS (2026-11-21)

> **Provenance:** the owner researched "installable agent skills (SKILL.md packs) plus
> the editing techniques they encode" and supplied a 128-link list (sections 1–8 below
> preserve it, condensed from the original PDF). Owner's own caveats are preserved in
> §5. The list was NOT taken on faith: entries are tagged by what we did with them.
>
> **Tags**
> - `[VERIFIED]` — we opened the repo/PyPI page (live check 2026-11-21) and confirmed it exists as described.
> - `[OWNER-LIST]` — description preserved from the owner's doc; we did NOT open it. Treat as a lead, not a fact.
> - Fit: `ADOPTED` (pattern already landed here), `MINE` (worth extracting from, ranked),
>   `ALREADY-BUILT` (we have the equivalent — do not re-mine), `WATCH` (revisit when the
>   corresponding capability becomes a milestone), `N-A` (out of scope for an
>   evidence-driven bodycam documentary pipeline).
>
> **Security law (owner's caveat, ratified):** skills run scripts on your machine.
> Never install a skill pack without reading its SKILL.md and scripts. Section 1's
> curated hubs carry security grades — start there.

## 1. Verdict by category

| Owner section | Content | Verdict for DirectorOS |
|---|---|---|
| 1. Curated hubs (1–12) | awesome-lists of video skills, editors, MCP servers | HUB: keep #1 as the standing index (security-graded). Others: directories, not sources |
| 2. Editing skill packs (13–52) | cutting, captions, color, reels, shorts | Mostly N-A (Hormozi/reels/short-form). MINE: #14, #40, #41, #46, #79 patterns |
| 3. Programmatic / motion-graphics (53–64) | Remotion, HTML-to-video | WATCH: graphics stage could borrow Remotion skills; no current need (cards are FFmpeg-drawn) |
| 4. Pro-editor bridges (65–81) | Resolve / Premiere / Kdenlive MCP | ALREADY-DECIDED (ADR-0011): in-process registry now; mine the *patterns*, not the plugs. WATCH for direct-NLE export |
| 5. Big-tech platforms (82–99) | Adobe MAX / Resolve 21.1 / Flow / Runway news | News, not skills. WATCH: Premiere's local MCP is preview-only (private beta planned early 2027); Resolve 21.1 has a native MCP server — relevant only under ADR-0011's revisit clause |
| 6. Research with open code (100–113) | VideoAgent, Crayotter, EditDuet, CutClaw, GLANCE, LAVE, MMLVE… | MINE as papers: EditDuet verified + adopted (critic); others are long-form editing research — WATCH for retention/coverage laws |
| 7. Building blocks (114–121) | FFmpeg, auto-editor, PySceneDetect, OTIO, WhisperX, Kdenlive… | ALREADY-BUILT: FFmpeg, OTIO (canonical, ADR-0002), webrtcvad+Gladia ASR (word timestamps), scene detection. No new deps |
| 8. Technique write-ups (122–128) | practitioner posts | MINE opportunistically; #123 (eight skill files for post-production) may map to skill-file sections |

## 2. What we ALREADY have (do not re-mine)

| Capability | Their approach | Ours |
|---|---|---|
| Agent↔tools plug | MCP servers ( Resolve/Premiere/Kdenlive) | Typed in-process registry, 35 tools (ADR-0011) |
| Cut data interchange | OTIO export adapters | OTIO canonical from day 1 (ADR-0002) |
| Transcription | WhisperX word timestamps | Gladia v2 live ASR + webrtcvad evidence |
| Silence/filler cutting | auto-editor | ASR-word + VAD-span driven trims (measured windows, not silence heuristics) |
| Session memory | `project.md` / AGENTS.md / plan files | `session_memory.py` (plan/notes/status, atomic + torn-write fallback) |
| Self-evaluation of renders | video-use self-eval at cut boundaries | `critic.py` `qa.self_audit` + 11 caption QA gates + audit_technical probes |
| Read-back receipts | davinci-mcp "verified: true" | tool JSON receipts + checkpoint store; honest-absence ("not established") semantics in critic |
| Provenance lawbook | OpenChatCut | skill v0.5 provenance section (evidence-before-action law) |
| Craft constants | Netflix captions, YouTube audio chain | skill v0.5, sourced |

## 3. Ranked mining targets (what to extract, when)

1. **browser-use/video-use** `[VERIFIED]` github.com/browser-use/video-use — MINE.
   Confirmed live: "the LLM never watches the video, it READS it" — one ~12 KB packed
   transcript per source + on-demand filmstrip/waveform PNG composites only at decision
   points; self-evaluates rendered output at every cut boundary (fix+re-render max 3);
   session memory in `project.md`. **Extract:** the packed-reading-view idea for
   `story_map.py` (token budget already matches our 2.5k law) and the
   on-demand-visual-only-at-decision-points rule for vision calls.
2. **proofcut** `[OWNER-LIST]` (PyPI, 93 MCP tools, OTIO/Kdenlive export) — MINE the
   *verification habit* its description claims: measures frame counts because
   "ffmpeg and auto-editor exit successfully even on some failures". Our equivalent
   law is ffprobe-validated cache admission; extend to per-render frame-count truth
   if a discrepancy ever appears.
3. **notque/claude-code-toolkit: video-editing** `[OWNER-LIST]` — ADOPTED in
   principle: 6-layer split, AI judgment + deterministic FFmpeg/Remotion execution —
   exactly our registry/worker seam. Nothing more to extract unless we add a
   programmatic-graphics layer (then: Remotion skills #55).
4. **davinci-mcp (PyPI, claimed 299 typed actions)** `[OWNER-LIST]` — the read-back +
   honest-absence patterns are already adopted (ADR-0011). Re-verify the package
   exists before citing it anywhere (the PyPI page we confirmed was
   `davinci-resolve-mcp`, samuelgursky's, #65 `[VERIFIED]`).
5. **ChatCut talking-head-guide (#40)** `[OWNER-LIST]` — speech-led editing with
   independent treatments per utterance. Matches our speaker-colored caption +
   role-flow audit design; mine for treatment vocabulary (when we add per-speaker
   audio treatment).
6. **EditDuet / VideoAgent / CutClaw / GLANCE / LAVE / Crayotter (§6)** — research
   papers with open code; EditDuet `[VERIFIED]` arxiv.org/abs/2509.10761 already
   shaped the critic. WATCH the others for long-form retention/coverage techniques;
   do not import their frameworks.
7. **remotion-dev/skills (#55, 12 SKILL.md files)** `[OWNER-LIST]` — WATCH: only if
   case cards / graphics graduate from FFmpeg-drawn to programmatic scenes.

## 4. Fit-filter rules (why most of section 2 is N-A)

Our product is an evidence-driven documentary pipeline for law-enforcement footage:
multi-hour sources, speaker-colored caption truth, tracked redaction, QA gates,
$0 default cost. Skill packs aimed at Reels/Hormozi captions/vertical shorts/
CapCut drafts/keyframe animation optimize different constraints (hook density,
trend formats, speed over provenance). They fail the fit filter unless they encode a
*law* we lack. The fit filter is: does it improve **evidence → edit decision →
verified render**? Everything else is entertainment-form research.

## 5. Preserved inventory (condensed; owner's numbering)

Owner's preserved caveats: descriptions came from search results, "I haven't opened or
tested every repo… check each README before you install"; links #22–39 and #59–63 came
from curated lists, not visits; #115/#118/#121 from the researcher's own knowledge;
"security: skills run scripts on your machine."

**1. Curated hubs:** 1 zhuyansen/awesome-claude-video-skills `[VERIFIED]` (180 repos,
security-graded) · 2 PhiloLabs/awesome-video-agents · 3 szj2ys/awesome-video-editing ·
4 ningzimu/awesome-skills (Video & Motion) · 5 uhub/awesome-agent-skills ·
6 VoltAgent/awesome-agent-skills (1000+) · 7 agentskillshub.top/best/claude-video-skills
(534 ranked) · 8 HyperFrames: best open-source AI video editors · 9 varg.ai: 37 editing
tools per job · 10 ChatCut: best editing MCP servers · 11 DonkeyCut: AI editors
compared · 12 Valmera: agentic editors.

**2. Editing skill packs:** 13 WhiteTowerAI/cut-as-code · 14 browser-use/video-use
`[VERIFIED]` · 15 social-media-skills/video-editing-skill ·
16 orage-developers/skills-editor · 17 megaartaiking-cmd/agent-video-skills ·
18 6missedcalls/video-editing-skill · 19 riecodes/free247editor ·
20 znyupup/ai-video-editing-skill (five-act vlog) · 21 kavitarak/ai-video-skills ·
22 Agentchengfeng/chengfeng-videocut-skills · 23 kmilo93sd/agent-video-editor ·
24 konradre/video-production-skills · 25 Maciejdziuba/youtube-skills ·
26 pycodebr/heditor · 27 artemnovitckii/claude-editing-skills ·
28 artemnovitckii/claude-animation-skills · 29 alisadiq-ai/favstash-ai-video-editing-skills ·
30 RoverEva/eva-video-style · 31 AKMessi/vex · 32 zenstory-ai/video-recap-skills ·
33 FireRedTeam/FireRed-OpenStoryline · 34 Melvynx/video-agent-skills ·
35 video-db/skills (ingest→search→timeline→HLS) · 36 renezander030/capcut-edit ·
37 MartinDelophy/edit-timeline-studio · 38 Orkas-AI/video-router ·
39 HUANGCHIHHUNGLeo/claude-real-video · 40 ChatCut talking-head-guide ·
41 notque/claude-code-toolkit: video-editing · 42 jeremylongshore video-editor ·
43 Qwen video-edit plugin · 44 kemerd/premiere-agent · 45 Montaj ·
46 barefootford/buttercut · 47 digitalsamba/claude-code-video-toolkit ·
48 seedprod/video-editor-for-claude-code · 49 runesleo/claude-video-kit ·
50 brenoepics/video-to-skill · 51 Banuba/ai-skills · 52 ModelsLab/skills.

**3. Programmatic / motion graphics:** 53 heygen-com/hyperframes · 54 HyperFrames on
Tessl · 55 remotion-dev/skills (12 SKILL.md) · 56 calesthio/OpenMontage ·
57 shuffleo/production-crew · 58 machina-exm/film-studio-skills · 59 pexoai/pexo-skills ·
60 Vincentwei1021/video-shotcraft · 61 Vincentwei1021/video-talkcraft ·
62 Vincentwei1021/anything2explainer · 63 Cuongyd196/remotion-cuongit-template ·
64 0xsline/OpenChatCut (multi-track local editor + skills + MCP).

**4. Pro-editor bridges:** 65 samuelgursky/davinci-resolve-mcp `[VERIFIED]` ·
66 lordhoell/davinci-resolve-mcp (440+ tools claimed) · 67 davinci-mcp (PyPI, 299
typed actions claimed) · 68 davinci-resolve-cli (PyPI, SKILL.md + MCP) ·
69 liuluhaixiu/DaVinci-AutoEdit-Agent · 70 hetpatel-11/Adobe_Premiere_Pro_MCP ·
71 leancoderkavy/premiere-pro-mcp · 72 antipaster/Adobe-Premiere-Pro-MCP (170+) ·
73 ayushozha/AdobePremiereProMCP (7 skills) · 74 mikechambers/adb-mcp ·
75 AJSMonty/adobe-mcp · 76 SysAdminDoc/OpenCut · 77 felixkarevo/kdenlive-mcp
`[VERIFIED]` (Kdenlive 26.04.x + OTIO 0.18.1 export) · 78 relo-video/synthcut (94
tools) · 79 proofcut (PyPI, 93 tools, OTIO/Kdenlive) · 80 Ekaanth/OpenCut-AI ·
81 ChatCut-Inc/agent-plugin.

**5. Big-tech / commercial:** 82 TechCrunch: Adobe MAX 2026 MCP · 83 Adobe: Firefly +
Premiere Color Mode · 84 Cutsio on MAX 2026 · 85 RedShark: Premiere MCP · 86 DEV:
Adobe's MCP door · 87–90 Blackmagic/CG Channel/YMCinema/CineD on Resolve 21.1 ·
91 Google Flow (Veo 3.1) · 92 Pixo on Flow · 93 Tella: Descript MCP · 94 Runway Aleph +
RunAPI skill · 95 invideo agentic editing · 96 ChatCut on Product Hunt · 97–99
Rend/Vidova/Wireflow comparisons. (All `[OWNER-LIST]`, news-class.)

**6. Research (open code):** 100 VideoAgent paper · 101 HKUDS/VideoAgent ·
102 Crayotter paper · 103 idwts/Crayotter · 104 EditDuet `[VERIFIED]`
arxiv.org/abs/2509.10761 · 105 CutClaw paper · 106 GVCLab/CutClaw · 107 GLANCE ·
108 ZihaoLinQZ/GLANCE-Video-Editing-Agent · 109 LAVE · 110 Prompt-driven agentic
editing · 111 Wucy0519/MMLVE · 112 WeVisCut · 113 DIRECT.

**7. Building blocks:** 114 FFmpeg `[ALREADY-BUILT]` · 115 auto-editor ·
116 PySceneDetect · 117 OpenTimelineIO `[ALREADY-BUILT, ADR-0002]` · 118 WhisperX ·
119 Kdenlive · 120 Shotcut · 121 LosslessCut.

**8. Technique write-ups:** 122 Flavio Copes: letting an agent edit my videos ·
123 Chris Lema: eight skill files for post-production · 124 Four Claude Code editing
tools · 125 HackerNoon: transactional editing interfaces · 126 DEV: an editor skill
for OpenClaw · 127–128 NemoVideo skills repo + agent editing workflow.
