# Agent Research Dossier — verified against primary sources (2026-11-21)

> **Purpose:** the owner supplied a research document ("How big-tech teams build
> tool-calling, multi-agent systems", 10 pp., 36 primary links + a 128-link skill-mining
> list). The owner's rule: **nothing in it is trusted on faith** — every load-bearing
> claim was re-verified against the primary source before it touched this codebase.
> This file is the durable record of that verification, so no future session re-derives it.
> The owner's original list is preserved in [skills-mining.md](skills-mining.md).
>
> **Method:** each claim below was checked by opening the primary source (or, where
> marked, an authoritative secondary quoting it). Sources verified live 2026-11-21
> unless noted. The Anthropic posts moved from `/engineering/...` to `/research/...`
> paths during a site migration — old absolute links in third-party writeups may 404;
> the `/research/` paths below are the live ones.

---

## 1. Verified claims → what they changed here

| # | Claim (from owner's doc) | Primary source | Verdict | Landed as |
|---|---|---|---|---|
| 1 | "An agent is an LLM using tools autonomously in a loop" | Anthropic, *Building Effective Agents* — https://www.anthropic.com/research/building-effective-agents (2024-12-19) | CONFIRMED (also: start with composable patterns — chaining, routing — before autonomous loops) | `PlannerExecutor` plan→dispatch→observe loop (`agent_runtime/agents.py`) |
| 2 | Multi-agent: Opus-lead + Sonnet-subagents beat single agent by 90.2% on internal research eval; ~15× chat tokens; ~4× for single-agent | Anthropic, *How we built our multi-agent research system* — https://www.anthropic.com/research/multi-agent-research-system | CONFIRMED verbatim in the source — **and** the source itself warns most coding tasks have "fewer truly parallelizable pieces"; video cuts are sequential/interdependent | REJECTED for the editing loop; single agent + deterministic critic instead (ADR-0011) |
| 3 | Tool design for agents, not for developers: few tools, clear boundaries, compact results, descriptions written as carefully as prompts, **"run evaluations of your tools"** | Anthropic, *Writing effective tools for AI agents* — https://www.anthropic.com/research/writing-tools-for-agents (2025-09-11) | CONFIRMED | 35-tool typed registry with Pydantic schemas + per-tool docstrings; **`scripts/agent_evals.py`** (6 fixed scenarios) exists because of the eval sentence |
| 4 | Context engineering > prompt engineering; context is a finite "attention budget"; smallest high-signal token set wins | Anthropic, *Effective context engineering for AI agents* — https://www.anthropic.com/research/effective-context-engineering-for-ai-agents | CONFIRMED | story_map.py condenses a 43-min tape into ~2.5k tokens of 1-min buckets; tool results return compact JSON digests, never raw media |
| 5 | Long-running pattern: initializer sets up env once, **worker makes incremental progress and leaves artifacts; progress files ARE the memory** | Anthropic, *Effective harnesses for long-running agents* — https://www.anthropic.com/research/effective-harnesses-for-long-running-agents (2025-11-26) | CONFIRMED | `session_memory.py` — `session.plan` / `session.notes` / `session.status` (atomic writes, torn-write fallback, tested process-death resume) |
| 6 | MCP = agent↔tool plug ("vertical"); A2A = agent↔agent ("horizontal"); A2A adoption lagged MCP | A2A Protocol docs, *A2A and MCP: Detailed Comparison* — https://a2a-protocol.org/latest/topics/a2a-and-mcp | CONFIRMED — the horizontal/vertical framing is the docs' own | ADR-0011: no A2A, no MCP in the editing loop; single-team single-agent |
| 7 | MCP donated to the Linux Foundation's Agentic AI Foundation (Dec 2025); MCP open-sourced Nov 2024 | MCP blog, *MCP joins the Agentic AI Foundation* — https://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation (2025-12-09); LF press release https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation | CONFIRMED (AAIF co-founded with Block + OpenAI; also hosts AGENTS.md and goose) | Protocol context only — no code impact (ADR-0011 keeps in-process registry) |
| 8 | EditDuet = Editor agent + Critic agent for non-linear video editing | EditDuet — https://arxiv.org/abs/2509.10761 (Sandoval-Castaneda et al., 2025) | CONFIRMED | `critic.py` — `qa.self_audit` as a deterministic critic tool (our critic is code, not a second LLM: cheaper and non-negotiable) |
| 9 | Guardrails are runtime controls, not prompt instructions; external data is untrusted; irreversible actions need human approval | OWASP, *AI Agent Security Cheat Sheet* — https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html | CONFIRMED (cheat sheet exists and covers these classes) — **but see correction C1**: "fail-closed" + "policy separation" wording is NOT OWASP's | checkpoint-before-execute + undo + publish gate; `REDACT_MODE=tracked` refuses empty plans; agent writes only inside the project work dir |
| 10 | "Fail closed: unknown tools require approval" + decide/execute separation | Google, *An Introduction to Google's Approach to AI Agent Security* (Díaz, Kern, Olive) — accessed via Simon Willison's review https://simonw.substack.com/p/the-lethal-trifecta-for-ai-agents (Willison quotes it as "our aspirational framework for secure AI agents"); OpenAI, *A Practical Guide to Building Agents* — https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents (guardrails chapter) | CONFIRMED as Google/OpenAI language | Correction C1 below; **fail-closed WITH feedback** in both LLM backends (`llm.py`) — unknown tool names are rejected with the known-tool list so the model can recover |
| 11 | Framework table (Claude Agent SDK / OpenAI Agents SDK / Google ADK / LangGraph / MS Agent Framework / Pydantic AI) | Vendor docs (not individually opened this session — the doc itself flags comparisons as fast-rotating) | NOT RE-VERIFIED — deliberate: we are framework-free (own harness, pure httpx per SKILL.md), so the comparison was informational only | No adoption |

## 2. Corrections to the owner's doc (so we never re-derive these)

- **C1 — Attribution.** The doc attributed "fail closed: unknown tools require
  approval" and "separate deciding from executing" to the OWASP cheat sheet. OWASP's
  sheet covers the *class* of guidance, but the specific vocabulary and the
  decide/execute (policy/enforcer) split come from Google's agent-security framework
  and OpenAI's agents guide. Recorded as Google/OpenAI language in ADR-0011's context.
- **C2 — A2A.** The doc floated A2A as the plug standard for agent collaboration, then
  half-retracted it ("for a single-team project, you almost certainly don't need A2A").
  Verified and settled: A2A is for cross-org, cross-vendor agent fleets; our editing
  loop is one agent over an in-process registry. **ADR-0011** (2026-11-21) ratified:
  no A2A, no MCP for the editing loop; revisit only when an external NLE (Resolve,
  Premiere) must be driven directly.
- **C3 — Multi-agent economics.** The doc's 90.2% headline is real but scope-limited:
  the source attributes it to *breadth-first research evals*, and explicitly says
  domains with shared context and interdependent steps (coding, by extension
  sequential video cuts) are a poor fit today. We took the warning, not the headline.

## 3. Adopted-pattern → artifact map

| Research pattern | Artifact in this repo |
|---|---|
| Agent = loop (claim 1) | `agent_runtime/agents.py` `PlannerExecutor` |
| Typed tools + descriptions as prompts (claim 3) | `agent_runtime/registry.py` + 35 tools across `editing_tools` / `inspection_tools` / `workshop_tools` / `session_memory` / `critic` |
| Evals for tools (claim 3) | `scripts/agent_evals.py` — 6 hermetic scenarios, same set scores every agent version |
| Progress files ARE the memory (claim 5) | `agent_runtime/session_memory.py` (+ tested torn-write fallback) |
| Durable execution / survives crashes | `agent_runtime/checkpoint.py` (checkpoint-before-execute, rollback rebinds `REG_TIMELINE`), pipeline domain-table persistence (ADR-0009) |
| Editor/Critic pairing (claim 8) | `agent_runtime/critic.py` `qa.self_audit` — deterministic, honest-absence semantics |
| Fail-closed with feedback (claim 10) | `agent_runtime/llm.py` — unknown proposals rejected with the known-tool list fed back to the model |
| Guardrails ≠ prompts (claim 9) | publish gate, undo/checkpoint, destructive-tool consent (`trim_clip` flagged DESTRUCTIVE), work-copy caption edits (originals never touched) |
| Skills carry craft, not context (owner doc §6) | `packages/skills/bodycam_documentary.md` v0.5 — sourced craft constants (Netflix caption timing, silence thresholds, punch-in law, YouTube audio chain, OpenChatCut provenance lawbook) |
| Human approval before export | Studio publish gate; ship step is operator-gated |

## 4. Deliberately NOT adopted (with reasons, so no future session re-litigates)

- **MCP / A2A in the editing loop** — ADR-0011. Protocol tax + token explosion for a
  single-team, single-agent loop over an in-process registry. The *patterns* the
  pro-editor bridges proved were adopted instead (read-back receipts, honest-absence
  semantics, consent for destructive ops).
- **Multi-agent subagent fleets** — C3: cuts are interdependent; 15× tokens buys
  parallel breadth, which an edit does not have.
- **LangGraph / Temporal / MS Agent Framework now** — durable execution is already
  satisfied by domain-table persistence + checkpoint store (ADR-0004 keeps Temporal
  as a later swap, not a dependency).
- **Tracing vendors (LangSmith/Langfuse/Phoenix)** — noted, not installed; current
  tracing is the `agent_tool_logs` table + eval report JSON. Revisit if agent runs
  become opaque.

## 5. Re-verification pointers for future sessions

Anthropic posts live under `anthropic.com/research/...`: `building-effective-agents`,
`multi-agent-research-system`, `writing-tools-for-agents`,
`effective-context-engineering-for-ai-agents`,
`effective-harnesses-for-long-running-agents`. A2A comparison:
`a2a-protocol.org/latest/topics/a2a-and-mcp`. MCP governance:
`blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation`.
EditDuet: `arxiv.org/abs/2509.10761`. OWASP:
`cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html`.
If a claim in the owner's PDF lacks a row above, treat it as UNVERIFIED and check the
primary source before acting on it.
