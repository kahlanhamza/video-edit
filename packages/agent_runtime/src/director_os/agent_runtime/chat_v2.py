"""Chat agent v2 — the orchestrator turn loop (agent-frame-editor roadmap step 5).

One user message -> ONE bounded turn with the house laws enforced end-to-end:

- roles are roles, not services (ADR-0011): the SAME typed registry serves a
  selector (look: ops.list / transcript windows / frame.grab / selection
  helpers), an editor (mutate: ops.propose / selection.track / timeline
  tools) and a critic (qa.self_audit) — the model switches hats inside the
  turn, the tools stay the audited, checkpointed surface.
- maker-checker discipline (EditDuet + Azure patterns): the editor proposes
  calls, the critic audits the result, and ONE repair round is allowed —
  iteration caps + explicit completion criteria, never an infinite loop.
- fail-closed WITH feedback: an unknown tool or a failed dispatch becomes a
  tool result the next round sees (never silently dropped, never a crash).
- evidence-before-action and everything-is-an-op ride the tools themselves —
  the loop does not weaken them.

Context pack = op digest + evidence inventory + tool catalog: the agent
perceives the timeline state BEFORE touching it (timeline-aware law).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

MAX_TOOL_ROUNDS = 3  # Azure maker-checker law: capped, with fallback
MAX_REPAIR_ROUNDS = 1  # exactly one critic-driven repair — no loops
MAX_CATALOG_TOOLS = 48  # context protection: the catalog, not the prompt, explodes


class ChatTurnResult(BaseModel):
    reply: str
    rounds: int = 0
    calls: list[dict[str, Any]] = Field(default_factory=list)
    critic: dict[str, Any] | None = None
    version_before: int | None = None
    version: int | None = None
    detail: str = ""


class ChatLLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any] | None:
    """First balanced JSON object in the text (fenced or raw) — None on miss."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def build_context_pack(
    ops_store: Any,
    registry: Any,
    *,
    timeline_summary: str = "",
    extras: dict[str, str] | None = None,
) -> str:
    """What the agent perceives BEFORE acting: op digest + tool catalog.

    Bounded by construction: the op digest is the fold (counts + compact
    effects), the catalog is names + one-line purposes — never full payloads.
    """
    lines: list[str] = []
    if timeline_summary:
        lines.append(f"TIMELINE: {timeline_summary[:1200]}")
    try:
        fold = ops_store.fold()
        lines.append(
            f"OP LOG: rev {fold.rev} — {fold.active_ops} active, {fold.revoked_ops} revoked"
        )
        for e in fold.effects[:12]:
            tracked = " tracked" if e.get("mode") == "tracked" else ""
            lines.append(
                f"  - {e.get('kind')} [{e.get('x')},{e.get('y')} {e.get('w')}x{e.get('h')}]"
                f"{tracked} frames {e.get('frame_start')}..{e.get('frame_end')} "
                f"by {e.get('origin')} (op {str(e.get('op_id'))[:8]})"
            )
        if fold.caption_overrides:
            lines.append(f"  - caption overrides on cues {sorted(fold.caption_overrides)[:8]}")
    except Exception as e:  # noqa: BLE001 — a fold failure is context, not a crash
        lines.append(f"OP LOG: unavailable ({e})")
    catalog = []
    for name in registry.names()[:MAX_CATALOG_TOOLS]:
        try:
            spec = registry.get(name)
            catalog.append(f"  {name} — {str(spec.description)[:110]}")
        except Exception:  # noqa: BLE001
            continue
    lines.append("TOOLS (call by exact name):")
    lines.extend(catalog)
    for k, v in (extras or {}).items():
        lines.append(f"{k.upper()}: {v[:400]}")
    return "\n".join(lines)


_CHAT_SYSTEM = """You are DirectorOS's editing agent. One turn, three hats:
SELECTOR — look before you act (ops.list, transcript windows, frame.grab).
EDITOR — mutate ONLY through tools; every op is evidence-backed and undoable.
CRITIC — when asked to verify, run qa.self_audit and report honestly.

Laws you must obey:
- perceive first: read the context pack; never guess op ids or frame numbers.
- evidence-before-action: blur/track edits cite the frame you grabbed.
- frame-addressed ops: name frame_start/frame_end when the edit scopes frames.
- undo = revoke: fix a wrong op by revoking it (ops.revoke), never by silence.
- reply in the user's language, concretely, with op ids when things changed.

Respond with ONE JSON object, no prose outside it:
{"reply": <string|null — the user-facing answer when done>,
 "calls": [{"tool": "<exact name>", "args": {...}, "note": "<why>"}],
 "done": <bool>}
Finish (done=true) as soon as the request is satisfied; do not invent work."""


def run_chat_turn(
    backend: Any,
    registry: Any,
    executor: Any,
    ops_store: Any,
    *,
    user_message: str,
    context_pack: str,
    history: list[dict[str, str]] | None = None,
    timeline: Any = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    max_tokens: int = 2048,
) -> ChatTurnResult:
    """One chat turn: perceive -> act -> (critic -> one repair) -> answer."""
    calls_log: list[dict[str, Any]] = []
    last_reply = ""
    rounds = 0
    run_ok = True

    convo: list[dict[str, str]] = [
        {"role": "user", "content": f"CONTEXT PACK:\n{context_pack}"},
        *(history or [])[-8:],
        {"role": "user", "content": f"REQUEST: {user_message}"},
    ]

    for _round in range(max_tool_rounds):
        rounds += 1
        block = convo[-2:] if len(convo) > 2 else convo
        suffix = ""
        if calls_log:
            suffix = "\nLAST TOOL RESULTS:\n" + "\n".join(
                f"  {c['tool']}: {'ok' if c['ok'] else 'FAILED'} — {c['detail'][:220]}"
                for c in calls_log[-6:]
            )
        raw = backend.complete_json(
            _CHAT_SYSTEM,
            "\n".join(u["content"] for u in block) + suffix + "\nRespond with the JSON object now.",
            max_tokens=max_tokens,
        )
        parsed = _extract_json(raw) or {}
        reply = parsed.get("reply")
        if isinstance(reply, str) and reply.strip():
            last_reply = reply.strip()
        call_specs = parsed.get("calls") or []
        if not call_specs:
            # no calls this round: honor the reply, else one nudge round
            if last_reply or parsed.get("done") is True:
                break
            if _round == max_tool_rounds - 1:
                break
            continue

        # EXECUTE first — a model that sends calls with done=true still gets
        # them dispatched (an op proposed-but-never-run would be a silent lie)
        for spec in call_specs[:6]:  # a round executes at most 6 calls
            if not isinstance(spec, dict):
                continue
            tool = str(spec.get("tool", ""))
            raw_args = spec.get("args")
            args: dict[str, Any] = dict(raw_args) if isinstance(raw_args, dict) else {}
            note = str(spec.get("note", ""))
            if tool not in registry.names():
                # fail-closed WITH feedback: the next round sees the miss
                calls_log.append(
                    {
                        "tool": tool or "?",
                        "ok": False,
                        "detail": f"unknown tool (known: {', '.join(registry.names()[:12])}...)",
                    }
                )
                continue
            from director_os.agent_runtime.agents import ToolStep

            run = executor.execute_plan(
                [ToolStep(tool=tool, args=args, checkpoint_note=note)], timeline
            )
            timeline = run.timeline  # authoritative post-run object (rollback-safe)
            run_ok = run_ok and run.ok
            for sr in run.steps:
                calls_log.append({"tool": sr.tool, "ok": sr.ok, "detail": sr.detail})
            if not run.ok:
                break  # a failed step rolls back + the next round sees the failure

        if parsed.get("done") is True:
            break  # completion criterion met AFTER the calls landed

    # critic gate (maker-checker): audit, then at most ONE repair round
    critic: dict[str, Any] | None = None
    if "qa.self_audit" in registry.names() and calls_log:
        try:
            critic = registry.dispatch("qa.self_audit", {})
        except Exception as e:  # noqa: BLE001 — critic failure is reported, not fatal
            critic = {"ok": False, "detail": f"critic dispatch failed: {e}"}
        if (
            critic
            and critic.get("ok") is False
            and last_reply
            and rounds < max_tool_rounds + MAX_REPAIR_ROUNDS
        ):
            raw = backend.complete_json(
                _CHAT_SYSTEM,
                f"CONTEXT PACK:\n{context_pack}\nREQUEST: {user_message}\n"
                f"CRITIC VERDICT (failed): {json.dumps(critic)[:600]}\n"
                "Repair now: revoke/fix the failing ops, then answer. JSON object now.",
                max_tokens=max_tokens,
            )
            parsed = _extract_json(raw) or {}
            if isinstance(parsed.get("reply"), str) and parsed["reply"].strip():
                last_reply = parsed["reply"].strip()
            for spec in (parsed.get("calls") or [])[:4]:
                if isinstance(spec, dict) and spec.get("tool") in registry.names():
                    from director_os.agent_runtime.agents import ToolStep

                    run = executor.execute_plan(
                        [
                            ToolStep(
                                tool=str(spec["tool"]),
                                args=dict(spec["args"])
                                if isinstance(spec.get("args"), dict)
                                else {},
                                checkpoint_note=str(spec.get("note", "")),
                            )
                        ],
                        timeline,
                    )
                    timeline = run.timeline
                    for sr in run.steps:
                        calls_log.append({"tool": sr.tool, "ok": sr.ok, "detail": sr.detail})
            try:
                critic = registry.dispatch("qa.self_audit", {})
            except Exception as e:  # noqa: BLE001
                critic = {"ok": False, "detail": f"critic re-dispatch failed: {e}"}

    if not last_reply:
        # deterministic fallback — the turn NEVER answers with nothing
        ok_calls = [c for c in calls_log if c["ok"]]
        if ok_calls:
            last_reply = "Done: " + "; ".join(
                f"{c['tool']} — {c['detail'][:120]}" for c in ok_calls[:3]
            )
        elif calls_log:
            last_reply = "I could not complete that: " + "; ".join(
                f"{c['tool']} — {c['detail'][:120]}" for c in calls_log[:2]
            )
        else:
            last_reply = "I read the timeline but found nothing to change for that request."

    return ChatTurnResult(
        reply=last_reply,
        rounds=rounds,
        calls=calls_log,
        critic=critic,
        detail=f"{rounds} round(s), {len(calls_log)} call(s), "
        f"critic {'ok' if (critic or {}).get('ok') else 'n/a'}",
    )
