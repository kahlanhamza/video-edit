"use client";

// ChatPanel — the chat agent v2 surface: one message in, one bounded turn out.
// The turn is transparent: every tool call with its ok badge, the critic
// verdict, and the reply. Roles are roles, not services (ADR-0011) — the
// panel shows WHICH tools acted, never a black box.

import { useRef, useState } from "react";
import { api } from "@/lib/api";
import { Icon } from "./icons";

interface ChatCall {
  tool: string;
  ok: boolean;
  detail: string;
}

interface ChatTurn {
  role: "user" | "agent";
  text: string;
  calls?: ChatCall[];
  critic?: Record<string, unknown> | null;
  rounds?: number;
  pending?: boolean;
  error?: string;
}

const MAX_HISTORY = 12; // turn pairs kept in the request — context protection

export default function ChatPanel({ projectId }: { projectId: string }) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement | null>(null);

  async function send() {
    const message = input.trim();
    if (!message || busy) return;
    setInput("");
    setBusy(true);

    const history = turns
      .filter((t) => !t.pending && !t.error)
      .slice(-MAX_HISTORY)
      .map((t) => ({ role: t.role === "user" ? "user" : "assistant", content: t.text }));

    setTurns((prev) => [
      ...prev,
      { role: "user", text: message },
      { role: "agent", text: "", pending: true },
    ]);
    requestAnimationFrame(() => listRef.current?.scrollTo({ top: 1e9 }));

    try {
      const out = await api.agentChat(projectId, { message, history });
      setTurns((prev) => [
        ...prev.slice(0, -1),
        {
          role: "agent",
          text: out.reply,
          calls: out.calls,
          critic: out.critic,
          rounds: out.rounds,
        },
      ]);
    } catch (e) {
      setTurns((prev) => [
        ...prev.slice(0, -1),
        {
          role: "agent",
          text: "",
          error: e instanceof Error ? e.message : String(e),
        },
      ]);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => listRef.current?.scrollTo({ top: 1e9 }));
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col border-t border-[var(--line-1)]">
      <div className="panel-header !h-8">
        <span className="micro-label">Agent chat</span>
        <span
          className="chip !text-[9px]"
          title="selector · editor · critic — one bounded turn, checkpointed tools"
        >
          chat v2
        </span>
        <div className="flex-1" />
        {turns.length > 0 && (
          <button
            className="icon-btn !h-5 !w-5"
            title="clear the conversation"
            onClick={() => setTurns([])}
          >
            <Icon name="trash" size={11} />
          </button>
        )}
      </div>

      <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto px-2.5 py-2">
        {turns.length === 0 && (
          <p className="text-[11px] leading-relaxed text-[var(--tx-3)]">
            Ask the agent to act on the timeline — &quot;blur the watermark at
            1:20 and track it&quot;, &quot;what did the agent change so
            far?&quot;. Every tool call is checkpointed, undoable, and shown
            here.
          </p>
        )}
        {turns.map((t, i) => (
          <div key={i} className="mb-2">
            {t.role === "user" ? (
              <div className="ml-6 rounded-lg bg-[var(--bg-2)] px-2.5 py-1.5 text-[12px] text-[var(--tx-1)]">
                {t.text}
              </div>
            ) : t.pending ? (
              <div className="mr-6 flex items-center gap-2 text-[11px] text-[var(--tx-3)]">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--acc)]" />
                thinking…
              </div>
            ) : (
              <div className="mr-6">
                {t.calls && t.calls.length > 0 && (
                  <div className="mb-1 space-y-0.5">
                    {t.calls.map((c, j) => (
                      <div
                        key={j}
                        className="flex items-start gap-1.5 text-[10px] leading-snug"
                        title={c.detail}
                      >
                        <span
                          className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${
                            c.ok ? "bg-[var(--ok)]" : "bg-[var(--err)]"
                          }`}
                        />
                        <span className="font-mono text-[var(--tx-2)]">{c.tool}</span>
                      </div>
                    ))}
                  </div>
                )}
                {t.critic && (
                  <div
                    className={`mb-1 inline-block rounded px-1.5 py-0.5 text-[9px] uppercase tracking-wider ${
                      t.critic.ok
                        ? "bg-[var(--ok-bg, rgba(64,192,120,0.14))] text-[var(--ok)]"
                        : "bg-[rgba(226,163,54,0.14)] text-[var(--warn)]"
                    }`}
                    title={String(t.critic.detail ?? "")}
                  >
                    critic {t.critic.ok ? "green" : "flagged"}
                  </div>
                )}
                {t.error ? (
                  <p className="text-[11px] text-[var(--err)]">{t.error}</p>
                ) : (
                  <p className="text-[12px] leading-relaxed text-[var(--tx-1)]">{t.text}</p>
                )}
                {typeof t.rounds === "number" && (
                  <p className="mt-0.5 text-[9px] text-[var(--tx-3)]">
                    {t.rounds} round{t.rounds === 1 ? "" : "s"}
                  </p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="flex shrink-0 items-center gap-1.5 border-t border-[var(--line-1)] px-2 py-1.5">
        <input
          className="input min-w-0 flex-1 !text-[12px]"
          placeholder={busy ? "agent is working…" : "tell the agent what to change…"}
          value={input}
          disabled={busy}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <button
          className="btn !h-7 !px-2 !text-[11px]"
          disabled={busy || !input.trim()}
          onClick={() => void send()}
        >
          <Icon name="play" size={11} />
        </button>
      </div>
    </div>
  );
}
