"use client";

// ConsolePanel — the embedded sandbox console: a real terminal feed with
// level tones, filter, and follow-the-tail. Streamed over SSE by useLiveJob.

import { useEffect, useMemo, useRef, useState } from "react";
import type { LiveJobState } from "@/lib/livejob";

const LEVEL_TONE: Record<string, string> = {
  info: "text-[var(--tx-2)]",
  warn: "text-[var(--warn)]",
  error: "text-[var(--err)]",
};

export default function ConsolePanel({ live }: { live: LiveJobState }) {
  const [follow, setFollow] = useState(true);
  const [filter, setFilter] = useState("");
  const logRef = useRef<HTMLDivElement | null>(null);

  const logs = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return live.logs;
    return live.logs.filter((l) => l.message.toLowerCase().includes(q));
  }, [live.logs, filter]);

  useEffect(() => {
    if (follow && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs.length, follow]);

  return (
    <div className="flex min-h-0 flex-1 flex-col border-t border-[var(--line-1)]">
      <div className="panel-header !h-8">
        <span className="micro-label">Sandbox console</span>
        <div className="flex-1" />
        <input
          className="input !h-5 !w-28 !rounded !px-1.5 !text-[10px]"
          placeholder="filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <button
          className="flex items-center gap-1 text-[10px] text-[var(--tx-3)] hover:text-[var(--tx-1)]"
          onClick={() => setFollow((v) => !v)}
          title="auto-scroll to the newest line"
        >
          <span className="toggle !h-[13px] !w-[23px]" data-on={follow} />
          follow
        </button>
      </div>
      <div
        ref={logRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          if (!atBottom && follow) setFollow(false);
        }}
        className="min-h-0 flex-1 overflow-y-auto bg-[var(--bg-0)] px-3 py-2"
      >
        {logs.length === 0 ? (
          <p className="console-line text-[var(--tx-3)]">
            $ waiting for pipeline output…
          </p>
        ) : (
          logs.map((l) => (
            <p key={l.id} className={`console-line ${LEVEL_TONE[l.level] ?? "text-[var(--tx-2)]"}`}>
              <span className="select-none text-[var(--tx-3)]">$ </span>
              {l.message}
            </p>
          ))
        )}
      </div>
    </div>
  );
}
