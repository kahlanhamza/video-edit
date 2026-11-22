"use client";

// RunConsole — the embedded sandbox console: every pipeline step with its own
// progress bar, plus a terminal-style live log feed. Data comes from the
// useLiveJob SSE stream; nothing here talks to the network itself.

import { useEffect, useMemo, useRef, useState } from "react";
import { JobStep } from "@/lib/api";
import { LiveJobState, runSummary } from "@/lib/livejob";

const STEP_ICON: Record<JobStep["status"], string> = {
  done: "✓",
  failed: "✗",
  skipped: "»",
  running: "◈",
};

const STEP_TONE: Record<JobStep["status"], string> = {
  done: "text-emerald-400",
  failed: "text-red-400",
  skipped: "text-slate-500",
  running: "text-amber-300",
};

const BAR_TONE: Record<JobStep["status"], string> = {
  done: "bg-emerald-500",
  failed: "bg-red-500",
  skipped: "bg-slate-700",
  running: "bg-amber-400",
};

const LOG_TONE: Record<string, string> = {
  info: "text-slate-300",
  warn: "text-amber-300",
  error: "text-red-400",
};

function fmtDuration(ms: number): string {
  if (ms <= 0) return "";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export default function RunConsole({ live }: { live: LiveJobState }) {
  const [follow, setFollow] = useState(true);
  const [filter, setFilter] = useState("");
  const logRef = useRef<HTMLDivElement | null>(null);
  const summary = runSummary(live);
  const runningStep = live.steps.find((s) => s.status === "running");

  const logs = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return live.logs;
    return live.logs.filter((l) => l.message.toLowerCase().includes(q));
  }, [live.logs, filter]);

  // follow mode: pin the log feed to the bottom as lines arrive
  useEffect(() => {
    if (follow && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs.length, follow]);

  const pct = live.job?.progress_pct ?? 0;
  const stage = live.job?.stage ?? "idle";

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* header: stage + overall progress + counters */}
      <div className="border-b border-slate-800 px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <span
              className={`inline-block h-2 w-2 shrink-0 rounded-full ${
                live.job?.error
                  ? "bg-red-500"
                  : live.done
                    ? "bg-emerald-500"
                    : live.connected
                      ? "animate-pulse bg-amber-400"
                      : "bg-slate-600"
              }`}
              title={live.connected ? "streaming" : live.done ? "finished" : "reconnecting…"}
            />
            <span className="truncate text-sm font-semibold text-slate-100">
              stage {stage}
            </span>
            {live.timelineVersion != null && (
              <span className="shrink-0 rounded bg-violet-500/10 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-violet-300">
                v{live.timelineVersion}
              </span>
            )}
          </div>
          <span className="shrink-0 text-xs tabular-nums text-slate-400">
            {pct.toFixed(0)}% · {summary.done}✓ {summary.failed}✗ {summary.skipped}»
          </span>
        </div>
        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded bg-slate-800">
          <div
            className={`h-1.5 rounded transition-all duration-300 ${
              live.job?.error ? "bg-red-500" : "bg-amber-400"
            }`}
            style={{ width: `${Math.min(100, pct)}%` }}
          />
        </div>
        {live.job?.error && (
          <p className="mt-1.5 line-clamp-2 rounded bg-red-500/10 px-2 py-1 text-xs text-red-300">
            {live.job.error}
          </p>
        )}
      </div>

      {/* step list: one progress bar per pipeline step */}
      <ol className="max-h-[38vh] overflow-y-auto border-b border-slate-800 px-3 py-2">
        {live.steps.length === 0 && (
          <li className="text-xs text-slate-500">
            no steps yet — press Run and every stage appears here live
          </li>
        )}
        {live.steps.map((s) => (
          <li key={s.id} className="py-1">
            <div className="flex items-center gap-2">
              <span className={`shrink-0 text-xs ${STEP_TONE[s.status]}`}>
                {STEP_ICON[s.status]}
              </span>
              <span
                className={`min-w-0 flex-1 truncate text-xs ${
                  s.status === "running"
                    ? "font-semibold text-amber-200"
                    : s.status === "failed"
                      ? "text-red-300"
                      : "text-slate-300"
                }`}
                title={s.detail || s.label}
              >
                {s.label}
              </span>
              {s.status === "running" && (
                <span className="shrink-0 text-[10px] tabular-nums text-amber-300/80">
                  {s.pct.toFixed(0)}%
                </span>
              )}
              {s.duration_ms > 0 && (
                <span className="shrink-0 text-[10px] tabular-nums text-slate-500">
                  {fmtDuration(s.duration_ms)}
                </span>
              )}
            </div>
            {(s.status === "running" || s.status === "done") && s.pct > 0 && (
              <div className="ml-6 mt-0.5 h-1 w-[calc(100%-1.5rem)] overflow-hidden rounded bg-slate-800">
                <div
                  className={`h-1 rounded ${BAR_TONE[s.status]} transition-all duration-300`}
                  style={{ width: `${Math.min(100, s.pct)}%` }}
                />
              </div>
            )}
            {s.status === "running" && s.detail && (
              <p className="ml-6 truncate text-[10px] text-slate-500">{s.detail}</p>
            )}
          </li>
        ))}
        {runningStep?.detail && (
          <li className="pt-1 text-[10px] text-slate-500">now: {runningStep.detail}</li>
        )}
      </ol>

      {/* sandbox log feed */}
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="flex items-center gap-2 px-3 pt-2">
          <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
            sandbox console
          </span>
          <input
            className="input !h-6 flex-1 !py-0 text-xs"
            placeholder="filter lines…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <label className="flex shrink-0 items-center gap-1 text-[10px] text-slate-400">
            <input
              type="checkbox"
              checked={follow}
              onChange={(e) => setFollow(e.target.checked)}
            />
            follow
          </label>
        </div>
        <div
          ref={logRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
            if (!atBottom && follow) setFollow(false);
          }}
          className="mt-1.5 min-h-0 flex-1 overflow-y-auto px-3 pb-2 font-mono text-[11px] leading-4"
        >
          {logs.length === 0 && (
            <p className="text-slate-600">$ waiting for pipeline output…</p>
          )}
          {logs.map((l) => (
            <p
              key={l.id}
              className={`whitespace-pre-wrap break-words ${LOG_TONE[l.level] ?? "text-slate-300"}`}
            >
              <span className="select-none text-slate-600">$ </span>
              {l.message}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}
