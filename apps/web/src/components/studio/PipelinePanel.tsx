"use client";

// PipelinePanel — the observability heart: every pipeline step with its own
// progress bar, live stage + overall progress, error surface. Data comes from
// the useLiveJob SSE stream (nothing here talks to the network).

import { useEffect, useRef } from "react";
import { Icon } from "./icons";
import type { LiveJobState } from "@/lib/livejob";
import { runSummary } from "@/lib/livejob";

function fmtDuration(ms: number): string {
  if (ms <= 0) return "";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export default function PipelinePanel({ live }: { live: LiveJobState }) {
  const summary = runSummary(live);
  const pct = live.job?.progress_pct ?? 0;
  const stage = live.job?.stage ?? "idle";
  const stepsRef = useRef<HTMLDivElement | null>(null);
  const runningId = live.steps.find((s) => s.status === "running")?.id ?? null;

  // follow the running step — the list always shows what is happening now
  // (the build-log pattern; no dead fold, no hunting for the active row)
  useEffect(() => {
    const el = stepsRef.current;
    if (!el || runningId == null) return;
    const row = el.querySelector<HTMLElement>(`[data-step-id="${runningId}"]`);
    if (row) {
      const target = row.offsetTop - el.clientHeight / 2 + row.offsetHeight / 2;
      el.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
    }
  }, [runningId, live.steps.length]);

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="panel-header !justify-between">
        <span className="micro-label">Pipeline</span>
        <div className="flex items-center gap-2">
          <span className="tcode text-[11px] text-[var(--tx-2)]">
            {summary.done}✓ · {summary.skipped}» · {summary.failed}✗
          </span>
          <span
            className={`flex items-center gap-1.5 text-[11px] font-medium ${
              live.job?.error
                ? "text-[var(--err)]"
                : live.done
                  ? "text-[var(--ok)]"
                  : live.connected
                    ? "text-[var(--acc)]"
                    : "text-[var(--tx-3)]"
            }`}
          >
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${
                live.job?.error
                  ? "bg-[var(--err)]"
                  : live.done
                    ? "bg-[var(--ok)]"
                    : live.connected
                      ? "animate-pulse bg-[var(--acc)]"
                      : "bg-[var(--tx-3)]"
              }`}
            />
            {live.job?.error ? "error" : live.done ? "done" : live.connected ? "live" : "idle"}
          </span>
        </div>
      </div>

      {/* overall */}
      <div className="border-b border-[var(--line-1)] px-3 py-2.5">
        <div className="flex items-baseline justify-between gap-2">
          <span className="truncate text-[12px] font-semibold text-[var(--tx-1)]">
            {stage}
          </span>
          <span className="tcode text-[12px] text-[var(--tx-2)]">
            {pct.toFixed(0)}%
          </span>
        </div>
        <div className="bar-track mt-1.5">
          <div
            className="bar-fill"
            data-tone={live.job?.error ? "err" : undefined}
            style={{ width: `${Math.min(100, pct)}%` }}
          />
        </div>
        {live.job?.error && (
          <p className="mt-2 line-clamp-3 rounded border border-[rgba(241,109,109,.25)] bg-[rgba(241,109,109,.08)] px-2 py-1.5 text-[11px] leading-relaxed text-[var(--err)]">
            {live.job.error}
          </p>
        )}
      </div>

      {/* steps — one progress bar each; follows the running step */}
      <div className="relative min-h-0 flex-1">
        <div
          ref={stepsRef}
          className="tl-scroll absolute inset-0 overflow-y-auto px-2 py-1.5 pb-3"
        >
          {live.steps.length === 0 && (
            <p className="px-1 py-3 text-[11px] leading-relaxed text-[var(--tx-3)]">
              Every stage appears here the moment it starts — ingest, perception,
              planning, redaction, render — each with its own progress bar.
            </p>
          )}
          {live.steps.map((s) => {
          const tone =
            s.status === "done"
              ? "var(--ok)"
              : s.status === "failed"
                ? "var(--err)"
                : s.status === "running"
                  ? "var(--acc)"
                  : "var(--tx-3)";
          return (
            <div key={s.id} data-step-id={s.id} className="px-1 py-1">
              <div className="flex items-center gap-2">
                <span className="w-3 shrink-0" style={{ color: tone }}>
                  {s.status === "running" ? (
                    <span className="inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-current" />
                  ) : s.status === "done" ? (
                    <Icon name="check" size={12} />
                  ) : s.status === "failed" ? (
                    <Icon name="x" size={12} />
                  ) : (
                    <span className="text-[10px]">»</span>
                  )}
                </span>
                <span
                  className={`min-w-0 flex-1 truncate text-[11.5px] ${
                    s.status === "running"
                      ? "font-semibold text-[var(--tx-1)]"
                      : s.status === "failed"
                        ? "text-[var(--err)]"
                        : "text-[var(--tx-2)]"
                  }`}
                  title={s.detail || s.label}
                >
                  {s.label}
                </span>
                {s.status === "running" && (
                  <span className="tcode shrink-0 text-[10px] text-[var(--acc)]">
                    {s.pct.toFixed(0)}%
                  </span>
                )}
                {s.duration_ms > 0 && (
                  <span className="shrink-0 text-[10px] tabular-nums text-[var(--tx-3)]">
                    {fmtDuration(s.duration_ms)}
                  </span>
                )}
              </div>
              <div className="ml-5 mt-1 h-[3px] overflow-hidden rounded-full bg-[var(--bg-3)]">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${s.status === "done" ? 100 : Math.min(100, s.pct)}%`,
                    background:
                      s.status === "failed" ? "var(--err)" : s.status === "done" ? "var(--ok)" : "var(--acc)",
                  }}
                />
              </div>
              {s.status === "running" && s.detail && (
                <p className="ml-5 mt-0.5 truncate text-[10px] text-[var(--tx-3)]">
                  {s.detail}
                </p>
              )}
            </div>
          );
          })}
        </div>
        {/* fade so the fold reads as "more below", never as a cut */}
        <div
          className="pointer-events-none absolute inset-x-0 bottom-0 h-6"
          style={{ background: "linear-gradient(to top, var(--bg-1), transparent)" }}
        />
      </div>
    </div>
  );
}
