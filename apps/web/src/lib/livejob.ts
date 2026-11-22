// Pure live-job state machine — the full-visibility console feed.
// No DOM, no React: the hook (hooks/useLiveJob.ts) feeds SSE events in and
// the studio renders the state out. Step identity is the durable row id
// (`seq_step`) so replays and reconnects upsert instead of duplicating.

import { JobLog, JobStep } from "./api";

export interface LiveJobState {
  job: { stage: string; progress_pct: number; error: string | null } | null;
  steps: JobStep[];
  logs: JobLog[];
  timelineVersion: number | null;
  timelineNote: string;
  connected: boolean;
  done: boolean;
  /** number of events folded in — the UI can key off activity */
  eventsSeen: number;
}

export const MAX_LOG_LINES = 800;

export const initialLiveJobState: LiveJobState = {
  job: null,
  steps: [],
  logs: [],
  timelineVersion: null,
  timelineNote: "",
  connected: false,
  done: false,
  eventsSeen: 0,
};

export type LiveEvent =
  | { type: "snapshot"; steps: JobStep[]; logs: JobLog[] }
  | { type: "job"; stage: string; progress_pct: number; error: string | null }
  | {
      type: "step";
      seq_step: number;
      key: string;
      label: string;
      status: JobStep["status"];
      pct: number;
      detail: string;
      order: number;
      duration_ms?: number;
    }
  | { type: "log"; level: JobLog["level"]; message: string; seq?: number }
  | { type: "timeline"; version: number; note?: string }
  | { type: "connected" }
  | { type: "disconnected" }
  | { type: "done"; stage?: string; progress_pct?: number; error?: string | null };

function upsertStep(steps: JobStep[], incoming: JobStep): JobStep[] {
  const idx = steps.findIndex((s) => s.id === incoming.id);
  if (idx === -1) {
    // insert sorted by id (insertion order == creation order)
    const next = [...steps, incoming];
    next.sort((a, b) => a.id - b.id);
    return next;
  }
  const next = [...steps];
  next[idx] = { ...next[idx], ...incoming };
  return next;
}

export function liveJobReducer(state: LiveJobState, ev: LiveEvent): LiveJobState {
  switch (ev.type) {
    case "connected":
      if (state.connected) return state;
      return { ...state, connected: true };
    case "disconnected":
      if (!state.connected) return state;
      return { ...state, connected: false };
    case "snapshot":
      return {
        ...state,
        steps: ev.steps,
        logs: ev.logs.slice(-MAX_LOG_LINES),
        eventsSeen: state.eventsSeen + 1,
      };
    case "job":
      return {
        ...state,
        job: { stage: ev.stage, progress_pct: ev.progress_pct, error: ev.error },
        eventsSeen: state.eventsSeen + 1,
      };
    case "step":
      return {
        ...state,
        steps: upsertStep(state.steps, {
          id: ev.seq_step,
          key: ev.key,
          label: ev.label,
          status: ev.status,
          pct: ev.status === "done" ? 100 : ev.pct,
          detail: ev.detail,
          order: ev.order,
          duration_ms: ev.duration_ms ?? 0,
        }),
        eventsSeen: state.eventsSeen + 1,
      };
    case "log":
      return {
        ...state,
        logs: [
          ...state.logs,
          { id: ev.seq ?? (state.logs.at(-1)?.id ?? 0) + 1, level: ev.level, message: ev.message },
        ].slice(-MAX_LOG_LINES),
        eventsSeen: state.eventsSeen + 1,
      };
    case "timeline":
      return {
        ...state,
        timelineVersion: ev.version,
        timelineNote: ev.note ?? "",
        eventsSeen: state.eventsSeen + 1,
      };
    case "done":
      return {
        ...state,
        done: true,
        connected: false,
        job: {
          stage: ev.stage ?? state.job?.stage ?? "",
          progress_pct: ev.progress_pct ?? state.job?.progress_pct ?? 100,
          error: ev.error ?? state.job?.error ?? null,
        },
        eventsSeen: state.eventsSeen + 1,
      };
    default:
      return state;
  }
}

/** The step the console should highlight (the running one, else the last). */
export function currentStep(steps: JobStep[]): JobStep | null {
  const running = steps.find((s) => s.status === "running");
  if (running) return running;
  return steps.length ? steps[steps.length - 1] : null;
}

/** Overall console health: derived from steps + job row. */
export function runSummary(state: LiveJobState): {
  done: number;
  failed: number;
  skipped: number;
  running: number;
} {
  const acc = { done: 0, failed: 0, skipped: 0, running: 0 };
  for (const s of state.steps) {
    if (s.status in acc) acc[s.status as keyof typeof acc] += 1;
  }
  return acc;
}
