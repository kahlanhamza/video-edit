"use client";

// useLiveJob — SSE subscription to /jobs/{id}/live with snapshot bootstrap and
// exponential reconnect. Feeds every frame through the pure liveJobReducer.
//
// The resume law (owner 2026-11-23): Run reuses the SAME job id, so a plain
// [jobId] dependency would never re-run the effect after a failure closed the
// stream — "run again does nothing". `reconnect()` bumps an epoch that reopens
// the EventSource even when the job id is unchanged.

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { api, liveSseUrl } from "@/lib/api";
import { initialLiveJobState, liveJobReducer, LiveEvent, LiveJobState } from "@/lib/livejob";

const RECONNECT_BASE_MS = 1200;
const MAX_RETRIES = 6;

export interface LiveJobHandle {
  state: LiveJobState;
  /** force a fresh snapshot (e.g. after Run dispatches a new job) */
  resync: () => void;
  /** reopen the SSE stream even for the SAME job id (Run-again after failure) */
  reconnect: () => void;
}

export function useLiveJob(
  projectId: string | null,
  jobId: string | null,
  onTimelineRevision?: (version: number, note: string) => void,
): LiveJobHandle {
  const [state, dispatch] = useReducer(liveJobReducer, initialLiveJobState);
  // epoch: bumped by reconnect() to force the effect to re-run (the stream
  // latches closed on `done` — including the failure path — and a resumed job
  // keeps its id, so the id alone can never retrigger it)
  const [epoch, setEpoch] = useState(0);
  const esRef = useRef<EventSource | null>(null);
  const retriesRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closedRef = useRef(false);
  const timelineCbRef = useRef(onTimelineRevision);
  timelineCbRef.current = onTimelineRevision;

  const resync = useCallback(() => {
    if (!projectId || !jobId) return;
    api
      .jobSnapshot(projectId, jobId)
      .then((snap) =>
        dispatch({ type: "snapshot", steps: snap.steps, logs: snap.logs }),
      )
      .catch(() => {
        /* project/job may not exist yet — the SSE stream will populate */
      });
  }, [projectId, jobId]);

  const reconnect = useCallback(() => {
    setEpoch((e) => e + 1);
  }, []);

  useEffect(() => {
    if (!projectId || !jobId) return;
    closedRef.current = false;
    retriesRef.current = 0;

    const connect = () => {
      if (closedRef.current) return;
      esRef.current?.close();
      const es = new EventSource(liveSseUrl(projectId, jobId));
      esRef.current = es;

      es.onopen = () => {
        retriesRef.current = 0;
        dispatch({ type: "connected" });
        resync(); // fresh steps+logs every time the stream opens
      };

      const listener = (kind: string) => (ev: MessageEvent) => {
        let data: Record<string, unknown>;
        try {
          data = JSON.parse(ev.data);
        } catch {
          return;
        }
        const event = frameToEvent(kind, data);
        if (event) dispatch(event);
        if (kind === "timeline" && data.version != null) {
          timelineCbRef.current?.(Number(data.version), String(data.note ?? ""));
        }
        if (kind === "done") {
          es.close();
          closedRef.current = true;
          dispatch({ type: "disconnected" });
        }
      };

      for (const kind of ["job", "step", "log", "timeline", "error", "done"]) {
        es.addEventListener(kind, listener(kind) as EventListener);
      }

      es.onerror = () => {
        es.close();
        dispatch({ type: "disconnected" });
        if (closedRef.current) return;
        retriesRef.current += 1;
        if (retriesRef.current > MAX_RETRIES) return; // stay dormant; Run re-opens
        const delay = RECONNECT_BASE_MS * 2 ** (retriesRef.current - 1);
        timerRef.current = setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      closedRef.current = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      esRef.current?.close();
      esRef.current = null;
    };
  }, [projectId, jobId, epoch, resync]);

  return { state, resync, reconnect };
}

/** Map an SSE frame to a reducer event; unknown shapes are ignored loudly-ish. */
export function frameToEvent(
  kind: string,
  data: Record<string, unknown>,
): LiveEvent | null {
  switch (kind) {
    case "job":
      return {
        type: "job",
        stage: String(data.stage ?? ""),
        progress_pct: Number(data.progress_pct ?? 0),
        error: (data.error as string | null) ?? null,
      };
    case "step":
      return {
        type: "step",
        seq_step: Number(data.seq_step ?? 0),
        key: String(data.key ?? ""),
        label: String(data.label ?? ""),
        status: (data.status as "running" | "done" | "failed" | "skipped") ?? "running",
        pct: Number(data.pct ?? 0),
        detail: String(data.detail ?? ""),
        order: Number(data.order ?? 0),
        duration_ms: data.duration_ms != null ? Number(data.duration_ms) : undefined,
      };
    case "log":
      return {
        type: "log",
        level: (data.level as "info" | "warn" | "error") ?? "info",
        message: String(data.message ?? ""),
        seq: data.seq != null ? Number(data.seq) : undefined,
      };
    case "timeline":
      return {
        type: "timeline",
        version: Number(data.version ?? 0),
        note: String(data.note ?? ""),
      };
    case "error":
      return {
        type: "job",
        stage: String(data.stage ?? ""),
        progress_pct: Number(data.progress_pct ?? 0),
        error: String(data.message ?? "pipeline error"),
      };
    case "done":
      return {
        type: "done",
        stage: data.stage != null ? String(data.stage) : undefined,
        progress_pct: data.progress_pct != null ? Number(data.progress_pct) : undefined,
        error: (data.error as string | null) ?? null,
      };
    default:
      return null;
  }
}
