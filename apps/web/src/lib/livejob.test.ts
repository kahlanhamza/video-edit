// Pure reducer tests for the live console feed — no DOM, no EventSource.

import { describe, expect, it } from "vitest";
import {
  currentStep,
  initialLiveJobState,
  liveJobReducer,
  runSummary,
  LiveEvent,
} from "./livejob";

function fold(events: LiveEvent[]) {
  return events.reduce(liveJobReducer, initialLiveJobState);
}

describe("liveJobReducer", () => {
  it("bootstraps from a snapshot and merges live deltas", () => {
    let s = fold([{ type: "snapshot", steps: [], logs: [] }]);
    s = fold([
      { type: "connected" },
      {
        type: "step",
        seq_step: 1,
        key: "ingest",
        label: "Ingest source",
        status: "running",
        pct: 0,
        detail: "",
        order: 1,
      },
      {
        type: "step",
        seq_step: 1,
        key: "ingest",
        label: "Ingest source",
        status: "running",
        pct: 55,
        detail: "probing",
        order: 1,
      },
      {
        type: "step",
        seq_step: 1,
        key: "ingest",
        label: "Ingest source",
        status: "done",
        pct: 100,
        detail: "",
        order: 1,
        duration_ms: 1200,
      },
      { type: "log", level: "info", message: "proxy ready", seq: 7 },
      { type: "job", stage: "perception", progress_pct: 22, error: null },
    ]);
    expect(s.steps).toHaveLength(1); // upserted, not duplicated
    expect(s.steps[0].status).toBe("done");
    expect(s.steps[0].pct).toBe(100);
    expect(s.logs).toEqual([{ id: 7, level: "info", message: "proxy ready" }]);
    expect(s.job?.stage).toBe("perception");
    expect(s.connected).toBe(true);
  });

  it("inserts steps in creation order regardless of arrival order", () => {
    const s = fold([
      { type: "step", seq_step: 3, key: "qa", label: "QA", status: "running", pct: 0, detail: "", order: 3 },
      { type: "step", seq_step: 1, key: "ingest", label: "Ingest", status: "running", pct: 0, detail: "", order: 1 },
    ]);
    expect(s.steps.map((x) => x.id)).toEqual([1, 3]);
  });

  it("marks done on the terminal frame and disconnects", () => {
    const s = fold([
      { type: "connected" },
      { type: "job", stage: "human_gate", progress_pct: 100, error: null },
      { type: "done", stage: "human_gate", progress_pct: 100, error: null },
    ]);
    expect(s.done).toBe(true);
    expect(s.connected).toBe(false);
    expect(s.job?.stage).toBe("human_gate");
  });

  it("caps the log buffer at 800 lines", () => {
    const events: LiveEvent[] = [];
    for (let i = 0; i < 850; i++) {
      events.push({ type: "log", level: "info", message: `line ${i}`, seq: i + 1 });
    }
    const s = fold(events);
    expect(s.logs).toHaveLength(800);
    expect(s.logs[0].message).toBe("line 50");
    expect(s.logs.at(-1)?.message).toBe("line 849");
  });

  it("records timeline revisions", () => {
    const s = fold([{ type: "timeline", version: 4, note: "story map" }]);
    expect(s.timelineVersion).toBe(4);
    expect(s.timelineNote).toBe("story map");
  });

  it("treats an error job frame as non-terminal state, not done", () => {
    const s = fold([{ type: "job", stage: "qa", progress_pct: 90, error: "loudness" }]);
    expect(s.done).toBe(false);
    expect(s.job?.error).toBe("loudness");
  });
});

describe("console derivations", () => {
  it("currentStep prefers the running step", () => {
    const s = fold([
      { type: "step", seq_step: 1, key: "a", label: "A", status: "done", pct: 100, detail: "", order: 1 },
      { type: "step", seq_step: 2, key: "b", label: "B", status: "running", pct: 30, detail: "", order: 2 },
    ]);
    expect(currentStep(s.steps)?.key).toBe("b");
  });

  it("runSummary counts statuses", () => {
    const s = fold([
      { type: "step", seq_step: 1, key: "a", label: "A", status: "done", pct: 100, detail: "", order: 1 },
      { type: "step", seq_step: 2, key: "b", label: "B", status: "failed", pct: 40, detail: "x", order: 2 },
      { type: "step", seq_step: 3, key: "c", label: "C", status: "skipped", pct: 0, detail: "", order: 3 },
    ]);
    expect(runSummary(s)).toEqual({ done: 1, failed: 1, skipped: 1, running: 0 });
  });
});
