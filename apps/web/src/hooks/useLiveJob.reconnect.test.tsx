// useLiveJob reconnect law (owner 2026-11-23): Run reuses the SAME job id on
// resume, so after a failure closes the stream ("done" event), only an epoch
// bump can reopen it. reconnect() must construct a fresh EventSource even
// when jobId is unchanged — this is the "run again does nothing" fix.

import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useLiveJob } from "./useLiveJob";

/** Minimal EventSource fake: records instances, lets tests push frames. */
class FakeEventSource {
  static instances: FakeEventSource[] = [];
  url: string;
  closed = false;
  listeners = new Map<string, ((ev: { data: string }) => void)[]>();
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(kind: string, fn: (ev: { data: string }) => void) {
    this.listeners.set(kind, [...(this.listeners.get(kind) ?? []), fn]);
  }

  close() {
    this.closed = true;
  }

  emit(kind: string, data: unknown) {
    for (const fn of this.listeners.get(kind) ?? []) fn({ data: JSON.stringify(data) });
  }
}

const snapshotMock = vi.fn().mockResolvedValue({ steps: [], logs: [] });

vi.mock("@/lib/api", () => ({
  api: { jobSnapshot: (...a: unknown[]) => snapshotMock(...a) },
  liveSseUrl: (pid: string, jid: string) => `/sse/${pid}/${jid}`,
}));

function lastSource(): FakeEventSource {
  return FakeEventSource.instances[FakeEventSource.instances.length - 1];
}

describe("useLiveJob reconnect", () => {
  beforeEach(() => {
    FakeEventSource.instances = [];
    snapshotMock.mockClear();
    vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("opens one stream per job id and folds job frames", () => {
    const { result } = renderHook(() => useLiveJob("p1", "j1"));
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(lastSource().url).toBe("/sse/p1/j1");

    act(() => {
      lastSource().emit("job", { stage: "perception", progress_pct: 22, error: null });
    });
    expect(result.current.state.job?.stage).toBe("perception");
    expect(result.current.state.job?.error).toBeNull();
  });

  it("reopens the stream for the SAME job id after a failure closed it", () => {
    const { result } = renderHook(() => useLiveJob("p1", "j1"));

    // pipeline failure: error frame then terminal done -> stream latches closed
    act(() => {
      lastSource().emit("error", { stage: "perception", message: "gladia failed" });
      lastSource().emit("done", { stage: "perception", progress_pct: 22 });
    });
    expect(lastSource().closed).toBe(true);
    expect(result.current.state.done).toBe(true);
    expect(result.current.state.job?.error).toBe("gladia failed");

    // Run-again: same id, epoch bump -> a FRESH EventSource is constructed
    act(() => result.current.reconnect());
    expect(FakeEventSource.instances).toHaveLength(2);
    expect(lastSource().closed).toBe(false);
    expect(lastSource().url).toBe("/sse/p1/j1");

    // the reopened stream clears the terminal latch (resume is visible again)
    act(() => lastSource().onopen?.());
    expect(result.current.state.done).toBe(false);
    expect(result.current.state.connected).toBe(true);

    // and the API's resume event (error cleared) folds through
    act(() => {
      lastSource().emit("job", { stage: "perception", progress_pct: 22, error: null });
    });
    expect(result.current.state.job?.error).toBeNull();
  });
});
