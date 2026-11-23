// useOpsStream law (op log v1): every op event must eventually fire exactly
// ONE debounced callback per animation-frame burst — a burst of N agent ops
// triggers ONE refetch of the folded program, not N. Unmount closes the
// stream; the durable table replays on reconnect so nothing is missed.

import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useOpsStream } from "./useOpsStream";

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  url: string;
  closed = false;
  listeners = new Map<string, (() => void)[]>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(kind: string, fn: () => void) {
    this.listeners.set(kind, [...(this.listeners.get(kind) ?? []), fn]);
  }

  close() {
    this.closed = true;
  }

  emit(kind: string) {
    for (const fn of this.listeners.get(kind) ?? []) fn();
  }
}

const opsStreamUrlMock = vi.fn(
  (pid: string) => `/ops-stream/${pid}`,
);

vi.mock("@/lib/api", () => ({
  api: { opsStreamUrl: (...a: unknown[]) => opsStreamUrlMock(...a) },
}));

function lastSource(): FakeEventSource {
  return FakeEventSource.instances[FakeEventSource.instances.length - 1];
}

const rafQueue: FrameRequestCallback[] = [];
const raf = vi.fn((cb: FrameRequestCallback) => {
  rafQueue.push(cb);
  return rafQueue.length;
});

function flushRaf() {
  const pending = rafQueue.splice(0);
  for (const cb of pending) cb(0);
}

describe("useOpsStream", () => {
  beforeEach(() => {
    FakeEventSource.instances = [];
    rafQueue.length = 0;
    vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);
    vi.stubGlobal("requestAnimationFrame", raf);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("subscribes to the project op stream", () => {
    renderHook(() => useOpsStream("p1", () => {}));
    expect(lastSource().url).toBe("/ops-stream/p1");
  });

  it("coalesces an op burst into ONE callback per frame", () => {
    const onOps = vi.fn();
    renderHook(() => useOpsStream("p1", onOps));
    const es = lastSource();
    act(() => {
      es.emit("op");
      es.emit("op");
      es.emit("op");
      flushRaf();
    });
    expect(onOps).toHaveBeenCalledTimes(1);
    act(() => {
      es.emit("op");
      flushRaf();
    });
    expect(onOps).toHaveBeenCalledTimes(2);
  });

  it("unmount closes the stream and disables the subscription", () => {
    const { unmount } = renderHook(() => useOpsStream("p1", () => {}));
    const es = lastSource();
    unmount();
    expect(es.closed).toBe(true);
  });

  it("does not subscribe without a project id", () => {
    renderHook(() => useOpsStream(null, () => {}));
    expect(FakeEventSource.instances).toHaveLength(0);
  });
});
