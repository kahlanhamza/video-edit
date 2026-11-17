// Hook wiring tests — keyboard -> reducer -> <video> transport.

import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import { useNLEShortcuts } from "./useNLEShortcuts";

/** jsdom has no media implementation — drive the hook with a fake element. */
function fakeVideo(initialTime = 3.5) {
  const el = {
    currentTime: initialTime,
    duration: 60,
    paused: true,
    playbackRate: 1,
    play: vi.fn(function (this: { paused: boolean }) {
      this.paused = false;
      return Promise.resolve();
    }),
    pause: vi.fn(function (this: { paused: boolean }) {
      this.paused = true;
    }),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  };
  return el as unknown as HTMLVideoElement;
}

function keyEvent(code: string, over: Partial<KeyboardEventInit> = {}) {
  return new KeyboardEvent("keydown", { code, bubbles: true, cancelable: true, ...over });
}

describe("useNLEShortcuts", () => {
  let video: HTMLVideoElement;

  beforeEach(() => {
    video = fakeVideo();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("Space toggles transport: play then pause", () => {
    const { result } = renderHook(() => useNLEShortcuts({ videoRef: { current: video } }));
    act(() => window.dispatchEvent(keyEvent("Space")));
    expect(result.current.state.playing).toBe(true);
    expect(video.play as Mock).toHaveBeenCalled(); // effect applied 1x forward
    act(() => window.dispatchEvent(keyEvent("Space")));
    expect(result.current.state.playing).toBe(false);
    expect(video.pause as Mock).toHaveBeenCalled();
  });

  it("L presses raise playbackRate 1x -> 2x -> 4x", () => {
    const { result } = renderHook(() => useNLEShortcuts({ videoRef: { current: video } }));
    act(() => {
      window.dispatchEvent(keyEvent("KeyL"));
      window.dispatchEvent(keyEvent("KeyL"));
      window.dispatchEvent(keyEvent("KeyL"));
    });
    expect(result.current.state.rate).toBe(4);
    expect(video.playbackRate).toBe(4);
  });

  it("K stops: paused and rate re-armed", () => {
    const { result } = renderHook(() => useNLEShortcuts({ videoRef: { current: video } }));
    act(() => window.dispatchEvent(keyEvent("KeyL"))); // effects flush: play() runs
    act(() => window.dispatchEvent(keyEvent("KeyK"))); // effects flush: pause() runs
    expect(result.current.state).toMatchObject({ playing: false, rate: 0 });
    expect(video.pause as Mock).toHaveBeenCalled();
    expect(video.playbackRate).toBe(1);
  });

  it("ArrowLeft steps back exactly one frame (1/fps)", () => {
    renderHook(() => useNLEShortcuts({ videoRef: { current: video }, fps: 30 }));
    act(() => window.dispatchEvent(keyEvent("ArrowLeft")));
    expect(video.currentTime).toBeCloseTo(3.5 - 1 / 30, 6);
  });

  it("ArrowRight steps forward and clamps to duration", () => {
    renderHook(() => useNLEShortcuts({ videoRef: { current: video }, fps: 30 }));
    act(() => window.dispatchEvent(keyEvent("ArrowRight")));
    expect(video.currentTime).toBeCloseTo(3.5 + 1 / 30, 6);
    video.currentTime = 60;
    act(() => window.dispatchEvent(keyEvent("ArrowRight")));
    expect(video.currentTime).toBe(60);
  });

  it("I / O mark in/out from the video clock and report via onInOut", () => {
    const onInOut = vi.fn();
    const { result } = renderHook(() =>
      useNLEShortcuts({ videoRef: { current: video }, onInOut }),
    );
    act(() => window.dispatchEvent(keyEvent("KeyI")));
    video.currentTime = 7.25; // editor seeks forward
    act(() => window.dispatchEvent(keyEvent("KeyO")));
    expect(result.current.state.inS).toBe(3.5);
    expect(result.current.state.outS).toBe(7.25);
    expect(onInOut).toHaveBeenLastCalledWith(3.5, 7.25);
  });

  it("S splits and fires onSplit; Ctrl+Z undoes the split", () => {
    const onSplit = vi.fn();
    const { result } = renderHook(() =>
      useNLEShortcuts({ videoRef: { current: video }, onSplit }),
    );
    act(() => window.dispatchEvent(keyEvent("KeyS")));
    expect(result.current.state.splits).toEqual([3.5]);
    expect(onSplit).toHaveBeenCalledWith(3.5);
    act(() => window.dispatchEvent(keyEvent("KeyZ", { ctrlKey: true })));
    expect(result.current.state.splits).toEqual([]);
  });

  it("ignores transport keys while typing in an input", () => {
    const input = document.createElement("input");
    document.body.appendChild(input);
    const { result } = renderHook(() => useNLEShortcuts({ videoRef: { current: video } }));
    act(() => input.dispatchEvent(keyEvent("Space")));
    expect(result.current.state.playing).toBe(false);
    act(() => input.dispatchEvent(keyEvent("KeyS")));
    expect(result.current.state.splits).toEqual([]);
    input.remove();
  });

  it("enabled=false detaches the key listener", () => {
    const { result, rerender } = renderHook(
      ({ enabled }: { enabled: boolean }) =>
        useNLEShortcuts({ videoRef: { current: video }, enabled }),
      { initialProps: { enabled: true } },
    );
    expect(result.current.active).toBe(true);
    rerender({ enabled: false });
    act(() => window.dispatchEvent(keyEvent("Space")));
    expect(result.current.state.playing).toBe(false);
    expect(result.current.active).toBe(false);
  });
});
