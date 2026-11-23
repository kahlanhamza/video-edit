// effectWindowFrames — the FX-track edge-drag math (seconds → frame window,
// clamped, never inverted). The drag lands an adjust op with this window.
import { describe, expect, it } from "vitest";
import { effectWindowFrames } from "./nle";

// 30 fps, 100 s tape → last frame index 2999
const FPS = 30;
const DUR = 100;

describe("effectWindowFrames", () => {
  it("converts seconds to raw frames", () => {
    const out = effectWindowFrames("end", 10, FPS, DUR, { start: 0, end: null });
    expect(out).toEqual({ frame_start: 0, frame_end: 300 });
  });

  it("pins the other edge when the op had no window (whole-tape box)", () => {
    const out = effectWindowFrames("start", 5, FPS, DUR, { start: null, end: null });
    expect(out).toEqual({ frame_start: 150, frame_end: 2999 });
    const end = effectWindowFrames("end", 5, FPS, DUR, { start: null, end: null });
    expect(end).toEqual({ frame_start: 0, frame_end: 150 });
  });

  it("clamps inside the tape", () => {
    const out = effectWindowFrames("end", 500, FPS, DUR, { start: 100, end: null });
    expect(out!.frame_end).toBe(2999);
  });

  it("refuses an inverted or zero-length window", () => {
    expect(
      effectWindowFrames("start", 20, FPS, DUR, { start: null, end: 300 }),
    ).toBeNull(); // start frame 600 > end 300
    expect(
      effectWindowFrames("end", 10, FPS, DUR, { start: 300, end: null }),
    ).toBeNull(); // end 300 <= start 300
  });

  it("refuses a degenerate tape", () => {
    expect(effectWindowFrames("end", 1, 0, DUR, { start: 0, end: null })).toBeNull();
    expect(effectWindowFrames("end", 1, FPS, 0, { start: 0, end: null })).toBeNull();
  });
});
