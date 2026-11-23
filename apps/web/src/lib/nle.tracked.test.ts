import { describe, expect, it } from "vitest";
import { trackedBoxAt } from "./nle";

const hits = [
  { t_s: 1.0, x: 10, y: 20, w: 30, h: 40, score: 0.9, interpolated: false },
  { t_s: 2.0, x: 20, y: 30, w: 30, h: 40, score: 0.8, interpolated: true },
  { t_s: 3.0, x: 40, y: 50, w: 30, h: 40, score: 0.7, interpolated: false },
];

describe("trackedBoxAt", () => {
  it("returns the nearest hit while inside the tracked span", () => {
    expect(trackedBoxAt(hits, 1.9)).toEqual({
      x: 20,
      y: 30,
      w: 30,
      h: 40,
      interpolated: true,
      score: 0.8,
    });
    expect(trackedBoxAt(hits, 3.0)?.x).toBe(40);
    expect(trackedBoxAt(hits, 1.0)?.x).toBe(10);
  });

  it("returns null outside the span — the selection is not active there", () => {
    expect(trackedBoxAt(hits, 0.5)).toBeNull();
    expect(trackedBoxAt(hits, 3.5)).toBeNull();
  });

  it("returns null for an empty path", () => {
    expect(trackedBoxAt([], 1.0)).toBeNull();
  });

  it("is frame-boundary tolerant", () => {
    expect(trackedBoxAt(hits, 0.9999)?.x).toBe(10);
    expect(trackedBoxAt(hits, 3.0001)?.x).toBe(40);
  });
});
