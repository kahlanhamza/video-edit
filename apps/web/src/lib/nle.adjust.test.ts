// adjustedBox — the pure math behind manual box adjust (drag move / corner
// resize). The component renders this preview live and lands it as the
// adjust op on release; the clamping laws are the real contract.
import { describe, expect, it } from "vitest";
import { MIN_ADJUST_PX, adjustedBox, type AdjustDrag } from "./nle";

const MOVE: AdjustDrag = {
  opId: "op-1",
  mode: "move",
  origRaw: { x: 100, y: 100, w: 200, h: 80 },
  startDisplay: { x: 50, y: 50 },
};
const RESIZE: AdjustDrag = { ...MOVE, mode: "resize" };

// display -> raw scale: 2 raw px per display px, tape 1920x1080
const SCALE = { x: 2, y: 2 };

describe("adjustedBox — move", () => {
  it("translates the raw rect by the scaled pointer delta", () => {
    const out = adjustedBox(MOVE, { x: 60, y: 40 }, SCALE, 1920, 1080);
    // dx = (60-50)*2 = +20 raw, dy = (40-50)*2 = -20 raw
    expect(out).toEqual({ x: 120, y: 80, w: 200, h: 80 });
  });

  it("clamps inside the tape frame — never negative, never past the edge", () => {
    const out = adjustedBox(MOVE, { x: 0, y: 0 }, SCALE, 1920, 1080);
    expect(out.x).toBe(0); // 100 - 100 raw → clamped at the left edge
    expect(out.y).toBe(0);
    const right = adjustedBox(MOVE, { x: 5000, y: 5000 }, SCALE, 1920, 1080);
    expect(right.x).toBe(1920 - 200);
    expect(right.y).toBe(1080 - 80);
  });

  it("keeps the size untouched while moving", () => {
    const out = adjustedBox(MOVE, { x: 75, y: 65 }, SCALE, 1920, 1080);
    expect(out.w).toBe(200);
    expect(out.h).toBe(80);
  });
});

describe("adjustedBox — resize", () => {
  it("grows the bottom-right corner, keeps the top-left pinned", () => {
    const out = adjustedBox(RESIZE, { x: 70, y: 40 }, SCALE, 1920, 1080);
    expect(out.x).toBe(100);
    expect(out.y).toBe(100);
    expect(out.w).toBe(200 + 40); // (70-50)*2 = 40 raw growth
    expect(out.h).toBe(80 - 20); // (40-50)*2 = -20 raw shrink
  });

  it("never shrinks below the minimum coverage box", () => {
    // cursor pulled far past the top-left: dx = (-46-50)*2 = -192 raw
    const out = adjustedBox(RESIZE, { x: -46, y: -46 }, SCALE, 1920, 1080);
    expect(out.w).toBe(MIN_ADJUST_PX);
    expect(out.h).toBe(MIN_ADJUST_PX);
  });

  it("clamps growth at the tape edges", () => {
    const far: AdjustDrag = {
      ...RESIZE,
      origRaw: { x: 1900, y: 1060, w: 16, h: 16 },
    };
    const out = adjustedBox(far, { x: 5000, y: 5000 }, SCALE, 1920, 1080);
    expect(out.x + out.w).toBeLessThanOrEqual(1920);
    expect(out.y + out.h).toBeLessThanOrEqual(1080);
    expect(out.w).toBeGreaterThanOrEqual(MIN_ADJUST_PX);
    expect(out.h).toBeGreaterThanOrEqual(MIN_ADJUST_PX);
  });
});

describe("adjustedBox — tiny tape guard", () => {
  it("stays sane on a degenerate (sub-minimum) tape — the tape edge wins", () => {
    const out = adjustedBox(MOVE, { x: 60, y: 40 }, SCALE, 4, 4);
    expect(out.x).toBeGreaterThanOrEqual(0);
    expect(out.y).toBeGreaterThanOrEqual(0);
    expect(out.x + out.w).toBeLessThanOrEqual(4);
    expect(out.y + out.h).toBeLessThanOrEqual(4);
  });
});
