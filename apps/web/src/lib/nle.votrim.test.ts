// voTrimWindow — the A1 audio edge-drag math: the trim cuts INSIDE the TTS
// stem (vo window), never moves a source window. Bounds: stem end, VO_MIN
// window; a no-op drag returns null so no pointless op dispatches.
import { describe, expect, it } from "vitest";
import { voTrimWindow } from "./nle";

const CLIP = {
  start_s: 10,
  dur_s: 8.35, // 8s TTS + 0.35 tail pad
  vo_in_s: 0,
  vo_out_s: null as number | null,
  tts_duration_s: 8.0,
};

describe("voTrimWindow", () => {
  it("skips head dead air (left edge drag right)", () => {
    const out = voTrimWindow("start", 10 + 1.2, CLIP);
    expect(out).toEqual({ vo_in_s: 1.2, vo_out_s: null });
  });

  it("cuts the tail (right edge drag left)", () => {
    const out = voTrimWindow("end", 10 + 5.5, CLIP);
    expect(out).toEqual({ vo_in_s: 0, vo_out_s: 5.5 });
  });

  it("respects an already-trimmed window on both edges", () => {
    const trimmed = { ...CLIP, vo_in_s: 1.0, vo_out_s: 6.0 };
    const left = voTrimWindow("start", 10 + 0.5, trimmed); // 0.5 into block -> in 1.5
    expect(left).toEqual({ vo_in_s: 1.5, vo_out_s: 6.0 });
    // block-local 4.0 = window length 4.0 (audio rides from 0) -> out 1.0 + 4.0
    const right = voTrimWindow("end", 10 + 4.0, trimmed);
    expect(right).toEqual({ vo_in_s: 1.0, vo_out_s: 5.0 });
  });

  it("clamps the in-point so the window never goes under VO_MIN", () => {
    const out = voTrimWindow("start", 10 + 7.9, CLIP); // 7.9 -> beyond 8 - 0.5
    expect(out!.vo_in_s).toBe(7.5);
  });

  it("clamps the out-point to the stem end", () => {
    const out = voTrimWindow("end", 10 + 99, CLIP); // drag far past the block
    expect(out).toBeNull(); // full window again — no-op, no dispatch
  });

  it("left drag far left is a no-op (in stays 0)", () => {
    expect(voTrimWindow("start", 9, CLIP)).toBeNull(); // before the block start
  });

  it("falls back to the block length when the TTS manifest has not landed", () => {
    const noStem = { ...CLIP, tts_duration_s: null };
    const out = voTrimWindow("end", 10 + 6.0, noStem);
    expect(out).toEqual({ vo_in_s: 0, vo_out_s: 6 });
    const left = voTrimWindow("start", 10 + 7.95, noStem); // bound = in + dur - 0.5 = 7.85
    expect(left!.vo_in_s).toBe(7.85);
  });

  it("rounds to milliseconds (no float dust in the dispatched op)", () => {
    const out = voTrimWindow("start", 10 + 1.2345678, CLIP);
    expect(out!.vo_in_s).toBe(1.235);
  });
});
