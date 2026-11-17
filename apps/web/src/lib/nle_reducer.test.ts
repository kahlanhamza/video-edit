// Pure reducer unit tests — issue #19 acceptance: "scoring unit tests …
// shortcuts hook unit-tested (pure reducer)".

import { describe, expect, it } from "vitest";
import { initialNLEState, nleReducer, type NLEState } from "./nle_reducer";

const s = (over: Partial<NLEState> = {}): NLEState => ({ ...initialNLEState, ...over });

describe("nleReducer transport", () => {
  it("Space toggles play/pause and re-arms rate 1", () => {
    let st = nleReducer(initialNLEState, { type: "togglePlay" });
    expect(st.playing).toBe(true);
    expect(st.rate).toBe(1);
    st = nleReducer(st, { type: "togglePlay" });
    expect(st.playing).toBe(false);
    expect(st.rate).toBe(0);
  });

  it("K stops hard and re-arms 1x", () => {
    let st = nleReducer(initialNLEState, { type: "shuttle", dir: 1 });
    st = nleReducer(st, { type: "shuttle", dir: 1 }); // 2x
    st = nleReducer(st, { type: "stop" });
    expect(st).toMatchObject({ playing: false, rate: 0 });
    st = nleReducer(st, { type: "togglePlay" });
    expect(st.rate).toBe(1);
  });

  it("L steps 1x -> 2x -> 4x and caps at 4x", () => {
    let st = nleReducer(initialNLEState, { type: "shuttle", dir: 1 });
    expect(st.rate).toBe(1);
    st = nleReducer(st, { type: "shuttle", dir: 1 });
    expect(st.rate).toBe(2);
    st = nleReducer(st, { type: "shuttle", dir: 1 });
    expect(st.rate).toBe(4);
    st = nleReducer(st, { type: "shuttle", dir: 1 });
    expect(st.rate).toBe(4);
  });

  it("J flips to reverse and restarts the speed rung at 1x", () => {
    let st = nleReducer(initialNLEState, { type: "shuttle", dir: 1 });
    st = nleReducer(st, { type: "shuttle", dir: 1 }); // 2x forward
    st = nleReducer(st, { type: "shuttle", dir: -1 });
    expect(st.rate).toBe(-1);
    st = nleReducer(st, { type: "shuttle", dir: -1 });
    expect(st.rate).toBe(-2);
  });

  it("J after a hard stop starts reverse at 1x (not from the stale forward rate)", () => {
    // regression: a phantom rung was born from the stopped rate
    let st = nleReducer(initialNLEState, { type: "shuttle", dir: 1 });
    st = nleReducer(st, { type: "stop" });
    st = nleReducer(st, { type: "shuttle", dir: -1 });
    expect(st.rate).toBe(-1);
  });
});

describe("nleReducer markers", () => {
  it("I sets in; an in after the out drags the out with it", () => {
    let st = s({ outS: 10 });
    st = nleReducer(st, { type: "markIn", t: 12 });
    expect(st.inS).toBe(12);
    expect(st.outS).toBeNull();
  });

  it("O sets out; an out before the in drags the in with it", () => {
    let st = s({ inS: 5 });
    st = nleReducer(st, { type: "markOut", t: 3 });
    expect(st.outS).toBe(3);
    expect(st.inS).toBeNull();
  });

  it("S adds ascending split markers and dedupes ms-quantized times", () => {
    let st = nleReducer(initialNLEState, { type: "split", t: 4.4999999 });
    st = nleReducer(st, { type: "split", t: 4.49999991 });
    st = nleReducer(st, { type: "split", t: 1.25 });
    expect(st.splits).toEqual([1.25, 4.5]);
  });

  it("Ctrl+Z undoes marker changes in LIFO order; empty stack is a no-op", () => {
    let st = nleReducer(initialNLEState, { type: "markIn", t: 2 });
    st = nleReducer(st, { type: "split", t: 3 });
    st = nleReducer(st, { type: "markOut", t: 8 });
    expect(st).toMatchObject({ inS: 2, outS: 8, splits: [3] });
    st = nleReducer(st, { type: "undo" });
    expect(st.outS).toBeNull();
    st = nleReducer(st, { type: "undo" });
    expect(st.splits).toEqual([]);
    st = nleReducer(st, { type: "undo" });
    expect(st.inS).toBeNull();
    st = nleReducer(st, { type: "undo" });
    expect(st.undoStack.length).toBe(0);
  });
});

describe("nleReducer native sync", () => {
  it("follows the element pausing itself", () => {
    let st = s({ playing: true, rate: 2 });
    st = nleReducer(st, { type: "syncPlaying", playing: false });
    expect(st).toMatchObject({ playing: false, rate: 0 });
  });

  it("follows the element starting to play, arming rate 1 from a stop", () => {
    let st = nleReducer(initialNLEState, { type: "syncPlaying", playing: true });
    expect(st).toMatchObject({ playing: true, rate: 1 });
  });
});
