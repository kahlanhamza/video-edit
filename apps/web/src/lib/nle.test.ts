// Pure NLE timeline math — lane routing, timecodes, adaptive ticks, zoom.

import { describe, expect, it } from "vitest";
import { NleCaption, NleClip, NleTimeline } from "./api";
import {
  buildLanes,
  captionAt,
  clamp,
  clampZoom,
  clipAfter,
  clipAt,
  formatTimecode,
  formatTimecodeFrames,
  formatTimecodeShort,
  laneForClip,
  programClips,
  rulerTicks,
  scrubBackground,
  scrubFrameCount,
  scrubFrameIndex,
  snapTime,
  sourceTimeAt,
  brollSourceAt,
  tickStep,
  timelineWidth,
} from "./nle";

function clip(partial: Partial<NleClip>): NleClip {
  return {
    track: "V1_master",
    kind: "raw",
    name: "clip",
    start_s: 0,
    dur_s: 2,
    end_s: 2,
    master_index: 0,
    act: null,
    script: null,
    template: null,
    audio_mode: null,
    source_start_s: 0,
    ...partial,
  };
}

describe("laneForClip", () => {
  it("routes narration and audio tracks to the audio lane", () => {
    expect(laneForClip(clip({ kind: "narration" }))).toBe("narration");
    expect(laneForClip(clip({ track: "A1_narration" }))).toBe("narration");
  });

  it("routes overlay cards to the cards lane", () => {
    expect(laneForClip(clip({ track: "V2_overlay_cards", kind: "card" }))).toBe("card");
  });

  it("keeps master footage in the video lane", () => {
    expect(laneForClip(clip({}))).toBe("video");
  });
});

describe("buildLanes", () => {
  it("sorts each lane by start time and orders lanes video/cards/audio", () => {
    const lanes = buildLanes([
      clip({ name: "late", start_s: 8 }),
      clip({ name: "early", start_s: 2 }),
      clip({ name: "vo", start_s: 3, kind: "narration", track: "A1_narration" }),
      clip({ name: "card", start_s: 0, kind: "card", track: "V2_overlay_cards" }),
    ]);
    expect(lanes.map((l) => l.id)).toEqual(["video", "card", "narration"]);
    expect(lanes[0].clips.map((c) => c.name)).toEqual(["early", "late"]);
    expect(lanes[0].label).toBe("Video");
    expect(lanes[1].label).toBe("Cards");
    expect(lanes[2].label).toBe("Audio");
  });

  it("returns no lanes for an empty cut", () => {
    expect(buildLanes([])).toEqual([]);
  });
});

describe("formatTimecode", () => {
  it("formats mm:ss.tenths", () => {
    expect(formatTimecode(0)).toBe("00:00.0");
    expect(formatTimecode(65.42)).toBe("01:05.4");
    expect(formatTimecode(600)).toBe("10:00.0");
  });

  it("clamps negative input to zero", () => {
    expect(formatTimecode(-3)).toBe("00:00.0");
  });
});

describe("tickStep", () => {
  it("widens the tick interval as the zoom shrinks", () => {
    const tight = tickStep(200); // px/s high — sub-second ticks fine
    const wide = tickStep(4); // px/s low — minute ticks required
    expect(tight).toBeLessThan(wide);
    expect(tickStep(4) * 4).toBeGreaterThanOrEqual(70);
    expect(tickStep(100)).toBe(1);
  });
});

describe("zoom + geometry", () => {
  it("clamps zoom into the usable band", () => {
    expect(clampZoom(0.5)).toBeGreaterThanOrEqual(4);
    expect(clampZoom(10000)).toBeLessThanOrEqual(240);
    expect(clampZoom(30)).toBe(30);
  });

  it("computes content width and clamps", () => {
    expect(timelineWidth(30, 24)).toBe(720);
    expect(clamp(15, 0, 10)).toBe(10);
  });
});

// --- pro timeline additions ---------------------------------------------------

describe("formatTimecodeFrames", () => {
  it("renders HH:MM:SS:FF at 30fps", () => {
    expect(formatTimecodeFrames(0)).toBe("00:00:00:00");
    expect(formatTimecodeFrames(65.5, 30)).toBe("00:01:05:15");
    expect(formatTimecodeFrames(3661.25, 30)).toBe("01:01:01:07");
  });
  it("clamps negatives to zero", () => {
    expect(formatTimecodeFrames(-3)).toBe("00:00:00:00");
  });
});

describe("formatTimecodeShort", () => {
  it("renders mm:ss", () => {
    expect(formatTimecodeShort(0)).toBe("00:00");
    expect(formatTimecodeShort(125)).toBe("02:05");
  });
});

describe("rulerTicks", () => {
  it("marks majors on the step grid and minors between", () => {
    const ticks = rulerTicks(0, 10, 100); // 100 px/s -> step 1s? no: >=70px -> 1s
    const majors = ticks.filter((t) => t.major);
    expect(majors[0].t).toBe(0);
    expect(majors[1].t).toBe(1);
    const firstMinor = ticks.find((t) => !t.major);
    expect(firstMinor?.t).toBeCloseTo(0.25, 5);
  });
  it("covers the visible span only", () => {
    const ticks = rulerTicks(50, 60, 10); // 10 px/s -> step 10s
    expect(ticks.length).toBeGreaterThan(0);
    expect(ticks[0].t).toBeGreaterThanOrEqual(50 - 1e-6);
    expect(ticks[ticks.length - 1].t).toBeLessThanOrEqual(60 + 10 + 1e-6);
  });
});

describe("snapTime", () => {
  it("snaps within tolerance", () => {
    expect(snapTime(2.4, 0.5, 0.15)).toBe(2.5);
  });
  it("keeps raw time outside tolerance", () => {
    expect(snapTime(2.2, 0.5, 0.1)).toBe(2.2);
  });
});

describe("scrubsheet math", () => {
  it("counts frames at ~1 per 2s, capped at 40", () => {
    expect(scrubFrameCount(10)).toBe(5);
    expect(scrubFrameCount(999)).toBe(40);
    expect(scrubFrameCount(0)).toBe(1);
  });
  it("maps source time to a frame index", () => {
    expect(scrubFrameIndex(0, 10)).toBe(0);
    expect(scrubFrameIndex(4, 10)).toBe(2);
    expect(scrubFrameIndex(99, 10)).toBe(4); // clamped to n-1
  });
  it("background style shows one sprite frame per cell", () => {
    const bg = scrubBackground(0, 5, "/scrub.jpg");
    expect(bg.backgroundSize).toBe("500% 100%");
    expect(bg.backgroundPosition).toBe("0% 0%");
    const last = scrubBackground(4, 5, "/scrub.jpg");
    expect(last.backgroundPosition).toBe("100% 0%");
    const single = scrubBackground(0, 1, "/s.jpg");
    expect(single.backgroundPosition).toBe("0% 0%");
  });
});

// --- program-monitor engine (timeline-true preview) ---------------------------

function nleWith(clips: NleClip[], captions: NleCaption[] = []): NleTimeline {
  return {
    project_id: "p",
    version: 1,
    duration_s: clips.reduce((a, c) => Math.max(a, c.end_s), 0),
    asset: null,
    clips,
    captions,
    effects: [],
  };
}

describe("programClips", () => {
  it("keeps only master clips with a source window, in timeline order", () => {
    const raw = clip({ name: "raw_0", start_s: 0, dur_s: 5, end_s: 5, source_start_s: 100 });
    const narr = clip({
      kind: "narration",
      track: "V1_master",
      name: "narr_1",
      start_s: 5,
      dur_s: 10,
      end_s: 15,
      source_start_s: 200,
    });
    const card = clip({ track: "V2_overlay_cards", kind: "card", name: "card", source_start_s: null });
    const noSrc = clip({ name: "mystery", source_start_s: null });
    const out = programClips(nleWith([noSrc, narr, card, raw]));
    expect(out.map((c) => c.name)).toEqual(["raw_0", "narr_1"]);
  });
  it("null nle yields empty", () => {
    expect(programClips(null)).toEqual([]);
  });
});

describe("clipAt / clipAfter / sourceTimeAt", () => {
  const clips = [
    clip({ name: "a", start_s: 0, dur_s: 10, end_s: 10, source_start_s: 100 }),
    clip({ name: "b", start_s: 20, dur_s: 5, end_s: 25, source_start_s: 0 }),
  ];
  it("finds the clip containing t, null in gaps", () => {
    expect(clipAt(clips, 0)?.name).toBe("a");
    expect(clipAt(clips, 9.9)?.name).toBe("a");
    expect(clipAt(clips, 10)).toBeNull();
    expect(clipAt(clips, 22)?.name).toBe("b");
    expect(clipAt(clips, 25)).toBeNull();
  });
  it("finds the next clip at/after t", () => {
    expect(clipAfter(clips, 11)?.name).toBe("b");
    expect(clipAfter(clips, 25)).toBeNull();
    expect(clipAfter(clips, 0)?.name).toBe("a");
  });
  it("maps program time into the source window, clamped at the edges", () => {
    expect(sourceTimeAt(clips[0], 0)).toBe(100);
    expect(sourceTimeAt(clips[0], 5)).toBe(105);
    expect(sourceTimeAt(clips[0], 10)).toBe(110); // clamped end
    expect(sourceTimeAt(clips[0], 20)).toBeNull(); // outside
  });
});

describe("brollSourceAt", () => {
  it("single-window narration rides the clip's own source window", () => {
    const narr = clip({
      kind: "narration",
      start_s: 0,
      dur_s: 10,
      end_s: 10,
      source_start_s: 50,
      broll_windows: [[50, 60]],
    });
    expect(brollSourceAt(narr, 0)).toBe(50);
    expect(brollSourceAt(narr, 9.5)).toBe(59.5);
  });
  it("cycles multi-window b-roll exactly like the renderer", () => {
    // window0 (input0) = clip's own range [0,10]; windows[1..] = meta windows
    const narr = clip({
      kind: "narration",
      start_s: 0,
      dur_s: 30,
      end_s: 30,
      source_start_s: 0,
      broll_windows: [
        [0, 10],
        [100, 110],
        [200, 210],
      ],
    });
    // tau 0..10 -> window0, 10..20 -> [100..110), 20..30 -> [200..210)
    expect(brollSourceAt(narr, 0)).toBe(0);
    expect(brollSourceAt(narr, 9.9)).toBeCloseTo(9.9, 5);
    expect(brollSourceAt(narr, 10)).toBe(100);
    expect(brollSourceAt(narr, 19.5)).toBeCloseTo(109.5, 5);
    expect(brollSourceAt(narr, 20)).toBe(200);
    expect(brollSourceAt(narr, 29)).toBeCloseTo(209, 5);
  });
  it("returns null outside the clip", () => {
    const narr = clip({ kind: "narration", start_s: 5, dur_s: 10, end_s: 15, source_start_s: 0 });
    expect(brollSourceAt(narr, 4.9)).toBeNull();
    expect(brollSourceAt(narr, 15.1)).toBeNull();
  });
});

describe("captionAt", () => {
  const cues: NleCaption[] = [
    { index: 0, start_ms: 0, end_ms: 2000, text: "hello", speaker: "Officer" },
    { index: 1, start_ms: 3000, end_ms: 5000, text: "world", speaker: "" },
  ];
  it("finds the on-screen cue and null between cues", () => {
    expect(captionAt(cues, 1)?.text).toBe("hello");
    expect(captionAt(cues, 2.5)).toBeNull();
    expect(captionAt(cues, 4)?.text).toBe("world");
    expect(captionAt(cues, 9)).toBeNull();
  });
});
