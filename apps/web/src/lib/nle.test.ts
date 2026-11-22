// Pure NLE timeline math — lane routing, timecodes, adaptive ticks, zoom.

import { describe, expect, it } from "vitest";
import { NleClip } from "./api";
import {
  buildLanes,
  clamp,
  clampZoom,
  formatTimecode,
  laneForClip,
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
