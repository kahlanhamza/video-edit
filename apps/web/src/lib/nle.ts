// Pure NLE timeline math — lane routing, timecodes, adaptive ruler ticks.
// No DOM, no React: components/studio/Timeline.tsx renders what these
// functions derive, so zoom/geometry semantics stay unit-tested.

import { NleClip } from "./api";

export interface NleLane {
  id: string;
  label: string;
  tone: "video" | "narration" | "card" | "other";
  clips: NleClip[];
}

/** Which lane a track item belongs to, by its OTIO track name + clip kind. */
export function laneForClip(clip: NleClip): NleLane["tone"] {
  if (clip.track.startsWith("A1") || clip.kind === "narration") return "narration";
  if (clip.track.startsWith("V2") || clip.kind === "card") return "card";
  return "video";
}

const LANE_LABELS: Record<NleLane["tone"], string> = {
  video: "Video",
  narration: "Audio",
  card: "Cards",
  other: "Other",
};

const LANE_ORDER: NleLane["tone"][] = ["video", "card", "narration", "other"];

/** Group every timeline clip into stable render lanes (video / cards / audio). */
export function buildLanes(clips: NleClip[]): NleLane[] {
  const byTone = new Map<NleLane["tone"], NleLane>();
  for (const clip of clips) {
    const tone = laneForClip(clip);
    let lane = byTone.get(tone);
    if (!lane) {
      lane = { id: tone, label: LANE_LABELS[tone], tone, clips: [] };
      byTone.set(tone, lane);
    }
    lane.clips.push(clip);
  }
  const lanes = LANE_ORDER.filter((t) => byTone.has(t)).map((t) => byTone.get(t)!);
  for (const lane of lanes) lane.clips.sort((a, b) => a.start_s - b.start_s);
  return lanes;
}

/** mm:ss.t — the NLE ruler format (sub-second precision at the end). */
export function formatTimecode(t: number): string {
  const safe = Math.max(0, t);
  const m = Math.floor(safe / 60);
  const s = Math.floor(safe % 60);
  const tenths = Math.floor((safe * 10) % 10);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${tenths}`;
}

const NICE_STEPS = [
  0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600,
];

/** Adaptive ruler spacing: a nice interval that keeps ticks >= ~70px apart. */
export function tickStep(pxPerSecond: number): number {
  for (const step of NICE_STEPS) {
    if (step * pxPerSecond >= 70) return step;
  }
  return NICE_STEPS[NICE_STEPS.length - 1];
}

export function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

/** Effective px/s with the zoom clamp applied (4 px/s .. 240 px/s). */
export const MIN_PX_PER_S = 4;
export const MAX_PX_PER_S = 240;

export function clampZoom(pxPerSecond: number): number {
  return clamp(pxPerSecond, MIN_PX_PER_S, MAX_PX_PER_S);
}

/** Content width in px for the given duration + zoom. */
export function timelineWidth(durationS: number, pxPerSecond: number): number {
  return Math.max(1, Math.ceil(durationS * pxPerSecond));
}
