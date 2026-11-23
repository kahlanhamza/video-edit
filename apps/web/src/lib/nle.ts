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

// --- additions for the pro timeline (pure, unit-tested) ---------------------

/** HH:MM:SS:FF — the professional ruler/timecode format (frames at fps). */
export function formatTimecodeFrames(t: number, fps = 30): string {
  const safe = Math.max(0, t);
  const frames = Math.floor((safe * fps) % fps);
  const total = Math.floor(safe);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}:${pad(frames)}`;
}

/** mm:ss — compact ruler label for low zoom levels. */
export function formatTimecodeShort(t: number): string {
  const safe = Math.max(0, t);
  const m = Math.floor(safe / 60);
  const s = Math.floor(safe % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export interface RulerTick {
  t: number;
  /** major ticks carry a label; minor are just marks */
  major: boolean;
}

/**
 * Full ruler tick model for the visible span [viewStartS, viewEndS]:
 * major ticks every `step` seconds with `divisions` minor marks between.
 * Mirrors Premiere/Resolve rulers (adaptive density, labels only on major).
 */
export function rulerTicks(
  viewStartS: number,
  viewEndS: number,
  pxPerSecond: number,
  divisions = 4,
): RulerTick[] {
  const step = tickStep(pxPerSecond);
  const out: RulerTick[] = [];
  const first = Math.floor(viewStartS / step) * step;
  for (let t = first; t <= viewEndS + step; t += step / divisions) {
    const snapped = Number(t.toFixed(4));
    if (snapped < viewStartS - 1e-6) continue;
    out.push({ t: snapped, major: Math.abs((snapped / step) % 1) < 1e-6 });
  }
  return out;
}

/** Snap a time to the nearest `grid` seconds within `toleranceS`, else raw. */
export function snapTime(t: number, grid: number, toleranceS: number): number {
  if (grid <= 0) return t;
  const nearest = Math.round(t / grid) * grid;
  return Math.abs(nearest - t) <= toleranceS ? Number(nearest.toFixed(4)) : t;
}

// --- scrubsheet math (assets/{id}/scrubsheet: 1 row, ~1 frame / 2 s) --------

export const SCRUB_INTERVAL_S = 2.0;
export const SCRUB_MAX_THUMBS = 40;

/** Frame count in the scrubsheet sprite for a source of `durationS`. */
export function scrubFrameCount(durationS: number): number {
  return Math.min(
    SCRUB_MAX_THUMBS,
    Math.max(1, Math.ceil(Math.max(durationS, 0.1) / SCRUB_INTERVAL_S)),
  );
}

/** Sprite frame index covering source-time `t` (clamped to the sprite). */
export function scrubFrameIndex(t: number, durationS: number): number {
  const n = scrubFrameCount(durationS);
  return clamp(Math.floor(t / SCRUB_INTERVAL_S), 0, n - 1);
}

/**
 * CSS background-position for showing sprite frame `i` of `n` in a cell
 * (classic one-row sprite slider: n*100% size, i/(n-1)*100% position).
 */
export function scrubBackground(i: number, n: number, url: string): {
  backgroundImage: string;
  backgroundSize: string;
  backgroundPosition: string;
} {
  const pos = n > 1 ? (i / (n - 1)) * 100 : 0;
  return {
    backgroundImage: `url(${url})`,
    backgroundSize: `${n * 100}% 100%`,
    backgroundPosition: `${pos}% 0%`,
  };
}

/**
 * The tracked box to draw at playhead time `tS` — the per-frame path of a
 * tracked blur selection (step 3). `null` when the playhead sits outside the
 * tracked span: the selection is NOT active there, so drawing anything would
 * lie about what the program blurs. The box follows the object as you scrub.
 */
export interface TrackedBox {
  x: number;
  y: number;
  w: number;
  h: number;
  interpolated: boolean;
  score: number;
}

export function trackedBoxAt(
  hits: Array<{ t_s: number; x: number; y: number; w: number; h: number; score: number; interpolated: boolean }>,
  tS: number,
): TrackedBox | null {
  if (hits.length === 0) return null;
  const first = hits[0];
  const last = hits[hits.length - 1];
  if (tS < first.t_s - 1 / 1000 || tS > last.t_s + 1 / 1000) return null;
  let best = first;
  let bestD = Math.abs(first.t_s - tS);
  for (const h of hits) {
    const d = Math.abs(h.t_s - tS);
    if (d < bestD) {
      best = h;
      bestD = d;
    }
  }
  return { x: best.x, y: best.y, w: best.w, h: best.h, interpolated: best.interpolated, score: best.score };
}
