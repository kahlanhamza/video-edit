"use client";

// Timeline — the professional multi-track NLE timeline.
// Lanes: Video (master cut, trimmable), Cards (overlays), Audio (narration),
// Captions (per-cue chips), Effects (redaction windows). The playhead rides
// the proxy player; the agent's changes arrive via live timeline revisions.

import { useCallback, useMemo, useRef, useState } from "react";
import { NleCaption, NleClip, NleEffect, NleTimeline } from "@/lib/api";
import {
  buildLanes,
  clamp,
  clampZoom,
  formatTimecode,
  tickStep,
  timelineWidth,
} from "@/lib/nle";

const LANE_H = 34;
const RULER_H = 26;

const LANE_TONE: Record<string, { bg: string; border: string; label: string }> = {
  video: { bg: "bg-sky-500/25", border: "border-sky-400/70", label: "text-sky-200" },
  narration: { bg: "bg-emerald-500/25", border: "border-emerald-400/70", label: "text-emerald-200" },
  card: { bg: "bg-violet-500/25", border: "border-violet-400/70", label: "text-violet-200" },
  other: { bg: "bg-slate-600/30", border: "border-slate-500/60", label: "text-slate-200" },
};

const KIND_BADGE: Record<string, string> = {
  raw: "RAW",
  narration: "VO",
  card: "CARD",
};

type DragMode =
  | { kind: "none" }
  | { kind: "seek" }
  | { kind: "trim-left"; clip: NleClip; startPx: number }
  | { kind: "trim-right"; clip: NleClip; startPx: number };

export interface TimelineProps {
  nle: NleTimeline | null;
  playheadS: number;
  onSeek: (t: number) => void;
  onTrimClip?: (itemIndex: number, newStartS: number, newDurS: number) => void;
  onSplitAtPlayhead?: (playheadS: number) => void;
  onDeleteEffect?: (effectId: string) => void;
}

export default function Timeline({
  nle,
  playheadS,
  onSeek,
  onTrimClip,
  onSplitAtPlayhead,
  onDeleteEffect,
}: TimelineProps) {
  const [pxPerS, setPxPerS] = useState(24);
  const [selected, setSelected] = useState<string | null>(null);
  const [drag, setDrag] = useState<DragMode>({ kind: "none" });
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const dragTimeRef = useRef(0); // pointer time during a trim drag
  const laneLabelW = 88;

  const duration = nle?.duration_s ?? 0;
  const width = timelineWidth(duration, pxPerS);
  const lanes = useMemo(() => (nle ? buildLanes(nle.clips) : []), [nle]);
  const captions = nle?.captions ?? [];
  const effects = nle?.effects ?? [];
  const zoom = clampZoom(pxPerS);
  const step = tickStep(zoom);

  const xToTime = useCallback(
    (clientX: number): number => {
      const el = scrollRef.current;
      if (!el) return 0;
      const rect = el.getBoundingClientRect();
      const x = clientX - rect.left + el.scrollLeft - laneLabelW;
      return clamp(x / zoom, 0, Math.max(duration, 0.001));
    },
    [zoom, duration, laneLabelW],
  );

  const selectedClip = useMemo(
    () => (nle?.clips ?? []).find((c) => `${c.track}:${c.name}:${c.start_s}` === selected) ?? null,
    [nle, selected],
  );

  const timeToPx = (t: number) => t * zoom;

  function onRulerDown(ev: React.MouseEvent) {
    onSeek(xToTime(ev.clientX));
    setDrag({ kind: "seek" });
  }

  function onMouseMove(ev: React.MouseEvent) {
    const t = xToTime(ev.clientX);
    dragTimeRef.current = t;
    if (drag.kind === "seek") onSeek(t);
  }

  function endDrag() {
    if (drag.kind === "trim-left" && drag.clip && onTrimClip) {
      const c = drag.clip;
      if (c.master_index != null && c.source_start_s != null) {
        // left edge moved by (pointerT - clipStart) timeline seconds; the
        // source in-point moves with it, duration shrinks by the same delta
        const newSourceStart = Math.max(0, c.source_start_s + (dragTimeRef.current - c.start_s));
        const newDur = c.dur_s + (c.source_start_s - newSourceStart);
        if (newDur > 0.1) {
          onTrimClip(
            c.master_index,
            Number(newSourceStart.toFixed(3)),
            Number(newDur.toFixed(3)),
          );
        }
      }
    }
    if (drag.kind === "trim-right" && drag.clip && onTrimClip) {
      const c = drag.clip;
      if (c.master_index != null && c.source_start_s != null) {
        const newDur = Math.max(0.1, dragTimeRef.current - c.start_s);
        onTrimClip(c.master_index, c.source_start_s, Number(newDur.toFixed(3)));
      }
    }
    setDrag({ kind: "none" });
  }

  const ticks: number[] = useMemo(() => {
    const out: number[] = [];
    for (let t = 0; t <= duration + step; t += step) out.push(Number(t.toFixed(3)));
    return out;
  }, [duration, step]);

  if (!nle) {
    return (
      <div className="flex h-44 items-center justify-center rounded-lg border border-slate-800 bg-slate-900/60 text-sm text-slate-500">
        no timeline yet — run the pipeline and the agent&apos;s cut appears here live
      </div>
    );
  }

  const canEdit = (c: NleClip) => c.track === "V1_master" && c.master_index != null && c.source_start_s != null && !!onTrimClip;

  return (
    <div className="select-none rounded-lg border border-slate-800 bg-slate-900/60">
      {/* toolbar: zoom + split + clip info */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 px-3 py-2">
        <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
          timeline v{nle.version ?? "—"}
        </span>
        <span className="text-xs tabular-nums text-slate-400">
          {formatTimecode(playheadS)} / {formatTimecode(duration)}
        </span>
        <div className="flex-1" />
        {selectedClip && (
          <span className="max-w-[42ch] truncate rounded bg-slate-800 px-2 py-0.5 text-[11px] text-slate-300" title={selectedClip.name}>
            {selectedClip.name} · {selectedClip.dur_s.toFixed(2)}s
            {selectedClip.script ? ` · “${selectedClip.script.slice(0, 40)}”` : ""}
          </span>
        )}
        <button
          className="btn-ghost !px-2 !py-0.5 text-xs"
          disabled={!onSplitAtPlayhead}
          title="split the master clip under the playhead"
          onClick={() => onSplitAtPlayhead?.(playheadS)}
        >
          split at playhead
        </button>
        <button className="btn-ghost !px-2 !py-0.5 text-xs" onClick={() => setPxPerS(clampZoom(zoom * 1.4))}>
          +
        </button>
        <button className="btn-ghost !px-2 !py-0.5 text-xs" onClick={() => setPxPerS(clampZoom(zoom / 1.4))}>
          −
        </button>
        <button
          className="btn-ghost !px-2 !py-0.5 text-xs"
          title="zoom to fit"
          onClick={() => {
            const el = scrollRef.current;
            if (el && duration > 0) setPxPerS(clampZoom((el.clientWidth - laneLabelW - 8) / duration));
          }}
        >
          fit
        </button>
      </div>

      {/* scrollable tracks */}
      <div
        ref={scrollRef}
        className="overflow-x-auto overflow-y-hidden"
        onMouseMove={onMouseMove}
        onMouseUp={endDrag}
        onMouseLeave={endDrag}
      >
        <div style={{ width: width + laneLabelW }} className="relative">
          {/* ruler */}
          <div
            className="relative cursor-crosshair border-b border-slate-800"
            style={{ height: RULER_H }}
            onMouseDown={onRulerDown}
          >
            <div style={{ width: laneLabelW }} className="pointer-events-none absolute left-0 top-0 h-full bg-slate-900/95" />
            <div style={{ marginLeft: laneLabelW }} className="relative h-full">
              {ticks.map((t) => (
                <div
                  key={t}
                  className="absolute top-0 h-full border-l border-slate-700/70 pl-1 text-[9px] tabular-nums text-slate-500"
                  style={{ left: timeToPx(t) }}
                >
                  {formatTimecode(t)}
                </div>
              ))}
            </div>
          </div>

          {/* clip lanes */}
          {lanes.map((lane) => {
            const tone = LANE_TONE[lane.tone] ?? LANE_TONE.other;
            return (
              <div key={lane.id} className="relative border-b border-slate-800/60" style={{ height: LANE_H }}>
                <div style={{ width: laneLabelW }} className="absolute left-0 top-0 z-10 flex h-full items-center bg-slate-900/95 px-2">
                  <span className={`text-[10px] font-semibold uppercase tracking-wide ${tone.label}`}>
                    {lane.label}
                  </span>
                </div>
                <div className="relative h-full" style={{ marginLeft: laneLabelW, width }}>
                  {lane.clips.map((c) => {
                    const key = `${c.track}:${c.name}:${c.start_s}`;
                    const isSel = key === selected;
                    const w = Math.max(3, c.dur_s * zoom);
                    return (
                      <div
                        key={key}
                        className={`absolute top-1 flex h-[26px] cursor-pointer items-center overflow-hidden rounded border px-1 ${tone.bg} ${isSel ? "ring-2 ring-amber-400" : tone.border}`}
                        style={{ left: timeToPx(c.start_s), width: w }}
                        title={`${c.name} · ${c.dur_s.toFixed(2)}s${c.act ? ` · ${c.act}` : ""}${c.script ? ` · ${c.script.slice(0, 60)}` : ""}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelected(isSel ? null : key);
                          if (lane.tone === "video") onSeek(c.start_s);
                        }}
                      >
                        <span className="mr-1 shrink-0 rounded bg-black/40 px-1 text-[8px] font-bold tracking-wide text-slate-200">
                          {KIND_BADGE[c.kind] ?? c.kind.slice(0, 4).toUpperCase()}
                        </span>
                        <span className="truncate text-[10px] text-slate-100">{c.name}</span>
                        {isSel && canEdit(c) && (
                          <>
                            <span
                              className="absolute left-0 top-0 h-full w-1.5 cursor-ew-resize bg-amber-400/80"
                              title="drag to trim in-point"
                              onMouseDown={(e) => {
                                e.stopPropagation();
                                setDrag({ kind: "trim-left", clip: c, startPx: c.start_s * zoom });
                              }}
                            />
                            <span
                              className="absolute right-0 top-0 h-full w-1.5 cursor-ew-resize bg-amber-400/80"
                              title="drag to trim out-point"
                              onMouseDown={(e) => {
                                e.stopPropagation();
                                setDrag({ kind: "trim-right", clip: c, startPx: c.start_s * zoom });
                              }}
                            />
                          </>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}

          {/* captions lane */}
          <div className="relative border-b border-slate-800/60" style={{ height: LANE_H }}>
            <div style={{ width: laneLabelW }} className="absolute left-0 top-0 z-10 flex h-full items-center bg-slate-900/95 px-2">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-amber-200">
                Captions
              </span>
            </div>
            <div className="relative h-full" style={{ marginLeft: laneLabelW, width }}>
              {captions.length === 0 && (
                <span className="absolute left-2 top-2 text-[10px] text-slate-600">no cues yet</span>
              )}
              {captions.map((cue: NleCaption) => {
                const start = cue.start_ms / 1000;
                const w = Math.max(2, ((cue.end_ms - cue.start_ms) / 1000) * zoom);
                return (
                  <div
                    key={cue.index}
                    className="absolute top-1 flex h-[26px] cursor-pointer items-center overflow-hidden rounded border border-amber-400/50 bg-amber-500/15 px-1 hover:bg-amber-500/25"
                    style={{ left: timeToPx(start), width: w }}
                    title={`${cue.speaker}: ${cue.text}`}
                    onClick={() => onSeek(start)}
                  >
                    <span className="truncate text-[10px] text-amber-100">{cue.text}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* effects lane (redaction windows) */}
          <div className="relative" style={{ height: LANE_H }}>
            <div style={{ width: laneLabelW }} className="absolute left-0 top-0 z-10 flex h-full items-center bg-slate-900/95 px-2">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-rose-300">
                Effects
              </span>
            </div>
            <div className="relative h-full" style={{ marginLeft: laneLabelW, width }}>
              {effects.length === 0 && (
                <span className="absolute left-2 top-2 text-[10px] text-slate-600">no blur/redaction windows</span>
              )}
              {effects.map((fx: NleEffect) => {
                const start = fx.t_start_s ?? 0;
                const end = fx.t_end_s ?? duration;
                const w = Math.max(4, (end - start) * zoom);
                return (
                  <div
                    key={fx.id}
                    className="group absolute top-1 flex h-[26px] items-center overflow-hidden rounded border border-rose-400/60 bg-rose-500/20 px-1"
                    style={{ left: timeToPx(start), width: w }}
                    title={`${fx.label || "blur"} · ${fx.mode}${fx.t_start_s != null ? ` · ${start.toFixed(1)}s–${end.toFixed(1)}s` : " · full length"} · box ${fx.w}x${fx.h}@${fx.x},${fx.y}`}
                    onClick={() => onSeek(start)}
                  >
                    <span className="truncate text-[10px] text-rose-100">
                      blur · {fx.label || fx.mode}
                    </span>
                    {onDeleteEffect && fx.created_by === "human_edit" && (
                      <button
                        className="ml-1 shrink-0 rounded px-1 text-[10px] text-rose-300 hover:bg-rose-500/40"
                        title="delete this studio-drawn box"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteEffect(fx.id);
                        }}
                      >
                        del
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* playhead — spans ruler + all lanes */}
          <div
            className="pointer-events-none absolute top-0 w-0.5 bg-amber-400"
            style={{ left: laneLabelW + timeToPx(clamp(playheadS, 0, duration)), height: RULER_H + lanes.length * LANE_H + 2 * LANE_H }}
          >
            <span className="absolute -left-1 top-0 block h-2 w-2.5 rounded-b bg-amber-400" />
          </div>
        </div>
      </div>
      {/* trim drags are tracked by onMouseMove on the scroll container above */}
      <p className="border-t border-slate-800 px-3 py-1.5 text-[10px] text-slate-500">
        click a clip to select · drag its amber edges to trim in/out (dispatches{" "}
        <span className="text-slate-400">timeline.trim_clip</span>) · split at playhead ·
        agent changes stream in live
      </p>
    </div>
  );
}
