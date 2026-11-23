"use client";

// Timeline — the professional NLE timeline (design audit: Premiere/Resolve).
//
// Anatomy, top to bottom:
//   toolbar  — version chip · split · snap magnet · zoom slider + fit
//   ruler    — adaptive major/minor ticks, HH:MM:SS:FF labels, seek + drag
//   tracks   — V2 Cards · V1 Master (filmstrip from the scrubsheet sprite)
//              A1 Audio (VO blocks) · CC Captions · FX Effects
// Every track has a real header (name chip, eye/lock/mute) like Premiere.
// The playhead spans ruler + tracks with a flag handle in the ruler.

import { useCallback, useMemo, useRef, useState } from "react";
import { assetScrubUrl } from "@/lib/api";
import {
  buildLanes,
  clamp,
  clampZoom,
  formatTimecodeFrames,
  formatTimecodeShort,
  rulerTicks,
  scrubBackground,
  scrubFrameCount,
  scrubFrameIndex,
  snapTime,
  timelineWidth,
} from "@/lib/nle";
import type { NleCaption, NleClip, NleEffect, NleTimeline } from "@/lib/api";
import { Icon } from "./icons";

const TRACK_HEAD_W = 148;
const RULER_H = 26;
const H_VIDEO = 56;
const H_CARD = 38;
const H_AUDIO = 32;
const H_CC = 24;
const H_FX = 24;

type DragMode =
  | { kind: "none" }
  | { kind: "seek" }
  | { kind: "trim-left"; clip: NleClip }
  | { kind: "trim-right"; clip: NleClip };

export interface TimelineProps {
  nle: NleTimeline | null;
  playheadS: number;
  onSeek: (t: number) => void;
  onTrimClip?: (itemIndex: number, newStartS: number, newDurS: number) => void;
  onSplitAtPlayhead?: (playheadS: number) => void;
  onDeleteEffect?: (effectId: string) => void;
}

interface TrackDef {
  id: string;
  name: string;
  kind: "video" | "card" | "audio" | "cc" | "fx";
  height: number;
  canHide: boolean;
  canMute: boolean;
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
  const [snap, setSnap] = useState(true);
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const [locked, setLocked] = useState<Record<string, boolean>>({});
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const dragTimeRef = useRef(0);

  const duration = nle?.duration_s ?? 0;
  const width = timelineWidth(duration, pxPerS);
  const zoom = clampZoom(pxPerS);
  const lanes = useMemo(() => (nle ? buildLanes(nle.clips) : []), [nle]);
  const captions = nle?.captions ?? [];
  const effects = nle?.effects ?? [];
  const fps = nle?.asset?.fps ?? 30;

  // fixed track stack — always all five, like a real NLE
  const tracks: TrackDef[] = [
    { id: "card", name: "V2", kind: "card", height: H_CARD, canHide: true, canMute: false },
    { id: "video", name: "V1", kind: "video", height: H_VIDEO, canHide: true, canMute: false },
    { id: "audio", name: "A1", kind: "audio", height: H_AUDIO, canHide: false, canMute: true },
    { id: "cc", name: "CC", kind: "cc", height: H_CC, canHide: true, canMute: true },
    { id: "fx", name: "FX", kind: "fx", height: H_FX, canHide: true, canMute: false },
  ];

  const laneFor = (kind: TrackDef["kind"]) =>
    kind === "card" ? lanes.find((l) => l.tone === "card")
    : kind === "video" ? lanes.find((l) => l.tone === "video")
    : kind === "audio" ? lanes.find((l) => l.tone === "narration")
    : null;

  const totalH =
    RULER_H + tracks.reduce((a, t) => a + (hidden[t.id] ? 0 : t.height), 0);

  const xToTime = useCallback(
    (clientX: number): number => {
      const el = scrollRef.current;
      if (!el) return 0;
      const rect = el.getBoundingClientRect();
      const x = clientX - rect.left + el.scrollLeft - TRACK_HEAD_W;
      let t = clamp(x / zoom, 0, Math.max(duration, 0.001));
      if (snap) t = snapTime(t, 0.5, 4 / zoom);
      return t;
    },
    [zoom, duration, snap],
  );

  const timeToPx = (t: number) => t * zoom;

  const selectedClip = useMemo(
    () => (nle?.clips ?? []).find((c) => `${c.track}:${c.name}:${c.start_s}` === selected) ?? null,
    [nle, selected],
  );

  function onRulerDown(ev: React.MouseEvent) {
    onSeek(xToTime(ev.clientX));
    setDrag({ kind: "seek" });
  }

  function onPointerMove(ev: React.MouseEvent) {
    const t = xToTime(ev.clientX);
    dragTimeRef.current = t;
    if (drag.kind === "seek") onSeek(t);
  }

  function endDrag() {
    if (drag.kind === "trim-left" && onTrimClip) {
      const c = drag.clip;
      if (c.master_index != null && c.source_start_s != null) {
        const newSourceStart = Math.max(
          0,
          c.source_start_s + (dragTimeRef.current - c.start_s),
        );
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
    if (drag.kind === "trim-right" && onTrimClip) {
      const c = drag.clip;
      if (c.master_index != null && c.source_start_s != null) {
        const newDur = Math.max(0.1, dragTimeRef.current - c.start_s);
        onTrimClip(c.master_index, c.source_start_s, Number(newDur.toFixed(3)));
      }
    }
    setDrag({ kind: "none" });
  }

  const ticks = useMemo(() => {
    const el = scrollRef.current;
    const viewW = (el?.clientWidth ?? 900) - TRACK_HEAD_W;
    const start = (el?.scrollLeft ?? 0) / zoom;
    return rulerTicks(start, start + viewW / zoom, zoom);
  }, [zoom, duration, nle]); // eslint-disable-line react-hooks/exhaustive-deps

  const canEdit = (c: NleClip) =>
    c.track === "V1_master" && c.master_index != null && c.source_start_s != null && !!onTrimClip;

  const scrubN = nle?.asset ? scrubFrameCount(nle.asset.duration_s) : 0;
  const scrubUrl = nle?.asset ? assetScrubUrl(nle.asset.id) : "";

  function zoomFit() {
    const el = scrollRef.current;
    if (el && duration > 0) {
      setPxPerS(clampZoom((el.clientWidth - TRACK_HEAD_W - 8) / duration));
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* ---- toolbar ---- */}
      <div className="flex h-9 shrink-0 items-center gap-1.5 border-b border-[var(--line-1)] bg-[var(--bg-2)] px-2.5">
        <span className="chip chip-acc">v{nle?.version ?? "—"}</span>
        <span className="tcode text-[12px] text-[var(--tx-2)]">
          {formatTimecodeFrames(playheadS, fps)}
          <span className="text-[var(--tx-3)]"> / {formatTimecodeFrames(duration, fps)}</span>
        </span>
        <span className="mx-1 h-4 w-px bg-[var(--line-2)]" />
        <button
          className="icon-btn"
          disabled={!onSplitAtPlayhead}
          title="Split clip at playhead (S)"
          onClick={() => onSplitAtPlayhead?.(playheadS)}
        >
          <Icon name="scissors" />
        </button>
        <button
          className="icon-btn"
          data-active={snap}
          title="Snapping — clips and the playhead magnetize to half-second grid and clip edges"
          onClick={() => setSnap((v) => !v)}
        >
          <Icon name="magnet" />
        </button>
        <div className="flex-1" />
        {selectedClip && (
          <span className="chip max-w-[320px]" title={selectedClip.name}>
            <span className="truncate">
              {selectedClip.name} · {selectedClip.dur_s.toFixed(2)}s
              {selectedClip.script ? ` · ${selectedClip.script.slice(0, 36)}` : ""}
            </span>
          </span>
        )}
        <button
          className="icon-btn"
          title="Zoom out (−)"
          onClick={() => setPxPerS(clampZoom(zoom / 1.5))}
        >
          <Icon name="zoomOut" />
        </button>
        <input
          type="range"
          min={4}
          max={240}
          step={1}
          value={zoom}
          className="h-1 w-24 cursor-pointer accent-[var(--acc)]"
          onChange={(e) => setPxPerS(clampZoom(Number(e.target.value)))}
          aria-label="timeline zoom"
        />
        <button
          className="icon-btn"
          title="Zoom in (+)"
          onClick={() => setPxPerS(clampZoom(zoom * 1.5))}
        >
          <Icon name="zoomIn" />
        </button>
        <button className="icon-btn" title="Zoom to fit" onClick={zoomFit}>
          <Icon name="fit" />
        </button>
      </div>

      {/* ---- tracks (scrollable) ---- */}
      <div
        ref={scrollRef}
        className="tl-scroll min-h-0 flex-1 overflow-x-auto overflow-y-hidden bg-[var(--bg-0)]"
        onMouseMove={onPointerMove}
        onMouseUp={endDrag}
        onMouseLeave={endDrag}
      >
        {!nle ? (
          <TimelineEmpty />
        ) : (
          <div className="relative select-none" style={{ width: width + TRACK_HEAD_W, height: totalH }}>
            {/* ruler */}
            <div
              className="sticky left-0 z-30 cursor-crosshair border-b border-[var(--line-1)] bg-[var(--bg-1)]"
              style={{ height: RULER_H, width: width + TRACK_HEAD_W }}
              onMouseDown={onRulerDown}
            >
              <div
                className="absolute left-0 top-0 z-10 h-full border-r border-[var(--line-1)] bg-[var(--bg-2)]"
                style={{ width: TRACK_HEAD_W }}
              >
                <span className="micro-label absolute left-2.5 top-1/2 -translate-y-1/2">
                  {formatTimecodeShort(duration)} total
                </span>
              </div>
              {ticks.map((tk) => (
                <div
                  key={tk.t}
                  className="absolute top-0 h-full"
                  style={{ left: TRACK_HEAD_W + timeToPx(tk.t) }}
                >
                  <div
                    className={
                      tk.major
                        ? "h-[10px] w-px border-l border-[var(--tx-3)]"
                        : "h-[5px] w-px border-l border-[var(--line-2)]"
                    }
                  />
                  {tk.major && (
                    <span className="tcode absolute left-1 top-[9px] text-[10px] leading-none text-[var(--tx-3)]">
                      {zoom >= 60
                        ? formatTimecodeFrames(tk.t, fps)
                        : formatTimecodeShort(tk.t)}
                    </span>
                  )}
                </div>
              ))}
            </div>

            {/* track rows */}
            {tracks.map((track) => {
              const lane = laneFor(track.kind);
              const isHidden = !!hidden[track.id];
              const isLocked = !!locked[track.id];
              return (
                <div
                  key={track.id}
                  className="relative border-b border-[var(--line-1)]"
                  style={{ height: isHidden ? 0 : track.height, overflow: "hidden" }}
                >
                  {/* header */}
                  <div
                    className="absolute left-0 top-0 z-10 flex h-full items-center gap-1.5 border-r border-[var(--line-1)] bg-[var(--bg-2)] px-2"
                    style={{ width: TRACK_HEAD_W, height: track.height }}
                  >
                    <span
                      className="flex h-[17px] min-w-[22px] items-center justify-center rounded px-1 text-[10px] font-bold"
                      style={{
                        color: `var(--clip-${track.kind === "card" ? "card" : track.kind === "video" ? "video" : track.kind === "audio" ? "audio" : track.kind === "cc" ? "cc" : "fx"})`,
                        background: "rgba(255,255,255,0.06)",
                      }}
                    >
                      {track.name}
                    </span>
                    <span className="text-[10px] font-medium tracking-wide text-[var(--tx-2)]">{trackLabel(track.kind)}</span>
                    <div className="flex-1" />
                    {track.canMute && (
                      <button
                        className="icon-btn !h-5 !w-5"
                        title="mute / solo monitoring"
                        onClick={() =>
                          setHidden((h) => ({ ...h, [track.id]: !h[track.id] }))
                        }
                      >
                        <Icon name="mute" size={12} />
                      </button>
                    )}
                    {track.canHide && (
                      <button
                        className="icon-btn !h-5 !w-5"
                        title={isHidden ? "show track" : "hide track"}
                        onClick={() =>
                          setHidden((h) => ({ ...h, [track.id]: !h[track.id] }))
                        }
                      >
                        <Icon name={isHidden ? "eyeOff" : "eye"} size={12} />
                      </button>
                    )}
                    <button
                      className="icon-btn !h-5 !w-5"
                      title={isLocked ? "unlock track" : "lock track (blocks edits)"}
                      onClick={() => setLocked((l) => ({ ...l, [track.id]: !l[track.id] }))}
                    >
                      <Icon name={isLocked ? "lock" : "unlock"} size={12} />
                    </button>
                  </div>

                  {/* lane content */}
                  {!isHidden && (
                    <div
                      className="relative h-full bg-[#17181c]"
                      style={{ marginLeft: TRACK_HEAD_W, width }}
                    >
                      {/* grid lines every second */}
                      <GridLines width={width} zoom={zoom} height={track.height} />
                      {track.kind === "cc" && (
                        <CaptionTrack
                          captions={captions}
                          zoom={zoom}
                          onSeek={onSeek}
                          duration={duration}
                        />
                      )}
                      {track.kind === "fx" && (
                        <EffectTrack
                          effects={effects}
                          zoom={zoom}
                          duration={duration}
                          onSeek={onSeek}
                          onDelete={onDeleteEffect}
                        />
                      )}
                      {lane &&
                        lane.clips.map((c) => (
                          <ClipBlock
                            key={`${c.track}:${c.name}:${c.start_s}`}
                            clip={c}
                            kind={track.kind}
                            zoom={zoom}
                            selected={
                              selected === `${c.track}:${c.name}:${c.start_s}`
                            }
                            locked={isLocked}
                            scrubN={scrubN}
                            scrubUrl={scrubUrl}
                            sourceDuration={nle.asset?.duration_s ?? duration}
                            onSelect={(k) => setSelected((s) => (s === k ? null : k))}
                            onSeek={onSeek}
                            onTrimStart={(clip) => {
                              setDrag({ kind: "trim-left", clip });
                            }}
                            onTrimEnd={(clip) => {
                              setDrag({ kind: "trim-right", clip });
                            }}
                            canEdit={canEdit(c)}
                          />
                        ))}
                      {lane && lane.clips.length === 0 && (
                        <span className="micro-label absolute left-2.5 top-1/2 -translate-y-1/2 opacity-40">
                          {trackEmptyLabel(track.kind)}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              );
            })}

            {/* playhead — spans ruler + all tracks; above the sticky ruler
                header so it stays visible at the left edge (Premiere behaviour) */}
            <div
              className="playhead-line"
              style={{
                left: TRACK_HEAD_W + timeToPx(clamp(playheadS, 0, duration)),
                height: totalH - RULER_H,
                top: RULER_H,
                zIndex: 40,
              }}
            />
            <div
              className="playhead-flag"
              style={{
                left: TRACK_HEAD_W + timeToPx(clamp(playheadS, 0, duration)) - 5.5,
                zIndex: 41,
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function trackLabel(kind: TrackDef["kind"]): string {
  return {
    video: "master cut",
    card: "cards",
    audio: "narration",
    cc: "captions",
    fx: "effects",
  }[kind];
}

function trackEmptyLabel(kind: TrackDef["kind"]): string {
  return {
    video: "no cut yet",
    card: "—",
    audio: "—",
    cc: "no cues",
    fx: "—",
  }[kind];
}

function GridLines({ width, zoom, height }: { width: number; zoom: number; height: number }) {
  const stepS = zoom >= 120 ? 1 : zoom >= 30 ? 5 : 30;
  const lines: number[] = [];
  for (let t = stepS; t * zoom < width; t += stepS) lines.push(t);
  if (lines.length > 400) return null;
  return (
    <>
      {lines.map((t) => (
        <div
          key={t}
          className="absolute top-0 h-full border-l border-[#1f2025]"
          style={{ left: t * zoom }}
        />
      ))}
      <div className="absolute inset-0" style={{ height }} />
    </>
  );
}

function ClipBlock({
  clip,
  kind,
  zoom,
  selected,
  locked,
  scrubN,
  scrubUrl,
  sourceDuration,
  onSelect,
  onSeek,
  onTrimStart,
  onTrimEnd,
  canEdit,
}: {
  clip: NleClip;
  kind: TrackDef["kind"];
  zoom: number;
  selected: boolean;
  locked: boolean;
  scrubN: number;
  scrubUrl: string;
  sourceDuration: number;
  onSelect: (key: string) => void;
  onSeek: (t: number) => void;
  onTrimStart: (clip: NleClip) => void;
  onTrimEnd: (clip: NleClip) => void;
  canEdit: boolean;
}) {
  const key = `${clip.track}:${clip.name}:${clip.start_s}`;
  const w = Math.max(3, clip.dur_s * zoom);
  const label =
    kind === "audio" ? `VO · ${clip.name}` : kind === "card" ? `CARD · ${clip.name}` : clip.name;

  // filmstrip cells: one scrubsheet frame per 2 s of SOURCE time
  const cells: React.ReactNode[] = [];
  if (kind === "video" && scrubN > 0 && w > 24) {
    const cellWS = SCRUB_CELL_S * zoom;
    const nCells = Math.min(80, Math.ceil(w / cellWS) + 1);
    for (let i = 0; i < nCells; i++) {
      const tLocal = i * SCRUB_CELL_S;
      if (tLocal > clip.dur_s) break;
      const srcT = (clip.source_start_s ?? 0) + tLocal;
      cells.push(
        <div
          key={i}
          className="clip-film"
          style={{
            left: tLocal * zoom,
            width: Math.min(cellWS, w - tLocal * zoom),
            ...scrubBackground(
              scrubFrameIndex(srcT, sourceDuration),
              scrubN,
              scrubUrl,
            ),
          }}
        />,
      );
    }
  }

  return (
    <div
      className="clip"
      data-selected={selected}
      style={{
        left: clip.start_s * zoom,
        width: w,
        background: `var(--clip-${kind === "video" ? "video" : kind === "card" ? "card" : "audio"})`,
      }}
      title={`${clip.name} · ${clip.dur_s.toFixed(2)}s${clip.act ? ` · ${clip.act}` : ""}${clip.script ? ` · ${clip.script.slice(0, 70)}` : ""}`}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(key);
        onSeek(clip.start_s);
      }}
    >
      {cells}
      {/* label bar over the filmstrip */}
      <div
        className="absolute inset-x-0 bottom-0 flex items-center px-1.5"
        style={{
          height: kind === "video" ? 15 : "100%",
          background:
            kind === "video"
              ? "linear-gradient(to top, rgba(6,8,10,.95) 0%, rgba(6,8,10,.75) 55%, rgba(6,8,10,.25) 100%)"
              : undefined,
        }}
      >
        <span className="truncate text-[10px] font-medium leading-none text-white/90">
          {label}
        </span>
        {clip.audio_mode && (
          <span className="ml-auto shrink-0 rounded bg-black/40 px-1 text-[8px] font-bold text-white/80">
            {clip.audio_mode === "source_audio"
              ? "SRC"
              : clip.audio_mode === "vo_only"
                ? "VO"
                : clip.audio_mode.toUpperCase()}
          </span>
        )}
      </div>
      {selected && canEdit && !locked && (
        <>
          <span
            className="trim-handle left-0"
            title="drag to trim in-point"
            onMouseDown={(e) => {
              e.stopPropagation();
              onTrimStart(clip);
            }}
          />
          <span
            className="trim-handle right-0"
            title="drag to trim out-point"
            onMouseDown={(e) => {
              e.stopPropagation();
              onTrimEnd(clip);
            }}
          />
        </>
      )}
    </div>
  );
}

const SCRUB_CELL_S = 2;

function CaptionTrack({
  captions,
  zoom,
  onSeek,
  duration,
}: {
  captions: NleCaption[];
  zoom: number;
  onSeek: (t: number) => void;
  duration: number;
}) {
  if (captions.length === 0) return null;
  return (
    <>
      {captions.map((cue) => {
        const start = cue.start_ms / 1000;
        const w = Math.max(2, ((cue.end_ms - cue.start_ms) / 1000) * zoom);
        return (
          <div
            key={cue.index}
            className="clip !cursor-pointer"
            style={{
              left: start * zoom,
              width: w,
              top: 2,
              bottom: 2,
              background: "var(--clip-cc)",
              borderColor: "rgba(255,255,255,.08)",
            }}
            title={`${cue.speaker}: ${cue.text}`}
            onClick={() => onSeek(start)}
          >
            <div className="absolute inset-0 flex items-center px-1.5">
              <span className="truncate text-[9px] leading-none text-white/90">
                {cue.text}
              </span>
            </div>
          </div>
        );
      })}
      <span className="hidden">{duration}</span>
    </>
  );
}

function EffectTrack({
  effects,
  zoom,
  duration,
  onSeek,
  onDelete,
}: {
  effects: NleEffect[];
  zoom: number;
  duration: number;
  onSeek: (t: number) => void;
  onDelete?: (id: string) => void;
}) {
  if (effects.length === 0) return null;
  return (
    <>
      {effects.map((fx) => {
        const start = fx.t_start_s ?? 0;
        const end = fx.t_end_s ?? duration;
        const w = Math.max(6, (end - start) * zoom);
        const agent = fx.origin === "agent";
        return (
          <div
            key={fx.id}
            className="clip group"
            style={{
              left: start * zoom,
              width: w,
              top: 2,
              bottom: 2,
              background: "var(--clip-fx)",
              outline: agent ? "1px dashed rgba(255,255,255,0.55)" : undefined,
              outlineOffset: -1,
            }}
            title={`${fx.label || "blur"} · ${fx.mode}${agent ? " · agent op" : ""}${fx.t_start_s != null ? ` · ${start.toFixed(1)}s–${end.toFixed(1)}s` : " · full length"} · box ${fx.w}x${fx.h}@${fx.x},${fx.y}`}
            onClick={() => onSeek(start)}
          >
            <div className="absolute inset-0 flex items-center gap-1 px-1.5">
              <span className="truncate text-[9px] leading-none text-white/90">
                {fx.label || "blur"} · {fx.mode}
              </span>
              {agent && (
                <span
                  className="shrink-0 rounded bg-black/45 px-1 text-[8px] font-semibold uppercase tracking-wider leading-[1.4] text-white/85"
                  title="applied by the agent — every op is on the log with its evidence"
                >
                  AI
                </span>
              )}
              {onDelete && (fx.op_id || fx.created_by === "human_edit") && (
                <button
                  className="ml-auto hidden shrink-0 rounded bg-black/40 px-1 text-[9px] leading-none text-white/80 hover:bg-[var(--err)] hover:text-white group-hover:block"
                  title={fx.op_id ? "revoke this op (history kept)" : "delete this studio-drawn box"}
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(fx.id);
                  }}
                >
                  ✕
                </button>
              )}
            </div>
          </div>
        );
      })}
    </>
  );
}

function TimelineEmpty() {
  return (
    <div className="flex h-full min-h-[120px] flex-col items-center justify-center gap-2">
      <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--bg-2)] text-[var(--tx-3)]">
        <Icon name="clapper" size={20} />
      </span>
      <p className="text-[12px] text-[var(--tx-2)]">
        The agent&apos;s cut lands here — live.
      </p>
      <p className="text-[11px] text-[var(--tx-3)]">
        Press <span className="chip">Run pipeline</span> in the top bar; tracks
        fill as the director plans, trims and captions.
      </p>
    </div>
  );
}
