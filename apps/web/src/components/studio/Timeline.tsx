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

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  voTrimWindow,
} from "@/lib/nle";
import type { NleCaption, NleClip, NleEffect, NleTimeline } from "@/lib/api";
import { effectWindowFrames } from "@/lib/nle";
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
  | { kind: "trim-right"; clip: NleClip }
  // FX-track edge drag: retime the op's frame window (adjust endpoint carries
  // frame_start/frame_end; the box geometry passes through unchanged)
  | { kind: "fx-start"; fx: NleEffect }
  | { kind: "fx-end"; fx: NleEffect };

export interface TimelineProps {
  nle: NleTimeline | null;
  playheadS: number;
  onSeek: (t: number) => void;
  onTrimClip?: (itemIndex: number, newStartS: number, newDurS: number) => void;
  /** Drag an A1 audio block edge -> new VO window inside the TTS stem. */
  onTrimAudio?: (itemIndex: number, voInS: number, voOutS: number | null) => void;
  onSplitAtPlayhead?: (playheadS: number) => void;
  onDeleteEffect?: (effectId: string) => void;
  /** Drag an op-backed effect block's edge → new frame window (raw frames). */
  onAdjustEffectWindow?: (opId: string, frameStart: number, frameEnd: number) => void;
  /** Delete the selected master clip (raw or narration) — server checkpoints. */
  onDeleteClip?: (clip: NleClip) => void;
  /** Open the insert popover (raw clip / narration / card at the playhead). */
  onInsert?: () => void;
  /** Version-pointer undo/redo (checkpoint history — nothing is destroyed). */
  canUndo?: boolean;
  canRedo?: boolean;
  onUndo?: () => void;
  onRedo?: () => void;
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
  onTrimAudio,
  onSplitAtPlayhead,
  onDeleteEffect,
  onAdjustEffectWindow,
  onDeleteClip,
  onInsert,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
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
    // A1 audio trim: the edge drag cuts INSIDE the TTS stem (vo window),
    // the twin math of the master trim below — narration blocks never move
    // source windows, they re-cut the voiceover itself.
    if (
      (drag.kind === "trim-left" || drag.kind === "trim-right") &&
      drag.clip.kind === "narration" &&
      onTrimAudio
    ) {
      const c = drag.clip;
      if (c.master_index != null) {
        const win = voTrimWindow(
          drag.kind === "trim-left" ? "start" : "end",
          dragTimeRef.current,
          c,
        );
        if (win) onTrimAudio(c.master_index, win.vo_in_s, win.vo_out_s);
      }
      setDrag({ kind: "none" });
      return;
    }
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
    if (
      (drag.kind === "fx-start" || drag.kind === "fx-end") &&
      onAdjustEffectWindow &&
      drag.fx.op_id
    ) {
      const win = effectWindowFrames(
        drag.kind === "fx-start" ? "start" : "end",
        dragTimeRef.current,
        fps,
        duration,
        { start: drag.fx.frame_start ?? null, end: drag.fx.frame_end ?? null },
      );
      if (win) {
        onAdjustEffectWindow(drag.fx.op_id, win.frame_start, win.frame_end);
      }
    }
    setDrag({ kind: "none" });
  }

  // CR-013: ticks used to read scrollLeft once at render time (no scroll
  // listener) — scrolling right past the first viewport left the ruler blank.
  // Scroll events (rAF-throttled) now drive the visible tick window.
  const [scrollState, setScrollState] = useState({ left: 0, viewW: 900 });
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() =>
        setScrollState({ left: el.scrollLeft, viewW: el.clientWidth }),
      );
    };
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(raf);
    };
  }, [nle]);

  const ticks = useMemo(
    () =>
      rulerTicks(
        scrollState.left / zoom,
        (scrollState.left + scrollState.viewW - TRACK_HEAD_W) / zoom,
        zoom,
      ),
    [scrollState, zoom],
  );

  const canEdit = (c: NleClip) =>
    c.track === "V1_master" && c.master_index != null && c.source_start_s != null && !!onTrimClip;
  const canTrimAudio = (c: NleClip) =>
    c.kind === "narration" && c.master_index != null && !!onTrimAudio;

  const scrubN = nle?.asset ? scrubFrameCount(nle.asset.duration_s) : 0;
  const scrubUrl = nle?.asset ? assetScrubUrl(nle.asset.id) : "";

  // A real NLE timeline is editable: Delete removes the selected master clip
  // (server checkpoint = undoable), and the same key never fires while typing.
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key !== "Delete" && ev.key !== "Backspace") return;
      const el = ev.target;
      if (el instanceof HTMLElement && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable))
        return;
      if (!selectedClip || !onDeleteClip) return;
      ev.preventDefault();
      onDeleteClip(selectedClip);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedClip, onDeleteClip]);

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
          disabled={!onDeleteClip || !selectedClip || selectedClip.master_index == null}
          title={
            selectedClip
              ? `Delete "${selectedClip.name}" (${selectedClip.dur_s.toFixed(1)}s) — checkpointed, Ctrl+Z-capable via history`
              : "select a clip, then Delete"
          }
          onClick={() => selectedClip && onDeleteClip?.(selectedClip)}
        >
          <Icon name="trash" />
        </button>
        <button
          className="icon-btn"
          disabled={!onUndo || !canUndo}
          title="Undo — activate the previous timeline checkpoint (Ctrl+Z when the video surface has focus)"
          onClick={() => onUndo?.()}
        >
          <Icon name="undo" />
        </button>
        <button
          className="icon-btn"
          disabled={!onRedo || !canRedo}
          title="Redo — activate the next timeline checkpoint"
          onClick={() => onRedo?.()}
        >
          <Icon name="redo" />
        </button>
        <button
          className="icon-btn"
          disabled={!onInsert}
          title="Add to the timeline at the playhead — raw clip from the source, narration VO, or a card"
          onClick={() => onInsert?.()}
        >
          <Icon name="plus" />
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
      <div className="relative min-h-0 flex-1">
        <div
          ref={scrollRef}
          className="tl-scroll h-full min-h-0 overflow-x-auto overflow-y-hidden bg-[var(--bg-0)]"
          onMouseMove={onPointerMove}
          onMouseUp={endDrag}
          onMouseLeave={endDrag}
        >
        {!nle ? (
          <TimelineEmpty />
        ) : (
          <div className="relative select-none" style={{ width: width + TRACK_HEAD_W, height: totalH }}>
            {/* ruler — spans the full content width; the fixed corner lives in
                the header overlay below (sticky was a no-op: the ruler spanned
                its own containing block, CR-013) */}
            <div
              className="cursor-crosshair border-b border-[var(--line-1)] bg-[var(--bg-1)]"
              style={{ height: RULER_H, width: width + TRACK_HEAD_W }}
              onMouseDown={onRulerDown}
            >
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
                  {/* header cells live in the pinned overlay — the scrolled
                      copy used to slide away and leave unlabeled tracks */}

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
                          onDragStartEdge={(fx) => setDrag({ kind: "fx-start", fx })}
                          onDragEndEdge={(fx) => setDrag({ kind: "fx-end", fx })}
                        />
                      )}
                      {/* V1 also renders narration blocks as muted B-ROLL
                          ghosts — the b-roll the renderer cuts under every VO
                          is part of the program and must be visible in the
                          timeline, not hidden behind the A1 lane */}
                      {track.kind === "video" &&
                        lanes
                          .find((l) => l.tone === "narration")
                          ?.clips.map((c) => (
                            <ClipBlock
                              key={`broll:${c.track}:${c.name}:${c.start_s}`}
                              clip={c}
                              kind={track.kind}
                              zoom={zoom}
                              selected={false}
                              locked
                              scrubN={scrubN}
                              scrubUrl={scrubUrl}
                              sourceDuration={nle.asset?.duration_s ?? duration}
                              onSelect={() => onSeek(c.start_s)}
                              onSeek={onSeek}
                              onTrimStart={() => {}}
                              onTrimEnd={() => {}}
                              canEdit={false}
                              isAudio={false}
                              isBroll
                            />
                          ))}
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
                            canEdit={canEdit(c) || canTrimAudio(c)}
                            isAudio={c.kind === "narration"}
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

        {/* fixed left column (CR-013): track labels + controls pinned to the
            viewport edge — scrolled lane content passes UNDER it, like a real
            NLE. pointer-events sit on the cells only, so lane clicks in the
            first 148px still work when nothing is scrolled over them. During
            an active drag the overlay goes fully inert — a pointer crossing
            the header column must NOT fire the scroll container's mouseleave
            and cancel a running trim. */}
        {nle && (
          <div
            className="pointer-events-none absolute inset-y-0 left-0 z-50"
            style={{ width: TRACK_HEAD_W }}
          >
            <div
              className={`absolute left-0 top-0 border-b border-r border-[var(--line-1)] bg-[var(--bg-2)] ${
                drag.kind === "none" ? "pointer-events-auto" : "pointer-events-none"
              }`}
              style={{ height: RULER_H, width: TRACK_HEAD_W }}
            >
              <span className="micro-label absolute left-2.5 top-1/2 -translate-y-1/2">
                {formatTimecodeShort(duration)} total
              </span>
            </div>
            <div className="absolute left-0" style={{ top: RULER_H }}>
              {tracks.map((track) => {
                const isHidden = !!hidden[track.id];
                const isLocked = !!locked[track.id];
                return (
                  <div
                    key={track.id}
                    className={`flex items-center gap-1.5 border-b border-r border-[var(--line-1)] bg-[var(--bg-2)] px-2 ${
                      drag.kind === "none" ? "pointer-events-auto" : "pointer-events-none"
                    }`}
                    style={{
                      width: TRACK_HEAD_W,
                      height: isHidden ? 0 : track.height,
                      overflow: "hidden",
                    }}
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
                    <span className="text-[10px] font-medium tracking-wide text-[var(--tx-2)]">
                      {trackLabel(track.kind)}
                    </span>
                    <div className="flex-1" />
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
                );
              })}
            </div>
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
  isAudio,
  isBroll,
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
  isAudio?: boolean;
  /** V1 ghost of a narration block: the muted b-roll the renderer cuts
   *  under the voiceover — dashed, dimmed, seek-on-click only. */
  isBroll?: boolean;
}) {
  const key = `${clip.track}:${clip.name}:${clip.start_s}`;
  const w = Math.max(3, clip.dur_s * zoom);
  const label = isBroll
    ? `B-ROLL · muted · ${clip.name}`
    : kind === "audio"
      ? `VO · ${clip.name}`
      : kind === "card"
        ? `CARD · ${clip.name}`
        : clip.name;
  // A1: which b-roll windows ride under this VO (the renderer cycles them)
  const brollList = clip.broll_windows ?? null;
  // A1 audio trim truth: the audio extent inside the block (tail pad renders
  // darker so the operator sees exactly how much VO is there)
  const audioLen =
    isAudio && (clip.vo_out_s != null || clip.tts_duration_s != null || (clip.vo_in_s ?? 0) > 0)
      ? Math.min(
          (clip.vo_out_s ?? clip.tts_duration_s ?? clip.dur_s + (clip.vo_in_s ?? 0)) -
            (clip.vo_in_s ?? 0),
          clip.dur_s,
        )
      : null;
  const trimmed = isAudio && ((clip.vo_in_s ?? 0) > 0 || clip.vo_out_s != null);

  // filmstrip cells: one scrubsheet frame per 2 s of SOURCE time
  const cells: React.ReactNode[] = [];
  if ((kind === "video" || isBroll) && scrubN > 0 && w > 24) {
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
        ...(isBroll
          ? {
              borderStyle: "dashed",
              borderColor: "rgba(114,208,165,.55)",
              background: "rgba(38,68,56,.45)",
              opacity: 0.88,
            }
          : {}),
      }}
      title={`${clip.name} · ${clip.dur_s.toFixed(2)}s${
        isAudio && trimmed
          ? ` · VO ${(clip.vo_in_s ?? 0).toFixed(2)}–${
              clip.vo_out_s != null ? clip.vo_out_s.toFixed(2) : "end"
            }s`
          : ""
      }${
        brollList
          ? ` · b-roll ${brollList
              .map(([a, b]) => `${formatTimecodeShort(a)}–${formatTimecodeShort(b)}`)
              .join(" → ")}`
          : ""
      }${clip.act ? ` · ${clip.act}` : ""}${clip.script ? ` · ${clip.script.slice(0, 70)}` : ""}`}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(key);
        onSeek(clip.start_s);
      }}
    >
      {cells}
      {/* A1: tail pad zone — the block is longer than the trimmed VO here */}
      {isAudio && audioLen != null && audioLen < clip.dur_s - 0.01 && (
        <div
          className="absolute inset-y-0 right-0 bg-black/45"
          style={{ left: audioLen * zoom }}
          title="tail pad — air after the voiceover ends"
        />
      )}
      {isAudio && trimmed && (
        <span className="absolute right-1 top-1 z-10 rounded bg-black/55 px-1 text-[8px] font-bold tracking-wide text-white/85">
          VO TRIM
        </span>
      )}
      {/* A1 b-roll badge: the muted source windows under this voiceover —
          invisible b-roll is indistinguishable from missing b-roll */}
      {isAudio && (
        <span
          className={`absolute left-1 top-1 z-10 rounded px-1 text-[8px] font-bold tracking-wide ${
            brollList
              ? "bg-[rgba(20,66,46,.8)] text-[#9fe3c0]"
              : "bg-[rgba(96,48,20,.8)] text-[#f0c084]"
          }`}
          title={
            brollList
              ? `b-roll windows: ${brollList
                  .map(([a, b]) => `${formatTimecodeShort(a)}–${formatTimecodeShort(b)}`)
                  .join(" → ")}`
              : "no b-roll windows in metadata — the render will be black here"
          }
        >
          {brollList ? `B-ROLL ×${brollList.length}` : "NO B-ROLL"}
        </span>
      )}
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
  onDragStartEdge,
  onDragEndEdge,
}: {
  effects: NleEffect[];
  zoom: number;
  duration: number;
  onSeek: (t: number) => void;
  onDelete?: (id: string) => void;
  onDragStartEdge?: (fx: NleEffect) => void;
  onDragEndEdge?: (fx: NleEffect) => void;
}) {
  if (effects.length === 0) return null;
  return (
    <>
      {effects.map((fx) => {
        const start = fx.t_start_s ?? 0;
        const end = fx.t_end_s ?? duration;
        const w = Math.max(6, (end - start) * zoom);
        const agent = fx.origin === "agent";
        const adjustable = !!fx.op_id && fx.mode !== "tracked" && !!onDragStartEdge;
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
            {adjustable && (
              <>
                <span
                  className="trim-handle left-0"
                  title="drag to retime the effect's start frame (release lands the adjust op)"
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    onDragStartEdge?.(fx);
                  }}
                />
                <span
                  className="trim-handle right-0"
                  title="drag to retime the effect's end frame (release lands the adjust op)"
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    onDragEndEdge?.(fx);
                  }}
                />
              </>
            )}
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
