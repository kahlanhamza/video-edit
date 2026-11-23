"use client";

// ProgramMonitor — the NLE program viewer: letterboxed proxy on true black,
// mouse-drawn blur boxes over it, and a professional transport bar
// (go-start / step-back / play / step-fwd / go-end + hero timecode).
// Keyboard: Space play · J/K/L shuttle · I/O in-out · arrows frame-step.

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { Asset, NleEffect, OpRecord, SelectionOut, TrackHit, api, assetProxyUrl, assetScrubUrl } from "@/lib/api";
import {
  AdjustDrag,
  TrackedBox,
  adjustedBox,
  formatTimecodeFrames,
  scrubBackground,
  scrubFrameCount,
  scrubFrameIndex,
  trackedBoxAt,
} from "@/lib/nle";
import { useNLEShortcuts } from "@/hooks/useNLEShortcuts";
import { Icon } from "./studio/icons";

export interface MonitorHandle {
  seek: (t: number) => void;
}

interface Draft {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export const ProgramMonitor = forwardRef<
  MonitorHandle,
  {
    projectId: string;
    asset: Asset;
    fps: number;
    onTimeUpdate: (t: number) => void;
    onChanged: () => void;
  }
>(function ProgramMonitor(
  { projectId, asset, fps, onTimeUpdate, onChanged },
  ref,
) {
  const [regions, setRegions] = useState<NleEffect[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showExisting, setShowExisting] = useState(true);
  const [drawMode, setDrawMode] = useState(false);
  // manual adjust (dual-modality law: agent ops stay human-refinable):
  // click a static op-backed box to select it, drag to move, drag the corner
  // handle to resize, Alt+arrows to nudge — release lands an adjust op
  // (append-only: a NEW op supersedes the old one, coverage never gaps)
  const [selectedOp, setSelectedOp] = useState<string | null>(null);
  const [adjustDrag, setAdjustDrag] = useState<AdjustDrag | null>(null);
  const [preview, setPreview] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [videoReady, setVideoReady] = useState(false);
  const [tracking, setTracking] = useState<string | null>(null);
  const [trackedPaths, setTrackedPaths] = useState<Record<string, TrackHit[]>>({});
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const { state: nle } = useNLEShortcuts({ videoRef, fps, enabled: !drawMode });

  useImperativeHandle(ref, () => ({
    seek: (t: number) => {
      const v = videoRef.current;
      if (v) v.currentTime = t;
      setTime(t);
    },
  }));

  const onTimeUpdateRef = useRef(onTimeUpdate);
  onTimeUpdateRef.current = onTimeUpdate;

  const reload = useCallback(async () => {
    try {
      // The op fold merges op-backed blur boxes with legacy regions — one
      // source of truth for what the program actually blurs (op log v1).
      const state = await api.listOps(projectId);
      setRegions(state.fold.effects);
      setError(null);
      // tracked selections (step 3): fetch each op's per-frame path once —
      // the monitor then draws the box FOLLOWING the object as you scrub
      const need = state.fold.effects.filter(
        (r) => r.mode === "tracked" && r.op_id && !trackedPaths[r.op_id],
      );
      if (need.length > 0) {
        const loaded = await Promise.all(
          need.map(async (r) => {
            try {
              const op = await api.getOp(r.op_id!);
              const hits = (op.params?.tracked as TrackHit[] | undefined) ?? [];
              return [r.op_id!, hits] as const;
            } catch {
              return [r.op_id!, [] as TrackHit[]] as const;
            }
          }),
        );
        setTrackedPaths((prev) => {
          const next = { ...prev };
          for (const [id, hits] of loaded) next[id] = hits;
          return next;
        });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  useEffect(() => {
    reload();
  }, [reload]);

  function toDisplayCoords(ev: React.MouseEvent): { x: number; y: number } {
    const rect = wrapRef.current!.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(ev.clientX - rect.left, rect.width)),
      y: Math.max(0, Math.min(ev.clientY - rect.top, rect.height)),
    };
  }

  function onMouseDown(ev: React.MouseEvent) {
    const p = toDisplayCoords(ev);
    setDraft({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
  }

  function onMouseMove(ev: React.MouseEvent) {
    if (adjustDrag) {
      const p = toDisplayCoords(ev);
      const rect = wrapRef.current!.getBoundingClientRect();
      setPreview(
        adjustedBox(
          adjustDrag,
          p,
          { x: asset.width / rect.width, y: asset.height / rect.height },
          asset.width,
          asset.height,
        ),
      );
      return;
    }
    if (!draft) return;
    const p = toDisplayCoords(ev);
    setDraft({ ...draft, x1: p.x, y1: p.y });
  }

  /** Begin a manual adjust drag on one static op-backed box. */
  function startAdjust(r: NleEffect, mode: "move" | "resize", ev: React.MouseEvent) {
    if (drawMode || !r.op_id) return; // draw mode owns the surface; tracked ops follow the object
    ev.stopPropagation();
    videoRef.current?.pause();
    setSelectedOp(r.op_id);
    setAdjustDrag({
      opId: r.op_id,
      mode,
      origRaw: { x: r.x, y: r.y, w: r.w, h: r.h },
      startDisplay: toDisplayCoords(ev),
    });
    setPreview({ x: r.x, y: r.y, w: r.w, h: r.h });
  }

  async function landAdjust() {
    if (!adjustDrag || !preview) return;
    const patch = {
      ...preview,
      note: `manually ${adjustDrag.mode === "move" ? "moved" : "resized"} in the studio`,
    };
    setAdjustDrag(null);
    setPreview(null);
    try {
      const next: OpRecord = await api.adjustOp(projectId, adjustDrag.opId, patch);
      setSelectedOp(next.id); // the box keeps its identity through the chain hop
      await reload();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function onMouseUp() {
    if (adjustDrag) {
      await landAdjust();
      return;
    }
    if (!draft) return;
    const dx = Math.abs(draft.x1 - draft.x0);
    const dy = Math.abs(draft.y1 - draft.y0);
    setDraft(null);
    if (dx < 8 || dy < 8) return;
    const rect = wrapRef.current!.getBoundingClientRect();
    const sx = asset.width / rect.width;
    const sy = asset.height / rect.height;
    try {
      // A drawn box is an OP (origin=human) — the log keeps who drew what;
      // the API dual-writes the render-path region automatically.
      await api.createOp(projectId, {
        kind: "blur",
        origin: "human",
        evidence: { source: "program_monitor_draw" },
        params: {
          x: Math.round(Math.min(draft.x0, draft.x1) * sx),
          y: Math.round(Math.min(draft.y0, draft.y1) * sy),
          w: Math.max(1, Math.round(dx * sx)),
          h: Math.max(1, Math.round(dy * sy)),
          sigma: 24,
        },
      });
      await reload();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  // Alt+arrows nudge the selected box (1 raw px, Shift = 10) — frame-precise
  // manual refinement without fighting the J/K/L/arrows transport shortcuts.
  useEffect(() => {
    if (!selectedOp || drawMode) return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") {
        setSelectedOp(null);
        return;
      }
      if (!ev.altKey || !ev.key.startsWith("Arrow")) return;
      ev.preventDefault();
      const r = regions.find((x) => x.op_id === selectedOp);
      if (!r) return;
      const step = ev.shiftKey ? 10 : 1;
      const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[
        ev.key as "ArrowLeft" | "ArrowRight" | "ArrowUp" | "ArrowDown"
      ];
      const x = Math.max(0, Math.min(asset.width - r.w, r.x + d[0]));
      const y = Math.max(0, Math.min(asset.height - r.h, r.y + d[1]));
      api
        .adjustOp(projectId, selectedOp, { x, y, w: r.w, h: r.h, note: "nudged with arrow keys" })
        .then(async (next) => {
          setSelectedOp(next.id);
          await reload();
          onChanged();
        })
        .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedOp, drawMode, regions, projectId, asset.width, asset.height, onChanged]); // eslint-disable-line react-hooks/exhaustive-deps

  async function removeRegion(r: NleEffect) {
    try {
      if (r.op_id) {
        await api.revokeOp(projectId, r.op_id, "removed from the program monitor");
      } else {
        await api.deleteRedaction(projectId, r.id);
      }
      await reload();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  /** Track this box from the playhead: promptable selection (step 3). The
   * drawn box seeds the prompt; the tracked op REPLACES the static one
   * (revoked — history kept) so the program never double-blurs. */
  async function trackRegion(r: NleEffect) {
    if (!r.op_id) return; // legacy regions have no op to upgrade
    setTracking(r.op_id);
    setError(null);
    const anchor = Math.min(Math.max(0, Math.round(time * fps)), Math.max(0, Math.round(asset.duration_s * fps) - 1));
    const span = Math.min(12, Math.max(1, asset.duration_s - anchor / fps));
    try {
      const sel: SelectionOut = await api.createSelection(asset.id, {
        kind: "box",
        x: r.x,
        y: r.y,
        w: r.w,
        h: r.h,
        anchor_frame: anchor,
        frame_start: anchor,
        frame_end: Math.min(Math.round(asset.duration_s * fps) - 1, anchor + Math.round(span * fps)),
      });
      await api.revokeOp(projectId, r.op_id, `superseded by tracked op ${sel.op_id?.slice(0, 8)}`);
      if (sel.op_id) {
        setTrackedPaths((prev) => ({ ...prev, [sel.op_id!]: sel.hits }));
      }
      await reload();
      onChanged();
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setTracking(null);
    }
  }

  const rect = wrapRef.current?.getBoundingClientRect();
  const scaleX = rect ? rect.width / asset.width : 0;
  const scaleY = rect ? rect.height / asset.height : 0;

  const transport = (fn: (v: HTMLVideoElement) => void) => () => {
    if (videoRef.current) fn(videoRef.current);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* ---- viewer toolbar ---- */}
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-[var(--line-1)] bg-[var(--bg-2)] px-2.5">
        <span className="micro-label">Program</span>
        <div className="flex-1" />
        <button
          className="icon-btn"
          data-active={drawMode}
          title="Draw mode — drag a box on the video to add a privacy blur (raw-frame coords)"
          onClick={() => {
            setDrawMode(!drawMode);
            setDraft(null);
          }}
        >
          <Icon name="wand" />
        </button>
        <button
          className="icon-btn"
          data-active={showExisting}
          title={showExisting ? "hide blur boxes" : "show blur boxes"}
          onClick={() => setShowExisting((v) => !v)}
        >
          <Icon name={showExisting ? "eye" : "eyeOff"} />
        </button>
        <span className="chip" title="blur boxes on the op log (agent + human)">
          blur boxes: {regions.length}
        </span>
      </div>

      {/* ---- video surface ---- */}
      <div className="relative flex min-h-0 flex-1 items-center justify-center bg-black">
        <div
          ref={wrapRef}
          className={`relative h-full max-h-full select-none ${drawMode ? "cursor-crosshair" : ""}`}
          onMouseDown={drawMode ? onMouseDown : undefined}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={() => {
            if (adjustDrag) {
              setAdjustDrag(null);
              setPreview(null);
            }
          }}
        >
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video
            ref={videoRef}
            src={assetProxyUrl(asset.id)}
            className="h-full w-auto"
            style={{ aspectRatio: `${asset.width} / ${asset.height}` }}
            preload="metadata"
            playsInline
            onLoadedData={() => setVideoReady(true)}
            onCanPlay={() => setVideoReady(true)}
            onTimeUpdate={(ev) => {
              const t = ev.currentTarget.currentTime;
              setTime(t);
              onTimeUpdateRef.current?.(t);
            }}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onClick={() => {
              const v = videoRef.current;
              if (!v) return;
              if (selectedOp) {
                setSelectedOp(null); // clicking the surface deselects, Premiere-style
                return;
              }
              v.paused ? v.play() : v.pause();
            }}
          />
          {/* poster overlay — the real first scrub frame until the video decodes */}
          {!videoReady && (
            <div
              className="pointer-events-none absolute inset-0 flex items-center justify-center"
              aria-hidden
            >
              <div
                className="h-full w-full"
                style={scrubBackground(
                  scrubFrameIndex(0, asset.duration_s),
                  scrubFrameCount(asset.duration_s),
                  assetScrubUrl(asset.id),
                )}
              />
            </div>
          )}
          {nle.inS !== null && (
            <span
              className="pointer-events-none absolute inset-y-1 border-l-2 border-[var(--ok)]"
              title={`in ${nle.inS.toFixed(2)}s`}
              style={{ left: `${Math.min(100, (nle.inS / asset.duration_s) * 100)}%` }}
            />
          )}
          {nle.outS !== null && (
            <span
              className="pointer-events-none absolute inset-y-1 border-r-2 border-[var(--acc)]"
              title={`out ${nle.outS.toFixed(2)}s`}
              style={{ left: `${Math.min(100, (nle.outS / asset.duration_s) * 100)}%` }}
            />
          )}
          {nle.splits.map((t) => (
            <span
              key={t}
              title={`split ${t.toFixed(2)}s`}
              className="pointer-events-none absolute top-1 h-2.5 w-0.5 bg-[var(--warn)]"
              style={{ left: `${Math.min(100, (t / asset.duration_s) * 100)}%` }}
            />
          ))}
          {showExisting &&
            scaleX > 0 &&
            regions.map((r) => {
              // tracked selections draw the box AT THE PLAYHEAD (it follows
              // the object as you scrub); static boxes draw their stored rect
              // — or the LIVE adjust preview while one is being dragged
              const path = r.op_id ? trackedPaths[r.op_id] : undefined;
              const tb: TrackedBox | null =
                r.mode === "tracked" && path && path.length > 0 ? trackedBoxAt(path, time) : null;
              const isDragged = adjustDrag?.opId === r.op_id && !!preview;
              const bx = isDragged ? preview!.x : tb ? tb.x : r.x;
              const by = isDragged ? preview!.y : tb ? tb.y : r.y;
              const bw = isDragged ? preview!.w : tb ? tb.w : r.w;
              const bh = isDragged ? preview!.h : tb ? tb.h : r.h;
              if (r.mode === "tracked" && !tb && path && path.length > 0) return null; // outside the tracked span
              const adjustable = !!r.op_id && r.mode !== "tracked" && !drawMode;
              const isSelected = selectedOp !== null && selectedOp === r.op_id;
              return (
                <div
                  key={r.id}
                  className={`absolute border bg-[rgba(241,109,109,0.14)] ${adjustable ? "cursor-move" : "pointer-events-none"}`}
                  style={{
                    left: bx * scaleX,
                    top: by * scaleY,
                    width: bw * scaleX,
                    height: bh * scaleY,
                    borderStyle:
                      isSelected || isDragged
                        ? "solid"
                        : r.origin === "agent" || tb?.interpolated
                          ? "dashed"
                          : "solid",
                    borderColor:
                      isSelected || isDragged
                        ? "var(--acc)"
                        : r.mode === "tracked"
                          ? "var(--warn, #e2a336)"
                          : "var(--err)",
                    boxShadow: isSelected || isDragged ? "0 0 0 1px var(--acc)" : undefined,
                  }}
                  onMouseDown={(ev) => {
                    if (adjustable) startAdjust(r, "move", ev);
                  }}
                  title={
                    r.mode === "tracked"
                    ? `${r.label || "tracked"} · follows object @ playhead ${bx}x${by}`
                    : `${r.label || "blur"} ${r.w}x${r.h}@${r.x},${r.y} · ${r.origin ?? "legacy"}${adjustable ? " · drag to move, corner to resize" : ""}`
                  }
                >
                  {adjustable && isSelected && (
                    <span
                      className="absolute -bottom-1.5 -right-1.5 h-3 w-3 cursor-nwse-resize rounded-sm border border-[var(--acc)] bg-[var(--bg-2)]"
                      title="drag to resize (release lands the corrected op)"
                      onMouseDown={(ev) => startAdjust(r, "resize", ev)}
                    />
                  )}
                </div>
              );
            })}
          {draft && (
            <div
              className="pointer-events-none absolute border border-dashed border-[var(--acc)] bg-[var(--acc-bg)]"
              style={{
                left: Math.min(draft.x0, draft.x1),
                top: Math.min(draft.y0, draft.y1),
                width: Math.abs(draft.x1 - draft.x0),
                height: Math.abs(draft.y1 - draft.y0),
              }}
            />
          )}
        </div>
      </div>

      {/* ---- transport bar ---- */}
      <div className="flex h-11 shrink-0 items-center gap-1 border-t border-[var(--line-1)] bg-[var(--bg-2)] px-2.5">
        <button
          className="icon-btn"
          title="go to start (Home)"
          onClick={transport((v) => {
            v.currentTime = 0;
            onTimeUpdate(0);
          })}
        >
          <Icon name="goStart" />
        </button>
        <button
          className="icon-btn"
          title="step back one frame (←)"
          onClick={transport((v) => {
            v.pause();
            v.currentTime = Math.max(0, v.currentTime - 1 / fps);
          })}
        >
          <Icon name="stepBack" />
        </button>
        <button
          className="icon-btn !h-8 !w-8 rounded-full border border-[var(--line-2)] text-[var(--tx-1)] hover:border-[var(--acc)] hover:text-[var(--acc)]"
          title="play / pause (Space)"
          onClick={transport((v) => (v.paused ? v.play() : v.pause()))}
        >
          <Icon name={playing ? "pause" : "play"} size={18} />
        </button>
        <button
          className="icon-btn"
          title="step forward one frame (→)"
          onClick={transport((v) => {
            v.pause();
            v.currentTime = Math.min(asset.duration_s, v.currentTime + 1 / fps);
          })}
        >
          <Icon name="stepFwd" />
        </button>
        <button
          className="icon-btn"
          title="go to end (End)"
          onClick={transport((v) => {
            v.currentTime = asset.duration_s;
          })}
        >
          <Icon name="goEnd" />
        </button>
        <span className="tcode tcode-hero ml-2">
          {formatTimecodeFrames(time, fps)}
        </span>
        <span className="tcode text-[12px] text-[var(--tx-3)]">
          / {formatTimecodeFrames(asset.duration_s, fps)}
        </span>
        <div className="flex-1" />
        <span className="chip" title="source resolution and frame rate">
          {asset.width}×{asset.height} · {fps.toFixed(2)} fps
        </span>
      </div>

      {/* ---- blur region list (contextual, under transport) ---- */}
      {(regions.length > 0 || drawMode) && (
        <div className="max-h-24 shrink-0 overflow-y-auto border-t border-[var(--line-1)] px-2.5 py-1.5">
          {drawMode && (
            <p className="text-[11px] text-[var(--tx-2)]">
              Draw mode on — drag a box over a face or watermark; it rides the
              raw footage at render time.
            </p>
          )}
          {regions.map((r) => (
            <div
              key={r.id}
              data-selected={r.op_id !== null && selectedOp === r.op_id}
              className="group flex cursor-pointer items-center justify-between rounded px-1 py-0.5 hover:bg-[var(--bg-2)] data-[selected=true]:bg-[var(--acc-bg)]"
              title={r.op_id && r.mode !== "tracked" ? "click to select — drag the box on the video or nudge with Alt+arrows" : undefined}
              onClick={() => setSelectedOp((s) => (r.op_id && s !== r.op_id ? r.op_id : null))}
            >
              <span className="tcode text-[11px] text-[var(--tx-2)]">
                {r.label || "blur"} · {r.w}×{r.h} @ {r.x},{r.y}
                {r.origin === "agent" && (
                  <span className="ml-1.5 rounded bg-[var(--acc-bg)] px-1 text-[9px] uppercase tracking-wider text-[var(--acc)]">
                    agent
                  </span>
                )}
                {r.mode === "tracked" && r.op_id && trackedPaths[r.op_id]?.length > 0 && (
                  <span
                    className="ml-1.5 rounded bg-[var(--bg-3, #26292f)] px-1 text-[9px] uppercase tracking-wider text-[var(--warn, #e2a336)]"
                    title={`per-frame path · ${trackedPaths[r.op_id].length} frames`}
                  >
                    tracked · {trackedPaths[r.op_id].length}f
                  </span>
                )}
              </span>
              <span className="flex items-center gap-0.5">
                {r.op_id && r.mode !== "tracked" && (
                  <button
                    className="icon-btn !h-5 !w-5 opacity-0 group-hover:opacity-100"
                    disabled={tracking === r.op_id}
                    title="Track this object from the playhead — the blur follows it across frames (selection service)"
                    onClick={() => trackRegion(r)}
                  >
                    <Icon name="wand" size={12} />
                  </button>
                )}
                <button
                  className="icon-btn !h-5 !w-5 opacity-0 group-hover:opacity-100"
                  title="remove blur box (op-backed boxes are revoked — history kept)"
                  onClick={() => removeRegion(r)}
                >
                  <Icon name="trash" size={12} />
                </button>
              </span>
            </div>
          ))}
          {error && <p className="text-[11px] text-[var(--err)]">{error}</p>}
        </div>
      )}

      {/* keyboard legend */}
      <div className="flex h-6 shrink-0 items-center gap-3 border-t border-[var(--line-1)] px-2.5 text-[10px] text-[var(--tx-3)]">
        <span>Space play</span>
        <span>J·K·L shuttle</span>
        <span>I / O mark in-out</span>
        <span>←/→ frame step</span>
        <span>S split at playhead</span>
        <span>Ctrl+Z undo</span>
        <span title="with a box selected: drag to move, corner to resize, Alt+arrows nudge (Shift = 10 px), Esc deselect">Alt+←↑↓→ nudge box</span>
      </div>
    </div>
  );
});

export function MonitorEmpty({ onRun }: { onRun?: () => void }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 bg-black">
      <span className="flex h-14 w-14 items-center justify-center rounded-xl bg-[var(--bg-2)] text-[var(--tx-3)]">
        <Icon name="media" size={26} />
      </span>
      <div className="text-center">
        <p className="text-[13px] font-medium text-[var(--tx-1)]">
          No program yet
        </p>
        <p className="mt-1 max-w-[300px] text-[11px] leading-relaxed text-[var(--tx-3)]">
          Ingest builds a 480p scrub proxy automatically. Run the pipeline and
          the monitor comes alive here.
        </p>
      </div>
      {onRun && (
        <button className="btn" onClick={onRun}>
          <Icon name="play" size={12} /> Run pipeline
        </button>
      )}
    </div>
  );
}
