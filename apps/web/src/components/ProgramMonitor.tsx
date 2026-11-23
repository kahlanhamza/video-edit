"use client";

// ProgramMonitor — the NLE program viewer. TWO clocks, one truth:
//
//   RAW mode     the 480p proxy plays like a source viewer (pre-pipeline).
//   FOLLOW mode  (default once a cut exists) the proxy is driven BY THE
//                TIMELINE: program seconds map into each master clip's
//                source window, narration blocks ride their muted b-roll
//                windows (the renderer's cycle, mirrored frame-true in
//                lib/nle.brollSourceAt) with the synthesized TTS stem on a
//                parallel <audio> element, captions burn in from the cue
//                store, and clip boundaries hard-cut exactly like the render.
//
// The preview must never lie about the cut (studio acceptance): a monitor
// that plays the raw 43-minute source while the timeline says otherwise is
// worse than no monitor. Blur boxes stay RAW-frame addressed (they ride the
// source footage), so they keep using the video element's raw clock.
//
// Keyboard: Space play · J/K/L shuttle · I/O in-out · arrows frame-step
// (program time in follow mode) · S split at playhead · Ctrl+Z undo op.

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Asset,
  NleCaption,
  NleEffect,
  NleTimeline,
  OpRecord,
  SelectionOut,
  TrackHit,
  api,
  assetProxyUrl,
  assetScrubUrl,
  narrationClipUrl,
} from "@/lib/api";
import {
  AdjustDrag,
  TrackedBox,
  adjustedBox,
  brollSourceAt,
  captionAt,
  clipAfter,
  clipAt,
  formatTimecodeFrames,
  programClips,
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
    /** The op-log rev the page last saw — when it advances (agent op, another
     * tab, an undo elsewhere) the monitor refetches the fold so blur boxes
     * converge LIVE, not just the timeline. */
    opsRev?: number | null;
    /** The edited timeline — powers FOLLOW mode (timeline-true preview). */
    nle?: NleTimeline | null;
    onTimeUpdate: (t: number) => void;
    onChanged: () => void;
  }
>(function ProgramMonitor(
  { projectId, asset, fps, opsRev, nle, onTimeUpdate, onChanged },
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
  // `time` is the DISPLAY clock: program seconds in follow mode, raw source
  // seconds otherwise. `rawT` is always the video element's source clock —
  // blur boxes and tracked paths are raw-footage addressed.
  const [time, setTime] = useState(0);
  const [rawT, setRawT] = useState(0);
  const [followOn, setFollowOn] = useState(true);
  const [videoReady, setVideoReady] = useState(false);
  const [tracking, setTracking] = useState<string | null>(null);
  const [trackedPaths, setTrackedPaths] = useState<Record<string, TrackHit[]>>({});
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const pClips = useMemo(() => programClips(nle), [nle]);
  const hasProgram = pClips.length > 0;
  const follow = followOn && hasProgram;
  const programDuration = follow ? (nle?.duration_s ?? asset.duration_s) : asset.duration_s;

  // latest-ref mirrors for the video element event handlers (they must not
  // re-bind on every clock tick)
  const pClipsRef = useRef(pClips);
  pClipsRef.current = pClips;
  const followRef = useRef(follow);
  followRef.current = follow;
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const timeRef = useRef(time);
  timeRef.current = time;
  const rawTRef = useRef(rawT);
  rawTRef.current = rawT;
  const progDurRef = useRef(programDuration);
  progDurRef.current = programDuration;
  const activeClipRef = useRef<number>(-1); // index into pClips, -1 = gap
  const onTimeUpdateRef = useRef(onTimeUpdate);
  onTimeUpdateRef.current = onTimeUpdate;

  /** Map a program time into the proxy video (follow) or straight (raw). */
  const seek = useCallback(
    (t: number) => {
      const v = videoRef.current;
      const clamped = Math.min(Math.max(0, t), progDurRef.current);
      setTime(clamped);
      timeRef.current = clamped;
      if (!followRef.current) {
        if (v) v.currentTime = clamped;
        setRawT(clamped);
        return;
      }
      const clips = pClipsRef.current;
      const c = clipAt(clips, clamped);
      if (v) {
        if (c) {
          const src = brollSourceAt(c, clamped) ?? c.source_start_s ?? 0;
          v.currentTime = Math.min(Math.max(0, src), Math.max(0, asset.duration_s - 0.05));
          activeClipRef.current = clips.indexOf(c);
        } else {
          activeClipRef.current = -1; // gap: freeze the tape, black overlay shows
        }
      }
      syncAudio(clamped);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [asset.duration_s, projectId],
  );

  useImperativeHandle(ref, () => ({ seek }));

  /** FOLLOW ENGINE — the program clock owns playback (rAF ticker).
   *
   * The video element is a commanded display device: each tick advances the
   * PROGRAM clock at the tape's playbackRate, drives the proxy to the
   * b-roll-cycled source position for that program second, hard-cuts at clip
   * boundaries (exactly like the render), skips gaps, and re-syncs the TTS
   * bus. Deriving the program time from video timeupdate instead breaks on
   * multi-window b-roll: the tape plays OUTSIDE the clip's own source range
   * (windows [1..N] live anywhere in the source), so there is no linear map
   * from tape seconds to program seconds — the clock must lead, not follow. */
  useEffect(() => {
    if (!follow || !playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const v = videoRef.current;
      if (v) {
        const rate = v.playbackRate || 1;
        let pt = timeRef.current + ((now - last) / 1000) * rate;
        const clips = pClipsRef.current;
        let c = activeClipRef.current >= 0 ? clips[activeClipRef.current] : null;
        if (!c || pt >= c.end_s - 1e-6 || pt < c.start_s - 1e-6) {
          if (rate < 0 && c) {
            // reverse shuttle past this clip's head: land on the PREVIOUS
            // clip's out-point (or the program start when none is left)
            const prevIdx = activeClipRef.current - 1;
            const prev = prevIdx >= 0 ? clips[prevIdx] : null;
            if (!prev) {
              pt = 0;
              timeRef.current = pt;
              setTime(pt);
              onTimeUpdateRef.current?.(pt);
              v.pause();
              return;
            }
            c = prev;
            activeClipRef.current = prevIdx;
            pt = Math.min(pt, prev.end_s);
          } else {
            // forward boundary (or resuming in a gap): next clip at/after here
            const next = clipAfter(clips, c ? c.end_s : pt);
            if (!next) {
              pt = progDurRef.current;
              timeRef.current = pt;
              setTime(pt);
              onTimeUpdateRef.current?.(pt);
              v.pause();
              return; // program over — ticker dies with the pause
            }
            c = next;
            activeClipRef.current = clips.indexOf(next);
            pt = Math.max(pt, next.start_s);
          }
        }
        timeRef.current = pt;
        setTime(pt);
        onTimeUpdateRef.current?.(pt);
        const expected = brollSourceAt(c, pt) ?? c.source_start_s ?? 0;
        if (Math.abs(v.currentTime - expected) > 0.35) {
          v.currentTime = expected; // hard cut / drift re-sync
        }
        syncAudio(pt);
      }
      last = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [follow, playing]);

  /** Keep the narration TTS stem glued to the program clock. Called on every
   * seek and every follow tick — cheap (one clipAt walk) and self-correcting
   * (drift > 180 ms re-seeks the audio element). */
  function syncAudio(pt: number) {
    const a = audioRef.current;
    const v = videoRef.current;
    if (!a || !v) return;
    if (!followRef.current) {
      a.pause();
      v.volume = 1;
      return;
    }
    const c = clipAt(pClipsRef.current, pt);
    if (c && c.kind === "narration" && c.narration_key) {
      const url = narrationClipUrl(projectId, c.narration_key);
      if (!a.src.endsWith(url)) a.src = url;
      const voIn = c.vo_in_s ?? 0;
      const voOut = c.vo_out_s ?? c.tts_duration_s ?? null;
      let target = voIn + (pt - c.start_s);
      if (voOut != null) target = Math.min(target, voOut);
      target = Math.max(0, target);
      if (a.readyState >= 1 && Math.abs(a.currentTime - target) > 0.18) {
        a.currentTime = target;
      }
      a.playbackRate = v.playbackRate || 1;
      if (playingRef.current) void a.play().catch(() => {});
      else a.pause();
      v.volume = Math.min(1, Math.max(0, c.broll_audio_volume ?? 0));
    } else {
      a.pause();
      v.volume = c && c.audio_mode === "silence" ? 0 : 1;
    }
  }

  const { state: markers } = useNLEShortcuts({
    videoRef,
    fps,
    enabled: !drawMode,
    onUndo: undoNewestOp,
    timeSource: () =>
      followRef.current ? timeRef.current : (videoRef.current?.currentTime ?? timeRef.current),
    seekTo: (t) => {
      const v = videoRef.current;
      if (v) v.pause();
      seek(t);
    },
    maxTime: programDuration,
  });

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
    // fires on mount, project change, and every op-log advance the page
    // reports (SSE → reloadNle → ops_rev) — the monitor converges live
  }, [reload, opsRev]);

  // The cut changed (dispatch, agent run, undo): re-anchor the tape to the
  // SAME program second so the monitor instantly reflects the new edit.
  useEffect(() => {
    if (follow) seek(timeRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pClips]);

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

  /** Ctrl+Z = undo the NEWEST op (revoke — history kept). The fold sorts by
   *  rev, so repeated undos naturally walk back one op at a time; returns
   *  false when no op-backed box exists, falling through to marker undo. */
  function undoNewestOp(): boolean {
    const latest = regions.reduce<NleEffect | null>((acc, r) => {
      if (!r.op_id) return acc;
      if (!acc || (r.op_rev ?? 0) > (acc.op_rev ?? 0)) return r;
      return acc;
    }, null);
    if (!latest?.op_id) return false;
    void api
      .revokeOp(projectId, latest.op_id, "undo (Ctrl+Z)")
      .then(async () => {
        await reload();
        onChanged();
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    return true;
  }

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
    const anchor = Math.min(Math.max(0, Math.round(rawTRef.current * fps)), Math.max(0, Math.round(asset.duration_s * fps) - 1));
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

  const activeClip = follow ? clipAt(pClips, time) : null;
  const inGap = follow && activeClip === null;
  const cue: NleCaption | null = follow ? captionAt(nle?.captions ?? [], time) : null;
  const narrationLive = activeClip?.kind === "narration";

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* ---- viewer toolbar ---- */}
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-[var(--line-1)] bg-[var(--bg-2)] px-2.5">
        <span className="micro-label">Program</span>
        {hasProgram && (
          <button
            className="chip"
            data-active={follow}
            title={
              follow
                ? "FOLLOW mode: the preview plays the TIMELINE — clip cuts, narration b-roll, captions. Click to scrub the raw source instead."
                : "RAW mode: the preview scrubs the raw source. Click to follow the edited timeline."
            }
            onClick={() => setFollowOn((v) => !v)}
            style={follow ? { background: "var(--acc-bg)", color: "var(--acc)" } : undefined}
          >
            {follow ? "following timeline" : "raw source"}
          </button>
        )}
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
              // raw clock only — blur boxes + tracked paths are raw-footage
              // addressed; the FOLLOW program clock lives in the rAF engine
              const r = ev.currentTarget.currentTime;
              setRawT(r);
              rawTRef.current = r;
              if (!followRef.current) {
                setTime(r);
                timeRef.current = r;
                onTimeUpdateRef.current?.(r);
              }
            }}
            onPlay={() => {
              setPlaying(true);
              playingRef.current = true;
              syncAudio(timeRef.current);
            }}
            onPause={() => {
              setPlaying(false);
              playingRef.current = false;
              syncAudio(timeRef.current);
            }}
            onRateChange={(ev) => {
              if (audioRef.current) {
                audioRef.current.playbackRate = ev.currentTarget.playbackRate || 1;
              }
            }}
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
          {/* the narration TTS bus — a parallel element the engine keeps
              sample-glued to the program clock (muted b-roll on the video) */}
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <audio ref={audioRef} preload="auto" className="hidden" />
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
          {/* gap overlay: follow mode outside any clip — the render is black here */}
          {follow && inGap && (
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-1 bg-black/92">
              <span className="text-[12px] font-medium text-[var(--tx-2)]">gap</span>
              <span className="text-[10px] text-[var(--tx-3)]">
                nothing on V1 here — the render is black
              </span>
            </div>
          )}
          {/* narration badge: which bus is live */}
          {follow && narrationLive && (
            <div className="pointer-events-none absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/65 px-2 py-1 text-[10px] font-medium text-[var(--ok)] backdrop-blur-sm">
              <span className="h-1.5 w-1.5 rounded-full bg-[var(--ok)]" />
              narration VO · b-roll muted
            </div>
          )}
          {/* captions (follow mode): the cue store is the baked truth */}
          {follow && cue && (
            <div className="pointer-events-none absolute inset-x-3 bottom-2.5 flex justify-center">
              <p
                className="max-w-[92%] rounded bg-black/70 px-2.5 py-1 text-center text-[13px] font-medium leading-snug text-white [text-shadow:0_1px_2px_rgba(0,0,0,.9)]"
                title={`cue ${cue.index}${cue.speaker ? ` · ${cue.speaker}` : ""}`}
              >
                {cue.speaker && (
                  <span className="mr-1.5 text-[10px] uppercase tracking-wider text-[var(--acc)]">
                    {cue.speaker}
                  </span>
                )}
                {cue.text}
              </p>
            </div>
          )}
          {markers.inS !== null && (
            <span
              className="pointer-events-none absolute inset-y-1 border-l-2 border-[var(--ok)]"
              title={`in ${markers.inS.toFixed(2)}s`}
              style={{ left: `${Math.min(100, (markers.inS / programDuration) * 100)}%` }}
            />
          )}
          {markers.outS !== null && (
            <span
              className="pointer-events-none absolute inset-y-1 border-r-2 border-[var(--acc)]"
              title={`out ${markers.outS.toFixed(2)}s`}
              style={{ left: `${Math.min(100, (markers.outS / programDuration) * 100)}%` }}
            />
          )}
          {markers.splits.map((t) => (
            <span
              key={t}
              title={`split ${t.toFixed(2)}s`}
              className="pointer-events-none absolute top-1 h-2.5 w-0.5 bg-[var(--warn)]"
              style={{ left: `${Math.min(100, (t / programDuration) * 100)}%` }}
            />
          ))}
          {showExisting &&
            scaleX > 0 &&
            regions.map((r) => {
              // tracked selections draw the box AT THE PLAYHEAD (it follows
              // the object as you scrub); static boxes draw their stored rect
              // — or the LIVE adjust preview while one is being dragged.
              // Raw-footage clock: rawT, never the program clock.
              const path = r.op_id ? trackedPaths[r.op_id] : undefined;
              const tb: TrackedBox | null =
                r.mode === "tracked" && path && path.length > 0 ? trackedBoxAt(path, rawT) : null;
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
          onClick={() => {
            videoRef.current?.pause();
            seek(0);
          }}
        >
          <Icon name="goStart" />
        </button>
        <button
          className="icon-btn"
          title="step back one frame (←)"
          onClick={() => {
            videoRef.current?.pause();
            seek(timeRef.current - 1 / fps);
          }}
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
          onClick={() => {
            videoRef.current?.pause();
            seek(timeRef.current + 1 / fps);
          }}
        >
          <Icon name="stepFwd" />
        </button>
        <button
          className="icon-btn"
          title="go to end (End)"
          onClick={() => {
            videoRef.current?.pause();
            seek(programDuration);
          }}
        >
          <Icon name="goEnd" />
        </button>
        <span className="tcode tcode-hero ml-2">
          {formatTimecodeFrames(time, fps)}
        </span>
        <span className="tcode text-[12px] text-[var(--tx-3)]">
          / {formatTimecodeFrames(programDuration, fps)}
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
        <span title="revokes the newest op on the log (history kept); falls back to in/out marker undo">Ctrl+Z undo op</span>
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
