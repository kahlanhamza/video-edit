"use client";

// useNLEShortcuts — Studio NLE keyboard control (issue #19).
// Maps the standard shortcuts onto a <video> element and the pure
// nleReducer: Space play/pause, J/K/L shuttle, I/O in/out, ←/→ frame step,
// S split, Ctrl+Z undo. Reverse shuttle is emulated with a rAF scrub loop
// (HTML5 video cannot play backwards natively). Typing in inputs never
// triggers transport.

import { useEffect, useReducer, useRef, useState } from "react";
import { initialNLEState, nleReducer } from "@/lib/nle_reducer";

export interface UseNLEShortcutsOptions {
  /** The controlled <video> element. */
  videoRef: React.RefObject<HTMLVideoElement | null>;
  /** Frame step for ←/→ (media fps, default 30). */
  fps?: number;
  /** Fires on S with the split time (seconds). */
  onSplit?: (t: number) => void;
  /** Fires whenever in/out changes (either null = not set). */
  onInOut?: (inS: number | null, outS: number | null) => void;
  /** False disables the key listener entirely (e.g. while drawing a blur box). */
  enabled?: boolean;
  /** Ctrl+Z: return true when a SERVER-side undo consumed the event (op
   * revoke); false falls through to the local marker undo. Latest-callback
   * ref — safe to pass an inline closure. */
  onUndo?: () => boolean;
}

const EDITABLE = new Set(["INPUT", "TEXTAREA", "SELECT"]);

function isTyping(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  return EDITABLE.has(el.tagName) || el.isContentEditable;
}

export function useNLEShortcuts({
  videoRef,
  fps = 30,
  onSplit,
  onInOut,
  enabled = true,
  onUndo,
}: UseNLEShortcutsOptions) {
  const [state, dispatch] = useReducer(nleReducer, initialNLEState);
  const [active, setActive] = useState(false);
  const rafRef = useRef<number | null>(null);
  // Latest-callback refs so the key handler never goes stale.
  const cbs = useRef({ onSplit, onInOut, onUndo });
  cbs.current = { onSplit, onInOut, onUndo };

  const lastMarkers = useRef<{ inS: number | null; outS: number | null }>({
    inS: state.inS,
    outS: state.outS,
  });

  // Report in/out changes upward once per change.
  useEffect(() => {
    if (state.inS !== lastMarkers.current.inS || state.outS !== lastMarkers.current.outS) {
      lastMarkers.current = { inS: state.inS, outS: state.outS };
      cbs.current.onInOut?.(state.inS, state.outS);
    }
  }, [state.inS, state.outS]);

  // Follow native pause/play (user clicked the element's own controls, video ended).
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const sync = () => dispatch({ type: "syncPlaying", playing: !v.paused });
    v.addEventListener("pause", sync);
    v.addEventListener("play", sync);
    return () => {
      v.removeEventListener("pause", sync);
      v.removeEventListener("play", sync);
    };
  }, [videoRef]);

  // Transport application: forward speed = native playbackRate; reverse =
  // rAF scrub loop (HTML5 video cannot play backwards natively).
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (!state.playing || state.rate === 0) {
      if (!v.paused) v.pause();
      v.playbackRate = 1;
      return;
    }

    if (state.rate > 0) {
      v.playbackRate = state.rate;
      if (v.paused) void v.play().catch(() => {}); // autoplay policy — stay paused silently
      return;
    }

    // Reverse scrub at |rate|x.
    v.playbackRate = 1;
    if (!v.paused) v.pause();
    let last = performance.now();
    const step = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      v.currentTime = Math.max(0, v.currentTime - Math.abs(state.rate) * dt);
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [state.playing, state.rate, videoRef]);

  // Keyboard binding.
  useEffect(() => {
    if (!enabled) {
      setActive(false);
      return;
    }

    const applyMarkers = (kind: "in" | "out" | "split") => {
      const v = videoRef.current;
      const time = v ? v.currentTime : 0;
      if (kind === "in") dispatch({ type: "markIn", t: time });
      else if (kind === "out") dispatch({ type: "markOut", t: time });
      else {
        dispatch({ type: "split", t: time });
        cbs.current.onSplit?.(time);
      }
    };

    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target)) return;
      const v = videoRef.current;
      const step = 1 / Math.max(1, fps);

      switch (e.code) {
        case "Space":
          e.preventDefault();
          dispatch({ type: "togglePlay" });
          break;
        case "KeyJ":
          e.preventDefault();
          dispatch({ type: "shuttle", dir: -1 });
          break;
        case "KeyK":
          e.preventDefault();
          dispatch({ type: "stop" });
          break;
        case "KeyL":
          e.preventDefault();
          dispatch({ type: "shuttle", dir: 1 });
          break;
        case "KeyI":
          e.preventDefault();
          applyMarkers("in");
          break;
        case "KeyO":
          e.preventDefault();
          applyMarkers("out");
          break;
        case "KeyS":
          e.preventDefault();
          applyMarkers("split");
          break;
        case "ArrowLeft":
        case "ArrowRight":
          e.preventDefault();
          if (v) {
            v.playbackRate = 1;
            const delta = e.code === "ArrowLeft" ? -step : step;
            v.currentTime = Math.min(
              Math.max(0, v.currentTime + delta),
              v.duration || Number.MAX_SAFE_INTEGER,
            );
          }
          break;
        case "KeyZ":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            // a server-side op undo (revoke newest op) wins when available;
            // the local marker stack is the fallback, never a double action
            if (cbs.current.onUndo?.()) return;
            dispatch({ type: "undo" });
          }
          break;
        default:
          break;
      }
    };

    window.addEventListener("keydown", onKey);
    setActive(true);
    return () => {
      window.removeEventListener("keydown", onKey);
      setActive(false);
    };
  }, [enabled, fps, videoRef]);

  return { state, dispatch, active };
}
