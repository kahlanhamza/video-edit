"use client";

// Proxy player with a mouse-drawn blur canvas (M3 acceptance: "draw a blur box
// with your mouse over a corner watermark"). Boxes are drawn in display space,
// stored in RAW-FOOTAGE pixel space, and ride every raw segment at render time.
// Draw mode is a toggle so native scrubbing keeps working when it is off.

import { useCallback, useEffect, useRef, useState } from "react";
import { Asset, Redaction, api, assetProxyUrl } from "@/lib/api";
import { useNLEShortcuts } from "@/hooks/useNLEShortcuts";

interface Draft {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export default function ProxyPlayer({
  projectId,
  asset,
  onChanged,
  onTimeUpdate,
}: {
  projectId: string;
  asset: Asset;
  onChanged: () => void;
  /** live playhead seconds — the studio timeline follows the video */
  onTimeUpdate?: (t: number) => void;
}) {
  const [regions, setRegions] = useState<Redaction[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showExisting, setShowExisting] = useState(true);
  const [drawMode, setDrawMode] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Studio NLE shortcuts (issue #19): Space/J/K/L/I/O/←/→/S/Ctrl+Z.
  // Disabled while a blur box is being drawn — the mouse owns the surface then.
  const { state: nle } = useNLEShortcuts({ videoRef, enabled: !drawMode });

  // playhead feed for the studio timeline — a no-op when no consumer passed
  const onTimeUpdateRef = useRef(onTimeUpdate);
  onTimeUpdateRef.current = onTimeUpdate;
  const handleTimeUpdate = useCallback(
    (ev: React.SyntheticEvent<HTMLVideoElement>) => {
      onTimeUpdateRef.current?.(ev.currentTarget.currentTime);
    },
    [],
  );

  const reload = useCallback(async () => {
    try {
      setRegions(await api.listRedactions(projectId));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
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
    if (!draft) return;
    const p = toDisplayCoords(ev);
    setDraft({ ...draft, x1: p.x, y1: p.y });
  }

  async function onMouseUp() {
    if (!draft) return;
    const dx = Math.abs(draft.x1 - draft.x0);
    const dy = Math.abs(draft.y1 - draft.y0);
    setDraft(null);
    if (dx < 8 || dy < 8) return; // click, not a drag
    const rect = wrapRef.current!.getBoundingClientRect();
    // display space -> raw-footage pixel space
    const sx = asset.width / rect.width;
    const sy = asset.height / rect.height;
    const x = Math.round(Math.min(draft.x0, draft.x1) * sx);
    const y = Math.round(Math.min(draft.y0, draft.y1) * sy);
    const w = Math.max(1, Math.round(dx * sx));
    const h = Math.max(1, Math.round(dy * sy));
    try {
      await api.addRedaction(projectId, { x, y, w, h });
      await reload();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function removeRegion(id: string) {
    try {
      await api.deleteRedaction(projectId, id);
      await reload();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const rect = wrapRef.current?.getBoundingClientRect();
  const scaleX = rect ? rect.width / asset.width : 0;
  const scaleY = rect ? rect.height / asset.height : 0;

  return (
    <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
      <div
        ref={wrapRef}
        className={`relative w-full select-none rounded bg-black ${drawMode ? "cursor-crosshair" : ""}`}
        onMouseDown={drawMode ? onMouseDown : undefined}
        onMouseMove={drawMode ? onMouseMove : undefined}
        onMouseUp={drawMode ? onMouseUp : undefined}
      >
        <video
          ref={videoRef}
          src={assetProxyUrl(asset.id)}
          controls
          className="w-full"
          preload="metadata"
          onTimeUpdate={handleTimeUpdate}
        />
        {nle.inS !== null && (
          <span
            className="pointer-events-none absolute top-1 bottom-1 border-l-2 border-emerald-400"
            title={`in ${nle.inS.toFixed(2)}s`}
            style={{ left: `${Math.min(100, (nle.inS / asset.duration_s) * 100)}%` }}
          />
        )}
        {nle.outS !== null && (
          <span
            className="pointer-events-none absolute top-1 bottom-1 border-r-2 border-sky-400"
            title={`out ${nle.outS.toFixed(2)}s`}
            style={{ left: `${Math.min(100, (nle.outS / asset.duration_s) * 100)}%` }}
          />
        )}
        {nle.splits.map((t) => (
          <span
            key={t}
            title={`split ${t.toFixed(2)}s`}
            className="pointer-events-none absolute top-1 h-2 w-0.5 bg-amber-400"
            style={{ left: `${Math.min(100, (t / asset.duration_s) * 100)}%` }}
          />
        ))}
        {showExisting &&
          scaleX > 0 &&
          regions.map((r) => (
            <div
              key={r.id}
              className="pointer-events-none absolute border-2 border-red-500/80 bg-red-500/15"
              style={{
                left: r.x * scaleX,
                top: r.y * scaleY,
                width: r.w * scaleX,
                height: r.h * scaleY,
              }}
              title={`${r.label || "blur"} ${r.w}x${r.h}@${r.x},${r.y}`}
            />
          ))}
        {draft && (
          <div
            className="pointer-events-none absolute border-2 border-dashed border-amber-400 bg-amber-400/10"
            style={{
              left: Math.min(draft.x0, draft.x1),
              top: Math.min(draft.y0, draft.y1),
              width: Math.abs(draft.x1 - draft.x0),
              height: Math.abs(draft.y1 - draft.y0),
            }}
          />
        )}
      </div>

      <div className="space-y-2 text-sm">
        <button
          className={`btn-ghost w-full ${drawMode ? "!bg-amber-500/20 text-amber-300" : ""}`}
          onClick={() => {
            setDrawMode(!drawMode);
            setDraft(null);
          }}
        >
          {drawMode ? "stop drawing" : "draw blur box"}
        </button>
        <p className="text-xs text-slate-500">
          {drawMode
            ? "drag on the video to box a watermark / face"
            : `enable draw mode to place a blur box — raw-frame coords ${asset.width}x${asset.height}`}
        </p>
        <label className="flex items-center gap-2 text-xs text-slate-400">
          <input
            type="checkbox"
            checked={showExisting}
            onChange={(e) => setShowExisting(e.target.checked)}
          />
          show {regions.length} blur box(es)
        </label>
        <p className="rounded bg-slate-900/60 p-2 text-[11px] leading-4 text-slate-500">
          <span className="text-slate-300">NLE keys</span> — space play · J/K/L shuttle ·
          I/O in/out · ←/→ frame step · S split · Ctrl+Z undo
        </p>
        {error && <p className="text-xs text-red-400">{error}</p>}
        <ul className="space-y-1">
          {regions.map((r) => (
            <li
              key={r.id}
              className="flex items-center justify-between rounded bg-slate-900 px-2 py-1 text-xs text-slate-300"
            >
              <span className="truncate">
                {r.label || "blur"} · {r.w}x{r.h} @ {r.x},{r.y}
              </span>
              <button
                className="text-red-400 hover:text-red-300"
                onClick={() => removeRegion(r.id)}
              >
                del
              </button>
            </li>
          ))}
          {regions.length === 0 && <li className="text-xs text-slate-600">no blur boxes yet</li>}
        </ul>
      </div>
    </div>
  );
}
