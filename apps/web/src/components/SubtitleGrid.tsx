"use client";

// Subtitle grid editor (M3 acceptance: "edit a subtitle line in the browser").
// Saving a row PATCHes the cue; the backend re-aligns word timings via difflib
// (guardrail 5) and rebuilds subtitles.ass + subtitles.srt deterministically.

import { useCallback, useEffect, useState } from "react";
import { Cue, api, srtDownloadUrl } from "@/lib/api";

function fmtMs(ms: number): string {
  const s = Math.floor(ms / 1000);
  const cs = Math.floor((ms % 1000) / 10);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
}

interface RowState {
  text: string;
  speaker: string;
  startMs: string;
  endMs: string;
  saving: boolean;
  saved: boolean;
}

/** Timing text field parse: whole ms >= 0; empty string = untouched. */
function parseMs(v: string): number | null {
  const t = v.trim();
  if (!/^\d+$/.test(t)) return null;
  return Number(t);
}

export default function SubtitleGrid({
  projectId,
  onSeek,
}: {
  projectId: string;
  /** click a row timecode to send the program monitor there */
  onSeek?: (t: number) => void;
}) {
  const [cues, setCues] = useState<Cue[]>([]);
  const [rows, setRows] = useState<Record<number, RowState>>({});
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const reload = useCallback(async () => {
    try {
      const list = await api.listSubtitles(projectId);
      setCues(list);
      setRows(
        Object.fromEntries(
          list.map((c) => [
            c.index,
            {
              text: c.text,
              speaker: c.speaker,
              startMs: String(c.start_ms),
              endMs: String(c.end_ms),
              saving: false,
              saved: false,
            },
          ]),
        ),
      );
      setError(null);
      setLoaded(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [projectId]);

  useEffect(() => {
    reload();
  }, [reload]);

  function updateLocal(index: number, patch: Partial<RowState>) {
    setRows((r) => ({ ...r, [index]: { ...r[index], ...patch, saved: false } }));
  }

  async function save(index: number) {
    const row = rows[index];
    if (!row?.text.trim()) return;
    const startMs = parseMs(row.startMs);
    const endMs = parseMs(row.endMs);
    if (startMs === null || endMs === null) {
      setError(`cue ${index}: timing must be whole milliseconds`);
      return;
    }
    if (endMs <= startMs) {
      setError(`cue ${index}: end must be after start (a zero-length cue is never valid)`);
      return;
    }
    updateLocal(index, { saving: true });
    try {
      const timing = { start_ms: startMs, end_ms: endMs };
      const updated = await api.editSubtitle(
        projectId,
        index,
        row.text.trim(),
        row.speaker,
        timing,
      );
      setCues((cs) => cs.map((c) => (c.index === index ? updated : c)));
      setRows((r) => ({
        ...r,
        [index]: {
          text: updated.text,
          speaker: updated.speaker,
          startMs: String(updated.start_ms),
          endMs: String(updated.end_ms),
          saving: false,
          saved: true,
        },
      }));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      updateLocal(index, { saving: false });
    }
  }

  if (loaded && cues.length === 0) {
    return (
      <p className="text-[12px] leading-relaxed text-[var(--tx-3)]">
        No cues yet — transcription lands them here (and on the CC track) once
        the pipeline runs.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-[11px] text-[var(--tx-3)]">
          {cues.length} cue(s) · text saves re-align word timings (difflib, no
          drift) · timing saves move the whole window (words rescale)
        </p>
        <a className="btn-ghost !h-6 !px-2 !text-[11px]" href={srtDownloadUrl(projectId)} download>
          download .srt
        </a>
      </div>
      {error && <p className="text-[11px] text-[var(--err)]">{error}</p>}
      <div className="overflow-y-auto rounded-md border border-[var(--line-1)]">
        <table className="w-full table-fixed text-[12px]">
          <thead className="sticky top-0 bg-[var(--bg-2)] text-left">
            <tr>
              <th className="micro-label w-9 px-2 py-1.5">#</th>
              <th className="micro-label w-24 px-1 py-1.5">in/out</th>
              <th className="micro-label w-24 px-1 py-1.5">speaker</th>
              <th className="micro-label px-1 py-1.5">text</th>
              <th className="w-14 px-1 py-1.5"></th>
            </tr>
          </thead>
          <tbody>
            {cues.map((c) => {
              const row = rows[c.index];
              const dirty =
                row &&
                (row.text !== c.text ||
                  row.speaker !== c.speaker ||
                  row.startMs !== String(c.start_ms) ||
                  row.endMs !== String(c.end_ms));
              return (
                <tr key={c.index} className="border-t border-[var(--line-1)] align-top">
                  <td className="px-2 py-1 text-[var(--tx-3)]">{c.index}</td>
                  <td className="px-1 py-1">
                    <button
                      className="tcode mb-0.5 block text-[10px] text-[var(--tx-3)] hover:text-[var(--acc)]"
                      title="seek program monitor here"
                      onClick={() => onSeek?.(c.start_ms / 1000)}
                    >
                      {fmtMs(c.start_ms)}–{fmtMs(c.end_ms)}
                    </button>
                    <span className="flex items-center gap-0.5">
                      <input
                        className="input !h-5 !w-14 !px-1 !text-[10px]"
                        value={row?.startMs ?? String(c.start_ms)}
                        title="cue start (ms) — saved with the row; word timings rescale into the new window"
                        onChange={(e) => updateLocal(c.index, { startMs: e.target.value })}
                      />
                      <input
                        className="input !h-5 !w-14 !px-1 !text-[10px]"
                        value={row?.endMs ?? String(c.end_ms)}
                        title="cue end (ms) — saved with the row; must be inside the tape"
                        onChange={(e) => updateLocal(c.index, { endMs: e.target.value })}
                      />
                    </span>
                  </td>
                  <td className="px-1 py-1">
                    <input
                      className="input !h-6 !px-1.5 !text-[11px]"
                      value={row?.speaker ?? c.speaker}
                      onChange={(e) => updateLocal(c.index, { speaker: e.target.value })}
                    />
                  </td>
                  <td className="px-1 py-1">
                    <textarea
                      className="input !px-1.5 !py-1 !text-[11px]"
                      rows={1}
                      value={row?.text ?? c.text}
                      onChange={(e) => updateLocal(c.index, { text: e.target.value })}
                    />
                  </td>
                  <td className="px-1 py-1 text-right">
                    <button
                      className="btn-ghost !h-6 !px-2 !text-[11px]"
                      disabled={!dirty || row?.saving}
                      onClick={() => save(c.index)}
                    >
                      {row?.saving ? "…" : row?.saved ? "ok" : "save"}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
