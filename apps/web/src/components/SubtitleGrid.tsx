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
  saving: boolean;
  saved: boolean;
}

export default function SubtitleGrid({ projectId }: { projectId: string }) {
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
            { text: c.text, speaker: c.speaker, saving: false, saved: false },
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
    updateLocal(index, { saving: true });
    try {
      const updated = await api.editSubtitle(projectId, index, row.text.trim(), row.speaker);
      setCues((cs) => cs.map((c) => (c.index === index ? updated : c)));
      setRows((r) => ({ ...r, [index]: { text: updated.text, speaker: updated.speaker, saving: false, saved: true } }));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      updateLocal(index, { saving: false });
    }
  }

  if (loaded && cues.length === 0) {
    return <p className="text-sm text-slate-500">no cues yet — run the pipeline first</p>;
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-slate-500">
          {cues.length} cue(s) · word timings re-aligned on save (difflib, no drift)
        </p>
        <a className="btn-ghost text-xs" href={srtDownloadUrl(projectId)} download>
          download .srt
        </a>
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}
      <div className="max-h-96 overflow-y-auto rounded border border-slate-800">
        <table className="w-full table-fixed text-sm">
          <thead className="sticky top-0 bg-slate-900 text-left text-xs uppercase text-slate-500">
            <tr>
              <th className="w-10 px-2 py-1">#</th>
              <th className="w-28 px-2 py-1">in/out</th>
              <th className="w-32 px-2 py-1">speaker</th>
              <th className="px-2 py-1">text</th>
              <th className="w-16 px-2 py-1"></th>
            </tr>
          </thead>
          <tbody>
            {cues.map((c) => {
              const row = rows[c.index];
              const dirty = row && (row.text !== c.text || row.speaker !== c.speaker);
              return (
                <tr key={c.index} className="border-t border-slate-800 align-top">
                  <td className="px-2 py-1 text-slate-500">{c.index}</td>
                  <td className="px-2 py-1 font-mono text-xs text-slate-400">
                    {fmtMs(c.start_ms)}–{fmtMs(c.end_ms)}
                  </td>
                  <td className="px-2 py-1">
                    <input
                      className="input px-1 py-0.5 text-xs"
                      value={row?.speaker ?? c.speaker}
                      onChange={(e) => updateLocal(c.index, { speaker: e.target.value })}
                    />
                  </td>
                  <td className="px-2 py-1">
                    <textarea
                      className="input w-full px-1 py-0.5 text-xs"
                      rows={1}
                      value={row?.text ?? c.text}
                      onChange={(e) => updateLocal(c.index, { text: e.target.value })}
                    />
                  </td>
                  <td className="px-2 py-1 text-right">
                    <button
                      className="btn-ghost px-2 py-0.5 text-xs"
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
