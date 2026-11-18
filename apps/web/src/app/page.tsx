"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, Project, SourceIn } from "@/lib/api";

interface VideoRow {
  id: number;
  url: string;
}

interface SourceRow {
  id: number;
  kind: "url" | "text";
  label: string;
  url: string;
  text: string;
}

interface PendingUpload {
  id: number;
  file: File;
}

const REMOVE_BTN =
  "shrink-0 rounded border border-slate-700 px-2 py-1 text-xs text-slate-500 hover:border-red-500 hover:text-red-400";

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [title, setTitle] = useState("");
  const [videoRows, setVideoRows] = useState<VideoRow[]>([{ id: 1, url: "" }]);
  const [autoDiscover, setAutoDiscover] = useState(true);
  const [sourceRows, setSourceRows] = useState<SourceRow[]>([]);
  const [pendingUploads, setPendingUploads] = useState<PendingUpload[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const nextId = useRef(2);
  const fileInput = useRef<HTMLInputElement | null>(null);

  const refresh = useCallback(async () => {
    try {
      setProjects(await api.listProjects());
      setError(null);
    } catch (e) {
      setError(
        `gateway unreachable (${e instanceof Error ? e.message : e}) — start it with \`make api\``,
      );
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function updateVideo(id: number, url: string) {
    setVideoRows((rows) => rows.map((r) => (r.id === id ? { ...r, url } : r)));
  }

  function updateSource(
    id: number,
    patch: Partial<Pick<SourceRow, "label" | "url" | "text">>,
  ) {
    setSourceRows((rows) => rows.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  }

  function removeSourceRow(id: number) {
    setSourceRows((rows) => rows.filter((r) => r.id !== id));
  }

  function addSourceRow(kind: "url" | "text") {
    setSourceRows((rows) => [
      ...rows,
      { id: nextId.current++, kind, label: "", url: "", text: "" },
    ]);
  }

  function onFilesPicked(files: FileList | null) {
    if (!files || files.length === 0) return;
    const picked = Array.from(files).filter((f) => /\.(pdf|txt|md)$/i.test(f.name));
    setPendingUploads((list) => [
      ...list,
      ...picked.map((file) => ({ id: nextId.current++, file })),
    ]);
    if (fileInput.current) fileInput.current.value = ""; // allow re-picking the same file
  }

  async function create() {
    if (!title.trim() || busy) return;
    setBusy(true);
    try {
      const videoUrls = videoRows
        .map((r) => r.url.trim())
        .filter((u) => u.length > 0);
      const sources: SourceIn[] = sourceRows
        .map((r) => ({
          kind: r.kind,
          label: r.label.trim() || undefined,
          url: r.kind === "url" ? r.url.trim() || undefined : undefined,
          text: r.kind === "text" ? r.text.trim() || undefined : undefined,
        }))
        .filter((r) => r.url !== undefined || r.text !== undefined);
      const created = await api.createProject(title.trim(), {
        ...(videoUrls.length > 0 ? { video_urls: videoUrls } : {}),
        auto_discover: autoDiscover,
        ...(sources.length > 0 ? { sources } : {}),
      });
      // uploads run sequentially after the case exists; failures are
      // collected per-source so the rest of the case is not lost.
      const failed: string[] = [];
      for (const p of pendingUploads) {
        try {
          await api.uploadSource(created.id, p.file, p.file.name);
        } catch (e) {
          failed.push(`${p.file.name}: ${e instanceof Error ? e.message : e}`);
        }
      }
      setTitle("");
      setVideoRows([{ id: nextId.current++, url: "" }]);
      setAutoDiscover(true);
      setSourceRows([]);
      setPendingUploads([]);
      await refresh();
      if (failed.length > 0) {
        setError(
          `case created, but ${failed.length} upload(s) failed — ${failed.join(" · ")}`,
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <section className="card">
        <h1 className="text-lg font-bold">Cases</h1>
        <p className="mt-1 text-sm text-slate-400">
          Multi-source ingest — one case, 1..N bodycam videos plus case sources
          (PDFs, URLs, pasted notes), distilled into a case dossier.
        </p>

        <div className="mt-3 flex gap-2">
          <input
            className="input"
            placeholder="New case title (e.g. 2024-03-11 traffic stop FOIA)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && create()}
          />
          <button className="btn" onClick={create} disabled={busy || !title.trim()}>
            {busy ? "Creating…" : "Create case"}
          </button>
        </div>

        <div className="mt-4">
          <h2 className="text-sm font-semibold text-slate-300">Video input(s)</h2>
          <div className="mt-2 space-y-2">
            {videoRows.map((row, i) => (
              <div key={row.id} className="flex gap-2">
                <input
                  className="input"
                  placeholder={
                    i === 0
                      ? "Bodycam / source video URL"
                      : `Camera ${i + 1} / alternate source URL`
                  }
                  value={row.url}
                  onChange={(e) => updateVideo(row.id, e.target.value)}
                />
                <button
                  className={REMOVE_BTN}
                  onClick={() =>
                    setVideoRows((rows) => rows.filter((r) => r.id !== row.id))
                  }
                  title="Remove video"
                  aria-label="Remove video"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
          <button
            className="btn-ghost mt-2"
            onClick={() =>
              setVideoRows((rows) => [...rows, { id: nextId.current++, url: "" }])
            }
          >
            + Add another camera / source video
          </button>
        </div>

        <label className="mt-3 flex items-start gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            className="mt-0.5 h-4 w-4 accent-amber-500"
            checked={autoDiscover}
            onChange={(e) => setAutoDiscover(e.target.checked)}
          />
          <span>
            Auto-discover linked summary &amp; docs from source page
            <span className="mt-0.5 block text-xs text-slate-500">
              Follows the video page (Wikimedia/COPA etc.) and pulls linked PDFs +
              summaries into the case dossier.
            </span>
          </span>
        </label>

        <div className="mt-4">
          <h2 className="text-sm font-semibold text-slate-300">Case sources</h2>
          <div className="mt-2 flex flex-wrap gap-2">
            <button className="btn-ghost" onClick={() => addSourceRow("url")}>
              + URL
            </button>
            <button className="btn-ghost" onClick={() => addSourceRow("text")}>
              + Paste Text / Notes
            </button>
            <button className="btn-ghost" onClick={() => fileInput.current?.click()}>
              + Upload File
            </button>
            <input
              ref={fileInput}
              type="file"
              multiple
              accept=".pdf,.txt,.md"
              className="hidden"
              onChange={(e) => onFilesPicked(e.target.files)}
            />
          </div>
          {sourceRows.length === 0 && pendingUploads.length === 0 && (
            <p className="mt-2 text-xs text-slate-500">
              Optional — link articles and dockets, paste FOIA notes / sentencing
              updates, or attach PDF / TXT / MD files to enrich the case dossier.
            </p>
          )}

          <div className="mt-3 space-y-2">
            {sourceRows.map((row) =>
              row.kind === "url" ? (
                <div
                  key={row.id}
                  className="flex flex-wrap gap-2 rounded border border-slate-800 bg-slate-950/40 p-2"
                >
                  <input
                    className="input min-w-0 flex-1"
                    placeholder="https:// source URL — article, docket, press release…"
                    value={row.url}
                    onChange={(e) => updateSource(row.id, { url: e.target.value })}
                  />
                  <input
                    className="input w-full sm:w-56"
                    placeholder="Label (optional)"
                    value={row.label}
                    onChange={(e) => updateSource(row.id, { label: e.target.value })}
                  />
                  <button
                    className={REMOVE_BTN}
                    onClick={() => removeSourceRow(row.id)}
                    title="Remove source"
                    aria-label="Remove source"
                  >
                    ✕
                  </button>
                </div>
              ) : (
                <div
                  key={row.id}
                  className="space-y-2 rounded border border-slate-800 bg-slate-950/40 p-2"
                >
                  <div className="flex gap-2">
                    <input
                      className="input"
                      placeholder="Label (optional) — e.g. FOIA notes / sentencing update"
                      value={row.label}
                      onChange={(e) => updateSource(row.id, { label: e.target.value })}
                    />
                    <button
                      className={REMOVE_BTN}
                      onClick={() => removeSourceRow(row.id)}
                      title="Remove source"
                      aria-label="Remove source"
                    >
                      ✕
                    </button>
                  </div>
                  <textarea
                    className="input"
                    rows={3}
                    placeholder="Paste notes / sentencing updates / custom context for the dossier…"
                    value={row.text}
                    onChange={(e) => updateSource(row.id, { text: e.target.value })}
                  />
                </div>
              ),
            )}
          </div>

          {pendingUploads.length > 0 && (
            <ul className="mt-2 space-y-1">
              {pendingUploads.map((p) => (
                <li
                  key={p.id}
                  className="flex items-center justify-between gap-2 rounded border border-slate-800 bg-slate-950/40 px-2 py-1 text-xs text-slate-300"
                >
                  <span className="min-w-0 truncate">
                    <span className="text-slate-500">upload · </span>
                    {p.file.name}{" "}
                    <span className="text-slate-500">({fmtSize(p.file.size)})</span>
                  </span>
                  <button
                    className={REMOVE_BTN}
                    onClick={() =>
                      setPendingUploads((list) => list.filter((x) => x.id !== p.id))
                    }
                    title="Remove upload"
                    aria-label="Remove upload"
                  >
                    ✕
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
      </section>

      <section className="space-y-2">
        {projects.length === 0 && !error && (
          <p className="text-sm text-slate-500">No cases yet.</p>
        )}
        {projects.map((p) => (
          <a
            key={p.id}
            href={`/projects/${p.id}`}
            className="card block hover:border-amber-600"
          >
            <div className="flex items-center justify-between">
              <span className="font-semibold">{p.title}</span>
              <span className="rounded bg-slate-800 px-2 py-0.5 text-xs uppercase tracking-wide text-slate-300">
                {p.status}
              </span>
            </div>
            <div className="mt-1 text-xs text-slate-500">
              {p.niche_skill} · {p.id.slice(0, 8)}…
            </div>
          </a>
        ))}
      </section>
    </div>
  );
}
