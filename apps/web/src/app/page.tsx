"use client";

// Cases — the Studio front door (app shell, no page scroll).
// Left: the new-case composer (1..N videos, sources, uploads, auto-discover).
// Right: the case grid. One action — create AND run; the agent takes over.

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { api, Project, SourceIn } from "@/lib/api";
import { Icon } from "@/components/studio/icons";

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

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const STATUS_CHIP: Record<string, string> = {
  created: "chip",
  running: "chip chip-acc",
  done: "chip chip-ok",
  failed: "chip chip-err",
  rendered: "chip chip-ok",
};

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

  // CR-010: the front door never refreshed — a case could sit "done" for
  // hours while the grid showed a stale chip. Poll while the tab is visible;
  // pipelines park at the gate on their own, the operator should see it.
  useEffect(() => {
    const t = setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, 10000);
    const onVisible = () => {
      if (document.visibilityState === "visible") void refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      clearInterval(t);
      document.removeEventListener("visibilitychange", onVisible);
    };
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
      // one action: create AND run — the agent takes it from here (download,
      // scrape sources, script, cut, render). Failures are resumable via the
      // project page's Run/Resume button.
      let runNote = "";
      if (videoUrls.length > 0) {
        try {
          const run = await api.runPipeline(created.id, videoUrls[0]);
          runNote = run.reused
            ? "resumed the previous pipeline run"
            : `pipeline started (${run.detail})`;
          // Honest multi-camera semantics (CR-006): the pipeline cuts the
          // PRIMARY camera only. Extra URLs stay as case sources — ingesting
          // them now would launch a SECOND full pipeline on the same project
          // and clobber the reviewed timeline version. Ingest them from
          // Media → Ingest when you deliberately want them processed.
          if (videoUrls.length > 1) {
            runNote += ` — camera 1 is the primary cut; ${videoUrls.length - 1} extra URL(s) kept as case sources (Media → Ingest to process them)`;
          }
        } catch (e) {
          runNote = `pipeline not started: ${e instanceof Error ? e.message : e}`;
        }
      }
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
          `case created, but ${failed.length} upload(s) failed — ${failed.join(" · ")} ${runNote}`,
        );
      } else if (runNote) {
        setError(`case created — ${runNote}`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[var(--bg-0)]">
      {/* ---- top bar ---- */}
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-[var(--line-1)] bg-[var(--bg-1)] px-4">
        <span className="flex h-6 w-6 items-center justify-center rounded bg-[var(--acc-bg)] text-[var(--acc)]">
          <Icon name="clapper" size={13} />
        </span>
        <span className="text-[13px] font-semibold tracking-wide text-[var(--tx-1)]">
          DirectorOS <span className="font-normal text-[var(--tx-3)]">Studio</span>
        </span>
        <span className="chip">cases</span>
        <div className="flex-1" />
        <Link href="/settings" className="btn-ghost !h-7">
          <Icon name="settings" size={13} /> Settings
        </Link>
      </header>

      {error && (
        <div className="flex shrink-0 items-center gap-2 border-b border-[var(--line-1)] bg-[var(--bg-1)] px-4 py-1.5 text-[11px] text-[var(--tx-2)]">
          <button
            className="icon-btn !h-4 !w-4"
            onClick={() => setError(null)}
            title="dismiss"
          >
            <Icon name="x" size={10} />
          </button>
          <span className="min-w-0 truncate">{error}</span>
        </div>
      )}

      {/* ---- workspace ---- */}
      <div className="grid min-h-0 flex-1 gap-0 lg:grid-cols-[400px_1fr]">
        {/* composer */}
        <aside className="flex min-h-0 flex-col overflow-y-auto border-r border-[var(--line-1)] bg-[var(--bg-1)]">
          <div className="panel-header">
            <span className="micro-label">New case</span>
          </div>
          <div className="space-y-5 p-4">
            <div>
              <h3 className="micro-label">Title</h3>
              <div className="mt-1.5 flex gap-1.5">
                <input
                  className="input"
                  placeholder="2024-03-11 traffic stop FOIA"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && create()}
                />
                <button
                  className="btn shrink-0"
                  onClick={create}
                  disabled={busy || !title.trim()}
                >
                  <Icon name="plus" size={12} />
                  {busy ? "Creating…" : "Create"}
                </button>
              </div>
              <p className="mt-1.5 text-[11px] leading-relaxed text-[var(--tx-3)]">
                One case, 1..N bodycam videos plus case sources — distilled
                into a case dossier. Creating runs the pipeline immediately.
              </p>
            </div>

            <div>
              <h3 className="micro-label">Video input</h3>
              <div className="mt-1.5 space-y-1.5">
                {videoRows.map((row, i) => (
                  <div key={row.id} className="flex gap-1.5">
                    <input
                      className="input"
                      placeholder={
                        i === 0
                          ? "Bodycam / source video URL"
                          : `Camera ${i + 1} / alternate source`
                      }
                      value={row.url}
                      onChange={(e) => updateVideo(row.id, e.target.value)}
                    />
                    {videoRows.length > 1 && (
                      <button
                        className="icon-btn"
                        onClick={() =>
                          setVideoRows((rows) => rows.filter((r) => r.id !== row.id))
                        }
                        title="Remove video"
                      >
                        <Icon name="x" size={12} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <button
                className="btn-ghost mt-1.5 !h-7 !text-[11px]"
                onClick={() =>
                  setVideoRows((rows) => [...rows, { id: nextId.current++, url: "" }])
                }
              >
                <Icon name="plus" size={11} /> Add another camera
              </button>
            </div>

            <div className="flex items-start gap-2.5">
              <button
                className="toggle mt-0.5"
                data-on={autoDiscover}
                onClick={() => setAutoDiscover((v) => !v)}
                aria-label="auto-discover linked documents"
              />
              <span className="text-[12px] text-[var(--tx-1)]">
                Auto-discover linked summaries &amp; docs
                <span className="mt-0.5 block text-[11px] leading-relaxed text-[var(--tx-3)]">
                  Follows the video page (Wikimedia/COPA etc.) and pulls linked
                  PDFs + summaries into the case dossier.
                </span>
              </span>
            </div>

            <div>
              <h3 className="micro-label">Case sources</h3>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                <button className="btn-ghost !h-7 !text-[11px]" onClick={() => addSourceRow("url")}>
                  <Icon name="plus" size={11} /> URL
                </button>
                <button className="btn-ghost !h-7 !text-[11px]" onClick={() => addSourceRow("text")}>
                  <Icon name="plus" size={11} /> Notes
                </button>
                <button
                  className="btn-ghost !h-7 !text-[11px]"
                  onClick={() => fileInput.current?.click()}
                >
                  <Icon name="upload" size={11} /> File
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
                <p className="mt-1.5 text-[11px] leading-relaxed text-[var(--tx-3)]">
                  Optional — link articles and dockets, paste FOIA notes, or
                  attach PDF / TXT / MD to enrich the dossier.
                </p>
              )}

              <div className="mt-2 space-y-1.5">
                {sourceRows.map((row) =>
                  row.kind === "url" ? (
                    <div
                      key={row.id}
                      className="flex flex-wrap gap-1.5 rounded-md border border-[var(--line-1)] bg-[var(--bg-2)] p-2"
                    >
                      <input
                        className="input min-w-0 flex-1"
                        placeholder="https:// source URL — article, docket…"
                        value={row.url}
                        onChange={(e) => updateSource(row.id, { url: e.target.value })}
                      />
                      <input
                        className="input w-full sm:w-48"
                        placeholder="Label (optional)"
                        value={row.label}
                        onChange={(e) => updateSource(row.id, { label: e.target.value })}
                      />
                      <button
                        className="icon-btn"
                        onClick={() => removeSourceRow(row.id)}
                        title="Remove source"
                      >
                        <Icon name="x" size={12} />
                      </button>
                    </div>
                  ) : (
                    <div
                      key={row.id}
                      className="space-y-1.5 rounded-md border border-[var(--line-1)] bg-[var(--bg-2)] p-2"
                    >
                      <div className="flex gap-1.5">
                        <input
                          className="input"
                          placeholder="Label — e.g. FOIA notes / sentencing update"
                          value={row.label}
                          onChange={(e) => updateSource(row.id, { label: e.target.value })}
                        />
                        <button
                          className="icon-btn"
                          onClick={() => removeSourceRow(row.id)}
                          title="Remove source"
                        >
                          <Icon name="x" size={12} />
                        </button>
                      </div>
                      <textarea
                        className="input"
                        rows={3}
                        placeholder="Paste notes / sentencing updates / custom context…"
                        value={row.text}
                        onChange={(e) => updateSource(row.id, { text: e.target.value })}
                      />
                    </div>
                  ),
                )}
              </div>

              {pendingUploads.length > 0 && (
                <ul className="mt-1.5 space-y-1">
                  {pendingUploads.map((p) => (
                    <li
                      key={p.id}
                      className="flex items-center justify-between gap-2 rounded-md border border-[var(--line-1)] bg-[var(--bg-2)] px-2 py-1 text-[11px] text-[var(--tx-2)]"
                    >
                      <span className="min-w-0 truncate">
                        {p.file.name}{" "}
                        <span className="text-[var(--tx-3)]">({fmtSize(p.file.size)})</span>
                      </span>
                      <button
                        className="icon-btn !h-5 !w-5"
                        onClick={() =>
                          setPendingUploads((list) => list.filter((x) => x.id !== p.id))
                        }
                        title="Remove upload"
                      >
                        <Icon name="x" size={11} />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </aside>

        {/* case grid */}
        <main className="min-h-0 overflow-y-auto p-5">
          <div className="flex items-center justify-between">
            <h2 className="text-[13px] font-semibold text-[var(--tx-1)]">
              All cases
            </h2>
            <span className="chip">{projects.length}</span>
          </div>
          {projects.length === 0 && (
            <div className="mt-16 flex flex-col items-center gap-3 text-center">
              <span className="flex h-14 w-14 items-center justify-center rounded-xl bg-[var(--bg-2)] text-[var(--tx-3)]">
                <Icon name="folder" size={26} />
              </span>
              <p className="text-[13px] text-[var(--tx-2)]">No cases yet</p>
              <p className="max-w-[320px] text-[11px] leading-relaxed text-[var(--tx-3)]">
                Compose one on the left — paste a bodycam URL, attach case
                sources, and the agent researches, cuts and captions it
                end-to-end.
              </p>
            </div>
          )}
          <div className="mt-4 grid gap-2.5 sm:grid-cols-2 xl:grid-cols-3">
            {projects.map((p) => (
              <a
                key={p.id}
                href={`/projects/${p.id}`}
                className="panel group flex flex-col gap-3 p-4 transition-colors hover:border-[var(--acc-line)]"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-[13px] font-medium text-[var(--tx-1)]">
                    {p.title}
                  </span>
                  <span className={STATUS_CHIP[p.status.toLowerCase()] ?? "chip"}>
                    {p.status}
                  </span>
                </div>
                <div className="flex items-center justify-between text-[10px] text-[var(--tx-3)]">
                  <span className="uppercase tracking-wider">
                    {p.niche_skill.replaceAll("_", " ")}
                  </span>
                  <span className="tcode">{p.id.slice(0, 8)}</span>
                </div>
                <span className="flex items-center gap-1 text-[11px] font-medium text-[var(--acc)] opacity-0 transition-opacity group-hover:opacity-100">
                  Open studio <Icon name="stepFwd" size={11} />
                </span>
              </a>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
