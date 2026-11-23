"use client";

// Cases — the Studio front door (app shell, no page scroll).
// Left: the new-case composer (1..N videos, sources, uploads, auto-discover).
// Right: the case grid with LIVE pipeline progress. One action — create AND
// run; the operator lands straight in the editor and watches the download.

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
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

interface Toast {
  kind: "ok" | "err" | "info";
  message: string;
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

/** pipeline stage → what the operator actually did (no jargon on the door) */
const STAGE_LABEL: Record<string, string> = {
  ingest: "Downloading & preparing media",
  perception: "Transcribing audio",
  proofread: "Proofreading captions",
  vision: "Vision indexing",
  planning: "Cutting the story",
  director: "Directing the re-cut",
  redaction: "Redacting privacy regions",
  narration: "Writing narration",
  audio_prep: "Enhancing audio",
  render: "Rendering",
  qa: "Quality check",
  publish: "Publishing",
  human_gate: "Waiting for your review",
  awaiting_review: "Waiting for your review",
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
  const [toast, setToast] = useState<Toast | null>(null);
  const nextId = useRef(2);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const router = useRouter();

  const showToast = useCallback((kind: Toast["kind"], message: string) => {
    setToast({ kind, message });
  }, []);

  // success/info toasts auto-dismiss; errors stay until the operator closes
  // them (a self-erasing error is a lie — CR-011)
  useEffect(() => {
    if (!toast || toast.kind === "err") return;
    const t = setTimeout(() => setToast(null), 6000);
    return () => clearTimeout(t);
  }, [toast]);

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

  // CR-010, adaptive: while ANY case runs, poll fast (4s) so download/
  // transcription progress is visible; a quiet grid polls slowly (30s).
  // The old fixed 10s poll either starved running cases or churned idle ones.
  const anyRunning = projects.some((p) => p.job_running);
  useEffect(() => {
    const interval = anyRunning ? 4000 : 30000;
    const t = setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, interval);
    const onVisible = () => {
      if (document.visibilityState === "visible") void refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      clearInterval(t);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh, anyRunning]);

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
      // One action: create AND run. With a video URL the worker starts
      // DOWNLOADING immediately (the job registers instantly, the download
      // streams progress) — so the honest move is to land the operator in
      // the editor where the pipeline panel shows the bytes moving. There
      // is nothing to watch on this page; staying here is the "Creating…
      // forever" bug in a nicer costume.
      let runFailed: string | null = null;
      if (videoUrls.length > 0) {
        try {
          const run = await api.runPipeline(created.id, videoUrls[0]);
          if (run.reused) {
            showToast("info", `Resumed the previous pipeline run for “${created.title}”`);
          }
          // Honest multi-camera semantics (CR-006): the pipeline cuts the
          // PRIMARY camera only. Extra URLs stay as case sources.
          if (videoUrls.length > 1) {
            showToast(
              "info",
              `Camera 1 is the primary cut — ${videoUrls.length - 1} extra URL(s) kept as case sources (Media → Ingest to process them)`,
            );
          }
        } catch (e) {
          runFailed = e instanceof Error ? e.message : String(e);
        }
      }
      // uploads run after the case exists; failures are collected per-source
      // so the rest of the case is not lost.
      const failed: string[] = [];
      for (const p of pendingUploads) {
        try {
          await api.uploadSource(created.id, p.file, p.file.name);
        } catch (e) {
          failed.push(`${p.file.name}: ${e instanceof Error ? e.message : e}`);
        }
      }
      // reset the composer before navigating (a stale form on back is a trap)
      setTitle("");
      setVideoRows([{ id: nextId.current++, url: "" }]);
      setAutoDiscover(true);
      setSourceRows([]);
      setPendingUploads([]);
      if (runFailed || failed.length > 0) {
        // the case exists and the operator can retry there — but the failure
        // must be SEEN, so it rides into the editor as a sticky toast
        if (runFailed) {
          showToast("err", `Case created, pipeline not started — ${runFailed}`);
          await refresh();
          return;
        }
        showToast("err", `Case created, ${failed.length} upload(s) failed — ${failed.join(" · ")}`);
      }
      router.push(`/projects/${created.id}`);
    } catch (e) {
      showToast("err", e instanceof Error ? e.message : String(e));
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
        <div className="flex shrink-0 items-center gap-2 border-b border-[var(--line-1)] bg-[var(--bg-1)] px-4 py-1.5 text-[12px] text-[var(--tx-2)]">
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
            {projects.map((p) => {
              const running = p.job_running;
              const pct = p.job_pct ?? 0;
              return (
                <a
                  key={p.id}
                  href={`/projects/${p.id}`}
                  className="panel card-lift group flex flex-col gap-3 p-4 transition-colors hover:border-[var(--acc-line)]"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[13px] font-medium text-[var(--tx-1)]">
                      {p.title}
                    </span>
                    {running ? (
                      <span className="chip chip-acc">
                        <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                        running
                      </span>
                    ) : (
                      <span className={STATUS_CHIP[p.status.toLowerCase()] ?? "chip"}>
                        {p.status}
                      </span>
                    )}
                  </div>
                  {/* live pipeline truth on the card itself — a case that is
                      downloading shows WHERE it is, so nobody clicks around
                      hunting for feedback (the owner's 2026-11-23 report) */}
                  {running && (
                    <div>
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="truncate text-[11px] text-[var(--tx-2)]">
                          {STAGE_LABEL[p.job_stage ?? ""] ?? p.job_stage ?? "working"}
                        </span>
                        <span className="tcode text-[11px] text-[var(--acc)]">
                          {(p.job_pct ?? 0).toFixed(0)}%
                        </span>
                      </div>
                      <div className="bar-track mt-1.5">
                        <div
                          className="bar-fill bar-indeterminate"
                          style={{ width: `${Math.min(100, pct)}%` }}
                        />
                      </div>
                    </div>
                  )}
                  <div className="flex items-center justify-between text-[10.5px] text-[var(--tx-3)]">
                    <span className="uppercase tracking-wider">
                      {p.niche_skill.replaceAll("_", " ")}
                    </span>
                    <span className="tcode">{p.id.slice(0, 8)}</span>
                  </div>
                  <span className="flex items-center gap-1 text-[11px] font-medium text-[var(--acc)] opacity-0 transition-opacity group-hover:opacity-100">
                    Open studio <Icon name="stepFwd" size={11} />
                  </span>
                </a>
              );
            })}
          </div>
        </main>
      </div>

      {/* toast — floating, bottom-right, Material-snackbar energy */}
      {toast && (
        <div
          role="status"
          className={`toast ${toast.kind === "err" ? "toast-err" : toast.kind === "ok" ? "toast-ok" : ""}`}
        >
          <span
            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
              toast.kind === "err"
                ? "bg-[var(--err)]"
                : toast.kind === "ok"
                  ? "bg-[var(--ok)]"
                  : "bg-[var(--acc)]"
            }`}
          />
          <span className="min-w-0 flex-1 leading-relaxed">{toast.message}</span>
          <button
            className="icon-btn !h-5 !w-5 shrink-0"
            onClick={() => setToast(null)}
            title="dismiss"
          >
            <Icon name="x" size={11} />
          </button>
        </div>
      )}
    </div>
  );
}
