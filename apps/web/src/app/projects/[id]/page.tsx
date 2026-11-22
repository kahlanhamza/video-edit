"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ProxyPlayer from "@/components/ProxyPlayer";
import SubtitleGrid from "@/components/SubtitleGrid";
import {
  api,
  Asset,
  CaseDossier,
  CaseSourceOut,
  DispatchResult,
  Job,
  Project,
  TimelineState,
  sseUrl,
} from "@/lib/api";

const TOOLS = [
  {
    id: "timeline.trim_clip",
    args: { item_index: 0, new_start_s: 1.0, new_dur_s: 3.0 },
    hint: "trim master clip 0",
  },
  {
    id: "timeline.split_clip",
    args: { item_index: 0, at_local_s: 1.5 },
    hint: "split master clip 0",
  },
  {
    id: "redaction.add_blur",
    args: { item_index: 0, bbox: [40, 40, 300, 120], sigma: 24 },
    hint: "privacy blur on clip 0",
  },
  {
    id: "cards.add",
    args: {
      template: "location_intro",
      title: "CASE FILE",
      start_s: 0.0,
      dur_s: 3.0,
    },
    hint: "location card at 0s",
  },
];

// --- WS4: Case Sources & Dossier ---

const SOURCE_KIND_BADGE: Record<string, string> = {
  url: "bg-sky-500/10 text-sky-300",
  pdf: "bg-rose-500/10 text-rose-300",
  text: "bg-emerald-500/10 text-emerald-300",
  source_page: "bg-violet-500/10 text-violet-300",
};

const SOURCE_STATUS_CHIP: Record<string, string> = {
  pending: "bg-slate-800 text-slate-400",
  extracting: "bg-amber-500/10 text-amber-300 animate-pulse",
  ready: "bg-emerald-500/10 text-emerald-300",
  failed: "bg-red-500/10 text-red-400",
};

function isEmptyDossier(d: CaseDossier): boolean {
  return (
    !d.subject_name &&
    d.age == null &&
    !d.role_or_agency &&
    !d.incident_datetime &&
    !d.location &&
    !d.inciting_reason &&
    (d.key_facts?.length ?? 0) === 0 &&
    (d.charges?.length ?? 0) === 0 &&
    (d.test_results?.length ?? 0) === 0 &&
    !d.verdict_outcome
  );
}

function DossierView({ dossier }: { dossier: CaseDossier }) {
  const sourceLabel = (sourceId: string) =>
    dossier.sources?.find((s) => s.source_id === sourceId)?.label ??
    `${sourceId.slice(0, 8)}…`;
  const facts = dossier.key_facts ?? [];
  return (
    <div className="mt-2 space-y-3">
      <div className="text-sm">
        <span className="font-semibold text-slate-100">
          {dossier.subject_name ?? "Unknown subject"}
        </span>
        {dossier.age != null && (
          <span className="text-slate-400"> · age {dossier.age}</span>
        )}
        {dossier.role_or_agency && (
          <span className="text-slate-400"> · {dossier.role_or_agency}</span>
        )}
      </div>

      <dl className="space-y-1 text-sm">
        {(
          [
            ["Incident", dossier.incident_datetime],
            ["Location", dossier.location],
            ["Inciting reason", dossier.inciting_reason],
          ] as Array<[string, string | null]>
        ).map(([label, value]) => (
          <div key={label} className="flex gap-2">
            <dt className="w-28 shrink-0 text-xs uppercase tracking-wide text-slate-500">
              {label}
            </dt>
            <dd className="min-w-0 flex-1 text-slate-200">{value ?? "—"}</dd>
          </div>
        ))}
      </dl>

      {dossier.verdict_outcome && (
        <p className="rounded bg-amber-500/10 px-2 py-1 text-sm text-amber-300">
          Verdict / outcome: {dossier.verdict_outcome}
        </p>
      )}

      {facts.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Key facts
          </h4>
          <ul className="mt-1 space-y-1">
            {facts.map((f, i) => (
              <li key={i} className="text-sm">
                <span className="text-slate-200">{f.fact}</span>
                {f.date && (
                  <span className="ml-1 text-xs text-slate-500">({f.date})</span>
                )}
                <span className="block text-xs text-slate-500" title={f.citation}>
                  — from {sourceLabel(f.source_id)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {(dossier.charges?.length ?? 0) > 0 && (
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Charges
          </h4>
          <div className="mt-1 flex flex-wrap gap-1">
            {(dossier.charges ?? []).map((c, i) => (
              <span
                key={i}
                className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-300"
              >
                {c}
              </span>
            ))}
          </div>
        </div>
      )}

      {(dossier.test_results?.length ?? 0) > 0 && (
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Test results
          </h4>
          <div className="mt-1 flex flex-wrap gap-1">
            {(dossier.test_results ?? []).map((t, i) => (
              <span
                key={i}
                className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-300"
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const [id, setId] = useState<string | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [timeline, setTimeline] = useState<TimelineState | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [progress, setProgress] = useState<{ stage: string; pct: number; error: string | null } | null>(null);
  const [dispatchLog, setDispatchLog] = useState<DispatchResult[]>([]);
  const [publish, setPublish] = useState<{ video_path: string; title: string; base_description: string }>({
    video_path: "",
    title: "",
    base_description: "Rendered by DirectorOS — private review draft.",
  });
  const [publishResult, setPublishResult] = useState<string | null>(null);
  const [ingestUrl, setIngestUrl] = useState("");
  const [ingestBusy, setIngestBusy] = useState(false);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const [runBusy, setRunBusy] = useState(false);
  const [runNote, setRunNote] = useState<string | null>(null);
  const [sources, setSources] = useState<CaseSourceOut[]>([]);
  const [dossier, setDossier] = useState<CaseDossier | null>(null);
  const [sourcesBusy, setSourcesBusy] = useState(false);
  const [refreshNote, setRefreshNote] = useState<string | null>(null);
  const [addSourceKind, setAddSourceKind] = useState<"url" | "text">("url");
  const [addSourceUrl, setAddSourceUrl] = useState("");
  const [addSourceText, setAddSourceText] = useState("");
  const [addSourceLabel, setAddSourceLabel] = useState("");
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    params.then((p) => setId(p.id));
  }, [params]);

  const refresh = useCallback(async () => {
    if (!id) return;
    try {
      setProject(await api.getProject(id));
      setJobs(await api.listJobs(id));
      setTimeline(await api.timelineState(id));
      setAssets(await api.listAssets(id));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [id]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // WS4: case sources + dossier — 404 / empty just means "not built yet"
  const loadSources = useCallback(async () => {
    if (!id) return;
    try {
      setSources(await api.listSources(id));
    } catch {
      setSources([]);
    }
    try {
      setDossier(await api.getDossier(id));
    } catch {
      setDossier(null);
    }
  }, [id]);

  useEffect(() => {
    loadSources();
  }, [loadSources]);

  // live progress over SSE for the newest job
  useEffect(() => {
    if (!id || jobs.length === 0) return;
    const latest = jobs[jobs.length - 1];
    esRef.current?.close();
    const es = new EventSource(sseUrl(id, latest.id));
    es.addEventListener("progress", (ev) => {
      const d = JSON.parse((ev as MessageEvent).data);
      setProgress({ stage: d.stage, pct: d.progress_pct, error: d.error });
    });
    es.addEventListener("done", (ev) => {
      const d = JSON.parse((ev as MessageEvent).data);
      setProgress({ stage: d.stage, pct: d.progress_pct, error: d.error });
      es.close();
      refresh();
    });
    es.onerror = () => es.close();
    esRef.current = es;
    return () => es.close();
  }, [id, jobs, refresh]);

  async function runOrResume() {
    setRunBusy(true);
    setRunNote(null);
    try {
      const r = await api.runPipeline(id!);
      setRunNote(
        r.reused
          ? `Resumed job ${r.job_id.slice(0, 8)}… from where it stopped — finished stages are never redone.`
          : `Started job ${r.job_id.slice(0, 8)}… (${r.detail}).`,
      );
      await refresh();
    } catch (e) {
      setRunNote(e instanceof Error ? e.message : String(e));
    } finally {
      setRunBusy(false);
    }
  }

  async function doIngest() {
    if (!ingestUrl.trim()) return;
    setIngestBusy(true);
    setIngestStatus(null);
    try {
      const r = await api.ingestUrl(id!, ingestUrl.trim());
      setIngestStatus(
        `${r.detail} — job ${r.job_id.slice(0, 8)}… watching progress below`,
      );
      setIngestUrl("");
      await refresh();
    } catch (e) {
      setIngestStatus(`failed: ${e instanceof Error ? e.message : e}`);
    } finally {
      setIngestBusy(false);
    }
  }

  async function dispatch(tool: string, args: Record<string, unknown>) {
    try {
      const result = await api.dispatchTool(id!, tool, args, "studio dispatch");
      setDispatchLog((log) => [result, ...log].slice(0, 10));
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function refreshAndReplan() {
    if (!id || sourcesBusy) return;
    setSourcesBusy(true);
    setRefreshNote(null);
    try {
      const r = await api.refreshSources(id);
      await loadSources();
      await refresh();
      setRefreshNote(`refreshed ${r.refreshed} source(s) — dossier re-planned`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSourcesBusy(false);
    }
  }

  async function addSourceLater() {
    if (!id || sourcesBusy) return;
    const kind = addSourceKind;
    const url = addSourceUrl.trim();
    const text = addSourceText.trim();
    if (kind === "url" && !url) return;
    if (kind === "text" && !text) return;
    setSourcesBusy(true);
    setRefreshNote(null);
    try {
      await api.addSource(
        id,
        kind === "text"
          ? { kind, label: addSourceLabel.trim() || undefined, text }
          : { kind, label: addSourceLabel.trim() || undefined, url },
      );
      setAddSourceUrl("");
      setAddSourceText("");
      setAddSourceLabel("");
      await loadSources();
      setRefreshNote("source attached — hit Refresh & re-plan to extract + re-cut");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSourcesBusy(false);
    }
  }

  async function doPublish() {
    try {
      const r = await api.publishDraft(id!, publish);
      setPublishResult(`draft ${r.video_id} (${r.privacy_status}, captions: ${r.caption_attached})`);
    } catch (e) {
      setPublishResult(`failed: ${e instanceof Error ? e.message : e}`);
    }
  }

  if (!id) return <p className="text-slate-500">Loading…</p>;

  const latest = jobs[jobs.length - 1];
  const pct = progress?.pct ?? latest?.progress_pct ?? 0;
  const playable = assets.find((a) => a.has_proxy);

  return (
    <div className="space-y-6">
      <section className="card">
        <div className="flex items-center justify-between">
          <h1 className="text-lg font-bold">{project?.title ?? id}</h1>
          <span className="rounded bg-slate-800 px-2 py-0.5 text-xs uppercase text-slate-300">
            {project?.status ?? "…"}
          </span>
        </div>
        {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
        {latest && (
          <div className="mt-3">
            <div className="flex justify-between text-xs text-slate-400">
              <span>
                job {latest.id.slice(0, 8)}… · stage {progress?.stage ?? latest.stage}
              </span>
              <span>{pct.toFixed(0)}%</span>
            </div>
            <div className="mt-1 h-2 w-full rounded bg-slate-800">
              <div
                className={`h-2 rounded ${latest.error ? "bg-red-500" : "bg-amber-500"}`}
                style={{ width: `${Math.min(100, pct)}%` }}
              />
            </div>
            {(progress?.error ?? latest.error) && (
              <p className="mt-1 text-xs text-red-400">{progress?.error ?? latest.error}</p>
            )}
          </div>
        )}
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button className="btn" onClick={runOrResume} disabled={runBusy}>
            {runBusy
              ? "Starting…"
              : latest && (progress?.error ?? latest.error)
                ? "Resume pipeline"
                : latest
                  ? "Run again"
                  : "Run pipeline"}
          </button>
          <span className="text-xs text-slate-500">
            scrape sources → download video → transcribe → cut → render → QA — a
            failed step continues where it stopped
          </span>
        </div>
        {runNote && <p className="mt-2 text-xs text-slate-300">{runNote}</p>}
      </section>

      <section className="card">
        <h2 className="font-semibold">Paste media URL</h2>
        <p className="mt-1 text-xs text-slate-500">
          FOIA portal / CDN link — the gateway downloads it (UA-aware), registers
          a queued job and the fleet advances ingest → perception → render. No
          terminal involved.
        </p>
        <div className="mt-3 flex gap-2">
          <input
            className="input"
            placeholder="https://portal.example/case/bodycam.mp4"
            value={ingestUrl}
            onChange={(e) => setIngestUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doIngest()}
          />
          <button className="btn" onClick={doIngest} disabled={ingestBusy || !ingestUrl.trim()}>
            {ingestBusy ? "Ingesting…" : "Ingest"}
          </button>
        </div>
        {ingestStatus && <p className="mt-2 text-sm text-slate-300">{ingestStatus}</p>}
      </section>

      <section className="card">
        <h2 className="font-semibold">Proxy scrub + blur canvas</h2>
        {playable ? (
          <div className="mt-3">
            <ProxyPlayer projectId={id} asset={playable} onChanged={refresh} />
          </div>
        ) : (
          <p className="mt-2 text-sm text-slate-500">
            no 480p proxy yet — ingest builds one automatically (480p + scrubsheet + 64k audio)
          </p>
        )}
      </section>

      <section className="card">
        <div className="flex items-center justify-between gap-2">
          <div>
            <h2 className="font-semibold">Case Sources &amp; Dossier</h2>
            <p className="mt-0.5 text-xs text-slate-500">
              every source feeding this case — the dossier is re-planned from them
              on refresh
            </p>
          </div>
          <button
            className="btn-ghost"
            onClick={refreshAndReplan}
            disabled={sourcesBusy}
          >
            {sourcesBusy ? "Refreshing…" : "Refresh & re-plan"}
          </button>
        </div>

        <div className="mt-3 grid gap-4 lg:grid-cols-2">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Sources ({sources.length})
            </h3>
            <ul className="mt-2 space-y-1">
              {sources.map((s) => (
                <li key={s.id} className="flex items-center gap-2 text-sm">
                  <span
                    className={`rounded px-2 py-0.5 text-xs uppercase tracking-wide ${
                      SOURCE_KIND_BADGE[s.kind] ?? "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {s.kind}
                  </span>
                  <span
                    className="min-w-0 flex-1 truncate text-slate-300"
                    title={s.label}
                  >
                    {s.label || `${s.id.slice(0, 8)}…`}
                  </span>
                  <span
                    className={`rounded px-2 py-0.5 text-xs uppercase tracking-wide ${
                      SOURCE_STATUS_CHIP[s.status] ?? "bg-slate-800 text-slate-400"
                    }`}
                    title={s.status === "failed" && s.error ? s.error : undefined}
                  >
                    {s.status}
                  </span>
                </li>
              ))}
              {sources.length === 0 && (
                <li className="text-sm text-slate-500">
                  no sources yet — attach one below (a link, pasted notes or a
                  file), or create a case with video URLs (auto-discover on)
                </li>
              )}
            </ul>
            {refreshNote && (
              <p className="mt-2 text-xs text-emerald-400">{refreshNote}</p>
            )}
            <div className="mt-3 space-y-2 border-t border-slate-800 pt-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                + attach more sources to this case
              </p>
              <div className="flex gap-2">
                <select
                  className="input w-28"
                  value={addSourceKind}
                  onChange={(e) => setAddSourceKind(e.target.value as "url" | "text")}
                >
                  <option value="url">URL</option>
                  <option value="text">Notes</option>
                </select>
                <input
                  className="input"
                  placeholder={
                    addSourceKind === "url"
                      ? "https://court.example/case-doc (news, court page, direct .pdf)"
                      : "label (e.g. sentencing update)"
                  }
                  value={addSourceKind === "url" ? addSourceUrl : addSourceLabel}
                  onChange={(e) =>
                    addSourceKind === "url"
                      ? setAddSourceUrl(e.target.value)
                      : setAddSourceLabel(e.target.value)
                  }
                />
              </div>
              {addSourceKind === "text" && (
                <textarea
                  className="input"
                  rows={2}
                  placeholder="paste notes — FOIA updates, sentencing, corrections"
                  value={addSourceText}
                  onChange={(e) => setAddSourceText(e.target.value)}
                />
              )}
              <button
                className="btn-ghost"
                onClick={addSourceLater}
                disabled={
                  sourcesBusy ||
                  (addSourceKind === "url" ? !addSourceUrl.trim() : !addSourceText.trim())
                }
              >
                Attach source
              </button>
            </div>
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              CaseDossier
            </h3>
            {dossier && !isEmptyDossier(dossier) ? (
              <DossierView dossier={dossier} />
            ) : (
              <p className="mt-2 text-sm text-slate-500">
                No dossier yet — add sources and refresh.
              </p>
            )}
          </div>
        </div>
      </section>

      <section className="card">
        <h2 className="font-semibold">Subtitle grid</h2>
        <div className="mt-3">
          <SubtitleGrid projectId={id} />
        </div>
      </section>

      <section className="card">
        <h2 className="font-semibold">Timeline versions</h2>
        <p className="mt-1 text-xs text-slate-500">
          active: {timeline?.active_version ?? "—"} · every mutation is a durable
          checkpoint; failed dispatches roll back automatically
        </p>
        <ul className="mt-2 space-y-1">
          {timeline?.versions.map((v) => (
            <li
              key={v.version_num}
              className={`flex justify-between rounded px-2 py-1 text-sm ${
                v.version_num === timeline.active_version
                  ? "bg-amber-500/10 text-amber-300"
                  : "text-slate-300"
              }`}
            >
              <span>
                v{v.version_num} · {v.created_by}
              </span>
              <span className="truncate text-xs text-slate-500">{v.checkpoint_note}</span>
            </li>
          ))}
          {timeline?.versions.length === 0 && (
            <li className="text-sm text-slate-500">no checkpoints yet</li>
          )}
        </ul>
      </section>

      <section className="card">
        <h2 className="font-semibold">Director tools (checkpoint-first)</h2>
        <div className="mt-2 flex flex-wrap gap-2">
          {TOOLS.map((t) => (
            <button
              key={t.id}
              className="btn-ghost"
              onClick={() => dispatch(t.id, t.args)}
              title={t.id}
            >
              {t.hint}
            </button>
          ))}
        </div>
        {dispatchLog.length > 0 && (
          <ul className="mt-3 space-y-1 text-xs">
            {dispatchLog.map((d, i) => (
              <li
                key={i}
                className={d.ok ? "text-emerald-400" : "text-red-400"}
              >
                {d.tool} → {d.ok ? `v${d.version}` : `ROLLED BACK to v${d.rolled_back_to}`} · {d.detail}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card">
        <h2 className="font-semibold">Human Review Gate — YouTube private draft</h2>
        <div className="mt-2 grid gap-2">
          <input
            className="input"
            placeholder="master path (e.g. output/pipeline/<id>/final.mp4)"
            value={publish.video_path}
            onChange={(e) => setPublish({ ...publish, video_path: e.target.value })}
          />
          <input
            className="input"
            placeholder="draft title"
            value={publish.title}
            onChange={(e) => setPublish({ ...publish, title: e.target.value })}
          />
          <textarea
            className="input"
            rows={2}
            placeholder="description"
            value={publish.base_description}
            onChange={(e) => setPublish({ ...publish, base_description: e.target.value })}
          />
          <button
            className="btn w-fit"
            onClick={doPublish}
            disabled={!publish.video_path || !publish.title}
          >
            Publish private draft
          </button>
          {publishResult && <p className="text-sm text-slate-300">{publishResult}</p>}
        </div>
      </section>
    </div>
  );
}
