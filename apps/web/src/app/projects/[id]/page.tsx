"use client";

// Director Studio — the full-visibility editing workbench.
//
// Top bar: project identity + live stage/progress + Run/Stop/Delete.
// Main grid: proxy player + blur canvas (left), embedded sandbox console (right).
// Below: the multi-track NLE timeline (video / cards / audio / captions /
// effects) with manual trim/split, then tabbed inspector panels (subtitles,
// sources & dossier, director tools, publish, ingest).

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import ProxyPlayer from "@/components/ProxyPlayer";
import SubtitleGrid from "@/components/SubtitleGrid";
import RunConsole from "@/components/studio/RunConsole";
import Timeline from "@/components/studio/Timeline";
import { useLiveJob } from "@/hooks/useLiveJob";
import {
  api,
  Asset,
  CaseDossier,
  CaseSourceOut,
  DispatchResult,
  Job,
  NleTimeline,
  Project,
  TimelineState,
} from "@/lib/api";

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

const STUDIO_TABS = ["subtitles", "sources", "tools", "publish"] as const;
type StudioTab = (typeof STUDIO_TABS)[number];

const TAB_LABEL: Record<StudioTab, string> = {
  subtitles: "Subtitles",
  sources: "Sources & Dossier",
  tools: "Director tools",
  publish: "Publish",
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

export default function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const [id, setId] = useState<string | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [timeline, setTimeline] = useState<TimelineState | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [nle, setNle] = useState<NleTimeline | null>(null);
  const [playheadS, setPlayheadS] = useState(0);
  const [tab, setTab] = useState<StudioTab>("subtitles");
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
  const [stopBusy, setStopBusy] = useState(false);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const router = useRouter();
  const [sources, setSources] = useState<CaseSourceOut[]>([]);
  const [dossier, setDossier] = useState<CaseDossier | null>(null);
  const [sourcesBusy, setSourcesBusy] = useState(false);
  const [refreshNote, setRefreshNote] = useState<string | null>(null);
  const [addSourceKind, setAddSourceKind] = useState<"url" | "text">("url");
  const [addSourceUrl, setAddSourceUrl] = useState("");
  const [addSourceText, setAddSourceText] = useState("");
  const [addSourceLabel, setAddSourceLabel] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [editBusy, setEditBusy] = useState(false);

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

  const reloadNle = useCallback(async () => {
    if (!id) return;
    try {
      setNle(await api.nleTimeline(id));
    } catch {
      setNle(null); // 404 = no timeline yet
    }
  }, [id]);

  useEffect(() => {
    refresh();
    reloadNle();
  }, [refresh, reloadNle]);

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

  const latestJob = jobs.length ? jobs[jobs.length - 1] : null;

  // live console + timeline revisions over SSE
  const onTimelineRevision = useCallback(() => {
    reloadNle();
    refresh();
  }, [reloadNle, refresh]);
  const { state: live } = useLiveJob(id, latestJob?.id ?? null, onTimelineRevision);

  // when a run finishes, pull the final state (thumbnails, versions, NLE)
  const prevDoneRef = useRef(false);
  useEffect(() => {
    if (live.done && !prevDoneRef.current) {
      refresh();
      reloadNle();
    }
    prevDoneRef.current = live.done;
  }, [live.done, refresh, reloadNle]);

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
        `${r.detail} — job ${r.job_id.slice(0, 8)}… watching the console`,
      );
      setIngestUrl("");
      await refresh();
    } catch (e) {
      setIngestStatus(`failed: ${e instanceof Error ? e.message : e}`);
    } finally {
      setIngestBusy(false);
    }
  }

  async function dispatchTool(
    tool: string,
    args: Record<string, unknown>,
  ): Promise<DispatchResult | null> {
    try {
      const result = await api.dispatchTool(id!, tool, args, "studio manual edit");
      setDispatchLog((log) => [result, ...log].slice(0, 10));
      await reloadNle();
      await refresh();
      return result;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
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

  // --- timeline manual edits ---
  async function handleTrim(itemIndex: number, newStartS: number, newDurS: number) {
    setEditBusy(true);
    await dispatchTool("timeline.trim_clip", {
      item_index: itemIndex,
      new_start_s: newStartS,
      new_dur_s: newDurS,
    });
    setEditBusy(false);
  }

  async function handleSplitAtPlayhead(t: number) {
    if (!nle) return;
    // the master clip under the playhead (video lane, timeline-local seconds)
    const clip = nle.clips.find(
      (c) => c.track === "V1_master" && t >= c.start_s && t < c.end_s,
    );
    if (!clip || clip.master_index == null) {
      setError(`no master clip under ${t.toFixed(2)}s to split`);
      return;
    }
    setEditBusy(true);
    await dispatchTool("timeline.split_clip", {
      item_index: clip.master_index,
      at_local_s: Number((t - clip.start_s).toFixed(3)),
    });
    setEditBusy(false);
  }

  async function handleDeleteEffect(effectId: string) {
    if (!id) return;
    setEditBusy(true);
    try {
      await api.deleteRedaction(id, effectId);
      await reloadNle();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setEditBusy(false);
    }
  }

  if (!id) return <p className="text-slate-500">Loading…</p>;

  const pct = live.job?.progress_pct ?? latestJob?.progress_pct ?? 0;
  const stage = live.job?.stage ?? latestJob?.stage ?? "idle";
  const jobError = live.job?.error ?? latestJob?.error ?? null;
  const playable = assets.find((a) => a.has_proxy);

  return (
    <div className="space-y-4">
      {/* ---------------- top bar: identity + live status + operator controls ---------------- */}
      <section className="card sticky top-0 z-20 !py-3 backdrop-blur">
        <div className="flex flex-wrap items-center gap-3">
          <Link
            href="/"
            className="btn-ghost !border-slate-700 !px-2 !py-1 text-xs"
            title="back to all cases"
          >
            ←
          </Link>
          <div className="min-w-0">
            <h1 className="truncate text-base font-bold">{project?.title ?? id}</h1>
            <p className="text-[10px] uppercase tracking-wide text-slate-500">
              {project?.niche_skill} · latest job {latestJob ? `${latestJob.id.slice(0, 8)}…` : "none"}
            </p>
          </div>
          <span
            className={`rounded px-2 py-0.5 text-xs uppercase ${
              jobError
                ? "bg-red-500/15 text-red-300"
                : live.done
                  ? "bg-emerald-500/15 text-emerald-300"
                  : "bg-slate-800 text-slate-300"
            }`}
          >
            {project?.status ?? "…"}
          </span>

          {/* live stage + overall progress */}
          <div className="min-w-[180px] flex-1">
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>
                stage <span className="font-semibold text-slate-200">{stage}</span>
              </span>
              <span className="tabular-nums">{pct.toFixed(0)}%</span>
            </div>
            <div className="mt-1 h-1.5 w-full rounded bg-slate-800">
              <div
                className={`h-1.5 rounded transition-all duration-300 ${
                  jobError ? "bg-red-500" : "bg-amber-400"
                }`}
                style={{ width: `${Math.min(100, pct)}%` }}
              />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button className="btn" onClick={runOrResume} disabled={runBusy}>
              {runBusy
                ? "Starting…"
                : jobError
                  ? "Resume pipeline"
                  : latestJob
                    ? "Run again"
                    : "Run pipeline"}
            </button>
            <button
              className="btn-ghost"
              onClick={async () => {
                setStopBusy(true);
                try {
                  const out = await api.stopProject(id);
                  setRunNote(out.note);
                } catch (e) {
                  setRunNote(e instanceof Error ? e.message : String(e));
                } finally {
                  setStopBusy(false);
                }
              }}
              disabled={stopBusy || !latestJob}
              title="parks the job at the next stage boundary — completed stages are kept"
            >
              {stopBusy ? "Stopping…" : "Stop"}
            </button>
            <button
              className="btn-ghost !text-red-400 hover:!border-red-500"
              onClick={async () => {
                if (
                  !window.confirm(
                    "Delete this project, all its jobs and all pipeline output? This cannot be undone.",
                  )
                )
                  return;
                setDeleteBusy(true);
                try {
                  await api.deleteProject(id);
                  router.push("/");
                } catch (e) {
                  setError(e instanceof Error ? e.message : String(e));
                  setDeleteBusy(false);
                }
              }}
              disabled={deleteBusy}
              title="removes every DB row + the pipeline output tree for this project"
            >
              {deleteBusy ? "Deleting…" : "Delete"}
            </button>
            <Link href="/settings" className="btn-ghost text-xs" title="credentials vault">
              ⚙
            </Link>
          </div>
        </div>
        {(runNote || jobError) && (
          <p className={`mt-2 text-xs ${jobError ? "text-red-400" : "text-slate-300"}`}>
            {jobError ?? runNote}
          </p>
        )}
        {error && !runNote && !jobError && (
          <p className="mt-2 text-xs text-red-400">{error}</p>
        )}
      </section>

      {/* ---------------- main grid: player + live console ---------------- */}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <section className="card min-w-0">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Proxy scrub + blur canvas</h2>
            {nle?.asset && (
              <span className="text-[10px] text-slate-500">
                source {nle.asset.width}x{nle.asset.height} @ {nle.asset.fps.toFixed(2)}fps ·{" "}
                {nle.asset.duration_s.toFixed(1)}s
              </span>
            )}
          </div>
          {playable ? (
            <div className="mt-3">
              <ProxyPlayer
                projectId={id}
                asset={playable}
                onChanged={() => {
                  reloadNle();
                }}
                onTimeUpdate={setPlayheadS}
              />
            </div>
          ) : (
            <p className="mt-2 text-sm text-slate-500">
              no 480p proxy yet — ingest builds one automatically (480p + scrubsheet + 64k audio)
            </p>
          )}
        </section>

        <section className="card flex min-h-[420px] flex-col !p-0 xl:h-[520px]">
          <RunConsole live={live} />
        </section>
      </div>

      {/* ---------------- full-width live NLE timeline ---------------- */}
      <section className="card">
        <Timeline
          nle={nle}
          playheadS={playheadS}
          onSeek={setPlayheadS}
          onTrimClip={handleTrim}
          onSplitAtPlayhead={handleSplitAtPlayhead}
          onDeleteEffect={handleDeleteEffect}
        />
        {editBusy && <p className="mt-1 text-xs text-amber-300">dispatching edit…</p>}
        <p className="mt-1 text-xs text-slate-500">
          active checkpoint: v{timeline?.active_version ?? "—"} · every mutation is a durable
          checkpoint; failed dispatches roll back automatically
        </p>
      </section>

      {/* ---------------- tabbed inspector ---------------- */}
      <section className="card">
        <div className="flex flex-wrap gap-1 border-b border-slate-800 pb-2">
          {STUDIO_TABS.map((t) => (
            <button
              key={t}
              className={`rounded px-3 py-1 text-sm ${
                tab === t
                  ? "bg-amber-500/15 font-semibold text-amber-300"
                  : "text-slate-400 hover:text-slate-200"
              }`}
              onClick={() => setTab(t)}
            >
              {TAB_LABEL[t]}
            </button>
          ))}
        </div>

        <div className="pt-3">
          {tab === "subtitles" && <SubtitleGrid projectId={id} />}

          {tab === "sources" && (
            <div>
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs text-slate-500">
                  every source feeding this case — the dossier is re-planned from them on refresh
                </p>
                <button className="btn-ghost" onClick={refreshAndReplan} disabled={sourcesBusy}>
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
                        <span className="min-w-0 flex-1 truncate text-slate-300" title={s.label}>
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
                        no sources yet — attach one below, or create a case with video URLs
                      </li>
                    )}
                  </ul>
                  {refreshNote && <p className="mt-2 text-xs text-emerald-400">{refreshNote}</p>}
                  <div className="mt-3 space-y-2 border-t border-slate-800 pt-3">
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

              <div className="mt-4 border-t border-slate-800 pt-3">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Paste media URL
                </h3>
                <p className="mt-1 text-xs text-slate-500">
                  FOIA portal / CDN link — the gateway downloads it (UA-aware) and the fleet
                  advances ingest → perception → render. Watch the live console.
                </p>
                <div className="mt-2 flex gap-2">
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
              </div>
            </div>
          )}

          {tab === "tools" && (
            <div>
              <p className="text-xs text-slate-500">
                checkpoint-first director tools — same engine the LLM director uses
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {TOOLS.map((t) => (
                  <button
                    key={t.id}
                    className="btn-ghost"
                    onClick={() => dispatchTool(t.id, t.args)}
                    title={t.id}
                  >
                    {t.hint}
                  </button>
                ))}
              </div>
              {dispatchLog.length > 0 && (
                <ul className="mt-3 space-y-1 text-xs">
                  {dispatchLog.map((d, i) => (
                    <li key={i} className={d.ok ? "text-emerald-400" : "text-red-400"}>
                      {d.tool} →{" "}
                      {d.ok ? `v${d.version}` : `ROLLED BACK to v${d.rolled_back_to}`} · {d.detail}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {tab === "publish" && (
            <div className="grid gap-2 md:max-w-xl">
              <p className="text-xs text-slate-500">
                Human Review Gate — approve the cut as a YouTube Private Draft
              </p>
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
          )}
        </div>
      </section>
    </div>
  );
}
