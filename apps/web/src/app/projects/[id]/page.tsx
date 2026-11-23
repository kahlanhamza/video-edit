"use client";

// Director Studio — the professional NLE workbench (design system: see
// apps/web/src/app/globals.css + docs/DESIGN_SYSTEM.md).
//
// Full-viewport app layout, no page scroll:
//   ┌ top bar: identity · live stage/progress · hero timecode · Run/Stop/Del ┐
//   │ rail │ left panel      │ program monitor + transport │ pipeline + console│
//   └ timeline dock: 5 tracks, ruler, playhead, snapping, zoom ┘
// The left rail switches Media / Sources / Captions / Tools / Publish.

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import SubtitleGrid from "@/components/SubtitleGrid";
import {
  MonitorEmpty,
  MonitorHandle,
  ProgramMonitor,
} from "@/components/ProgramMonitor";
import ChatPanel from "@/components/studio/ChatPanel";
import ConsolePanel from "@/components/studio/ConsolePanel";
import { Icon, IconName } from "@/components/studio/icons";
import PipelinePanel from "@/components/studio/PipelinePanel";
import Timeline from "@/components/studio/Timeline";
import { useLiveJob } from "@/hooks/useLiveJob";
import { useOpsStream } from "@/hooks/useOpsStream";
import {
  api,
  Asset,
  CaseDossier,
  CaseSourceOut,
  DispatchResult,
  Job,
  NleTimeline,
  ExportGate,
  Project,
  TimelineState,
} from "@/lib/api";

// --- WS4: Case Sources & Dossier --------------------------------------------

const SOURCE_KIND_CHIP: Record<string, string> = {
  url: "chip chip-acc",
  pdf: "chip chip-err",
  text: "chip chip-ok",
  source_page: "chip",
};

const SOURCE_STATUS_CHIP: Record<string, string> = {
  pending: "chip",
  extracting: "chip chip-warn",
  ready: "chip chip-ok",
  failed: "chip chip-err",
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
    <div className="space-y-3">
      <div className="text-[13px]">
        <span className="font-semibold text-[var(--tx-1)]">
          {dossier.subject_name ?? "Unknown subject"}
        </span>
        {dossier.age != null && (
          <span className="text-[var(--tx-2)]"> · age {dossier.age}</span>
        )}
        {dossier.role_or_agency && (
          <span className="text-[var(--tx-2)]"> · {dossier.role_or_agency}</span>
        )}
      </div>

      <dl className="space-y-1.5 text-[12px]">
        {(
          [
            ["Incident", dossier.incident_datetime],
            ["Location", dossier.location],
            ["Inciting reason", dossier.inciting_reason],
          ] as Array<[string, string | null]>
        ).map(([label, value]) => (
          <div key={label} className="flex gap-2">
            <dt className="micro-label w-24 shrink-0 pt-0.5">{label}</dt>
            <dd className="min-w-0 flex-1 text-[var(--tx-1)]">{value ?? "—"}</dd>
          </div>
        ))}
      </dl>

      {dossier.verdict_outcome && (
        <p className="rounded border border-[rgba(245,181,68,.25)] bg-[rgba(245,181,68,.07)] px-2 py-1.5 text-[12px] text-[var(--warn)]">
          Verdict: {dossier.verdict_outcome}
        </p>
      )}

      {facts.length > 0 && (
        <div>
          <h4 className="micro-label">Key facts</h4>
          <ul className="mt-1.5 space-y-1.5">
            {facts.map((f, i) => (
              <li key={i} className="text-[12px] leading-relaxed">
                <span className="text-[var(--tx-1)]">{f.fact}</span>
                {f.date && (
                  <span className="ml-1 text-[11px] text-[var(--tx-3)]">({f.date})</span>
                )}
                <span className="block text-[11px] text-[var(--tx-3)]" title={f.citation}>
                  — {sourceLabel(f.source_id)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {(dossier.charges?.length ?? 0) > 0 && (
        <div>
          <h4 className="micro-label">Charges</h4>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {(dossier.charges ?? []).map((c, i) => (
              <span key={i} className="chip">{c}</span>
            ))}
          </div>
        </div>
      )}

      {(dossier.test_results?.length ?? 0) > 0 && (
        <div>
          <h4 className="micro-label">Test results</h4>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {(dossier.test_results ?? []).map((t, i) => (
              <span key={i} className="chip">{t}</span>
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

type LeftView = "media" | "sources" | "captions" | "tools" | "publish";

const VIEWS: { id: LeftView; label: string; icon: IconName }[] = [
  { id: "media", label: "Media", icon: "media" },
  { id: "sources", label: "Sources", icon: "folder" },
  { id: "captions", label: "Captions", icon: "captions" },
  { id: "tools", label: "Director tools", icon: "wand" },
  { id: "publish", label: "Publish", icon: "upload" },
];

export default function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const [id, setId] = useState<string | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [timeline, setTimeline] = useState<TimelineState | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [nle, setNle] = useState<NleTimeline | null>(null);
  const [gate, setGate] = useState<ExportGate | null>(null);
  const [playheadS, setPlayheadS] = useState(0);
  const [view, setView] = useState<LeftView>("media");
  const [dispatchLog, setDispatchLog] = useState<DispatchResult[]>([]);
  const [publish, setPublish] = useState<{
    video_path: string;
    title: string;
    base_description: string;
  }>({
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
  const monitorRef = useRef<MonitorHandle>(null);

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
    // export gate rides the same refetch cycle (ops stream + live.done)
    try {
      setGate(await api.exportGate(id));
    } catch {
      /* gate state is additive — never blocks the page */
    }
  }, [id]);

  useEffect(() => {
    refresh();
    reloadNle();
  }, [refresh, reloadNle]);

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

  const onTimelineRevision = useCallback(() => {
    reloadNle();
    refresh();
  }, [reloadNle, refresh]);
  const { state: live, reconnect: reconnectLive } = useLiveJob(
    id,
    latestJob?.id ?? null,
    onTimelineRevision,
  );

  // op log v1: agent ops (and human ops from other tabs) land live — one
  // debounced refetch per burst, no job required.
  useOpsStream(id, reloadNle);

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
      // Run reuses the failed job's id on resume — the SSE stream latched
      // closed on the failure, so force it open even when the id is unchanged
      reconnectLive();
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
        `${r.detail} — job ${r.job_id.slice(0, 8)}… watch the console`,
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
      setPublishResult(
        `draft ${r.video_id} (${r.privacy_status}, captions: ${r.caption_attached})`,
      );
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

  // A1 audio trim: the edge drag lands a VO window inside the TTS stem
  async function handleTrimAudio(itemIndex: number, voInS: number, voOutS: number | null) {
    setEditBusy(true);
    await dispatchTool("timeline.trim_audio", {
      item_index: itemIndex,
      vo_in_s: voInS,
      vo_out_s: voOutS,
    });
    setEditBusy(false);
  }

  async function handleSplitAtPlayhead(t: number) {
    if (!nle) return;
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
      const fx = nle?.effects.find((e) => e.id === effectId);
      if (fx?.op_id) {
        await api.revokeOp(id, fx.op_id, "removed from the studio timeline");
      } else {
        await api.deleteRedaction(id, effectId);
      }
      await reloadNle();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setEditBusy(false);
    }
  }

  async function handleAdjustEffectWindow(opId: string, frameStart: number, frameEnd: number) {
    if (!id || !nle) return;
    const fx = nle.effects.find((e) => e.op_id === opId);
    if (!fx) return;
    setEditBusy(true);
    try {
      // geometry carries over; the drag only retimes the frame window
      await api.adjustOp(id, opId, {
        x: fx.x,
        y: fx.y,
        w: fx.w,
        h: fx.h,
        frame_start: frameStart,
        frame_end: frameEnd,
        note: "window retimed from the FX track",
      });
      await reloadNle();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setEditBusy(false);
    }
  }

  const seek = useCallback((t: number) => {
    setPlayheadS(t);
    monitorRef.current?.seek(t);
  }, []);

  if (!id) return <p className="p-6 text-[var(--tx-3)]">Loading…</p>;

  const pct = live.job?.progress_pct ?? latestJob?.progress_pct ?? 0;
  const stage = live.job?.stage ?? latestJob?.stage ?? "idle";
  const jobError = live.job?.error ?? latestJob?.error ?? null;
  const playable = assets.find((a) => a.has_proxy);
  const fps = nle?.asset?.fps ?? 30;
  const running = !live.done && !jobError && live.connected;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[var(--bg-0)]">
      {/* ================= top bar ================= */}
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-[var(--line-1)] bg-[var(--bg-1)] px-3">
        <Link
          href="/"
          className="icon-btn"
          title="back to all cases"
        >
          <Icon name="back" />
        </Link>
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded bg-[var(--acc-bg)] text-[var(--acc)]">
            <Icon name="clapper" size={13} />
          </span>
          <span className="text-[13px] font-semibold tracking-wide text-[var(--tx-1)]">
            DirectorOS <span className="font-normal text-[var(--tx-3)]">Studio</span>
          </span>
        </div>
        <span className="h-5 w-px bg-[var(--line-2)]" />
        <div className="min-w-0">
          <h1 className="truncate text-[13px] font-medium text-[var(--tx-1)]">
            {project?.title ?? id}
          </h1>
          <p className="truncate text-[10px] uppercase tracking-wider text-[var(--tx-3)]">
            {project?.niche_skill.replaceAll("_", " ")}
            {latestJob ? ` · job ${latestJob.id.slice(0, 8)}…` : " · no jobs yet"}
          </p>
        </div>
        <span
          className={`chip ${
            jobError ? "chip-err" : live.done ? "chip-ok" : running ? "chip-acc" : ""
          }`}
        >
          {project?.status ?? "…"}
        </span>

        {/* single source of truth: live stage + overall progress */}
        <div className="hidden min-w-[200px] max-w-[340px] flex-1 items-center gap-2.5 md:flex">
          <span className="truncate text-[11px] text-[var(--tx-2)]">
            <span className="micro-label">stage</span>{" "}
            <span className="font-medium text-[var(--tx-1)]">{stage}</span>
          </span>
          <div className="bar-track min-w-[80px] flex-1">
            <div
              className="bar-fill"
              data-tone={jobError ? "err" : live.done ? "ok" : undefined}
              style={{ width: `${Math.min(100, pct)}%` }}
            />
          </div>
          <span className="tcode text-[11px] text-[var(--tx-2)]">
            {pct.toFixed(0)}%
          </span>
        </div>

        <div className="flex-1" />

        {/* export gate (step 6): the ONLY render — dirty state lives on the
            server (baked_ops_rev stamped by the bake), never client-trusted */}
        <button
          className={gate?.dirty ? "btn" : "btn-ghost"}
          onClick={runOrResume}
          disabled={runBusy || running}
          title={
            gate?.exported_once
              ? gate?.dirty
                ? gate.pending_ops > 0
                  ? `${gate.pending_ops} op event(s) since the last export — re-export re-bakes only the dirty segments`
                  : "Timeline edited since the last export (trim/split/move) — re-export bakes the new cut"
                : `Exported (ops rev ${gate.baked_ops_rev ?? "—"}, version ${gate.baked_timeline_version ?? "—"}) — the program on screen is what is on disk`
              : "First export: bakes the program (tracked ops + blur regions) into the deliverable"
          }
        >
          <Icon name="media" size={11} />
          {gate?.dirty || !gate?.exported_once
            ? `Export${gate && gate.exported_once && gate.pending_ops > 0 ? ` · ${gate.pending_ops} new` : ""}`
            : "Exported ✓"}
        </button>
        <button className="btn" onClick={runOrResume} disabled={runBusy}>
          <Icon name="play" size={11} />
          {runBusy
            ? "Starting…"
            : jobError
              ? "Resume"
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
          <Icon name="stop" size={11} /> Stop
        </button>
        <button
          className="btn-danger"
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
          <Icon name="trash" size={12} />
          {deleteBusy ? "Deleting…" : "Delete"}
        </button>
        <Link href="/settings" className="icon-btn" title="credentials vault">
          <Icon name="settings" />
        </Link>
      </header>

      {/* error / run note strip */}
      {(jobError || runNote || error) && (
        <div
          className={`flex shrink-0 items-center gap-2 border-b px-3 py-1.5 text-[11px] ${
            jobError
              ? "border-[rgba(241,109,109,.3)] bg-[rgba(241,109,109,.08)] text-[var(--err)]"
              : "border-[var(--line-1)] bg-[var(--bg-1)] text-[var(--tx-2)]"
          }`}
        >
          <button
            className="icon-btn !h-4 !w-4"
            onClick={() => {
              setRunNote(null);
              setError(null);
            }}
            title="dismiss"
          >
            <Icon name="x" size={10} />
          </button>
          <span className="min-w-0 truncate">{jobError ?? runNote ?? error}</span>
        </div>
      )}

      {/* ================= body ================= */}
      <div className="flex min-h-0 flex-1">
        {/* icon rail */}
        <nav className="flex w-11 shrink-0 flex-col items-center gap-1 border-r border-[var(--line-1)] bg-[var(--bg-1)] py-2">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              className="icon-btn !h-9 !w-9"
              data-active={view === v.id}
              title={v.label}
              onClick={() => setView(v.id)}
            >
              <Icon name={v.icon} size={17} />
            </button>
          ))}
          <div className="flex-1" />
          <Link href="/settings" className="icon-btn !h-9 !w-9" title="Settings">
            <Icon name="settings" size={16} />
          </Link>
        </nav>

        {/* left panel */}
        <aside className="flex w-[300px] shrink-0 flex-col border-r border-[var(--line-1)] bg-[var(--bg-1)]">
          <div className="panel-header">
            <span className="micro-label">
              {VIEWS.find((v) => v.id === view)?.label}
            </span>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            {view === "media" && (
              <div className="space-y-4">
                <div>
                  <h3 className="micro-label">Ingest media</h3>
                  <p className="mt-1 text-[11px] leading-relaxed text-[var(--tx-3)]">
                    FOIA portal / CDN link — the gateway downloads it (UA-aware)
                    and the fleet advances ingest → perception → render.
                  </p>
                  <div className="mt-2 flex gap-1.5">
                    <input
                      className="input"
                      placeholder="https://portal.example/case/bodycam.mp4"
                      value={ingestUrl}
                      onChange={(e) => setIngestUrl(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && doIngest()}
                    />
                    <button
                      className="btn shrink-0"
                      onClick={doIngest}
                      disabled={ingestBusy || !ingestUrl.trim()}
                    >
                      <Icon name="upload" size={12} />
                    </button>
                  </div>
                  {ingestStatus && (
                    <p className="mt-1.5 text-[11px] text-[var(--tx-2)]">{ingestStatus}</p>
                  )}
                </div>

                <div>
                  <h3 className="micro-label">Assets ({assets.length})</h3>
                  <div className="mt-1.5 space-y-1.5">
                    {assets.length === 0 && (
                      <p className="text-[11px] text-[var(--tx-3)]">
                        no assets yet — ingest creates the 480p proxy bundle
                      </p>
                    )}
                    {assets.map((a) => (
                      <div
                        key={a.id}
                        className="flex items-center gap-2 rounded-md border border-[var(--line-1)] bg-[var(--bg-2)] px-2 py-1.5"
                      >
                        <span className="text-[var(--tx-3)]">
                          <Icon name="film" size={14} />
                        </span>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-[11px] text-[var(--tx-1)]">
                            {a.role.replaceAll("_", " ")}
                          </p>
                          <p className="tcode text-[10px] text-[var(--tx-3)]">
                            {a.duration_s.toFixed(1)}s
                          </p>
                        </div>
                        <span className={`chip ${a.has_proxy ? "chip-ok" : ""}`}>
                          {a.has_proxy ? "480p" : "raw"}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="micro-label">Checkpoints</h3>
                  <p className="mt-1 text-[11px] leading-relaxed text-[var(--tx-3)]">
                    active v{timeline?.active_version ?? "—"} · every mutation is
                    a durable checkpoint; failed dispatches roll back
                    automatically.
                  </p>
                </div>
              </div>
            )}

            {view === "sources" && (
              <div className="space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <h3 className="micro-label">Sources ({sources.length})</h3>
                  <button className="btn-ghost !h-6 !px-2 !text-[11px]" onClick={refreshAndReplan} disabled={sourcesBusy}>
                    {sourcesBusy ? "Refreshing…" : "Refresh & re-plan"}
                  </button>
                </div>
                <ul className="space-y-1">
                  {sources.map((s) => (
                    <li key={s.id} className="flex items-center gap-1.5">
                      <span className={`shrink-0 ${SOURCE_KIND_CHIP[s.kind] ?? "chip"}`}>
                        {s.kind}
                      </span>
                      <span
                        className="min-w-0 flex-1 truncate text-[11.5px] text-[var(--tx-1)]"
                        title={s.label}
                      >
                        {s.label || `${s.id.slice(0, 8)}…`}
                      </span>
                      <span
                        className={`shrink-0 ${SOURCE_STATUS_CHIP[s.status] ?? "chip"}`}
                        title={s.status === "failed" && s.error ? s.error : undefined}
                      >
                        {s.status}
                      </span>
                    </li>
                  ))}
                  {sources.length === 0 && (
                    <li className="text-[11px] text-[var(--tx-3)]">
                      no sources yet — attach one below, or create a case with video URLs
                    </li>
                  )}
                </ul>
                {refreshNote && (
                  <p className="text-[11px] text-[var(--ok)]">{refreshNote}</p>
                )}

                <div className="space-y-2 border-t border-[var(--line-1)] pt-3">
                  <div className="flex gap-1.5">
                    <select
                      className="input !w-20"
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
                          ? "https://court.example/case-doc"
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
                    className="btn-ghost w-full"
                    onClick={addSourceLater}
                    disabled={
                      sourcesBusy ||
                      (addSourceKind === "url" ? !addSourceUrl.trim() : !addSourceText.trim())
                    }
                  >
                    <Icon name="plus" size={12} /> Attach source
                  </button>
                </div>

                <div className="border-t border-[var(--line-1)] pt-3">
                  <h3 className="micro-label">Case dossier</h3>
                  {dossier && !isEmptyDossier(dossier) ? (
                    <div className="mt-2">
                      <DossierView dossier={dossier} />
                    </div>
                  ) : (
                    <p className="mt-1.5 text-[11px] text-[var(--tx-3)]">
                      No dossier yet — add sources and refresh.
                    </p>
                  )}
                </div>
              </div>
            )}

            {view === "captions" && <SubtitleGrid projectId={id} onSeek={seek} />}

            {view === "tools" && (
              <div className="space-y-3">
                <p className="text-[11px] leading-relaxed text-[var(--tx-3)]">
                  Checkpoint-first director tools — the same engine the LLM
                  director uses. Every dispatch is versioned and rolls back on
                  failure.
                </p>
                <div className="grid grid-cols-2 gap-1.5">
                  {TOOLS.map((t) => (
                    <button
                      key={t.id}
                      className="btn-ghost !h-8 justify-start !text-[11px]"
                      onClick={() => dispatchTool(t.id, t.args)}
                      title={t.id}
                    >
                      {t.hint}
                    </button>
                  ))}
                </div>
                {editBusy && (
                  <p className="text-[11px] text-[var(--warn)]">dispatching edit…</p>
                )}
                {dispatchLog.length > 0 && (
                  <ul className="space-y-1 text-[11px]">
                    {dispatchLog.map((d, i) => (
                      <li
                        key={i}
                        className={d.ok ? "text-[var(--ok)]" : "text-[var(--err)]"}
                      >
                        {d.tool} →{" "}
                        {d.ok ? `v${d.version}` : `ROLLED BACK to v${d.rolled_back_to}`} ·{" "}
                        {d.detail}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {view === "publish" && (
              <div className="space-y-2">
                <p className="text-[11px] leading-relaxed text-[var(--tx-3)]">
                  Human Review Gate — approve the cut as a YouTube Private Draft.
                </p>
                <input
                  className="input"
                  placeholder="master path (output/pipeline/<id>/final.mp4)"
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
                  rows={3}
                  placeholder="description"
                  value={publish.base_description}
                  onChange={(e) =>
                    setPublish({ ...publish, base_description: e.target.value })
                  }
                />
                <button
                  className="btn"
                  onClick={doPublish}
                  disabled={!publish.video_path || !publish.title}
                >
                  <Icon name="upload" size={12} /> Publish private draft
                </button>
                {publishResult && (
                  <p className="text-[11px] text-[var(--tx-2)]">{publishResult}</p>
                )}
              </div>
            )}
          </div>
        </aside>

        {/* program monitor */}
        <main className="flex min-w-0 flex-1 flex-col bg-black">
          {playable ? (
            <ProgramMonitor
              ref={monitorRef}
              projectId={id}
              asset={playable}
              fps={fps}
              opsRev={nle?.ops_rev ?? null}
              onTimeUpdate={setPlayheadS}
              onChanged={() => reloadNle()}
            />
          ) : (
            <MonitorEmpty onRun={runOrResume} />
          )}
        </main>

        {/* pipeline + console */}
        <aside className="flex w-[340px] shrink-0 flex-col overflow-hidden border-l border-[var(--line-1)] bg-[var(--bg-1)] xl:w-[380px]">
          <div className="flex min-h-0 flex-[1.15] flex-col">
            <PipelinePanel live={live} />
          </div>
          <div className="flex min-h-0 flex-[1.2] flex-col">
            <ChatPanel projectId={id} />
          </div>
          <div className="flex min-h-0 flex-1 flex-col">
            <ConsolePanel live={live} />
          </div>
        </aside>
      </div>

      {/* ================= timeline dock ================= */}
      <section className="flex h-[268px] shrink-0 flex-col border-t border-[var(--line-1)] bg-[var(--bg-1)]">
        <Timeline
          nle={nle}
          playheadS={playheadS}
          onSeek={seek}
          onTrimClip={handleTrim}
          onTrimAudio={handleTrimAudio}
          onSplitAtPlayhead={handleSplitAtPlayhead}
          onDeleteEffect={handleDeleteEffect}
          onAdjustEffectWindow={handleAdjustEffectWindow}
        />
      </section>
    </div>
  );
}
