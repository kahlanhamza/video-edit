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
  Deliverables,
  DispatchResult,
  Job,
  NleClip,
  NleTimeline,
  ExportGate,
  Project,
  TimelineState,
  TimelineVersion,
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

type LeftView = "media" | "sources" | "captions" | "tools" | "publish";
const VIEWS: { id: LeftView; label: string; icon: IconName }[] = [
  { id: "media", label: "Media", icon: "media" },
  { id: "sources", label: "Sources", icon: "folder" },
  { id: "captions", label: "Captions", icon: "captions" },
  { id: "tools", label: "Agent activity", icon: "wand" },
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
  const [publishBusy, setPublishBusy] = useState(false);
  const [deliverables, setDeliverables] = useState<Deliverables | null>(null);
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
  // insert popover (raw clip / narration / card at the playhead); the undo
  // history rides the existing `timeline` state (GET /timeline versions)
  const [insertOpen, setInsertOpen] = useState(false);
  const [insertTab, setInsertTab] = useState<"raw" | "narration" | "card">("raw");
  const [insertSrcStart, setInsertSrcStart] = useState("0");
  const [insertDur, setInsertDur] = useState("8");
  const [insertScript, setInsertScript] = useState("");
  const [insertBrollStart, setInsertBrollStart] = useState("");
  const [insertCardTitle, setInsertCardTitle] = useState("");
  const [insertCardSubtitle, setInsertCardSubtitle] = useState("");

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

  // CR-009: when the SSE stream is dead but the newest job is unfinished,
  // poll the server so progress + job truth keep moving (and `running` stays
  // honest — buttons stay locked while the worker renders). Healthy SSE
  // streams make this a no-op.
  const pollJob = jobs.length ? jobs[jobs.length - 1] : null;
  const pollNeeded =
    pollJob != null &&
    !pollJob.error &&
    pollJob.progress_pct < 100 &&
    pollJob.stage !== "human_gate" &&
    pollJob.stage !== "awaiting_review" &&
    (!live.connected || live.done);
  useEffect(() => {
    if (!pollNeeded) return;
    const t = setInterval(() => {
      void refresh();
    }, 5000);
    return () => clearInterval(t);
  }, [pollNeeded, refresh]);

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

  // Export is NOT a re-run: when the gate is clean the disk already carries
  // the reviewed program — the button must say so and do nothing (CR-004:
  // Export used to be a disguised pipeline restart, clickable after "Exported ✓").
  const exportClean = !!gate && gate.exported_once && !gate.dirty;
  async function doExport() {
    if (exportClean) return;
    await runOrResume();
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
    setPublishBusy(true);
    setPublishResult(null);
    try {
      const r = await api.publishDraft(id!, {
        ...publish,
        srt_path: deliverables?.srt_path ?? null,
      });
      setPublishResult(
        r.dry_run
          ? `SIMULATED — no YouTube call was made (draft id ${r.video_id}). Add YOUTUBE_CLIENT_ID / SECRET / REFRESH_TOKEN in Settings, or upload ship_pack/final_1080p.mp4 manually.`
          : `REAL YouTube private draft created: ${r.video_id} (${r.privacy_status}, captions: ${r.caption_attached}) — review it at youtube.com and press Publish there.`,
      );
    } catch (e) {
      setPublishResult(`failed: ${e instanceof Error ? e.message : e}`);
    } finally {
      setPublishBusy(false);
    }
  }

  // the publish view must show what the disk ACTUALLY carries — nobody
  // hand-types a filesystem path (CR-003)
  const loadDeliverables = useCallback(async () => {
    if (!id) return;
    try {
      const d = await api.getDeliverables(id);
      setDeliverables(d);
      setPublish((p) => ({
        ...p,
        video_path: p.video_path || d.master_path || "",
        title: p.title || d.suggested_title || "",
      }));
    } catch {
      setDeliverables(null); // 404 / no outputs yet
    }
  }, [id]);

  useEffect(() => {
    if (view === "publish") void loadDeliverables();
  }, [view, loadDeliverables]);

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

  // --- timeline editing v2: delete / add / undo / redo (all server-checkpointed) ---

  async function handleDeleteClip(clip: NleClip) {
    if (clip.master_index == null) return;
    setEditBusy(true);
    await dispatchTool("timeline.remove_clip", { item_index: clip.master_index });
    setEditBusy(false);
  }

  const activeVersion = timeline?.active_version ?? nle?.version ?? null;
  const versions: TimelineVersion[] = timeline?.versions ?? [];
  const canUndo =
    activeVersion != null && versions.some((v) => v.version_num < activeVersion);
  const canRedo =
    activeVersion != null && versions.some((v) => v.version_num > activeVersion);

  async function activateRelative(dir: -1 | 1) {
    if (!id || activeVersion == null) return;
    const candidates = versions
      .map((v) => v.version_num)
      .filter((n) => (dir === -1 ? n < activeVersion : n > activeVersion));
    if (candidates.length === 0) return;
    const target = dir === -1 ? Math.max(...candidates) : Math.min(...candidates);
    setEditBusy(true);
    try {
      const r = await api.activateVersion(id, target);
      setRunNote(
        dir === -1
          ? `Undo — activated timeline v${r.active_version} (was v${r.previous_version ?? "—"}). Nothing is deleted; Redo walks forward again.`
          : `Redo — activated timeline v${r.active_version}.`,
      );
      await reloadNle();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setEditBusy(false);
    }
  }

  /** master item index under the playhead (insert BEFORE it), null = append */
  function insertIndexAtPlayhead(): number | null {
    if (!nle) return null;
    const c = nle.clips.find(
      (x) => x.track === "V1_master" && playheadS >= x.start_s && playheadS < x.end_s,
    );
    return c?.master_index ?? null;
  }

  async function handleInsertRaw() {
    const start = Number(insertSrcStart);
    const dur = Number(insertDur);
    if (!Number.isFinite(start) || start < 0 || !Number.isFinite(dur) || dur <= 0) {
      setError("source start must be >= 0 s and duration > 0 s");
      return;
    }
    setEditBusy(true);
    await dispatchTool("timeline.insert_raw_clip", {
      source_start_s: start,
      dur_s: dur,
      at_index: insertIndexAtPlayhead(),
    });
    setEditBusy(false);
    setInsertOpen(false);
  }

  async function handleInsertNarration() {
    const dur = Number(insertDur);
    const script = insertScript.trim();
    if (!script) {
      setError("write the narration script first");
      return;
    }
    if (!Number.isFinite(dur) || dur < 1 || dur > 120) {
      setError("narration duration must be 1..120 s");
      return;
    }
    const broll = insertBrollStart.trim();
    setEditBusy(true);
    await dispatchTool("timeline.add_narration", {
      script_text: script,
      dur_s: dur,
      at_index: insertIndexAtPlayhead(),
      ...(broll !== "" && Number.isFinite(Number(broll))
        ? { broll_source_start_s: Number(broll) }
        : {}),
    });
    setEditBusy(false);
    setInsertOpen(false);
    setInsertScript("");
  }

  async function handleInsertCard() {
    const title = insertCardTitle.trim();
    const dur = Number(insertDur);
    if (!title) {
      setError("card title is required");
      return;
    }
    if (!Number.isFinite(dur) || dur < 0.5 || dur > 60) {
      setError("card duration must be 0.5..60 s");
      return;
    }
    setEditBusy(true);
    await dispatchTool("cards.add", {
      template: "evidence_fact",
      title,
      subtitle: insertCardSubtitle.trim(),
      start_s: playheadS,
      dur_s: dur,
    });
    setEditBusy(false);
    setInsertOpen(false);
    setInsertCardTitle("");
    setInsertCardSubtitle("");
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
  // running = SERVER truth first (the newest job is unfinished and not parked
  // at the gate), SSE liveness second. CR-009: a dropped SSE stream used to
  // flip this false and re-enable Run/Export mid-run while the worker was
  // still rendering — the buttons lied and a click double-dispatched.
  const latestJobUnfinished =
    latestJob != null &&
    !latestJob.error &&
    latestJob.progress_pct < 100 &&
    latestJob.stage !== "human_gate" &&
    latestJob.stage !== "awaiting_review";
  const running = latestJobUnfinished || (!live.done && !jobError && live.connected);
  const streamStale = latestJobUnfinished && !live.connected && !live.done;

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
          {streamStale && (
            <span
              className="chip"
              title="the live event stream dropped — progress comes from server polling until it reconnects; Run/Export stay locked while the worker is busy"
            >
              reconnecting…
            </span>
          )}
        </div>

        <div className="flex-1" />

        {/* export gate (step 6): the ONLY render — dirty state lives on the
            server (baked_ops_rev stamped by the bake), never client-trusted.
            When the gate is clean the button is DISABLED, not a disguised re-run. */}
        <button
          className={gate?.dirty ? "btn" : "btn-ghost"}
          onClick={doExport}
          disabled={runBusy || running || exportClean}
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
        <button
          className="btn"
          onClick={runOrResume}
          disabled={runBusy || running}
          title={
            running
              ? "The pipeline is already working — progress is on the right. Stop it first to restart or switch sources."
              : jobError
                ? "Resume from the failed stage — finished stages are never redone"
                : latestJob
                  ? "Re-advance the pipeline (finished stages short-circuit)"
                  : "Run the full pipeline"
          }
        >
          <Icon name="play" size={11} />
          {runBusy
            ? "Starting…"
            : running
              ? "Running…"
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
                  Every timeline mutation — yours (trim/split handles) or the
                  agent's (chat) — is a checkpointed op. Failed dispatches roll
                  back automatically; this log is the record. Edits are made on
                  the timeline itself or by telling the chat agent what to cut —
                  never by firing blind preset tools (CR-005: the old demo
                  buttons silently re-cut clip 0 with hardcoded args).
                </p>
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
                {deliverables === null ? (
                  <p className="text-[11px] text-[var(--tx-3)]">
                    Nothing rendered yet — run the pipeline first; the master and
                    ship pack appear here automatically.
                  </p>
                ) : (
                  <div className="rounded border border-[var(--line-1)] bg-[var(--bg-2)] p-2 text-[11px] text-[var(--tx-2)]">
                    <div className="tcode text-[var(--tx-3)]">SHIPPABLE MASTER</div>
                    <div className="mt-1 break-all text-[var(--tx-1)]">
                      {deliverables.master_path ?? "— no master on disk yet"}
                    </div>
                    {deliverables.master_size_bytes != null && (
                      <div className="mt-1 text-[var(--tx-3)]">
                        {(deliverables.master_size_bytes / 1e6).toFixed(1)} MB
                        {deliverables.master_is_post_fx ? " · post-fx (captions + blur baked)" : " · pre-post-fx master"}
                      </div>
                    )}
                    {deliverables.pack_files.length > 0 && (
                      <div className="mt-2">
                        <div className="tcode text-[var(--tx-3)]">SHIP PACK — {deliverables.pack_files.length} file(s)</div>
                        <div className="mt-1 break-all text-[var(--tx-3)]">
                          {deliverables.pack_files.map((f) => f.name).join(" · ")}
                        </div>
                      </div>
                    )}
                    {!deliverables.youtube_configured && (
                      <div className="mt-2 text-[var(--tx-3)]">
                        YouTube OAuth not configured — publishing records a SIMULATION only.
                        Add YOUTUBE_CLIENT_ID / SECRET / REFRESH_TOKEN in Settings for a real
                        upload, or upload <span className="tcode">ship_pack/final_1080p.mp4</span> to
                        YouTube Studio manually (chapters + captions are in the pack).
                      </div>
                    )}
                  </div>
                )}
                {deliverables?.gate_dirty && (
                  <p className="text-[11px] leading-relaxed text-[var(--warn,#e2b341)]">
                    Export gate is dirty — the timeline on screen is NOT the file below.
                    Hit Export first, then publish.
                  </p>
                )}
                <input
                  className="input"
                  placeholder="master path (prefilled from the ship pack)"
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
                  disabled={!publish.video_path || !publish.title || publishBusy}
                >
                  <Icon name="upload" size={12} />
                  {publishBusy ? "Publishing…" : "Publish private draft"}
                </button>
                {publishResult && (
                  <p className="whitespace-pre-wrap text-[11px] leading-relaxed text-[var(--tx-2)]">
                    {publishResult}
                  </p>
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
              nle={nle}
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
      <section className="relative flex h-[268px] shrink-0 flex-col border-t border-[var(--line-1)] bg-[var(--bg-1)]">
        {/* insert popover — Add (+) on the timeline toolbar opens it; lives
            here so it can overlay the timeline without clipping */}
        {insertOpen && (
          <div className="absolute bottom-[270px] left-1/2 z-50 w-[420px] -translate-x-1/2 rounded-lg border border-[var(--line-2)] bg-[var(--bg-2)] p-3 shadow-2xl">
            <div className="mb-2.5 flex items-center gap-1.5">
              <span className="micro-label">Insert at {playheadS.toFixed(1)}s</span>
              <div className="flex-1" />
              {(["raw", "narration", "card"] as const).map((t) => (
                <button
                  key={t}
                  className={`chip ${insertTab === t ? "chip-acc" : ""}`}
                  onClick={() => setInsertTab(t)}
                >
                  {t === "raw" ? "Raw clip" : t === "narration" ? "Narration VO" : "Card"}
                </button>
              ))}
              <button className="icon-btn !h-5 !w-5" onClick={() => setInsertOpen(false)} title="close">
                <Icon name="x" size={11} />
              </button>
            </div>

            {insertTab === "raw" && (
              <div className="space-y-2">
                <p className="text-[11px] leading-relaxed text-[var(--tx-3)]">
                  Pull a stretch of the SOURCE into the cut at the playhead —
                  e.g. a cold-open climax from later in the tape. Seconds are
                  source footage time.
                </p>
                <div className="flex gap-2">
                  <label className="flex-1">
                    <span className="micro-label">source start (s)</span>
                    <input className="input mt-1" value={insertSrcStart} onChange={(e) => setInsertSrcStart(e.target.value)} inputMode="decimal" />
                  </label>
                  <label className="flex-1">
                    <span className="micro-label">duration (s)</span>
                    <input className="input mt-1" value={insertDur} onChange={(e) => setInsertDur(e.target.value)} inputMode="decimal" />
                  </label>
                </div>
                <button className="btn w-full" disabled={editBusy} onClick={handleInsertRaw}>
                  <Icon name="plus" size={12} /> Insert raw clip
                </button>
              </div>
            )}

            {insertTab === "narration" && (
              <div className="space-y-2">
                <p className="text-[11px] leading-relaxed text-[var(--tx-3)]">
                  A voiceover block over muted b-roll. Leave b-roll empty to ride
                  the scene the previous clip showed. TTS is synthesized on the
                  next render/export.
                </p>
                <textarea
                  className="input"
                  rows={3}
                  placeholder="narration script — what the viewer hears"
                  value={insertScript}
                  onChange={(e) => setInsertScript(e.target.value)}
                />
                <div className="flex gap-2">
                  <label className="flex-1">
                    <span className="micro-label">duration (s, 1–120)</span>
                    <input className="input mt-1" value={insertDur} onChange={(e) => setInsertDur(e.target.value)} inputMode="decimal" />
                  </label>
                  <label className="flex-1">
                    <span className="micro-label">b-roll source start (optional)</span>
                    <input className="input mt-1" placeholder="previous clip's scene" value={insertBrollStart} onChange={(e) => setInsertBrollStart(e.target.value)} inputMode="decimal" />
                  </label>
                </div>
                <button className="btn w-full" disabled={editBusy || !insertScript.trim()} onClick={handleInsertNarration}>
                  <Icon name="plus" size={12} /> Add narration VO
                </button>
              </div>
            )}

            {insertTab === "card" && (
              <div className="space-y-2">
                <p className="text-[11px] leading-relaxed text-[var(--tx-3)]">
                  A graphic card (evidence fact) overlaid at the playhead.
                </p>
                <input className="input" placeholder="card title" value={insertCardTitle} onChange={(e) => setInsertCardTitle(e.target.value)} />
                <div className="flex gap-2">
                  <input className="input flex-1" placeholder="subtitle (optional)" value={insertCardSubtitle} onChange={(e) => setInsertCardSubtitle(e.target.value)} />
                  <label className="w-28">
                    <span className="micro-label">duration (s)</span>
                    <input className="input mt-1" value={insertDur} onChange={(e) => setInsertDur(e.target.value)} inputMode="decimal" />
                  </label>
                </div>
                <button className="btn w-full" disabled={editBusy || !insertCardTitle.trim()} onClick={handleInsertCard}>
                  <Icon name="plus" size={12} /> Add card
                </button>
              </div>
            )}
          </div>
        )}

        <Timeline
          nle={nle}
          playheadS={playheadS}
          onSeek={seek}
          onTrimClip={handleTrim}
          onTrimAudio={handleTrimAudio}
          onSplitAtPlayhead={handleSplitAtPlayhead}
          onDeleteEffect={handleDeleteEffect}
          onAdjustEffectWindow={handleAdjustEffectWindow}
          onDeleteClip={handleDeleteClip}
          onInsert={() => setInsertOpen((v) => !v)}
          canUndo={canUndo}
          canRedo={canRedo}
          onUndo={() => activateRelative(-1)}
          onRedo={() => activateRelative(1)}
        />
      </section>
    </div>
  );
}
