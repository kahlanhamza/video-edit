"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ProxyPlayer from "@/components/ProxyPlayer";
import SubtitleGrid from "@/components/SubtitleGrid";
import { api, Asset, DispatchResult, Job, Project, TimelineState, sseUrl } from "@/lib/api";

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
        {!latest && (
          <p className="mt-2 text-sm text-slate-500">
            No pipeline job yet — start one from the worker:
            <code className="ml-2 rounded bg-slate-950 px-1.5 py-0.5 text-xs">
              celery -A director_os.worker.celery_app.celery_app worker -l info
            </code>
          </p>
        )}
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
