// DirectorOS gateway client — typed fetch helpers (M3 Director Studio).
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export interface Project {
  id: string;
  title: string;
  niche_skill: string;
  status: string;
}

export interface Job {
  id: string;
  stage: string;
  progress_pct: number;
  error: string | null;
}

export interface RunResult {
  project_id: string;
  job_id: string;
  reused: boolean;
  dispatched: boolean;
  detail: string;
  stage: string;
  progress_pct: number;
}

// --- full-visibility Studio: live steps + console logs + NLE timeline ---

export interface JobStep {
  id: number;
  key: string;
  label: string;
  status: "running" | "done" | "failed" | "skipped";
  pct: number;
  detail: string;
  order: number;
  duration_ms: number;
}

export interface JobLog {
  id: number;
  level: "info" | "warn" | "error";
  message: string;
}

export interface LiveJobSnapshot {
  steps: JobStep[];
  logs: JobLog[];
}

export interface NleClip {
  track: string;
  kind: string; // raw | narration | card | ...
  name: string;
  start_s: number;
  dur_s: number;
  end_s: number;
  /** index on the master track INCLUDING gaps — the DirectorToolbox item_index */
  master_index: number | null;
  act: string | null;
  script: string | null;
  template: string | null;
  audio_mode: string | null;
  /** in-point in SOURCE footage seconds (trim math) */
  source_start_s: number | null;
  /** A1 audio trim: the VO window inside the TTS stem + the measured stem
   *  (optional — snapshots predate the trim, undefined = untrimmed full stem) */
  vo_in_s?: number;
  vo_out_s?: number | null;
  tts_duration_s?: number | null;
}

export interface NleCaption {
  index: number;
  start_ms: number;
  end_ms: number;
  text: string;
  speaker: string;
}

export interface NleEffect {
  id: string;
  op_id: string | null;
  op_rev: number | null;
  kind?: string;
  x: number;
  y: number;
  w: number;
  h: number;
  sigma: number;
  label: string;
  mode: string;
  t_start_s: number | null;
  t_end_s: number | null;
  frame_start?: number | null;
  frame_end?: number | null;
  created_by: string;
  /** "agent" | "human" | "legacy" — op provenance for the timeline badge. */
  origin?: string;
}

export interface NleAsset {
  id: string;
  duration_s: number;
  fps: number;
  width: number;
  height: number;
  has_proxy: boolean;
}

export interface NleTimeline {
  project_id: string;
  version: number | null;
  duration_s: number;
  asset: NleAsset | null;
  clips: NleClip[];
  captions: NleCaption[];
  effects: NleEffect[];
  /** Op log cursor + counts (op log v1): the fold the timeline renders. */
  ops_rev?: number;
  ops_counts?: { active: number; revoked: number };
  caption_overrides?: Record<string, Record<string, unknown>>;
}

// --- op log v1: everything-is-an-op ---
export interface OpRecord {
  rev: number;
  id: string;
  project_id: string;
  asset_id: string;
  kind: string;
  frame_start: number | null;
  frame_end: number | null;
  track: string;
  params: Record<string, unknown>;
  origin: "agent" | "human";
  evidence: Record<string, unknown>;
  session_id: string;
  revokes_op_id: string | null;
  revoked: boolean;
  created_at: string;
}

export interface ExportGate {
  ops_rev: number;
  baked_ops_rev: number | null;
  active_version?: number | null;
  baked_timeline_version?: number | null;
  dirty: boolean;
  pending_ops: number;
  exported_once: boolean;
}

export interface OpsState {
  project_id: string;
  rev: number;
  active_ops: number;
  revoked_ops: number;
  ops: OpRecord[];
  fold: {
    rev: number;
    active_ops: number;
    revoked_ops: number;
    effects: NleEffect[];
    caption_overrides: Record<string, Record<string, unknown>>;
    other: Array<Record<string, unknown>>;
  };
}

// --- selection service (agent-frame-editor step 3): tracked blur paths ---
export interface TrackHit {
  t_s: number;
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  interpolated: boolean;
}

export interface SelectionOut {
  backend: string;
  anchor_frame: number;
  frame_start: number;
  frame_end: number;
  n_hits: number;
  n_interpolated: number;
  mean_score: number;
  hits: TrackHit[];
  union_box: { x: number; y: number; w: number; h: number };
  op_id: string | null;
  detail: string;
}

export interface TimelineVersion {
  version_num: number;
  created_by: string;
  checkpoint_note: string;
  created_at: string | null;
}

export interface TimelineState {
  active_version: number | null;
  versions: TimelineVersion[];
}

export interface DispatchResult {
  ok: boolean;
  agent: string;
  tool: string;
  version_before: number | null;
  version: number | null;
  rolled_back_to: number | null;
  detail: string;
}

export interface Asset {
  id: string;
  role: string;
  duration_s: number;
  width: number;
  height: number;
  has_proxy: boolean;
}

export interface Cue {
  index: number;
  text: string;
  start_ms: number;
  end_ms: number;
  speaker: string;
}

export interface Redaction {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  sigma: number;
  label: string;
  created_by: string;
  created_at: string;
}

// --- WS4: multi-source case ingest + CaseDossier ---

/** Source payload accepted at case creation (POST /projects). */
export interface SourceIn {
  kind: "url" | "text";
  label?: string;
  url?: string;
  text?: string;
}

/** Source payload accepted by POST /projects/{id}/sources. */
export interface AddSourceIn {
  kind: "url" | "text" | "source_page";
  label?: string;
  url?: string;
  text?: string;
}

export type CaseSourceKind = "source_page" | "url" | "pdf" | "text";
export type CaseSourceStatus = "pending" | "extracting" | "ready" | "failed";

export interface CaseSourceOut {
  id: string;
  kind: CaseSourceKind;
  label: string;
  status: CaseSourceStatus;
  error: string | null;
  created_at: string;
}

export interface DossierFact {
  fact: string;
  date: string | null;
  source_id: string;
  citation: string;
}

export interface DossierSourceRef {
  source_id: string;
  kind: string;
  label: string;
  citation: string;
}

export interface CaseDossier {
  subject_name: string | null;
  age: number | null;
  role_or_agency: string | null;
  incident_datetime: string | null;
  location: string | null;
  inciting_reason: string | null;
  key_facts: DossierFact[];
  charges: string[];
  test_results: string[];
  verdict_outcome: string | null;
  sources: DossierSourceRef[];
}

export interface RefreshSourcesResult {
  refreshed: number;
  dossier: CaseDossier;
}

// --- Settings UI: provider credentials vault + live probes (ADR-0012) ---

export interface SettingsField {
  key: string;
  kind: "secret" | "text";
  label: string;
  placeholder: string;
  configured: boolean;
  source: "studio" | "env" | null;
  /** masked hint for secrets (raw values NEVER leave the API), full for text */
  value: string;
}

export interface SettingsProvider {
  id: string;
  label: string;
  hint: string;
  probe: string | null;
  fields: SettingsField[];
}

export interface ProvidersOut {
  ok: boolean;
  version: number;
  encrypted_at_rest: boolean;
  providers: SettingsProvider[];
}

export interface ProbeOut {
  ok: boolean;
  latency_ms: number;
  detail: string;
  extra: Record<string, string>;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    ...init,
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail ?? `${r.status} ${r.statusText}`);
  }
  // 204 / empty body (DELETE endpoints) — parsing json on an empty stream
  // throws "Unexpected end of JSON input" and the caller sees a fake failure
  if (r.status === 204 || r.headers.get("content-length") === "0") {
    return undefined as T;
  }
  const text = await r.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

export const api = {
  listProjects: () => req<Project[]>("/projects"),
  createProject: (
    title: string,
    opts?: {
      niche_skill?: string;
      video_urls?: string[];
      auto_discover?: boolean;
      sources?: SourceIn[];
    },
  ) => {
    const body: {
      title: string;
      niche_skill?: string;
      video_urls?: string[];
      auto_discover?: boolean;
      sources?: SourceIn[];
    } = { title, niche_skill: opts?.niche_skill ?? "bodycam_documentary" };
    if (opts?.video_urls) body.video_urls = opts.video_urls;
    // only sent when true — unchecked means "let the pipeline default"
    if (opts?.auto_discover) body.auto_discover = opts.auto_discover;
    if (opts?.sources) body.sources = opts.sources;
    return req<Project>("/projects", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },
  getProject: (id: string) => req<Project>(`/projects/${id}`),
  stopProject: (id: string) =>
    req<{ stopped: boolean; note: string }>(`/projects/${id}/stop`, { method: "POST" }),
  deleteProject: (id: string) => req<undefined>(`/projects/${id}`, { method: "DELETE" }),
  runPipeline: (id: string, videoUrl?: string, customHeaders?: Record<string, string>) =>
    req<RunResult>(`/projects/${id}/run`, {
      method: "POST",
      body: JSON.stringify({
        video_url: videoUrl || null,
        custom_headers: customHeaders ?? null,
      }),
    }),
  ingestUrl: (id: string, url: string, customHeaders?: Record<string, string>) =>
    req<{ job_id: string; source_path: string; enqueued: boolean; detail: string }>(
      `/projects/${id}/ingest`,
      {
        method: "POST",
        body: JSON.stringify({ url, custom_headers: customHeaders ?? {} }),
      },
    ),
  listJobs: (id: string) => req<Job[]>(`/projects/${id}/jobs`),
  timelineState: (id: string) => req<TimelineState>(`/projects/${id}/timeline`),
  // --- full-visibility Studio: live console + NLE timeline ---
  jobSnapshot: (id: string, jobId: string) =>
    req<LiveJobSnapshot>(`/projects/${id}/jobs/${jobId}/snapshot`),
  nleTimeline: (id: string) => req<NleTimeline>(`/projects/${id}/timeline/nle`),
  dispatchTool: (
    id: string,
    tool: string,
    args: Record<string, unknown>,
    checkpointNote = "",
  ) =>
    req<DispatchResult>(`/projects/${id}/agent/dispatch`, {
      method: "POST",
      body: JSON.stringify({ tool, args, checkpoint_note: checkpointNote }),
    }),
  publishDraft: (
    id: string,
    payload: {
      video_path: string;
      title: string;
      base_description?: string;
      srt_path?: string | null;
      chapter_titles?: string[];
      chapter_starts_s?: number[];
      tags?: string[];
    },
  ) =>
    req<{
      project_id: string;
      video_id: string;
      privacy_status: string;
      caption_attached: boolean;
    }>(`/projects/${id}/publish-draft`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  // --- M3 close-out: proxy player + subtitle grid + blur canvas
  listAssets: (id: string) => req<Asset[]>(`/projects/${id}/assets`),
  listSubtitles: (id: string) => req<Cue[]>(`/projects/${id}/subtitles`),
  editSubtitle: (
    id: string,
    index: number,
    text: string,
    speaker?: string,
    timing?: { start_ms: number; end_ms: number },
  ) =>
    req<Cue>(`/projects/${id}/subtitles/${index}`, {
      method: "PATCH",
      body: JSON.stringify({
        text,
        speaker: speaker ?? null,
        start_ms: timing?.start_ms ?? null,
        end_ms: timing?.end_ms ?? null,
      }),
    }),
  listRedactions: (id: string) => req<Redaction[]>(`/projects/${id}/redactions`),
  addRedaction: (
    id: string,
    box: { x: number; y: number; w: number; h: number; sigma?: number; label?: string },
  ) =>
    req<Redaction>(`/projects/${id}/redactions`, {
      method: "POST",
      body: JSON.stringify(box),
    }),
  deleteRedaction: (id: string, regionId: string) =>
    req<void>(`/projects/${id}/redactions/${regionId}`, { method: "DELETE" }),
  // --- op log v1: every edit is an op (agent or human); undo = revoke ---
  listOps: (id: string) => req<OpsState>(`/projects/${id}/ops`),
  createOp: (
    id: string,
    op: {
      kind: string;
      params?: Record<string, unknown>;
      origin?: "agent" | "human";
      evidence?: Record<string, unknown>;
      frame_start?: number | null;
      frame_end?: number | null;
      track?: string;
      note?: string;
    },
  ) =>
    req<OpRecord>(`/projects/${id}/ops`, {
      method: "POST",
      body: JSON.stringify(op),
    }),
  revokeOp: (id: string, opId: string, note = "") =>
    req<OpRecord>(`/projects/${id}/ops/${opId}/revoke`, {
      method: "POST",
      body: JSON.stringify({ note }),
    }),
  // manual adjust: a NEW op supersedes the old one (append-only law) —
  // corrected geometry + adjusts_op_id chain, old op revoked, coverage never gaps
  adjustOp: (
    id: string,
    opId: string,
    patch: {
      x: number;
      y: number;
      w: number;
      h: number;
      sigma?: number;
      frame_start?: number;
      frame_end?: number;
      note?: string;
    },
  ) =>
    req<OpRecord>(`/projects/${id}/ops/${opId}/adjust`, {
      method: "POST",
      body: JSON.stringify(patch),
    }),
  // --- chat agent v2 (roadmap step 5): one bounded orchestrator turn ---
  agentChat: (
    id: string,
    body: {
      message: string;
      niche?: string;
      history?: Array<{ role: string; content: string }>;
    },
  ) =>
    req<{
      ok: boolean;
      reply: string;
      rounds: number;
      calls: Array<{ tool: string; ok: boolean; detail: string }>;
      critic: Record<string, unknown> | null;
      version: number | null;
      detail: string;
    }>(`/projects/${id}/agent/chat`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  importLegacyOps: (id: string) =>
    req<{ imported: number; skipped: number; total_regions: number }>(
      `/projects/${id}/ops/import-legacy`,
      { method: "POST" },
    ),
  opsStreamUrl: (id: string, since = 0) =>
    `${API_BASE}/projects/${id}/ops/stream?since=${since}`,
  // --- selection service: promptable per-object tracking (step 3) ---
  createSelection: (
    assetId: string,
    body: {
      kind: "box" | "point";
      x: number;
      y: number;
      w?: number;
      h?: number;
      anchor_frame: number;
      frame_start: number;
      frame_end: number;
      backend?: string;
      create_blur_op?: boolean;
      sigma?: number;
      label?: string;
    },
  ) =>
    req<SelectionOut>(`/assets/${assetId}/selection`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getOp: (opId: string) => req<OpRecord>(`/ops/${opId}`),
  // --- export gate (step 6): the render stamps the cursor server-side ---
  exportGate: (id: string) =>
    req<{
      ops_rev: number;
      baked_ops_rev: number | null;
      active_version?: number | null;
      baked_timeline_version?: number | null;
      dirty: boolean;
      pending_ops: number;
      exported_once: boolean;
    }>(`/projects/${id}/export/gate`),
  selectionBackends: () =>
    req<Record<string, boolean>>(`/selection/backends`),
  // --- WS4: multi-source case ingest + CaseDossier
  listSources: (id: string) => req<CaseSourceOut[]>(`/projects/${id}/sources`),
  addSource: (id: string, body: AddSourceIn) =>
    req<CaseSourceOut>(`/projects/${id}/sources`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  uploadSource: async (id: string, file: File, label?: string) => {
    // dedicated fetch — the multipart boundary must be set by the browser,
    // so this must not go through the JSON `req` helper.
    const form = new FormData();
    form.append("file", file);
    if (label) form.append("label", label);
    const r = await fetch(`${API_BASE}/projects/${id}/sources/upload`, {
      method: "POST",
      body: form,
      cache: "no-store",
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail ?? `${r.status} ${r.statusText}`);
    }
    return r.json() as Promise<CaseSourceOut>;
  },
  getDossier: (id: string) => req<CaseDossier>(`/projects/${id}/dossier`),
  refreshSources: (id: string) =>
    req<RefreshSourcesResult>(`/projects/${id}/sources/refresh`, {
      method: "POST",
    }),
  // --- Settings: credentials vault + connection probes
  listProviders: () => req<ProvidersOut>("/settings/providers"),
  saveCredentials: (entries: { key: string; value: string }[]) =>
    req<ProvidersOut>("/settings/credentials", {
      method: "PUT",
      body: JSON.stringify({ entries }),
    }),
  deleteCredential: (key: string) =>
    req<ProvidersOut>(`/settings/credentials/${encodeURIComponent(key)}`, {
      method: "DELETE",
    }),
  testProvider: (provider: string, params: Record<string, string>) =>
    req<ProbeOut>("/settings/test", {
      method: "POST",
      body: JSON.stringify({ provider, params }),
    }),
};

export function assetProxyUrl(assetId: string): string {
  return `${API_BASE}/assets/${assetId}/proxy`;
}

export function assetScrubUrl(assetId: string): string {
  return `${API_BASE}/assets/${assetId}/scrubsheet`;
}

export function srtDownloadUrl(projectId: string): string {
  return `${API_BASE}/projects/${projectId}/subtitles/srt`;
}

export function sseUrl(projectId: string, jobId: string): string {
  return `${API_BASE}/projects/${projectId}/jobs/${jobId}/events?poll_s=0.5&max_seconds=600`;
}

export function liveSseUrl(projectId: string, jobId: string): string {
  return `${API_BASE}/projects/${projectId}/jobs/${jobId}/live?poll_s=0.4&max_seconds=7200`;
}
