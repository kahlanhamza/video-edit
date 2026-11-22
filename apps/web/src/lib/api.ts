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
  editSubtitle: (id: string, index: number, text: string, speaker?: string) =>
    req<Cue>(`/projects/${id}/subtitles/${index}`, {
      method: "PATCH",
      body: JSON.stringify({ text, speaker: speaker ?? null }),
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
