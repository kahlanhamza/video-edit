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
  return r.json() as Promise<T>;
}

export const api = {
  listProjects: () => req<Project[]>("/projects"),
  createProject: (title: string, nicheSkill = "bodycam_documentary") =>
    req<Project>("/projects", {
      method: "POST",
      body: JSON.stringify({ title, niche_skill: nicheSkill }),
    }),
  getProject: (id: string) => req<Project>(`/projects/${id}`),
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
