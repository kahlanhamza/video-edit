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
};

export function sseUrl(projectId: string, jobId: string): string {
  return `${API_BASE}/projects/${projectId}/jobs/${jobId}/events?poll_s=0.5&max_seconds=600`;
}
