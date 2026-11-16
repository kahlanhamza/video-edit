"use client";

import { useCallback, useEffect, useState } from "react";
import { api, Project } from "@/lib/api";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [title, setTitle] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

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

  async function create() {
    if (!title.trim()) return;
    setBusy(true);
    try {
      await api.createProject(title.trim());
      setTitle("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <section className="card">
        <h1 className="text-lg font-bold">Cases</h1>
        <p className="mt-1 text-sm text-slate-400">
          One case = one bodycam source, one autonomous pipeline run, one human
          review gate.
        </p>
        <div className="mt-3 flex gap-2">
          <input
            className="input"
            placeholder="New case title (e.g. 2024-03-11 traffic stop FOIA)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && create()}
          />
          <button className="btn" onClick={create} disabled={busy || !title.trim()}>
            {busy ? "Creating…" : "Create case"}
          </button>
        </div>
        {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
      </section>

      <section className="space-y-2">
        {projects.length === 0 && !error && (
          <p className="text-sm text-slate-500">No cases yet.</p>
        )}
        {projects.map((p) => (
          <a
            key={p.id}
            href={`/projects/${p.id}`}
            className="card block hover:border-amber-600"
          >
            <div className="flex items-center justify-between">
              <span className="font-semibold">{p.title}</span>
              <span className="rounded bg-slate-800 px-2 py-0.5 text-xs uppercase tracking-wide text-slate-300">
                {p.status}
              </span>
            </div>
            <div className="mt-1 text-xs text-slate-500">
              {p.niche_skill} · {p.id.slice(0, 8)}…
            </div>
          </a>
        ))}
      </section>
    </div>
  );
}
