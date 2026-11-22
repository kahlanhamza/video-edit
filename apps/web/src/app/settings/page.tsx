"use client";

import { useCallback, useEffect, useState } from "react";
import { api, ProbeOut, ProvidersOut, SettingsProvider } from "@/lib/api";

/**
 * Settings — provider credentials vault (ADR-0012).
 *
 * One card per provider: paste keys, TEST the live connection, SAVE into the
 * vault (encrypted at rest when SETTINGS_SECRET_KEY is set). Saved keys
 * override .env at runtime — no restart: the gateway reloads the overlay on
 * save, the worker fleet re-reads it per pipeline job.
 */

interface ProbeState {
  status: "idle" | "loading" | ProbeOut;
}

// OpenAI-compatible presets — one click fills Base URL + Model.
const LLM_PRESETS: { name: string; base: string; model: string; note: string }[] = [
  { name: "Meta Muse", base: "https://api.meta.ai/v1", model: "muse-ember-1.0", note: "default, cheap tool loops" },
  { name: "Google AI Studio", base: "", model: "gemini-3.8-flash", note: "native Gemini — Base URL stays blank" },
  { name: "OpenRouter", base: "https://openrouter.ai/api/v1", model: "vendor/model:free", note: "free models rotate" },
  { name: "Groq", base: "https://api.groq.com/openai/v1", model: "llama-3.3-70b-versatile", note: "fast free tier" },
  { name: "DeepSeek", base: "https://api.deepseek.com/v1", model: "deepseek-chat", note: "cheap reasoning" },
  { name: "Ollama (local)", base: "http://localhost:11434/v1", model: "llama3.1", note: "no key needed" },
  { name: "LM Studio (local)", base: "http://localhost:1234/v1", model: "local-model", note: "whatever you loaded" },
];

function SourceBadge({ source }: { source: "studio" | "env" | null }) {
  if (source === "studio")
    return (
      <span className="rounded bg-emerald-900/60 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300">
        studio
      </span>
    );
  if (source === "env")
    return (
      <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400">.env</span>
    );
  return (
    <span className="rounded bg-slate-900 px-1.5 py-0.5 text-[10px] text-slate-600">unset</span>
  );
}

function ProbeChip({ state }: { state: ProbeState }) {
  if (state.status === "idle") return null;
  if (state.status === "loading")
    return <span className="text-xs text-amber-400">testing connection…</span>;
  const p = state.status;
  return (
    <div
      className={`mt-2 rounded border px-2 py-1.5 text-xs ${
        p.ok
          ? "border-emerald-700 bg-emerald-950/40 text-emerald-300"
          : "border-red-800 bg-red-950/40 text-red-300"
      }`}
    >
      <span className="font-semibold">{p.ok ? "CONNECTED" : "FAILED"}</span>
      {` · ${p.latency_ms} ms · ${p.detail}`}
      {Object.keys(p.extra).length > 0 && (
        <span className="text-slate-400">
          {" "}
          ·{" "}
          {Object.entries(p.extra)
            .map(([k, v]) => `${k}: ${v}`)
            .join(", ")}
        </span>
      )}
    </div>
  );
}

export default function SettingsPage() {
  const [catalog, setCatalog] = useState<ProvidersOut | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [probes, setProbes] = useState<Record<string, ProbeState>>({});
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setCatalog(await api.listProviders());
      setNote(null);
    } catch (e) {
      setNote(`gateway unreachable (${e instanceof Error ? e.message : e})`);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function setValue(key: string, value: string) {
    setValues((v) => ({ ...v, [key]: value }));
  }

  async function test(provider: SettingsProvider) {
    if (!provider.probe) return;
    setProbes((s) => ({ ...s, [provider.id]: { status: "loading" } }));
    try {
      // typed values override saved ones — the button tests what WILL be used
      const params: Record<string, string> = {};
      for (const f of provider.fields) {
        const typed = (values[f.key] ?? "").trim();
        if (typed) params[f.key] = typed;
      }
      const result = await api.testProvider(provider.probe, params);
      setProbes((s) => ({ ...s, [provider.id]: { status: result } }));
    } catch (e) {
      setProbes((s) => ({
        ...s,
        [provider.id]: {
          status: {
            ok: false,
            latency_ms: 0,
            detail: e instanceof Error ? e.message : String(e),
            extra: {},
          },
        },
      }));
    }
  }

  async function save(provider: SettingsProvider) {
    const entries = provider.fields
      .map((f) => ({ key: f.key, value: (values[f.key] ?? "").trim() }))
      .filter((e) => e.value !== "");
    if (entries.length === 0) {
      setNote(`${provider.label}: nothing to save (type a value first)`);
      return;
    }
    setBusy(true);
    try {
      const out = await api.saveCredentials(entries);
      setCatalog(out);
      // clear the typed box for what was just saved
      setValues((v) => {
        const next = { ...v };
        for (const e of entries) delete next[e.key];
        return next;
      });
      setProbes((s) => ({ ...s, [provider.id]: { status: "idle" } }));
      setNote(
        `saved ${entries.length} key(s) for ${provider.label} — active immediately (workers pick it up on the next job)`,
      );
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function clearField(key: string) {
    setBusy(true);
    try {
      const out = await api.deleteCredential(key);
      setCatalog(out);
      setNote(`removed studio override for ${key} (falls back to .env)`);
    } catch (e) {
      setNote(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function applyPreset(preset: (typeof LLM_PRESETS)[number]) {
    setValues((v) => ({
      ...v,
      LLM_BASE_URL: preset.base,
      LLM_MODEL: preset.model,
    }));
    setNote(`preset applied: ${preset.name} (${preset.note}) — paste your key, then Test`);
  }

  return (
    <div className="space-y-6">
      <section className="card">
        <h1 className="text-lg font-bold">Settings — provider connections</h1>
        <p className="mt-1 text-sm text-slate-400">
          Paste keys, hit <span className="text-slate-200">Test connection</span> to verify the
          live endpoint, then <span className="text-slate-200">Save</span>. Saved keys override
          .env at runtime — no restart needed. Secrets are stored server-side and never returned
          to the browser (masked hints only).
        </p>
        {catalog && !catalog.encrypted_at_rest && (
          <p className="mt-2 rounded border border-slate-800 bg-slate-950/60 px-2 py-1.5 text-xs text-slate-400">
            Vault at rest: <span className="text-amber-400">plaintext</span> (same threat model as
            your .env file). Set <code className="text-slate-300">SETTINGS_SECRET_KEY</code> in
            .env to encrypt stored keys.
          </p>
        )}
        {note && <p className="mt-2 text-xs text-amber-300">{note}</p>}
      </section>

      {!catalog && <section className="card text-sm text-slate-400">loading providers…</section>}

      {catalog?.providers.map((provider) => (
        <section key={provider.id} className="card">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="text-sm font-bold text-slate-100">{provider.label}</h2>
              <p className="mt-1 max-w-3xl text-xs text-slate-400">{provider.hint}</p>
            </div>
            {provider.probe && (
              <button className="btn-ghost shrink-0" onClick={() => test(provider)} disabled={busy}>
                Test connection
              </button>
            )}
          </div>

          {provider.id === "llm" && (
            <div className="mt-3">
              <div className="text-xs font-semibold text-slate-400">Presets</div>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {LLM_PRESETS.map((p) => (
                  <button
                    key={p.name}
                    className="btn-ghost !px-2 !py-1 text-xs"
                    onClick={() => applyPreset(p)}
                    title={p.note}
                  >
                    {p.name}
                  </button>
                ))}
              </div>
              <p className="mt-1.5 text-[11px] text-slate-500">
                Any endpoint that speaks the OpenAI chat-completions wire format connects here.
                Agent-style task APIs (e.g. Manus&apos; task API) are not chat-completions — a
                gateway that exposes one does.
              </p>
            </div>
          )}

          <div className="mt-3 space-y-2">
            {provider.fields.map((f) => (
              <div key={f.key}>
                <div className="mb-1 flex items-center gap-2">
                  <label className="text-xs font-medium text-slate-300">{f.label}</label>
                  <SourceBadge source={f.source} />
                  {f.kind === "secret" && f.source === "studio" && (
                    <button
                      className="text-[10px] text-slate-500 underline hover:text-red-400"
                      onClick={() => clearField(f.key)}
                      disabled={busy}
                    >
                      remove studio override
                    </button>
                  )}
                </div>
                <input
                  className="input"
                  type={f.kind === "secret" ? "password" : "text"}
                  placeholder={
                    f.kind === "secret" && f.configured
                      ? `saved: ${f.value} — type a new key to replace`
                      : f.placeholder || f.label
                  }
                  value={values[f.key] ?? ""}
                  onChange={(e) => setValue(f.key, e.target.value)}
                  autoComplete="off"
                />
              </div>
            ))}
          </div>

          <div className="mt-3 flex items-center gap-2">
            <button className="btn" onClick={() => save(provider)} disabled={busy}>
              Save
            </button>
            <span className="text-xs text-slate-500">
              blank fields are left untouched — saved keys win over .env
            </span>
          </div>

          <ProbeChip state={probes[provider.id] ?? { status: "idle" }} />
        </section>
      ))}
    </div>
  );
}
