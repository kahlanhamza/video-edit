"use client";

// Settings — provider connections vault (ADR-0012), designed like a real
// product settings surface:
//   - one card per provider, status dot + last probe result shown instantly
//     (persisted locally; re-test any time with the SAVED key — no re-entry)
//   - typed values override saved ones: Test tests what WILL be used
//   - Save writes the vault; workers pick keys up on the next job, no restart

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api, ProbeOut, ProvidersOut, SettingsProvider } from "@/lib/api";
import { Icon } from "@/components/studio/icons";

interface ProbeState {
  status: "idle" | "loading" | ProbeOut;
}

// last known connection state per provider, so a revisit shows Connected
// without touching the network (localStorage key)
const LS_KEY = "directoros.probe.v1";

function loadProbeMemory(): Record<string, ProbeOut> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    return raw ? (JSON.parse(raw) as Record<string, ProbeOut>) : {};
  } catch {
    return {};
  }
}

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
    return <span className="chip chip-ok">saved</span>;
  if (source === "env") return <span className="chip">.env</span>;
  return <span className="chip">unset</span>;
}

function ProbeResult({ state }: { state: ProbeState }) {
  if (state.status === "idle") return null;
  if (state.status === "loading")
    return (
      <p className="mt-3 flex items-center gap-2 text-[11.5px] text-[var(--tx-2)]">
        <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--acc)]" />
        testing connection…
      </p>
    );
  const p = state.status;
  return (
    <div
      className={`mt-3 rounded-md border px-3 py-2 text-[11.5px] leading-relaxed ${
        p.ok
          ? "border-[rgba(63,214,143,.3)] bg-[rgba(63,214,143,.07)] text-[var(--ok)]"
          : "border-[rgba(241,109,109,.3)] bg-[rgba(241,109,109,.07)] text-[var(--err)]"
      }`}
    >
      <span className="font-semibold">{p.ok ? "Connected" : "Failed"}</span>
      {` · ${p.latency_ms} ms · ${p.detail}`}
      {Object.keys(p.extra).length > 0 && (
        <span className="text-[var(--tx-3)]">
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
  const [memory, setMemory] = useState<Record<string, ProbeOut>>({});
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [unreachable, setUnreachable] = useState<string | null>(null);

  useEffect(() => {
    setMemory(loadProbeMemory());
  }, []);

  const refresh = useCallback(async () => {
    try {
      setCatalog(await api.listProviders());
      setUnreachable(null);
      setNote(null);
    } catch (e) {
      setUnreachable(e instanceof Error ? e.message : String(e));
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
      // typed values override saved ones — the button tests what WILL be used.
      // with empty inputs the SAVED vault key is tested (no re-entry needed).
      const params: Record<string, string> = {};
      for (const f of provider.fields) {
        const typed = (values[f.key] ?? "").trim();
        if (typed) params[f.key] = typed;
      }
      const result = await api.testProvider(provider.probe, params);
      setProbes((s) => ({ ...s, [provider.id]: { status: result } }));
      setMemory((m) => {
        const next = { ...m, [provider.id]: result };
        try {
          window.localStorage.setItem(LS_KEY, JSON.stringify(next));
        } catch {
          /* storage full/blocked — fine */
        }
        return next;
      });
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
      setValues((v) => {
        const next = { ...v };
        for (const e of entries) delete next[e.key];
        return next;
      });
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
    <div className="flex h-screen flex-col overflow-hidden bg-[var(--bg-0)]">
      {/* ---- top bar ---- */}
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-[var(--line-1)] bg-[var(--bg-1)] px-4">
        <Link href="/" className="icon-btn" title="back to cases">
          <Icon name="back" />
        </Link>
        <span className="flex h-6 w-6 items-center justify-center rounded bg-[var(--acc-bg)] text-[var(--acc)]">
          <Icon name="clapper" size={13} />
        </span>
        <span className="text-[13px] font-semibold tracking-wide text-[var(--tx-1)]">
          DirectorOS <span className="font-normal text-[var(--tx-3)]">Studio</span>
        </span>
        <span className="chip">settings</span>
        <div className="flex-1" />
        <span className="chip">{catalog?.providers.length ?? "…"} providers</span>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-4 p-6">
          <section>
            <h1 className="text-[15px] font-semibold text-[var(--tx-1)]">
              Provider connections
            </h1>
            <p className="mt-1 text-[12px] leading-relaxed text-[var(--tx-2)]">
              Paste keys, hit <span className="text-[var(--tx-1)]">Test</span> to
              verify the live endpoint, then <span className="text-[var(--tx-1)]">Save</span>.
              Saved keys override .env at runtime — no restart. Secrets stay
              server-side (masked hints only) and are tested from the vault on
              every revisit: <span className="text-[var(--tx-1)]">no re-entry needed</span>.
            </p>
            {catalog && !catalog.encrypted_at_rest && (
              <p className="mt-2 rounded-md border border-[rgba(245,181,68,.25)] bg-[rgba(245,181,68,.07)] px-3 py-2 text-[11.5px] leading-relaxed text-[var(--warn)]">
                Vault at rest: plaintext (same threat model as your .env file).
                Set <code>SETTINGS_SECRET_KEY</code> in .env to encrypt stored keys.
              </p>
            )}
            {note && <p className="mt-2 text-[11.5px] text-[var(--tx-2)]">{note}</p>}
          </section>

          {!catalog && unreachable && (
            <section className="panel flex flex-col items-center gap-3 p-10 text-center">
              <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-[var(--bg-2)] text-[var(--err)]">
                <Icon name="x" size={22} />
              </span>
              <div>
                <p className="text-[13px] font-medium text-[var(--tx-1)]">
                  Gateway unreachable
                </p>
                <p className="mt-1 max-w-[360px] break-words text-[11px] leading-relaxed text-[var(--tx-3)]">
                  {unreachable} — start the gateway (make api) and retry.
                </p>
              </div>
              <button className="btn" onClick={refresh}>
                Retry
              </button>
            </section>
          )}

          {!catalog && !unreachable && (
            <section className="panel flex items-center justify-center gap-2.5 p-10 text-[12px] text-[var(--tx-3)]">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--acc)]" />
              loading providers…
            </section>
          )}

          {catalog?.providers.map((provider) => {
            const mem = memory[provider.id];
            const state = probes[provider.id] ??
              (mem ? { status: mem } : { status: "idle" as const });
            const connected = mem?.ok === true;
            const failed = mem?.ok === false;
            const configured = provider.fields.some((f) => f.configured);
            const dotClass = connected
              ? "bg-[var(--ok)]"
              : failed
                ? "bg-[var(--err)]"
                : configured
                  ? "bg-[var(--warn)]"
                  : "bg-[var(--tx-3)]";
            const dotTitle = connected
              ? "last test: connected"
              : failed
                ? "last test: failed"
                : configured
                  ? "keys saved — not tested yet"
                  : "no keys";
            return (
              <section key={provider.id} className="panel">
                <div className="panel-header !h-11">
                  <span
                    className={`h-2 w-2 shrink-0 rounded-full ${dotClass}`}
                    title={dotTitle}
                  />
                  <h2 className="truncate text-[12.5px] font-semibold text-[var(--tx-1)]">
                    {provider.label}
                  </h2>
                  <span className="flex-1" />
                  {provider.probe && (
                    <button
                      className="btn-ghost !h-7 !text-[11px]"
                      onClick={() => test(provider)}
                      disabled={busy}
                    >
                      Test connection
                    </button>
                  )}
                </div>

                <div className="p-4">
                  <p className="text-[11.5px] leading-relaxed text-[var(--tx-2)]">
                    {provider.hint}
                  </p>

                  {provider.id === "llm" && (
                    <div className="mt-3">
                      <div className="micro-label">Presets</div>
                      <div className="mt-1.5 flex flex-wrap gap-1.5">
                        {LLM_PRESETS.map((p) => (
                          <button
                            key={p.name}
                            className="btn-ghost !h-6 !px-2 !text-[11px]"
                            onClick={() => applyPreset(p)}
                            title={p.note}
                          >
                            {p.name}
                          </button>
                        ))}
                      </div>
                      <p className="mt-1.5 text-[11px] leading-relaxed text-[var(--tx-3)]">
                        Any endpoint that speaks the OpenAI chat-completions wire
                        format connects here. Blank Base URL = native Google AI
                        Studio (tested with the Gemini REST probe).
                      </p>
                    </div>
                  )}

                  <div className="mt-3 grid gap-2.5 sm:grid-cols-2">
                    {provider.fields.map((f) => (
                      <div key={f.key}>
                        <div className="mb-1 flex items-center gap-2">
                          <label className="text-[11.5px] font-medium text-[var(--tx-1)]">
                            {f.label}
                          </label>
                          <SourceBadge source={f.source} />
                          {f.kind === "secret" && f.source === "studio" && (
                            <button
                              className="ml-auto text-[10px] text-[var(--tx-3)] underline hover:text-[var(--err)]"
                              onClick={() => clearField(f.key)}
                              disabled={busy}
                            >
                              remove
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

                  <div className="mt-3 flex items-center gap-3">
                    <button className="btn !h-7 !text-[11px]" onClick={() => save(provider)} disabled={busy}>
                      Save
                    </button>
                    <span className="text-[11px] text-[var(--tx-3)]">
                      blank fields are left untouched — saved keys win over .env
                    </span>
                  </div>

                  <ProbeResult state={state} />
                </div>
              </section>
            );
          })}
        </div>
      </div>
    </div>
  );
}
