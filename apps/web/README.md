# apps/web — Director Studio (Milestone 3)

Human review surface over the DirectorOS gateway. Next.js 15 + React 19 +
Tailwind v4. The backend it talks to is `apps/api` (FastAPI, port 8000).

## Run

```bash
make api            # FastAPI gateway on :8000
cd apps/web
npm install
npm run dev         # Studio on :3000
```

`NEXT_PUBLIC_API_BASE` overrides the gateway URL (default
`http://localhost:8000`).

## What it covers

- **Cases** — create/list projects (`POST/GET /projects`)
- **Pipeline progress** — live SSE stream of the newest job
  (`GET /projects/{id}/jobs/{id}/events`), stage + percent + error banner
- **Timeline versions** — durable checkpoint list with the active pointer
  (`GET /projects/{id}/timeline`)
- **Director tools** — one-click checkpoint-first dispatches
  (`POST /projects/{id}/agent/dispatch`); failed calls show their automatic
  rollback target
- **Human Review Gate** — publish the approved cut as a YouTube Private Draft
  (`POST /projects/{id}/publish-draft`)

M3 acceptance (issue #11): operator can create a case, watch the pipeline run
live, mutate the timeline through tools with visible rollback semantics, and
approve the cut to a private draft — all from the browser.
