# gpu_services/sam2 — SAM 2 local weight service

Promptable video segmentation (Meta SAM 2.1) as a **local weight service**:
one container owns the checkpoint, the Studio talks to it over HTTP, torch
never enters the default render path (SKILL law).

## Why a service

- **Zero heavy SDKs in the default path** — `packages/vision_fx` keeps torch
  out; the selection port (`vision_fx.selection`) gains an HTTP adapter
  (`Sam2ServiceBackend`) instead of an import.
- **Weights are durable** — the checkpoint lives on the `sam2weights` named
  volume: download once (`POST /weights/ensure`), restart forever.
- **Hermetic CI** — the service's tests mock the predictor and the client's
  tests mock the transport; no weights, no GPU, no network in CI.
- **Niche fit** — narration/commentary editing needs subject tracking for
  blur/redaction/punch-ins; SAM 2 is the occlusion-tolerant tracker behind
  the Studio's "click a thing, tell the agent what to do with it" loop
  (frame-level editing vision, PLAN §3).

## Endpoints

| Method | Path             | What                                                      |
| ------ | ---------------- | --------------------------------------------------------- |
| GET    | `/health`        | weights present? sha256? device? tracker loaded?          |
| POST   | `/weights/ensure`| idempotent download of the checkpoint (atomic, hashed)    |
| POST   | `/track`         | box/point prompt on one frame -> per-frame RAW-pixel boxes|

`/track` returns the exact hit shape the Studio's `TrackedSelection`
consumes (`t_s, x, y, w, h, score`) — downstream blur/redaction cannot tell
SAM 2 from the ncc-cpu backend.

## Run (docker compose)

```bash
docker compose --profile gpu up -d --build sam2
curl -s localhost:8300/health
curl -s -X POST localhost:8300/weights/ensure   # ~160 MB, once
```

Then set `SAM2_SERVICE_URL=http://sam2:8300` in `.env` (the compose wiring
already exports it to `api` + `worker`) and the Studio's selection panel
shows `sam2` as available. Without the service, everything falls back to
detector-assist / ncc-cpu exactly as before.

Manual trim knobs (`.env`): `SAM2_WEIGHTS_URL` (default: Meta sam2.1
Hiera-Tiny), `SAM2_WEIGHTS_SHA256` (pin the checkpoint),
`SAM2_WEIGHTS_DIR` / `SAM2_WEIGHTS` (host-side runs).
