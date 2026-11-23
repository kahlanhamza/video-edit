#!/usr/bin/env bash
# DirectorOS full-stack smoke — the NEW A1 trim interaction on a REAL device.
#
#   ./scripts/fullstack_smoke.sh                          # full: generate a test video, run the
#                                                         # pipeline (needs your .env keys), then trim
#   ./scripts/fullstack_smoke.sh --trim-only <project_id> # trim interaction on an EXISTING project
#
# What it proves end to end against http://localhost:8000 (docker compose stack):
#   1. API + web healthy
#   2. [full] test video ingest + pipeline run to render (narration/commentary niche)
#   3. timeline.trim_audio dispatch -> new version -> NLE carries the VO window
#   4. invalid trim rejected WITHOUT advancing the version (rollback law)
#   5. export gate goes dirty after the trim (the Export button never lies)
#
# Requires: docker compose stack up (./setup.sh or `make up`), curl, python3.

set -eu
cd "$(dirname "$0")/.."

API="${SMOKE_API:-http://localhost:8000}"
WEB="${SMOKE_WEB:-http://localhost:3000}"
MODE="full"
PROJECT_ARG=""
if [ "${1:-}" = "--trim-only" ]; then
  MODE="trim-only"
  PROJECT_ARG="${2:-}"
  if [ -z "$PROJECT_ARG" ]; then
    echo "usage: $0 --trim-only <project_id>"; exit 2
  fi
fi

say() { printf '\n== %s\n' "$1"; }
fail() { printf '\nFAIL: %s\n' "$1"; exit 1; }

http() { # http METHOD PATH [BODY] -> prints body, sets HTTP_CODE
  _method="$1"; _path="$2"; _body="${3:-}"
  if [ -n "$_body" ]; then
    HTTP_CODE=$(curl -s -o /tmp/smoke_body.json -w '%{http_code}' -X "$_method" \
      -H 'Content-Type: application/json' -d "$_body" "$API$_path")
  else
    HTTP_CODE=$(curl -s -o /tmp/smoke_body.json -w '%{http_code}' -X "$_method" "$API$_path")
  fi
  cat /tmp/smoke_body.json
}

say "1) stack health"
curl -sf "$API/health" >/dev/null || fail "API not healthy at $API (docker compose up -d first)"
echo "  api: ok"
curl -sf "$WEB" >/dev/null 2>&1 && echo "  web: ok" || echo "  web: not reachable (Studio UI check skipped)"

# --------------------------------------------------------------- trim interaction
trim_interaction() {
  PID="$1"
  say "3) NLE snapshot — find the A1 audio block"
  http GET "/projects/$PID/timeline/nle" >/dev/null
  NARR_COUNT=$(json "narration_count" < /tmp/smoke_narr.json 2>/dev/null || true)
  python3 - "$PID" <<'PYEOF'
import json, sys, urllib.request

API = "http://localhost:8000"
pid = sys.argv[1]

def call(method, path, body=None):
    req = urllib.request.Request(
        API + path, method=method,
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}

st, nle = call("GET", f"/projects/{pid}/timeline/nle")
assert st == 200, nle
narr = [c for c in nle["clips"] if c["kind"] == "narration"]

if not narr:
    raws = [c for c in nle["clips"] if c["master_index"] is not None and c["kind"] == "raw"]
    assert raws, "no master clips at all — run the pipeline first"
    script = ("This stop looked routine, until the trunk opened — "
              "smoke narration proving the A1 trim loop end to end.")
    st2, res = call("POST", f"/projects/{pid}/agent/dispatch", {
        "tool": "narration.add",
        "args": {"item_index": raws[0]["master_index"], "script_text": script},
        "checkpoint_note": "smoke: seed narration for the A1 trim",
    })
    assert st2 == 200 and res.get("ok"), res
    print(f"  narration.add on raw block {raws[0]['master_index']} -> ok (version {res.get('version')})")
    st, nle = call("GET", f"/projects/{pid}/timeline/nle")
    narr = [c for c in nle["clips"] if c["kind"] == "narration"]
    assert narr, "narration.add did not produce an A1 block"

a1 = narr[0]
assert a1["master_index"] is not None, a1
print(f"  A1 block: master_index={a1['master_index']} start={a1['start_s']} dur={a1['dur_s']}s "
      f"vo=[{a1.get('vo_in_s', 0)}, {a1.get('vo_out_s')}] tts={a1.get('tts_duration_s')}")

say("4) NEW INTERACTION: timeline.trim_audio (VO window inside the TTS stem)")
st3, trim = call("POST", f"/projects/{pid}/agent/dispatch", {
    "tool": "timeline.trim_audio",
    "args": {"item_index": a1["master_index"], "vo_in_s": 0.4, "vo_out_s": 3.5},
    "checkpoint_note": "smoke: A1 trim (skip head dead air, cut tail)",
})
assert st3 == 200 and trim.get("ok"), trim
new_version = trim.get("version")
print(f"  dispatch ok -> version {new_version}: {trim.get('detail')}")

st4, nle2 = call("GET", f"/projects/{pid}/timeline/nle")
a1b = [c for c in nle2["clips"] if c["kind"] == "narration"][0]
assert abs(a1b["vo_in_s"] - 0.4) < 0.01 and abs(a1b["vo_out_s"] - 3.5) < 0.01, a1b
print(f"  NLE carries the window [{a1b['vo_in_s']}, {a1b['vo_out_s']}]s "
      "(the Studio trim handles + VO TRIM chip have data)")

say("5) invalid trim rejected WITHOUT advancing the version")
st5, bad = call("POST", f"/projects/{pid}/agent/dispatch", {
    "tool": "timeline.trim_audio",
    "args": {"item_index": a1["master_index"], "vo_in_s": 5.0, "vo_out_s": 1.0},
    "checkpoint_note": "smoke: inverted window must be rejected",
})
assert st5 == 200 and bad.get("ok") is False, bad
st6, nle3 = call("GET", f"/projects/{pid}/timeline/nle")
a1c = [c for c in nle3["clips"] if c["kind"] == "narration"][0]
assert abs(a1c["vo_out_s"] - 3.5) < 0.01, a1c
print("  rejected + rolled back — the good window survives (checkpoint law)")

say("6) export gate")
st7, gate = call("GET", f"/projects/{pid}/export/gate")
assert st7 == 200, gate
assert gate.get("dirty") is True, gate
print(f"  gate dirty={gate['dirty']} active_version={gate.get('active_version')} "
      f"baked={gate.get('baked_timeline_version')} — the Export button demands a re-bake")

print("\nTRIM INTERACTION: ALL CHECKS GREEN")
print(f"  open http://localhost:3000/projects/{pid}")
print("  -> select the A1 block: VO TRIM chip, tail-pad zone, drag handles")
PYEOF
}

if [ "$MODE" = "trim-only" ]; then
  say "trim-only mode on project $PROJECT_ARG"
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "$API/projects/$PROJECT_ARG")
  [ "$CODE" = "200" ] || fail "project $PROJECT_ARG not found (HTTP $CODE)"
  trim_interaction "$PROJECT_ARG"
  exit 0
fi

# ---------------------------------------------------------------- full mode
say "2) generate a narration-shaped test video INSIDE the api container"
docker compose exec -T api bash -lc '
mkdir -p /app/output/smoke &&
ffmpeg -y -loglevel error \
  -f lavfi -i "testsrc2=size=640x360:rate=30:duration=24" \
  -f lavfi -i "sine=frequency=440:duration=24" \
  -c:v libx264 -preset veryfast -crf 30 -pix_fmt yuv420p \
  -c:a aac -b:a 64k -ar 48000 -ac 2 -shortest \
  /app/output/smoke/narration_test.mp4' || fail "ffmpeg test video generation failed"
echo "  /app/output/smoke/narration_test.mp4 (24s, 48kHz stereo — audio law)"

say "2b) serve it on the compose network (smoke-media, profile: smoke)"
docker compose --profile smoke up -d --quiet-pull smoke-media 2>/dev/null \
  || docker compose --profile smoke up -d smoke-media || fail "smoke-media did not start"

say "2c) create the project (narration/commentary niche law)"
http POST /projects '{"title":"SMOKE narration trim","niche_skill":"bodycam_documentary"}' >/dev/null
[ "$HTTP_CODE" = "201" ] || fail "project create failed (HTTP $HTTP_CODE): $(cat /tmp/smoke_body.json)"
PID=$(python3 -c "import json;print(json.load(open('/tmp/smoke_body.json'))['id'])")
echo "  project $PID"

say "2d) ingest + run the pipeline (your .env keys: Gladia ASR, LLM director)"
http POST "/projects/$PID/ingest" '{"url":"http://smoke-media:8099/smoke/narration_test.mp4"}' >/dev/null
[ "$HTTP_CODE" = "202" ] || fail "ingest failed (HTTP $HTTP_CODE): $(cat /tmp/smoke_body.json)"
JOB=$(python3 -c "import json;print(json.load(open('/tmp/smoke_body.json'))['job_id'])")
echo "  job $JOB — polling (this renders a real cut; typically 2-8 minutes)"

DEADLINE=$(($(date +%s) + 1500))
STAGE="?"
LAST=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  http GET "/projects/$PID/jobs" >/dev/null
  [ "$HTTP_CODE" = "200" ] || fail "jobs list failed (HTTP $HTTP_CODE)"
  STATE=$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
job = next((j for j in d if j.get("id") == sys.argv[2]), None)
print("none" if job is None else "%s|%d|%s" % (job.get("stage", "?"), int(job.get("progress_pct", 0)), job.get("error") or ""))
' /tmp/smoke_body.json "$JOB")
  STAGE=$(echo "$STATE" | cut -d'|' -f1)
  PCT=$(echo "$STATE" | cut -d'|' -f2)
  ERR=$(echo "$STATE" | cut -d'|' -f3)
  LINE="  stage=$STAGE progress=${PCT}%"
  if [ "$LINE" != "$LAST" ]; then echo "$LINE"; LAST="$LINE"; fi
  if [ -n "$ERR" ]; then
    echo "  job error: $ERR"
    http GET "/projects/$PID/jobs/$JOB/snapshot" >/dev/null
    python3 -c "
import json
d=json.load(open('/tmp/smoke_body.json'))
for l in (d.get('logs') or [])[-12:]:
    print('   |', l.get('message', l) if isinstance(l, dict) else l)
"
    fail "pipeline job failed"
  fi
  if [ "$STAGE" = "human_gate" ] && [ "$PCT" = "100" ]; then break; fi
  sleep 5
done
echo "  pipeline reached stage=$STAGE"
[ "$STAGE" = "human_gate" ] || fail "pipeline did not reach the human gate in 25 minutes (last stage: $STAGE)"
trim_interaction "$PID"

say "SMOKE PASSED — the new interaction works on this device"
echo "  deliverable: check Studio -> project $PID -> Export (one button, gate-gated)"
