#!/usr/bin/env bash
# Robust downloader for the real bodycam source (Wikimedia throttles: 429 + retry-after).
# Retries with backoff until the file lands and validates (size > 100 MB + webm magic).
set -u
URL="https://upload.wikimedia.org/wikipedia/commons/6/67/Bodycam-_Drunk_Cop_Arrested_While_On_Duty._Eustis%2C_FL._July_04%2C_2022..webm"
DEST="/home/z/my-project/director-os/data/media/bodycam_eustis.webm"
UA="DirectorOS-Pipeline/1.0 (E2E pipeline test; contact dev@director-os.local)"
LOG=/tmp/dl_bodycam.log
mkdir -p "$(dirname "$DEST")"
: > "$LOG"

for attempt in $(seq 1 40); do
  # resume partial download with -C -; Wikimedia supports range requests
  curl -s -A "$UA" -C - --max-time 1800 -o "$DEST" "$URL" >> "$LOG" 2>&1
  rc=$?
  size=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
  magic=$(head -c 4 "$DEST" 2>/dev/null | xxd -p 2>/dev/null || echo none)
  echo "[attempt $attempt] rc=$rc size=$size magic=$magic $(date -u +%H:%M:%S)" >> "$LOG"
  if [ "$size" -gt 100000000 ] && [ "$magic" = "1a45dfa3" ]; then
    echo "COMPLETE size=$size" >> "$LOG"
    exit 0
  fi
  # honor retry-after on fresh 429s; curl rc 0 with html body also counts as throttle
  if [ "$rc" -ne 0 ] || ! grep -q "1a45dfa3" <(head -c 4 "$DEST" | xxd -p 2>/dev/null); then
    head -c 200 "$DEST" 2>/dev/null | grep -qi "429 Too Many" && sleep 605 && rm -f "$DEST" && continue
    head -c 200 "$DEST" 2>/dev/null | grep -qi "<html" && rm -f "$DEST"
  fi
  sleep 30
done
echo "GAVE_UP" >> "$LOG"
exit 1
