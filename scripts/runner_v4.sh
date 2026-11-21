#!/bin/bash
# Detached v4 caption re-burn runner (survives tool-call boundaries).
cd /home/z/my-project/scripts
LOG=/home/z/my-project/pipeline_run/v3/runner_v4.log
echo "=== v4 runner start $(date) ===" >> "$LOG"
for ci in 0 1 2 3 4 5 6 7 8; do
  echo "--- burn $ci $(date)" >> "$LOG"
  timeout 3000 python3 caption_v4.py burn $ci >> "$LOG" 2>&1 || { echo "BURN $ci FAILED" >> "$LOG"; exit 1; }
done
echo "--- burnconcat $(date)" >> "$LOG"
timeout 600 python3 caption_v4.py burnconcat >> "$LOG" 2>&1 || exit 1
echo "--- mux $(date)" >> "$LOG"
timeout 900 python3 caption_v4.py mux >> "$LOG" 2>&1 || exit 1
echo "=== v4 runner done $(date) ===" >> "$LOG"
