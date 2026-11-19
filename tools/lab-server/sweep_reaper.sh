#!/usr/bin/env bash
# Watches the running FULL sweep and frees GPU + RAM once it finishes.
# Teardown happens ONLY on clean completion (the DONE marker is in the log). If the
# sweep crashed/was killed, the server is LEFT UP so the run can be inspected/resumed.
# Run detached (tmux 'reaper') so it survives SSH disconnects.
set -uo pipefail
LOG="$HOME/sweep_all_full.log"
RLOG="$HOME/sweep_reaper.log"
exec > "$RLOG" 2>&1

echo "[reaper] start $(date)"
SWEEP_PID="$(pgrep -u "$USER" -f sweep_all_full.sh | head -1 || true)"
if [ -z "$SWEEP_PID" ]; then
  echo "[reaper] sweep_all_full.sh not running at start — checking log for completion"
else
  echo "[reaper] waiting on sweep wrapper pid $SWEEP_PID"
  while kill -0 "$SWEEP_PID" 2>/dev/null; do sleep 120; done
  echo "[reaper] sweep wrapper exited $(date)"
fi
sleep 5

if grep -q "SWEEP_ALL_FULL DONE" "$LOG" 2>/dev/null; then
  echo "[reaper] CLEAN completion detected — tearing down policy server"
  tmux kill-session -t srv 2>/dev/null && echo "[reaper] killed tmux srv"
  sleep 3
  pkill -u "$USER" -f run_policy_server 2>/dev/null && echo "[reaper] reaped stray server proc"
  sleep 2
  pkill -u "$USER" -f _inductor/compile_worker 2>/dev/null && echo "[reaper] reaped stray compile workers"
  tmux kill-session -t sweepfull 2>/dev/null && echo "[reaper] closed tmux sweepfull"
  sleep 3
  echo "[reaper] teardown complete $(date). Remaining GPU procs:"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
  echo "[reaper] $USER python procs still alive:"
  pgrep -u "$USER" -a python || echo "  none"
else
  echo "[reaper] NO completion marker — sweep likely crashed/killed."
  echo "[reaper] LEAVING server UP for inspection/resume. No teardown performed."
fi
echo "[reaper] done $(date)"
