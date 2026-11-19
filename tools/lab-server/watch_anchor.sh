#!/usr/bin/env bash
# Block until the move_target d=0 anchor condition reports. That condition is an
# explicit no-op, so it must reproduce the clean baseline (~85-95%). If it does
# not, the Stage 3 pipeline is broken and the run should be killed.
F=~/projects/vla-failure-analysis/outputs/scene_move_target_smoke/distance_0/success.json
for i in $(seq 1 120); do
  [ -f "$F" ] && break
  tmux has-session -t scene 2>/dev/null || { echo "session gone"; break; }
  sleep 30
done
echo "=== SWAP result ==="
cat ~/projects/vla-failure-analysis/outputs/scene_swap_smoke/mode_swap/success.json 2>/dev/null
echo
echo "=== ANCHOR (move_target d=0) ==="
cat "$F" 2>/dev/null || echo "anchor not written yet"
