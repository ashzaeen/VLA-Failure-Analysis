#!/usr/bin/env bash
# Block until the Stage 3 smoke sweep finishes (or the tmux session dies), then
# print every per-condition result found. Intended to be run over ssh in the
# background so the completion notification arrives on its own.
for i in $(seq 1 720); do   # 720 * 30 s = 6 h ceiling
  if grep -q 'ALL SMOKE SWEEPS COMPLETE' ~/scene_all_smoke.log 2>/dev/null; then
    echo "=== SMOKE COMPLETE ==="
    break
  fi
  if ! tmux has-session -t scene 2>/dev/null; then
    echo "=== tmux session 'scene' GONE (crash or manual kill) ==="
    break
  fi
  sleep 30
done

cd ~/projects/vla-failure-analysis
for root in outputs/scene_swap_smoke outputs/scene_move_target_smoke outputs/scene_move_distractors_smoke; do
  echo "--- $root"
  [ -f "$root/SWEEP_SUMMARY.csv" ] && cat "$root/SWEEP_SUMMARY.csv"
  for f in "$root"/*/success.json; do
    [ -f "$f" ] && echo "  $f: $(tr -d '\n ' < "$f")"
  done
done
echo "=== tail of log ==="
grep -e 'DONE rc=' -e '^--- ' ~/scene_all_smoke.log | tail -30
