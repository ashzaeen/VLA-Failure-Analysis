#!/usr/bin/env bash
# Wait for the video smoke sweep, then: report every condition's success rate,
# classify what the robot actually disturbed, and package the rollout videos
# into one archive for transfer.
for i in $(seq 1 720); do
  grep -q 'ALL SMOKE SWEEPS COMPLETE' ~/scene_all_smoke_video.log 2>/dev/null && break
  tmux has-session -t scene 2>/dev/null || { echo "=== session gone ==="; break; }
  sleep 30
done

cd ~/projects/vla-failure-analysis
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vla-libero
export PYTHONNOUSERSITE=1 PYTHONPATH="$(pwd)"

echo "=========== SUCCESS RATES ==========="
for root in outputs/scene_swap_smoke outputs/scene_move_target_smoke outputs/scene_move_distractors_smoke; do
  echo "--- $root"
  [ -f "$root/SWEEP_SUMMARY.csv" ] && cat "$root/SWEEP_SUMMARY.csv"
done

echo
echo "=========== REPRODUCIBILITY vs FIRST PASS (no-video) ==========="
for root in outputs/scene_swap_smoke_novid outputs/scene_move_target_smoke_novid outputs/scene_move_distractors_smoke_novid; do
  [ -d "$root" ] || continue
  echo "--- $root"
  grep -h -e value -e succ "$root"/*/success.json 2>/dev/null | tr -d ' "' | paste -sd' '
done

echo
echo "=========== FAILURE MECHANISM ==========="
python scripts/analyze_scene_failures.py \
  outputs/scene_swap_smoke/*/results.json \
  outputs/scene_move_target_smoke/*/results.json \
  outputs/scene_move_distractors_smoke/*/results.json 2>&1

echo
echo "=========== VIDEO ARCHIVE ==========="
cd ~/projects/vla-failure-analysis/outputs
rm -f ~/stage3_videos.tar
tar cf ~/stage3_videos.tar \
  scene_swap_smoke/*/rollouts \
  scene_move_target_smoke/*/rollouts \
  scene_move_distractors_smoke/*/rollouts 2>/dev/null
ls -lh ~/stage3_videos.tar
find scene_swap_smoke scene_move_target_smoke scene_move_distractors_smoke \
  -name '*.mp4' 2>/dev/null | wc -l
