#!/usr/bin/env bash
# Stage 3 SMOKE: object-position perturbations, 2 trials/task = 20 eps/condition.
#
# Three experiments, run in ascending order of expected cost:
#   1. swap             - target <-> distractor slot exchange. No new positions are
#                         introduced (every object still sits on a slot the model has
#                         seen occupied), so this is the confound-free position test.
#   2. move_target      - dose-response: displace only the target by d metres.
#   3. move_distractors - control: displace the distractors by d, target stays home.
#
# d=0 is included in the move_* sweeps as an in-run baseline anchor: at distance 0 the
# perturber is an explicit no-op, so that condition must reproduce the clean baseline.
exec > >(tee -a "$HOME/scene_all_smoke.log") 2>&1
source ~/scene_common.sh

echo "=== [1/3] SWAP ==="
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0_scene.yaml \
  --param scene.mode --values swap \
  --fixed scene.seed=1,$FIXED_SMOKE \
  --num-trials 2 --seed 1 \
  --out-root outputs/scene_swap_smoke
echo "[scene] SWAP DONE rc=$? $(date)"

echo "=== [2/3] MOVE_TARGET ==="
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0_scene.yaml \
  --param scene.distance --values 0,0.05,0.10,0.15 \
  --fixed scene.mode=move_target,scene.seed=1,$FIXED_SMOKE \
  --num-trials 2 --seed 1 \
  --out-root outputs/scene_move_target_smoke
echo "[scene] MOVE_TARGET DONE rc=$? $(date)"

echo "=== [3/3] MOVE_DISTRACTORS ==="
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0_scene.yaml \
  --param scene.distance --values 0.05,0.10,0.15 \
  --fixed scene.mode=move_distractors,scene.seed=1,$FIXED_SMOKE \
  --num-trials 2 --seed 1 \
  --out-root outputs/scene_move_distractors_smoke
echo "[scene] MOVE_DISTRACTORS DONE rc=$? $(date)"
echo "[scene] ALL SMOKE SWEEPS COMPLETE $(date)"
