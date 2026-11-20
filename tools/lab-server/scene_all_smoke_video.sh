#!/usr/bin/env bash
# Stage 3 SMOKE, second pass: identical conditions and seeds to the first pass,
# but with rollout video ON and the object-disturbance instrumentation in place.
#
# NPZ stays OFF: the frame-collection fix means video no longer requires it, and
# the NPZs run ~100 MB/episode. Attention also stays off - it is not needed to
# establish the behavioural signature and roughly triples episode time.
#
# The first pass survives as outputs/scene_*_smoke_novid, so re-running the same
# seeds doubles as a reproducibility check (swap must land 0/20, d=0 must land
# 17/20 again).
exec > >(tee -a "$HOME/scene_all_smoke_video.log") 2>&1
source ~/scene_common.sh

# Override the smoke defaults: video on, NPZ off.
FIXED_VID="policy_client.include_attentions=false,policy_client.batch_size=1,policy_client.timeout_seconds=1200,logging.save_episode_npz=false,logging.save_rollout_video=true"

echo "=== [1/3] SWAP (video) ==="
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0_scene.yaml \
  --param scene.mode --values swap \
  --fixed scene.seed=1,$FIXED_VID \
  --num-trials 2 --seed 1 \
  --out-root outputs/scene_swap_smoke
echo "[scene] SWAP DONE rc=$? $(date)"

echo "=== [2/3] MOVE_TARGET (video) ==="
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0_scene.yaml \
  --param scene.distance --values 0,0.05,0.10,0.15 \
  --fixed scene.mode=move_target,scene.seed=1,$FIXED_VID \
  --num-trials 2 --seed 1 \
  --out-root outputs/scene_move_target_smoke
echo "[scene] MOVE_TARGET DONE rc=$? $(date)"

echo "=== [3/3] MOVE_DISTRACTORS (video) ==="
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0_scene.yaml \
  --param scene.distance --values 0.05,0.10,0.15 \
  --fixed scene.mode=move_distractors,scene.seed=1,$FIXED_VID \
  --num-trials 2 --seed 1 \
  --out-root outputs/scene_move_distractors_smoke
echo "[scene] MOVE_DISTRACTORS DONE rc=$? $(date)"
echo "[scene] ALL SMOKE SWEEPS COMPLETE $(date)"
