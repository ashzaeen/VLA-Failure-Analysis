#!/usr/bin/env bash
# FULL proprioception sweep: additive Gaussian noise std = 0,0.5,1,2,4,8 (0 = clean),
# alpha=1.0 beta=0 held fixed. 10 trials/task => 100 episodes/condition.
# Attention + video + NPZ enabled.
source ~/sweep_common_full.sh
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0.yaml \
  --param observation.proprio_noise_std \
  --values 0,0.5,1,2,4,8 \
  --fixed observation.beta=0,observation.alpha=1.0,$FIXED_FULL \
  --num-trials 10 --seed 1 \
  --out-root outputs/proprio_sweep_full
echo "[sweep-full] PROPRIO DONE rc=$? $(date)"
