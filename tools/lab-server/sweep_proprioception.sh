#!/usr/bin/env bash
# Proprioception smoke sweep: additive Gaussian noise std = 0,0.5,1,2,4,8
# (0 = clean), alpha=1.0 beta=0 held fixed, 2 trials/task.
source ~/sweep_common.sh
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0.yaml \
  --param observation.proprio_noise_std \
  --values 0,0.5,1,2,4,8 \
  --fixed observation.beta=0,observation.alpha=1.0,$FIXED_SPEED \
  --num-trials 2 --seed 1 \
  --out-root outputs/proprio_sweep_smoke
echo "[sweep] PROPRIO DONE rc=$? $(date)"
