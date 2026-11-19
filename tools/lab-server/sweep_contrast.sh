#!/usr/bin/env bash
# Contrast smoke sweep: alpha = 0.2..2.0 (clean=1.0), beta held at 0, 2 trials/task.
source ~/sweep_common.sh
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0.yaml \
  --param observation.alpha \
  --values 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0 \
  --fixed observation.beta=0,$FIXED_SPEED \
  --num-trials 2 --seed 1 \
  --out-root outputs/contrast_sweep_smoke
echo "[sweep] CONTRAST DONE rc=$? $(date)"
