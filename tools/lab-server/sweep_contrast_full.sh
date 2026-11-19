#!/usr/bin/env bash
# FULL contrast sweep: alpha = 0.2..2.0 (clean=1.0), beta held at 0.
# 10 trials/task => 100 episodes/condition. Attention + video + NPZ enabled.
source ~/sweep_common_full.sh
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0.yaml \
  --param observation.alpha \
  --values 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0 \
  --fixed observation.beta=0,$FIXED_FULL \
  --num-trials 10 --seed 1 \
  --out-root outputs/contrast_sweep_full
echo "[sweep-full] CONTRAST DONE rc=$? $(date)"
