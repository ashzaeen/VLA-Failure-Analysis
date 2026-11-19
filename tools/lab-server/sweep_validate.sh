#!/usr/bin/env bash
# Fast sanity run: ONE proprio-noise condition, 1 trial/task (10 episodes) to prove
# the new proprio perturbation + client + server round-trip works before the full smoke.
source ~/sweep_common.sh
python scripts/run_sweep.py \
  --base configs/runner/libero_pi0.yaml \
  --param observation.proprio_noise_std \
  --values 2.0 \
  --fixed observation.beta=0,observation.alpha=1.0,$FIXED_SPEED \
  --num-trials 1 --seed 1 \
  --out-root outputs/validate_proprio
echo "[sweep] VALIDATE DONE rc=$? $(date)"
