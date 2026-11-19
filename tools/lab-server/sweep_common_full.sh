#!/usr/bin/env bash
# Shared preamble for the FULL perturbation sweeps (100 eps/condition, attention +
# video + NPZ all ENABLED). Sourced by sweep_contrast_full.sh / sweep_proprio_full.sh.
set -uo pipefail
REPO="$HOME/projects/vla-failure-analysis"; cd "$REPO"
if ! curl -s -m 5 http://127.0.0.1:8001/health | grep -q ok; then
  echo "SERVER DOWN on :8001 — start ~/start_server_gpu1.sh first"; exit 2; fi
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vla-libero
export TMPDIR="$HOME/tmp" PYTHONNOUSERSITE=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(pwd)"
# FULL-run overrides: attention returned by the server, rollout video saved, and the
# per-episode NPZ saved (the NPZ is where attention_maps are persisted for analysis).
# batch_size kept at 1 for memory safety on the shared GPU; generous request timeout.
FIXED_FULL="policy_client.include_attentions=true,policy_client.batch_size=1,policy_client.timeout_seconds=1200,logging.save_episode_npz=true,logging.save_rollout_video=true,logging.save_attention=true"
echo "[sweep-full] python=$(which python) gpu=$CUDA_VISIBLE_DEVICES cwd=$(pwd) start=$(date)"
