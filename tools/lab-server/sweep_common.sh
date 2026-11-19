#!/usr/bin/env bash
# Shared preamble for the perturbation sweeps. Sourced by the per-sweep scripts.
set -uo pipefail
REPO="$HOME/projects/vla-failure-analysis"; cd "$REPO"
if ! curl -s -m 5 http://127.0.0.1:8001/health | grep -q ok; then
  echo "SERVER DOWN on :8001 — start ~/start_server_gpu1.sh first"; exit 2; fi
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vla-libero
export TMPDIR="$HOME/tmp" PYTHONNOUSERSITE=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(pwd)"
# Speed overrides common to every smoke condition (heavy logging off).
FIXED_SPEED="policy_client.include_attentions=false,policy_client.batch_size=1,policy_client.timeout_seconds=600,logging.save_episode_npz=false,logging.save_rollout_video=false"
echo "[sweep] python=$(which python) gpu=$CUDA_VISIBLE_DEVICES cwd=$(pwd) start=$(date)"
