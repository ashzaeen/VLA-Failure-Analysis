#!/usr/bin/env bash
# Brightness sweep: beta = 0,10,...,100; 100 eps each; same seed; no attention.
set -uo pipefail
REPO="$HOME/projects/vla-failure-analysis"; cd "$REPO"
if ! curl -s -m 3 http://127.0.0.1:8001/health | grep -q ok; then
  echo "SERVER DOWN ??? start ~/start_server.sh first"; exit 2; fi
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vla-libero
export TMPDIR="$HOME/tmp" PYTHONNOUSERSITE=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(pwd)"
mkdir -p ~/sweep_logs outputs/beta_sweep
echo "SWEEP_START $(date)"
for beta in 0 10 20 30 40 50 60 70 80 90 100; do
  OUT="outputs/beta_sweep/beta_${beta}"
  sed -e 's/num_trials_per_task: 50/num_trials_per_task: 10/' \
      -e 's/timeout_seconds: 300/timeout_seconds: 600/' \
      -e 's/include_attentions: true/include_attentions: false/' \
      -e 's/batch_size: 3/batch_size: 1/' \
      -e "s/beta: 30/beta: ${beta}/" \
      -e "s#output_dir: outputs/libero_object_eval_4_embeds#output_dir: ${OUT}#" \
      configs/runner/libero_pi0.yaml > ~/sweep_logs/runner_beta_${beta}.yaml
  echo "########## BETA=${beta} START $(date) ##########"
  python scripts/run_libero_eval.py --config ~/sweep_logs/runner_beta_${beta}.yaml > ~/sweep_logs/beta_${beta}.log 2>&1
  RC=$?
  RATE=$(grep -oE 'Overall Success Rate: [0-9]+/[0-9]+ = [0-9.]+' ~/sweep_logs/beta_${beta}.log | tail -1)
  echo "BETA=${beta} RESULT: ${RATE:-<none>} rc=${RC} $(date)"
done
echo "SWEEP_ALL_DONE $(date)"