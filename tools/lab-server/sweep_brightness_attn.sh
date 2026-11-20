#!/usr/bin/env bash
# Brightness sweep WITH attention: beta=0..100; 100 eps each; same seed.
# Per beta: eval (attention on) -> attention analysis (metrics + heatmap videos).
set -uo pipefail
REPO="$HOME/projects/vla-failure-analysis"; cd "$REPO"
if ! curl -s -m 3 http://127.0.0.1:8001/health | grep -q ok; then
  echo "SERVER DOWN ??? start ~/start_server.sh first"; exit 2; fi
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vla-libero
export TMPDIR="$HOME/tmp" PYTHONNOUSERSITE=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(pwd)" MPLBACKEND=Agg
mkdir -p ~/sweep_attn_logs
echo "SWEEP_ATTN_START $(date)"
for beta in 0 10 20 30 40 50 60 70 80 90 100; do
  OUT="outputs/beta_sweep_attn/beta_${beta}"
  sed -e 's/num_trials_per_task: 50/num_trials_per_task: 10/' \
      -e 's/timeout_seconds: 300/timeout_seconds: 600/' \
      -e 's/batch_size: 3/batch_size: 1/' \
      -e "s/beta: 30/beta: ${beta}/" \
      -e "s#output_dir: outputs/libero_object_eval_4_embeds#output_dir: ${OUT}#" \
      configs/runner/libero_pi0.yaml > ~/sweep_attn_logs/runner_beta_${beta}.yaml
  echo "########## BETA=${beta} EVAL START $(date) ##########"
  python scripts/run_libero_eval.py --config ~/sweep_attn_logs/runner_beta_${beta}.yaml > ~/sweep_attn_logs/eval_${beta}.log 2>&1
  ERC=$?
  RATE=$(grep -oE 'Overall Success Rate: [0-9]+/[0-9]+ = [0-9.]+' ~/sweep_attn_logs/eval_${beta}.log | tail -1)
  echo "BETA=${beta} EVAL_RATE: ${RATE:-<none>} rc=${ERC}"
  if [ "$ERC" -eq 0 ]; then
    echo "########## BETA=${beta} ANALYZE START $(date) ##########"
    python scripts/analyze_attention.py --npz_dir "${OUT}/episodes_npz" \
       --output_dir "${OUT}/attention_analysis" --head_idx 0 --patch_grid_size 16 --top_k 5 \
       > ~/sweep_attn_logs/analyze_${beta}.log 2>&1
    echo "BETA=${beta} ANALYZE rc=$?"
  fi
  echo "########## BETA=${beta} DONE $(date) ##########"
done
echo "SWEEP_ATTN_ALL_DONE $(date)"