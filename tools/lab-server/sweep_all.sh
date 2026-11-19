#!/usr/bin/env bash
# Runs both smoke sweeps back-to-back. `exec >` sends all output to a log file so
# progress survives disconnection and is readable without tmux capture-pane.
exec > ~/sweep_all.log 2>&1
echo "===== SWEEP_ALL START $(date) ====="
bash ~/sweep_contrast.sh
echo "----- contrast finished, starting proprio $(date) -----"
bash ~/sweep_proprio.sh
echo "===== SWEEP_ALL DONE $(date) ====="
