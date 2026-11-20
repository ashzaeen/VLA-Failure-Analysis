#!/usr/bin/env bash
# Runs both FULL sweeps (100 eps/condition, attention+video+NPZ) back-to-back.
# Proprio first (the scientifically interesting collapsing curve), then contrast.
# `exec >` sends all output to a log so progress survives disconnection.
exec > ~/sweep_all_full.log 2>&1
echo "===== SWEEP_ALL_FULL START $(date) ====="
bash ~/sweep_proprioception_full.sh
echo "----- proprio finished, starting contrast $(date) -----"
bash ~/sweep_contrast_full.sh
echo "===== SWEEP_ALL_FULL DONE $(date) ====="
