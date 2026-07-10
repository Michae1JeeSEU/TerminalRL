#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SEED="${SEED:-0}"

python -m terminal_marl.training.train_high_level_ppo \
  --training-config terminal_marl/configs/training/ppo_pilot_medium.yaml \
  --seed "${SEED}" \
  --device cpu \
  --n-envs 8 \
  --vec-env subproc

