#!/usr/bin/env bash
#SBATCH --job-name=terminal-ppo-eval
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err

set -euo pipefail

PARTITION="${PARTITION:-6240-36C-192G}"
echo "Partition variable can be changed before sbatch: ${PARTITION}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

MODEL_PATH="${MODEL_PATH:-runs/high_level_ppo/ppo_main_medium/model.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/high_level_ppo_eval}"

python -m terminal_marl.training.evaluate_high_level_policy \
  --model-path "${MODEL_PATH}" \
  --configs dynamic_hrl full_hrl_no_safety \
  --seeds 0 1 2 \
  --demand-levels medium high bursty \
  --episodes 3 \
  --output-dir "${OUTPUT_DIR}"

