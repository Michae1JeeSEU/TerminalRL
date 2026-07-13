#!/bin/bash
#SBATCH --job-name=terminal-ppo-eval
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module load anaconda3-2024.10-1 || true

if command -v conda >/dev/null 2>&1; then
  source activate "${CONDA_ENV:-terminal-marl}" || true
fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

MODEL_PATH="${MODEL_PATH:-runs/high_level_ppo/ppo_main_medium/model.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/high_level_ppo_eval}"

python -m terminal_marl.training.evaluate_high_level_policy \
  --model-path "${MODEL_PATH}" \
  --configs dynamic_hrl full_hrl_no_safety \
  --seeds 0 1 2 \
  --demand-levels medium high bursty \
  --episodes 3 \
  --output-dir "${OUTPUT_DIR}"

echo "Job finished"
