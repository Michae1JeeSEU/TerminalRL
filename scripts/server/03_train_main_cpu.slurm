#!/bin/bash
#SBATCH --job-name=terminal-ppo-main
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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

CONFIG_PATH="${CONFIG_PATH:-terminal_marl/configs/training/ppo_main_medium.yaml}"
SEED="${SEED:-0}"

python -m terminal_marl.training.train_high_level_ppo \
  --training-config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --device cpu \
  --n-envs 16 \
  --vec-env subproc

echo "Job finished"
