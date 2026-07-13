#!/bin/bash
#SBATCH --job-name=terminal-ppo-pilot
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

SEED="${SEED:-0}"

python -m terminal_marl.training.train_high_level_ppo \
  --training-config terminal_marl/configs/training/ppo_pilot_medium.yaml \
  --seed "${SEED}" \
  --device cpu \
  --n-envs 8 \
  --vec-env subproc

echo "Job finished"
