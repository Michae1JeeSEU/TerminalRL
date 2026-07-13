#!/bin/bash
#SBATCH --job-name=terminal-ppo-sweep
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

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

# Each row: seed demand learning_rate training_config
MATRIX=(
  "0 medium 0.00030 terminal_marl/configs/training/ppo_main_medium.yaml"
  "1 medium 0.00030 terminal_marl/configs/training/ppo_main_medium.yaml"
  "2 medium 0.00030 terminal_marl/configs/training/ppo_main_medium.yaml"
  "0 high 0.00025 terminal_marl/configs/training/ppo_main_high.yaml"
  "1 high 0.00025 terminal_marl/configs/training/ppo_main_high.yaml"
  "2 high 0.00025 terminal_marl/configs/training/ppo_main_high.yaml"
  "0 bursty 0.00020 terminal_marl/configs/training/ppo_main_bursty.yaml"
  "1 bursty 0.00020 terminal_marl/configs/training/ppo_main_bursty.yaml"
  "2 bursty 0.00020 terminal_marl/configs/training/ppo_main_bursty.yaml"
)

ROW="${MATRIX[${SLURM_ARRAY_TASK_ID:-0}]}"
read -r SEED DEMAND LR CONFIG_PATH <<< "${ROW}"

python -m terminal_marl.training.train_high_level_ppo \
  --training-config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --demand-level "${DEMAND}" \
  --learning-rate "${LR}" \
  --device cpu \
  --n-envs 16 \
  --vec-env subproc

echo "Job finished"
