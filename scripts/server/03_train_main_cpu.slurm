#!/usr/bin/env bash
#SBATCH --job-name=terminal-ppo-main
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err

set -euo pipefail

PARTITION="${PARTITION:-6240-36C-192G}"
if [[ -n "${SLURM_JOB_ID:-}" && -n "${PARTITION}" ]]; then
  echo "Requested partition variable: ${PARTITION}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Optional on clusters with environment modules:
# module load anaconda3-2024.10-1

CONFIG_PATH="${CONFIG_PATH:-terminal_marl/configs/training/ppo_main_medium.yaml}"
SEED="${SEED:-0}"

python -m terminal_marl.training.train_high_level_ppo \
  --training-config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --device cpu \
  --n-envs 16 \
  --vec-env subproc

