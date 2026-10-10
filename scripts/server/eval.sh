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

module load anaconda3-2024.10-1

source activate sumo-1.25

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m terminal_marl.experiments.reward_diagnostics \
  --config dynamic_hrl \
  --demand-level medium \
  --episodes 3 \
  --output-dir runs/reward_diagnostics_medium

echo "reward diagnostics finished"
