#!/bin/bash
#SBATCH --job-name=terminal-aggregate
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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

RESULTS_CSV="${RESULTS_CSV:-runs/high_level_ppo_eval/aggregate_results.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/high_level_ppo_eval}"

python -m terminal_marl.experiments.aggregate "${RESULTS_CSV}" "${OUTPUT_DIR}"
python - <<'PY'
import os
from pathlib import Path
from terminal_marl.experiments.plot_results import plot_summary
output_dir = Path(os.environ.get("OUTPUT_DIR", "runs/high_level_ppo_eval"))
plot_summary(output_dir / "aggregate_summary.csv", output_dir / "figures")
PY

echo "Job finished"
