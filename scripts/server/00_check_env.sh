#!/bin/bash
#SBATCH --job-name=terminal-check-env
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Optional on clusters with environment modules; mirrors the known server setup.
module load anaconda3-2024.10-1 || true

if command -v conda >/dev/null 2>&1; then
  source activate "${CONDA_ENV:-terminal-marl}" || true
fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -V
python - <<'PY'
import importlib.util
for name in ["numpy", "gymnasium", "stable_baselines3", "matplotlib", "pytest"]:
    print(f"{name}: {'ok' if importlib.util.find_spec(name) else 'missing'}")
PY
python -m pytest tests/test_training_config_loader.py tests/test_train_high_level_ppo_smoke.py -v

echo "Job finished"
