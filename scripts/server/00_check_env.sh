#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Optional on clusters with environment modules:
# module load anaconda3-2024.10-1

python -V
python - <<'PY'
import importlib.util
for name in ["numpy", "gymnasium", "stable_baselines3", "matplotlib", "pytest"]:
    print(f"{name}: {'ok' if importlib.util.find_spec(name) else 'missing'}")
PY
python -m pytest tests/test_training_config_loader.py tests/test_train_high_level_ppo_smoke.py -v

