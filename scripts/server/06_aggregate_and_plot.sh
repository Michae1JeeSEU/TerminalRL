#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

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
