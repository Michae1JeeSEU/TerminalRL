"""Frozen traffic/benchmark/dt/scheduler diagnostics with independentprocessjobs."""
import argparse
import json
import os
import subprocess
import sys
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
from ..core.rotating_slots.diagnostics import run_diagnostics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--pilot", choices=("traffic", "benchmark", "sweeps", "all"))
    p.add_argument("--pilot-window", type=float, default=2.)
    p.add_argument("--pilot-drain", type=float, default=0.)
    p.add_argument("--pilot-counts", default="10")
    p.add_argument("--pilot-repeats", type=int, default=1)
    p.add_argument("--pilot-duration", type=float, default=.2)
    p.add_argument("--code-commit")
    a = p.parse_args(argv)
    if a.code_commit: os.environ["TERMINAL_CODE_COMMIT"] = a.code_commit
    command = subprocess.list2cmdline([sys.executable, "-m", "terminal_marl.scripts.run_stage1_diagnostics", *(sys.argv[1:] if argv is None else argv)])
    result = run_diagnostics(a.output, workers=a.workers, pilot=a.pilot,
        pilot_window=a.pilot_window, pilot_drain=a.pilot_drain, pilot_counts=tuple(map(int, a.pilot_counts.split(","))),
        pilot_repeats=a.pilot_repeats, pilot_duration=a.pilot_duration, command=command)
    print(json.dumps({"protocol": result["protocol"], "coverage": result["coverage"], "stage1_status": result["stage1_status"]}, indent=2))
    return 1 if result["coverage"]["failures"] or result["coverage"]["missing"] else 0


if __name__ == "__main__": raise SystemExit(main())
