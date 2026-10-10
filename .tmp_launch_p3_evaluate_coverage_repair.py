"""Launch one bounded E04 coverage-repair evaluation arm with argv-list args."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path("/home/user/projects/terminal")
PYTHON = Path("/home/user/envs/terminal_env/bin/python")
EVALUATOR = PROJECT / ".tmp_p3_evaluate_coverage_repair.py"
CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_gate_coverage_repair_v2_20260925/fit853401/p3_gate_coverage_repair_v2.pt"
)
ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_gate_coverage_repair_v2_20260925/evaluation"
)
CONTROL = ROOT / "_control"


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _validate_args(args: list[str]) -> None:
    if any("\r" in arg or "\n" in arg for arg in args):
        raise RuntimeError(f"invalid newline in subprocess args: {args!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--run-index", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    if args.mode == "smoke" and args.run_index != 0:
        parser.error("smoke is fixed to run-index 0")
    return args


def main() -> int:
    args = _parse_args()
    label = f"smoke_run{args.run_index}" if args.mode == "smoke" else f"run{args.run_index}"
    output = ROOT / label
    max_steps = 2 if args.mode == "smoke" else 1800
    CONTROL.mkdir(parents=True, exist_ok=True)
    control_prefix = CONTROL / f"eval_{label}"
    control_files = [
        control_prefix.with_suffix(suffix)
        for suffix in (
            ".pid",
            ".launcher_pid",
            ".args.json",
            ".status",
            ".exitcode",
            ".stdout",
            ".stderr",
        )
    ]
    if output.exists():
        raise RuntimeError(f"refusing existing evaluation output: {output}")
    if any(path.exists() for path in control_files):
        raise RuntimeError("refusing existing evaluation control file")
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"checkpoint not found: {CHECKPOINT}")

    command = [
        str(PYTHON),
        str(EVALUATOR),
        "--checkpoint",
        str(CHECKPOINT),
        "--run-index",
        str(args.run_index),
        "--output",
        str(output),
        "--max-steps",
        str(max_steps),
    ]
    _validate_args(command)
    _write(control_prefix.with_suffix(".args.json"), json.dumps(command) + "\n")
    _write(control_prefix.with_suffix(".status"), "RUNNING\n")
    _write(control_prefix.with_suffix(".launcher_pid"), f"{os.getpid()}\n")
    with control_prefix.with_suffix(".stdout").open("w", encoding="utf-8") as stdout, control_prefix.with_suffix(
        ".stderr"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT),
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            stdout=stdout,
            stderr=stderr,
        )
        _write(control_prefix.with_suffix(".pid"), f"{process.pid}\n")
        rc = process.wait()
    _write(control_prefix.with_suffix(".exitcode"), f"{rc}\n")
    _write(control_prefix.with_suffix(".status"), f"exitcode={rc}\n{'COMPLETED' if rc == 0 else 'FAILED'}\n")
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"evaluation launcher failed: {exc}", file=sys.stderr)
        raise
