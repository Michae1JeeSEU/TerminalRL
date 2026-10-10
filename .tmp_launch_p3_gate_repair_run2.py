"""Launch the one-time corrected run2 gate-repair label collection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path("/home/user/projects/terminal")
PYTHON = Path("/home/user/envs/terminal_env/bin/python")
COLLECTOR = PROJECT / ".tmp_p3_gate_repair_collect.py"
ROOT = Path("/home/user/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925")
CONTROL = ROOT / "_control"
RUN_INDEX = 2


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _validate_args(args: list[str]) -> None:
    if any("\r" in arg or "\n" in arg for arg in args):
        raise RuntimeError(f"invalid newline in subprocess args: {args!r}")


def _run_arm(arm: str, output: Path) -> int:
    args = [
        str(PYTHON),
        str(COLLECTOR),
        "--run-index",
        str(RUN_INDEX),
        "--arm",
        arm,
        "--output",
        str(output),
    ]
    _validate_args(args)
    _write(CONTROL / f"run2.{arm}.args.json", json.dumps(args) + "\n")
    stdout_path = CONTROL / f"run2.{arm}.stdout"
    stderr_path = CONTROL / f"run2.{arm}.stderr"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            args,
            cwd=str(PROJECT),
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            stdout=stdout,
            stderr=stderr,
        )
        _write(CONTROL / f"run2.{arm}.pid", f"{process.pid}\n")
        return process.wait()


def main() -> int:
    CONTROL.mkdir(parents=True, exist_ok=True)
    outputs = {
        "original": ROOT / "collection" / "original_run2",
        "bounded": ROOT / "collection" / "bounded_run2",
    }
    control_files = [
        CONTROL / "run2.launcher.pid",
        CONTROL / "run2.corrected.status",
        CONTROL / "run2.corrected.exitcode",
        *(CONTROL / f"run2.{arm}.{suffix}" for arm in outputs for suffix in ("pid", "args.json", "stdout", "stderr")),
    ]
    if any(path.exists() for path in outputs.values()):
        raise RuntimeError("refusing existing valid run2 output")
    if any(path.exists() for path in control_files):
        raise RuntimeError("refusing existing corrected run2 control file")

    _write(CONTROL / "run2.launcher.pid", f"{os.getpid()}\n")
    _write(CONTROL / "run2.corrected.status", "RUNNING\n")
    try:
        original_rc = _run_arm("original", outputs["original"])
        _write(CONTROL / "run2.corrected.status", f"original_exit={original_rc}\n")
        if original_rc != 0:
            _write(CONTROL / "run2.corrected.exitcode", f"{original_rc}\n")
            _write(CONTROL / "run2.corrected.status", f"original_exit={original_rc}\nFAILED\n")
            return original_rc

        bounded_rc = _run_arm("bounded", outputs["bounded"])
        _write(
            CONTROL / "run2.corrected.status",
            f"original_exit=0\nbounded_exit={bounded_rc}\n{'COMPLETED' if bounded_rc == 0 else 'FAILED'}\n",
        )
        _write(CONTROL / "run2.corrected.exitcode", f"{bounded_rc}\n")
        return bounded_rc
    except BaseException as exc:
        _write(CONTROL / "run2.corrected.status", f"EXCEPTION={type(exc).__name__}: {exc}\nFAILED\n")
        _write(CONTROL / "run2.corrected.exitcode", "125\n")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run2 launcher failed: {exc}", file=sys.stderr)
        raise
