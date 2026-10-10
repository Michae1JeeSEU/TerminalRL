"""Prepare the registered gate-release-control phase launcher.

This helper is intentionally inert until called with one explicit phase.
The phase subprocess uses the server project checkout and writes only to the
registered training-output root.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


PROJECT = "/home/user/projects/terminal"
PYTHON = "/home/user/envs/terminal_env/bin/python"
MODULE = "terminal_marl.training.run_p3_gate_release_control_20260930"
OUTPUT = "/home/user/training_outputs/terminal/p3_followup_20260930/gate_release_control"
PHASES = ("preflight", "pilot", "full")


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return env


def _argv(phase: str) -> list[str]:
    if phase not in PHASES:
        raise ValueError(f"unsupported phase: {phase}")
    return [PYTHON, "-m", MODULE, "--phase", phase, "--output-dir", OUTPUT]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    args = parser.parse_args()
    phase = str(args.phase)
    log_path = Path(OUTPUT) / f"{phase}_v1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = _argv(phase)
    if phase == "preflight":
        with log_path.open("ab", buffering=0) as stream:
            result = subprocess.run(
                argv,
                cwd=PROJECT,
                env=_environment(),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                check=False,
            )
        print(json.dumps({"phase": phase, "returncode": result.returncode}, sort_keys=True))
        return int(result.returncode)

    with log_path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            argv,
            cwd=PROJECT,
            env=_environment(),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
    details: dict[str, object] = {
        "phase": phase,
        "pid": int(process.pid),
        "argv": argv,
        "cwd": PROJECT,
        "log": str(log_path),
    }
    try:
        details["proc_cwd"] = os.readlink(f"/proc/{process.pid}/cwd")
        details["proc_argv"] = (
            Path(f"/proc/{process.pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode()
            .strip()
        )
    except OSError as exc:
        details["proc_read_error"] = repr(exc)
    print(json.dumps(details, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
