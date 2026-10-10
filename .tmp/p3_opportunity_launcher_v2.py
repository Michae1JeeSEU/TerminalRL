"""Small remote phase launcher used only for the registered opportunity run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = "/home/user/projects/terminal"
PYTHON = "/home/user/envs/terminal_env/bin/python"
OUTPUT = "/home/user/training_outputs/terminal/p3_followup_20260930/opportunity_coverage"
MODULE = "terminal_marl.training.run_p3_opportunity_coverage_20260930"


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
    return [PYTHON, "-m", MODULE, "--phase", phase, "--output-dir", OUTPUT]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"preflight", "pilot"}:
        raise SystemExit("usage: launcher.py {preflight|pilot}")
    phase = sys.argv[1]
    log_name = f"{phase}_v1.log"
    log_path = Path(OUTPUT) / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if phase == "preflight":
        with log_path.open("ab", buffering=0) as stream:
            result = subprocess.run(
                _argv(phase),
                cwd=PROJECT,
                env=_environment(),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                check=False,
            )
        print(json.dumps({"phase": phase, "returncode": result.returncode}))
        return result.returncode

    with log_path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            _argv(phase),
            cwd=PROJECT,
            env=_environment(),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
    pid = process.pid
    details: dict[str, object] = {"phase": phase, "pid": pid}
    try:
        details["cwd"] = os.readlink(f"/proc/{pid}/cwd")
        details["argv"] = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    except OSError as exc:
        details["proc_read_error"] = repr(exc)
    print(json.dumps(details, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
