"""Bounded launcher for the frozen V7 paired evaluator.

The default invocation performs only the evaluator's check-only binding pass.
``--launch`` repeats that pass and then supervises exactly one full evaluator
process.  The smoke path is intentionally not part of this launcher.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path("/home/user/projects/terminal")
PYTHON = Path("/home/user/envs/terminal_env/bin/python")
FORMAL_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926"
)
OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_adaptive_kl_eval_v7_20260926"
)
CONTROL_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + "_control")
MODULE = "terminal_marl.training.evaluate_p3_adaptive_kl_v7"


def _env() -> dict[str, str]:
    value = dict(os.environ)
    value["OMP_NUM_THREADS"] = "1"
    value["MKL_NUM_THREADS"] = "1"
    return value


def _command(*, check_only: bool) -> list[str]:
    command = [
        str(PYTHON), "-m", MODULE,
        "--formal-root", str(FORMAL_ROOT),
        "--output", str(OUTPUT_ROOT),
    ]
    if check_only:
        command.append("--check-only")
    return command


def _check_only() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(check_only=True),
        cwd=str(REPO),
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)

    if OUTPUT_ROOT.exists():
        raise SystemExit(f"refusing existing evaluation output: {OUTPUT_ROOT}")
    if args.launch and CONTROL_ROOT.exists():
        raise SystemExit(f"refusing existing launcher control root: {CONTROL_ROOT}")

    check = _check_only()
    print(check.stdout, end="")
    if check.stderr:
        print(check.stderr, file=sys.stderr, end="")
    if check.returncode != 0 or not args.launch:
        return int(check.returncode)

    CONTROL_ROOT.mkdir(parents=True, exist_ok=False)
    command = _command(check_only=False)
    _write(CONTROL_ROOT / "command.json", {
        "cwd": str(REPO),
        "argv": command,
        "formal_root": str(FORMAL_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "smoke_output": None,
    })
    stdout_path = CONTROL_ROOT / "stdout.log"
    stderr_path = CONTROL_ROOT / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(REPO),
            env=_env(),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    _write(CONTROL_ROOT / "pid.json", {"pid": process.pid, "argv": command})
    _write(CONTROL_ROOT / "status.json", {"status": "RUNNING", "pid": process.pid})
    return_code = process.wait()
    _write(CONTROL_ROOT / "exitcode.json", {"status": "COMPLETED", "exit_code": return_code})
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
