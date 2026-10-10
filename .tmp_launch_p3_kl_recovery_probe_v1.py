"""Bounded launcher for the two-run KL recovery probe.

The default invocation performs only the runner's check-only preflight for
run0 and run2.  ``--launch`` repeats that preflight and starts one detached
supervisor per run only when both checks pass.  The supervisor is the only
process that writes the child exit-code file.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path("/home/user/projects/terminal")
PYTHON = Path("/home/user/envs/terminal_env/bin/python")
RUNNER_MODULE = "terminal_marl.training.run_p3_kl_recovery_probe_v1"
OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_kl_recovery_probe_v1_20260926"
)
CONTROL_ROOT = OUTPUT_ROOT / "_control"
RUN_INDICES = (0, 2)
SUPERVISE = "--_supervise"


def _env() -> dict[str, str]:
    result = os.environ.copy()
    result["OMP_NUM_THREADS"] = "1"
    result["MKL_NUM_THREADS"] = "1"
    return result


def _runner_argv(run_index: int, output: Path, *, check_only: bool) -> list[str]:
    argv = [
        str(PYTHON),
        "-m",
        RUNNER_MODULE,
        "--run-index",
        str(run_index),
        "--output",
        str(output),
    ]
    if check_only:
        argv.append("--check-only")
    return argv


def _validate_argv(argv: Sequence[str]) -> None:
    if any("\r" in item or "\n" in item for item in argv):
        raise RuntimeError(f"newline in subprocess argument: {argv!r}")


def _control_paths(run_index: int) -> dict[str, Path]:
    prefix = CONTROL_ROOT / f"run{run_index}"
    return {
        "command": prefix.with_suffix(".command.json"),
        "supervisor_pid": prefix.with_suffix(".supervisor_pid"),
        "pid": prefix.with_suffix(".pid"),
        "status": prefix.with_suffix(".status"),
        "exitcode": prefix.with_suffix(".exitcode"),
        "stdout": prefix.with_suffix(".stdout.log"),
        "stderr": prefix.with_suffix(".stderr.log"),
        "supervisor_stdout": prefix.with_suffix(".supervisor.stdout.log"),
        "supervisor_stderr": prefix.with_suffix(".supervisor.stderr.log"),
        "supervisor_error": prefix.with_suffix(".supervisor_error"),
    }


def _output_path(run_index: int) -> Path:
    return OUTPUT_ROOT / f"run{run_index}"


def _exclusive_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_status(path: Path, value: str) -> None:
    path.write_text(value + "\n", encoding="utf-8", newline="\n")


def _assert_clean() -> None:
    for run_index in RUN_INDICES:
        output = _output_path(run_index)
        if output.exists():
            raise FileExistsError(f"refusing existing output: {output}")
        for path in _control_paths(run_index).values():
            if path.exists():
                raise FileExistsError(f"refusing existing control path: {path}")


def _parse_runner_json(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for offset, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, end = decoder.raw_decode(text[offset:])
            except json.JSONDecodeError:
                continue
            if not text[offset + end :].strip() and isinstance(value, dict):
                return value
        return {}
    return value if isinstance(value, dict) else {}


def _check_one(run_index: int) -> dict[str, Any]:
    output = _output_path(run_index)
    argv = _runner_argv(run_index, output, check_only=True)
    _validate_argv(argv)
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        env=_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    parsed = _parse_runner_json(completed.stdout)
    result: dict[str, Any] = {
        "run_index": run_index,
        "returncode": completed.returncode,
        "status": parsed.get("status"),
        "check_only": parsed.get("check_only"),
        "output_created": parsed.get("output_created"),
        "output_existed": parsed.get("output_existed"),
        "output": str(output),
    }
    if "error" in parsed:
        result["error"] = parsed["error"]
    if completed.stderr.strip():
        result["stderr_nonempty"] = True
    result["passed"] = bool(
        completed.returncode == 0
        and parsed.get("status") == "CHECK_ONLY_OK"
        and parsed.get("check_only") is True
        and parsed.get("output_created") is False
        and parsed.get("output_existed") is False
    )
    return result


def _check_all() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for run_index in RUN_INDICES:
        try:
            result = _check_one(run_index)
        except Exception as exc:
            result = {
                "run_index": run_index,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        checks.append(result)
        if not result.get("passed"):
            failures.append(f"run{run_index}")
    if failures:
        raise RuntimeError("check-only failed for " + ", ".join(failures))
    return checks


def _supervise(run_index: int) -> int:
    paths = _control_paths(run_index)
    output = _output_path(run_index)
    argv = _runner_argv(run_index, output, check_only=False)
    _validate_argv(argv)
    child: subprocess.Popen[str] | None = None
    returncode = 1
    try:
        _write_status(paths["status"], "STARTING")
        with paths["stdout"].open("x", encoding="utf-8", newline="\n") as stdout, \
                paths["stderr"].open("x", encoding="utf-8", newline="\n") as stderr:
            child = subprocess.Popen(
                argv,
                cwd=PROJECT_ROOT,
                env=_env(),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                text=True,
            )
            _exclusive_text(paths["pid"], f"{child.pid}\n")
            _write_status(paths["status"], "RUNNING")
            returncode = int(child.wait())
        _exclusive_text(paths["exitcode"], f"{returncode}\n")
        _write_status(paths["status"], "COMPLETED" if returncode == 0 else "FAILED")
        return returncode
    except BaseException as exc:
        if child is not None:
            try:
                returncode = int(child.wait())
            except BaseException:
                returncode = 1
        try:
            _exclusive_text(
                paths["supervisor_error"],
                f"{type(exc).__name__}: {exc}\n",
            )
        except FileExistsError:
            pass
        try:
            if not paths["exitcode"].exists():
                _exclusive_text(paths["exitcode"], f"{returncode}\n")
            _write_status(paths["status"], "SUPERVISOR_FAILED")
        except OSError:
            pass
        return returncode if returncode != 0 else 1


def _launch_one(run_index: int) -> dict[str, Any]:
    paths = _control_paths(run_index)
    output = _output_path(run_index)
    supervisor_argv = [
        str(PYTHON),
        str(Path(__file__).resolve()),
        SUPERVISE,
        "--run-index",
        str(run_index),
    ]
    _validate_argv(supervisor_argv)
    _exclusive_text(paths["command"], json.dumps(_runner_argv(run_index, output, check_only=False)) + "\n")
    with paths["supervisor_stdout"].open("x", encoding="utf-8", newline="\n") as stdout, \
            paths["supervisor_stderr"].open("x", encoding="utf-8", newline="\n") as stderr:
        supervisor = subprocess.Popen(
            supervisor_argv,
            cwd=PROJECT_ROOT,
            env=_env(),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            text=True,
        )
    _exclusive_text(paths["supervisor_pid"], f"{supervisor.pid}\n")
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if paths["pid"].exists():
            child_pid = int(paths["pid"].read_text(encoding="utf-8").strip())
            return {
                "run_index": run_index,
                "supervisor_pid": supervisor.pid,
                "child_pid": child_pid,
                "output": str(output),
                "control": str(CONTROL_ROOT),
            }
        if paths["supervisor_error"].exists():
            error = paths["supervisor_error"].read_text(encoding="utf-8").strip()
            raise RuntimeError(f"run{run_index} supervisor failed: {error}")
        time.sleep(0.05)
    raise TimeoutError(f"run{run_index} supervisor did not publish child PID")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="run both check-only preflights, then launch both supervisors",
    )
    parser.add_argument(SUPERVISE, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-index", type=int, choices=RUN_INDICES, help=argparse.SUPPRESS)
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(values)
    if args._supervise:
        if args.run_index not in RUN_INDICES:
            raise SystemExit("--_supervise requires --run-index 0 or 2")
        return _supervise(int(args.run_index))
    if args.run_index is not None:
        raise SystemExit("--run-index is only valid with internal supervision")
    _assert_clean()
    checks = _check_all()
    print(json.dumps({"event": "check_only", "checks": checks}, sort_keys=True), flush=True)
    if not args.launch:
        return 0
    CONTROL_ROOT.mkdir(parents=True, exist_ok=False) if not CONTROL_ROOT.exists() else None
    started = [_launch_one(run_index) for run_index in RUN_INDICES]
    print(json.dumps({"event": "started", "runs": started}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"KL recovery launcher failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
