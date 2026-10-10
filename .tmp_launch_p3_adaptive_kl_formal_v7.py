"""Preflight and optionally launch the three-run V7 formal campaign.

The default invocation is check-only and creates no output or control files.
``--launch`` repeats all three check-only calls and then starts one detached
supervisor per run.  The supervisor records the child PID, status, logs, and
terminal exit code; it never retries a failed child.
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
RUNNER_MODULE = "terminal_marl.training.run_p3_adaptive_kl_formal_v7"
OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926"
)
CONTROL_ROOT = OUTPUT_ROOT / "_control"
SOURCE_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_gate_coverage_repair_v2_20260925/fit853401/"
    "p3_gate_coverage_repair_v2.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba"
)
ADMISSION_REPORT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_adaptive_kl_probe_v1_20260926/formal_v7_admission.json"
)
RUN_INDICES = (0, 1, 2)
SUPERVISE = "--_supervise"


def _env() -> dict[str, str]:
    result = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        result[name] = "1"
    return result


def _validate_fixed_inputs(source: Path, admission: Path) -> None:
    if source != SOURCE_CHECKPOINT:
        raise ValueError(f"source checkpoint must be the fixed 03c0 path: {SOURCE_CHECKPOINT}")
    if admission != ADMISSION_REPORT:
        raise ValueError(f"admission report must be the fixed path: {ADMISSION_REPORT}")
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint missing: {source}")
    if not admission.is_file():
        raise FileNotFoundError(f"formal admission report missing: {admission}")


def _runner_argv(
    run_index: int, output: Path, source: Path, admission: Path, *, check_only: bool,
) -> list[str]:
    if run_index not in RUN_INDICES:
        raise ValueError("formal run index must be 0, 1, or 2")
    argv = [
        str(PYTHON), "-m", RUNNER_MODULE,
        "--run-index", str(run_index),
        "--output", str(output),
        "--checkpoint", str(source),
        "--admission-report", str(admission),
    ]
    if check_only:
        argv.append("--check-only")
    return argv


def _validate_argv(argv: Sequence[str]) -> None:
    if any("\r" in item or "\n" in item for item in argv):
        raise RuntimeError(f"newline in subprocess argument: {argv!r}")


def _output_path(run_index: int) -> Path:
    return OUTPUT_ROOT / f"run{run_index}"


def _control_path(run_index: int) -> Path:
    return CONTROL_ROOT / f"run{run_index}"


def _paths(run_index: int) -> dict[str, Path]:
    prefix = _control_path(run_index)
    return {
        "command": prefix / "command.json",
        "supervisor_pid": prefix / "supervisor.pid",
        "pid": prefix / "pid",
        "status": prefix / "status.json",
        "exitcode": prefix / "exitcode",
        "stdout": prefix / "stdout.log",
        "stderr": prefix / "stderr.log",
        "supervisor_stdout": prefix / "supervisor.stdout.log",
        "supervisor_stderr": prefix / "supervisor.stderr.log",
        "supervisor_error": prefix / "supervisor_error",
    }


def _exclusive_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_status(path: Path, status: str, **fields: Any) -> None:
    path.write_text(
        json.dumps({"status": status, **fields}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _assert_clean() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing existing formal output/control root: {OUTPUT_ROOT}")
    if CONTROL_ROOT.exists():
        raise FileExistsError(f"refusing existing control root: {CONTROL_ROOT}")


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


def _check_one(run_index: int, source: Path, admission: Path) -> dict[str, Any]:
    output = _output_path(run_index)
    argv = _runner_argv(run_index, output, source, admission, check_only=True)
    _validate_argv(argv)
    completed = subprocess.run(
        argv, cwd=PROJECT_ROOT, env=_env(), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False,
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
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": parsed.get("source_checkpoint_sha256"),
    }
    if completed.stderr.strip():
        result["stderr_nonempty"] = True
    result["passed"] = bool(
        completed.returncode == 0
        and parsed.get("status") == "CHECK_ONLY_OK"
        and parsed.get("check_only") is True
        and parsed.get("output_created") is False
        and parsed.get("output_existed") is False
        and parsed.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256
    )
    return result


def _check_all(source: Path, admission: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for run_index in RUN_INDICES:
        try:
            result = _check_one(run_index, source, admission)
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
        raise RuntimeError("formal V7 check-only failed for " + ", ".join(failures))
    return checks


def _supervise(run_index: int, source: Path, admission: Path) -> int:
    paths = _paths(run_index)
    output = _output_path(run_index)
    argv = _runner_argv(run_index, output, source, admission, check_only=False)
    _validate_argv(argv)
    child: subprocess.Popen[str] | None = None
    returncode = 1
    try:
        _write_status(paths["status"], "STARTING", run_index=run_index)
        with paths["stdout"].open("x", encoding="utf-8", newline="") as stdout, \
                paths["stderr"].open("x", encoding="utf-8", newline="") as stderr:
            child = subprocess.Popen(
                argv, cwd=PROJECT_ROOT, env=_env(), stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True, text=True,
            )
            _exclusive_text(paths["pid"], f"{child.pid}\n")
            _write_status(paths["status"], "RUNNING", run_index=run_index, child_pid=child.pid)
            returncode = int(child.wait())
        _exclusive_text(paths["exitcode"], f"{returncode}\n")
        _write_status(
            paths["status"], "COMPLETED" if returncode == 0 else "FAILED",
            run_index=run_index, child_pid=child.pid if child else None,
            exitcode=returncode,
        )
        return returncode
    except BaseException as exc:
        try:
            _exclusive_text(paths["supervisor_error"], f"{type(exc).__name__}: {exc}\n")
        except FileExistsError:
            pass
        try:
            if not paths["exitcode"].exists():
                _exclusive_text(paths["exitcode"], f"{returncode}\n")
            _write_status(paths["status"], "SUPERVISOR_FAILED", run_index=run_index,
                          error=f"{type(exc).__name__}: {exc}")
        except OSError:
            pass
        return returncode if returncode != 0 else 1


def _launch_one(run_index: int, source: Path, admission: Path) -> dict[str, Any]:
    paths = _paths(run_index)
    output = _output_path(run_index)
    supervisor_argv = [
        str(PYTHON), str(Path(__file__).resolve()), SUPERVISE,
        "--run-index", str(run_index),
        "--source-checkpoint", str(source),
        "--admission-report", str(admission),
    ]
    _validate_argv(supervisor_argv)
    _exclusive_text(paths["command"], json.dumps(
        _runner_argv(run_index, output, source, admission, check_only=False),
        separators=(",", ":"),
    ) + "\n")
    with paths["supervisor_stdout"].open("x", encoding="utf-8", newline="") as stdout, \
            paths["supervisor_stderr"].open("x", encoding="utf-8", newline="") as stderr:
        supervisor = subprocess.Popen(
            supervisor_argv, cwd=PROJECT_ROOT, env=_env(), stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=True, text=True,
        )
    _exclusive_text(paths["supervisor_pid"], f"{supervisor.pid}\n")
    return {
        "run_index": run_index,
        "supervisor_pid": supervisor.pid,
        "child_pid": None,
        "status": "SUPERVISOR_STARTED",
        "output": str(output),
        "control": str(_control_path(run_index)),
    }


def _await_started(item: dict[str, Any]) -> dict[str, Any]:
    run_index = int(item["run_index"])
    paths = _paths(run_index)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if paths["pid"].is_file():
            child_pid = int(paths["pid"].read_text(encoding="utf-8").strip())
            status = "RUNNING"
            if paths["status"].is_file():
                try:
                    status = str(json.loads(paths["status"].read_text(encoding="utf-8"))[
                        "status"
                    ])
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    status = "RUNNING"
            return {**item, "child_pid": child_pid, "status": status}
        if paths["supervisor_error"].is_file():
            error = paths["supervisor_error"].read_text(encoding="utf-8").strip()
            raise RuntimeError(f"run{run_index} supervisor failed: {error}")
        time.sleep(0.05)
    raise TimeoutError(f"run{run_index} supervisor did not publish child PID")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true",
                        help="preflight all runs, then start detached supervisors")
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--admission-report", type=Path, default=ADMISSION_REPORT)
    parser.add_argument(SUPERVISE, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-index", type=int, choices=RUN_INDICES, help=argparse.SUPPRESS)
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    source = Path(args.source_checkpoint)
    admission = Path(args.admission_report)
    _validate_fixed_inputs(source, admission)
    if args._supervise:
        if args.run_index not in RUN_INDICES:
            raise SystemExit("--_supervise requires --run-index 0, 1, or 2")
        return _supervise(int(args.run_index), source, admission)
    if args.run_index is not None:
        raise SystemExit("--run-index is only valid with internal supervision")
    _assert_clean()
    checks = _check_all(source, admission)
    print(json.dumps({"event": "check_only", "checks": checks}, sort_keys=True), flush=True)
    if not args.launch:
        return 0
    CONTROL_ROOT.mkdir(parents=True, exist_ok=False)
    for run_index in RUN_INDICES:
        _control_path(run_index).mkdir(parents=True, exist_ok=False)
    launched = [_launch_one(run_index, source, admission) for run_index in RUN_INDICES]
    started = [_await_started(item) for item in launched]
    print(json.dumps({"event": "started", "runs": started}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"formal V7 launcher failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
