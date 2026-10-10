"""Prepare and launch the bounded V6 pilot/formal campaign on the lab server.

This helper is intentionally separate from the training runner.  It performs
the runner's check-only preflight first, refuses existing stage/run outputs,
and starts one detached supervisor only after the required checks pass.  The
supervisor owns the training child and records its PID and exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path("/home/user/projects/terminal")
PYTHON = Path("/home/user/envs/terminal_env/bin/python")
OUTPUT_ROOT = Path("/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925")
SOURCE_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_gate_coverage_repair_v2_20260925/fit853401/"
    "p3_gate_coverage_repair_v2.pt"
)
RUNNER_MODULE = "terminal_marl.training.run_v4_2_p3_gate_anchored_v6"
SCRIPT_PATH = Path(__file__).resolve()
FORMAL_RUNS = (0, 1, 2)
_SUPERVISE = "--_supervise"


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = "1"
    return env


def _runner_argv(
    *, stage: str, output: Path, run_index: int | None = None,
    admission_report: Path | None = None,
) -> list[str]:
    argv = [
        str(PYTHON), "-m", RUNNER_MODULE,
        "--stage", stage,
        "--checkpoint", str(SOURCE_CHECKPOINT),
        "--output", str(output),
    ]
    if stage == "formal":
        if run_index not in FORMAL_RUNS:
            raise ValueError("formal run index must be 0, 1, or 2")
        if admission_report is None:
            raise ValueError("formal admission report is required")
        argv.extend(["--run-index", str(run_index), "--admission-report", str(admission_report)])
    return argv


def _exclusive_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_status(control: Path, status: str, **fields: Any) -> None:
    payload = {"status": status, **fields}
    (control / "status.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _control_path(
    stage: str, run_index: int | None, pilot_attempt_name: str = "pilot",
) -> Path:
    name = pilot_attempt_name if stage == "pilot" else f"formal_run{run_index}"
    return OUTPUT_ROOT / "_control" / name


def _output_path(
    stage: str, run_index: int | None, pilot_attempt_name: str = "pilot",
) -> Path:
    if stage == "pilot":
        return OUTPUT_ROOT / pilot_attempt_name
    return OUTPUT_ROOT / "formal" / f"run{run_index}"


def _prepare_control(
    stage: str, run_index: int | None, pilot_attempt_name: str = "pilot",
) -> tuple[Path, Path]:
    control = _control_path(stage, run_index, pilot_attempt_name)
    output = _output_path(stage, run_index, pilot_attempt_name)
    if control.exists():
        raise FileExistsError(f"control directory already exists: {control}")
    if output.exists():
        raise FileExistsError(f"training output already exists: {output}")
    control.mkdir(parents=True, exist_ok=False)
    _exclusive_text(control / "stage.txt", stage + "\n")
    run_name = pilot_attempt_name if run_index is None else str(run_index)
    _exclusive_text(control / "run.txt", run_name + "\n")
    _write_status(control, "CHECK_ONLY_PENDING", stage=stage, run_index=run_index)
    return control, output


def _check_only(
    *, stage: str, output: Path, control: Path, run_index: int | None,
    admission_report: Path | None,
) -> dict[str, Any]:
    argv = _runner_argv(
        stage=stage, output=output, run_index=run_index,
        admission_report=admission_report,
    )
    completed = subprocess.run(
        [*argv, "--check-only"],
        cwd=REPO_ROOT,
        env=_server_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    _exclusive_text(control / "check_stdout.log", completed.stdout)
    _exclusive_text(control / "check_stderr.log", completed.stderr)
    _exclusive_text(control / "check_exitcode.txt", f"{completed.returncode}\n")
    parsed: dict[str, Any] = {}
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    result = {
        "stage": stage,
        "run_index": run_index,
        "returncode": completed.returncode,
        "runner_status": parsed.get("status"),
        "source_checkpoint_sha256": parsed.get("source_checkpoint_sha256"),
        "check_only": parsed.get("check_only"),
        "runner_argv": argv,
    }
    (control / "check_only.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    passed = completed.returncode == 0 and parsed.get("status") == "CHECK_ONLY_OK"
    _write_status(control, "CHECK_ONLY_OK" if passed else "CHECK_ONLY_FAILED", **result)
    if not passed:
        raise RuntimeError(
            f"check-only failed for {stage} {run_index}: "
            f"returncode={completed.returncode}, status={parsed.get('status')}"
        )
    return result


def _supervise(control: Path, runner_argv: Sequence[str]) -> int:
    """Run exactly one training child and record its terminal exit code."""
    _write_status(control, "SUPERVISOR_STARTING")
    stdout_handle = None
    stderr_handle = None
    try:
        stdout_handle = (control / "stdout.log").open("x", encoding="utf-8", newline="\n")
        stderr_handle = (control / "stderr.log").open("x", encoding="utf-8", newline="\n")
        child = subprocess.Popen(
            list(runner_argv),
            cwd=REPO_ROOT,
            env=_server_env(),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        _exclusive_text(control / "pid.txt", f"{child.pid}\n")
        _write_status(control, "RUNNING", child_pid=child.pid)
        returncode = child.wait()
        _exclusive_text(control / "exitcode.txt", f"{returncode}\n")
        _write_status(control, "EXITED", child_pid=child.pid, exitcode=returncode)
        return returncode
    except BaseException as exc:
        _exclusive_text(control / "supervisor_error.txt", f"{type(exc).__name__}: {exc}\n")
        _write_status(control, "SUPERVISOR_FAILED", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def _launch_supervisor(
    *, stage: str, output: Path, control: Path, run_index: int | None,
    admission_report: Path | None,
) -> dict[str, Any]:
    runner_argv = _runner_argv(
        stage=stage, output=output, run_index=run_index,
        admission_report=admission_report,
    )
    supervisor_argv = [
        str(PYTHON), str(SCRIPT_PATH), _SUPERVISE,
        "--control", str(control),
        "--runner-argv-json", json.dumps(runner_argv, separators=(",", ":")),
    ]
    supervisor_stdout = (control / "supervisor.stdout.log").open("x", encoding="utf-8")
    supervisor_stderr = (control / "supervisor.stderr.log").open("x", encoding="utf-8")
    supervisor = subprocess.Popen(
        supervisor_argv,
        cwd=REPO_ROOT,
        env=_server_env(),
        stdin=subprocess.DEVNULL,
        stdout=supervisor_stdout,
        stderr=supervisor_stderr,
        start_new_session=True,
    )
    supervisor_stdout.close()
    supervisor_stderr.close()
    _exclusive_text(control / "supervisor_pid.txt", f"{supervisor.pid}\n")

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if (control / "pid.txt").exists() or (control / "supervisor_error.txt").exists():
            break
        time.sleep(0.05)
    child_pid = None
    if (control / "pid.txt").exists():
        child_pid = int((control / "pid.txt").read_text(encoding="utf-8").strip())
    elif (control / "supervisor_error.txt").exists():
        raise RuntimeError((control / "supervisor_error.txt").read_text(encoding="utf-8").strip())
    else:
        raise TimeoutError(f"supervisor did not publish child PID: {control}")
    return {
        "stage": stage,
        "run_index": run_index,
        "control_dir": str(control),
        "output_dir": str(output),
        "supervisor_pid": supervisor.pid,
        "child_pid": child_pid,
        "runner_argv": runner_argv,
    }


def _run_campaign(
    stage: str, admission_report: Path | None, pilot_attempt_name: str,
) -> int:
    if stage == "pilot":
        controls = [(None, *_prepare_control(stage, None, pilot_attempt_name))]
    else:
        # Reserve all formal control/output paths before any check-only call or
        # launch, so a partial formal campaign cannot overwrite a prior one.
        controls = [(index, *_prepare_control(stage, index)) for index in FORMAL_RUNS]

    checks: list[dict[str, Any]] = []
    check_failures: list[str] = []
    for run_index, control, output in controls:
        try:
            checks.append(_check_only(
                stage=stage, output=output, control=control,
                run_index=run_index, admission_report=admission_report,
            ))
            print(json.dumps({"event": "check_only", **checks[-1]}, sort_keys=True), flush=True)
        except Exception as exc:
            check_failures.append(f"{stage} {run_index}: {type(exc).__name__}: {exc}")
            print(json.dumps({
                "event": "check_only_failed", "stage": stage,
                "run_index": run_index, "error": f"{type(exc).__name__}: {exc}",
            }, sort_keys=True), flush=True)

    if check_failures:
        raise RuntimeError("check-only failed; no training supervisor started: " + " | ".join(check_failures))

    # For formal, this loop is reached only after all three check-only calls
    # have passed.
    for run_index, control, output in controls:
        started = _launch_supervisor(
            stage=stage, output=output, control=control,
            run_index=run_index, admission_report=admission_report,
        )
        print(json.dumps({"event": "started", **started}, sort_keys=True), flush=True)
    return 0


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "formal"), required=True)
    parser.add_argument("--admission-report", type=Path)
    parser.add_argument("--pilot-attempt-name", choices=("pilot", "pilot_restart1"), default="pilot")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == _SUPERVISE:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(_SUPERVISE, action="store_true")
        parser.add_argument("--control", type=Path, required=True)
        parser.add_argument("--runner-argv-json", required=True)
        args = parser.parse_args(values)
        return _supervise(args.control, json.loads(args.runner_argv_json))

    args = _parse_args(values)
    if args.stage == "formal" and args.admission_report is None:
        raise SystemExit("--admission-report is required for --stage formal")
    if args.stage == "pilot" and args.admission_report is not None:
        raise SystemExit("--admission-report is only valid for --stage formal")
    if args.stage == "formal" and args.pilot_attempt_name != "pilot":
        raise SystemExit("--pilot-attempt-name is only valid for --stage pilot")
    return _run_campaign(args.stage, args.admission_report, args.pilot_attempt_name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
