"""Guarded launcher for the P3 neighbor-prefix coverage stages.

The prefix collector is ``terminal_marl.training.p3_neighbor_prefix_coverage_data``.
Its prepare-only mode performs the declared offline validation and writes the
selection/budget record without snapshot reads or native planner calls.  The
smoke and full stages use the same collector with their registered bounds; CV
is a separate explicitly requested stage guarded on completed full output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


SSH_TARGET = "lab-server"
REMOTE_PROJECT = "/home/user/projects/terminal"
REMOTE_PYTHON = "/home/user/envs/terminal_env/bin/python"
REMOTE_ROOT = "/home/user/training_outputs/terminal"

PREFIX_BASE = f"{REMOTE_ROOT}/p3_neighbor_prefix_coverage_20261010"
PREPARE_OUTPUT = f"{PREFIX_BASE}/prepare"
SMOKE_OUTPUT = f"{PREFIX_BASE}/smoke"
FULL_OUTPUT = f"{PREFIX_BASE}/full"
CV_OUTPUT = f"{PREFIX_BASE}/cv/nine_cycle"

MAX_CANDIDATES_PER_PARENT = 150
MAX_PLANS_PER_PARENT = 900
FULL_TASK_TIMEOUT = 5700
FULL_OUTER_TIMEOUT = 6000
SMOKE_TASK_TIMEOUT = 120
SMOKE_OUTER_TIMEOUT = 180
CV_OUTER_TIMEOUT = 4200
WORKERS = 4

# Keep the canonical roots in one place so a stage cannot silently select a
# different source set.
ORIGINAL_TRAIN = [
    f"{REMOTE_ROOT}/p3_v25_neutral_course_v2_20261008/train_flow6_mixed",
    f"{REMOTE_ROOT}/p3_v25_neutral_course_v2_20261008/train_dense24_fixed",
    f"{REMOTE_ROOT}/p3_v25_neutral_course_v2_20261008/train_dense24_mixed",
]
ADDITIONAL_TRAIN = [
    f"{REMOTE_ROOT}/p3_v26_resources_20261009/train_01",
    f"{REMOTE_ROOT}/p3_v26_resources_20261009/train_02",
    f"{REMOTE_ROOT}/p3_v26_resources_20261009/train_03",
]
SUPPLEMENTAL_TRAIN = [
    f"{REMOTE_ROOT}/p3_v26_supplemental_20261009/train_01",
    f"{REMOTE_ROOT}/p3_v26_supplemental_20261009/train_02",
    f"{REMOTE_ROOT}/p3_v26_supplemental_20261009/train_03",
]
OLD_WITNESS = f"{REMOTE_PROJECT}/.tmp/p3_neighbor_witness_train_20261009"
SUPPLEMENTAL_WITNESS = (
    f"{REMOTE_ROOT}/p3_neighbor_supplemental_witness_20261009/three_cycle"
)

# This is the source list used for the dynamic SHA check.  The expected values
# are computed at invocation time so the server must run the exact locally
# synced collector, data modules, actor, and focused tests.
HASH_FILES = [
    "terminal_marl/training/p3_neighbor_prefix_coverage_data.py",
    "terminal_marl/training/p3_v26_train_dataset.py",
    "terminal_marl/training/p3_v25_course_data.py",
    "terminal_marl/training/p3_v26_course_data.py",
    "terminal_marl/training/p3_v26_supplemental_data.py",
    "terminal_marl/training/p3_neighbor_witness_data.py",
    "terminal_marl/training/p3_neighbor_witness_dataset.py",
    "terminal_marl/scripts/diagnose_p3_intent_neighbor.py",
    "terminal_marl/algo/intent_neighbor_actor.py",
    "tests/test_p3_v26_train_dataset.py",
    "tests/test_p3_neighbor_witness_data.py",
    "tests/test_p3_neighbor_witness_dataset.py",
    "tests/test_p3_supplemental_neighbor_data.py",
    "tests/test_p3_neighbor_prefix_coverage_data.py",
    "tests/test_p3_neighbor_prefix_parallel.py",
    "tests/test_p3_neighbor_prefix_queries.py",
    ".tmp/run_p3_neighbor_prefix_20261010.py",
]

COLLECTOR_MODULE = "terminal_marl.training.p3_neighbor_prefix_coverage_data"


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def frozen_checks() -> str:
    missing = [path for path in HASH_FILES if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing local hash inputs: " + ", ".join(missing))
    return "".join(
        f"{hashlib.sha256(Path(path).read_bytes()).hexdigest()}  {path}\n"
        for path in HASH_FILES
    )


def env_command(argv: list[str]) -> str:
    command = [
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
    ]
    command += [REMOTE_PYTHON, *argv]
    return shlex.join(command)


def no_existing_output(lines: list[str], output: str) -> None:
    for suffix in ("", ".pid", ".root.pid", ".wrapper.pid", ".log", ".exit"):
        lines.append(f"test ! -e {shell_quote(output + suffix)}")


def require_json_status(lines: list[str], path: str, status: str) -> None:
    code = (
        "import json,sys; "
        "assert json.load(open(sys.argv[1],encoding='utf-8'))['status']==sys.argv[2]"
    )
    lines.append(env_command(["-c", code, path, status]))


def require_zero_exit(lines: list[str], output: str) -> None:
    lines.append(f"test \"$(cat {shell_quote(output + '.exit')})\" = 0")


def require_prepare_contract(lines: list[str]) -> None:
    code = (
        "import json,sys; "
        "r=json.load(open(sys.argv[1],encoding='utf-8')); "
        "p=json.load(open(sys.argv[2],encoding='utf-8')); "
        "s=json.load(open(sys.argv[3],encoding='utf-8')); "
        "assert r.get('status')=='PREPARED'; "
        "assert p.get('new_selected_candidate_count')==713; "
        "assert p.get('all_selected_candidate_count')==733; "
        "assert s.get('counts',{}).get('new_candidate_groups')==713"
    )
    lines.append(env_command([
        "-c", code,
        PREPARE_OUTPUT + "/result.json",
        PREPARE_OUTPUT + "/protocol.json",
        PREPARE_OUTPUT + "/selection.json",
    ]))


def require_smoke_contract(lines: list[str]) -> None:
    require_zero_exit(lines, SMOKE_OUTPUT)
    require_json_status(lines, SMOKE_OUTPUT + "/result.json", "SMOKE_OK")


def require_full_contract(lines: list[str]) -> None:
    require_zero_exit(lines, FULL_OUTPUT)
    require_json_status(lines, FULL_OUTPUT + "/result.json", "COMPLETE")


def common_lines(checks: str) -> list[str]:
    return [
        "set -eu",
        f"cd {shell_quote(REMOTE_PROJECT)}",
        f"printf %s {shell_quote(checks)} | sha256sum -c -",
        "test -z \"$(pgrep -f '^/home/user/envs/terminal_env/bin/python.*terminal_marl' || true)\"",
    ]


def background_stage(
    lines: list[str],
    command: str,
    output: str,
    *,
    outer_timeout: int,
    post_guard: str | None = None,
) -> None:
    log = output + ".log"
    root_pid = output + ".root.pid"
    wrapper_pid = output + ".wrapper.pid"
    exit_file = output + ".exit"
    body = "\n".join(
        [
            "set -u",
            f"cd {shell_quote(REMOTE_PROJECT)}",
            f"timeout {outer_timeout} {command} > {shell_quote(log)} 2>&1",
            "code=$?",
            *(["if [ \"$code\" -eq 0 ]; then", f"  {post_guard}", "  post_code=$?", "  if [ \"$post_code\" -ne 0 ]; then code=\"$post_code\"; fi", "fi"] if post_guard else []),
            f"printf '%s\\n' \"$code\" > {shell_quote(exit_file)}",
            "exit \"$code\"",
        ]
    )
    lines.extend(
        [
            f"nohup sh -c {shell_quote(body)} > /dev/null 2>&1 < /dev/null &",
            "wrapper_pid=$!",
            f"printf '%s\\n' \"$wrapper_pid\" > {shell_quote(root_pid)}",
            f"printf '%s\\n' \"$wrapper_pid\" > {shell_quote(wrapper_pid)}",
            "printf 'wrapper_pid=%s\\n' \"$wrapper_pid\"",
        ]
    )


def collector_argv(
    *, output: str, smoke: bool = False, prepare_only: bool = False
) -> list[str]:
    if smoke and prepare_only:
        raise ValueError("--smoke and --prepare-only are mutually exclusive")
    argv = [
        "-u", "-m", COLLECTOR_MODULE,
        "--old-train-directories", *ORIGINAL_TRAIN,
        "--additional-train-directories", *ADDITIONAL_TRAIN,
        "--supplemental-train-directories", *SUPPLEMENTAL_TRAIN,
        "--old-witness-directory", OLD_WITNESS,
        "--supplemental-witness-directory", SUPPLEMENTAL_WITNESS,
        "--output", output,
    ]
    if smoke:
        argv.append("--smoke")
    if prepare_only:
        argv.append("--prepare-only")
    return argv


def cv_command(output: str) -> str:
    return env_command(
        [
            "-u",
            "-m",
            "terminal_marl.scripts.diagnose_p3_intent_neighbor",
            "--train-directories",
            *ORIGINAL_TRAIN,
            "--additional-train-directories",
            *ADDITIONAL_TRAIN,
            "--witness-directory",
            OLD_WITNESS,
            "--supplemental-train-directories",
            *SUPPLEMENTAL_TRAIN,
            "--supplemental-witness-directory",
            SUPPLEMENTAL_WITNESS,
            "--prefix-witness-directory",
            FULL_OUTPUT,
            "--workers",
            str(WORKERS),
            "--output",
            output,
        ],
    )


def cv_guard_code(output: str) -> str:
    del output
    return (
        "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); "
        "p=r['protocol']; f=r['folds']; "
        "assert r['status']=='COMPLETE' and len(f)==9; "
        "assert p['gate_mode']=='first_feasible_union'; "
        "assert p['arms']==['native_pair_auxiliary']; "
        "assert p['source_group']=='old6_plus_supplemental3_plus_prefix'; "
        "assert p['qualification'] is False; "
        "assert all(x['arm']=='native_pair_auxiliary' and "
        "[m['update'] for m in x['curve']]==[0,100,500,1000,2000,5000] for x in f)"
    )


def prepare_spec() -> dict[str, object]:
    return {
        "status": "DRY_RUN_ONLY",
        "collector_module": COLLECTOR_MODULE,
        "collector_flags": [
            "--old-train-directories", "--additional-train-directories",
            "--supplemental-train-directories", "--old-witness-directory",
            "--supplemental-witness-directory", "--output", "--smoke",
            "--prepare-only",
        ],
        "stages": ["prepare", "smoke", "full", "cv"],
        "paths": {
            "base": PREFIX_BASE,
            "prepare": PREPARE_OUTPUT,
            "smoke": SMOKE_OUTPUT,
            "full": FULL_OUTPUT,
            "cv": CV_OUTPUT,
            "old_witness": OLD_WITNESS,
            "supplemental_witness": SUPPLEMENTAL_WITNESS,
        },
        "input_directories": {
            "original": ORIGINAL_TRAIN,
            "additional": ADDITIONAL_TRAIN,
            "supplemental": SUPPLEMENTAL_TRAIN,
        },
        "limits": {
            "max_candidates_per_parent": MAX_CANDIDATES_PER_PARENT,
            "max_plans_per_parent": MAX_PLANS_PER_PARENT,
            "total_candidates_all_nine": MAX_CANDIDATES_PER_PARENT * 9,
            "total_plans_all_nine": MAX_PLANS_PER_PARENT * 9,
            "full_task_timeout": FULL_TASK_TIMEOUT,
            "full_outer_timeout": FULL_OUTER_TIMEOUT,
            "smoke_task_timeout": SMOKE_TASK_TIMEOUT,
            "smoke_outer_timeout": SMOKE_OUTER_TIMEOUT,
            "smoke_requests": 3,
            "smoke_plans": 18,
            "cv_workers": WORKERS,
            "cv_outer_timeout": CV_OUTER_TIMEOUT,
        },
        "source_hashes": "computed dynamically at invocation",
        "side_effects": {
            "server_source_upload": False,
            "server_git": False,
            "query_or_training_at_prepare": False,
        },
}


def run_stage(stage: str, *, dry_run: bool = False) -> int:
    checks = frozen_checks()
    lines = common_lines(checks)
    if stage == "prepare":
        output = PREPARE_OUTPUT
        lines.append(f"mkdir -p {shell_quote(PREFIX_BASE)}")
        no_existing_output(lines, output)
        command = env_command(collector_argv(output=output, prepare_only=True))
        log = output + ".log"
        exit_file = output + ".exit"
        lines.append(
            f"if {command} > {shell_quote(log)} 2>&1; then code=0; "
            f"else code=$?; fi; printf '%s\\n' \"$code\" > {shell_quote(exit_file)}; exit \"$code\""
        )
    elif stage == "smoke":
        output = SMOKE_OUTPUT
        lines.extend([f"mkdir -p {shell_quote(PREFIX_BASE)}"])
        no_existing_output(lines, output)
        command = env_command(collector_argv(output=output, smoke=True))
        background_stage(lines, command, output, outer_timeout=SMOKE_OUTER_TIMEOUT)
    elif stage == "full":
        output = FULL_OUTPUT
        lines.extend([f"mkdir -p {shell_quote(PREFIX_BASE)}"])
        no_existing_output(lines, output)
        require_json_status(lines, PREPARE_OUTPUT + "/result.json", "PREPARED")
        require_zero_exit(lines, PREPARE_OUTPUT)
        require_prepare_contract(lines)
        require_smoke_contract(lines)
        command = env_command(collector_argv(output=output))
        background_stage(lines, command, output, outer_timeout=FULL_OUTER_TIMEOUT)
    elif stage == "cv":
        output = CV_OUTPUT
        lines.extend([f"mkdir -p {shell_quote(PREFIX_BASE + '/cv')}"])
        no_existing_output(lines, output)
        require_full_contract(lines)
        command = cv_command(output)
        background_stage(
            lines,
            command,
            output,
            outer_timeout=CV_OUTER_TIMEOUT,
            post_guard=env_command(["-c", cv_guard_code(output), output + "/result.json"]),
        )
    else:
        raise AssertionError(stage)

    if dry_run:
        print(json.dumps({
            "stage": stage,
            "ssh": ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", SSH_TARGET],
            "remote_script": "\n".join(lines),
            **prepare_spec(),
        }, ensure_ascii=False, sort_keys=True))
        return 0

    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            SSH_TARGET,
            "\n".join(lines),
        ],
        check=False,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "full", "cv"))
    parser.add_argument("--dry-run", action="store_true", help="print the guarded command without SSH execution")
    args = parser.parse_args()
    return run_stage(args.stage, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
