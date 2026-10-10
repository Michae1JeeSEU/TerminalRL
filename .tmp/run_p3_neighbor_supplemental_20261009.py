"""Guarded one-shot launcher for the supplemental P3 neighbor comparison stages.

This file is intentionally preparation-only until a caller selects a stage.  Each
selected stage performs one remote SSH invocation and refuses existing artifacts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import shlex
import subprocess
from pathlib import Path

SSH_TARGET = "lab-server"
REMOTE_PROJECT = "/home/user/projects/terminal"
REMOTE_PYTHON = "/home/user/envs/terminal_env/bin/python"
REMOTE_ROOT = "/home/user/training_outputs/terminal"
AUX_BASE = f"{REMOTE_ROOT}/p3_neighbor_supplemental_witness_20261009"
CV_BASE = f"{REMOTE_ROOT}/p3_neighbor_nine_cycle_cv_20261009"
OLD_WITNESS = f"{REMOTE_PROJECT}/.tmp/p3_neighbor_witness_train_20261009"

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

CHECK_TESTS = [
    "tests/test_p3_v26_train_dataset.py",
    "tests/test_p3_neighbor_witness_data.py",
    "tests/test_p3_neighbor_witness_dataset.py",
    "tests/test_intent_neighbor_actor.py",
    "tests/test_p3_supplemental_neighbor_data.py",
]
MODIFIED_MODULES = [
    "terminal_marl/training/p3_v26_train_dataset.py",
    "terminal_marl/training/p3_neighbor_witness_data.py",
    "terminal_marl/training/p3_neighbor_witness_dataset.py",
    "terminal_marl/scripts/diagnose_p3_intent_neighbor.py",
]
CV_EXTRA_FILES = [
    "terminal_marl/algo/intent_neighbor_actor.py",
    "terminal_marl/algo/ordered_proposal_actor.py",
    "terminal_marl/algo/joint_ppo_v4_2.py",
    "terminal_marl/training/p3_v25_set_learnability.py",
    "terminal_marl/training/p3_v3_bc_learnability.py",
    "docs/evidence/v4_2/p3_neighbor_observability_repair_20261009.md",
]
OLD_WITNESS_FILES = [
    f".tmp/p3_neighbor_witness_train_20261009/{name}"
    for name in ("protocol.json", "result.json", "witnesses.jsonl")
]
LAUNCHER_FILE = ".tmp/run_p3_neighbor_supplemental_20261009.py"
LOADER_CHECK_SCRIPT = ".tmp/check_p3_nine_cycle_inputs_20261009.py"
LOADER_SOURCE_FILES = tuple(
    MODIFIED_MODULES
    + [
        "terminal_marl/training/p3_v26_resource_inputs.py",
        "terminal_marl/algo/intent_neighbor_actor.py",
    ]
)
EXPECTED_NINE_SEEDS = (893511, 893512, 893513, 893601, 893602, 893603, 893604, 893605, 893606)


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def load_frozen_hash_files() -> list[str]:
    """Reuse the existing resource launcher list and add this launcher's checks."""
    tree = ast.parse(
        Path(".tmp/run_p3_v26_resources.py").read_text(encoding="utf-8")
    )
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "files"
            for target in node.targets
        )
    )
    value = assignment.value
    if not isinstance(value, ast.BinOp):
        raise RuntimeError("expected .tmp/run_p3_v26_resources.py files=[...] + tests")
    base_files = ast.literal_eval(value.left)
    files = list(base_files) + CHECK_TESTS + MODIFIED_MODULES
    files += CV_EXTRA_FILES + OLD_WITNESS_FILES + [LAUNCHER_FILE, LOADER_CHECK_SCRIPT]
    unique = list(dict.fromkeys(files))
    missing = [path for path in unique if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing local hash inputs: " + ", ".join(missing))
    return unique


def frozen_checks() -> str:
    return "".join(
        f"{hashlib.sha256(Path(path).read_bytes()).hexdigest()}  {path}\n"
        for path in load_frozen_hash_files()
    )


def env_command(argv: list[str], *, timeout_seconds: int | None = None) -> str:
    command = [
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
    ]
    if timeout_seconds is not None:
        command += ["timeout", str(timeout_seconds)]
    command += [REMOTE_PYTHON, *argv]
    return shlex.join(command)


def json_guard(code: str, paths: list[str]) -> str:
    return env_command(["-c", code, *paths])


def no_existing_output(lines: list[str], output: str) -> None:
    for suffix in ("", ".pid", ".wrapper.pid", ".log", ".exit"):
        lines.append(f"test ! -e {shell_quote(output + suffix)}")


def require_preflight(lines: list[str]) -> None:
    preflight_exit = f"{CV_BASE}/preflight.exit"
    preflight_log = f"{CV_BASE}/preflight.log"
    lines.extend(
        [
            f"test -s {shell_quote(preflight_log)}",
            f"test -e {shell_quote(preflight_exit)}",
            f'test "$(cat {shell_quote(preflight_exit)})" = 0',
        ]
    )


def require_status(
    lines: list[str], paths: list[str], status: str
) -> None:
    code = (
        "import json,sys; "
        "assert all(json.load(open(path, encoding='utf-8'))['status']==sys.argv[1] "
        "for path in sys.argv[2:])"
    )
    lines.append(
        json_guard(code, [status, *paths])
    )


def require_protocol_source_group(
    lines: list[str], protocol_paths: list[str], source_group: str
) -> None:
    code = (
        "import json,sys; "
        "assert all(json.load(open(path, encoding='utf-8'))['source_group']==sys.argv[1] "
        "for path in sys.argv[2:])"
    )
    lines.append(json_guard(code, [source_group, *protocol_paths]))


def require_loader_check(lines: list[str], loader_check: str) -> None:
    lines.append(f"test -s {shell_quote(loader_check)}")
    code = (
        "import hashlib,json,sys; "
        "r=json.load(open(sys.argv[1],encoding='utf-8')); "
        "assert r['status']=='PASS'; "
        f"required={list(LOADER_SOURCE_FILES)!r}; "
        "hashes=r['source_hashes']; "
        "assert isinstance(hashes,dict) and set(required).issubset(hashes); "
        "assert all(hashes[path]==hashlib.sha256(open(path,'rb').read()).hexdigest() "
        "for path in required); "
        f"assert r['source_file']=={LOADER_CHECK_SCRIPT!r}; "
        "source_path=r['source_file']; recorded=r['source_file_sha256']; "
        "actual=hashlib.sha256(open(source_path,'rb').read()).hexdigest(); "
        "assert actual==recorded"
    )
    lines.append(json_guard(code, [loader_check]))


def background_stage(
    lines: list[str], command: str, output: str, post_guard: str
) -> None:
    log = output + ".log"
    pid = output + ".pid"
    wrapper_pid = output + ".wrapper.pid"
    exit_file = output + ".exit"
    body = "\n".join(
        [
            "set -u",
            f"cd {shell_quote(REMOTE_PROJECT)}",
            f"{command} > {shell_quote(log)} 2>&1 &",
            "task_pid=$!",
            f"printf '%s\n' \"$task_pid\" > {shell_quote(pid)}",
            "wait \"$task_pid\"",
            "code=$?",
            'if [ "$code" -eq 0 ]; then',
            f"  {post_guard}",
            "  post_code=$?",
            '  if [ "$post_code" -ne 0 ]; then code="$post_code"; fi',
            "fi",
            f"printf '%s\n' \"$code\" > {shell_quote(exit_file)}",
            "exit \"$code\"",
        ]
    )
    lines.extend(
        [
            f"nohup sh -c {shell_quote(body)} > /dev/null 2>&1 < /dev/null &",
            "wrapper_pid=$!",
            f"printf '%s\n' \"$wrapper_pid\" > {shell_quote(wrapper_pid)}",
            "printf 'wrapper_pid=%s\n' \"$wrapper_pid\"",
        ]
    )


def common_lines(checks: str) -> list[str]:
    return [
        "set -eu",
        f"cd {shell_quote(REMOTE_PROJECT)}",
        f"printf %s {shell_quote(checks)} | sha256sum -c -",
        "test -z \"$(pgrep -f '^/home/user/envs/terminal_env/bin/python.*terminal_marl' || true)\"",
    ]


def run_stage(stage: str, loader_check: str | None) -> int:
    checks = frozen_checks()
    lines = common_lines(checks)
    preflight_log = f"{CV_BASE}/preflight.log"
    preflight_exit = f"{CV_BASE}/preflight.exit"

    if stage == "check":
        lines.extend(
            [
                f"mkdir -p {shell_quote(CV_BASE)}",
                f"test ! -e {shell_quote(preflight_log)}",
                f"test ! -e {shell_quote(preflight_exit)}",
                "set +e",
                env_command(["-m", "pytest", *CHECK_TESTS, "-q"])
                + f" > {shell_quote(preflight_log)} 2>&1",
                "code=$?",
                f"printf '%s\n' \"$code\" > {shell_quote(preflight_exit)}",
                f"cat {shell_quote(preflight_log)}",
                'exit "$code"',
            ]
        )
    elif stage == "synthetic":
        output = f"{CV_BASE}/smoke_synthetic"
        lines.extend([f"mkdir -p {shell_quote(CV_BASE)}"])
        require_preflight(lines)
        no_existing_output(lines, output)
        command = env_command(
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
                f"{AUX_BASE}/three_cycle",
                "--smoke",
                "--output",
                output,
            ],
            timeout_seconds=180,
        )
        guard = json_guard(
            "import json,sys; "
            "r=json.load(open(sys.argv[1],encoding='utf-8')); "
            "p=r['protocol']; folds=r['folds']; "
            "assert r['status']=='COMPLETE' and len(folds)==9; "
            "assert sorted(f['held_seed'] for f in folds)==list(range(1,10)); "
            "assert p['source_group']=='old6_plus_supplemental3'; "
            "assert p['gate_mode']=='first_feasible_union'; "
            "assert p['arms']==['native_pair_auxiliary']; "
            "assert all(f['arm']=='native_pair_auxiliary' and "
            "[point['update'] for point in f['curve']]==[0,1,2] "
            "for f in folds)",
            [output + "/result.json"],
        )
        background_stage(lines, command, output, guard)
    elif stage == "aux_smoke":
        output = f"{AUX_BASE}/smoke"
        lines.extend([f"mkdir -p {shell_quote(AUX_BASE)}"])
        require_preflight(lines)
        no_existing_output(lines, output)
        require_status(
            lines,
            [f"{path}/result.json" for path in SUPPLEMENTAL_TRAIN],
            "COLLECTION_OK",
        )
        command = env_command(
            [
                "-u",
                "-m",
                "terminal_marl.training.p3_neighbor_witness_data",
                "--supplemental",
                "--train-directories",
                *SUPPLEMENTAL_TRAIN,
                "--output",
                output,
                "--smoke",
            ],
            timeout_seconds=180,
        )
        guard = json_guard(
            "import json,sys; "
            "r=json.load(open(sys.argv[1],encoding='utf-8')); "
            "p=json.load(open(sys.argv[2],encoding='utf-8')); "
            "assert r['status']=='SMOKE_OK' and p['source_group']=='supplemental'",
            [output + "/result.json", output + "/protocol.json"],
        )
        background_stage(lines, command, output, guard)
    elif stage == "aux_full":
        output = f"{AUX_BASE}/three_cycle"
        lines.extend([f"mkdir -p {shell_quote(AUX_BASE)}"])
        require_preflight(lines)
        no_existing_output(lines, output)
        require_status(lines, [f"{AUX_BASE}/smoke/result.json"], "SMOKE_OK")
        require_status(
            lines,
            [f"{path}/result.json" for path in SUPPLEMENTAL_TRAIN],
            "COLLECTION_OK",
        )
        command = env_command(
            [
                "-u",
                "-m",
                "terminal_marl.training.p3_neighbor_witness_data",
                "--supplemental",
                "--train-directories",
                *SUPPLEMENTAL_TRAIN,
                "--output",
                output,
            ],
            timeout_seconds=1900,
        )
        guard = json_guard(
            "import json,sys; "
            "r=json.load(open(sys.argv[1],encoding='utf-8')); "
            "p=json.load(open(sys.argv[2],encoding='utf-8')); "
            "assert r['status']=='COMPLETE' and p['source_group']=='supplemental'",
            [output + "/result.json", output + "/protocol.json"],
        )
        background_stage(lines, command, output, guard)
    elif stage == "cv":
        if not loader_check:
            raise SystemExit("stage cv requires --loader-check PATH")
        output = f"{CV_BASE}/nine_cycle"
        lines.extend([f"mkdir -p {shell_quote(CV_BASE)}"])
        require_preflight(lines)
        no_existing_output(lines, output)
        require_status(
            lines,
            [f"{OLD_WITNESS}/result.json", f"{AUX_BASE}/three_cycle/result.json"],
            "COMPLETE",
        )
        require_status(
            lines,
            [f"{CV_BASE}/smoke_synthetic/result.json"],
            "COMPLETE",
        )
        require_protocol_source_group(
            lines,
            [f"{AUX_BASE}/three_cycle/protocol.json"],
            "supplemental",
        )
        require_loader_check(lines, loader_check)
        command = env_command(
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
                f"{AUX_BASE}/three_cycle",
                "--output",
                output,
            ],
            timeout_seconds=11400,
        )
        guard = json_guard(
            "import json,sys; "
            "r=json.load(open(sys.argv[1],encoding='utf-8')); "
            f"p=r['protocol']; folds=r['folds']; expected={list(EXPECTED_NINE_SEEDS)!r}; "
            "assert r['status']=='COMPLETE' and len(folds)==9; "
            "assert sorted(f['held_seed'] for f in folds)==expected; "
            "assert p['arms']==['native_pair_auxiliary']; "
            "assert p['gate_mode']=='first_feasible_union'; "
            "assert p['source_group']=='old6_plus_supplemental3'; "
            "assert p['supplemental_role']=='train_only_development_not_qualification'; "
            "assert p['qualification'] is False; "
            "assert all(f['arm']=='native_pair_auxiliary' and "
            "[point['update'] for point in f['curve']]==[0,100,500,1000,2000,5000] "
            "for f in folds)",
            [output + "/result.json"],
        )
        background_stage(lines, command, output, guard)
    else:
        raise AssertionError(f"unsupported stage: {stage}")

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
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one guarded supplemental P3 neighbor stage; "
            "maximal stages are single-shot and never reuse outputs."
        )
    )
    parser.add_argument(
        "stage", choices=("check", "synthetic", "aux_smoke", "aux_full", "cv")
    )
    parser.add_argument(
        "--loader-check",
        help=(
            "required remote nine-cycle loader-check JSON; cv verifies PASS, "
            "fixed source_hashes, and source_file_sha256 against source_file"
        ),
    )
    args = parser.parse_args()
    return run_stage(args.stage, args.loader_check)


if __name__ == "__main__":
    raise SystemExit(main())
