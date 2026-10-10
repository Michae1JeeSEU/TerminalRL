"""Guarded launcher for the registered fixed eight-arm V3 comparison.

The default invocation is a no-side-effect dry description.  Launching is
deliberately opt-in and requires the caller to provide the reviewed local
commit.  This helper does not construct an environment or policy itself; it
only validates the fixed output/checkpoint preconditions and starts the
registered evaluator as a detached server process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/user/projects/terminal")
PYTHON_EXECUTABLE = "/home/user/envs/terminal_env/bin/python"
EVALUATOR_MODULE = "terminal_marl.scripts.evaluate_p3_foundation_repair_20261002"
CHECKPOINT_SOURCE_PATH = (
    "/home/user/training_outputs/terminal/"
    "p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt"
)
CHECKPOINT_PATH = Path(CHECKPOINT_SOURCE_PATH)
CHECKPOINT_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
OUTPUT_ROOT = Path("/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002")
MATRIX_LOG = OUTPUT_ROOT / "matrix.log"
PROGRESS_JSON = OUTPUT_ROOT / "progress.json"
COMPARISON_JSON = OUTPUT_ROOT / "comparison.json"
LAUNCH_METADATA = OUTPUT_ROOT / "launch_metadata.json"

# These are the evaluator, dense24 builder, and execution/safety sources whose
# hashes are captured with the detached launch metadata.
CODE_PATHS = (
    "terminal_marl/scripts/evaluate_p3_foundation_repair_20261002.py",
    "terminal_marl/training/evaluate_p3_task_neutral.py",
    "terminal_marl/training/evaluate_p3_proposal_head_probe_v1.py",
    "terminal_marl/training/paired_gate_diagnostic.py",
    "terminal_marl/training/run_p3_forced_wait_frozen_20260930.py",
    "terminal_marl/training/terminal_lifecycle_metrics.py",
    "terminal_marl/envs/parallel_env.py",
    "terminal_marl/core/config.py",
    "terminal_marl/core/trajectory_generator.py",
    "terminal_marl/core/polynomial_trajectory.py",
    "terminal_marl/core/trajectory_time_parameterization.py",
    "terminal_marl/core/vehicle_dynamics.py",
    "terminal_marl/core/safety_filter.py",
    "terminal_marl/core/runtime_traffic_safety.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for relative in CODE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            missing.append(str(path))
            continue
        hashes[relative] = _sha256(path)
    if missing:
        raise RuntimeError("required source file(s) missing: " + ", ".join(missing))
    return hashes


def _guard() -> dict[str, Any]:
    if not PROJECT_ROOT.is_dir():
        raise RuntimeError(f"project root missing: {PROJECT_ROOT}")
    if not OUTPUT_ROOT.is_dir():
        raise RuntimeError(f"fixed output root missing: {OUTPUT_ROOT}")
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(f"fixed checkpoint missing: {CHECKPOINT_PATH}")
    if str(CHECKPOINT_PATH) != CHECKPOINT_SOURCE_PATH:
        raise RuntimeError("fixed checkpoint source path changed")
    observed_checkpoint_sha = _sha256(CHECKPOINT_PATH)
    if observed_checkpoint_sha.lower() != CHECKPOINT_SHA256.lower():
        raise RuntimeError(
            "fixed checkpoint SHA mismatch: "
            f"expected {CHECKPOINT_SHA256}, observed {observed_checkpoint_sha}"
        )

    forbidden = [PROGRESS_JSON, COMPARISON_JSON, MATRIX_LOG, LAUNCH_METADATA]
    existing = [str(path) for path in forbidden if path.exists()]
    existing.extend(str(path) for path in OUTPUT_ROOT.rglob("result.json"))
    if existing:
        raise RuntimeError("matrix output already exists: " + ", ".join(sorted(existing)))

    return {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": observed_checkpoint_sha,
        "code_hashes": _code_hashes(),
    }


def _dry_payload() -> dict[str, Any]:
    return {
        "status": "DRY_RUN",
        "launch_required": True,
        "argv": [PYTHON_EXECUTABLE, "-u", "-m", EVALUATOR_MODULE, "--run"],
        "cwd": str(PROJECT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "side_effects": "none",
    }


def _launch(commit: str) -> int:
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        print(json.dumps({"status": "ERROR", "error": "--commit must be a full 40-hex commit"}))
        return 2

    try:
        binding = _guard()
    except (OSError, RuntimeError) as exc:
        print(json.dumps({"status": "GUARD_REJECTED", "error": str(exc)}))
        return 2

    argv = [PYTHON_EXECUTABLE, "-u", "-m", EVALUATOR_MODULE, "--run"]
    log_handle = None
    process = None
    try:
        log_handle = MATRIX_LOG.open("x", encoding="utf-8")
        process = subprocess.Popen(
            argv,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        metadata = {
            "status": "launched",
            "launched_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": int(process.pid),
            "local_commit": commit.lower(),
            "argv": argv,
            "cwd": str(PROJECT_ROOT),
            "output_root": str(OUTPUT_ROOT),
            "matrix_log": str(MATRIX_LOG),
            "checkpoint_path": binding["checkpoint_path"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "code_hashes": binding["code_hashes"],
        }
        LAUNCH_METADATA.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "LAUNCH_UNCERTAIN" if process is not None else "LAUNCH_FAILED",
            "error": str(exc),
            "pid": None if process is None else int(process.pid),
        }))
        return 2
    finally:
        if log_handle is not None:
            log_handle.close()

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--commit")
    args = parser.parse_args(argv)
    if not args.launch:
        if args.commit is not None:
            parser.error("--commit requires --launch")
        print(json.dumps(_dry_payload(), indent=2, sort_keys=True))
        return 0
    if args.commit is None:
        print(json.dumps({
            "status": "ERROR",
            "error": "--launch requires --commit <full 40-hex commit>",
        }))
        return 2
    return _launch(args.commit)


if __name__ == "__main__":
    raise SystemExit(main())
