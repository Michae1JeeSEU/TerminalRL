"""Single missing f67 job, with separately identified inner reference workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.server import recover_rotating_slots_diagnostics as recovery
from scripts.server.parallel_rotating_slots_runtime import (
    ReferencePool, diagnostic_world_factory, execution_identity)
from terminal_marl.core.rotating_slots import diagnostics
from terminal_marl.core.rotating_slots.qualification import write_json


def parallel_job(job, *, inner_workers, recovery_commit, superseded_attempt=None):
    root = Path(job["output"])
    identity = execution_identity(recovery_commit, inner_workers)
    identity["status"] = "RUNNING"
    identity["actual_command"] = job["command"]
    identity["replacement_reason"] = "user_authorized_multicore_replacement"
    identity["superseded_attempt"] = str(superseded_attempt) if superseded_attempt else None
    identity["replacement_scope"] = (
        "replaces the single-core recovery attempt; does not infer another power loss "
        "or infer a cause from SSH availability; original-source interruption remains separate")
    identity["metrics_comparison_limit"] = (
        "parallel timing differs from original concurrent 20-job matrix; "
        "no method/baseline performance conclusion")
    metadata_path = root/"parallel_execution.json"
    write_json(metadata_path, identity)
    started = time.perf_counter()
    pool = ReferencePool(inner_workers, stats_path=root/"parallel_workers.json")
    try:
        with pool, diagnostic_world_factory(pool):
            row = diagnostics._diagnostic_job(job)
        identity.update(status=row["status"], workers=pool.summary(),
                        runner_wall_seconds=time.perf_counter()-started)
        row["execution"] = identity
        row["cpu_seconds_parent_only"] = row["cpu_seconds"]
        row["cpu_seconds_scope"] = "parent only; child completed-pad-task CPU recorded separately"
        row["child_task_cpu_seconds"] = identity["workers"]["child_task_cpu_seconds"]
        row["known_cpu_seconds_parent_plus_child_tasks"] = (
            row["cpu_seconds"] + row["child_task_cpu_seconds"])
        row["measured_components"]["scope"] = "parent process only; excludes all child CPU"
        row["measurement_instrumentation"].update(
            process_scope="parent only", child_profile="enabled per reference pad task",
            worker_measurement_scope=identity["workers"]["measurement_scope"])
        write_json(root/job["id"]/"result.json", row)
        write_json(metadata_path, identity)
        # The frozen world uses integer pad keys in memory; its JSON archive
        # uses string keys. Legacy recovery compares the returned row to the
        # actual JSON object, so use the same serialized representation.
        return recovery._read_json(root/job["id"]/"result.json")
    except BaseException as exc:
        pool.close()
        identity.update(status="FAILED", workers=pool.summary(),
            runner_wall_seconds=time.perf_counter()-started,
            failure={"type": type(exc).__name__, "message": str(exc)})
        write_json(metadata_path, identity)
        raise


def recover_parallel(source, output, *, code_commit, recovery_commit, inner_workers, command,
                     superseded_attempt=None):
    # Keep the original exact 61/60 validation, archive checks, and source guards.
    recovery._wrapper_identity(recovery_commit)
    if inner_workers < 1:
        raise ValueError("--inner-workers must be positive")
    def runner(job):
        return parallel_job(job, inner_workers=inner_workers, recovery_commit=recovery_commit,
                            superseded_attempt=superseded_attempt)
    report = recovery.recover(source, output, code_commit=code_commit, workers=1,
        command=command, recovery_commit=recovery_commit, runner=runner)
    restarted = next(r for r in report["jobs"] if r["id"] == recovery.INTERRUPTED_ID)
    report["execution"] = restarted["execution"]
    report["recovery"]["execution_metadata"] = str(Path(output)/"parallel_execution.json")
    write_json(Path(output)/"recovery_manifest.json", report["recovery"])
    write_json(Path(output)/"diagnostics.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-commit", choices=(recovery.FROZEN_COMMIT,), required=True)
    parser.add_argument("--recovery-commit", required=True)
    parser.add_argument("--inner-workers", type=int, required=True,
                        help="reference pad workers inside ONE independent frozen job")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--superseded-attempt", help="prior single-core recovery output; recorded only, never modified")
    args = parser.parse_args(argv)
    command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()),
                                      *(sys.argv[1:] if argv is None else argv)])
    try:
        recovery._wrapper_identity(args.recovery_commit)
        if args.inner_workers < 1:
            raise ValueError("--inner-workers must be positive")
        if args.preflight_only:
            plan = recovery.preflight(args.source, args.output, code_commit=args.code_commit,
                                      workers=1, command=command)
            print(json.dumps({"status": "PREFLIGHT_PASSED", "independent_jobs": 1,
                "execution": execution_identity(args.recovery_commit, args.inner_workers),
                "validated_completed_count": len(plan.rows), "output_created": False}, indent=2))
        else:
            result = recover_parallel(args.source, args.output, code_commit=args.code_commit,
                recovery_commit=args.recovery_commit, inner_workers=args.inner_workers, command=command,
                superseded_attempt=args.superseded_attempt)
            print(json.dumps({"status": "COMPLETED", "coverage": result["coverage"]}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "RECOVERY_NOT_COMPLETE", "error_type": type(exc).__name__,
                          "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
