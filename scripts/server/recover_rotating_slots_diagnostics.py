"""Restart only the power-interrupted f67 formal job, preserving its old attempt.

This wrapper is outside the frozen provenance glob. It does not restore a world
checkpoint, alter the protocol, run Git, or write into the original output tree.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _name in THREADS:
    os.environ[_name] = "1"
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from terminal_marl.core.rotating_slots import diagnostics
from terminal_marl.core.rotating_slots.fixtures import event_equivalent
from terminal_marl.core.rotating_slots.qualification import provenance, task_coverage, write_json
from terminal_marl.core.rotating_slots.replay import ReplayBundle

FROZEN_COMMIT = "f67b5c8462433ba701191c493a37e4a8e472334e"
FROZEN_CODE_SHA = "110be32211d37e82772d806d86ae599d0a58a32e01af3465be836a58de902750"
FROZEN_CONFIG_HASH = "0c7cce0c0c019d50f7bba5aecd5e8340925d0d16867f8f06eb51e89102343d53"
INTERRUPTED_ID = "traffic_rate180_seed71"
WRAPPER_VERSION = "1.0"
TAIL_BYTES = 131072


class RecoveryError(ValueError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    def invalid(value):
        raise RecoveryError("non-finite JSON value: " + value)
    return json.loads(Path(path).read_bytes(), parse_constant=invalid)


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrapper_identity(recovery_commit=None):
    if recovery_commit is not None:
        if not isinstance(recovery_commit, str) or len(recovery_commit) != 40:
            raise RecoveryError("--recovery-commit must be a supplied 40-digit commit hash")
        try:
            int(recovery_commit, 16)
        except ValueError as exc:
            raise RecoveryError("invalid supplied recovery commit") from exc
    path = Path(__file__).resolve()
    return {"version": WRAPPER_VERSION, "path": str(path), "sha256": _hash(path),
            "provided_recovery_commit": recovery_commit, "commit_verification": "external; no server Git lookup"}


def frozen_jobs():
    """The exact order/parameters of f67 run_diagnostics(pilot=None)."""
    jobs = [dict(id=f"traffic_rate{rate}_seed{seed}", kind="traffic", rate=rate,
                 seed=seed, window=3600., drain=1800.)
            for rate in (60, 120, 180) for seed in (11, 23, 47, 71, 101)]
    jobs += [dict(id=f"benchmark_n{n}_{mode}_r{repeat}", kind="benchmark",
                  count=n, mode=mode, repeat=repeat, duration=1.)
             for n in (10, 30, 60, 83) for mode in ("stable", "inflight_attempt")
             for repeat in range(5)]
    jobs += [dict(id=f"dt_{dt}", kind="dt", dt=dt) for dt in (.2, .1, .05)]
    jobs += [dict(id=f"scheduler_{period}", kind="scheduler", period=period,
                  seed=11, rate=120, window=3600., drain=1800.)
             for period in (1., 5., 10.)]
    return [dict(job, pilot=False) for job in jobs]


def _check_job(job, spec, output, command=None):
    if not isinstance(job, dict) or set(job) != set(spec) | {"output", "command"}:
        raise RecoveryError("job fields differ from frozen matrix: " + spec["id"])
    for key, expected in spec.items():
        value = job[key]
        if value != expected or isinstance(value, bool) != isinstance(expected, bool):
            raise RecoveryError(f"frozen job parameter mismatch: {spec['id']}.{key}")
    if not isinstance(job["output"], str) or not Path(job["output"]).is_absolute():
        raise RecoveryError("job output must be its actual absolute path")
    if Path(job["output"]).resolve() != output:
        raise RecoveryError("recorded job output differs from --source/--output: " + spec["id"])
    if not isinstance(job["command"], str) or not job["command"].strip():
        raise RecoveryError("missing original job command")
    if command is not None and job["command"] != command:
        raise RecoveryError("job command mismatch: " + spec["id"])


def _current_provenance(command, code_commit):
    if code_commit != FROZEN_COMMIT:
        raise RecoveryError("only the f67 frozen matrix is supported")
    os.environ["TERMINAL_CODE_COMMIT"] = code_commit
    diagnostics.limit_threads()
    info = provenance(command)
    if (info["code_commit"], info["code_sha256"], info["config_hash"]) != (
            FROZEN_COMMIT, FROZEN_CODE_SHA, FROZEN_CONFIG_HASH):
        raise RecoveryError("current frozen source/config fingerprint mismatch")
    return info


def _check_provenance(info, current, command):
    if not isinstance(info, dict):
        raise RecoveryError("missing result provenance")
    for field in ("code_commit", "code_sha256", "config_hash", "source_files_sha256"):
        if info.get(field) != current[field]:
            raise RecoveryError("result provenance mismatch: " + field)
    digest = sha256(json.dumps(info["source_files_sha256"], sort_keys=True).encode()).hexdigest()
    if digest != FROZEN_CODE_SHA or info.get("command") != command:
        raise RecoveryError("result source digest/command mismatch")


def _check_row(row, spec, output, current, command=None):
    if not isinstance(row, dict) or row.get("id") != spec["id"] or row.get("status") != "COMPLETED":
        raise RecoveryError("only real COMPLETED results may be reused: " + spec["id"])
    job = row.get("job")
    _check_job(job, spec, output, command)
    _check_provenance(row.get("provenance"), current, job["command"])
    root = output / spec["id"]
    replay = root / "replay"
    for key, path in (("replay_path", replay), ("events_path", replay / "events.jsonl"),
                      ("manifest_path", replay / "manifest.json")):
        if not isinstance(row.get(key), str) or Path(row[key]).resolve() != path.resolve():
            raise RecoveryError("result archive path mismatch: " + spec["id"] + "." + key)
    if not replay.resolve().is_relative_to(output):
        raise RecoveryError("replay escapes the selected output")
    archive = ReplayBundle.validate(replay)
    index = archive.index
    yaml_key = "new10.16/fixed_terminal_rotating_slots_stage1_config_v1_0.yaml"
    if (index.get("code_commit") != FROZEN_COMMIT or index.get("config_hash") != FROZEN_CONFIG_HASH
            or index.get("config_bytes_sha256") != current["source_files_sha256"][yaml_key]
            or index.get("command") != job["command"] or index.get("seed") != job.get("seed", 11)):
        raise RecoveryError("bundle/result provenance mismatch: " + spec["id"])
    metrics = _read_json(replay / "metrics.json")
    if row.get("metrics") != metrics or metrics.get("conservation") is not True:
        raise RecoveryError("bundle/result metrics mismatch: " + spec["id"])
    if row.get("state_count") != archive.state_count:
        raise RecoveryError("bundle/result state count mismatch: " + spec["id"])
    if spec["kind"] in ("traffic", "scheduler"):
        if row.get("manifest") != _read_json(replay / "manifest.json"):
            raise RecoveryError("bundle/result manifest mismatch: " + spec["id"])
    if spec["kind"] == "dt":
        events = [_read_json_line(line) for line in (replay / "events.jsonl").read_bytes().splitlines() if line.strip()]
        if row.get("events") != events:
            raise RecoveryError("bundle/result dt events mismatch")
    return replay / "replay.json"


def _read_json_line(line):
    return json.loads(line)


def _file_record(path, *, hashed=True):
    path = Path(path)
    stat = path.stat()
    result = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
              "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
    if hashed:
        result["sha256"] = _hash(path)
    return result


def _interrupted_evidence(root):
    progress = root / "progress.json"
    states = root / "states.jsonl"
    if not root.is_dir() or not progress.is_file() or not states.is_file():
        raise RecoveryError("missing interrupted job directory/progress/states")
    record = _file_record(states, hashed=False)
    with states.open("rb") as stream:
        start = max(0, record["bytes"] - TAIL_BYTES)
        stream.seek(start)
        tail = stream.read(TAIL_BYTES)
    lines = tail.split(b"\n")
    if start:
        lines = lines[1:]
    complete = lines[:-1]
    last = None
    for line in reversed(complete):
        try:
            frame = json.loads(line)
            last = {"time": frame.get("time"), "failed": frame.get("failed")}
            break
        except (ValueError, AttributeError):
            continue
    record["bounded_tail"] = {"maximum_bytes": TAIL_BYTES, "bytes_read": len(tail),
        "sha256": sha256(tail).hexdigest(), "base64": base64.b64encode(tail).decode("ascii"),
        "ends_with_newline": tail.endswith(b"\n"), "last_complete_projection": last}
    return {"job_directory": str(root), "progress": _file_record(progress),
            "progress_record": _read_json(progress), "states": record,
            "interpretation": "interrupted attempt retained; no checkpoint restore or physical failure inferred"}


@dataclass
class RecoveryPlan:
    source: Path
    output: Path
    specs: list
    source_jobs: list
    rows: dict
    current: dict
    evidence: dict


def preflight(source, output, *, code_commit, workers=1, command="preflight-only"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if workers != 1:
        raise RecoveryError("exactly one missing job requires --workers 1")
    if output.exists():
        raise FileExistsError("recovery output already exists: " + str(output))
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise RecoveryError("recovery output must be separate from the original tree")
    if not source.is_dir() or (source / "diagnostics.json").exists():
        raise RecoveryError("source must be the interrupted, unaggregated diagnostics directory")
    current = _current_provenance(command, code_commit)
    specs = frozen_jobs()
    jobs_path = source / "jobs.json"
    jobs = _read_json(jobs_path)
    if not isinstance(jobs, list):
        raise RecoveryError("jobs.json must be the original 61-element list")
    coverage = task_coverage([s["id"] for s in specs], jobs)
    if coverage["missing"] or coverage["duplicates"] or coverage["unexpected"] or len(jobs) != 61:
        raise RecoveryError("source jobs matrix coverage mismatch")
    by_id = {j["id"]: j for j in jobs}
    original_command = jobs[0].get("command")
    for spec in specs:
        _check_job(by_id[spec["id"]], spec, source, original_command)
    unexpected = {p.parent.name for p in source.glob("*/result.json")} - set(by_id)
    if unexpected:
        raise RecoveryError("unexpected result IDs: " + str(sorted(unexpected)))
    rows, records = {}, []
    for spec in specs:
        result_path = source / spec["id"] / "result.json"
        if spec["id"] == INTERRUPTED_ID:
            if result_path.exists():
                raise RecoveryError("the interrupted job already has a result; do not rerun it")
            continue
        row = _read_json(result_path)
        if row.get("job") != by_id[spec["id"]]:
            raise RecoveryError("source jobs/result job mismatch: " + spec["id"])
        replay_index = _check_row(row, spec, source, current, original_command)
        rows[spec["id"]] = row
        records.append({"id": spec["id"], "result": _file_record(result_path),
                        "replay_index": _file_record(replay_index)})
    interrupted = _interrupted_evidence(source / INTERRUPTED_ID)
    #Exercise aggregate input fields before the potentially long single job.
    aggregate([rows[s["id"]] for s in specs if s["id"] in rows], command)
    evidence = {"source_directory": str(source), "jobs": _file_record(jobs_path),
        "reused_results": records, "interrupted_attempt": interrupted,
        "frozen_provenance": {k: current[k] for k in ("code_commit", "code_sha256", "config_hash", "source_files_sha256")},
        "validated_completed_count": len(rows), "restart_job_id": INTERRUPTED_ID,
        "expected_job_ids": [s["id"] for s in specs]}
    return RecoveryPlan(source, output, specs, jobs, rows, current, evidence)


def aggregate(rows, command):
    """Equivalent f67 report fields, with canonical matrix ordering supplied by caller."""
    dtrows = [r for r in rows if r.get("job", {}).get("kind") == "dt" and r["status"] == "COMPLETED"]
    dtcomparison = {"status": "NOT_RUN"}
    if len(dtrows) == 3:
        energies = [r["energy_j"] for r in dtrows]
        minimums = [r["minimum"] for r in dtrows]
        dtcomparison = {"status": "MEASURED", "energy_relative_spread": max(energies) / min(energies) - 1,
            "energy_within_0p005": max(energies) / min(energies) - 1 <= .005,
            "events_within_contract_tolerance": all(event_equivalent(r["events"], dtrows[0]["events"]) for r in dtrows),
            "identical_committed_script": all(r["committed_script"] == dtrows[0]["committed_script"] for r in dtrows),
            "minimum_width_across_dt_m": max(x["upper_bound_m"] for x in minimums) - min(x["lower_bound_m"] for x in minimums)}
    return {"schema": "FT-RS-S1-1.0-diagnostics", "protocol": "FROZEN_FORMAL_MATRIX",
        "jobs": rows, "coverage": task_coverage([j["id"] for j in frozen_jobs()], rows),
        "benchmark_summary": diagnostics.benchmark_summary(rows), "dt_comparison": dtcomparison,
        "scheduler_sensitivity": [{"period": r["job"]["period"], "metrics": r.get("metrics"), "cpu_seconds": r.get("cpu_seconds"),
            "decision_count": len(r.get("decision_times", [])), "wall_seconds": r.get("wall_seconds")}
            for r in rows if r.get("job", {}).get("kind") == "scheduler"],
        "researcher_visual_acceptance": "PENDING", "stage1_status": "NOT_ACCEPTED", "command": command}


def _source_unchanged(plan):
    records = [plan.evidence["jobs"], plan.evidence["interrupted_attempt"]["progress"]]
    for row in plan.evidence["reused_results"]:
        records.extend((row["result"], row["replay_index"]))
    for record in records:
        if _hash(record["path"]) != record["sha256"]:
            raise RecoveryError("source changed during recovery: " + record["path"])
    record = plan.evidence["interrupted_attempt"]["states"]
    stat = Path(record["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
        raise RecoveryError("interrupted states changed during recovery")
    if (plan.source / INTERRUPTED_ID / "result.json").exists():
        raise RecoveryError("original interrupted attempt acquired a result during recovery")


def recover(source, output, *, code_commit, workers=1, command, recovery_commit=None, runner=None):
    wrapper_identity = _wrapper_identity(recovery_commit)
    plan = preflight(source, output, code_commit=code_commit, workers=workers, command=command)
    started, wall_start = _now(), time.perf_counter()
    plan.output.mkdir(parents=True, exist_ok=False)
    recovery = {"schema": "rotating-slots-f67-single-job-recovery-1.0", "status": "RUNNING",
        "wrapper": wrapper_identity,
        "command": command, "output_directory": str(plan.output), "started_at": started,
        "reason": {"external_event": "power loss reported by user", "process_absence": "verified externally; see launcher logs",
                   "wrapper_process_check": "not performed", "physical_failure_inferred": False},
        "checkpoint_resume": False, "restart_from_simulated_time": 0., "reexecuted_job_ids": [INTERRUPTED_ID],
        "reused_result_count": 60, "workers": 1, "worker_reason": "only one incomplete frozen job; completed jobs are never rerun",
        "threads": {k: os.environ.get(k) for k in THREADS}, "source": plan.evidence,
        "timing_scope": "old per-job measurements are preserved; interrupted and restarted attempts are not combined into continuous runtime"}
    manifest_path = plan.output / "recovery_manifest.json"
    write_json(manifest_path, recovery)
    spec = next(s for s in plan.specs if s["id"] == INTERRUPTED_ID)
    job = dict(spec, output=str(plan.output), command=command)
    jobs = [job if j["id"] == INTERRUPTED_ID else j for j in plan.source_jobs]
    write_json(plan.output / "jobs.json", jobs)
    write_json(plan.output / "progress.json", {"completed": 60, "total": 61, "restart_job_id": INTERRUPTED_ID,
        "coverage": task_coverage([s["id"] for s in plan.specs], list(plan.rows.values()))})
    try:
        row = (diagnostics._diagnostic_job if runner is None else runner)(job)
        result_path = plan.output / INTERRUPTED_ID / "result.json"
        if _read_json(result_path) != row:
            raise RecoveryError("dispatched result differs from the actual written result")
        _check_row(row, spec, plan.output, plan.current, command)
        _source_unchanged(plan)
        rows = [row if s["id"] == INTERRUPTED_ID else plan.rows[s["id"]] for s in plan.specs]
        report = aggregate(rows, command)
        coverage = report["coverage"]
        if coverage["actual_count"] != 61 or any(coverage[k] for k in ("missing", "unexpected", "duplicates", "failures", "pending")):
            raise RecoveryError("final 61-job coverage is incomplete")
        recovery.update(status="COMPLETED", finished_at=_now(), recovered_result=_file_record(result_path),
            timing={"wrapper_wall_seconds": time.perf_counter() - wall_start,
                    "restarted_job_wall_seconds": row.get("wall_seconds"),
                    "interrupted_last_progress_wall_seconds": plan.evidence["interrupted_attempt"]["progress_record"].get("wall_seconds")})
        report["recovery"] = recovery
        write_json(manifest_path, recovery)
        write_json(plan.output / "progress.json", {"completed": 61, "total": 61, "coverage": coverage})
        #No formal report exists until the one new real result and its archive pass validation.
        write_json(plan.output / "diagnostics.json", report)
        return report
    except Exception as exc:
        recovery.update(status="FAILED", finished_at=_now(), failure={"type": type(exc).__name__, "message": str(exc)},
                        wrapper_wall_seconds=time.perf_counter() - wall_start)
        write_json(manifest_path, recovery)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-commit", required=True, choices=(FROZEN_COMMIT,))
    parser.add_argument("--recovery-commit", help="supplied wrapper/orchestration commit, separate from frozen code identity")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()),
                                      *(sys.argv[1:] if argv is None else argv)])
    try:
        if args.preflight_only:
            wrapper_identity = _wrapper_identity(args.recovery_commit)
            plan = preflight(args.source, args.output, code_commit=args.code_commit, workers=args.workers, command=command)
            print(json.dumps({"status": "PREFLIGHT_PASSED", "output_created": False,
                              "wrapper": wrapper_identity, "evidence": plan.evidence}, indent=2))
        else:
            result = recover(args.source, args.output, code_commit=args.code_commit, workers=args.workers,
                             command=command, recovery_commit=args.recovery_commit)
            print(json.dumps({"status": "COMPLETED", "coverage": result["coverage"],
                              "report": str(Path(args.output).resolve() / "diagnostics.json")}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "RECOVERY_NOT_COMPLETE", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
