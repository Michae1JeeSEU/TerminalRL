"""Qualify G09 from 40 existing measured archives, without running simulations.

This supplementary entry point does not execute the frozen catalogue fixture or
assert completion of the other diagnostics jobs. Measurement identity belongs to
the consumed source; the independent verifier's identity is recorded separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.qualification import (
    Evidence, catalogue, provenance, task_coverage, write_json,
)
from terminal_marl.core.rotating_slots.replay import ReplayBundle
from terminal_marl.core.rotating_slots.world import AIRBORNE

CONFIG_KEY = "new10.16/fixed_terminal_rotating_slots_stage1_config_v1_0.yaml"
COMPONENTS = {"generation", "motion_power", "separation", "resources", "state_update",
              "serialization", "emergency", "other"}
IDENTITY_FIELDS = ("code_commit", "code_sha256", "config_hash", "source_files_sha256")
LIMITATION = ("Consumes 40 existing measured benchmarks; no absolute throughput pass threshold; "
              "does not certify performance of a newer simulator, newer admission rules or throughput, "
              "steady state, or real-time operation.")


class QualificationError(ValueError):
    pass


def benchmark_specs():
    return [dict(id=f"benchmark_n{n}_{mode}_r{repeat}", kind="benchmark", count=n,
                 mode=mode, repeat=repeat, duration=1., pilot=False)
            for n in (10, 30, 60, 83) for mode in ("stable", "inflight_attempt")
            for repeat in range(5)]


def _read(path):
    return json.loads(Path(path).read_bytes())


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise QualificationError(message)


def _number(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _identity(info, *, expected_commit, expected_sha, config):
    _require(isinstance(info, dict), "missing source provenance")
    _require(info.get("code_commit") == expected_commit, "source code_commit mismatch")
    _require(info.get("code_sha256") == expected_sha, "source code_sha256 mismatch")
    _require(info.get("config_hash") == config.config_hash, "source frozen config mismatch")
    hashes = info.get("source_files_sha256")
    _require(isinstance(hashes, dict) and bool(hashes), "missing source hash mapping")
    for name, digest in hashes.items():
        _require(isinstance(name, str) and name and not Path(name).is_absolute()
                 and ".." not in Path(name).parts and isinstance(digest, str)
                 and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid source hash mapping")
    _require(sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() == expected_sha,
             "source hash mapping aggregate mismatch")
    _require(hashes.get(CONFIG_KEY) == config.raw_hash, "source frozen config bytes mismatch")
    return {key: deepcopy(info[key]) for key in IDENTITY_FIELDS}


def _job(job, spec, source, command):
    _require(isinstance(job, dict) and set(job) == set(spec) | {"output", "command"},
             "benchmark job fields mismatch: " + spec["id"])
    for key, expected in spec.items():
        _require(job[key] == expected and isinstance(job[key], bool) == isinstance(expected, bool),
                 "benchmark protocol parameter mismatch: " + spec["id"] + "." + key)
    _require(isinstance(job["output"], str) and Path(job["output"]).is_absolute()
             and Path(job["output"]).resolve() == source, "benchmark output path mismatch")
    _require(isinstance(job["command"], str) and bool(job["command"].strip())
             and job["command"] == command, "benchmark original command mismatch")


def _measurements(row, spec):
    _require(all(_number(row.get(k), positive=True) for k in ("wall_seconds", "cpu_seconds", "peak_rss_bytes")),
             "benchmark requires positive wall/CPU/RSS measurements: " + spec["id"])
    _require(isinstance(row.get("peak_rss_source"), str) and row["peak_rss_source"] not in ("", "UNAVAILABLE"),
             "missing RSS measurement scope")
    _require(row.get("simulated_seconds") == spec["duration"], "benchmark simulated duration mismatch")
    ratio = row.get("sim_seconds_per_wall_second")
    _require(_number(ratio, positive=True) and math.isclose(ratio, spec["duration"] / row["wall_seconds"], rel_tol=1e-12),
             "benchmark throughput field mismatch")
    components = row.get("measured_components", {})
    exclusive = components.get("exclusive_cpu_seconds", {})
    _require(set(exclusive) == COMPONENTS and all(_number(v) for v in exclusive.values())
             and sum(exclusive.values()) > 0, "invalid measured exclusive component timings")
    _require(isinstance(components.get("classification"), str) and bool(components["classification"]),
             "missing component timing classification")
    _require(isinstance(components.get("functions"), list), "missing measured function timings")
    viewer = components.get("viewer", {})
    _require(viewer.get("enabled") is False and _number(viewer.get("cpu_seconds"))
             and viewer["cpu_seconds"] == 0, "benchmark measurement requires disabled viewer and zero viewer CPU")
    for function in components["functions"]:
        _require(all(_number(function.get(k)) for k in ("exclusive_cpu_seconds", "inclusive_cpu_seconds")),
                 "invalid measured function timings")
    instrumentation = row.get("measurement_instrumentation", {})
    _require(instrumentation.get("profiler") == "cProfile" and instrumentation.get("clock") == "process_time"
             and instrumentation.get("profiler_enabled_during_run") is True
             and instrumentation.get("wall_and_cpu_include_profiler_overhead") is True,
             "missing original measurement instrumentation")


def validate_source(source, *, expected_code_commit, expected_code_sha256):
    """Read only the benchmark subset; no 61-job terminal state is required."""
    source = Path(source).resolve()
    _require(re.fullmatch(r"[0-9a-f]{40}", expected_code_commit) is not None, "expected commit must be a full SHA")
    _require(re.fullmatch(r"[0-9a-f]{64}", expected_code_sha256) is not None, "expected code SHA256 invalid")
    config = load_config()
    jobs_path = source / "jobs.json"
    jobs_hash = _hash(jobs_path)
    jobs = _read(jobs_path)
    _require(isinstance(jobs, list) and all(isinstance(j, dict) for j in jobs), "jobs.json must contain job objects")
    specs = benchmark_specs()
    expected_ids = [s["id"] for s in specs]
    selected = [j for j in jobs if j.get("kind") == "benchmark" or str(j.get("id", "")).startswith("benchmark_")]
    _require(all(isinstance(j.get("id"), str) for j in selected), "missing benchmark ID")
    coverage = task_coverage(expected_ids, selected)
    _require(len(selected) == 40 and not any(coverage[k] for k in ("missing", "unexpected", "duplicates")),
             "benchmark jobs coverage mismatch")
    # Do not read incomplete non-benchmark jobs while they are still running.
    unexpected = {p.parent.name for p in source.glob("benchmark_*/result.json")} - set(expected_ids)
    scheduled_ids = {j.get("id") for j in jobs if isinstance(j.get("id"), str)}
    for path in source.glob("*/result.json"):
        if path.parent.name not in scheduled_ids:
            extra = _read(path)
            if isinstance(extra, dict) and extra.get("job", {}).get("kind") == "benchmark":
                unexpected.add(path.parent.name)
    _require(not unexpected, "unexpected benchmark result IDs: " + str(sorted(unexpected)))
    by_id = {j["id"]: j for j in selected}
    command = selected[0].get("command")
    rows, records, measured_identity = [], [], None
    for spec in specs:
        job = by_id[spec["id"]]
        _job(job, spec, source, command)
        result_path = source / spec["id"] / "result.json"
        result_hash = _hash(result_path)
        row = _read(result_path)
        _require(isinstance(row, dict) and row.get("id") == spec["id"]
                 and row.get("status") == "COMPLETED", "benchmark result must be COMPLETED: " + spec["id"])
        _require(row.get("job") == job, "jobs/result benchmark parameters mismatch")
        identity = _identity(row.get("provenance"), expected_commit=expected_code_commit,
                             expected_sha=expected_code_sha256, config=config)
        if measured_identity is None:
            measured_identity = identity
        _require(identity == measured_identity, "mixed source identity/hash mapping")
        _require(row["provenance"].get("command") == command, "benchmark provenance command mismatch")
        _measurements(row, spec)
        inflight = spec["mode"] == "inflight_attempt"
        airborne = spec["count"] + int(inflight)
        _require(row.get("actual_airborne_count") == airborne and row.get("inflight_admitted") is inflight,
                 "benchmark actual airborne/inflight mismatch: " + spec["id"])
        replay = source / spec["id"] / "replay"
        for key, path in (("replay_path", replay), ("events_path", replay / "events.jsonl"),
                          ("manifest_path", replay / "manifest.json")):
            _require(isinstance(row.get(key), str) and Path(row[key]).is_absolute()
                     and Path(row[key]).resolve() == path.resolve(), "benchmark archive path mismatch: " + key)
        _require(replay.resolve().is_relative_to(source), "benchmark replay escapes source")
        archive = ReplayBundle.validate(replay)
        index = archive.index
        _require(index.get("code_commit") == expected_code_commit and index.get("config_hash") == config.config_hash
                 and index.get("config_bytes_sha256") == config.raw_hash and index.get("command") == command
                 and index.get("seed") == 11, "benchmark archive provenance mismatch")
        metrics = _read(replay / "metrics.json")
        _require(row.get("metrics") == metrics and metrics.get("conservation") is True
                 and metrics.get("physical_airborne_count") == airborne, "benchmark archive metrics mismatch")
        _require(row.get("state_count") == archive.state_count and archive.time_bounds == (0., spec["duration"]),
                 "benchmark recorded states/duration mismatch")
        last_frame = None
        with (replay / "states.jsonl").open("rb") as stream:
            for line in stream:
                if line.strip():
                    last_frame = json.loads(line)
        aircraft = last_frame.get("aircraft", [])
        ids = [a.get("id") for a in aircraft]
        _require(len(ids) == len(set(ids)) and sum(a.get("status") in AIRBORNE for a in aircraft) == airborne,
                 "benchmark recorded aircraft count mismatch")
        if inflight:
            _require(any(a.get("id") == "inflight" and a.get("admitted") is True
                         and a.get("status") in AIRBORNE for a in aircraft), "benchmark recorded inflight admission mismatch")
        if inflight:
            events = [_read_line(line) for line in (replay / "events.jsonl").read_bytes().splitlines() if line.strip()]
            _require(any(event.get("type") == "admitted" and event.get("aircraft_id") == "inflight" for event in events),
                     "benchmark replay lacks inflight admission event")
        records.append({"id": spec["id"], "result_path": str(result_path), "result_sha256": result_hash,
                        "replay_path": str(archive.path), "replay_index_sha256": _hash(replay / "replay.json"),
                        "state_count": archive.state_count})
        rows.append(row)
    _require(_hash(jobs_path) == jobs_hash, "source jobs changed during validation")
    for record in records:
        _require(_hash(record["result_path"]) == record["result_sha256"], "source result changed during validation")
    coverage = task_coverage(expected_ids, rows)
    return {"source_directory": str(source), "jobs_path": str(jobs_path), "jobs_sha256": jobs_hash,
            "coverage": coverage, "measurement_identity": measured_identity,
            "records": records, "jobs": rows, "benchmark_summary": summarize(rows)}


def _read_line(line):
    return json.loads(line)


def _summary(values):
    values = sorted(values)
    return {"median": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1]}


def summarize(rows):
    groups = []
    for n in (10, 30, 60, 83):
        for mode in ("stable", "inflight_attempt"):
            values = [r for r in rows if (r["job"]["count"], r["job"]["mode"]) == (n, mode)]
            groups.append({"initial_stable_count": n, "mode": mode, "repeats": len(values),
                "p95_definition": "nearest_rank (five repeats: maximum)",
                "wall_seconds": _summary([r["wall_seconds"] for r in values]),
                "cpu_seconds": _summary([r["cpu_seconds"] for r in values]),
                "peak_rss_bytes": _summary([r["peak_rss_bytes"] for r in values]),
                "peak_rss_sources": sorted({r["peak_rss_source"] for r in values}),
                "exclusive_cpu_seconds": {key: _summary([r["measured_components"]["exclusive_cpu_seconds"][key]
                                                        for r in values]) for key in sorted(COMPONENTS)},
                "raw_measurements": [{key: deepcopy(r[key]) for key in (
                    "id", "wall_seconds", "cpu_seconds", "peak_rss_bytes", "peak_rss_source",
                    "measured_components", "measurement_instrumentation")} for r in values],
                "component_scope": "Original exclusive groups retained; inclusive function times are retained individually and never summed.",
                "absolute_performance_pass_threshold": None, "interpretation": LIMITATION})
    return groups


def qualify(source, output, *, expected_code_commit, expected_code_sha256, command):
    source, output = Path(source).resolve(), Path(output).resolve()
    _require(not output.exists(), "qualification output already exists")
    _require(not output.is_relative_to(source) and not source.is_relative_to(output), "output must be separate from source")
    started = time.perf_counter()
    consumed = validate_source(source, expected_code_commit=expected_code_commit, expected_code_sha256=expected_code_sha256)
    current = provenance(command)
    verifier = {"entry_point": str(Path(__file__).resolve()), "entry_point_sha256": _hash(__file__),
                "command": command, "environment": current,
                "commit_identity_scope": "Current environment's supplied/local commit label; server supplied labels are not Git-verified.",
                "current_source_matches_measured_source": current["code_sha256"] == expected_code_sha256,
                "original_catalogue_fixture_executed": False, "new_simulations_executed": 0}
    info = dict(consumed["measurement_identity"], command=command, measurement_identity_scope="consumed benchmarks",
                verifier=verifier)
    case_dir = output / "G09"
    evidence = Evidence("G09", case_dir, info)
    evidence.action("consume_40_existing_benchmark_archives", {"source": str(source), "limitations": LIMITATION,
        "new_simulations_executed": 0, "original_catalogue_fixture_executed": False})
    evidence.observe("existing_benchmark_validation", {k: v for k, v in consumed.items() if k != "jobs"})
    evidence.observe("qualification_scope", LIMITATION)
    evidence.check(True, "exact 40 completed source benchmarks and content-verified real replay bundles",
                   "N10/30/60/83 x stable/inflight_attempt x repeats0..4", consumed["coverage"])
    evidence.replay_paths = [record["replay_path"] for record in consumed["records"]]
    # Deliberately do not call save_replays: these are the real consumed archives.
    write_json(case_dir / "fixture.json", {"seed": 11, "actions": evidence.actions, "attempted_plans": [],
        "entry_point": "supplementary_existing_benchmark_verifier", "original_catalogue_fixture_executed": False})
    write_json(case_dir / "assertions.json", evidence.assertions)
    write_json(case_dir / "observations.json", evidence.observations)
    write_json(case_dir / "environment.json", info)
    (case_dir / "execution.log").write_text(json.dumps(evidence.actions, ensure_ascii=False) + "\nPASS: " + LIMITATION + "\n", encoding="utf8")
    normative = catalogue()
    case = next(c for c in normative if c["id"] == "G09")
    row = {**case, "status": "PASS", "reason": LIMITATION, "seed": 11,
        "elapsed_wall_seconds": time.perf_counter() - started, "actual": evidence.observations,
        "assertion_count": len(evidence.assertions), "replay_paths": evidence.replay_paths,
        **{key: info[key] for key in ("code_sha256", "config_hash", "code_commit")},
        "measurement_identity_scope": "consumed benchmarks", "verifier": verifier,
        "command": command, "researcher_visual_acceptance": "NOT_APPLICABLE"}
    for key, filename in (("assertions_path", "assertions.json"), ("log_path", "execution.log"),
                          ("fixture_path", "fixture.json"), ("environment_path", "environment.json")):
        row[key] = str(case_dir / filename)
    write_json(case_dir / "result.json", row)
    cases = [row if c["id"] == "G09" else dict(c, status="NOT_RUN", reason="not executed by this supplementary verifier")
             for c in normative]
    report = {"schema": "FT-RS-S1-1.0-qualification-results", "catalogue_is_results": False,
        "cases": cases, "coverage": task_coverage(["G09"], [row]), "counts": dict(Counter(c["status"] for c in cases)),
        "automatic_status": "IMPLEMENTATION_IN_PROGRESS", "visualization_integration_status": "NOT_COMPLETE",
        "researcher_visual_acceptance": "PENDING", "stage1_status": "NOT_ACCEPTED", "command": command,
        "qualification_entry_point": "supplementary_existing_benchmark_verifier", "verifier": verifier,
        "source_evidence": {k: v for k, v in consumed.items() if k != "jobs"}}
    write_json(output / "results.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    args = parser.parse_args(argv)
    command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()),
                                      *(sys.argv[1:] if argv is None else argv)])
    try:
        report = qualify(args.source, args.output, expected_code_commit=args.expected_code_commit,
                         expected_code_sha256=args.expected_code_sha256, command=command)
        print(json.dumps({"status": "PASS", "executed_case_ids": ["G09"], "coverage": report["coverage"],
                          "report": str(Path(args.output).resolve() / "results.json")}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "QUALIFICATION_NOT_COMPLETE", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
