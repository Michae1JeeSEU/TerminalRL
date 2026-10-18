"""Tiny hash-verified test archives; no measured benchmark or simulation runs."""
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path

import pytest

from scripts.server import qualify_rotating_slots_benchmarks as qualifier
from terminal_marl.core.rotating_slots import diagnostics
from terminal_marl.core.rotating_slots.qualification import write_json
from terminal_marl.core.rotating_slots.replay import ReplayBundle

COMMIT = "a" * 40
COMMAND = "synthetic unit-test fixture only; not research measurements"


def change(path, update):
    value = json.loads(path.read_bytes())
    update(value)
    write_json(path, value)


def hashes(root):
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def existing(tmp_path, monkeypatch):
    source = tmp_path / "existing"
    source.mkdir()
    config = qualifier.load_config()
    source_hashes = {qualifier.CONFIG_KEY: config.raw_hash, "synthetic_fixture_only.py": "b" * 64}
    code_sha = sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    info = dict(code_commit=COMMIT, code_sha256=code_sha, config_hash=config.config_hash,
                source_files_sha256=source_hashes, command=COMMAND)
    jobs = [dict(spec, output=str(source), command=COMMAND) for spec in qualifier.benchmark_specs()]
    # An unfinished non-benchmark job is deliberately present, without result.
    write_json(source / "jobs.json", [dict(id="traffic_still_running", kind="traffic"), *jobs])
    for job in jobs:
        root = source / job["id"]
        inflight = job["mode"] == "inflight_attempt"
        n = job["count"] + int(inflight)
        aircraft = [dict(id=f"hold_{i}", status="HOLD_INNER", admitted=True) for i in range(job["count"])]
        if inflight:
            aircraft.append(dict(id="inflight", status="INBOUND", admitted=True))
        metrics = dict(conservation=True, physical_airborne_count=n, synthetic_test_fixture=True)
        record = dict(states=[dict(time=t, aircraft=aircraft) for t in (0., 1.)], metrics=metrics,
            manifest=dict(seed=11, requests=[]), trajectories={},
            events=[dict(time=0., type="admitted", aircraft_id="inflight")] if inflight else [])
        archive = ReplayBundle.write(root / "replay", record=record, eager=False,
                                     code_commit=COMMIT, command=COMMAND, seed=11)
        components = dict(exclusive_cpu_seconds={key: .1 for key in qualifier.COMPONENTS},
            classification="synthetic mutually exclusive test groups", functions=[dict(function="test_only",
                exclusive_cpu_seconds=.1, inclusive_cpu_seconds=100.)],
            viewer=dict(enabled=False, cpu_seconds=0.))
        row = dict(id=job["id"], job=job, status="COMPLETED", provenance=deepcopy(info),
            metrics=metrics, actual_airborne_count=n, inflight_admitted=inflight,
            wall_seconds=2. + job["repeat"], cpu_seconds=1. + job["repeat"], peak_rss_bytes=1024 + job["repeat"],
            peak_rss_source="synthetic_process_lifetime_fixture", simulated_seconds=1.,
            sim_seconds_per_wall_second=1. / (2. + job["repeat"]), measured_components=components,
            measurement_instrumentation=dict(profiler="cProfile", clock="process_time",
                profiler_enabled_during_run=True, wall_and_cpu_include_profiler_overhead=True),
            replay_path=str(archive.path), events_path=str(archive.path / "events.jsonl"),
            manifest_path=str(archive.path / "manifest.json"), state_count=2)
        write_json(root / "result.json", row)
    calls = []
    original = ReplayBundle.validate.__func__

    def validate(cls, path):
        calls.append(str(Path(path).resolve()))
        return original(cls, path)

    monkeypatch.setattr(ReplayBundle, "validate", classmethod(validate))
    # Every available simulation/benchmark entry raises if accidentally reached.
    for name in ("run_diagnostics", "run_benchmarks", "execute_jobs", "_diagnostic_job", "_new_world"):
        monkeypatch.setattr(diagnostics, name, lambda *a, **kw: pytest.fail("simulation entry was called"))
    return source, code_sha, calls


def invoke(existing, output):
    source, code_sha, _ = existing
    return qualifier.qualify(source, output, expected_code_commit=COMMIT,
                              expected_code_sha256=code_sha, command="pytest independent verifier")


def test_pass_consumes_exact_40_without_other_terminal_jobs_or_source_edits(existing, tmp_path):
    source, code_sha, calls = existing
    before = hashes(source)
    output = tmp_path / "qualification"
    report = invoke(existing, output)
    row = next(r for r in report["cases"] if r["id"] == "G09")
    assert len(calls) == 40 and set(calls) == set(row["replay_paths"])
    assert hashes(source) == before and not (source / "diagnostics.json").exists()
    assert len([r for r in report["cases"] if r["status"] != "NOT_RUN"]) == 1
    assert row["status"] == "PASS" and row["code_commit"] == COMMIT and row["code_sha256"] == code_sha
    assert report["coverage"]["actual_count"] == 1
    assert report["source_evidence"]["coverage"]["actual_count"] == 40
    assert row["verifier"]["current_source_matches_measured_source"] is False
    assert row["verifier"]["original_catalogue_fixture_executed"] is False
    assert row["verifier"]["new_simulations_executed"] == 0
    assert row["verifier"]["entry_point_sha256"] == qualifier._hash(qualifier.__file__)
    assert "newer simulator" in row["reason"] and "no absolute throughput" in row["reason"]
    assert not list(output.rglob("analytic_replay"))
    assert not (output / "diagnostics.json").exists()
    groups = report["source_evidence"]["benchmark_summary"]
    assert len(groups) == 8 and all(g["repeats"] == 5 for g in groups)
    for group in groups:
        assert group["wall_seconds"] == dict(median=4., p95=6.)
        assert group["cpu_seconds"] == dict(median=3., p95=5.)
        assert group["peak_rss_bytes"] == dict(median=1026, p95=1028)
        assert group["exclusive_cpu_seconds"]["generation"] == dict(median=.1, p95=.1)
        assert group["raw_measurements"][0]["measured_components"]["functions"][0]["inclusive_cpu_seconds"] == 100.
        assert group["absolute_performance_pass_threshold"] is None
    assert json.loads((output / "results.json").read_bytes()) == report
    assert "FROZEN_FORMAL_MATRIX" not in inspect.getsource(qualifier)
    for entry in ("_diagnostic_job(", "run_benchmarks(", "run_diagnostics(", "execute_case("):
        assert entry not in inspect.getsource(qualifier)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected", "missing_result", "extra_result"])
def test_missing_duplicate_or_unexpected_benchmark_is_rejected(existing, tmp_path, mutation):
    source, _, _ = existing
    if mutation == "missing_result":
        (source / "benchmark_n10_stable_r0/result.json").unlink()
    elif mutation == "extra_result":
        write_json(source / "benchmark_surprise/result.json", {})
    else:
        def update(jobs):
            if mutation == "missing": jobs.pop()
            elif mutation == "duplicate": jobs.append(deepcopy(jobs[1]))
            else: jobs[-1]["id"] = "benchmark_unexpected"
        change(source / "jobs.json", update)
    with pytest.raises((ValueError, FileNotFoundError)):
        invoke(existing, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("field", ["code_commit", "code_sha256", "config_hash", "source_files_sha256"])
def test_mixed_or_tampered_source_provenance_is_rejected(existing, tmp_path, field):
    source, _, _ = existing
    change(source / "benchmark_n10_stable_r1/result.json",
           lambda row: row["provenance"].__setitem__(field, {} if field == "source_files_sha256" else "wrong"))
    with pytest.raises(ValueError):
        invoke(existing, tmp_path / "refused")


def test_source_mapping_aggregate_is_recomputed(existing, tmp_path):
    source, _, _ = existing
    change(source / "benchmark_n10_stable_r0/result.json", lambda row:
           row["provenance"]["source_files_sha256"].__setitem__("synthetic_fixture_only.py", "c" * 64))
    with pytest.raises(ValueError, match="aggregate mismatch"):
        invoke(existing, tmp_path / "refused")


@pytest.mark.parametrize("mutation", ["archive", "metrics", "index", "airborne", "inflight", "wall", "cpu", "component", "status", "protocol"])
def test_archive_and_measurement_tampering_is_rejected(existing, tmp_path, mutation):
    source, _, _ = existing
    root = source / "benchmark_n10_inflight_attempt_r0"
    if mutation == "archive":
        (root / "replay/states.jsonl").write_bytes(b'{"time":0}\n')
    elif mutation == "index":
        change(root / "replay/replay.json", lambda index: index.__setitem__("code_commit", "wrong"))
    elif mutation == "protocol":
        change(source / "jobs.json", lambda jobs: next(j for j in jobs if j["id"] == root.name).__setitem__("duration", .2))
    else:
        def update(row):
            if mutation == "metrics": row["metrics"]["physical_airborne_count"] = 999
            elif mutation == "airborne": row["actual_airborne_count"] = 10
            elif mutation == "inflight": row["inflight_admitted"] = False
            elif mutation == "wall": row["wall_seconds"] = 0
            elif mutation == "cpu": row["cpu_seconds"] = -1
            elif mutation == "component": row["measured_components"]["exclusive_cpu_seconds"]["generation"] = -1
            elif mutation == "status": row["status"] = "BLOCKED"
        change(root / "result.json", update)
    with pytest.raises(ValueError):
        invoke(existing, tmp_path / "refused")


def test_cli_requires_explicit_matching_identity_and_fresh_output(existing, tmp_path, capsys):
    source, code_sha, _ = existing
    args = ["--source", str(source), "--output", str(tmp_path / "cli"),
            "--expected-code-commit", COMMIT, "--expected-code-sha256", code_sha]
    assert qualifier.main(args) == 0
    assert json.loads(capsys.readouterr().out)["executed_case_ids"] == ["G09"]
    assert qualifier.main(args) == 1
    assert "already exists" in capsys.readouterr().err
    args[3] = str(tmp_path / "mismatch")
    args[-1] = "d" * 64
    assert qualifier.main(args) == 1
    assert "source code_sha256 mismatch" in capsys.readouterr().err
    assert not (tmp_path / "mismatch").exists()


@pytest.mark.parametrize("viewer,passes", [
    (dict(enabled=False, cpu_seconds=0.), True),
    (dict(enabled=True, cpu_seconds=0.), False),
    (dict(enabled=False, cpu_seconds=.1), False),
    (dict(enabled=False, cpu_seconds=False), False),
    ({}, False),
])
def test_measurement_requires_disabled_viewer_and_zero_viewer_cpu(viewer, passes):
    row = dict(wall_seconds=2., cpu_seconds=1., peak_rss_bytes=1024,
        peak_rss_source="synthetic_fixture", simulated_seconds=1., sim_seconds_per_wall_second=.5,
        measured_components=dict(exclusive_cpu_seconds={key: .1 for key in qualifier.COMPONENTS},
            classification="fixture", functions=[], viewer=viewer),
        measurement_instrumentation=dict(profiler="cProfile", clock="process_time",
            profiler_enabled_during_run=True, wall_and_cpu_include_profiler_overhead=True))
    if passes:
        qualifier._measurements(row, qualifier.benchmark_specs()[0])
    else:
        with pytest.raises(ValueError, match="disabled viewer and zero viewer CPU"):
            qualifier._measurements(row, qualifier.benchmark_specs()[0])
