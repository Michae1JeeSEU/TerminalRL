import json
import math
import subprocess
import sys

import pytest

import terminal_marl.analysis.full_service_capacity as harness
from terminal_marl.analysis.full_service_capacity import (
    cohort_metrics, design_metadata, digest, generate_manifest, make_config,
    prepare_jobs, profiles, run_job, soc_from_uniform, task_from_uniform, validate_spec,
)


def test_manifest_pairing_reproducibility_and_profile_mapping():
    base = generate_manifest(123, 240., 3600., "half6")
    assert base == generate_manifest(123, 240., 3600., "half6")
    assert base != generate_manifest(124, 240., 3600., "half6")
    for name in profiles():
        other = generate_manifest(123, 240., 3600., name)
        assert other["latent_marks"] == base["latent_marks"]
        assert other["latent_marks_hash"] == base["latent_marks_hash"]
        for request, mark in zip(other["manifest"]["requests"], base["latent_marks"]):
            assert request["initial_soc"] == soc_from_uniform(mark["soc_uniform"], profiles()[name])
            assert request["service_class"] == task_from_uniform(mark["task_uniform"], profiles()[name])
    assert task_from_uniform(.6, profiles()["high_variance66"]) == "long"
    assert soc_from_uniform(.5, profiles()["low_soc"]) == pytest.approx(.6)
    assert all(r["release_time"] < 3600 for r in base["manifest"]["requests"])


def test_single_manifest_and_validation():
    single = generate_manifest(12, 120, 50, "half6", "single")["manifest"]["requests"]
    assert len(single) == 1
    assert single[0]["release_time"] == 0
    assert single[0]["initial_soc"] == .85
    assert single[0]["service_class"] == "standard"
    with pytest.raises(ValueError, match="warmup"):
        validate_spec(dict(demand_kind="single", warmup_s=1))
    for bad in (dict(methods=["learned"]), dict(rates_per_hour=[0]), dict(seeds=[True]), dict(measurement_s=0), dict(drain_s=math.inf)):
        with pytest.raises(ValueError):
            validate_spec(bad)


def test_cohort_accounting_pending_reroute_failure_and_touchdown_soc():
    requests = [dict(id=str(i), release_time=t, service_class="short" if i % 2 else "standard")
                for i, t in enumerate((0., 10., 20., 30., 40., 50., 60., 100.))]
    aircraft = {
        "0": dict(touchdown_time=5., status="CLEARED", soc=.1),
        "1": dict(touchdown_time=25., status="CLEARED", soc=.1),
        "2": dict(touchdown_time=45., status="CLEARED", soc=.9),
        "3": dict(touchdown_time=None, status="REROUTED"),
        "5": dict(touchdown_time=None, status="HOLD_OUTER"),
        "6": dict(touchdown_time=None, status="ENERGY_FAILURE"),
    }
    events = [dict(type="touchdown", aircraft_id=key, details=dict(soc=soc, emergency=emergency))
              for key, soc, emergency in (("0", .5, False), ("1", .3, False), ("2", .4, True))]
    result = cohort_metrics(requests, aircraft, events, start_s=10, end_s=100, sla_delay_s=20)
    assert result["offered"] == 6
    assert [result[k] for k in ("normal_touchdown", "emergency", "failure", "rerouted", "upstream_pending", "airborne_pending")] == [1] * 6
    assert result["normal_within_sla_and_reserve_ge_025"] == 1
    assert result["normal_ontime_fraction_all_offered"] == 1 / 6
    assert result["normal_delay_quantiles_s"]["p50"] == 15
    assert result["population_conservation"]
    assert result["per_class"]["standard"]["offered"] == 3
    assert result["per_class"]["short"]["offered"] == 3
    for field in ("offered", "normal_touchdown", "normal_within_sla", "normal_reserve_ge_025",
                  "normal_within_sla_and_reserve_ge_025", "emergency", "failure", "rerouted", "upstream_pending", "airborne_pending"):
        assert sum(row[field] for row in result["per_class"].values()) == result[field]


def test_design_bound_geometry_and_moments():
    for name, inner_k in (("radius250", 15), ("half6", 18), ("radius350", 21)):
        metadata = design_metadata(profiles()[name])
        inner = next(r for r in metadata["rings"] if r["id"] == "inner")
        assert inner["design_K"] == inner_k
        assert inner["speed_mps"] == pytest.approx(inner["radius_m"] * inner["angular_speed_radps"])
        assert inner["stable_holding_power_w"] > 0
        assert metadata["service_mean_s"] == 66
        assert metadata["protected_descent_s"] == 13.5
        assert metadata["pad_workload_necessary_upper_per_h"] == 3600 * 6 / 79.5
    assert design_metadata(profiles()["deterministic66"])["service_variance_s2"] == 0
    assert design_metadata(profiles()["high_variance66"])["service_variance_s2"] > design_metadata(profiles()["half6"])["service_variance_s2"]


def test_empty_short_world_output_and_completed_only_resume(tmp_path):
    spec = dict(profiles=["half6"], methods=["energy_slack"], seeds=[11], rates_per_hour=[.000001],
                warmup_s=0., measurement_s=.01, drain_s=0.)
    jobs = prepare_jobs(spec, tmp_path)
    assert len(jobs) == 1
    report = run_job(jobs[0])
    assert report["status"] == "COMPLETED"
    directory = tmp_path / "jobs/half6_rate1e-06_seed11_energy_slack"
    result = json.loads((directory / "results.json").read_bytes())
    assert result["measurement_cohort"]["offered"] == 0
    assert result["safety"]["conservation"]
    assert result["callback_count"] == 1
    assert len(result["boundaries"]) == 3
    assert result["observation_window"]["pad_busy_utilization"] == 0
    assert (directory / "decisions.jsonl").exists()
    timeseries = [json.loads(line) for line in (directory / "timeseries.jsonl").read_bytes().splitlines()]
    assert [row["sim_time_s"] for row in timeseries] == [0., .01]
    assert all(row["upstream"] == row["airborne"] == row["pad_busy_count"] == 0 for row in timeseries)
    progress = json.loads((directory / "progress.json").read_bytes())
    assert progress["latest_callback_wall_s"] == result["latest_callback_wall_s"]
    assert 0 <= progress["latest_callback_wall_s"] < result["wall_s"]
    assert json.loads((directory / "aircraft.json").read_bytes()) == {}
    assert run_job(jobs[0])["skipped"] is True
    changed = dict(spec, measurement_s=.02)
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_jobs(changed, tmp_path)
    config = json.loads((directory / "config.json").read_bytes())
    config["pads"]["count"] = 3
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="stored config identity mismatch"):
        run_job(jobs[0])
    with pytest.raises(ValueError, match="config.json identity mismatch"):
        prepare_jobs(spec, tmp_path)


def test_cli_help_does_not_require_solver():
    result = subprocess.run([sys.executable, "scripts/analysis/run_full_service_capacity.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--workers" in result.stdout


def test_solver_failure_is_explicit_and_retried_not_completed(tmp_path, monkeypatch):
    jobs = prepare_jobs(dict(profiles=["half6"], methods=["rh_cp_sat"],
        rates_per_hour=[.000001], measurement_s=.01, drain_s=0.), tmp_path)
    calls = []
    def unavailable():
        calls.append(1)
        raise RuntimeError("deliberate solver preflight failure")
    monkeypatch.setattr(harness, "solver_preflight", unavailable)
    assert run_job(jobs[0])["status"] == "FAILED"
    assert run_job(jobs[0])["status"] == "FAILED"
    assert len(calls) == 2
    result = json.loads((tmp_path / "jobs/half6_rate1e-06_seed11_rh_cp_sat/results.json").read_bytes())
    assert result["exception_type"] == "RuntimeError"
    assert "deliberate solver preflight failure" in result["traceback"]
    assert "measurement_cohort" not in result
    assert result["params"]["method"] == "rh_cp_sat"


def test_spawn_empty_jobs_preserve_coverage_and_identity(tmp_path):
    spec = dict(profiles=["half6"], methods=["energy_slack"], seeds=[11, 23],
                rates_per_hour=[.000001], warmup_s=0., measurement_s=.01, drain_s=0.)
    cpus = harness.available_cpus()
    rows = harness.run_campaign(spec, tmp_path, workers=min(2, len(cpus)))
    assert len(rows) == 2
    assert all(row["status"] == "COMPLETED" for row in rows)
    summary = json.loads((tmp_path / "summary.json").read_bytes())
    assert summary["expected_jobs"] == summary["finished_jobs"] == 2
    assert summary["all_completed"]
    for row in rows:
        from pathlib import Path
        directory = Path(row["directory"])
        identity = json.loads((directory / "identity.json").read_bytes())
        result = json.loads((directory / "results.json").read_bytes())
        assert result["identity_hash"] == digest(identity)
        assert result["assigned_cpu"] is None
        assert result["measurement_cohort"]["offered"] == 0
