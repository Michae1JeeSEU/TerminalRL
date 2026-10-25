"""Synthetic event/output fixtures; these tests do not run new experiments."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from terminal_marl.analysis.full_capacity_report import (
    aggregate_rows, analyze_campaign, analyze_job, holding_energy_audit,
    paired_differences, plot_report, write_report,
)
from terminal_marl.analysis.full_service_capacity import (
    atomic_json, cohort_metrics, digest, make_config, prepare_jobs, profiles,
)


def event(kind, time, aircraft="req_000000", **details):
    return dict(type=kind, time=time, aircraft_id=aircraft, details=details)


def chain(aircraft="req_000000"):
    plans = {}
    events = []
    for key, kind, start, end, ring in (
        ("in", "INBOUND", 0., 20., "outer"),
        ("om", "TRANSFER", 30., 50., "middle"),
        ("mi", "TRANSFER", 60., 80., "inner"),
        ("fin", "FINAL_NORMAL", 90., 120., None),
    ):
        plans[key] = dict(aircraft_id=aircraft, trajectory_hash=key, kind=kind,
                          start_time=start, end_time=end, target_ring=ring)
        events.append(event("admitted" if kind == "INBOUND" else "maneuver_started", start, aircraft, trajectory_hash=key))
        if ring is not None:
            events.append(event("inbound_completed" if kind == "INBOUND" else "transfer_completed", end, aircraft))
    # This path was planned, never started, and must not create residence.
    plans["unused"] = dict(aircraft_id=aircraft, trajectory_hash="unused", kind="TRANSFER",
                            start_time=21., end_time=22., target_ring="inner")
    return plans, events


def test_executed_holding_chain_ignores_unexecuted_plan():
    config = make_config(profiles()["half6"])
    capacity = config.energy.reference_capacity_Ah * config.energy.equivalent_voltage_V * 3600.
    plans, events = chain("a")
    state = dict(admitted=True, admission_time=0., status="CLEARED", soc=.8, holding_time=30., energy_j=capacity * .05)
    audit = holding_energy_audit({"a": state}, events, plans,
        [dict(id="a", initial_soc=.85)], config, final_time_s=200.)
    assert audit["ring_holding_s"] == dict(outer=10., middle=10., inner=10.)
    assert audit["holding_time_max_abs_residual_s"] == 0.
    row = audit["per_aircraft"][0]
    assert row["derived_move_energy_j"] == pytest.approx(state["energy_j"] - row["stable_holding_energy_j"])
    assert row["battery_identity_residual_j"] == pytest.approx(0., abs=1e-8)


def test_admitted_inbound_requires_matching_admission_time():
    config = make_config(profiles()["half6"])
    plan = dict(aircraft_id="a", trajectory_hash="in", kind="INBOUND", start_time=3., end_time=23., target_ring="outer")
    events = [event("admitted", 3., "a", trajectory_hash="in"), event("inbound_completed", 23., "a")]
    state = dict(admitted=True, admission_time=3., status="HOLD_OUTER", soc=.7, holding_time=7., energy_j=100.)
    audit = holding_energy_audit({"a": state}, events, {"in": plan}, [dict(id="a", initial_soc=.8)],
                                 config, final_time_s=30.)
    assert audit["ring_holding_s"]["outer"] == 7.
    assert audit["holding_time_max_abs_residual_s"] == 0.
    state["admission_time"] = 4.
    omitted = holding_energy_audit({"a": state}, events, {"in": plan}, [dict(id="a", initial_soc=.8)],
                                   config, final_time_s=30.)
    assert omitted["omitted_aircraft"] == 1


def test_holding_failure_uses_archive_time_and_event_fallback():
    config = make_config(profiles()["half6"])
    plan = dict(aircraft_id="a", trajectory_hash="in", kind="INBOUND", start_time=0., end_time=20., target_ring="outer")
    events = [event("maneuver_started", 0., "a", trajectory_hash="in"),
              event("inbound_completed", 20., "a"), event("aircraft_energy_failure", 40., "a")]
    state = dict(admitted=True, status="ENERGY_MISSION_FAILED", soc=0., holding_time=20., energy_j=100.)
    inputs = ({"a": state}, events, {"in": plan}, [dict(id="a", initial_soc=.6)], config)
    for failures in ({"a": {"time": 40.}}, {}):
        audit = holding_energy_audit(*inputs, final_time_s=200., failures=failures)
        assert audit["ring_holding_s"]["outer"] == 20.
        assert audit["per_aircraft"][0]["terminal_time_s"] == 40.
    omitted = holding_energy_audit({"a": state}, events[:-1], {"in": plan},
        [dict(id="a", initial_soc=.6)], config, final_time_s=200.)
    assert omitted["omitted_aircraft"] == 1
    assert "failure timestamp" in " ".join(omitted["per_aircraft"][0]["reasons"])


def test_missing_started_trajectory_omits_audit():
    config = make_config(profiles()["half6"])
    audit = holding_energy_audit({"a": dict(status="HOLD_OUTER", soc=.7, holding_time=10.)},
        [event("maneuver_started", 0., "a", trajectory_hash="absent")], {},
        [dict(id="a", initial_soc=.8)], config, final_time_s=20.)
    assert audit["omitted_aircraft"] == 1
    assert audit["audited_aircraft"] == 0


def synthetic_completed(job):
    """Serialize a small, explicitly synthetic completed-output fixture."""
    directory = Path(job["directory"])
    identity = job["identity"]
    params = identity["job"]
    config = make_config(profiles()[params["profile"]])
    request = json.loads((directory / "manifest.json").read_bytes())["manifest"]["requests"][0]
    capacity = config.energy.reference_capacity_Ah * config.energy.equivalent_voltage_V * 3600.
    state = dict(admitted=True, admission_time=0., status="CLEARED", soc=.8, touchdown_time=120.,
                 holding_time=30., energy_j=capacity * (request["initial_soc"] - .8))
    plans, events = chain()
    events.append(event("touchdown", 120., soc=.8, emergency=False))
    aircraft = {"req_000000": state}
    measurement = cohort_metrics([request], aircraft, events, start_s=0., end_s=100., sla_delay_s=130.)
    result = dict(status="COMPLETED", identity_hash=digest(identity), params=params,
        measurement_cohort=measurement, safety=dict(failures={}, conservation=True),
        boundaries=[dict(label="measurement_start", sim_time_s=0., upstream=0, airborne=0),
                    dict(label="measurement_end", sim_time_s=100., upstream=0, airborne=1),
                    dict(label="final", sim_time_s=200., upstream=0, airborne=0)],
        observation_window=dict(touchdown_rate_per_h=0., normal_touchdown_rate_per_h=0., pad_busy_utilization=.1),
        energy_identity=[dict(residual_j=0.)], wall_s=1., cpu_s=.8)
    atomic_json(directory / "results.json", result)
    atomic_json(directory / "aircraft.json", aircraft)
    atomic_json(directory / "trajectories.json", plans)
    (directory / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events), encoding="utf-8")
    return result


@pytest.fixture
def campaign(tmp_path):
    spec = dict(profiles=["half6"], methods=["energy_slack", "rh_cp_sat"], seeds=[11],
                rates_per_hour=[120.], demand_kind="single", warmup_s=0., measurement_s=100.,
                drain_s=100., sla_delay_s=130.)
    jobs = prepare_jobs(spec, tmp_path / "campaign")
    for job in jobs:
        synthetic_completed(job)
    atomic_json(tmp_path / "campaign/summary.json", dict(expected_jobs=2, finished_jobs=2, all_completed=True,
        jobs=[dict(directory=job["directory"], status="COMPLETED") for job in jobs]))
    return tmp_path / "campaign", jobs


def test_job_bound_denominators_and_cohort_audit(campaign):
    _, jobs = campaign
    report = analyze_job(jobs[0]["directory"])
    row = report["row"]
    assert row["offered"] == 1
    assert row["demand_kind"] == "single"
    assert row["normal_fraction_all_offered"] == 1.
    assert row["ontime_reserve_fraction_all_offered"] == 1.
    assert row["cohort_normal_touchdowns_per_h"] == 36.
    assert row["observation_normal_touchdowns_per_h"] == 0.
    assert row["finite_integer_count_upper"] == 1
    assert row["finite_bound_observed_count_le_upper"] is True
    assert row["finite_bound_violation"] is False
    assert row["loss_0.00_offered_rate_upper_per_h"] == pytest.approx(271.6981132075472)
    assert row["loss_0.05_offered_rate_upper_per_h"] == pytest.approx(296.60144181256436)
    assert row["stored_cohort_matches"] is True
    assert report["mechanism_audit"]["audited_aircraft"] == 1


def test_complete_campaign_outputs_and_paired_order(campaign, tmp_path):
    directory, _ = campaign
    report = analyze_campaign(directory)
    assert report["coverage"]["all_completed"] is True
    assert report["coverage"]["complete_jobs"] == 2
    pair = report["paired"]["method_cp_minus_es"][0]
    assert pair["manifest_hash_matches"] is True
    assert pair["config_hash_matches"] is True
    assert pair["normal_fraction_all_offered_delta"] == 0.
    output = tmp_path / "report"
    write_report(report, output)
    assert json.loads((output / "report.json").read_bytes())["coverage"]["all_completed"] is True
    assert (output / "per_job.csv").read_text(encoding="utf-8").count("\n") == 3
    assert (output / "paired_methods.csv").exists()
    assert json.loads((output / "per_job.json").read_bytes()) == report["per_job"]


def test_partial_and_failed_campaign_never_marked_complete(campaign):
    directory, jobs = campaign
    (Path(jobs[1]["directory"]) / "results.json").unlink()
    partial = analyze_campaign(directory)
    assert partial["coverage"]["complete_jobs"] == 1
    assert partial["coverage"]["incomplete_jobs"] == 1
    assert partial["coverage"]["all_completed"] is False
    assert partial["paired"]["method_cp_minus_es"] == []
    atomic_json(Path(jobs[1]["directory"]) / "results.json", dict(status="FAILED", exception="synthetic failure"))
    failed = analyze_campaign(directory)
    assert failed["coverage"]["failed_jobs"] == 1
    assert failed["coverage"]["all_completed"] is False


def test_missing_duplicate_and_hash_mismatch_are_reported(campaign):
    directory, jobs = campaign
    duplicate = directory / "jobs/duplicate"
    duplicate.mkdir()
    atomic_json(duplicate / "identity.json", jobs[0]["identity"])
    report = analyze_campaign(directory)
    assert report["coverage"]["duplicate_jobs"] == 1
    assert report["coverage"]["complete_jobs"] == 1
    assert report["coverage"]["all_completed"] is False
    (duplicate / "identity.json").unlink()
    duplicate.rmdir()
    config_path = Path(jobs[0]["directory"]) / "config.json"
    config = json.loads(config_path.read_bytes())
    config["pads"]["count"] = 99
    atomic_json(config_path, config)
    invalid = analyze_campaign(directory)
    assert invalid["coverage"]["failed_jobs"] == 1
    assert "config" in invalid["issues"]["failed"][0]["error"]
    # Rename a directory outside jobs so that the expected identity is absent.
    Path(jobs[1]["directory"]).rename(directory / "removed_job")
    missing = analyze_campaign(directory)
    assert missing["coverage"]["missing_jobs"] == 1


def test_mean_ranges_and_pair_differences_require_matching_inputs(campaign):
    _, jobs = campaign
    base = analyze_job(jobs[0]["directory"])["row"]
    es = dict(base, method="energy_slack", normal_fraction_all_offered=.8,
              ontime_reserve_fraction_all_offered=.8, observed_joint_fraction_meets_095=False)
    cp = dict(base, method="rh_cp_sat", normal_fraction_all_offered=.9,
              ontime_reserve_fraction_all_offered=.9, observed_joint_fraction_meets_095=False)
    seed2 = dict(es, seed=23, normal_fraction_all_offered=1., observed_joint_fraction_meets_095=True)
    aggregate = aggregate_rows([seed2, es])[0]
    assert aggregate["number_seeds"] == 2
    assert aggregate["normal_fraction_all_offered_mean"] == pytest.approx(.9)
    assert aggregate["normal_fraction_all_offered_min"] == .8
    assert aggregate["normal_fraction_all_offered_max"] == 1.
    assert aggregate["observed_seeds_meeting_joint_095"] == 1
    factor = dict(es, profile="low_soc", normal_fraction_all_offered=.7)
    paired = paired_differences([cp, factor, seed2, es])
    assert paired["method_cp_minus_es"][0]["normal_fraction_all_offered_delta"] == pytest.approx(.1)
    assert paired["factors_minus_half6"][0]["normal_fraction_all_offered_delta"] == pytest.approx(-.1)
    assert len(paired["method_cp_minus_es"]) == 1
    assert paired["method_difference_summary"][0]["number_paired_seeds"] == 1
    assert paired["method_difference_summary"][0]["normal_fraction_all_offered_delta_mean"] == pytest.approx(.1)
    cp["manifest_hash"] = "different"
    factor["latent_marks_hash"] = "different"
    rejected = paired_differences([cp, es, factor])
    assert rejected["method_cp_minus_es"] == []
    assert rejected["factors_minus_half6"] == []


def test_cli_returns_nonzero_for_partial_campaign(campaign, tmp_path):
    directory, jobs = campaign
    (Path(jobs[1]["directory"]) / "results.json").unlink()
    path = Path(__file__).resolve().parents[1] / "scripts/analysis/analyze_full_service_capacity.py"
    spec = importlib.util.spec_from_file_location("capacity_report_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--campaign", str(directory), "--output", str(tmp_path / "cli_report")]) == 1


def test_single_request_plot_skips_before_matplotlib_import(campaign, tmp_path, monkeypatch):
    directory, _ = campaign
    report = analyze_campaign(directory)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    result = plot_report(report, tmp_path / "no_placeholder_figures")
    assert result["status"] == "skipped"
    assert "placeholder rate" in result["reason"]
    assert not (tmp_path / "no_placeholder_figures").exists()
