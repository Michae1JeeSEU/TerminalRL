"""Synthetic same-source split campaigns, never new experimental evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from terminal_marl.analysis.full_capacity_study import analyze_study, write_report
from terminal_marl.analysis.full_service_capacity import atomic_json, digest, prepare_jobs, validate_spec
from tests.test_full_capacity_report import synthetic_completed


def spec():
    return dict(profiles=["half6"], methods=["energy_slack", "rh_cp_sat"], seeds=[11, 23],
        rates_per_hour=[120.], demand_kind="single", warmup_s=0., measurement_s=100.,
        drain_s=100., sla_delay_s=130., progress_interval_s=60.)


def make_campaign(path, selected_spec):
    jobs = prepare_jobs(selected_spec, path)
    for job in jobs:
        result = synthetic_completed(job)
        result["safety"]["terminated_invalid"] = False
        atomic_json(Path(job["directory"]) / "results.json", result)
    atomic_json(path / "summary.json", dict(expected_jobs=len(jobs), finished_jobs=len(jobs), all_completed=True,
        jobs=[dict(directory=job["directory"], status="COMPLETED") for job in jobs]))
    return jobs


@pytest.fixture
def split(tmp_path):
    expected = spec()
    paths = [tmp_path / "a", tmp_path / "b"]
    jobs = [make_campaign(path, dict(expected, seeds=[seed])) for path, seed in zip(paths, (11, 23))]
    return expected, paths, jobs


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_split_campaigns_complete_explicit_grid_and_write(split, tmp_path):
    expected, paths, _ = split
    report = analyze_study(paths, expected)
    assert report["coverage"]["all_completed"]
    assert report["coverage"]["expected_jobs"] == report["coverage"]["complete_jobs"] == 4
    assert len(report["paired"]["method_cp_minus_es"]) == 2
    assert len(report["jobs"]) == 4
    assert all(job["mechanism_audit"]["audited_aircraft"] == 1 for job in report["jobs"])
    assert {row["source_hash"] for row in report["per_job"]} == {report["source_hash"]}
    write_report(report, tmp_path / "report")
    assert read(tmp_path / "report/report.json")["coverage"]["all_completed"]


def test_missing_selected_keys_not_inferred_from_present_union(split):
    expected, paths, _ = split
    report = analyze_study(paths[:1], expected)
    assert report["coverage"]["expected_jobs"] == 4
    assert report["coverage"]["missing_jobs"] == 2
    assert not report["coverage"]["all_completed"]


def test_selected_failure_retains_raw_result_and_summary(split):
    expected, paths, jobs = split
    target = Path(jobs[0][0]["directory"])
    atomic_json(target / "results.json", dict(status="INVALID", exception="synthetic invalid continuation"))
    report = analyze_study(paths, expected)
    assert report["coverage"]["failed_jobs"] == 1
    assert not report["coverage"]["all_completed"]
    record = next(row for row in report["selected_job_records"] if row["directory"] == str(target))
    assert record["result"]["exception"] == "synthetic invalid continuation"
    assert record["summary_rows"][0]["status"] == "COMPLETED"


def test_duplicate_selected_key_is_excluded_from_aggregates(split):
    expected, paths, _ = split
    report = analyze_study(paths + paths[:1], expected)
    assert report["coverage"]["duplicate_keys"] == 2
    assert report["coverage"]["complete_jobs"] == 2
    assert {row["seed"] for row in report["per_job"]} == {23}
    assert not report["coverage"]["all_completed"]


@pytest.mark.parametrize("field", ["source", "runtime", "policy", "protocol"])
def test_incompatible_campaign_identities_rejected(split, field):
    expected, paths, _ = split
    raw = read(paths[1] / "campaign.json")
    if field == "source":
        raw["source"]["files"]["synthetic.py"] = "changed"
        raw["source"]["sha256"] = digest(raw["source"]["files"])
    elif field == "runtime":
        raw["runtime"]["interpreter"] = "/another/python"
    elif field == "policy":
        raw["policy"] = dict(raw["policy"], synthetic="changed")
    else:
        raw["spec"]["drain_s"] += 1
    atomic_json(paths[1] / "campaign.json", raw)
    with pytest.raises(ValueError, match="mismatch|protocol"):
        analyze_study(paths, expected)


def test_excluded_invalid_parent_jobs_do_not_poison_selected_study(tmp_path):
    expected = dict(spec(), seeds=[11])
    parent_spec = dict(expected, profiles=["half6", "radius250"])
    path = tmp_path / "old_screen"
    jobs = make_campaign(path, parent_spec)
    summary = read(path / "summary.json")
    for job, row in zip(jobs, summary["jobs"]):
        if job["identity"]["job"]["profile"] == "radius250":
            atomic_json(Path(job["directory"]) / "results.json", dict(status="INVALID", exception="old invalid geometry"))
            row.update(status="INVALID", exception="old invalid geometry")
    summary["all_completed"] = False
    atomic_json(path / "summary.json", summary)
    report = analyze_study([path], expected)
    assert report["coverage"]["all_completed"]
    assert report["coverage"]["excluded_jobs"] == 2
    assert report["source_campaigns"][0]["coverage"]["failed_jobs"] == 2
    assert not report["source_campaigns"][0]["coverage"]["all_completed"]
    assert all(row["exclusion_reason"] == "not_in_selected_study" and row["status"] == "INVALID"
               and row["result"]["exception"] == "old invalid geometry" for row in report["excluded_job_records"])


@pytest.mark.parametrize("change", ["absent", "missing_row", "duplicate_row", "wrong_status", "wrong_path", "wrong_finished"])
def test_selected_summary_correspondence_required(split, change):
    expected, paths, _ = split
    summary_path = paths[0] / "summary.json"
    summary = read(summary_path)
    if change == "absent":
        summary_path.unlink()
    else:
        if change == "missing_row":
            summary["jobs"].pop()
            summary["finished_jobs"] -= 1
        elif change == "duplicate_row":
            summary["jobs"].append(deepcopy(summary["jobs"][0]))
            summary["finished_jobs"] += 1
        elif change == "wrong_status":
            summary["jobs"][0]["status"] = "RUNNING"
        elif change == "wrong_path":
            summary["jobs"][0]["directory"] = "/another/job"
        else:
            summary["finished_jobs"] = 0
        atomic_json(summary_path, summary)
    report = analyze_study(paths, expected)
    assert not report["coverage"]["all_completed"]
    assert report["coverage"]["complete_jobs"] < 4


def test_job_identity_cannot_override_parent_source(split):
    expected, paths, jobs = split
    target = Path(jobs[0][0]["directory"])
    identity = read(target / "identity.json")
    identity["runtime"]["cp_workers"] = 99
    atomic_json(target / "identity.json", identity)
    result = read(target / "results.json")
    result["identity_hash"] = digest(identity)
    atomic_json(target / "results.json", result)
    report = analyze_study(paths, expected)
    assert not report["coverage"]["all_completed"]
    assert "parent campaign" in report["issues"]["failed"][0]["validation_errors"][0]


@pytest.mark.parametrize("change", ["cohort", "terminated_invalid", "missing_invalid_flag", "world_conservation", "omitted_audit"])
def test_completed_results_require_true_cohort_and_safety_audits(split, change):
    expected, paths, jobs = split
    target = Path(jobs[0][0]["directory"])
    result = read(target / "results.json")
    if change == "cohort":
        result["measurement_cohort"]["offered"] += 1
    elif change == "terminated_invalid":
        result["safety"]["terminated_invalid"] = True
    elif change == "missing_invalid_flag":
        result["safety"].pop("terminated_invalid")
    elif change == "world_conservation":
        result["safety"]["conservation"] = False
    else:
        plans = read(target / "trajectories.json")
        plans.pop("om")
        atomic_json(target / "trajectories.json", plans)
    atomic_json(target / "results.json", result)
    report = analyze_study(paths, expected)
    assert report["coverage"]["complete_jobs"] == 3
    assert report["coverage"]["failed_jobs"] == 1
    assert not report["coverage"]["all_completed"]
    assert report["issues"]["failed"][0]["validation_errors"]
    raw = next(row for row in report["selected_job_records"] if row["directory"] == str(target))
    assert raw["status"] == "COMPLETED"
    assert raw["result"] == result


def test_incomplete_unreadable_selected_and_unknown_identity(split):
    expected, paths, jobs = split
    (Path(jobs[0][0]["directory"]) / "results.json").unlink()
    (Path(jobs[0][1]["directory"]) / "results.json").write_text("broken", encoding="utf-8")
    extra = paths[0] / "jobs/unreadable"
    extra.mkdir()
    report = analyze_study(paths, expected)
    assert report["coverage"]["incomplete_jobs"] == 1
    assert report["coverage"]["failed_jobs"] == 1
    assert report["coverage"]["unreadable_jobs"] == 1
    assert not report["coverage"]["all_completed"]


def test_analysis_specs_prevent_runner_launch_and_preserve_protocol():
    root = Path(__file__).resolve().parents[1] / "experiments/service_capacity"
    base = read(root / "full_factorial_screen_v1.json")
    for name, count in (("full_geometry_study_v2.json", 36), ("full_nongeometry_study_v1.json", 72)):
        envelope = read(root / name)
        assert envelope["purpose"] == "analysis_only"
        assert envelope["automatic_launch"] is False
        assert envelope["expected_jobs"] == count
        with pytest.raises(ValueError, match="unknown spec fields"):
            validate_spec(envelope)
        for field in base.keys() - {"profiles", "seeds"}:
            assert envelope["spec"][field] == base[field]
    for name, profiles, seeds in (
        ("full_geometry_remaining_with_controls_v2.json", ["half6", "radius250", "radius350"], [23, 71]),
        ("full_geometry_control_seed11_v2.json", ["half6"], [11]),
    ):
        value = read(root / name)
        assert value == dict(base, profiles=profiles, seeds=seeds)


def test_cli_exit_status_and_envelope(split, tmp_path):
    expected, paths, _ = split
    file = Path(__file__).resolve().parents[1] / "scripts/analysis/analyze_full_capacity_study.py"
    module_spec = importlib.util.spec_from_file_location("study_cli", file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    source = tmp_path / "expected.json"
    atomic_json(source, dict(purpose="analysis_only", automatic_launch=False, expected_jobs=4, spec=expected))
    args = ["--spec", str(source), "--output", str(tmp_path / "report")]
    assert module.main(args + ["--campaign", str(paths[0])]) == 1
    assert module.main(args + ["--campaign", str(paths[0]), "--campaign", str(paths[1])]) == 0
    assert read(tmp_path / "report/report.json")["coverage"]["all_completed"]
