"""Evidencepipeline tests; these are notthe94-case qualification itself."""
import json
from pathlib import Path
import pytest

from terminal_marl.core.rotating_slots.qualification import catalogue, run_catalogue, task_coverage, merge_reports
from terminal_marl.core.rotating_slots.diagnostics import run_diagnostics
from terminal_marl.core.rotating_slots.replay import ReplayBundle


def test_catalogue94_is_explicit_and_unique():
    rows = catalogue()
    assert len(rows) == 94 and len({r["id"] for r in rows}) == 94
    assert sum(r["id"][0] in "ABCDEFG" for r in rows) == 82
    assert sum(r["id"].startswith("V") for r in rows) == 12
    assert all(r["status"] == "NOT_RUN" for r in rows)


def test_amended_execution_metadata_preserves_frozen_catalogue_and_merge(tmp_path,monkeypatch):
    from terminal_marl.core.rotating_slots import fixtures,qualification
    from terminal_marl.core.rotating_slots.world import POLICY_ID
    original_bytes=qualification.CATALOGUE.read_bytes()
    original={r["id"]:r for r in catalogue()}
    def minimal(case,e):
        e.check(True,"metadata-only pipeline fixture",True,True)
    monkeypatch.setattr(fixtures,"execute_case",minimal)
    first=run_catalogue(tmp_path/"f08",cases=["F08"])
    second=run_catalogue(tmp_path/"f09",cases=["F09"])
    assert first["policy_id"]==second["policy_id"]==POLICY_ID
    for name,report in (("F08",first),("F09",second)):
        row=next(r for r in report["cases"] if r["id"]==name)
        assert row["status"]=="PASS" and row["policy_id"]==POLICY_ID
        assert row["legacy_catalogue_expected"]==original[name]["expected"]
        assert row["expected"]!=row["legacy_catalogue_expected"]
        assert row["effective_criteria_source"].endswith("amendment_20261018.md")
        stored=json.loads(Path(row["fixture_path"]).read_bytes())
        assert stored["policy_id"]==POLICY_ID
    merged=merge_reports([tmp_path/"f08/results.json",tmp_path/"f09/results.json"],tmp_path/"merge")
    assert merged["policy_id"]==POLICY_ID
    assert next(r for r in merged["cases"] if r["id"]=="F08")==next(r for r in first["cases"] if r["id"]=="F08")
    assert qualification.CATALOGUE.read_bytes()==original_bytes
    assert catalogue()==list(original.values())


def test_actual_assertions_have_paths_and_unselected_cases_do_not_pass(tmp_path):
    result = run_catalogue(tmp_path/"catalogue", cases=["A01", "D01", "D02", "D04", "V12"])
    by_id = {r["id"]: r for r in result["cases"]}
    for case in ("A01", "D01", "D02", "D04"):
        row = by_id[case]
        assert row["status"] == "PASS" and row["assertion_count"] > 0
        assert Path(row["fixture_path"]).is_file() and Path(row["log_path"]).is_file()
        checks = json.loads(Path(row["assertions_path"]).read_text(encoding="utf8"))
        assert checks and all(c["passed"] and "expected" in c and "actual" in c for c in checks)
        assert row["code_sha256"] and row["config_hash"]
        for bundle in row["replay_paths"]: ReplayBundle.validate(bundle)
    assert by_id["A02"]["status"] == "NOT_RUN"
    assert by_id["V12"]["status"] == "HUMAN_PENDING"
    assert result["stage1_status"] == "NOT_ACCEPTED"
    assert result["researcher_visual_acceptance"] == "PENDING"


def test_unconstructed_fixture_is_blocked_with_reason_not_pass(tmp_path, monkeypatch):
    from terminal_marl.core.rotating_slots import fixtures
    def unavailable(case, evidence):
        raise fixtures.FixtureBlocked("deliberatelymissingfixture")
    monkeypatch.setattr(fixtures, "execute_case", unavailable)
    result = run_catalogue(tmp_path/"blocked", cases=["A01"])
    row = next(r for r in result["cases"] if r["id"] == "A01")
    assert row["status"] == "BLOCKED" and "deliberatelymissingfixture" in row["reason"]
    assert result["coverage"]["failures"] == ["A01"]
    assert result["automatic_status"] == "IMPLEMENTATION_IN_PROGRESS"


def test_coverage_detects_duplicates_missing_and_failures():
    actual = [{"id": "A01", "status": "PASS"}, {"id": "A01", "status": "PASS"}, {"id": "A03", "status": "FAIL"}]
    check = task_coverage(["A01", "A02", "A03"], actual)
    assert check["duplicates"] == ["A01"] and check["missing"] == ["A02"] and check["failures"] == ["A03"]


def test_serial_parallel_diagnostic_task_and_outcome_parity(tmp_path):
    serial = run_diagnostics(tmp_path/"serial", workers=1, pilot="traffic", pilot_window=.4, pilot_drain=.2)
    parallel = run_diagnostics(tmp_path/"parallel", workers=2, pilot="traffic", pilot_window=.4, pilot_drain=.2)
    assert serial["coverage"] == parallel["coverage"]
    assert serial["coverage"]["actual_count"] == 2 and not serial["coverage"]["failures"]
    a = {r["id"]: r for r in serial["jobs"]}; b = {r["id"]: r for r in parallel["jobs"]}
    assert a.keys() == b.keys()
    for case in a:
        assert a[case]["manifest"] == b[case]["manifest"] and a[case]["metrics"] == b[case]["metrics"]
        assert a[case]["decision_times"] == b[case]["decision_times"]
        assert a[case]["state_count"] == b[case]["state_count"]
        assert a[case]["measured_components"]["viewer"]["enabled"] is False
        assert all(value == "1" for value in b[case]["provenance"]["threads"].values())
        ReplayBundle.validate(a[case]["replay_path"])
        ReplayBundle.validate(b[case]["replay_path"])


def test_merge_preserves_paths_and_rejects_conflicting_provenance(tmp_path):
    run_catalogue(tmp_path/"first", cases=["A01"])
    run_catalogue(tmp_path/"second", cases=["D01", "V12"])
    first, second = tmp_path/"first/results.json", tmp_path/"second/results.json"
    result = merge_reports([first, second], tmp_path/"merged")
    rows = {r["id"]: r for r in result["cases"]}
    assert rows["A01"]["status"] == rows["D01"]["status"] == "PASS"
    assert rows["A02"]["status"] == "NOT_RUN" and rows["V12"]["status"] == "HUMAN_PENDING"
    assert Path(rows["A01"]["assertions_path"]).parent == tmp_path/"first/A01"
    assert result["stage1_status"] == "NOT_ACCEPTED"
    changed = json.loads(second.read_text(encoding="utf8"))
    next(r for r in changed["cases"] if r["id"] == "D01")["code_sha256"] = "different"
    bad = tmp_path/"bad.json"; bad.write_text(json.dumps(changed), encoding="utf8")
    with pytest.raises(ValueError, match="codehash/commit mismatch"):
        merge_reports([first, bad], tmp_path/"rejected")


@pytest.mark.parametrize("protocol", ["PILOT_NOT_FORMAL_MATRIX", "FROZEN_FORMAL_MATRIX"])
def test_g09_cannot_pass_from_pilot_or_missing_benchmark_rows(tmp_path, monkeypatch, protocol):
    source = tmp_path/"diagnostics.json"
    source.write_text(json.dumps({"schema": "FT-RS-S1-1.0-diagnostics", "protocol": protocol, "jobs": []}), encoding="utf8")
    monkeypatch.setenv("TERMINAL_DIAGNOSTIC_REPORT", str(source))
    result = run_catalogue(tmp_path/"case", cases=["G09"])
    row = next(r for r in result["cases"] if r["id"] == "G09")
    assert row["status"] == "FAIL" and row["assertion_count"] > 0
    assert result["automatic_status"] == "IMPLEMENTATION_IN_PROGRESS"


def test_audit_cli_writes_report_for_real_physical_archive(tmp_path, capsys):
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    from terminal_marl.scripts.run_stage1_tests import main
    world = RotatingSlotWorld()
    world.seed_service("service", 0, remaining=.125)
    world.advance(.2)
    bundle = ReplayBundle.write(tmp_path/"physical", world=world, code_commit="audit-cli-fixture", seed=11)
    output = tmp_path/"audit_output"
    code = main(["--audit", str(bundle.path), "--output", str(output)])
    report = json.loads((output/"audit.json").read_text(encoding="utf8"))
    assert code == 0 and report["status"] == "PASS" and report["issue_count"] == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_actual_arrival_callback_serializes_frozen_metrics(tmp_path):
    from collections.abc import Mapping
    from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    from terminal_marl.core.rotating_slots.qualification import write_json
    request = ArrivalRequest("actual_arrival", 0., 0, .9, "short")
    world = RotatingSlotWorld(manifest=ArrivalManifest((request,), seed=11))
    observed = []
    progress = tmp_path/"callback_progress.json"
    def callback(snapshot, port):
        if request.id not in port.aircraft:
            outcome = port.admit(request.id)
            assert outcome.accepted
        metrics, aircraft = port.metrics(), port.aircraft
        observed.append((metrics, aircraft))
        write_json(progress, {"status": "RUNNING", "metrics": metrics, "aircraft": aircraft})
    world.advance(0., callback)
    report = json.loads(progress.read_text(encoding="utf8"))
    assert observed and all(isinstance(x, Mapping) for x in observed[0])
    assert report["metrics"]["offered"] == report["metrics"]["admitted"] == 1
    assert report["aircraft"][request.id]["status"] == "INBOUND"
