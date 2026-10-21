"""Output-contract checks for the read-only normal-action diagnostic."""
import json
from types import SimpleNamespace

import pytest

from scripts.diagnostics import probe_stage2_normal_actions as probe
from terminal_marl.core.rotating_slots.models import CandidateResult, ValidationResult


def test_isolated_catalog_output_is_serializable_and_read_only(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("diagnostic attempted a world mutation")

    for name in ("advance", "submit", "commit_validated_prefix", "refine_planning_boundary"):
        monkeypatch.setattr(probe.RotatingSlotWorld, name, forbidden)
    report = probe.run_probe(catalog_only=True, isolated_aircraft="outer_00", max_wall_seconds=12.)
    assert report["status"] == "completed" and not report["partial"]
    assert report["state_unchanged"] and report["read_only"]
    assert report["snapshot"]["aircraft_count"] == 1
    catalog = report["catalog"]
    assert catalog["budget"]["candidate_seconds"] == 3.
    assert catalog["budget"]["max_per_aircraft"] == 64
    assert catalog["budget"]["max_attempts"] == 2048
    assert catalog["generate_calls"]
    call = catalog["generate_calls"][0]
    assert {"aircraft_id", "source_ring", "source_slot", "target_ring", "target_slot",
            "pad_id", "start", "duration", "attempts_before", "attempts_after",
            "returned_plan", "budget_truncated"} <= call.keys()
    aircraft = catalog["aircraft"]["initial_outer_00"]
    assert {"move_count", "first_target_distribution", "references", "columns"} <= aircraft.keys()
    assert all({"chain", "first_start_le_now_plus_5", "prefix_eligible", "reference"} <= c.keys()
               for c in aircraft["columns"])
    assert all({"kind", "chain", "accepted", "reason", "details"} <= c.keys()
               for c in catalog["single_checks"])
    assert {"wall_s", "cpu_s"} <= report["timings"]["catalog"].keys()
    assert json.loads(json.dumps(report, allow_nan=False))["schema"] == "stage2_normal_action_probe_v1"


def test_witness_output_separates_motion_from_snapshot_prefix_check(monkeypatch):
    world = probe.busy_fixture(probe.frozen_manifest())
    snapshot = world.build_planning_snapshot()
    checks = []

    def generate(aircraft_id, source_ring, source_slot, target, start_time, **kwargs):
        duration = kwargs["durations"][0]
        plan = SimpleNamespace(aircraft_id=aircraft_id, source_ring=source_ring,
            source_slot=source_slot, target_ring="middle", target_slot=target, pad_id=None,
            start_time=start_time, end_time=start_time+duration, trajectory_hash=f"test-{target}")
        return CandidateResult("FEASIBLE", plan)

    def evaluate(snapshot_arg, plans):
        assert snapshot_arg is snapshot and len(plans) == 1
        checks.append(plans[0].trajectory_hash)
        return ValidationResult(False, "CONFLICT", "RESOURCE_CONFLICT", {"resource": ("middle", 0)})

    monkeypatch.setattr(probe, "generate_transfer", generate)
    monkeypatch.setattr(probe, "evaluate_planning_snapshot", evaluate)
    report = {}
    probe.collect_witnesses(snapshot, ["outer_00"], report, probe.Deadline(5.))
    witnesses = report["witnesses"]
    assert witnesses["status"] == "completed" and witnesses["separate_from_catalog_budget"]
    row = witnesses["aircraft"][0]
    assert row["duration_domain"] == list(range(20, 181, 5))
    assert len(checks) == len(row["targets"]) > 0
    assert all(t["motion_feasible"] and len(t["attempts"]) == 1 for t in row["targets"])
    assert all(t["prefix_check"]["reason"] == "RESOURCE_CONFLICT" for t in row["targets"])
    assert world.build_planning_snapshot().version == snapshot.version
    json.dumps(probe.json_value(report), allow_nan=False)


def test_cli_refuses_existing_output_and_unimplemented_checkpoint(tmp_path):
    output = tmp_path/"existing.json"
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        probe.main(["--output", str(output)])
    assert exc.value.code == 2 and output.read_text(encoding="utf-8") == "sentinel"
    with pytest.raises(SystemExit) as exc:
        probe.main(["--checkpoint", str(tmp_path/"unknown.json"), "--output", str(tmp_path/"new.json")])
    assert exc.value.code == 2 and not (tmp_path/"new.json").exists()


def test_nonfinite_values_are_explicit_json_strings():
    value = probe.json_value({"values": [float("inf"), float("-inf"), float("nan")]})
    assert json.loads(json.dumps(value, allow_nan=False))["values"] == ["+Infinity", "-Infinity", "NaN"]
