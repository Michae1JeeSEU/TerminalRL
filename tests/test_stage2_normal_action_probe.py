"""Output-contract checks for the read-only normal-action diagnostic."""
import json
from dataclasses import replace
from pathlib import Path
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


def test_cli_refuses_existing_output_and_missing_checkpoint(tmp_path):
    output = tmp_path/"existing.json"
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        probe.main(["--output", str(output)])
    assert exc.value.code == 2 and output.read_text(encoding="utf-8") == "sentinel"
    with pytest.raises(SystemExit) as exc:
        probe.main(["--checkpoint", str(tmp_path/"unknown.json"), "--output", str(tmp_path/"new.json")])
    assert exc.value.code == 2 and not (tmp_path/"new.json").exists()


@pytest.fixture
def checkpoint_case(tmp_path):
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    root = tmp_path/"campaign"
    directory = root/"energy_slack"
    directory.mkdir(parents=True)
    world = probe.RotatingSlotWorld(manifest=probe.frozen_manifest(), experiment_policy_id=probe.STAGE2_POLICY_ID)
    world.seed_hold("initial_outer_00", "outer", 0, .8)
    world.advance(1.)
    provenance = probe.campaign.source_provenance()
    identity = probe.campaign.checkpoint_identity(provenance, probe.campaign.digest(world.manifest.to_dict()))
    for name, value in (("manifest.json", world.manifest.to_dict()), ("provenance.json", provenance),
                        ("campaign.json", {"identity": identity})):
        probe.campaign.atomic_json(root/name, value)
    path = directory/"checkpoint_0001.json"
    envelope = probe.campaign.save_checkpoint(path, world, NormalSchedulerCallback(), identity)
    sentinel = root/"must_not_open_or_truncate.bin"
    sentinel.write_bytes(b"keep this entire stream\n")
    envelope["payload"]["sink"] = {"path": str(sentinel), "offset": 0}
    envelope["sha256"] = probe.campaign.digest(envelope["payload"])
    probe.campaign.atomic_json(path, envelope)
    return path, sentinel


def test_checkpoint_probe_decodes_without_sink_or_execution_or_future_requests(checkpoint_case, monkeypatch):
    path, sentinel = checkpoint_case
    original = path.read_bytes()
    original_open = Path.open
    def guarded_open(self, mode="r", *args, **kwargs):
        assert self != sentinel, "diagnostic opened checkpoint sink"
        assert not any(flag in mode for flag in ("+", "w", "a")), mode
        return original_open(self, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("diagnostic called execution/restoration")
    for name in ("advance", "submit", "commit_validated_prefix", "refine_planning_boundary"):
        monkeypatch.setattr(probe.RotatingSlotWorld, name, forbidden)
    monkeypatch.setattr(probe.campaign, "restore_checkpoint", forbidden)
    observed = []
    def inspect_snapshot(snapshot, report, deadline):
        assert snapshot.time == 1. and not hasattr(snapshot, "manifest")
        assert not snapshot.requests  # All frozen releases occur after this boundary.
        observed.append(snapshot.time)
        report["catalog"] = {"status": "completed"}
    monkeypatch.setattr(probe, "collect_catalog", inspect_snapshot)
    report = probe.run_probe(checkpoint=path, catalog_only=True, max_wall_seconds=12.)
    assert report["status"] == "completed", report
    assert observed == [1.] and report["snapshot"]["time"] == 1.
    assert report["checkpoint"]["sink_metadata_ignored"] and report["checkpoint"]["state_sink"] is None
    assert report["state_unchanged"] and report["checkpoint_input_unchanged"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("corruption", ["schema", "digest", "identity", "config", "manifest"])
def test_checkpoint_probe_rejects_corrupt_inputs(checkpoint_case, corruption):
    from terminal_marl.core.rotating_slots.demand import ArrivalManifest
    path, _ = checkpoint_case
    envelope = probe.campaign.load_json(path)
    payload = envelope["payload"]
    if corruption == "schema":
        payload["schema"] = "unsupported"
    elif corruption == "identity":
        payload["identity"]["source_hash"] = "other"
    elif corruption in ("config", "manifest"):
        codec = probe.campaign.Stage2Codec()
        state = codec.decode(payload["world"])
        if corruption == "config":
            state["config"] = replace(state["config"], config_hash="other")
        else:
            state["manifest"] = ArrivalManifest((), 71, 180., 3600.)
        payload["world"] = codec.encode(state)
    if corruption != "digest":
        envelope["sha256"] = probe.campaign.digest(payload)
    else:
        envelope["sha256"] = "wrong"
    probe.campaign.atomic_json(path, envelope)
    with pytest.raises(probe.campaign.CheckpointError):
        probe.load_checkpoint_read_only(path)


def test_first_accepted_witness_screens_then_stops_per_aircraft(monkeypatch):
    world = probe.busy_fixture(probe.frozen_manifest())
    snapshot = world.build_planning_snapshot()
    calls = []
    monkeypatch.setattr(probe, "screen_normal_templates", lambda *args: (
        {(t, d): ("HORIZONTAL_JERK",) if d == 20 else None for t in range(28) for d in range(20,181,5)}, False))
    def generate(aid, ring, slot, target, start, **kwargs):
        duration = kwargs["durations"][0]
        calls.append((target, duration))
        plan = SimpleNamespace(aircraft_id=aid, source_ring=ring, source_slot=slot,
            target_ring="middle", target_slot=target, pad_id=None, start_time=start,
            end_time=start+duration, trajectory_hash=f"{target}-{duration}")
        return CandidateResult("FEASIBLE", plan)
    monkeypatch.setattr(probe, "generate_transfer", generate)
    monkeypatch.setattr(probe, "evaluate_planning_snapshot", lambda s, plans:
        ValidationResult(plans[0].end_time-s.time == 30, "SAFE", ""))
    report = {}
    probe.collect_witnesses(snapshot, ["outer_00"], report, probe.Deadline(5.), first_accepted=True, screen=True)
    row = report["witnesses"]["aircraft"][0]
    assert calls == [(0,25), (0,30)]
    assert row["accepted"] and row["accepted_plan"]["duration"] == 30
    assert row["screened_rejections"] == 1 and len(row["targets"]) == 1


def test_auto_witness_prioritizes_no_move_nonemergency_unfixed_holds():
    world = probe.busy_fixture(probe.frozen_manifest())
    snapshot = world.build_planning_snapshot()
    eligible = sorted(a.id for a in snapshot.aircraft.values() if a.status == "HOLD_OUTER")
    report = {"catalog": {"aircraft": {aid: {"move_count": 0 if aid in eligible[-3:] else 1} for aid in eligible}}}
    assert probe.auto_witnesses(snapshot, report, 3) == eligible[-3:]
    assert all(row["basis"] == "catalog_no_MOVE" for row in report["witness_selection"]["selected"])


def test_nonfinite_values_are_explicit_json_strings():
    value = probe.json_value({"values": [float("inf"), float("-inf"), float("nan")]})
    assert json.loads(json.dumps(value, allow_nan=False))["values"] == ["+Infinity", "-Infinity", "NaN"]
