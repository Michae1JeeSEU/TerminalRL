"""FT-RS console conversion uses verified recordings without renderer imports."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.replay import ReplayBundle, _plain
from terminal_marl.core.rotating_slots.trajectory import generate_transfer, make_stable_plan
from terminal_marl.visualization.console_server import ConsoleInputError, ConsoleSession, load_replay


def _bundle(tmp_path):
    config = load_config()
    plan = make_stable_plan("original/uav-17", "outer", 0, end_time=2.)
    states = []
    for timestamp, status in ((0., "HOLD_OUTER"), (.2, "LANDED_SERVICE"), (.33, "CLEARED"), (.4, "ENERGY_MISSION_FAILED")):
        k = plan.state(timestamp)
        aircraft = {"id": "original/uav-17", "status": status, "soc": .8-timestamp*.01,
            "p": k.p, "v": k.v, "ring": "outer" if timestamp == 0. else None,
            "slot": 0 if timestamp == 0. else None, "pad_id": 0 if timestamp == .2 else None,
            "service_end": .33 if timestamp == .2 else None,
            "trajectory_hash": plan.trajectory_hash if timestamp == 0. else None,
            "emergency": {"declared": False}}
        if timestamp == 0.:
            aircraft["battery_power_w"] = 42.
        states.append({"time": timestamp, "aircraft": [aircraft], "slots": [
            {"resource": ["middle", 0], "aircraft_id": aircraft["id"], "start": .3,
             "end": 1., "kind": "target", "committed": True}],
            "pads": [{"resource": 0, "aircraft_id": aircraft["id"], "start": .2, "end": .33,
                      "kind": "service", "committed": True}],
            "metrics": {"upstream_waiting": 2, "number": 7, "flag": True, "nonnumeric": "x"},
            "upstream": [{"id": "external-request"}]})
    record = {"config_hash": config.config_hash, "sample_dt_s": .2, "states": states,
        "manifest": {"seed": 11}, "trajectories": {plan.trajectory_hash: plan.to_dict()},
        "metrics": {"minimum_separation": {"lower_bound_m": 4., "upper_bound_m": 5., "time": .1}},
        "events": [{"time": .2, "type": "touchdown", "aircraft_id": "original/uav-17", "details": {"soc": .798}},
                   {"time": .33, "type": "pad_service_completed", "aircraft_id": "original/uav-17", "details": None},
                   {"time": .4, "type": "aircraft_energy_failure", "aircraft_id": "original/uav-17", "details": {"reason": "battery"}},
                   {"time": .1, "type": "rescue_opportunity_attempt", "details": {"large": [1]*100}}],
        "rejections": []}
    return ReplayBundle.write(tmp_path/"experiment"/"method"/"bundle", record=record,
                              code_commit="recorded-commit", seed=11)


def test_bundle_directory_and_index_preserve_recorded_semantics(tmp_path):
    bundle = _bundle(tmp_path)
    title, frames, metadata = load_replay(bundle.path)
    assert load_replay(bundle.path/"replay.json") == (title, frames, metadata)
    assert [frame["time"] for frame in frames] == [0., .2, .33, .4]
    assert [frame["step_count"] for frame in frames] == [0, 1, 2, 3]
    assert metadata["step_count_semantics"] == "recorded_frame_index"
    first = frames[0]
    uav = first["uavs"][0]
    assert uav["aircraft_id"] == "original/uav-17" and uav["uid"] == 0 and uav["arrival_id"] == 0
    assert uav["position"] == list(bundle.plan(uav["trajectory_plan_id"]).state(0.).p)
    assert uav["status"] == "HOLD_OUTER" and uav["slot_index"] == 0
    assert uav["trajectory_points"] == [] and uav["battery_power_w"] == 42.
    assert frames[1]["uavs"][0]["battery_power_w"] is None
    assert first["telemetry"] == {"upstream_waiting": 2, "number": 7}
    assert first["external_queue_size"] == 2 and first["active_count"] == 1
    assert frames[1]["service_count"] == 1 and frames[1]["completed_count"] == 0 and frames[1]["active_count"] == 0
    assert frames[2]["completed_count"] == 1 and frames[3]["active_count"] == 0
    assert frames[3]["uavs"][0]["status"] == "ENERGY_MISSION_FAILED"
    assert [event["outcome"] for event in frames[-1]["terminal_events"]] == ["touchdown", "completed", "failed_energy"]
    assert all(event["aircraft_id"] == "original/uav-17" for event in frames[-1]["terminal_events"])
    assert frames[2]["terminal_events"][1]["soc_source"] == "recorded_sample"
    assert len(frames[1]["terminal_events"]) == 1 and len(frames[2]["terminal_events"]) == 2
    assert "events" not in first["rotating_slots"] and "diagnostic_rejections_not_executed" not in first["rotating_slots"]
    pad = frames[1]["rotating_slots"]["pads"][0]
    assert pad["occupant_id"] == "original/uav-17" and pad["service_end"] == .33
    assert pad["service_remaining"] == pytest.approx(.13)
    slot = next(slot for slot in first["rotating_slots"]["slots"] if slot["id"] == "middle:0")
    assert slot["reservations"][0]["future_interval"] is True
    assert slot["reservations"][0]["commitment"] == "COMMITTED"
    assert first["rotating_slots"]["separation_evidence"]["scope"] == "whole_episode"
    assert metadata["source"] == metadata["replay_kind"] == "rotating_slots"
    assert metadata["interpolation"] == first["interpolation"] == "none"
    assert metadata["code_commit"] == "recorded-commit" and metadata["seed"] == 11
    assert metadata["replay_index_sha256"] == sha256((bundle.path/"replay.json").read_bytes()).hexdigest()
    json.dumps(frames, allow_nan=False)


@pytest.mark.parametrize("mutation", ["tamper", "missing"])
def test_invalid_bundle_is_rejected_without_legacy_fallback(tmp_path, mutation):
    bundle = _bundle(tmp_path)
    path = bundle.path/"events.jsonl"
    if mutation == "tamper":
        path.write_bytes(b"{}\n")
    else:
        path.unlink()
    with pytest.raises(ConsoleInputError, match="invalid rotating_slots ReplayBundle"):
        load_replay(bundle.path)


def test_replay_load_reference_and_failure_do_not_mutate_session(tmp_path):
    bundle = _bundle(tmp_path)
    session = ConsoleSession(token="test")
    try:
        loaded = session.load_replay_path(bundle.path)
        assert loaded["status"] == "replay" and loaded["generation"] == 1
        original = session.export_payload()
        reference = session.load_replay_path(bundle.path, reference=True)
        assert reference["metadata"]["source"] == "rotating_slots"
        assert session.export_payload() == original and session.generation == 1
        with pytest.raises(ConsoleInputError):
            session.load_replay_path(bundle.path/"missing")
        assert session.export_payload() == original and session.generation == 1
    finally:
        session.close()


def test_adapter_import_does_not_load_vtk_or_pyvista():
    command = "import sys; import terminal_marl.visualization.rotating_slots_console; assert not any(k == 'vtk' or k.startswith('vtkmodules') or k == 'pyvista' or k.endswith('pyvista_viewer') for k in sys.modules)"
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_frame_bound_rejects_without_truncation(tmp_path):
    from terminal_marl.visualization.rotating_slots_console import load_rotating_slots_replay
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="no frames were dropped"):
        load_rotating_slots_replay(bundle.path, max_frames=3)


def test_active_maneuver_is_bounded_and_dispatch_is_latest_recorded(tmp_path):
    source = _plain(_bundle(tmp_path).data)
    candidate = generate_transfer("original/uav-17", "outer", 0, 0, 0., durations=(40., 60., 80.))
    assert candidate.accepted
    plan = candidate.plan
    source["trajectories"][plan.trajectory_hash] = plan.to_dict()
    source["states"][0]["aircraft"][0].update(status="TRANSFER_O_M", trajectory_hash=plan.trajectory_hash)
    source["metrics"]["experiment_policy_id"] = "rs_s2_loss_continue_throughput_v1_0"
    source["decisions"] = [{"time": 0., "method": "recorded_method"}]
    source["dispatch_timings"] = [{"time": 0., "dispatch_wall_s": 1.}, {"time": .3, "dispatch_wall_s": 2.}]
    bundle = ReplayBundle.write(tmp_path/"active", record=source, code_commit="recorded-commit")
    _, frames, metadata = load_replay(bundle.path)
    points = frames[0]["uavs"][0]["trajectory_points"]
    assert len(points) == 24 and points[0] == list(plan.state(0.).p) and points[-1] == list(plan.state(plan.end_time).p)
    assert frames[0]["rotating_slots"]["latest_dispatch_timing"]["dispatch_wall_s"] == 1.
    assert frames[-1]["rotating_slots"]["latest_dispatch_timing"]["dispatch_wall_s"] == 2.
    assert metadata["method"] == "recorded_method" and metadata["method_source"] == "recorded_decisions"


def test_terminal_missing_soc_remains_nullable_and_member_size_is_bounded(tmp_path):
    from terminal_marl.visualization.rotating_slots_console import load_rotating_slots_replay
    source = _plain(_bundle(tmp_path).data)
    source["events"].append({"time": .2, "type": "rerouted", "aircraft_id": "external-unknown", "details": None})
    bundle = ReplayBundle.write(tmp_path/"nullable", record=source, code_commit="recorded-commit")
    _, frames, _ = load_replay(bundle.path)
    event = next(event for event in frames[-1]["terminal_events"] if event["aircraft_id"] == "external-unknown")
    assert event["soc"] is None and event["energy_remaining"] is None and event["soc_source"] == "unavailable"
    with pytest.raises(ValueError, match="size limit|exceed"):
        load_rotating_slots_replay(bundle.path, max_bytes=100)
