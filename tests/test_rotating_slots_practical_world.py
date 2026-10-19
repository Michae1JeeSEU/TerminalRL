"""Bounded runtime checks; interpolation is never a planning certificate."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.energy import EnergyResult
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.performance.checkpoint import (
    AdvanceCursor, CheckpointIdentity, POLICY_ID, StateCodec,
    capture_checkpoint, restore_checkpoint,
)


def test_normal_gate_skips_reference_and_telemetry_is_one_second(monkeypatch):
    world = RotatingSlotWorld()
    world.seed_hold("hold", "inner", 0, soc=.9)
    monkeypatch.setattr(world, "_reference", lambda *a: pytest.fail("high SOC reference search"))
    world.advance(2.13, False)
    assert [f["time"] for f in world.records] == pytest.approx([0., 1., 2., 2.13])
    stats = world.practical_runtime_stats()["counters"]
    assert stats["soc_gate_skips"] == 3
    assert stats["monitor_throttled"] >= 8
    assert not world._practical_runtime["table"]
    assert world.aircraft["hold"].emergency.q_e is None
    assert world.to_dict()["execution_dt"] == .2
    assert world.to_dict()["telemetry_dt_s"] == 1.


def test_offsecond_release_maneuver_touchdown_service_and_descent_frames():
    request = ArrivalRequest("queued", .37, 1, .8, "short")
    world = RotatingSlotWorld(manifest=ArrivalManifest((request,)))
    world.seed_service("service", 1, remaining=.37)
    world.seed_hold("final", "inner", 0, soc=.9, service_class="short")
    plan = generate_final("final", "inner", 0, 0, .13).plan
    assert plan is not None and world.submit((plan,)).accepted
    world.advance(plan.end_time+60., False)
    descent = world._descent_start(plan)
    times = [f["time"] for f in world.records]
    for timestamp in (.13, .37, descent, plan.end_time, plan.end_time+60.):
        assert timestamp in times
    assert world.aircraft["final"].touchdown_time == plan.end_time
    assert world.aircraft["final"].service_end == plan.end_time+60.
    assert world.aircraft["final"].status == "CLEARED"
    for event in world.events:
        assert event["time"] in times
    frame = next(f for f in world.records if f["time"] == descent)
    aircraft = next(a for a in frame["aircraft"] if a["id"] == "final")
    assert aircraft["p"] == pytest.approx(plan.state(descent).p, abs=1e-9)


def test_projected_soc_cheap_bound_avoids_remaining_energy(monkeypatch):
    world = RotatingSlotWorld()
    world.seed_hold("a", soc=.9)
    plan = generate_transfer("a", "outer", 0, 0, 0.).plan
    assert plan is not None
    world.active["a"] = plan
    monkeypatch.setattr(world, "_energy", lambda *args: pytest.fail("unnecessary remaining-energy integration"))
    monkeypatch.setattr(world, "_reference", lambda *args: pytest.fail("unnecessary reference"))
    assert not world._update_emergency()
    assert world.practical_runtime_stats()["counters"]["projected_bound_skips"] == 1


def test_committed_transfer_projected_low_soc_and_low_soc_fast_monitor(monkeypatch):
    world = RotatingSlotWorld()
    world.auto_emergency = False
    world.seed_hold("a", soc=.36)
    plan = generate_transfer("a", "outer", 0, 0, 0.).plan
    assert plan is not None
    world.active["a"] = plan
    monkeypatch.setattr(world, "_energy", lambda *args: EnergyResult(1., .04, 0., True))
    calls = []
    monkeypatch.setattr(world, "_monitor_reference", lambda *args: calls.append(world.time) or .05)
    for timestamp in (0., .2, .4):
        world.time = timestamp
        world._update_emergency()
    assert calls == [0., .2, .4]
    assert world.aircraft["a"].emergency.pending
    assert world.aircraft["a"].soc == .36  # projected query is not execution


def test_exact_imminent_hint_bypasses_high_soc_gate(monkeypatch):
    world = RotatingSlotWorld()
    world.seed_hold("a", "inner", soc=.4)
    world._practical_runtime["imminent"]["a"] = dict(time=.13, ring="inner", slot=0, version=0)
    world.time = .13
    calls = []
    monkeypatch.setattr(world, "_reference", lambda *args: calls.append(world.time) or (.14, None))
    monkeypatch.setattr(world, "_monitor_reference", lambda *args: pytest.fail("imminent query must be exact"))
    assert world._update_emergency()
    assert calls == [.13] and world.aircraft["a"].emergency.eligible


def test_high_soc_missing_reference_rechecks_exact_every_physics_step(monkeypatch):
    world = RotatingSlotWorld()
    world.seed_hold("a", "inner", soc=.9)
    a = world.aircraft["a"]
    world.aircraft["a"] = replace(a, emergency=replace(a.emergency, reference_missing=True))
    calls = []
    answers = iter((None, None, .06))
    monkeypatch.setattr(world, "_reference", lambda *args: calls.append(world.time) or (next(answers), None))
    monkeypatch.setattr(world, "_monitor_reference", lambda *args: pytest.fail("missing reference must use original exact routine"))
    for timestamp in (0., .2, .4):
        world.time = timestamp
        world._update_emergency()
    assert calls == [0., .2, .4]
    assert world.aircraft["a"].emergency.q_e == .06
    assert not world.aircraft["a"].emergency.reference_missing
    assert not world.aircraft["a"].emergency.eligible
    world.time = .6
    world._update_emergency()
    assert calls == [0., .2, .4]  # resolved high-SOC state resumes normal cadence
    assert world.practical_runtime_stats()["counters"]["exact_reference_calls"] == 3


def test_real_piecewise_phase_samples_agree_and_are_not_constant():
    world = RotatingSlotWorld()
    # First uncovered call must return the exact original answer while warming
    # only six of 720 cells. These cells have no sampled feasibility/duration switch.
    q0 = world._monitor_reference("inner", 0, 2.3, 0, .35, 0.)
    assert q0 == world._reference("inner", 0, 2.3)[0]
    estimates = [world._monitor_reference("inner", 0, t, 0, .35, 0.) for t in (2.4, 2.5)]
    exact = [world._reference("inner", 0, t)[0] for t in (2.4, 2.5)]
    assert estimates == pytest.approx(exact, abs=2e-5)
    assert estimates[0] != estimates[1]
    stats = world.practical_runtime_stats()
    assert stats["table_cells"] == 6 and stats["counters"]["table_samples"] == 18
    assert stats["counters"]["table_hits"] == 2
    assert stats["precompute_seconds"] > 0


def test_table_switch_uncertainty_and_critical_reserve_are_exact(monkeypatch):
    world = RotatingSlotWorld()
    # Construct a continuous synthetic cell covering all six pads.
    width = 2*math.pi/240
    for cell in (3, 43, 83, 123, 163, 203):
        world._practical_runtime["table"][f"inner:{cell}"] = [[.05, 35.], [.051, 35.], [.052, 35.]]
    calls = []
    monkeypatch.setattr(world, "_reference", lambda *args: calls.append(args) or (.077, None))
    monkeypatch.setattr(world, "_reference_pad", lambda *args: pytest.fail("covered cell sampled again"))
    assert world._monitor_reference("inner", 0, 3.2*width/.035, 0, .3, 0.) == .077
    assert len(calls) == 1  # reserve-critical, even though interpolation exists
    world._practical_runtime["table"]["inner:3"][1][1] = 40.
    assert world._monitor_reference("inner", 0, 3.2*width/.035, 0, .35, 0.) == .077
    assert len(calls) == 2  # sampled feasibility/duration switch
    world._practical_runtime["table"]["inner:3"] = [[None, None]]*3
    assert world._monitor_reference("inner", 0, 3.2*width/.035, 0, .35, 0.) == .077
    assert len(calls) == 3


def test_table_persistence_identity_and_legacy_checkpoint_defaults(tmp_path):
    path = tmp_path/"reference.json"
    world = RotatingSlotWorld(reference_cache_path=path)
    world._practical_runtime["table"]["inner:3"] = [[.05, 35.], [.051, 35.], [.052, 35.]]
    world.flush_reference_table()
    restored = RotatingSlotWorld(reference_cache_path=path)
    assert restored._practical_runtime["table"] == world._practical_runtime["table"]
    assert restored.practical_runtime_stats()["counters"]["table_loaded"] == 1
    data = json.loads(path.read_text())
    data["identity"]["profile_hash"] = "different"
    path.write_text(json.dumps(data))
    mismatch = RotatingSlotWorld(reference_cache_path=path)
    assert not mismatch._practical_runtime["table"]
    assert mismatch.practical_runtime_stats()["counters"]["table_identity_rejected"] == 1

    world.seed_hold("a", "inner", soc=.9)
    world._base(world.aircraft["a"])
    config = load_config()
    identity = CheckpointIdentity(POLICY_ID, config.config_hash, config.profile_hash, config.raw_hash,
                                  "rules", "candidates", "numerics", "implementation")
    payload = capture_checkpoint(world, identity=identity, cursor=AdvanceCursor(2., "loop_start", "test"))
    cold = restore_checkpoint(payload, expected_identity=identity, cache_mode="cold").world
    assert not cold._practical_runtime["base_plans"] and not cold._practical_runtime["table"]
    assert StateCodec().encode(cold.aircraft) == StateCodec().encode(world.aircraft)
    legacy = deepcopy(payload)
    legacy.pop("optional_runtime")
    baseline = restore_checkpoint(legacy, expected_identity=identity).world
    assert baseline._runtime_enabled()
    assert StateCodec().encode(baseline.aircraft) == StateCodec().encode(world.aircraft)
    assert baseline.configure_practical_runtime(False) is baseline


def test_stable_base_reused_past_archive_end_and_runtime_period_validation():
    world = RotatingSlotWorld()
    world.seed_hold("a", "inner", soc=.9)
    a = world.aircraft["a"]
    plan = world._base(a)
    world.time = 1001.
    assert world._base(a) is plan
    assert world._base(a).state(world.time) == plan.state(world.time)
    assert world.practical_runtime_stats()["counters"]["base_plan_hits"] == 2
    with pytest.raises(ValueError):
        world.configure_practical_runtime(telemetry_period=.1)
    with pytest.raises(ValueError):
        world.configure_practical_runtime(soc_threshold=.3)
