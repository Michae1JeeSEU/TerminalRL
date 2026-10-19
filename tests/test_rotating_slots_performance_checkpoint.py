"""Bounded P1 evidence; small scenes are not full traffic qualification."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math

import pytest

from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.driver import DeterministicDriver
from terminal_marl.core.rotating_slots.energy import EnergyResult
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink
from terminal_marl.core.rotating_slots.trajectory import make_ground_plan, make_stable_plan
from terminal_marl.core.rotating_slots.world import AircraftState, EmergencyStatus, RotatingSlotWorld
from terminal_marl.core.rotating_slots.performance.checkpoint import (
    AdvanceCursor, CheckpointError, CheckpointIdentity, CheckpointStore, POLICY_ID,
    StateCodec, capture_checkpoint, load_checkpoint, register_equivalence_migration,
    restore_checkpoint, write_checkpoint,
)
from terminal_marl.core.rotating_slots.performance.measurements import Measurements, PassiveHeartbeat
from terminal_marl.core.rotating_slots.performance.observation import (
    CutRequests, SuspendAtCut, make_observed_advance,
)


def identity():
    c = load_config()
    return CheckpointIdentity(POLICY_ID, c.config_hash, c.profile_hash, c.raw_hash,
                              "rules", "candidates", "numerics", "implementation-r0")


def cursor(target=2., phase="loop_start"):
    return AdvanceCursor(target, phase, "arrival_window")


def checkpoint(world=None, driver=None):
    return capture_checkpoint(world or RotatingSlotWorld(), identity=identity(), cursor=cursor(), driver=driver)


def mixed_world():
    world = RotatingSlotWorld()
    world.manifest = ArrivalManifest((ArrivalRequest("future", 1., 0, .8, "short"),), 11, 60.)
    world.requests = {"upstream": ArrivalRequest("upstream", 0., 1, .75, "long")}
    world.upstream_reasons = {"upstream": {"reason": "WAIT", "last_attempt_time": .125}}
    hold = make_stable_plan("active", "outer", 3, end_time=8.)
    soft = make_stable_plan("soft", "middle", 1, start_time=1., end_time=9.)
    for plan in (hold, soft):
        world.trajectories[plan.trajectory_hash] = plan.to_dict()
    world.active["active"] = hold
    world.soft["soft"] = soft
    world.aircraft["active"] = AircraftState("active", "HOLD_OUTER", .6000000000000001,
        *tuple(hold.state(0.).__dict__[key] for key in ("p", "v", "a", "jerk")), ring="outer", slot=3,
        emergency=EmergencyStatus(declared=True, charged=True, q_e=.12), energy_j=1234., peak_seconds=119.9999,
        trajectory_hash=hold.trajectory_hash, normal_final_touchdown_soc=.2500000001)
    world.aircraft["soft"] = replace(world.aircraft["active"], id="soft", ring="middle", slot=1)
    world.aircraft["ground"] = AircraftState("ground", "LANDED_SERVICE", .3, (100., 0., 0.),
        (0., 0., 0.), (0., 0., 0.), (0., 0., 0.), pad_id=0, service_end=1.)
    world.slots.reserve(("outer", 3), "active", 0., math.inf, plan_hash=hold.trajectory_hash, committed=True)
    world.pads.reserve(0, "ground", 0., 1., committed=True)
    world.dependencies = {soft.trajectory_hash: (hold.trajectory_hash,)}
    world._source_clear = {hold.trajectory_hash: (.5, ("outer", 3))}
    world.events.append({"time": 0., "type": "source_slot_cleared", "aircraft_id": None,
                         "details": {"trajectory_hash": hold.trajectory_hash}})
    world.rejections.append({"reason": "prior", "details": {2: (math.inf, -0.)}})
    world.costs["emergency"] = 3600.
    world.pad_busy_seconds = {5: 3., 0: 2., 2: 1.}
    world._q_cache[("outer", 3, 0.)] = (.1, hold)
    world._energy_cache[(hold.trajectory_hash, 0., .2)] = EnergyResult(1., .01, .001, True)
    world._reference_templates[("outer", 0, 0)] = 25.
    world._record()
    return world


def semantic(world):
    value = deepcopy(world.to_dict())
    for key in ("cpu", "cpu_seconds", "cache", "cache_hits", "reference_counts"):
        value.pop(key, None)
    # R0 metrics expose timers/cache diagnostics that are not physical state.
    if "metrics" in value:
        for key in ("cpu", "cache", "reference_counts"):
            value["metrics"].pop(key, None)
    for frame in value.get("states", []):
        for key in ("cpu", "cache", "reference_counts"):
            frame.get("metrics", {}).pop(key, None)
    return value


def small_event_world(sink=None):
    world = RotatingSlotWorld(state_sink=sink, manifest=ArrivalManifest((
        ArrivalRequest("r0", .4, 0, .8, "short"),
        ArrivalRequest("r1", .40000000001, 1, .7, "long"))))
    world.aircraft["ground"] = AircraftState("ground", "LANDED_SERVICE", .3,
        (100., 0., 0.), (0., 0., 0.), (0., 0., 0.), (0., 0., 0.), pad_id=0, service_end=.4)
    world._source_clear = {"earlier": (0., None), "same-time": (.4, None)}
    world.events.append({"time": 0., "type": "source_slot_cleared", "aircraft_id": None,
                         "details": {"trajectory_hash": "earlier"}})
    world._record()
    return world


def test_E01_E06_full_state_inventory_exact_roundtrip():
    world = mixed_world()
    driver = DeterministicDriver(); driver.calls = [0., .125]
    world.decision_callback = driver
    payload = checkpoint(world, driver)
    restored = restore_checkpoint(payload, expected_identity=identity(), cache_mode="warm")
    codec = StateCodec()
    for name in world.__dict__:
        if name not in ("state_sink", "decision_callback", "_scheduler_port"):
            assert codec.encode(world.__dict__[name]) == codec.encode(restored.world.__dict__[name]), name
    assert restored.world.decision_callback is restored.driver
    assert restored.driver.calls == [0., .125]
    assert list(restored.world.pad_busy_seconds) == [5, 0, 2]
    assert restored.world.slots.reservations[0].end == math.inf
    assert math.copysign(1., restored.world.rejections[-1]["details"][2][1]) == -1.
    assert restored.world._scheduler_port.time == world.time


def test_unknown_world_field_type_and_callback_fail_closed():
    world = RotatingSlotWorld(); world.unregistered = 1
    with pytest.raises(CheckpointError, match="world fields"):
        checkpoint(world)
    world = RotatingSlotWorld(); world.costs["unknown"] = object()
    with pytest.raises(CheckpointError, match="unsupported state type"):
        checkpoint(world)
    with pytest.raises(CheckpointError, match="DeterministicDriver"):
        checkpoint(RotatingSlotWorld(), lambda *_: None)


@pytest.mark.parametrize("phase", ["loop_start", "returned"])
def test_E02_E03_E05_A07_observation_suspend_resume_same_events(phase):
    plain = small_event_world(); plain.advance(1., False)
    baseline = semantic(plain)
    world = small_event_world()
    observed = make_observed_advance(RotatingSlotWorld)
    saved = []
    def hook(w, cur, driver):
        if not saved and ((phase == "loop_start" and w.time >= .4) or cur.phase == phase == "returned"):
            saved.append(capture_checkpoint(w, identity=identity(), cursor=cur, driver=driver))
            raise SuspendAtCut()
    with pytest.raises(SuspendAtCut):
        observed.run(world, 1., False, hook, outer_phase="arrival_window")
    restored = restore_checkpoint(saved[0], expected_identity=identity())
    observed.resume(restored)
    assert semantic(restored.world) == baseline
    assert sum(e["type"] == "source_slot_cleared" and e["details"]["trajectory_hash"] == "earlier"
               for e in restored.world.events) == 1
    assert sum(e["type"] == "pad_service_completed" for e in restored.world.events) == 1
    assert restored.cursor.original_target == 1.
    assert restored.cursor.outer_phase == "arrival_window"


def test_E04_non_grid_request_does_not_split_step():
    baseline = RotatingSlotWorld(); baseline.advance(1., False)
    observed = make_observed_advance(RotatingSlotWorld)
    world = RotatingSlotWorld()
    evidence = []
    requests = CutRequests(lambda w, c, d, e: evidence.append(e), requested_sim=(.33,), clock=lambda: 0.)
    observed.run(world, 1., False, requests)
    assert len(evidence) == 1
    assert evidence[0]["actual_sim_time"] == .4
    assert evidence[0]["simulation_deferral_s"] == [.4 - .33]
    assert world.decision_times == baseline.decision_times
    assert [r["time"] for r in world.records] == [r["time"] for r in baseline.records]
    assert semantic(world) == semantic(baseline)
    assert observed.patch.count("+        _performance_hook") == 1
    assert "self._advance_energy(next_time)" in observed.original_source
    assert "self._advance_energy(next_time)" in observed.generated_source


def test_wall_and_milestone_cuts_keep_independent_simulation_cadence():
    from types import SimpleNamespace
    wall = [0.]
    captured = []
    requests = CutRequests(lambda w, c, d, e: captured.append(e),
                           clock=lambda: wall[0], requested_sim=(15., 90., 95.))
    world = SimpleNamespace(time=0.)
    for second in range(1, 362):
        world.time = float(second)
        wall[0] = second * 300.
        requests(world, cursor(400.), False)
        assert requests.last_sim == float(second // 30 * 30)
        assert requests.last_wall == wall[0]
    periodic = [e for e in captured if "simulation_interval" in e["reasons"]]
    assert [e["actual_sim_time"] for e in periodic] == list(range(30, 361, 30))
    assert periodic[2]["reasons"] == ["simulation_interval", "wall_interval", "requested_milestone"]
    assert periodic[2]["requested_sim_times"] == [90., 90.]
    assert requests.requested_sim == []


def test_frequent_wall_and_simulation_cuts_preserve_original_physics():
    baseline = small_event_world(); baseline.advance(1., False)
    world = small_event_world()
    wall = [0.]
    def clock():
        wall[0] += 300.
        return wall[0]
    cuts = CutRequests(lambda *args: None, clock=clock, sim_interval=.3,
                       requested_sim=(.33, .8))
    make_observed_advance(RotatingSlotWorld).run(world, 1., False, cuts)
    assert len(cuts.captures) >= 5
    assert [e["actual_sim_time"] for e in cuts.captures
            if "simulation_interval" in e["reasons"]] == [.4, .8]
    assert semantic(world) == semantic(baseline)


def test_E09_cold_warm_cache_and_policy_table():
    world = mixed_world()
    payload = checkpoint(world)
    cold = restore_checkpoint(payload, expected_identity=identity(), cache_mode="cold").world
    warm = restore_checkpoint(payload, expected_identity=identity(), cache_mode="warm").world
    assert not cold._q_cache and not cold._energy_cache and not cold._reference_templates
    assert warm._q_cache and warm._energy_cache and warm._reference_templates
    # Keep the same R0 canonical representative rule after clearing memoization.
    assert cold.REFERENCE_PHASE_BINS == warm.REFERENCE_PHASE_BINS == 24
    # A cheap real continuation uses cached/uncached stable ground states.
    event = small_event_world()
    event._reference_templates[("outer", 0, 0)] = 25.
    payload = checkpoint(event, False)
    a = restore_checkpoint(payload, expected_identity=identity(), cache_mode="cold")
    b = restore_checkpoint(payload, expected_identity=identity(), cache_mode="warm")
    a.world.advance(1., False); b.world.advance(1., False)
    assert semantic(a.world) == semantic(b.world)


def test_E09_exact_energy_query_cold_warm_parity():
    world = RotatingSlotWorld()
    plan = make_stable_plan("query", "outer", 0, end_time=10.)
    expected = world._energy(plan, 0., .2)
    payload = checkpoint(world, False)
    cold = restore_checkpoint(payload, expected_identity=identity(), cache_mode="cold").world
    warm = restore_checkpoint(payload, expected_identity=identity(), cache_mode="warm").world
    assert cold._energy(plan, 0., .2) == warm._energy(plan, 0., .2) == expected
    assert cold.cache_hits == world.cache_hits
    assert warm.cache_hits == world.cache_hits + 1


def test_E02_effective_deterministic_driver_and_default_binding():
    world = RotatingSlotWorld(); driver = DeterministicDriver()
    world.decision_callback = driver
    observed = make_observed_advance(RotatingSlotWorld)
    saved = []
    def capture(w, cur, effective):
        if not saved:
            saved.append(capture_checkpoint(w, identity=identity(), cursor=cur, driver=effective))
            raise SuspendAtCut()
    with pytest.raises(SuspendAtCut):
        observed.run(world, 6., None, capture)
    restored = restore_checkpoint(saved[0], expected_identity=identity())
    observed.resume(restored)
    baseline = RotatingSlotWorld(); plain_driver = DeterministicDriver()
    baseline.advance(6., plain_driver)
    assert restored.driver.calls == plain_driver.calls == [0., 5.]
    assert restored.world.decision_callback is restored.driver
    assert semantic(restored.world) == semantic(baseline)


def test_A07_real_inner_hold_hook_on_off_and_interruption():
    """Actual core emergency refresh and energy path, without test replacements."""
    def scene():
        world = RotatingSlotWorld()
        world.seed_hold("real-inner-hold", "inner", 0, soc=.9)
        return world
    plain = scene(); plain.advance(.4, False)
    observed = make_observed_advance(RotatingSlotWorld)
    enabled = scene()
    cuts = []
    observed.run(enabled, .4, False, lambda w, c, d: cuts.append((w.time, c.phase)))
    interrupted = scene(); saved = []
    def capture(w, cur, driver):
        if w.time == .2 and cur.phase == "loop_start":
            saved.append(capture_checkpoint(w, identity=identity(), cursor=cur, driver=driver))
            raise SuspendAtCut()
    with pytest.raises(SuspendAtCut):
        observed.run(interrupted, .4, False, capture)
    restored = restore_checkpoint(saved[0], expected_identity=identity(), cache_mode="cold")
    observed.resume(restored)
    assert semantic(enabled) == semantic(plain) == semantic(restored.world)
    assert cuts == [(.2, "loop_start"), (.4, "loop_start"), (.4, "returned")]
    aircraft = plain.aircraft["real-inner-hold"]
    assert aircraft.soc < .9 and aircraft.emergency.q_e is None
    assert plain.practical_runtime_stats()["counters"]["soc_gate_skips"] > 0
    assert plain.auto_emergency is enabled.auto_emergency is restored.world.auto_emergency is True
    for world in (enabled, restored.world):
        actual = world.aircraft["real-inner-hold"]
        assert actual.soc.hex() == aircraft.soc.hex()
        assert actual.emergency.q_e is aircraft.emergency.q_e is None
        assert actual.trajectory_hash == aircraft.trajectory_hash
        assert world.events == plain.events and world.rejections == plain.rejections
        assert world.decision_times == plain.decision_times


def test_E07_stream_prefix_pending_same_time_replacement_and_tail(tmp_path):
    sink = BufferedStateJSONLSink(tmp_path / "old" / "states.jsonl")
    world = small_event_world(sink)
    observed = make_observed_advance(RotatingSlotWorld)
    saved = []
    def hook(w, c, d):
        if not saved and w.time >= .4:
            saved.append(capture_checkpoint(w, identity=identity(), cursor=c, driver=d))
            raise SuspendAtCut()
    with pytest.raises(SuspendAtCut):
        observed.run(world, 1., False, hook)
    payload = saved[0]
    # This is uncommitted post-checkpoint evidence, including same-time update.
    world._event("after_checkpoint"); world._record(); sink.flush(); sink.close()
    with sink.path.open("ab") as out:
        out.write(b'{"partial":')
    old_bytes = sink.path.read_bytes()
    restored = restore_checkpoint(payload, expected_identity=identity(), new_attempt_dir=tmp_path / "new")
    observed.resume(restored); restored.world.state_sink.close()
    assert sink.path.read_bytes() == old_bytes
    rows = [json.loads(x) for x in restored.world.state_sink.path.read_bytes().splitlines()]
    assert [r["time"] for r in rows] == sorted(set(r["time"] for r in rows))
    assert rows[-1]["time"] == 1.
    assert all(e["type"] != "after_checkpoint" for e in restored.world.events)
    assert len(rows) == restored.world._record_count
    direct_sink = BufferedStateJSONLSink(tmp_path / "direct" / "states.jsonl")
    direct = small_event_world(direct_sink); direct.advance(1., False); direct_sink.close()
    direct_rows = [json.loads(x) for x in direct_sink.path.read_bytes().splitlines()]
    for frames in (rows, direct_rows):
        for frame in frames:
            for key in ("cpu", "cache", "reference_counts"):
                frame.get("metrics", {}).pop(key, None)
    assert rows == direct_rows


def test_E07_prefix_corruption_refused_before_new_attempt(tmp_path):
    sink = BufferedStateJSONLSink(tmp_path / "old" / "states.jsonl")
    world = RotatingSlotWorld(state_sink=sink); world.advance(.4, False)
    payload = checkpoint(world, False); sink.close()
    data = bytearray(sink.path.read_bytes()); data[0] ^= 1; sink.path.write_bytes(data)
    target = tmp_path / "new"
    with pytest.raises(CheckpointError, match="prefix hash"):
        restore_checkpoint(payload, expected_identity=identity(), new_attempt_dir=target)
    assert not target.exists()


def test_E07_pending_corruption_and_wrong_prefix_count(tmp_path):
    sink = BufferedStateJSONLSink(tmp_path / "old" / "states.jsonl")
    world = RotatingSlotWorld(state_sink=sink); world.advance(.4, False)
    payload = checkpoint(world, False); sink.close()
    bad = deepcopy(payload)
    bad["state_stream"]["pending_sha256"] = "wrong"
    with pytest.raises(CheckpointError, match="pending frame"):
        restore_checkpoint(bad, expected_identity=identity(), new_attempt_dir=tmp_path / "new")
    bad = deepcopy(payload)
    bad["state_stream"]["committed_count"] += 1
    with pytest.raises(CheckpointError, match="count/latest"):
        restore_checkpoint(bad, expected_identity=identity(), new_attempt_dir=tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_E07_atomic_complete_retention(tmp_path, monkeypatch):
    import terminal_marl.core.rotating_slots.performance.checkpoint as module
    store = CheckpointStore(tmp_path)
    payload = checkpoint()
    first = store.save(payload, pinned=True)
    for i in range(11):
        store.save(payload, corpus=True)
    assert len(store.entries) == 9  # one pin and the most recent eight corpus cuts
    assert first.exists()
    index_before = store.index_path.read_bytes()
    original = module.os.replace
    def fail(source, target):
        if str(target).endswith("checkpoint-00000012.json"):
            raise OSError("injected atomic rename failure")
        return original(source, target)
    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        store.save(payload)
    assert store.index_path.read_bytes() == index_before
    assert not list(tmp_path.glob("*.tmp"))
    for entry in store.entries:
        load_checkpoint(tmp_path / entry["file"], expected_identity=identity())


def test_E08_hash_identity_schema_and_migration(tmp_path):
    path = tmp_path / "checkpoint.json"
    payload = checkpoint()
    write_checkpoint(path, payload)
    assert load_checkpoint(path, expected_identity=identity()) == payload
    document = json.loads(path.read_bytes()); document["payload"]["schema"] = "wrong"
    path.write_text(json.dumps(document))
    with pytest.raises(CheckpointError, match="content hash"):
        load_checkpoint(path, expected_identity=identity())
    for field in ("config_hash", "profile_hash", "physical_config_bytes_sha256", "effective_rules_hash",
                  "candidate_policy_hash", "numerical_policy_hash", "implementation_sha256"):
        with pytest.raises(CheckpointError, match="identity mismatch"):
            restore_checkpoint(payload, expected_identity=replace(identity(), **{field: "wrong"}))
    with pytest.raises(CheckpointError, match="schema/policy"):
        replace(identity(), policy_id="legacy")
    target = replace(identity(), implementation_sha256="qualified-second-implementation")
    register_equivalence_migration("test-qualified-pair", identity(), target)
    restored = restore_checkpoint(payload, expected_identity=target, migration_id="test-qualified-pair")
    assert restored.world.time == 0.
    with pytest.raises(CheckpointError, match="migration"):
        restore_checkpoint(payload, expected_identity=target, migration_id="trust_me")


def test_E08_archive_hash_corruption_is_rejected_before_attempt(tmp_path):
    world = mixed_world()
    payload = checkpoint(world)
    codec = StateCodec()
    state = codec.decode(payload["world"])
    first = next(iter(state["trajectories"].values()))
    first["trajectory_hash"] = "corrupted"
    payload["world"] = codec.encode(state)
    path = tmp_path / "corrupted-reenveloped.json"
    write_checkpoint(path, payload)
    payload = load_checkpoint(path, expected_identity=identity())
    with pytest.raises(CheckpointError, match="trajectory.*hash"):
        restore_checkpoint(payload, expected_identity=identity(), new_attempt_dir=tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_E10_replay_is_not_checkpoint():
    with pytest.raises(CheckpointError, match="checkpoint fields"):
        restore_checkpoint({"states": [{"time": 0.}]}, expected_identity=identity())
    payload = checkpoint(); payload["kind"] = "single_point_fixture"
    with pytest.raises(CheckpointError, match="not a resumable"):
        restore_checkpoint(payload, expected_identity=identity())


def test_measurements_clock_names_and_bounded_profile(tmp_path):
    wall, cpu = [1.], [10.]
    measurements = Measurements(wall_clock=lambda: wall[0], cpu_clock=lambda: cpu[0])
    with measurements.span("rollout"):
        wall[0] += 3.; cpu[0] += 2.
    report = measurements.snapshot()
    assert report["end_to_end_wall_s"] == 3.
    assert report["parent_process_cpu_s"] == 2.
    assert report["process_tree_cpu_s"] is None
    with pytest.raises(ValueError, match="benchmark"):
        with measurements.profile_batch(tmp_path / "forbidden.prof"):
            pass
    assert not (tmp_path / "forbidden.prof").exists()
    profile = Measurements("profile")
    with profile.profile_batch(tmp_path / "batch.prof"):
        sum(range(12))
    assert (tmp_path / "batch.prof").stat().st_size > 0


def test_heartbeat_emits_without_advancing_world(tmp_path):
    wall = [0.]
    world = RotatingSlotWorld()
    before = StateCodec().encode(world.records)
    heartbeat = PassiveHeartbeat(tmp_path / "heartbeat.json", clock=lambda: wall[0])
    heartbeat.publish(complete_sim_time=world.time, context={"phase": "reference", "pad": 2})
    wall[0] = 301.
    state = heartbeat.emit()
    assert state["status"] == "SLOW_OPERATION"
    assert state["last_complete_sim_time"] == 0.
    assert state["active_context"]["pad"] == 2
    assert StateCodec().encode(world.records) == before
    assert world.time == 0. and world.decision_times == []


def test_heartbeat_observer_failure_is_exposed(tmp_path):
    heartbeat = PassiveHeartbeat(tmp_path / "heartbeat.json")
    heartbeat.observer_error = OSError("injected output error")
    with pytest.raises(RuntimeError, match="observer failed") as error:
        heartbeat.check_health()
    assert isinstance(error.value.__cause__, OSError)
