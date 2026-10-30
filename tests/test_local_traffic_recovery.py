"""Real continuous trajectories for action-local certificates and standby."""
from dataclasses import replace
import time

import pytest

from tests.test_rotating_slots_latency_revalidation import setup, StatefulFinals
from tests.test_rotating_slots_planning_latency import Durations
from terminal_marl.core.rotating_slots import planning_latency as latency
from terminal_marl.core.rotating_slots.local_traffic_recovery import envelope_safe
from terminal_marl.core.rotating_slots.stage2_campaign import (
    save_checkpoint, restore_checkpoint, checkpoint_identity, digest)
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


def partial_fixture(*, cascade=False, repair_seconds=0., lead_seconds=30.):
    class Proposals(StatefulFinals):
        def __call__(self, port):
            if not cascade: return super().__call__(port)
            snapshot = port.build_planning_snapshot()
            self.calls.append(snapshot)
            self.plans = (
                generate_final("a", "inner", 0, 0, snapshot.time, config=snapshot.config).plan,
                generate_final("b", "inner", 6, 3, snapshot.time, config=snapshot.config).plan,
                generate_transfer("c", "middle", 0, 0, snapshot.time,
                                  durations=(60.,), config=snapshot.config).plan)
            evaluation = port.evaluate_submission(self.plans)
            assert evaluation.accepted, evaluation
            assert port.commit_validated_prefix(evaluation).status == "DEFERRED"
    callback = Proposals(pairs=(("a", 0), ("b", 3)))
    world = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    for aid, ring, slot in (("a", "inner", 0), ("b", "inner", 6), ("e", "inner", 9)):
        world.seed_hold(aid, ring, slot, .9)
    if cascade: world.seed_hold("c", "middle", 0, .9)
    clock = Durations(5. if lead_seconds == 20. else 10., 3.)
    scheduler = latency.LatencyAwareScheduler(callback, clock=clock, traffic_policy=latency.LOCAL_TRAFFIC_POLICY,
        lead_seconds=lead_seconds)
    world.decision_callback = scheduler
    changed = []
    def occupy(snapshot, port):
        if snapshot.time+1.e-8 < lead_seconds-10. or changed: return
        aircraft = world.aircraft["e"]
        world.aircraft["e"] = replace(aircraft, emergency=replace(aircraft.emergency, eligible=True, declared=True))
        plan = generate_final("e", "inner", 9, 0, world.time, emergency=True, config=world.config).plan
        result = port.submit((plan,))
        assert result.accepted, result
        changed.append(plan)
        clock.durations = iter((0., 2., 0., 0., repair_seconds, 0.))
    world.admission_callback = occupy
    return world, scheduler, callback, clock, changed


def test_real_late_outer_arrival_keeps_original_final_certificate_and_readiness():
    world, _, callback, clock, admitted = setup(arrivals=(19.5,))
    scheduler = latency.LatencyAwareScheduler(callback, clock=clock, traffic_policy=latency.LOCAL_TRAFFIC_POLICY,
        lead_seconds=20.)
    world.decision_callback = scheduler
    world.advance(18.)
    ready = scheduler.records[0]["ready_time"]
    world.advance(20.01)
    record = scheduler.records[0]
    assert admitted == ["new0"]
    assert record["status"] == "COMMITTED", record
    assert record["ready_time"] == ready and record["revalidation_count"] == 0
    assert record["guard_local_recovery"][-1]["envelope_proofs"] >= 1
    assert record["guard_local_recovery"][-1]["affected_action_hashes"] == []
    assert world.active["a"].trajectory_hash == callback.plans[0].trajectory_hash


@pytest.mark.parametrize("arrivals", ((18.,), (18., 19.)))
def test_checkpoint_after_unrelated_outer_guard_preserves_ready_final_and_archive(tmp_path, arrivals):
    from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink
    world, _, callback, clock, admitted = setup(arrivals=arrivals)
    archive = tmp_path/"archived.states.jsonl"
    sink = BufferedStateJSONLSink(archive)
    for frame in world.records: sink(frame)
    world.state_sink = sink
    scheduler = latency.LatencyAwareScheduler(callback, clock=clock, lead_seconds=20.,
        traffic_policy=latency.LOCAL_TRAFFIC_POLICY)
    world.decision_callback = scheduler
    world.advance(17.)
    original_ready = scheduler.records[0]["ready_time"]
    original_hash = callback.plans[0].trajectory_hash
    world.advance(19.25)
    assert admitted == ["new"+str(index) for index in range(len(arrivals))]
    assert scheduler.pending["stage"] == "PREPARED"
    record = scheduler.records[0]
    assert record["ready_time"] == original_ready and record["revalidation_count"] == 0
    assert record["guard_local_recovery"][-1]["accepted"]
    token = scheduler.pending["token"]
    assert world._planning_latency_packets[token]["local_traffic_source"].time+1.e-8 >= arrivals[-1]
    identity = checkpoint_identity(dict(source_hash="test", base_config_hash=world.config.config_hash,
        overlay_sha256="test"), digest(world.manifest.to_dict()))
    checkpoint = tmp_path/"after_unrelated_guard.checkpoint.json"
    save_checkpoint(checkpoint, world, scheduler, identity)
    sink.close()
    archived_bytes = archive.read_bytes()
    restored_scheduler = latency.LatencyAwareScheduler(StatefulFinals(), clock=Durations(), lead_seconds=20.,
        traffic_policy=latency.LOCAL_TRAFFIC_POLICY)
    restored = restore_checkpoint(checkpoint, restored_scheduler, identity,
        stream_copy=tmp_path/"restored.states.jsonl")
    try:
        assert restored_scheduler.records[0]["ready_time"] == original_ready
        assert restored._planning_latency_packets[token]["plans"][0].trajectory_hash == original_hash
        restored.advance(20.01, restored_scheduler)
        assert restored_scheduler.records[0]["status"] == "COMMITTED"
        assert restored_scheduler.records[0]["ready_time"] == original_ready
        assert restored.active["a"].trajectory_hash == original_hash
        assert archive.read_bytes() == archived_bytes
    finally:
        restored.state_sink.close()


def test_true_partial_resource_conflict_delivers_only_jointly_certified_standby():
    world, scheduler, callback, _, changed = partial_fixture()
    world.advance(30.01)
    record = scheduler.records[0]
    assert changed and record["status"] == "COMMITTED", record
    assert set(world.active) == {"b", "e"}
    assert world.aircraft["a"].status == "HOLD_INNER"
    assert record["submitted"] == 1
    assert record["standby_ready_time"] <= record["ready_deadline"]
    assert record["retained_trajectory_hashes"] == [callback.plans[1].trajectory_hash]
    event = record["revalidations"][0]["local_recovery"]
    assert event["affected_action_hashes"] == [callback.plans[0].trajectory_hash]
    assert event["withdrawn_action_reasons"][callback.plans[0].trajectory_hash] == "RESOURCE_DEPENDENCY_CHANGED"
    assert event["pairs_checked"] > 0  # includes b versus a's restored hold.
    assert record["recovery_delivery"] == "standby"
    attempt = record["repair"]["repair_attempts"][0]
    details = attempt["validation_details"]
    assert {r["aircraft_id"] for r in details["blocking_reservations"]} == {"e"}
    assert details["blocking_reservations"][0]["plan_hash"] == changed[0].trajectory_hash
    assert {r["trajectory_hash"] for r in details["proposed_actions"]} == {
        callback.plans[0].trajectory_hash, callback.plans[1].trajectory_hash}


def test_removed_final_recursively_withdraws_its_slot_release_dependent_transfer():
    world, scheduler, callback, _, _ = partial_fixture(cascade=True)
    world.advance(30.01)
    record = scheduler.records[0]
    assert record["status"] == "COMMITTED", record
    a, b, c = callback.plans
    event = record["revalidations"][0]["local_recovery"]
    assert set(event["affected_action_hashes"]) == {a.trajectory_hash, c.trajectory_hash}
    assert (a.trajectory_hash, c.trajectory_hash) in event["recursive_release_edges"]
    assert world.aircraft["a"].status == "HOLD_INNER" and world.aircraft["c"].status == "HOLD_MIDDLE"
    assert set(world.active) == {"b", "e"}
    assert world.active["b"].trajectory_hash == b.trajectory_hash


def test_compressed_repair_keeps_timely_ready_standby_and_checkpoint(tmp_path):
    world, scheduler, callback, _, _ = partial_fixture(repair_seconds=15., lead_seconds=20.)
    world.advance(13.)
    assert scheduler.pending["stage"] == "REPAIRING", scheduler.pending
    packet = world._planning_latency_packets[scheduler.pending["token"]]
    assert packet["ready_at"] <= 19.
    assert tuple(packet["plans"]) == (callback.plans[1],)
    identity = checkpoint_identity(dict(source_hash="test", base_config_hash=world.config.config_hash,
        overlay_sha256="test"), digest(world.manifest.to_dict()))
    path = tmp_path/"standby_repair.checkpoint.json"
    save_checkpoint(path, world, scheduler, identity)
    restored_scheduler = latency.LatencyAwareScheduler(StatefulFinals(), clock=Durations(),
        traffic_policy=latency.LOCAL_TRAFFIC_POLICY, lead_seconds=20.)
    restored = restore_checkpoint(path, restored_scheduler, identity)
    restored.advance(20.01, restored_scheduler)
    record = restored_scheduler.records[0]
    assert record["status"] == "COMMITTED" and record["recovery_delivery"] == "standby", record
    assert record["ready_time"] == packet["ready_at"]
    assert record["repair_work_s"] >= 15.
    assert set(restored.active) == {"b", "e"}


def test_same_ring_label_cannot_prove_separation_for_actual_inward_future_trajectory():
    world = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_hold("a", "outer", 0, .9)
    world.seed_hold("b", "middle", 0, .9)
    plan = generate_transfer("a", "outer", 0, 0, 30., config=world.config).plan
    first = world._traffic(world.aircraft["a"], {"a": plan})
    second = world._traffic(world.aircraft["b"])
    assert not envelope_safe(first, second, 30., plan.end_time+10., world.config)


def test_unknown_pair_is_withdrawn_and_never_executed(monkeypatch):
    from terminal_marl.core.rotating_slots.models import ValidationResult
    from terminal_marl.core.rotating_slots import world as world_module
    world, scheduler, _, _, _ = partial_fixture()
    world.advance(19.)
    original = world_module.validate_separation
    def unknown_rollback_pair(first, second, *args, **kwargs):
        if (hasattr(first, "plans") and hasattr(second, "plans")
                and {first.aircraft_id, second.aircraft_id} == {"a", "b"}):
            return ValidationResult(False, "UNKNOWN", "PAIR_UNRESOLVED")
        return original(first, second, *args, **kwargs)
    monkeypatch.setattr(world_module, "validate_separation", unknown_rollback_pair)
    world.advance(30.01)
    record = scheduler.records[0]
    assert record["status"] == "REJECTED" and record["reason"] == "PAIR_UNRESOLVED", record
    assert "a" not in world.active and "b" not in world.active


def test_natural_transfer_completion_keeps_exact_target_interval_certificate():
    world = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_hold("a", "inner", 0, .9)
    world.seed_hold("b", "outer", 18, .9)
    transfer = generate_transfer("b", "outer", 18, 14, 0., config=world.config).plan
    assert world.submit((transfer,)).accepted
    timing = latency.PlanningTiming(0., 60., 59.)
    final = generate_final("a", "inner", 0, 0, 60., config=world.config).plan
    result, token = latency.prepare_submission(world, (final,), timing,
        traffic_policy=latency.LOCAL_TRAFFIC_POLICY)
    assert result.accepted, result
    world.advance(transfer.end_time+1., decision_callback=False)
    guarded = latency.guard_submission(world, token)
    assert guarded.accepted, guarded
    assert guarded.details["affected_aircraft_ids"] == []
    assert guarded.details["reused_pairs"] >= 1 and guarded.details["envelope_proofs"] == 0


def test_extended_horizon_requires_proof_for_unchanged_pair(monkeypatch):
    from terminal_marl.core.rotating_slots import local_traffic_recovery as recovery
    world = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_hold("a", "outer", 0, .9)
    world.seed_hold("b", "outer", 18, .9)
    world.seed_hold("c", "outer", 9, .9)
    timing = latency.PlanningTiming(0., 30., 29.)
    proposal = generate_transfer("a", "outer", 0, 0, 30., config=world.config).plan
    result, token = latency.prepare_submission(world, (proposal,), timing,
        traffic_policy=latency.LOCAL_TRAFFIC_POLICY)
    assert result.accepted, result
    world.advance(10., decision_callback=False)
    old_end = world._planning_latency_packets[token]["pair_certificates"][("a", "b")][3]
    later = generate_transfer("c", "outer", 9, 7, 100., config=world.config).plan
    assert later is not None and world.submit((later,)).accepted
    original = recovery.envelope_safe
    calls = []
    def observe(first, second, start, end, config):
        calls.append(({first.aircraft_id, second.aircraft_id}, start, end))
        if {first.aircraft_id, second.aircraft_id} == {"a", "b"}: return False
        return original(first, second, start, end, config)
    monkeypatch.setattr(recovery, "envelope_safe", observe)
    guarded = latency.guard_submission(world, token)
    assert not guarded.accepted and guarded.reason == "TRAFFIC_REVALIDATION_REQUIRED"
    assert guarded.details["affected_aircraft_ids"] == ["c"]
    assert guarded.details["horizon_extension_pairs"] >= 1
    assert any(ids == {"a", "b"} and end > old_end for ids, _, end in calls)


def test_changed_again_rechecks_ready_standby_before_publication():
    world, scheduler, callback, clock, _ = partial_fixture(repair_seconds=15.)
    world.seed_hold("f", "inner", 3, .9)
    original = world.admission_callback
    changed_again = []
    def second_change(snapshot, port):
        original(snapshot, port)
        if snapshot.time+1.e-8 < 25. or changed_again: return
        aircraft = world.aircraft["f"]
        world.aircraft["f"] = replace(aircraft, emergency=replace(aircraft.emergency, eligible=True, declared=True))
        plan = generate_final("f", "inner", 3, 3, world.time, emergency=True, config=world.config).plan
        result = port.submit((plan,))
        assert result.accepted, result
        changed_again.append(plan)
        clock.durations = iter((0., 1., 0.))
    world.admission_callback = second_change
    world.advance(30.01)
    record = scheduler.records[0]
    assert changed_again and record["revalidation_count"] >= 2, record
    assert "a" not in world.active and "b" not in world.active
    world.advance(40.)
    assert record["status"] == "REJECTED", record
    assert "a" not in world.active and "b" not in world.active
    assert world.active["f"] == changed_again[0]


def test_ambiguous_original_pair_validation_begins_only_after_standby_is_ready(monkeypatch):
    from terminal_marl.core.rotating_slots import local_traffic_recovery as recovery
    from terminal_marl.core.rotating_slots import world as world_module
    from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
    from terminal_marl.core.rotating_slots.models import ValidationResult
    class Mixed(StatefulFinals):
        def __call__(self, port):
            snapshot = port.build_planning_snapshot(); self.calls.append(snapshot)
            self.plans = (generate_transfer("a", "outer", 0, 0, snapshot.time, config=snapshot.config).plan,
                generate_final("b", "inner", 6, 3, snapshot.time, config=snapshot.config).plan)
            evaluated = port.evaluate_submission(self.plans)
            assert evaluated.accepted, evaluated
            port.commit_validated_prefix(evaluated)
    world = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID,
        manifest=ArrivalManifest((ArrivalRequest("new", 10., 2, .9, "standard"),)))
    world.seed_hold("a", "outer", 0, .9); world.seed_hold("b", "inner", 6, .9)
    callback, clock = Mixed(), Durations(5., 3.)
    scheduler = latency.LatencyAwareScheduler(callback, clock=clock, lead_seconds=20.,
        traffic_policy=latency.LOCAL_TRAFFIC_POLICY)
    world.decision_callback = scheduler
    def admit(snapshot, port):
        if "new" in world.aircraft or "new" not in port.requests: return
        assert port.admit("new").accepted
        clock.durations = iter((0., 1., 0., 0., 15., 0.))
    world.admission_callback = admit
    original_envelope = recovery.envelope_safe
    def conservative_filter(first, second, *args):
        if {first.aircraft_id, second.aircraft_id} == {"a", "new"}: return False
        return original_envelope(first, second, *args)
    monkeypatch.setattr(recovery, "envelope_safe", conservative_filter)
    original_validator = world_module.validate_separation
    observations = []
    def original_pair(first, second, *args, **kwargs):
        if (hasattr(first, "plans") and hasattr(second, "plans")
                and {first.aircraft_id, second.aircraft_id} == {"a", "new"}
                and any(p.aircraft_id == "a" for p in (*first.plans, *second.plans))):
            packet = world._planning_latency_packets[scheduler.pending["token"]]
            observations.append((world.time, packet["ready_at"], tuple(packet["plans"])))
            assert packet["ready_at"] is not None and packet["ready_at"] <= world.time
            return ValidationResult(False, "UNKNOWN", "AMBIGUOUS_REPAIR_TIMEOUT")
        return original_validator(first, second, *args, **kwargs)
    monkeypatch.setattr(world_module, "validate_separation", original_pair)
    world.advance(20.01)
    record = scheduler.records[0]
    assert observations and observations[0][2] == (callback.plans[1],)
    assert record["status"] == "COMMITTED" and record["recovery_delivery"] == "standby", record
    assert "a" not in world.active and world.active["b"] == callback.plans[1]
    assert record["repair_work_s"] >= 15.


def test_recovery_scores_all_options_with_same_fresh_service_objective():
    from terminal_marl.core.rotating_slots.local_traffic_recovery import _current_scores
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider
    world = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_hold("a", "inner", 0, .9); world.seed_hold("b", "inner", 6, .9)
    timing = latency.PlanningTiming(0., 20., 19.)
    a = generate_final("a", "inner", 0, 0, 20., config=world.config).plan
    b = generate_final("b", "inner", 6, 3, 20., config=world.config).plan
    result, token = latency.prepare_submission(world, (a, b), timing,
        traffic_policy=latency.LOCAL_TRAFFIC_POLICY)
    assert result.accepted, result
    provider = CandidateProvider(planning_profile="downstream_service_v8", evaluation_profile="service_opportunity_v1")
    packet = world._planning_latency_packets[token]
    scores, evidence = _current_scores(world, packet, provider, ((b,), (a, b), (b,)), time.perf_counter()+10.)
    assert scores is not None, evidence
    assert scores[0] == scores[2] and all(len(score) == 6 for score in scores)
    assert evidence["scope"] == "current_executable_prefixes"
    assert evidence["replan_ids"] == ("a", "b")
    assert evidence["objective_schema"] == ("F", "-L", "R", "G_service", "E", "tau")
    assert evidence["frame_id"] and evidence["snapshot_id"] != packet["source"].version


def test_low_soc_withdrawn_hold_cannot_get_standby_energy_certificate():
    world = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.rescue_rule_profile = "rescue_need_v7"
    world.seed_hold("a", "inner", 0, .9); world.seed_hold("b", "inner", 6, .9)
    timing = latency.PlanningTiming(0., 20., 19.)
    plans = (generate_final("a", "inner", 0, 0, 20., config=world.config).plan,
             generate_final("b", "inner", 6, 3, 20., config=world.config).plan)
    result, token = latency.prepare_submission(world, plans, timing,
        traffic_policy=latency.LOCAL_TRAFFIC_POLICY)
    assert result.accepted, result
    world.advance(10., decision_callback=False)
    world.aircraft["a"] = replace(world.aircraft["a"], soc=.03)
    result = latency.revalidate_submission(world, token, max_work_s=8.)
    assert not result.accepted and result.reason == "FALLBACK_ENERGY_INSUFFICIENT", result
    row = result.details["fallback_energy"][0]
    assert row["interval"] == (20., 40.)
    assert row["start_soc_lower"] > 0 and row["end_soc_lower"] <= 0
    assert row["precision"] == "fine"
    assert not world.active and not world.soft
    assert len(world._planning_latency_packets) == 1


def test_repeated_fallback_soc_basis_change_rechecks_energy_without_withdrawing_unrelated_action():
    world, scheduler, callback, _, _ = partial_fixture(repair_seconds=15.)
    world.rescue_rule_profile = "rescue_need_v7"
    world.advance(23.)
    assert scheduler.pending["stage"] == "REPAIRING"
    token = scheduler.pending["token"]
    packet = world._planning_latency_packets[token]
    assert packet["fallback_aircraft_ids"] == ("a",)
    assert packet["ready_at"] is not None
    world.aircraft["a"] = replace(world.aircraft["a"], soc=.03)
    guarded = latency.guard_submission(world, token)
    assert not guarded.accepted and guarded.reason == "TRAFFIC_REVALIDATION_REQUIRED"
    assert guarded.details["fallback_revalidation_required"]
    assert guarded.details["affected_action_hashes"] == []
    assert guarded.details["retained_action_hashes"] == [callback.plans[1].trajectory_hash]
    result = latency.revalidate_submission(world, token, max_work_s=5.)
    assert not result.accepted and result.reason == "FALLBACK_ENERGY_INSUFFICIENT", result
    assert result.details["fallback_energy"][0]["aircraft_id"] == "a"
    assert "b" not in world.active


def test_partial_repair_promotion_preserves_remaining_fallback_applicability(monkeypatch):
    from terminal_marl.core.rotating_slots import local_traffic_recovery as recovery
    from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
    world = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID,
        manifest=ArrivalManifest((ArrivalRequest("new", 10., 2, .9, "standard"),)))
    world.rescue_rule_profile = "rescue_need_v7"
    for aid, ring, slot in (("a", "outer", 0), ("b", "inner", 6), ("c", "outer", 18)):
        world.seed_hold(aid, ring, slot, .9)
    timing = latency.PlanningTiming(0., 20., 19.)
    plans = (generate_transfer("a", "outer", 0, 0, 20., config=world.config).plan,
        generate_final("b", "inner", 6, 3, 20., config=world.config).plan,
        generate_transfer("c", "outer", 18, 14, 20., config=world.config).plan)
    result, token = latency.prepare_submission(world, plans, timing,
        traffic_policy=latency.LOCAL_TRAFFIC_POLICY)
    assert result.accepted, result
    world.advance(10., decision_callback=False)
    assert world.admit("new").accepted
    original = recovery.envelope_safe
    def inconclusive(first, second, *args):
        ids = {first.aircraft_id, second.aircraft_id}
        if ids in ({"a", "new"}, {"c", "new"}): return False
        return original(first, second, *args)
    monkeypatch.setattr(recovery, "envelope_safe", inconclusive)
    result = latency.revalidate_submission(world, token, max_work_s=8.)
    assert result.accepted, result
    world.advance(12., decision_callback=False)
    assert latency.complete_revalidation(world, token, result.details["attempt"], promote=True).accepted
    assert latency.mark_submission_ready(world, token).accepted
    # Synthetic score vectors select a particular valid partial repair; the
    # separate adapter test checks actual same-frame six-objective evaluation.
    def select_partial(world, packet, provider, alternatives, deadline):
        values = [(0, 0, 0, 2+i, 0, 0) for i in range(len(alternatives))]
        values[1] = (0, 0, 0, 0, 0, 0)
        return values, dict(scope="controlled_promotion_fixture", scores=values)
    monkeypatch.setattr(recovery, "_current_scores", select_partial)
    repair = recovery.repair_submission(world, token, None, max_work_s=6.)
    assert repair.accepted and repair.status == "REPAIRING", repair
    world.advance(15., decision_callback=False)
    promoted = recovery.complete_repair(world, token)
    assert promoted.accepted and promoted.status == "REPAIRED", promoted
    packet = world._planning_latency_packets[token]
    assert {p.aircraft_id for p in packet["plans"]} == {"a", "b"}
    assert packet["fallback_aircraft_ids"] == ("c",)
    assert packet["fallback_energy"][0]["aircraft_id"] == "c"
    world.aircraft["c"] = replace(world.aircraft["c"], soc=.03)
    changed = latency.guard_submission(world, token)
    assert not changed.accepted and changed.reason == "TRAFFIC_REVALIDATION_REQUIRED"
    assert changed.details["fallback_revalidation_required"]
    assert changed.details["affected_action_hashes"] == []
