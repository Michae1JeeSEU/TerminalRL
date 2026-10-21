"""Rejection-only canonical screening and high-target Stage2 regressions."""
import pytest

from terminal_marl.core.rotating_slots import trajectory
from terminal_marl.core.rotating_slots.energy import practical_numeric_stats
from terminal_marl.core.rotating_slots.stage2_campaign import busy_fixture, frozen_manifest
from terminal_marl.core.rotating_slots.stage2_planning import PlanningBudget, ReferenceProvider, _node
from terminal_marl.core.rotating_slots.world import evaluate_planning_snapshot


@pytest.fixture(scope="module")
def busy():
    world = busy_fixture(frozen_manifest())
    return world, world.build_planning_snapshot()


@pytest.mark.parametrize(("ring", "target", "duration"), [
    ("outer", 1, 40), ("outer", 26, 80), ("outer", 27, 45),
    ("middle", 17, 40), ("inner", 5, 35),
])
def test_busy_high_target_witness_survives_and_uses_real_prefix_checker(busy, ring, target, duration):
    world, snapshot = busy
    aircraft = snapshot.aircraft[f"initial_{ring}_00"]
    screen, _ = trajectory.screen_normal_templates(ring, aircraft.slot, snapshot.time, snapshot.config)
    assert screen[(target, duration)] is None  # UNKNOWN, never a certificate.
    generate = trajectory.generate_final if ring == "inner" else trajectory.generate_transfer
    result = generate(aircraft.id, ring, aircraft.slot, target, snapshot.time,
        source_version=aircraft.version, config=snapshot.config, durations=(duration,))
    assert result.accepted
    checked = evaluate_planning_snapshot(snapshot, (result.plan,))
    assert checked.accepted and not checked.refinement_required
    assert world.build_planning_snapshot().version == snapshot.version


@pytest.mark.parametrize(("ring", "slot", "target", "start"), [
    ("outer", 18, 14, 0.), ("outer", 36, 27, 75.),
    ("middle", 9, 6, 15.), ("middle", 27, 17, 125.),
])
def test_multiple_source_slots_and_epochs_preserve_feasible_transfers(ring, slot, target, start):
    screen, _ = trajectory.screen_normal_templates(ring, slot, start)
    assert screen[(target, 40)] is None
    assert trajectory.generate_transfer("single", ring, slot, target, start, durations=(40,)).accepted


@pytest.mark.parametrize(("slot", "pad", "start"), [(0, 5, 0.), (3, 0, 45.), (17, 4, 125.)])
def test_final_phase_screen_keeps_first_exactly_feasible_duration(slot, pad, start):
    result = trajectory.generate_final("single", "inner", slot, pad, start)
    assert result.accepted
    approach_duration = int(round(result.plan.segments[0].duration))
    screen, _ = trajectory.screen_normal_templates("inner", slot, start)
    assert screen[(pad, approach_duration)] is None


def test_screen_rejects_both_branches_with_original_validator_witness_reason():
    screen, _ = trajectory.screen_normal_templates("outer", 0, 0.)
    result = trajectory.generate_transfer("single", "outer", 0, 1, 0., durations=(20,))
    assert not result.accepted
    assert screen[(1, 20)] == tuple(row["reason"] for row in result.details["attempts"])
    assert all(reason == "HORIZONTAL_JERK" for reason in screen[(1, 20)])


def test_screen_builds_no_full_plans_and_is_not_a_feasibility_certificate(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("screen constructed a full TrajectoryPlan")
    monkeypatch.setattr(trajectory, "TrajectoryPlan", forbidden)
    screen, _ = trajectory.screen_normal_templates("outer", 11, 22.)
    assert any(value is None for value in screen.values())
    assert any(value is not None for value in screen.values())


def test_reference_screen_counts_are_separate_from_original_construction_budget(busy):
    _, snapshot = busy
    aircraft = snapshot.aircraft["initial_outer_00"]
    budget = PlanningBudget(max_per_aircraft=4, max_attempts=4)
    before = practical_numeric_stats().get("transfer_feasibility_calls", 0)
    ReferenceProvider().get(snapshot, aircraft, budget=budget)
    actual = practical_numeric_stats().get("transfer_feasibility_calls", 0)-before
    assert 0 < actual == sum(budget.attempts.values()) <= 4
    assert budget.screening["templates"] > actual
    assert budget.screening["witness_rejected"] > 0
    assert "geometry_template_screen" in budget.timings


def test_offset_enumeration_yields_for_round_robin_before_certification(busy):
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, EvaluationContext
    _, snapshot = busy
    aircraft = snapshot.aircraft["initial_outer_00"]
    budget = PlanningBudget()
    provider = CandidateProvider()
    paths = provider._paths(snapshot, aircraft, _node(snapshot, aircraft), budget, EvaluationContext(snapshot))
    assert next(paths) == ("SCREEN", None, None)
    assert not budget.attempts
    assert budget.screening["grid_calls"] == 1
    assert 0 < budget.screening["descriptors_queued"] <= 30


def test_stage_lower_bound_is_below_valid_remaining_chains_at_multiple_phases(busy):
    import random
    from terminal_marl.core.rotating_slots.geometry import ring_config
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider
    _, snapshot = busy
    rng = random.Random(71)
    assert [CandidateProvider._remaining_lower(snapshot, ring) for ring in ("outer", "middle", "inner")] == [123.5, 83.5, 43.5]
    for ring, slot, start in (("outer", 0, 0.), ("outer", 18, 75.),
                              ("middle", 9, 15.), ("middle", 27, 125.), ("inner", 17, 55.)):
        current, source_slot, clock = ring, slot, start
        bound = CandidateProvider._remaining_lower(snapshot, ring)
        while current != "inner":
            next_ring = {"outer": "middle", "middle": "inner"}[current]
            source_count = ring_config(current, snapshot.config).slot_count
            target_count = ring_config(next_ring, snapshot.config).slot_count
            target = round(source_slot*target_count/source_count) % target_count
            result = trajectory.generate_transfer("lower_bound", current, source_slot, target, clock)
            assert result.accepted
            clock, current, source_slot = result.plan.end_time, next_ring, target
        pads=list(range(6)); rng.shuffle(pads)
        results=[trajectory.generate_final("lower_bound", current, source_slot, pad, clock) for pad in pads]
        result=next((result for result in results if result.accepted),None)
        assert result is not None
        assert result.plan.end_time-start+1.e-8 >= bound


def test_fixed_background_rejections_do_not_consume_accepted_first_target_cap(monkeypatch):
    from collections import Counter
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, EvaluationContext
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    world = RotatingSlotWorld()
    world.seed_hold("inner", "inner", 0, .9)
    world.seed_service("busy", 5, remaining=50.)
    snapshot = world.build_planning_snapshot()
    context = EvaluationContext(snapshot)
    provider = CandidateProvider()
    # Isolate this first target so other pads cannot fill the eight-column cap
    # before the delayed pad5 stream is reached. All offered trajectories and
    # background checks still use the real canonical generator and evaluator.
    original_cursors=provider._offset_cursors
    def only_pad5(*args,**kwargs):
        yield from (cursor for cursor in original_cursors(*args,**kwargs) if cursor["target"]==5)
    monkeypatch.setattr(provider,"_offset_cursors",only_pad5)
    checks = []
    original = context.check_single
    def record(column):
        result = original(column)
        if column.first_target == (None, None, 5):
            checks.append((column.plans[0].start_time, result.accepted, result.reason))
        return result
    context.check_single = record
    budget = PlanningBudget()
    catalog = provider.get_columns(snapshot, budget, context)
    assert sum(not accepted for _, accepted, _ in checks) >= 2
    accepted = [c for c in catalog.columns["inner"] if c.kind == "MOVE"]
    pad5 = [c for c in accepted if c.first_target == (None, None, 5)]
    assert pad5 and all(c.plans[0].start_time > 0 for c in pad5), (checks,dict(budget.attempts),provider.aligned_starts)
    assert max(Counter(c.first_target for c in accepted).values()) <= 2
    assert all(count <= 64 for count in budget.attempts.values())
    assert any(reason == "RESOURCE_CONFLICT" for _, ok, reason in checks if not ok)
    assert all(evaluate_planning_snapshot(snapshot, (c.plans[0],)).accepted for c in pad5)


@pytest.fixture(scope="module")
def reference_world():
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    world=RotatingSlotWorld()
    world.configure_practical_runtime(True)
    for ring,slot in (("outer",0),("middle",9),("inner",17)):
        world.seed_hold(ring,ring,slot,.9)
    return world


def test_reference_first_seeds_cover_every_ring_with_original_attempt_limits(monkeypatch,reference_world):
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, EvaluationContext
    snapshot=reference_world.build_planning_snapshot()
    # Make this an attempt-budget regression independent of host CPU speed.
    monkeypatch.setattr(PlanningBudget,"elapsed",property(lambda self:0.))
    provider=CandidateProvider()
    monkeypatch.setattr(provider,"_paths",lambda *args,**kwargs:iter(()))
    budget=PlanningBudget()
    context=EvaluationContext(snapshot)
    catalog=provider.get_columns(snapshot,budget,context)
    for ring,depth in (("outer",3),("middle",2),("inner",1)):
        moving=[c for c in catalog.columns[ring] if c.kind=="MOVE"]
        full=next(c for c in moving if c.plans[-1].pad_id is not None)
        assert len(full.plans)==depth and len(moving)<=8
        assert context.check_single(full).accepted
        assert evaluate_planning_snapshot(snapshot,(full.plans[0],)).accepted
        assert 0<budget.attempts[ring]<=64
    assert sum(budget.attempts.values())<=2048
    assert "reference_seed_generation" in budget.timings


def test_reference_seed_rejection_keeps_full_reference_for_slack(monkeypatch):
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, EvaluationContext
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    monkeypatch.setattr(PlanningBudget,"elapsed",property(lambda self:0.))
    world=RotatingSlotWorld();world.seed_hold("inner","inner",0,.9)
    snapshot=world.build_planning_snapshot()
    references=ReferenceProvider()
    reference=references.get(snapshot,snapshot.aircraft["inner"],budget=PlanningBudget())
    assert reference.status=="KNOWN"
    world.seed_service("busy",reference.plans[-1].pad_id,remaining=500.)
    snapshot=world.build_planning_snapshot();context=EvaluationContext(snapshot)
    checked=[];original=context.check_single
    def record(column):
        result=original(column);checked.append(result);return result
    monkeypatch.setattr(context,"check_single",record)
    provider=CandidateProvider(references)
    # This test isolates the complete reference's rejection and slack value;
    # the separate safe-progress pass may legitimately use another free pad.
    monkeypatch.setattr(provider,"_progress_steps",lambda *args,**kwargs:iter(()))
    monkeypatch.setattr(provider,"_paths",lambda *args,**kwargs:iter(()))
    catalog=provider.get_columns(snapshot,PlanningBudget(),context)
    assert checked and any(not r.accepted and r.reason=="RESOURCE_CONFLICT" for r in checked)
    assert not any(c.kind=="MOVE" for c in catalog.columns["inner"])
    node=_node(snapshot,snapshot.aircraft["inner"])
    assert references.cache[(snapshot.config.config_hash,node.ring,node.slot,node.time,node.version,"inner")]==reference
    from terminal_marl.core.rotating_slots.stage2_planning import energy_slack
    assert catalog.slacks["inner"]==energy_slack(snapshot,snapshot.aircraft["inner"],reference)
    assert catalog.slacks["inner"][0]!=1


def test_reference_seed_completes_outer_before_next_aircraft_and_interleaves_rings(monkeypatch):
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    monkeypatch.setattr(PlanningBudget,"elapsed",property(lambda self:0.))
    world=RotatingSlotWorld();world.configure_practical_runtime(True)
    for name,ring,slot in (("o0","outer",0),("o1","outer",18),
                           ("m0","middle",9),("i0","inner",0),("i1","inner",9)):
        world.seed_hold(name,ring,slot,.9)
    snapshot=world.build_planning_snapshot();provider=CandidateProvider()
    monkeypatch.setattr(provider,"_paths",lambda *args,**kwargs:iter(()))
    events=[];original=provider.references.build_steps
    def record(snapshot,a,node,budget):
        events.append(("start",a.id))
        result=yield from original(snapshot,a,node,budget)
        events.append(("finish",a.id,result.status))
        return result
    monkeypatch.setattr(provider.references,"build_steps",record)
    catalog=provider.get_columns(snapshot,PlanningBudget())
    # Inspect actual completed normal trajectories as well as visit order.
    assert [event[1] for event in events if event[0]=="start"][:5]==["o0","m0","i0","o1","i1"]
    assert events.index(("finish","o0","KNOWN"))<events.index(("start","i0"))<events.index(("start","o1"))
    assert any(len(c.plans)==3 and c.plans[-1].pad_id is not None for c in catalog.columns["o0"] if c.kind=="MOVE")


@pytest.mark.parametrize(("ring","soc","peak"),[("outer",.001,0.),("inner",.25,0.),("middle",.9,121.)])
def test_reference_seed_replays_transfer_final_reserve_and_peak(monkeypatch,reference_world,ring,soc,peak):
    from dataclasses import replace
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider
    monkeypatch.setattr(PlanningBudget,"elapsed",property(lambda self:0.))
    snapshot=reference_world.build_planning_snapshot();a=snapshot.aircraft[ring]
    reference=ReferenceProvider().get(snapshot,a,budget=PlanningBudget())
    budget=PlanningBudget()
    seed=CandidateProvider()._reference_seed(snapshot,a,replace(_node(snapshot,a),soc=soc,peak=peak),reference,budget)
    assert seed is None and budget.screening["seed_energy_rejected"]==1
    assert reference.status=="KNOWN" and reference.plans[-1].pad_id is not None


def test_reference_seed_matches_emergency_truncation_and_immediate_inner_exception(monkeypatch,reference_world):
    from dataclasses import replace
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, HORIZON
    monkeypatch.setattr(PlanningBudget,"elapsed",property(lambda self:0.))
    snapshot=reference_world.build_planning_snapshot();provider=CandidateProvider()
    monkeypatch.setattr(provider,"_emergency_prediction",lambda *args:"UNKNOWN")
    for ring in ("outer","middle"):
        a=snapshot.aircraft[ring];node=_node(snapshot,a)
        reference=ReferenceProvider().get(snapshot,a,budget=PlanningBudget())
        seed=provider._reference_seed(snapshot,a,node,reference,PlanningBudget())
        if ring=="outer":
            assert seed==(reference.plans[:1],"UNKNOWN")
        else:
            assert seed==(reference.plans,"UNKNOWN_IMMEDIATE_NORMAL_EXCEPTION")
            first=reference.plans[0]
            delayed=trajectory.generate_final(a.id,"inner",first.target_slot,reference.plans[-1].pad_id,
                first.end_time+5.,source_version=first.source_version+1,config=snapshot.config)
            assert delayed.accepted
            assert provider._reference_seed(snapshot,a,node,replace(reference,plans=(first,delayed.plan)),PlanningBudget())==(reference.plans[:1],"UNKNOWN")
    a=snapshot.aircraft["outer"];node=_node(snapshot,a)
    first=trajectory.generate_transfer(a.id,"outer",a.slot,0,HORIZON-20.,source_version=node.version,
        config=snapshot.config,durations=(40,)).plan
    assert first is not None
    reference=ReferenceProvider().get(snapshot,a,budget=PlanningBudget())
    assert provider._reference_seed(snapshot,a,node,replace(reference,plans=(first,*reference.plans[1:])),PlanningBudget())==((first,),"UNKNOWN")
