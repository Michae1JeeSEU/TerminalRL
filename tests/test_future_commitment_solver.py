"""Real service-objective selection and delayed first-action delivery.

The physical fixtures run locally even when OR-Tools is unavailable.  The CP
variants must also run on the server; a local skip is not solver evidence.
"""
from dataclasses import replace
import json

import pytest

from terminal_marl.core.rotating_slots.future_commitment import record_for_plan
from terminal_marl.core.rotating_slots.planning_latency import (
    PlanningTiming, mark_submission_ready, prepare_submission,
    project_visible_snapshot, publish_submission,
)
from terminal_marl.core.rotating_slots.selective_replanning import (
    PlanningScope, build_global_frame,
)
from terminal_marl.core.rotating_slots.stage2_downstream import SERVICE_EVALUATION_PROFILE
from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, EnergySlackScheduler, EvaluationContext, PlanningBudget,
    RollingHorizonCPSATScheduler, _service_empty_reason,
)
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


def _world(ring="middle"):
    world = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency = False
    world.seed_hold("a", ring, 0, .9)
    return world


def _transfer(world, start, *, ring="middle", target=0, duration=40.):
    generated = generate_transfer("a", ring, 0, target, start,
        durations=(duration,), config=world.config)
    assert generated.accepted and generated.plan is not None, generated
    return generated.plan


def _publish(world, timing, plans):
    assert timing.effective_time - timing.snapshot_time == pytest.approx(20.)
    prepared, token = prepare_submission(world, plans, timing)
    assert prepared.accepted and token is not None, prepared
    world.advance(world.time + 1., False)
    ready = mark_submission_ready(world, token)
    assert ready.accepted, ready
    world.advance(timing.effective_time, False)
    published = publish_submission(world, token)
    assert published.accepted, published
    return published


def _problem(world, timing, paths, *, aid="a"):
    snapshot = project_visible_snapshot(world.build_planning_snapshot(),
        timing.effective_time, timing=timing)
    provider = CandidateProvider(planning_profile="downstream_service_v8",
        evaluation_profile=SERVICE_EVALUATION_PROFILE)
    frame = build_global_frame(snapshot, provider, EvaluationContext(snapshot))
    eligible = tuple(a.id for a in snapshot.aircraft.values()
        if a.status.startswith(("HOLD_", "TRANSFER_")))
    scope = PlanningScope(snapshot.version, snapshot.time, eligible, (aid,),
        tuple(sorted(set(snapshot.aircraft) - {aid})), actual_k=1,
        objective_frame_id=frame.identity)
    budget = PlanningBudget(soft_seconds=60., watchdog_seconds=60.,
        candidate_seconds=15., solver_seconds=10., max_refinement_rounds=8)
    context = EvaluationContext(snapshot, budget=budget, scope=scope, global_frame=frame)
    aircraft = snapshot.aircraft[aid]
    columns = [provider._column(snapshot, aircraft, (), kind, context)
        for kind in ("KEEP", "WAIT")]
    columns += [provider._column(snapshot, aircraft, tuple(path), "MOVE", context)
        for path in paths]
    assert all(c is not None for c in columns)
    catalog = provider._progressive_catalog(snapshot, budget, context,
        {aid: columns}, {aid: columns[0].reference}, [aircraft])
    return snapshot, context, budget, catalog


def _solve(engine, snapshot, context, budget, catalog):
    if engine == "cp":
        pytest.importorskip("ortools.sat.python.cp_model")
        scheduler = RollingHorizonCPSATScheduler()
    else:
        scheduler = EnergySlackScheduler()
    result = scheduler.plan(snapshot, catalog, context, budget)
    assert result.certified, result
    assert len(result.objective) == 6
    assert result.objective == catalog.downstream.key(result.selected)
    assert context.check_commit_prefix(result.selected).accepted
    if engine == "cp":
        assert result.solver_records
        components = {row.get("objective_components") for row in result.solver_records}
        assert components == {("F", "-L"), ("R",), ("G_service",), ("E",), ("tau",)}
        for row in result.solver_records:
            if "model_integer_key" in row:
                assert row["model_integer_key"] == row["canonical_selected_integer_key"]
    print(json.dumps(dict(engine=engine, status=result.status,
        selected=[c.kind for c in result.selected], objective=result.objective,
        prefix_starts=[p.start_time for p in result.prefix]), sort_keys=True))
    return result


@pytest.mark.parametrize("engine", ("energy_slack", "cp"))
def test_keep_existing_absolute_reservation_beats_worse_move_across_real_replans(engine):
    world = _world()
    old = _transfer(world, 150.3)
    _publish(world, PlanningTiming.from_lead(0., 20.), (old,))
    original_id = record_for_plan(world, old)["reservation_id"]
    # Each projection uses its own 20 s lead. The known absolute appointment
    # must never be reconstructed as effective_time + the original wait.
    for source_time in (20., 22., 24.):
        world.advance(source_time, False)
        timing = PlanningTiming.from_lead(source_time, 20.)
        worse = _transfer(world, 150.3, target=1)
        snapshot, context, budget, catalog = _problem(world, timing, ((worse,),))
        keep, _, move = catalog.columns["a"]
        keep_key = catalog.downstream.key((keep,))
        move_key = catalog.downstream.key((move,))
        assert keep_key[:4] == move_key[:4]
        assert keep_key[4] < move_key[4] and keep_key < move_key
        assert context.check_single(move).accepted
        print(json.dumps(dict(case="absolute_keep", effective=snapshot.time,
            old_start=old.start_time, keep=keep_key, alternative=move_key)))
        assert context.all_plans("a", ()) == (old,)
        assert context.all_plans("a", (worse,)) == (worse,)
        result = _solve(engine, snapshot, context, budget, catalog)
        assert not result.prefix and result.selected[0].kind in ("KEEP", "WAIT")
        assert _service_empty_reason(result, catalog, context, budget) == "WAITING_FOR_RESERVATION_START"
        assert snapshot.soft["a"].start_time == world.soft["a"].start_time == 150.3
        assert record_for_plan(world, old)["reservation_id"] == original_id


@pytest.mark.parametrize("engine", ("energy_slack", "cp"))
def test_better_real_six_objective_move_atomically_replaces_old_calendar(engine):
    world = _world()
    old = _transfer(world, 150.3)
    _publish(world, PlanningTiming.from_lead(0., 20.), (old,))
    previous = record_for_plan(world, old)
    timing = PlanningTiming.from_lead(world.time, 20.)
    replacement = _transfer(world, 90.3)
    snapshot, context, budget, catalog = _problem(world, timing, ((replacement,),))
    keep, _, move = catalog.columns["a"]
    assert catalog.downstream.key((move,)) < catalog.downstream.key((keep,))
    print(json.dumps(dict(case="replacement", keep=catalog.downstream.key((keep,)),
        replacement=catalog.downstream.key((move,)))))
    assert context.check_single(move).accepted
    result = _solve(engine, snapshot, context, budget, catalog)
    assert result.prefix == (replacement,)
    assert world.soft["a"] == old and previous["state"] == "RESERVED"
    _publish(world, timing, result.prefix)
    current = record_for_plan(world, replacement)
    assert world.soft["a"] == replacement and current["state"] == "RESERVED"
    assert previous["state"] == "CANCELLED"
    assert previous["replacement_reservation_id"] == current["reservation_id"]
    rows = world.build_planning_snapshot().slots
    assert any(r.aircraft_id == "a" and r.resource == ("inner", 0)
        and r.end > replacement.end_time for r in rows)
    world.advance(replacement.start_time + .001, False)
    assert world.active["a"] == replacement


@pytest.mark.parametrize("engine", ("energy_slack", "cp"))
def test_equivalent_service_prefers_real_lower_energy_delayed_outer_wait(engine):
    world = _world("outer")
    for index in range(6):
        world.seed_hold("inner_" + str(index), "inner", 3 * index, .9)
        world.seed_service("ground_" + str(index), index, 120.)
    timing = PlanningTiming.from_lead(0., 20.)
    immediate = _transfer(world, 20., ring="outer")
    delayed = _transfer(world, 40., ring="outer")
    snapshot, context, budget, catalog = _problem(world, timing,
        ((immediate,), (delayed,)))
    keep, _, now, later = catalog.columns["a"]
    now_key, now_rows = catalog.downstream.evaluate((now,))
    later_key, later_rows = catalog.downstream.evaluate((later,))
    assert now_key[:4] == later_key[:4]
    nr, lr = now_rows["a"], later_rows["a"]
    assert nr["service_start"] == lr["service_start"]
    assert nr["estimated_touchdown"] == lr["estimated_touchdown"]
    assert lr["total_j"] < nr["total_j"]
    actual = context.energy_portions(((delayed, delayed.start_time, delayed.end_time),))[0]
    assert lr["explicit_prefix_j"] >= actual
    assert lr["estimated_wait_seconds_by_ring"]["outer"] > nr["estimated_wait_seconds_by_ring"]["outer"]
    assert later_key < now_key
    print(json.dumps(dict(case="equivalent_service_outer_wait", immediate=now_key,
        delayed=later_key, service_start=nr["service_start"],
        immediate_energy_j=nr["total_j"], delayed_energy_j=lr["total_j"],
        transfer_energy_j=actual)))
    # KEEP is also a real, lower-energy outer waiting option in this fixture.
    # Solve the two explicit inward alternatives to isolate where they wait.
    catalog = replace(catalog, columns={"a": (now, later)})
    result = _solve(engine, snapshot, context, budget, catalog)
    assert result.prefix == (delayed,)


@pytest.mark.parametrize("engine", ("energy_slack", "cp"))
@pytest.mark.parametrize("obstacle", ("past_start", "source_version", "active_continuation"))
def test_undeliverable_prediction_is_interface_obstacle_not_reasonable_wait(engine, obstacle):
    world = _world()
    timing = PlanningTiming.from_lead(0., 20.)
    if obstacle == "past_start":
        stale = _transfer(world, 10.)
        expected = "FIRST_ACTION_OUTSIDE_DELIVERY_WINDOW"
    elif obstacle == "source_version":
        stale = replace(_transfer(world, 40.), source_version=999, trajectory_hash="")
        expected = "FIRST_ACTION_SOURCE_NOT_EFFECTIVE_HOLD"
    else:
        current = _transfer(world, 0.)
        assert world.submit((current,)).accepted
        world.advance(.001, False)
        timing = PlanningTiming.from_lead(world.time, 20.)
        stale = generate_final("a", "inner", 0, 5, current.end_time,
            source_version=1, config=world.config).plan
        assert stale is not None
        expected = "EXISTING_EFFECTIVE_COMMITMENT"
    snapshot, context, budget, catalog = _problem(world, timing, ((stale,),))
    move = catalog.columns["a"][2]
    checked = context.check_single(move)
    assert not checked.accepted and checked.reason == "DELIVERY_INTERFACE_OBSTACLE"
    assert checked.details["obstacle"] == expected
    result = _solve(engine, snapshot, context, budget, catalog)
    assert not result.prefix and all(c.kind != "MOVE" for c in result.selected)
    if engine == "cp":
        assert move.signature in context._cp_continuation["single_exclusions"]
    assert _service_empty_reason(result, catalog, context, budget) == "DELIVERY_INTERFACE_OBSTACLE"


@pytest.mark.parametrize("engine", ("energy_slack", "cp"))
def test_delayed_903_first_action_selected_without_delivering_prediction_suffix(engine):
    world = _world()
    world.advance(50.289792, False)
    timing = PlanningTiming.from_lead(world.time, 20.)
    first = _transfer(world, 90.3)
    final = generate_final("a", "inner", 0, 5, first.end_time,
        source_version=1, config=world.config).plan
    assert final is not None
    snapshot, context, budget, catalog = _problem(world, timing, ((first, final),))
    move = catalog.columns["a"][2]
    assert context.check_single(move).accepted
    result = _solve(engine, snapshot, context, budget, catalog)
    assert result.selected[0].plans == (first, final)
    assert result.prefix == (first,)
    _publish(world, timing, result.prefix)
    assert world.soft["a"] == first and first.start_time == 90.3
    assert record_for_plan(world, first)["state"] == "RESERVED"
    assert record_for_plan(world, final) is None
    assert not any(r.aircraft_id == "a" for r in world.build_planning_snapshot().pads)
