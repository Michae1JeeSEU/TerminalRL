"""Plan continuation retains serialized geometry beyond maneuver endpoints."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from terminal_marl.core.rotating_slots import energy, safety, trajectory as tr
from terminal_marl.core.rotating_slots.capacity_config import make_capacity_config
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.geometry import slot_state


@pytest.fixture(scope='module', params=(250., 300., 350.))
def geometry_plans(request):
    radius = request.param
    config = load_config() if radius == 300. else make_capacity_config(inner_radius_m=radius)
    slot = config.airspace.rings[2].slot_count - 1
    stable = tr.make_stable_plan('stable', 'inner', slot, 12.3, 32.3, config=config)
    transfer = tr.generate_transfer('transfer', 'middle', 0, slot, 12.3, config=config)
    final = tr.generate_final('final', 'inner', slot, 0, 12.3, config=config)
    assert transfer.accepted and final.accepted
    return config, (stable, transfer.plan, final.plan)


def assert_state(actual, expected):
    for name in ('p', 'v', 'a', 'jerk'):
        assert getattr(actual, name) == pytest.approx(getattr(expected, name), abs=2e-9, rel=2e-12)


def circular_sides(plan):
    yield plan.start_time, plan.source_ring, plan.source_slot, -1
    if plan.target_ring is not None:
        yield plan.end_time, plan.target_ring, plan.target_slot, 1


def test_continuation_states_use_encoded_geometry(geometry_plans):
    config, plans = geometry_plans
    for plan in plans:
        for time, ring, slot, direction in circular_sides(plan):
            for offset in (1e-6, .2, 4.7, 240.):
                t = time + direction * offset
                assert_state(plan.state(t), slot_state(ring, slot, t, config))
            # Maneuver endpoints are C2; their jerk can differ from the orbit.
            boundary = slot_state(ring, slot, time, config)
            actual = plan.state(time)
            for name in ('p', 'v', 'a'):
                assert getattr(actual, name) == pytest.approx(getattr(boundary, name), abs=2e-9)
            adjacent = plan.state(time + direction * 1e-7)
            assert math.dist(adjacent.p, actual.p) < 3e-6
            assert math.dist(adjacent.v, actual.v) < 2e-7
            assert math.dist(adjacent.a, actual.a) < 2e-8


def test_continuation_intervals_and_bounds_enclose_matching_states(geometry_plans):
    config, plans = geometry_plans
    for plan in plans:
        ranges = [(plan.start_time-4.7, plan.start_time-.2)]
        if plan.target_ring is not None:
            ranges += [(plan.end_time+.2, plan.end_time+4.7),
                       (plan.start_time-1., plan.end_time+1.)]
        else:
            ranges += [(plan.end_time+.2, plan.end_time+4.7)]
        for start, end in ranges:
            segments = tuple(plan.intervals(start, end))
            for segment in segments:
                lo, hi = max(start, segment.start_time), min(end, segment.end_time)
                for t in np.linspace(lo, hi, 9):
                    # At a maneuver endpoint the STABLE extension is C2 only.
                    state = segment.state(t)
                    actual = plan.state(t)
                    names = ('p', 'v', 'a') if t in (plan.start_time, plan.end_time) else ('p', 'v', 'a', 'jerk')
                    for name in names:
                        assert getattr(state, name) == pytest.approx(getattr(actual, name), abs=2e-9)
                    if segment.kind == 'STABLE':
                        ring, slot = ((plan.source_ring, plan.source_slot) if hi <= plan.start_time
                                      else (plan.target_ring, plan.target_slot))
                        assert_state(state, slot_state(ring, slot, t, config))
            for derivative, name in enumerate(('p', 'v', 'a', 'jerk')):
                bounds = plan.bounds(start, end, derivative)
                for t in np.linspace(start, end, 21):
                    for value, (lo, hi) in zip(getattr(plan.state(t), name), bounds):
                        assert lo <= value <= hi


def test_serialized_and_replaced_plans_need_no_default_config(geometry_plans, monkeypatch):
    _, plans = geometry_plans
    queries = [(plan, [(time+direction*offset) for time, _, _, direction in circular_sides(plan)
                       for offset in (.2, 240.)]) for plan in plans]
    expected = [[plan.state(t) for t in times] for plan, times in queries]
    monkeypatch.setattr(tr, 'load_config', lambda: pytest.fail('continuation loaded default geometry'))
    monkeypatch.setattr(tr, 'slot_state', lambda *args: pytest.fail('continuation consulted slot metadata'))
    for (plan, times), states in zip(queries, expected):
        restored = tr.TrajectoryPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        assert restored.to_dict() == plan.to_dict()
        clone = replace(restored, aircraft_id='clone', trajectory_hash='')
        shift = 20.
        shifted = replace(restored, start_time=plan.start_time+shift, end_time=plan.end_time+shift,
                          segments=tuple(replace(s, start_time=s.start_time+shift) for s in plan.segments),
                          trajectory_hash='')
        for t, state in zip(times, states):
            assert_state(restored.state(t), state)
            assert_state(clone.state(t), state)
            assert_state(shifted.state(t+shift), state)
        for start, end in ((plan.start_time-4., plan.start_time-1.),
                           (plan.end_time+1., plan.end_time+4.)):
            assert tuple(restored.intervals(start, end)) == tuple(plan.intervals(start, end))
            assert restored.bounds(start, end) == plan.bounds(start, end)


def test_continuation_energy_and_separation_retain_custom_radius(geometry_plans):
    config, plans = geometry_plans
    stable, transfer, final = plans
    for plan, start, end in ((stable, stable.start_time-4., stable.start_time-1.),
                             (transfer, transfer.end_time+1., transfer.end_time+4.),
                             (final, final.start_time-4., final.start_time-1.)):
        assert safety.validate_plan(plan, config).accepted
        actual = energy.integrate_energy(plan, start, end, config=config)
        reference = energy.integrate_energy(stable, start, end, config=config)
        assert actual.valid and reference.valid
        assert actual.joules == pytest.approx(reference.joules, rel=2e-12)
    neighbor = tr.make_stable_plan('neighbor', 'inner', 0, config=config)
    start, end = transfer.end_time+.1, transfer.end_time+.3
    separation = safety.validate_separation(transfer, neighbor, start, end, config)
    assert separation.accepted
    distance = math.dist(slot_state('inner', transfer.target_slot, start, config).p,
                         slot_state('inner', 0, start, config).p)
    assert separation.details['lower_bound_m'] <= distance+1e-9
    assert separation.details['upper_bound_m'] == pytest.approx(distance, abs=1e-9)


def test_inbound_preadmission_and_ground_continuation_are_preserved():
    config = make_capacity_config(inner_radius_m=350.)
    result = tr.generate_inbound('inbound', 0, 1, 0., config=config, durations=[55.])
    assert result.accepted
    inbound = result.plan
    assert inbound.state(inbound.start_time-1.) == inbound.segments[0].state(inbound.start_time)
    with pytest.raises(ValueError, match='before admission'):
        tuple(inbound.intervals(inbound.start_time-1., inbound.start_time))
    assert_state(inbound.state(inbound.end_time+4.), slot_state('outer', 1, inbound.end_time+4., config))
    ground = tr.make_ground_plan('ground', 0, 12.3, 32.3, config)
    for plan in (ground, tr.generate_final('final', 'inner', 20, 0, 12.3, config=config).plan):
        state = plan.state(plan.end_time+1.)
        assert state.p == plan.state(plan.end_time).p
        assert state.v == state.a == state.jerk == (0., 0., 0.)
        assert tuple(plan.intervals(plan.end_time+1., plan.end_time+2.))[0].kind == 'GROUND'
