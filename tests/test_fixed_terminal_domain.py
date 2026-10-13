from dataclasses import FrozenInstanceError, replace
from math import pi
from random import Random
import pytest

from terminal_marl.core.fixed_terminal.domain import Aircraft, Ring, Segment, TerminalConfig, distance
from terminal_marl.core.fixed_terminal.routes import build_plan, plan_energy, required_energy, segment_energy
from terminal_marl.core.fixed_terminal.scenarios import generate_scenario, truncated_normal


def test_frozen_geometry_capacity_and_compatibility():
    c = TerminalConfig()
    assert [(c.geometric_capacity(i), c.operational_capacity(i)) for i in range(3)] == [(25, 6), (41, 8), (58, 9)]
    with pytest.raises(FrozenInstanceError):
        c.rings = ()
    with pytest.raises(FrozenInstanceError):
        c.rings[0].radius = 99
    with pytest.raises(ValueError, match="holding speed"):
        replace(c, max_speed=2)
    with pytest.raises(ValueError, match="adjacent"):
        replace(c, rings=(Ring(12, 6), Ring(13, 7)))
    with pytest.raises(ValueError, match="emergency pad"):
        replace(c, emergency_pad=(-10, 0, 0))
    rings = list(c.rings)
    copied = replace(c, rings=rings)
    rings[0] = Ring(99, 99)
    assert copied.rings == c.rings and isinstance(copied.rings, tuple)
    with pytest.raises(ValueError, match="finite emergency"):
        replace(c, emergency_pad=(float("nan"), 0., 0.))


def test_high_density_initial_state_every_ring_safe_and_ongoing():
    c = TerminalConfig()
    scenario = generate_scenario(c, 101, "high", "low")
    initial = [r for r in scenario.requests if r.ring is not None]
    assert [sum(r.ring == i for r in initial) for i in range(3)] == [6, 8, 9]
    assert any(r.arrival_time > c.horizon/2 for r in scenario.requests)
    aircraft = [Aircraft(r.aircraft_id, 0, r.soc, r.soc, r.ring, r.theta) for r in initial]
    assert min(distance(a.position(0, c), b.position(0, c)) for i, a in enumerate(aircraft) for b in aircraft[i+1:]) >= c.protected_distance


def test_rejection_sampling_and_independent_demand_stream():
    c = TerminalConfig()
    a = generate_scenario(c, 23, "high", "high")
    b = generate_scenario(c, 23, "high", "low")
    assert a == generate_scenario(c, 23, "high", "high")
    assert [r.arrival_time for r in a.requests] == [r.arrival_time for r in b.requests]
    assert [r.soc for r in a.requests] != [r.soc for r in b.requests]
    values = [truncated_normal(Random(seed), .5, .6, .1, .9) for seed in range(100)]
    assert all(.1 < value < .9 for value in values)

    class Draws:
        def __init__(self):
            self.values = iter([-1, 2, .3])

        def gauss(self, mean, std):
            return next(self.values)

    assert truncated_normal(Draws(), .5, .1, .1, .9) == .3


def test_mode_energy_and_position_dependent_normal_emergency_requirements():
    c = TerminalConfig()
    for mode, expected in [("hold", .008), ("transfer", .012), ("descent", .009), ("landing", .014)]:
        segment = Segment(0, 10, mode, (1, 0, 1), (1, 0, 1), omega=.12)
        assert segment_energy(segment, c) == pytest.approx(expected)
    descending_transfer = Segment(0, 10, "transfer", (20, 0, 10), (12, 0, 6))
    assert segment_energy(descending_transfer, c) == pytest.approx(.021)
    a = Aircraft("a", 0, .8, .8, 2, 0)
    b = replace(a, theta=pi)
    assert required_energy(b, 0, c) > required_energy(a, 0, c)
    assert required_energy(b, 0, c, emergency=True) < required_energy(b, 0, c)
    assert plan_energy(build_plan(a, 0, c), c)+c.energy.reserve == required_energy(a, 0, c)


def test_joint_downstream_windows_and_path_continuity():
    c = TerminalConfig()
    a = Aircraft("a", 0, .8, .8, 2, .5)
    base = build_plan(a, 0, c)
    varied = build_plan(a, 0, c, (1, 2, 0))
    assert varied.end-base.end == pytest.approx(3*c.lap_time)
    holds = {s.ring for s in varied.segments if s.mode == "hold"}
    assert holds == {0, 1, 2}
    first_transfer_base = next(r for r in base.reservations if r.resource == "transfer:1:0")
    first_transfer_varied = next(r for r in varied.reservations if r.resource == "transfer:1:0")
    assert first_transfer_varied.start-first_transfer_base.start == pytest.approx(2*c.lap_time)
    normal_pad_base = max(r.end for r in base.reservations if r.resource == "normal_pad")
    normal_pad_varied = max(r.end for r in varied.reservations if r.resource == "normal_pad")
    assert normal_pad_varied-normal_pad_base == pytest.approx(3*c.lap_time)
    for first, second in zip(varied.segments, varied.segments[1:]):
        assert first.end == pytest.approx(second.start)
        assert distance(first.position(first.end), second.position(second.start)) < 1e-8
    assert all(s.speed_bound <= c.max_speed+1e-9 for s in varied.segments)
    with pytest.raises(ValueError):
        build_plan(a, 0, c, (0,))
    for invalid_ring in (-1, len(c.rings), True, 1.5):
        with pytest.raises(ValueError, match="ring index"):
            build_plan(replace(a, ring=invalid_ring), 0, c)
