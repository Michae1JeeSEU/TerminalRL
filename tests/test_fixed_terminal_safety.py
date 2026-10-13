from dataclasses import replace
from math import pi
import pytest
from terminal_marl.core.fixed_terminal.domain import Aircraft, Plan, Reservation, Segment, TerminalConfig
from terminal_marl.core.fixed_terminal.routes import build_plan, required_energy
from terminal_marl.core.fixed_terminal.safety import Feasibility, swept_spacing


def test_continuous_crossing_hidden_by_endpoint_sampling():
    first = Segment(0, 10, "transfer", (-10, 0, 5), (10, 0, 5))
    second = Segment(0, 10, "transfer", (0, -10, 5), (0, 10, 5))
    assert swept_spacing(first, second, 0, 10, 3) == "infeasible"
    circle = Segment(0, 2*pi/.12, "hold", (12, 0, 6), (12, 0, 6), 0, 0, .12)
    fixed = Segment(0, circle.end, "stationary", (-12, 0, 6), (-12, 0, 6))
    assert swept_spacing(circle, fixed, 0, circle.end, 3) == "infeasible"


def test_safety_rejects_reserve_noncanonical_route_and_resource_occupation():
    c, a = TerminalConfig(), Aircraft("a", 0, .8, .8, 0, 0)
    f = Feasibility(c)
    p = build_plan(a, 0, c)
    assert f.check(p, a, [a], 0).status == "feasible"
    assert f.check(replace(p, reservations=()), a, [a], 0).reason == "noncanonical_route_or_reservations"
    low = replace(a, soc=.06)
    assert f.check(build_plan(low, 0, c), low, [low], 0).reason == "insufficient_energy_reserve"
    busy = Aircraft("busy", 0, .8, .8, 0, 0, state="scheduled", plan=replace(p, aircraft_id="busy"))
    assert f.check(p, a, [a, busy], 0).reason.startswith("resource_conflict")


def test_capacity_is_independent_of_spatial_test_and_scheduler():
    c, f = TerminalConfig(), Feasibility(TerminalConfig())
    a = Aircraft("arrival", 0, .8, .8, 1, 0)
    # Each isolated holder is geometrically safe on the inner ring; together
    # they fill its fixed operational capacity before the proposed merge.
    holders = [Aircraft(f"holder-{i}", 0, .8, .8, 0, 2*pi*(i+.5)/c.operational_capacity(0))
               for i in range(c.operational_capacity(0))]
    result = f.check(build_plan(a, 0, c), a, [a]+holders, 0)
    assert result.reason == "operational_capacity:0"
    assert c.operational_capacity(0) == 6


def test_continuous_conflict_rejected_without_shared_resource():
    c, f = TerminalConfig(), Feasibility(TerminalConfig())
    a = Aircraft("arrival", 0, .8, .8, 1, 0)
    p = build_plan(a, 0, c)
    # Moving body crosses the radial route; no reservation names are shared.
    s = Segment(0, 10, "transfer", (16, -10, 8), (16, 10, 8))
    b = Aircraft("crossing", 0, .8, .8, 0, 0, state="scheduled",
                 plan=Plan("crossing", False, (), (s,), ()))
    assert f.check(p, a, [a, b], 0).reason == "continuous_spacing:crossing"


def test_emergency_eligibility_requires_normal_infeasible_and_valid_resources():
    c = TerminalConfig()
    f = Feasibility(c)
    a = Aircraft("a", 0, .09, .09, 2, pi)
    assert required_energy(a, 0, c, emergency=True) < a.soc < required_energy(a, 0, c)
    normal = f.search_normal(a, [a], 0, None)
    assert normal.status == "infeasible"
    emergency = f.search_emergency(a, [a], 0, None)
    assert emergency.status == "feasible"
    assert f.check(emergency.plan, a, [a], 0).status == "feasible"
    rich = replace(a, initial_soc=.2, soc=.2)
    assert f.check(build_plan(rich, 0, c, emergency=True), rich, [rich], 0).reason == "normal_plan_exists"
    exhausted = replace(a, soc=.065)
    assert f.search_emergency(exhausted, [exhausted], 0, None).reason == "emergency_energy_reserve"
    busy = replace(a, aircraft_id="busy", state="scheduled", plan=replace(emergency.plan, aircraft_id="busy"))
    assert f.check(emergency.plan, a, [a, busy], 0).reason.startswith("resource_conflict:emergency")


def test_unknown_search_cannot_authorize_emergency_and_horizon_is_not_deadline():
    c = TerminalConfig(horizon=1)
    f = Feasibility(c)
    a = Aircraft("a", 0, .4, .4, 2, pi)
    assert f.search_normal(a, [a], .99, 0).status == "unknown"
    emergency = build_plan(a, .99, c, emergency=True)
    assert f.check(emergency, a, [a], .99, search_limit=0).reason == "normal_feasibility_unknown"
    result = f.search_normal(a, [a], .99, None)
    assert result.status == "feasible" and result.plan.end > c.horizon
    assert f.search_emergency(a, [a], .99, None).reason == "normal_plan_exists"


def test_all_downstream_wait_vectors_in_energy_domain_are_searchable():
    from terminal_marl.core.fixed_terminal.safety import Check
    c = TerminalConfig()
    a = Aircraft("a", 0, .25, .25, 2, 0)

    class OnlyDownstreamWindow(Feasibility):
        def check(self, plan, aircraft, fleet, now, **kwargs):
            return Check("feasible", "test") if plan.wait_cycles == (1, 0, 0) else Check("infeasible", "test")

    result = OnlyDownstreamWindow(c).search_normal(a, [a], 0, None)
    assert result.status == "feasible" and result.plan.wait_cycles == (1, 0, 0)
