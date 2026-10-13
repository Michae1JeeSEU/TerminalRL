from copy import deepcopy
from dataclasses import replace
from math import pi
import pytest
from terminal_marl.core.fixed_terminal.domain import Aircraft, Plan, Reservation, Segment, TerminalConfig
from terminal_marl.core.fixed_terminal.routes import build_plan
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.core.fixed_terminal.baselines import joint_choices, Statistics


def env_for(*requests):
    return FixedTerminalEnv(TerminalConfig(), Scenario(1, "test", "test", tuple(requests), 1, 2))


def test_joint_plan_can_depend_on_another_holder_departing():
    requests = [Request(f"inner-{i}", 0, .8, 0, 2*pi*i/6) for i in range(6)]
    requests.append(Request("outer", 0, .8, 1, 1.))
    env = env_for(*requests)
    inner = build_plan(env.aircraft["inner-0"], 0, env.config)
    outer = build_plan(env.aircraft["outer"], 0, env.config)
    assert env.feasibility.check_batch((outer,), list(env.aircraft.values()), 0).reason == "operational_capacity:0"
    assert env.feasibility.check_batch((inner, outer), list(env.aircraft.values()), 0).status == "feasible"
    choices = joint_choices(env, {"outer": (outer,), "inner-0": (inner,)})
    assert len(choices[0]) == 2
    assert env.submit_batch(choices[0]).status == "feasible"
    assert env.aircraft["outer"].plan == outer and env.aircraft["inner-0"].plan == inner


def test_invalid_batch_is_atomic_and_commitments_cannot_be_replaced():
    env = env_for(Request("a", 0, .8, 0, 0), Request("b", 0, .8, 1, 1))
    a = build_plan(env.aircraft["a"], 0, env.config)
    b = build_plan(env.aircraft["b"], 0, env.config)
    malicious = replace(b, reservations=())
    before = deepcopy(env.aircraft), list(env.events)
    assert env.submit_batch((a, malicious)).status == "infeasible"
    assert (env.aircraft, env.events) == before
    assert env.submit_batch((a, a)).reason == "batch_duplicate_or_emergency"
    assert env.submit_batch((replace(a, emergency=True),)).reason == "batch_duplicate_or_emergency"
    assert env.submit_batch((a,)).status == "feasible"
    assert env.submit_batch((a,)).reason == "batch_requires_observed_holders"
    assert env.aircraft["a"].plan == a


def test_three_way_capacity_counts_all_selected_routes():
    env = env_for(Request("a", 0, .8, 1, 0), Request("b", 0, .8, 1, 1.2))
    # Exercise capacity independently from route-spacing and unary resources:
    # five holders plus two simultaneous incoming ring occupancy reservations.
    residents = [Aircraft(f"r{i}", 0, .8, .8, 0, 2*pi*i/5) for i in range(5)]
    a = build_plan(env.aircraft["a"], 0, env.config, (1, 0))
    b = build_plan(env.aircraft["b"], 0, env.config, (1, 0))
    b_future = replace(env.aircraft["b"], state="scheduled", plan=b)
    assert env.feasibility._capacity(a, residents, 0) is None
    assert env.feasibility._capacity(a, residents+[b_future], 0) == "operational_capacity:0"
    # Full batch invokes the same centralized multiway test.
    fleet = [env.aircraft["a"], env.aircraft["b"]]+residents
    assert env.feasibility.check_batch((a, b), fleet, 0).reason == "operational_capacity:0"


def test_joint_checker_enforces_protected_continuous_spacing():
    env = env_for(Request("a", 0, .8, 1, 0))
    plan = build_plan(env.aircraft["a"], 0, env.config)
    crossing = Segment(0, 10, "transfer", (16, -10, 8), (16, 10, 8))
    other = Aircraft("crossing", 0, .8, .8, 1, 2, state="scheduled",
                     plan=Plan("crossing", False, (), (crossing,), ()))
    check = env.feasibility.check_batch((plan,), list(env.aircraft.values())+[other], 0)
    assert check.status == "infeasible" and check.reason == "continuous_spacing:crossing"
