from copy import deepcopy
from dataclasses import replace
from math import pi
import pytest
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.core.fixed_terminal.routes import build_plan
from terminal_marl.core.fixed_terminal.value_analysis import system_value, visible_snapshot, evaluate_root, conditional_margin
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv


def env_for(*requests, horizon=20):
    return FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=horizon), Scenario(1, "test", "test", tuple(requests), 1, 2))


def test_value_counts_actual_completion_and_arrived_denominators():
    env = env_for(Request("a", 0, .8, 0, 0), Request("future", 15, .4, None, 0))
    env.submit_batch((build_plan(env.aircraft["a"], 0, env.config),))
    value = system_value(env)
    assert value["request_denominator"] == 2 and value["arrived_request_count"] == 1
    assert value["components"]["arrived_unresolved"] == 1 and value["not_arrived_count"] == 1
    assert value["J"][:4] == (0, 0, 0, -1)  # scheduled normal is not a completion
    while env.time < 10:
        env.step()
    later = system_value(env)
    assert later["components"]["normal_landed"] == 1
    assert later["components"]["arrived_unresolved"] == 0
    assert later["episode_cumulative"] is True


def test_visible_features_no_pending_and_rollout_does_not_mutate_original():
    env = env_for(Request("a", 0, .8, 0, .4), Request("secret", 15, .053, None, 0))
    before = deepcopy(env.__dict__)
    snapshot = visible_snapshot(env)
    assert [a["aircraft_id"] for a in snapshot["aircraft"]] == ["a"]
    assert "scenario" not in snapshot
    plan = build_plan(env.aircraft["a"], 0, env.config)
    first = evaluate_root(env, (plan,), (20,), realized_future=True)
    second = evaluate_root(env, (plan,), (20,), realized_future=True)
    assert {k: v for k, v in first["values"]["20"].items() if k != "elapsed_through_stop_seconds"} == {k: v for k, v in second["values"]["20"].items() if k != "elapsed_through_stop_seconds"}
    assert first["events"] == second["events"]
    assert env.aircraft == before["aircraft"] and env.events == before["events"] and env.time == before["time"]
    observed = evaluate_root(env, (plan,), (20,), realized_future=False)
    assert observed["values"]["20"]["request_denominator"] == 1
    assert first["values"]["20"]["request_denominator"] == 2


def test_nonzero_root_uses_global_cadence_and_cumulative_prefix():
    env = env_for(Request("a", 0, .8, 0, 2))
    while env.time < 7:
        env.step()
    value = system_value(env)
    result = evaluate_root(env, (), (20,), realized_future=True)
    assert [d["time"] for d in result["continuation_decisions"]] == [10., 15.]
    assert result["values"]["20"]["root_time"] == 7
    assert result["values"]["20"]["components"]["energy_consumed_soc_all_requests"] >= value["components"]["energy_consumed_soc_all_requests"]


def test_remove_one_infeasible_is_not_assigned_a_marginal():
    requests = [Request(f"inner-{i}", 0, .8, 0, 2*pi*i/6) for i in range(6)]
    requests.append(Request("outer", 0, .8, 1, 1.))
    env = env_for(*requests, horizon=30)
    inner = build_plan(env.aircraft["inner-0"], 0, env.config)
    outer = build_plan(env.aircraft["outer"], 0, env.config)
    margin = conditional_margin(env, (inner, outer), "inner-0")
    assert margin["status"] == "infeasible" and margin["marginal"] is None
    assert margin["reason"] == "without:operational_capacity:0"
    assert margin["additivity_claim"] is False


def test_root_safety_and_existing_commitments_unchanged():
    env = env_for(Request("a", 0, .8, 0, 0), Request("b", 0, .8, 1, 1))
    p = build_plan(env.aircraft["a"], 0, env.config)
    env.submit_batch((p,))
    bad = build_plan(env.aircraft["b"], 0, env.config)
    invalid = evaluate_root(env, (replace(bad, reservations=()),), (20,), realized_future=True)
    assert invalid["status"] == "infeasible"
    assert env.aircraft["a"].plan is p
