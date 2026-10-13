from copy import deepcopy
from dataclasses import replace
from math import pi
import pytest
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG
from terminal_marl.core.fixed_terminal.baselines import analyse_windows, rule_order, observed_clone, choose_normal_batch, normal_domain
from terminal_marl.core.fixed_terminal.domain import Aircraft, Plan, Reservation, Segment
from terminal_marl.core.fixed_terminal.routes import build_plan
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv


def env_for(*requests):
    return FixedTerminalEnv(DEFAULT_CONFIG, Scenario(1, "test", "test", tuple(requests), 1, 2))


def test_energy_slack_order_differs_from_raw_soc():
    # Higher SOC outer request has less task-energy slack than inner request.
    env = env_for(Request("inner", 0, .20, 0, 0), Request("outer", 0, .21, 2, pi))
    assert rule_order(env, "soc")[0].aircraft_id == "inner"
    assert rule_order(env, "energy_slack")[0].aircraft_id == "outer"
    assert rule_order(env, "fcfs")[0].aircraft_id == "inner"


def test_large_domain_deadline_certified_by_descending_lap_tiers():
    env = env_for(Request("a", 0, .9, 2, 0))
    row = analyse_windows(env, env.aircraft["a"], 256)
    assert row["domain_size"] > 256
    assert row["minimum_status"] == row["deadline_status"] == "exact"
    assert row["examined"] == 2
    assert row["latest_feasible_completion"] == pytest.approx(row["completion_energy_only_upper_bound"])


def test_deadline_depends_on_reserved_pad_and_reports_budget_bounds():
    env = env_for(Request("a", 0, .3, 0, 0))
    before = analyse_windows(env, env.aircraft["a"], 256)
    end = before["latest_feasible_completion"]
    busy = Segment(end-2, end+2, "stationary", (-50, 0, 0), (-50, 0, 0))
    env.aircraft["busy"] = Aircraft("busy", 0, .8, .8, 0, 0, state="scheduled",
                                   plan=Plan("busy", False, (), (busy,), (Reservation("normal_pad", end-2, end+2),)))
    after = analyse_windows(env, env.aircraft["a"], 256)
    assert after["deadline_status"] == "exact"
    assert after["latest_feasible_completion"] == pytest.approx(end-env.config.lap_time)
    capped = analyse_windows(env, env.aircraft["a"], 1)
    assert capped["minimum_status"] == "exact" and capped["deadline_status"] == "lower_bound"


def test_observed_clone_and_rolling_do_not_see_pending_or_mutate_live():
    env = env_for(Request("a", 0, .8, 0, 0), Request("secret-future", 1, .051, None, 0))
    view = observed_clone(env)
    assert "secret-future" not in view.aircraft
    assert all(r.aircraft_id != "secret-future" for r in view.scenario.requests)
    saved = deepcopy(env.aircraft), deepcopy(env.scenario), list(env.events), env.time
    settings = replace(DEFAULT_SCHEDULER_CONFIG, rolling_seconds=5, rolling_root_alternatives=2,
                       candidates_per_aircraft=2, catalogue_scan_limit=2, beam_width=4)
    decision = choose_normal_batch("rolling_horizon", view, settings)
    assert decision.statistics.lookahead_states >= 5
    assert (env.aircraft, env.scenario, env.events, env.time) == saved
    assert env.config == DEFAULT_CONFIG
    # The input guard also stops accidentally handing a live future-bearing env.
    with pytest.raises(ValueError, match="observed"):
        choose_normal_batch("fcfs", env)


def test_existing_commitment_is_identical_after_planning():
    env = env_for(Request("a", 0, .8, 0, 0), Request("b", 0, .8, 2, 2))
    plan = build_plan(env.aircraft["a"], 0, env.config)
    env.submit_batch((plan,))
    choose_normal_batch("joint_window", observed_clone(env))
    assert env.aircraft["a"].plan is plan
