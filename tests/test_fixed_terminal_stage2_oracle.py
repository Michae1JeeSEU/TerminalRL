from dataclasses import replace
import pytest
from terminal_marl.core.fixed_terminal.baselines import make_catalogue, objective, observed_clone, choose_normal_batch
from terminal_marl.core.fixed_terminal.oracle import finite_catalogue_milp, brute_force_catalogue, objective_gap
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.scripts.evaluate_fixed_terminal_stage2 import reference_cohort


def tiny():
    requests = (Request("a", 0, .4, 0, .2), Request("b", 0, .4, 1, 1.4))
    env = FixedTerminalEnv(DEFAULT_CONFIG, Scenario(1, "test", "test", requests, 1, 2))
    settings = replace(DEFAULT_SCHEDULER_CONFIG, oracle_candidates_per_aircraft=2)
    return env, make_catalogue(env, settings, reference=True)


def test_milp_matches_bruteforce_lexicographic_optimum():
    pytest.importorskip("scipy.optimize")
    env, catalogue = tiny()
    exact = brute_force_catalogue(env, catalogue)
    result = finite_catalogue_milp(env, catalogue, 10, 100)
    assert result.status == exact["status"] == "OPTIMAL"
    assert result.objective == pytest.approx(exact["objective"])
    assert result.certified_tiers == 3
    assert result.solver_bound_tier == "route_energy_soc"
    assert result.solver_dual_bound == pytest.approx(-result.objective[2])
    assert result.serializable()["solver_bound_scope"] == "current_finite_MILP_relaxation_tier_not_dynamic_bound"
    assert env.feasibility.check_batch(result.plans, list(env.aircraft.values()), env.time).status == "feasible"
    assert objective_gap(result.objective, result)["normal_count_gap"] == 0


def test_limit_never_claims_optimal_or_exposes_invalid_gap():
    pytest.importorskip("scipy.optimize")
    env, catalogue = tiny()
    result = finite_catalogue_milp(env, catalogue, 10, 0)
    assert result.status == "FEASIBLE_LIMIT" and result.certified_tiers == 0
    assert result.solver_dual_bound is None and result.solver_bound_tier is None
    assert not objective_gap((0, 0, 0), result)["valid"]


def test_closed_reference_is_original_round_robin_initial_subset():
    from terminal_marl.core.fixed_terminal.scenarios import generate_scenario
    original = generate_scenario(DEFAULT_CONFIG, 101, "high", "medium")
    cohort = reference_cohort(6, 101)
    assert [r.ring for r in cohort.requests] == [0, 1, 2, 0, 1, 2]
    assert all(r in original.requests and r.arrival_time == 0 for r in cohort.requests)


def test_reference_incumbents_are_from_the_same_restricted_catalogue():
    env, catalogue = tiny()
    allowed = {p for group in catalogue.values() for p in group}
    settings = replace(DEFAULT_SCHEDULER_CONFIG, rolling_seconds=5, rolling_root_alternatives=2)
    for method in ("fcfs", "soc", "energy_slack", "deadline_slack", "joint_window", "rolling_horizon"):
        decision = choose_normal_batch(method, observed_clone(env), settings, catalogue)
        assert set(decision.plans) <= allowed
        assert env.feasibility.check_batch(decision.plans, list(env.aircraft.values()), env.time).status == "feasible"
