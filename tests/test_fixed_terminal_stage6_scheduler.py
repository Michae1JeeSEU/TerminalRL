"""Nested-budget and certificate boundaries, with one real closed rollout path."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs import fixed_terminal_stage6 as config
from terminal_marl.core.fixed_terminal import assisted_scheduler as scheduler
from terminal_marl.core.fixed_terminal import validation_scheduler as previous
from terminal_marl.core.fixed_terminal.baselines import Statistics
from terminal_marl.core.fixed_terminal.reference import Root, action_key
from terminal_marl.core.fixed_terminal.routes import build_plan
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv


def tiny(*requests, horizon=12.):
    return FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=horizon),
                            Scenario(6001, "low", "medium", tuple(requests), 1, 2))


def value(n=0, e=0, f=0, c=1, delay=0.):
    return {"J": [-f, n, -e, -c, 0., -delay, 0.],
            "components": {"failed_energy": f, "normal_landed": n, "emergency_landed": e,
                           "arrived_unresolved": c, "observed_request_seconds": 0.,
                           "normal_delay_sum_seconds": delay, "energy_consumed_soc_all_requests": 0.},
            "arrived_request_count": n+e+f+c}


def fake_pool(monkeypatch, env, count=10, success_id=None):
    roots = [Root("hold", (), [], [], {})]+[
        Root(str(i), (build_plan(env.aircraft["a"], env.time, env.config, (i,)),), [], [], {})
        for i in range(count-1)]
    metadata = {"anchors": ["hold", "hold", "0"], "beam_order": [str(i) for i in range(count-1)],
                "statistics": asdict(Statistics(generated=2, examined=3)), "pool_validation_calls": count}
    calls = []
    def evaluate(view, root):
        calls.append(root.root_id)
        return {"status": "feasible", "value": value(n=1, c=0) if root.root_id == success_id else value(),
                "seconds": .001, "statistics": asdict(Statistics(generated=5, examined=7, joint_assignments=11)),
                "work": {"rollouts": 1, "simulation_steps": 12, "root_validation_calls": 2, "continuation_submission_calls": 2}}
    monkeypatch.setattr(scheduler, "make_pool", lambda *a: (roots, metadata))
    monkeypatch.setattr(scheduler, "closed_evaluation", evaluate)
    return roots, metadata, calls


def test_frozen_jobs_and_panel_scenarios():
    assert config.BUDGETS == (3, 5, 8) and config.POOL_LIMIT == 14
    assert config.EPSILON_NORMAL == 0. and config.COMPUTE_REDUCTION_TARGET == .10
    assert len(config.ARMS) == len(set(config.ARMS)) == 8
    for split, counts in [("test", (33, 15, 30)), ("pilot", (11, 5, 10))]:
        jobs = config.scenario_jobs(split)
        assert (len(jobs), len(config.panel_scenario_jobs(split)), len(config.panel_jobs(split))) == counts
        assert len({(r['seed'], r['load'], r['soc'], r['variant']) for r in jobs}) == len(jobs)
        assert {r['time'] for r in config.panel_jobs(split)} == {0., 60.}
    with pytest.raises(ValueError):
        config.scenario_jobs("train")


def test_nested_budget_rank_initial_then_beam_unique_cumulative_work(monkeypatch):
    env = tiny(Request("a", 0, .95, 0, .1))
    roots, _, calls = fake_pool(monkeypatch, env)
    scorer = SimpleNamespace(kind="rank", predict=lambda rows: [[float(i)] for i in range(len(rows))])
    row = scheduler.apply_assisted_scheduler(env, "rank_adaptive", scorer)
    stages = row["adaptive"]["stages"]
    assert [s["target_budget"] for s in stages] == [3, 5, 8, 10]
    assert [set(s["new_root_ids"]) for s in stages] == [{"hold", "0", "8"}, {"1", "2"}, {"3", "4", "5"}, {"6", "7"}]
    assert len(calls) == len(set(calls)) == len(roots) == row["budget"]
    assert row["adaptive"]["fallback_count"] == 3 and row["adaptive"]["stop_reason"] == "full_pool_exhausted"
    assert row["adaptive"]["quality_certified"] is False
    assert row["work"]["root_rollout_count"] == 10 and row["work"]["simulation_steps"] == 120
    assert row["work"]["root_validation_calls"] == 20 and row["work"]["continuation_submission_calls"] == 20
    assert row["work"]["final_submit_count"] == row["work"]["final_validation_calls"] == 1
    emergency = row["emergency_statistics"]
    assert row["statistics"]["generated"] == 52+emergency["generated"]
    assert row["statistics"]["examined"] == 73+emergency["examined"]
    assert env.search_candidates == 73+emergency["examined"] and env.decision_calls == 1
    assert stages[-1]["elapsed_seconds"] <= row["timing"]["total"]


@pytest.mark.parametrize("success,target", [("0", 3), ("3", 5), ("6", 8)])
def test_certificate_stops_after_completed_stage(monkeypatch, success, target):
    env = tiny(Request("a", 0, .95, 0, .1))
    _, _, calls = fake_pool(monkeypatch, env, success_id=success)
    row = scheduler.apply_assisted_scheduler(env, "beam_adaptive")
    assert row["budget"] == len(calls) == target
    assert row["chosen_root_id"] == success and row["adaptive"]["quality_certified"] is True
    assert row["adaptive"]["stop_reason"] == "visible_normal_upper_bound_reached"
    assert row["adaptive"]["stages"][-1]["certificate"]["new_failed"] == 0


def test_small_pool_deduplicates_clipped_stages(monkeypatch):
    env = tiny(Request("a", 0, .95, 0, .1))
    _, _, calls = fake_pool(monkeypatch, env, count=4)
    row = scheduler.apply_assisted_scheduler(env, "beam_adaptive")
    assert [s["target_budget"] for s in row["adaptive"]["stages"]] == [3, 4]
    assert len(calls) == len(set(calls)) == 4


def test_certificate_historical_counts_and_committed_emergency_conservative():
    counts = {"A": 4, "F0": 1, "E0": 1}
    result = scheduler.completion_certificate(value(n=2, f=1, e=1, c=0), counts)
    assert result["passed"] and result["normal_upper_bound"] == 2
    assert not scheduler.completion_certificate(value(n=1, f=2, e=1, c=0), counts)["passed"]
    assert not scheduler.completion_certificate(value(n=1, f=1, e=2, c=0), counts)["passed"]
    assert not scheduler.completion_certificate(value(n=1, f=1, e=1, c=1), counts)["passed"]
    # A committed emergency is not counted as already emergency_landed in E0.
    assert not scheduler.completion_certificate(value(n=0, e=1, c=0), {"A": 1, "F0": 0, "E0": 0})["passed"]
    bad = value(n=2, f=1, e=1, c=0); bad["arrived_request_count"] = 5
    with pytest.raises(ValueError, match="denominator"):
        scheduler.completion_certificate(bad, counts)


def test_visible_history_and_unarrived_are_not_certificate_input():
    env = tiny(Request("a", 0, .95, 0, .1), Request("hidden", 10, .3, None, 0.))
    env.aircraft["a"].state = "failed_energy"
    before = deepcopy(env.aircraft)
    assert scheduler.visible_counts(env) == {"A": 1, "F0": 1, "E0": 0}
    assert env.aircraft == before


def test_no_future_in_pool_features_or_rollouts(monkeypatch):
    env = tiny(Request("a", 0, .95, 0, .1), Request("hidden", 10, .3, None, 0.))
    before_hidden = deepcopy(env.aircraft["hidden"])
    roots, metadata, _ = fake_pool(monkeypatch, env, count=4)
    def assert_observed(view):
        assert set(view.aircraft) == {"a"}
        assert [r.aircraft_id for r in view.scenario.requests] == ["a"]
    def pool(view, limit):
        assert_observed(view);return roots, metadata
    original = scheduler.closed_evaluation
    def evaluate(view, root):
        assert_observed(view);return original(view, root)
    def features(view, plans):
        assert_observed(view);return [0.]*93
    monkeypatch.setattr(scheduler, "make_pool", pool)
    monkeypatch.setattr(scheduler, "closed_evaluation", evaluate)
    monkeypatch.setattr(scheduler, "extract_features", features)
    scorer = SimpleNamespace(kind="rank", predict=lambda rows: [[0.] for r in rows])
    row = scheduler.apply_assisted_scheduler(env, "gb_adaptive", scorer)
    assert env.aircraft["hidden"] == before_hidden and row["adaptive"]["visible_counts"]["A"] == 1


def test_constant_prediction_ties_follow_action_key(monkeypatch):
    env = tiny(Request("a", 0, .95, 0, .1))
    roots, metadata, _ = fake_pool(monkeypatch, env)
    predictions = [[1.]]*len(roots)
    expected = scheduler.shortlist(roots, metadata, "mlp_rank", 3, predictions)
    scorer = SimpleNamespace(kind="rank", predict=lambda rows: predictions)
    first = scheduler.apply_assisted_scheduler(env, "rank_adaptive", scorer)
    assert first["adaptive"]["stages"][0]["new_root_ids"] == [r.root_id for r in expected]
    assert first["chosen_root_id"] == min(roots, key=lambda r: action_key(r.plans)).root_id


@pytest.mark.parametrize("predictions", [None, [], [[float('nan')]]*4, [[float('inf')]]*4,
                                          [[True]]*4, [["1"]]*4, [[1., 2.]]*4, [1.]*4])
def test_invalid_scorer_outputs_reject_without_commit(monkeypatch, predictions):
    env = tiny(Request("a", 0, .95, 0, .1))
    fake_pool(monkeypatch, env, count=4)
    before = deepcopy(env.aircraft)
    with pytest.raises(ValueError):
        scheduler.apply_assisted_scheduler(env, "rank_adaptive", SimpleNamespace(kind="rank", predict=lambda rows: predictions))
    assert env.aircraft == before and env.events == []


@pytest.mark.parametrize("bad", [True, 2, 14., "14"])
def test_invalid_pool_budget_type(bad):
    with pytest.raises(ValueError, match="pool_limit"):
        scheduler.apply_assisted_scheduler(tiny(), "beam_adaptive", pool_limit=bad)


def test_single_root_all_arms_no_rollout_no_inference(monkeypatch):
    def forbidden(*args):
        raise AssertionError("single root attempted inference or rollout")
    monkeypatch.setattr(scheduler, "closed_evaluation", forbidden)
    scorer = SimpleNamespace(kind="rank", predict=forbidden)
    for arm in config.ARMS:
        row = scheduler.apply_assisted_scheduler(tiny(), arm, scorer)
        assert row["budget"] == row["work"]["root_rollout_count"] == 0
        assert len(row["selected_root_ids"]) == 1 and row["evaluations"] == {}
        assert row["adaptive"]["quality_certified"] is None and row["adaptive"]["stages"] == []
        assert row["adaptive"]["stop_reason"] == "single_root" and row["timing"]["inference"] == 0.
        assert row["work"]["final_submit_count"] == 1 and row["final_check"]["status"] == "feasible"


def test_final_validation_rejects_malicious_plan_before_emergency(monkeypatch):
    env = tiny(Request("a", 0, .95, 0, .1))
    plan = replace(build_plan(env.aircraft["a"], 0, env.config), wait_cycles=(999,))
    monkeypatch.setattr(scheduler, "make_pool", lambda *args: ([Root("bad", (plan,), [], [], {})],
                                                            {"statistics": asdict(Statistics()), "pool_validation_calls": 1}))
    monkeypatch.setattr(scheduler, "common_emergency_fallback", lambda *args: pytest.fail("emergency after rejection"))
    before = deepcopy(env.aircraft)
    with pytest.raises(RuntimeError, match="final independent batch validation"):
        scheduler.apply_assisted_scheduler(env, "beam_adaptive")
    assert env.aircraft == before and env.events == []


def test_real_full_matches_existing_full_and_serializes_steps(monkeypatch):
    env = tiny(Request("a", 0, .85, 0, .1), horizon=7.5)
    old = deepcopy(env)
    steps = []
    original_step = FixedTerminalEnv.step
    def step(e):
        steps.append(e.time);return original_step(e)
    monkeypatch.setattr(FixedTerminalEnv, "step", step)
    row = scheduler.apply_assisted_scheduler(env, "full_reference")
    assert len(steps) == row["work"]["simulation_steps"] == 8*row["budget"]
    prior = previous.apply_validation_scheduler(old, "full", budget=None)
    assert row["chosen_root_id"] == prior["chosen_root_id"]
    assert row["normal_plans"] == prior["normal_plans"]
    assert row["budget"] == row["pool_count"] and row["configured_budget"] is None
    assert env.aircraft == old.aircraft and env.config == old.config
    assert json.loads(json.dumps(row, allow_nan=False))["adaptive"]["visible_counts"] == {"A": 1, "F0": 0, "E0": 0}
