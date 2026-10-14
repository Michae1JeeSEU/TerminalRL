from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest

from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs import fixed_terminal_stage7 as config
from terminal_marl.core.fixed_terminal.scenarios import Scenario, Request
from terminal_marl.core.fixed_terminal.safety import Check
from terminal_marl.core.fixed_terminal.routes import build_plan
from terminal_marl.core.fixed_terminal.reference import action_key
from terminal_marl.core.fixed_terminal.reference import Root
from terminal_marl.core.fixed_terminal.baselines import joint_choices, Statistics, observed_clone
from terminal_marl.core.fixed_terminal.validation_scheduler import make_pool
from terminal_marl.core.fixed_terminal import search_scheduler as scheduler
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv


def tiny(*requests, horizon=12.):
    return FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=horizon), Scenario(7001, "low", "high", tuple(requests), 1, 2))


def synthetic_search():
    env = tiny(Request("a", 0, .95, 0, .1), Request("b", 0, .95, 0, 2.))
    state = scheduler._Search(observed_clone(env))
    state.catalogue = {k: tuple(build_plan(env.aircraft[k], 0., env.config, (i,)) for i in range(4)) for k in ("a", "b")}
    state.checked = lambda plans, final=False: (Check("feasible", "synthetic"), False)
    return env, state


def test_fixed_jobs_and_partial_budget_scope():
    assert len(config.scenario_jobs("train")) == 27
    assert len(config.scenario_jobs("val")) == 9
    assert len(config.scenario_jobs("test")) == 33
    assert len(config.panel_scenario_jobs("test")) == 15
    assert config.TEST_SEEDS == (7301, 7302, 7303)
    assert config.TRAINING == {"n_estimators": 100, "max_depth": 3, "learning_rate": .05, "random_state": 7401}


def test_true_partial_width_and_none_lineage():
    _, small = synthetic_search()
    _, full = synthetic_search()
    small.run(3)
    full.run(128)
    assert small.counters["child_attempts"] < full.counters["child_attempts"]
    assert small.stages[0]["layers"][0]["parent_frontier_count"] == 1
    holds = [n for n in small.nodes if n["depth"] == 2 and n["action_signature"] == []]
    assert len(holds) == 1 and holds[0]["selected"]
    assert [x["wait_cycles"] for x in holds[0]["lineage"]] == [None, None]
    assert holds[0]["parent_id"] is not None
    assert all(n["options_count"] == n["candidate_count"]+1 for n in small.nodes if n["depth"] < 2)
    assert all("features" not in n for n in small.nodes)


def test_learning_prunes_before_children_and_dynamic_width_is_layer_budget():
    _, state = synthetic_search()
    state.policy = SimpleNamespace(kind="expand", predict_expand=lambda rows: [0.] * len(rows))
    state.run(8, "hybrid")
    assert [r["width"] for r in state.stages[0]["layers"]] == [3, 3]
    assert state.counters["learned_pruned_nodes"] > 0
    assert state.counters["child_attempts"] == 10  # 5 root children, only HOLD's 5 children.
    assert all(not n["expanded"] for n in state.nodes if n["learned_pruned"])
    assert scheduler.dynamic_width([1., 1., 1.]) == 3
    assert scheduler.dynamic_width([1.] * 5) == 5
    assert scheduler.dynamic_width([1.] * 6) == 8


def test_reference_batch_keys_match_original_finite_beam():
    env = tiny(Request("a", 0, .95, 0, .1))
    state = scheduler._Search(observed_clone(env))
    old = joint_choices(observed_clone(env), state.catalogue, config.REFERENCE_SETTINGS, Statistics())
    roots, stage = state.run(128)
    expected, _ = make_pool(observed_clone(env))
    assert [action_key(r.plans) for r in roots] == [action_key(r.plans) for r in expected]
    assert [action_key(next(r for r in roots if r.root_id == rid).plans) for rid in stage["beam_order"]] == [action_key(x) for x in old[:12]]


def test_fallback_unique_cache_real_checks_and_checker_restoration():
    env = tiny(Request("a", 0, .95, 0, .1))
    checker = env.feasibility
    policy = SimpleNamespace(kind="expand", predict_expand=lambda rows: [float("nan")] * len(rows))
    row = scheduler.apply_search_scheduler(env, "learned_pruning", policy)
    assert row["adaptive"]["fallback_count"] == 1
    assert row["adaptive"]["fallback_reasons"][0].startswith("prediction_error:")
    assert row["work"]["unique_rollouts"] == len(row["evaluations"]) == len(set(row["evaluations"]))
    assert row["work"]["rollout_cache_hits"] >= 1
    assert row["work"]["candidate_checks"] == sum(v for k, v in row["work"].items() if k.endswith("_candidate_checks"))
    assert row["work"]["partial_batch_checks"] + row["work"]["partial_check_cache_hits"] == row["work"]["partial_check_attempts"]
    assert row["final_check"]["status"] == "feasible" and env.feasibility is checker
    assert len(row["search"]["stages"]) == 2
    assert any(len(r["provenance"]) > 1 for r in row["pool"])
    assert sum(row["timing"][k] for k in ("generation", "features", "inference", "exact_rollout", "final_validation_and_emergency")) <= row["timing"]["total"]+.001


def test_normal_none_path_cache_and_no_cross_method_cache():
    env = tiny(Request("a", 0, .95, 0, .1), horizon=1.)
    first = scheduler.apply_search_scheduler(deepcopy(env), "beam_adaptive")
    second = scheduler.apply_search_scheduler(deepcopy(env), "beam_adaptive")
    assert first["work"] == second["work"]
    assert first["work"]["partial_check_cache_hits"] > 0
    assert first["work"]["rollout_cache_hits"] > 0
    assert len(first["search"]["stages"]) == 4
    assert all(len(s["root_ids"]) <= 14 for s in first["search"]["stages"])


def test_later_stage_cannot_discard_better_cached_incumbent(monkeypatch):
    env = tiny(Request("a", 0, .95, 0, .1))
    state = scheduler._Search(observed_clone(env))
    first = Root("first", (build_plan(env.aircraft["a"], 0., env.config, (1,)),), [], [], {})
    later = Root("later", (build_plan(env.aircraft["a"], 0., env.config, (2,)),), [], [], {})
    state.all_roots = {first.root_id: first}
    monkeypatch.setattr(scheduler, "closed_evaluation", lambda view, root: {"status": "feasible", "value": {"J": [0, 1, 0, 0, 0., -1. if root.root_id == "first" else -2., 0.]}})
    assert state.evaluate([first])[0].root_id == "first"
    state.all_roots[later.root_id] = later
    assert state.evaluate([later])[0].root_id == "first"
    assert len(state.evaluations) == 2


def test_final_check_fallback_preserves_cost_and_stage_fields(monkeypatch):
    env = tiny(Request("a", 0, .95, 0, .1))
    original = scheduler._MeteredFeasibility.check_batch
    def injected(self, plans, fleet, now):
        checked = original(self, plans, fleet, now)
        return Check("unknown", "injected_final_check") if self.meter.phase == "pre_submit" else checked
    monkeypatch.setattr(scheduler._MeteredFeasibility, "check_batch", injected)
    policy = SimpleNamespace(kind="expand", predict_expand=lambda rows: [1.] * len(rows))
    row = scheduler.apply_search_scheduler(env, "learned_pruning", policy)
    assert row["adaptive"]["fallback_reasons"] == ["final_check:unknown:injected_final_check"]
    assert all("actual_widths" in stage for stage in row["adaptive"]["stages"])
    assert row["final_check"]["status"] == "feasible"
    assert row["work"]["final_validation_calls"] == 2
    assert sum(row["timing"][k] for k in ("generation", "features", "inference", "exact_rollout", "final_validation_and_emergency")) <= row["timing"]["total"]+.001


@pytest.mark.parametrize("value", [-.1, 1.1, True, "bad"])
def test_invalid_predictions(value):
    with pytest.raises(ValueError):
        scheduler._probabilities(SimpleNamespace(kind="expand", predict_expand=lambda rows: [value]), [[0.]])
