from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest

from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.core.fixed_terminal.baselines import Statistics
from terminal_marl.core.fixed_terminal.reference import Root, action_key, value_comparison_key
from terminal_marl.core.fixed_terminal.routes import build_plan
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.core.fixed_terminal import validation_scheduler as scheduler
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv


def tiny(*requests, horizon=20):
    return FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=horizon), Scenario(5001, "low", "medium", tuple(requests), 1, 2))


def test_total_budget_unique_anchors_refill_and_hybrid():
    roots = [Root(str(i), (SimpleNamespace(aircraft_id=str(i), wait_cycles=(0,)),), [], [], {}) for i in range(8)]
    meta = {"anchors": ["0", "0", "1"], "beam_order": [str(i) for i in range(6)]}
    for budget in (3, 5, 8):
        selected = scheduler.shortlist(roots, meta, "beam", budget)
        assert len(selected) == budget and {"0", "1"} <= {r.root_id for r in selected}
    assert {r.root_id for r in scheduler.shortlist(roots, meta, "beam", 5)} == {"0", "1", "2", "3", "4"}
    prediction = [[float(i)] for i in range(8)]
    assert scheduler.complete_order(roots, meta, "hybrid", prediction)[:4] == ["0", "7", "1", "6"]
    assert {r.root_id for r in scheduler.shortlist(roots, meta, "hybrid", 5, prediction)} == {"0", "1", "2", "6", "7"}
    assert scheduler.shortlist(roots, meta, "full", None) == roots
    with pytest.raises(ValueError, match="anchors"):
        scheduler.shortlist(roots, meta, "beam", 1)


def test_pool_observed_input_immutable_default_matches_stage4(monkeypatch):
    from terminal_marl.core.fixed_terminal.learning_scheduler import make_pool as previous_pool
    env = tiny(Request("a", 0, .8, 0, .1), Request("hidden", 12, .2, None, 0))
    before = deepcopy(env.aircraft)
    original = scheduler.make_catalogue
    def checked(view, *args):
        assert "hidden" not in view.aircraft
        assert [r.aircraft_id for r in view.scenario.requests] == ["a"]
        return original(view, *args)
    monkeypatch.setattr(scheduler, "make_catalogue", checked)
    roots, metadata = scheduler.make_pool(env)
    old, old_meta = previous_pool(env)
    assert [action_key(r.plans) for r in roots] == [action_key(r.plans) for r in old]
    assert metadata["anchors"] == old_meta["anchors"]
    assert env.aircraft == before


def test_closed_full_filtered_same_pool_exact_callcount_and_steps(monkeypatch):
    env = tiny(Request("a", 0, .8, 0, .1), horizon=7.5)
    roots, meta = scheduler.make_pool(env)
    calls, step_calls = [], []
    original_eval, original_step = scheduler.closed_evaluation, FixedTerminalEnv.step
    def counted_eval(e, r):
        calls.append(r.root_id)
        return original_eval(e, r)
    def counted_step(e):
        step_calls.append(e.time)
        return original_step(e)
    monkeypatch.setattr(scheduler, "closed_evaluation", counted_eval)
    monkeypatch.setattr(FixedTerminalEnv, "step", counted_step)
    best, values = scheduler.evaluate_selected(env, roots)
    assert calls == [r.root_id for r in roots]
    assert len(step_calls) == sum(v["work"]["simulation_steps"] for v in values.values()) == 8*len(roots)
    assert all(v["work"]["root_validation_calls"] == 2 for v in values.values())
    selected = scheduler.shortlist(roots, meta, "beam", 3)
    choice, subset = scheduler.evaluate_selected(env, selected)
    assert value_comparison_key(values[best.root_id]["value"]) >= value_comparison_key(subset[choice.root_id]["value"])
    assert len(calls) == len(roots)+len(selected)


def test_closed_statistics_sum_root_and_continuation(monkeypatch):
    env = tiny()
    value = {"J": [0]*7}
    monkeypatch.setattr(scheduler, "evaluate_root", lambda *a, **k: {
        "status": "feasible", "seconds": .1, "values": {"20": value},
        "root_emergency_statistics": asdict(Statistics(generated=2, examined=3)),
        "continuation_decisions": [{"statistics": asdict(Statistics(generated=5, examined=7, joint_assignments=11))},
                                   {"statistics": asdict(Statistics(generated=13, examined=17, joint_assignments=19))}]})
    result = scheduler.closed_evaluation(env, Root("hold", (), [], [], {}))
    assert result["statistics"]["generated"] == 20
    assert result["statistics"]["examined"] == 27 and result["statistics"]["joint_assignments"] == 30
    assert result["continuation_decision_count"] == 2 and "continuation_decisions" not in result


@pytest.mark.parametrize("window", (20., 60.))
def test_rolling_all_selected_roots_once_no_internal_truncation(window):
    env = tiny(Request("a", 0, .8, 0, .1), horizon=75)
    roots = [Root("hold", (), [], [], {})]+[Root(str(i), (build_plan(env.aircraft["a"], 0, env.config, (i,)),), [], [], {}) for i in range(4)]
    before = deepcopy(env.aircraft)
    chosen, results = scheduler.evaluate_selected(env, roots, window)
    assert len(results) == 5 and chosen in roots
    assert all(r["work"]["simulation_steps"] == int(window) and r["work"]["rollouts"] == 1 for r in results.values())
    assert all(r["evaluation_stop"] == window for r in results.values())
    assert env.aircraft == before


def test_single_root_all_methods_no_inference_no_rollout_final_check(monkeypatch):
    monkeypatch.setattr(scheduler, "evaluate_selected", lambda *a: (_ for _ in ()).throw(AssertionError("rollout")))
    scorer = SimpleNamespace(kind="rank", predict=lambda *a: (_ for _ in ()).throw(AssertionError("inference")))
    for method in ("beam", "full", "rolling", "linear_rank", "mlp_rank", "gradient_boost", "value", "hybrid"):
        row = scheduler.apply_validation_scheduler(tiny(), method, scorer, rolling_seconds=20. if method == "rolling" else None)
        assert row["work"]["root_rollout_count"] == 0 and row["work"]["final_submit_count"] == 1
        assert row["timing"]["inference"] == 0 and row["final_check"]["status"] == "feasible"


def test_malicious_final_plan_rejected_and_commitments_preserved(monkeypatch):
    env = tiny(Request("a", 0, .8, 0, 0))
    plan = build_plan(env.aircraft["a"], 0, env.config)
    wrong = replace(plan, wait_cycles=(999,))
    root = Root("bad", (wrong,), [], [], {})
    monkeypatch.setattr(scheduler, "make_pool", lambda *a: ([root], {"statistics": asdict(Statistics()), "pool_validation_calls": 1}))
    before = deepcopy(env.aircraft)
    with pytest.raises(RuntimeError, match="final independent"):
        scheduler.apply_validation_scheduler(env, "full")
    assert env.aircraft == before and not env.events
