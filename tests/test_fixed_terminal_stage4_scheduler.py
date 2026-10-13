from copy import deepcopy
from types import SimpleNamespace
from dataclasses import replace
import pytest
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.core.fixed_terminal.learning_scheduler import make_pool, shortlist, select_exact, apply_learning_scheduler, prediction_order
from terminal_marl.core.fixed_terminal.reference import Root, action_key, value_comparison_key
from terminal_marl.core.fixed_terminal.routes import build_plan


def tiny(*requests, horizon=20):
    return FixedTerminalEnv(replace(DEFAULT_CONFIG, horizon=horizon), Scenario(4001, "test", "test", tuple(requests), 1, 2))


def test_shared_pool_anchors_budget_shortlists_and_exact_selection():
    env = tiny(Request("a", 0, .7, 0, .1))
    before = deepcopy(env.aircraft)
    roots, meta = make_pool(env)
    assert len(roots) <= 14 and meta["settings"]["joint_aircraft_limit"] == 8
    assert meta["settings"]["catalogue_scan_limit"] == 48 and meta["settings"]["beam_width"] == 128
    assert {"HOLD", "FCFS"} <= {label for r in roots for label in r.labels}
    best, evaluated = select_exact(env, roots)
    for method, prediction in (("beam_topk", None), ("value_topk", [[0]*7 for _ in roots]), ("rank_topk", [[0] for _ in roots])):
        subset = shortlist(roots, meta, method, prediction)
        assert len(subset) <= 5 and set(meta["anchors"]) <= {r.root_id for r in subset}
        chosen, values = select_exact(env, subset)
        assert value_comparison_key(evaluated[best.root_id]["value"]) >= value_comparison_key(values[chosen.root_id]["value"])
    assert env.aircraft == before
    order = prediction_order(roots, [[0]*7 for _ in roots], "value")
    assert order == [r.root_id for r in sorted(roots, key=lambda r: action_key(r.plans))]


def test_single_root_fastpath_no_inference_or_rollout(monkeypatch):
    import terminal_marl.core.fixed_terminal.learning_scheduler as module
    env = tiny()
    monkeypatch.setattr(module, "select_exact", lambda *a: (_ for _ in ()).throw(AssertionError("rollout called")))
    scorer = SimpleNamespace(kind="rank", predict=lambda *a: (_ for _ in ()).throw(AssertionError("inference called")))
    row = apply_learning_scheduler(env, "rank_topk", scorer)
    assert row["degenerate"] and row["pool_count"] == 1 and row["evaluations"] == {}
    assert row["timing"]["inference"] == row["timing"]["exact_rollout"] == 0
    assert row["final_check"]["status"] == "feasible"


def test_final_public_batch_validation_rejects_malicious_root(monkeypatch):
    import terminal_marl.core.fixed_terminal.learning_scheduler as module
    env = tiny(Request("a", 0, .8, 0, 0))
    plan = build_plan(env.aircraft["a"], 0, env.config)
    malicious = replace(plan, wait_cycles=(999,))
    root = Root("malicious", (malicious,), [], [], {"status": "feasible"})
    monkeypatch.setattr(module, "make_pool", lambda e: ([root], {"statistics": {"examined": 0}}))
    before = deepcopy(env.aircraft)
    with pytest.raises(RuntimeError, match="final independent"):
        apply_learning_scheduler(env, "full_reference")
    assert env.aircraft == before and not env.events


def test_existing_commitments_remain_and_common_emergency_runs(monkeypatch):
    import terminal_marl.core.fixed_terminal.learning_scheduler as module
    env = tiny(Request("a", 0, .8, 0, 0))
    plan = build_plan(env.aircraft["a"], 0, env.config)
    env.submit_batch((plan,))
    calls = []
    original = module.common_emergency_fallback
    def wrapped(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(module, "common_emergency_fallback", wrapped)
    row = apply_learning_scheduler(env, "full_reference")
    assert calls and env.aircraft["a"].plan == plan and row["degenerate"]
