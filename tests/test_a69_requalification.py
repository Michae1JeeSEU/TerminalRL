import pickle
import ast
import time
from types import SimpleNamespace

import numpy as np

from terminal_marl.core.config import SimConfig
from terminal_marl.envs.high_level_wrapper import DynamicHighLevelWrapper
from terminal_marl.training.a65_counterfactual_audit import Candidate, candidate_set
from terminal_marl.training.a69_requalification import _branch, _meaningful, _oracle, evaluate_task


def _row(completed, waiting, *, safe=True):
    return {"safe": safe, "metrics": {"completed": completed, "all_request_waiting_area": waiting, "unresolved_count": 0., "unresolved_age_mean": 0., "unresolved_age_p95": 0., "service_delay_p95": 0., "service_delay_p99": 0., "energy_mean": 0., "priority_fairness_std": 0., "arrival_delay_by_id": {}}, "outcomes": {}, "shield_cost": 0., "candidate_execution": {"capacity_effective": False}}


def test_a69_operational_effect_is_not_labeled_noop():
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, total_requested_uavs=5, max_simulation_seconds=200.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    _, _, _, _, info = wrapper.step(np.asarray((0, 0, 0, 0, 0, 0, 0, -.2, 0, 0), np.float32))
    event = info["high_action"]
    assert event["operational_effective"]
    assert event["execution_outcome"] == "OPERATIONAL_APPLIED"


def test_a69_noop_does_not_change_control_parameters():
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, total_requested_uavs=5, max_simulation_seconds=200.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    before = wrapper.base.airspace.state_vector().copy()
    _, _, _, _, info = wrapper.step(np.zeros(10, np.float32))
    assert info["high_action"]["execution_outcome"] == "NO_OP"
    assert np.array_equal(before, wrapper.base.airspace.state_vector())


def test_a69_meaningful_threshold_is_not_relaxed():
    assert _meaningful({"completed": 1, "waiting_area": 0.})
    assert _meaningful({"completed": 0, "waiting_area": -50_000.})
    assert not _meaningful({"completed": 0, "waiting_area": -49_999.})


def test_a69_oracle_excludes_infeasible_candidate():
    candidates = {"no_op": Candidate("no_op", "NO_OP", (0.,) * 10), "unsafe": Candidate("unsafe", "GEOMETRY", (0.,) * 10), "headway": Candidate("headway", "HEADWAY", (0.,) * 10)}
    oracle = _oracle("s", {"no_op": _row(1, 100.), "unsafe": _row(9, 0., safe=False), "headway": _row(1, 0.)}, candidates)
    assert oracle["best_action"] == "headway"


def test_a69_branch_trace_keeps_legacy_collection_schema():
    """A6.9's richer labels remain consumable by the frozen metric collector."""
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, total_requested_uavs=3, max_simulation_seconds=20.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    row = _branch(pickle.dumps(wrapper, protocol=pickle.HIGHEST_PROTOCOL), candidate_set()[0], 1.)
    event = row["execution_trace"][0]
    assert event["outcome"] == "no_op"
    assert event["time"] == event["event_time"]
    assert "all_request_waiting_area" in row["metrics"]


def test_a69_candidate_set_is_canonical_10d_and_has_capacity_threshold_actions():
    candidates = candidate_set()
    assert len(candidates) == 16
    assert all(len(candidate.action) == 10 for candidate in candidates)
    assert {"outer_capacity_plus1", "middle_capacity_plus1", "inner_capacity_plus1"} <= {candidate.name for candidate in candidates}


def test_a69_runner_does_not_load_or_train_ppo():
    source = open("terminal_marl/training/a69_requalification.py", encoding="utf-8").read().lower()
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not any("ppo" in name or "checkpoint" in name for name in imported)
    assert not {"load_high_level", "train_ppo", "load_checkpoint"} & calls


def test_a69_task_duration_wraps_the_branch(monkeypatch):
    def slow_branch(*_args):
        time.sleep(.01)
        return {"ok": True}

    monkeypatch.setattr("terminal_marl.training.a69_requalification._branch", slow_branch)
    _, _, duration, row = evaluate_task(b"snapshot", "snapshot", candidate_set()[0], 1.)
    assert row == {"ok": True}
    assert duration >= .01


def test_a69_delayed_request_preserves_original_trace(monkeypatch):
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, total_requested_uavs=3, max_simulation_seconds=20.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    for _ in range(2):
        wrapper.base.step()
    coordinator = wrapper.coordinator

    class Verdict:
        def __init__(self, approved): self.approved = approved; self.reason = "predicted_conflict"
    rejected_update = wrapper.base.apply_high_level_action(np.zeros(10, np.float32), geometry_gate=False)
    applied_update = SimpleNamespace(structural_applied=True, capacity_before=(1, 1, 1), capacity_after=(2, 1, 1))
    allow = {"value": False}
    monkeypatch.setattr(coordinator, "_candidate_safety", lambda *_args: (applied_update if allow["value"] else rejected_update, Verdict(allow["value"])))
    monkeypatch.setattr(coordinator, "_repair_options", lambda: iter(()))
    monkeypatch.setattr(coordinator, "_atomic_commit", lambda _action, update: update)
    first = coordinator.coordinate(np.asarray((.6, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32), geometry_gate=True)
    assert first.execution_outcome == "STRUCTURAL_DELAYED"
    allow["value"] = True
    wrapper.base.now += config.dt
    coordinator.pending_until = wrapper.base.now
    coordinator._last_retry_signature = ()
    later = coordinator.advance()
    assert later.request_id == first.request_id
    assert later.original_request_time == first.original_request_time
    assert later.execution_outcome == "STRUCTURAL_DELAYED_THEN_APPLIED"
    assert later.application_delay_seconds == config.dt


def test_a69_unsafe_structural_action_fails_closed(monkeypatch):
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, total_requested_uavs=3, max_simulation_seconds=20.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    coordinator = wrapper.coordinator; before = wrapper.base.airspace.state_vector().copy()

    class Verdict:
        approved = False
        reason = "predicted_conflict"
    update = wrapper.base.apply_high_level_action(np.zeros(10, np.float32), geometry_gate=False)
    monkeypatch.setattr(coordinator, "_candidate_safety", lambda *_args: (update, Verdict()))
    monkeypatch.setattr(coordinator, "_repair_options", lambda: iter(()))
    decision = coordinator.coordinate(np.asarray((.6, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32), geometry_gate=True, allow_delay=False)
    assert decision.execution_outcome == "SAFETY_REJECTED"
    assert np.allclose(decision.applied_action, 0.)
    assert np.array_equal(before, wrapper.base.airspace.state_vector())


def test_a69_noop_crn_replay_is_deterministic():
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, total_requested_uavs=3, max_simulation_seconds=20.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    snapshot = pickle.dumps(wrapper, protocol=pickle.HIGHEST_PROTOCOL)
    first = _branch(snapshot, candidate_set()[0], 1.)
    second = _branch(snapshot, candidate_set()[0], 1.)
    assert first["metrics"] == second["metrics"]
    assert first["candidate_execution"] == second["candidate_execution"]
