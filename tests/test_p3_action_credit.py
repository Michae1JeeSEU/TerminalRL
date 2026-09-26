from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.core.low_level_joint_action_v4_2 import HierarchicalActionV42
from terminal_marl.training import diagnose_p3_action_credit as runner
from terminal_marl.training.p3_action_credit import branch_inputs, summarize_branches


def _actions_and_shortlists():
    actions = {
        "1": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "2": np.asarray([1.0, 0.1, -0.2], dtype=np.float32),
    }
    shortlists = {
        "2": tuple(HierarchicalActionV42(1, index).normalized for index in (2, 3, 4)),
    }
    return actions, shortlists


def test_branch_inputs_copy_and_preserve_other_uavs():
    actions, shortlists = _actions_and_shortlists()
    actions_before = {key: value.copy() for key, value in actions.items()}
    shortlists_before = {key: tuple(value.copy() for value in values) for key, values in shortlists.items()}
    branch_actions, branch_shortlists = branch_inputs(actions, shortlists, 2, None, np.zeros(25))

    assert np.array_equal(actions["2"], actions_before["2"])
    assert np.array_equal(branch_actions["1"], actions_before["1"])
    assert np.array_equal(actions["1"], actions_before["1"])
    assert set(shortlists) == set(shortlists_before)
    for key in shortlists:
        assert all(np.array_equal(left, right) for left, right in zip(shortlists[key], shortlists_before[key]))
    assert "2" not in branch_shortlists
    assert not np.array_equal(branch_actions["2"], actions["2"])


def test_release_shortlist_places_sampled_proposal_first_and_stable_ties():
    actions, shortlists = _actions_and_shortlists()
    logits = np.zeros(25, dtype=np.float64)
    logits[3] = 5.0
    # With equal remaining logits, stable order is 0, 1, ...; proposal 3 is
    # explicitly first even though it is already the highest logit.
    branch_actions, branch_shortlists = branch_inputs(actions, shortlists, 2, 3, logits)
    decoded = [HierarchicalActionV42(1, index).normalized for index in (3, 0, 1)]
    assert np.array_equal(branch_actions["2"], decoded[0])
    assert all(np.array_equal(left, right) for left, right in zip(branch_shortlists["2"], decoded))
    assert np.array_equal(actions["1"], branch_actions["1"])


def test_singleton_release_keeps_other_shortlists_and_only_requested_target():
    actions, shortlists = _actions_and_shortlists()
    branch_actions, branch_shortlists = branch_inputs(
        actions, shortlists, 2, 7, np.zeros(25), shortlist_k=1
    )
    assert np.array_equal(branch_actions["1"], actions["1"])
    assert all(
        np.array_equal(left, right)
        for left, right in zip(branch_shortlists["2"], (HierarchicalActionV42(1, 7).normalized,))
    )
    assert len(branch_shortlists["2"]) == 1
    assert np.array_equal(shortlists["2"][0], HierarchicalActionV42(1, 2).normalized)


def test_terminal_bootstrap_and_separate_weighted_reward_value_terms():
    wait = {"reward": 1.0, "next_value": 4.0, "terminated": False}
    releases = [{
        "reward": 3.0,
        "next_value": 8.0,
        "terminated": index == 0,
        "decision": "approved" if index == 0 else "rejected",
    } for index in range(25)]
    probabilities = np.full(25, 1.0 / 25.0)
    result = summarize_branches(wait, releases, probabilities)
    expected_next = (0.0 + 24.0 * 8.0) / 25.0
    assert result["expected_release_reward"] == pytest.approx(3.0)
    assert result["delta_reward"] == pytest.approx(2.0)
    assert result["discounted_delta_value"] == pytest.approx(0.99 * (expected_next - 4.0))
    assert result["delta_q"] == pytest.approx(result["delta_reward"] + result["discounted_delta_value"])
    assert result["approved_mass"] == pytest.approx(1.0 / 25.0)
    assert result["wait_q"] == pytest.approx(1.0 + 0.99 * 4.0)


def test_probability_shape_and_finiteness_are_checked():
    wait = {"reward": 0.0, "next_value": 0.0}
    releases = [{"reward": 0.0, "next_value": 0.0, "terminated": False, "decision": "rejected"}] * 25
    with pytest.raises(ValueError):
        summarize_branches(wait, releases, np.ones((5, 5)))
    bad = np.ones(25)
    bad[4] = np.nan
    with pytest.raises(ValueError):
        summarize_branches(wait, releases, bad)


def test_real_terminal_requires_arrivals_exhausted():
    env = SimpleNamespace(
        arrivals=SimpleNamespace(exhausted=lambda: False),
        active={},
        admission=SimpleNamespace(queue=SimpleNamespace(items=[])),
        service_uavs={},
    )
    assert runner._real_terminal(env) is False


def test_branch_result_uses_rollout_terminal_when_other_uav_remains(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.active = {1: object(), 2: object()}
            self.service_uavs = {}
            self.admission = SimpleNamespace(queue=SimpleNamespace(items=[]))
            self.arrivals = SimpleNamespace(exhausted=lambda: True)
            self.telemetry = {
                "collisions": 0,
                "nmac": 0,
                "hard_separation_violations": 0,
                "uncertified_execution": 0,
            }

        def step(self, actions, proposal_shortlists=None):
            return (
                {"1": np.zeros(88, dtype=np.float32), "2": np.zeros(88, dtype=np.float32)},
                {"1": 1.0, "2": 0.0},
                {"1": True, "2": False},
                {"1": False, "2": False},
                {
                    "reward_components": {"1": {"total": 1.0}},
                    "release_decisions": {"1": "approved"},
                },
            )

    monkeypatch.setattr(runner.diagnostic, "_restore_shadow_methods", lambda env: None)
    monkeypatch.setattr(runner.student_rollout_v4_2, "world_value", lambda policy, world: 2.0)
    base = FakeEnv()
    result = runner._branch_result(
        base,
        {"policy": object()},
        {"1": np.zeros(3, dtype=np.float32)},
        {},
        1,
        None,
        np.zeros(25, dtype=np.float32),
        runner._rng_snapshot(),
    )
    assert result["target_terminated"] is True
    assert result["terminated"] is False


def test_run_selected_captures_replayed_step_inputs(tmp_path, monkeypatch):
    tape = tmp_path / "tape.json"
    tape.write_text(json.dumps({"action_tape": [{"step": index} for index in range(4)]}), encoding="utf-8")
    captured = {}

    class FakeEnv:
        def __init__(self, config):
            self.step_count = 3
            self.active = {7: SimpleNamespace(arrival_id=70)}

        def _observations(self):
            return {"7": np.zeros(88, dtype=np.float32)}

        def step(self, actions, proposal_shortlists=None):
            captured["actions"] = actions
            captured["shortlists"] = proposal_shortlists
            return self._observations(), {}, {}, {}, {}

    def replay_action_tape(env, path):
        env.step(
            {"7": np.asarray([0.0, 1.0, 0.0], dtype=np.float32)},
            {"7": (np.asarray([0.0, 1.0, 0.0], dtype=np.float32),)},
        )
        return {"exact": True, "steps_replayed": 4, "steps_expected": 4}

    monkeypatch.setattr(runner, "TerminalParallelEnv", FakeEnv)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", staticmethod(replay_action_tape))
    monkeypatch.setattr(runner.SafetyTraceRecorder, "_state", staticmethod(lambda env: {"step": env.step_count}))
    monkeypatch.setattr(runner.collector, "_summary_outcomes", lambda path: {})
    monkeypatch.setattr(runner.collector, "_replay_outcomes", lambda env: {})
    monkeypatch.setattr(
        runner,
        "_evaluate_branches",
        lambda base_env, row, source_policy, actions, shortlists, branch_rng: captured.update(
            {"branch_actions": actions, "branch_shortlists": shortlists}
        ) or {"wait": {}, "releases": [], "summary": {}},
    )
    row = {
        "seed": 873101,
        "step": 3,
        "uid": 7,
        "arrival_id": 70,
        "observation": [0.0] * 88,
        "selection_reason": "test",
        "q_k3": 0.0,
        "source_proposal_logits": [0.0] * 25,
        "a_requested": {"7": [9.0, 9.0, 9.0]},
    }
    result = runner._run_selected(
        row,
        {"action_tape": str(tape), "scenario": "flow6"},
        None,
        {"policy": object()},
        smoke=False,
    )
    assert result["replay"]["exact"] is True
    assert np.array_equal(captured["actions"]["7"], np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.array_equal(captured["shortlists"]["7"][0], np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.array_equal(captured["branch_actions"]["7"], captured["actions"]["7"])


def test_evaluate_branches_restores_rng_on_branch_error(monkeypatch):
    monkeypatch.setattr(runner.SafetyTraceRecorder, "_state", staticmethod(lambda env: {"stub": True}))
    monkeypatch.setattr(runner.collector, "_digest", lambda value: "digest")
    monkeypatch.setattr(runner.collector.evaluator, "_model_digest", lambda policy: "model")

    def fail_branch(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        raise RuntimeError("branch failure")

    monkeypatch.setattr(runner, "_branch_result", fail_branch)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    before = runner._rng_snapshot()
    with pytest.raises(RuntimeError, match="branch failure"):
        runner._evaluate_branches(
            object(),
            {"uid": 1, "source_proposal_logits": [0.0] * 25, "proposal_probabilities": [1.0 / 25.0] * 25},
            {"policy": object()},
            {},
            {},
            before,
        )
    after = runner._rng_snapshot()
    assert runner._rng_equal(before, after)


def test_runtime_labels_use_singleton_approval_and_keep_catalogue_scope(monkeypatch):
    monkeypatch.setattr(runner.SafetyTraceRecorder, "_state", staticmethod(lambda env: {"stub": True}))
    monkeypatch.setattr(runner.collector, "_digest", lambda value: "digest")
    monkeypatch.setattr(runner.collector.evaluator, "_model_digest", lambda policy: "model")
    calls = []

    def fake_branch(
        base_env,
        source_policy,
        actions,
        shortlists,
        uid,
        proposal,
        logits,
        rng_before,
        *,
        shortlist_k=3,
    ):
        calls.append((proposal, shortlist_k))
        assert "2" in actions
        assert "2" in shortlists
        if proposal is None:
            decision = None
        elif shortlist_k == 3 and proposal == 4:
            decision = "approved"
        elif shortlist_k == 1 and proposal == 7:
            decision = "approved"
        else:
            decision = "safety_reject"
        return {
            "proposal": proposal,
            "reward": 0.0,
            "next_value": 0.0,
            "terminated": False,
            "decision": decision,
        }

    monkeypatch.setattr(runner, "_branch_result", fake_branch)
    row = {
        "uid": 1,
        "source_proposal_logits": [0.0] * 25,
        "proposal_probabilities": [1.0 / 25.0] * 25,
        "certified_proposals": [4],
    }
    result = runner._evaluate_branches(
        object(),
        row,
        {"policy": object()},
        {"1": np.zeros(3, dtype=np.float32), "2": np.zeros(3, dtype=np.float32)},
        {"2": (np.zeros(3, dtype=np.float32),)},
        runner._rng_snapshot(),
    )
    assert len(calls) == 51
    assert calls[:26] == [(None, 3)] + [(index, 3) for index in range(25)]
    assert calls[26:] == [(index, 1) for index in range(25)]
    assert result["catalogue_certified_proposals"] == [4]
    assert result["runtime_certified_proposals"] == [7]
    assert result["catalogue_false_positive_indices"] == [4]
    assert result["catalogue_false_negative_indices"] == [7]
    assert len(result["runtime_singleton_branches"]) == 25
    assert result["q_k3_runtime_labels"] == pytest.approx(1.0 / 25.0)
    assert result["summary"]["approved_mass"] == pytest.approx(1.0 / 25.0)
    assert result["runtime_q_vs_approved_mass_abs_error"] == pytest.approx(0.0, abs=1e-8)
    assert result["label_scope"] == "same_prestate_fixed_other_tape_actions_runtime_singleton"
