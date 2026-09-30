from __future__ import annotations

import copy
import json
import pickle
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from terminal_marl.core.low_level_joint_action_v4_2 import (
    RELEASE_GATE,
    WAIT_GATE,
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.training import paired_gate_diagnostic as diagnostic
from terminal_marl.training import student_rollout_v4_2
from terminal_marl.algo.joint_ppo_v4_2 import JointPPOV42
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.run_v4_2_hierarchical_s0_qualification import _config


class _TinyModel(nn.Module):
    def __init__(self, proposal_logits: np.ndarray | None = None) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 4)
        self.gate = nn.Linear(4, 2)
        self.proposal = nn.Linear(4, 25)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.copy_(torch.tensor([-10.0, 10.0]))
            self.proposal.weight.zero_()
            if proposal_logits is None:
                self.proposal.bias.zero_()
            else:
                self.proposal.bias.copy_(torch.as_tensor(proposal_logits, dtype=torch.float32))

    def distributions(self, observations: torch.Tensor):
        encoded = torch.tanh(self.encoder(observations))
        return (
            torch.distributions.Categorical(logits=self.gate(encoded)),
            torch.distributions.Categorical(logits=self.proposal(encoded)),
        )


class _TinyPolicy:
    observation_dim = 2
    device = torch.device("cpu")

    def __init__(self, proposal_logits: np.ndarray | None = None) -> None:
        self.model = _TinyModel(proposal_logits)


def _details(policy, env, observations, *, deterministic, shortlist_k):
    uids = sorted(observations)
    world = np.asarray([observations[uid] for uid in uids], dtype=np.float32)
    actor = np.ones(len(uids), dtype=bool)
    return (
        uids,
        world,
        actor,
        {},
        np.ones(len(uids), dtype=int),
        np.zeros(len(uids), dtype=int),
        np.zeros(len(uids)),
        np.zeros(len(uids)),
        {},
        {},
    )


class _FakeArrivals:
    def __init__(self, env):
        self.env = env

    def exhausted(self):
        return self.env.step_count >= 3

    def requests_until(self, _now):
        if self.env.step_count >= 3:
            return []
        return [
            SimpleNamespace(
                arrival_id=100 + self.env.step_count,
                request_time=float(self.env.step_count),
                initial_soc=0.9,
                base_priority=1,
                gate_id=0,
                mission_type=SimpleNamespace(value="nominal"),
            )
        ]


class _FakeEnv:
    def __init__(self):
        self.step_count = 0
        self.now = 0.0
        self._paired_seed = 91
        self.active = {1: object(), 2: object()}
        self.service_uavs = {}
        self.admission = SimpleNamespace(queue=SimpleNamespace(items=[]))
        self.arrivals = _FakeArrivals(self)
        self.ledger = SimpleNamespace(requested=set(), outcomes={})
        self.request_times = {}
        self.resolution_times = {}
        self.telemetry = {"collisions": 0.0, "nmac": 0.0}
        self.conflict_events = []
        self.selected_proposal = None

    def _observations(self):
        return {
            str(uid): np.asarray([float(uid), 0.0], dtype=np.float32)
            for uid in self.active
        }

    def step(self, actions, proposal_shortlists=None):
        del proposal_shortlists
        for request in self.arrivals.requests_until(self.now):
            self.ledger.requested.add(int(request.arrival_id))
            self.request_times[int(request.arrival_id)] = float(request.request_time)
        target = actions.get("1", actions.get(1))
        if target is not None:
            decoded = hierarchical_from_normalized_v4_2(target)
            if self.step_count == 0:
                self.selected_proposal = decoded.proposal if decoded.gate == RELEASE_GATE else None
        self.step_count += 1
        self.now = float(self.step_count)
        terminal_at = 2 if self.selected_proposal in {None, 0} else 3
        if self.step_count >= terminal_at:
            self.active = {}
            for arrival_id in list(self.ledger.requested):
                self.ledger.outcomes[arrival_id] = "completed"
                self.resolution_times[arrival_id] = self.now
        reward = float(self.selected_proposal or 0) if self.step_count >= 2 else 0.0
        following = self._observations()
        rewards = {"1": reward, "2": 0.0}
        terminated = {"1": not self.active, "2": not self.active}
        truncated = {"1": False, "2": False}
        info = {
            "time": self.now,
            "release_decisions": {"1": "approved"},
            "release_intent_telemetry": {"1": {"runtime_shortlist_rank": 1}},
            "reward_components": {"1": {"total": reward}},
        }
        return following, rewards, terminated, truncated, info


class _ImmediateArrivals(_FakeArrivals):
    def exhausted(self):
        return self.env.step_count >= 1


class _FirstStepRewardEnv(_FakeEnv):
    def __init__(self):
        super().__init__()
        self.arrivals = _ImmediateArrivals(self)

    def step(self, actions, proposal_shortlists=None):
        del proposal_shortlists
        for request in self.arrivals.requests_until(self.now):
            self.ledger.requested.add(int(request.arrival_id))
            self.request_times[int(request.arrival_id)] = float(request.request_time)
        target = actions.get("1", actions.get(1))
        selected = None
        if target is not None:
            decoded = hierarchical_from_normalized_v4_2(target)
            selected = decoded.proposal if decoded.gate == RELEASE_GATE else None
        self.selected_proposal = selected
        self.step_count = 1
        self.now = 1.0
        self.active = {}
        for arrival_id in list(self.ledger.requested):
            self.ledger.outcomes[arrival_id] = "completed"
            self.resolution_times[arrival_id] = self.now
        reward = float(selected or 0)
        return {}, {"1": reward}, {"1": True}, {"1": False}, {
            "time": self.now,
            "release_decisions": {"1": "approved"},
            "release_intent_telemetry": {"1": {"runtime_shortlist_rank": 1}},
            "reward_components": {"1": {"total": reward}},
        }


class _NaturalFailureEnv(_FirstStepRewardEnv):
    def step(self, actions, proposal_shortlists=None):
        following, rewards, terminated, truncated, info = super().step(actions, proposal_shortlists)
        for arrival_id in list(self.ledger.outcomes):
            self.ledger.outcomes[arrival_id] = "failed_energy"
        self.energy_outcomes = [{"arrival_id": arrival_id, "outcome": "failed_energy"}
                                for arrival_id in self.ledger.outcomes]
        return following, rewards, terminated, truncated, info


class _FallbackDecisionEnv(_FakeEnv):
    def step(self, actions, proposal_shortlists=None):
        following, rewards, terminated, truncated, info = super().step(actions, proposal_shortlists)
        if self.step_count == 1:
            target = actions.get("1", actions.get(1))
            decoded = hierarchical_from_normalized_v4_2(target)
            if decoded.proposal == 4:
                info["release_decisions"]["1"] = "approved"
                info["release_intent_telemetry"]["1"]["runtime_shortlist_rank"] = 2
            elif decoded.proposal == 5:
                info["release_decisions"]["1"] = "future_reserved"
                info["release_intent_telemetry"]["1"]["runtime_shortlist_rank"] = 1
        return following, rewards, terminated, truncated, info


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _same_rng(left, right):
    return (
        left[0] == right[0]
        and left[1][0] == right[1][0]
        and np.array_equal(left[1][1], right[1][1])
        and left[1][2:] == right[1][2:]
        and torch.equal(left[2], right[2])
    )


def test_keyed_request_is_uid_stable_and_rng_neutral(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)
    policy = _TinyPolicy()
    small = _FakeEnv()
    small.active = {1: object(), 2: object()}
    before = _rng_state()
    first = diagnostic.keyed_request(policy, small)
    after = _rng_state()
    larger = _FakeEnv()
    larger.active = {0: object(), 1: object(), 2: object()}
    second = diagnostic.keyed_request(policy, larger)
    assert _same_rng(before, after)
    assert np.array_equal(first.actions["1"], second.actions["1"])
    assert np.array_equal(first.actions["2"], second.actions["2"])
    assert diagnostic.keyed_uniform(7, 3, "1", "gate") == diagnostic.keyed_uniform(7, 3, "1", "gate")
    assert diagnostic.keyed_uniform(7, 3, "1", "gate") != diagnostic.keyed_uniform(7, 3, "1", "proposal")


def test_keyed_request_uses_sampled_proposal_first_and_stable_ties(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)
    policy = _TinyPolicy(np.zeros(25, dtype=np.float32))
    request = diagnostic.keyed_request(policy, _FakeEnv())
    action = hierarchical_from_normalized_v4_2(request.actions["1"])
    assert action.gate == RELEASE_GATE
    shortlist = [hierarchical_from_normalized_v4_2(value).proposal for value in request.shortlists["1"]]
    assert shortlist[0] == action.proposal
    assert shortlist[1:] == [value for value in range(25) if value != action.proposal][:2]


def test_paired_gate_weights_all_proposals_and_preserves_other_first_action(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)
    policy = _TinyPolicy(np.zeros(25, dtype=np.float32))
    env = _FakeEnv()
    first_actions = {
        "1": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "2": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    }
    report = diagnostic.paired_gate_value(
        policy,
        env,
        first_actions,
        {},
        1,
        [101],
        5,
    )
    assert len(report["wait"]) == 1
    assert len(report["releases"]) == 25
    values = [row["suffix_undiscounted_reward"] for row in report["releases"]]
    assert report["expected_release_suffix_undiscounted_reward"] == pytest.approx(24.0)
    assert report["expected_release_undiscounted_return"] == pytest.approx(24.0)
    assert report["expected_release_suffix_undiscounted_reward"] != max(values)
    assert all(row["first_step"]["reward"] == pytest.approx(0.0) for row in report["releases"])
    assert next(row for row in report["releases"] if row["proposal"] == 24)["undiscounted_return"] > next(
        row for row in report["releases"] if row["proposal"] == 1
    )["undiscounted_return"]
    other = [row["first_step"]["actions"]["2"] for row in report["releases"]]
    assert all(np.array_equal(value, other[0]) for value in other)
    assert report["terminal_complete"] is True
    assert "qualified_delta_Q" in report
    assert report["qualified_delta_Q"] is not None
    assert report["full_return_available"] is True
    assert report["model_digest"]
    assert report["continuation_policy_contract"] == "same_frozen_policy_for_first_q_and_all_continuations"
    assert report["mask_schema_contract"]["paired_value_composition"] == "same_frozen_policy_for_first_q_and_all_continuations"
    assert len(report["release_coverage_by_repeat"]) == 1
    assert np.isscalar(report["release_coverage_by_repeat"][0]["p_release"])
    assert len(report["release_coverage_by_repeat"][0]["proposal_probabilities"]) == 25
    json.dumps(report, allow_nan=False)


def test_paired_gate_qualified_delta_uses_full_return_when_only_first_step_differs(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)
    policy = _TinyPolicy(np.zeros(25, dtype=np.float32))
    report = diagnostic.paired_gate_value(
        policy,
        _FirstStepRewardEnv(),
        {"1": np.zeros(3, dtype=np.float32), "2": np.zeros(3, dtype=np.float32)},
        {},
        1,
        [103],
        2,
    )
    assert report["expected_release_suffix_undiscounted_reward"] == pytest.approx(0.0)
    assert report["expected_release_undiscounted_return"] == pytest.approx(12.0)
    assert report["qualified_delta_Q"] == pytest.approx(12.0)


def test_paired_gate_retains_branch_errors_and_censors_qualified_value(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)
    policy = _TinyPolicy(np.zeros(25, dtype=np.float32))
    original = _FakeEnv.step

    def fail_selected(self, actions, proposal_shortlists=None):
        if self.step_count == 0:
            selected = hierarchical_from_normalized_v4_2(actions["1"])
            if selected.proposal == 3:
                raise RuntimeError("synthetic branch failure")
        return original(self, actions, proposal_shortlists)

    monkeypatch.setattr(_FakeEnv, "step", fail_selected)
    report = diagnostic.paired_gate_value(
        policy,
        _FakeEnv(),
        {"1": np.asarray([0.0, 0.0, 0.0], dtype=np.float32), "2": np.zeros(3, dtype=np.float32)},
        {},
        1,
        [102],
        5,
    )
    failed = next(row for row in report["releases"] if row["proposal"] == 3)
    assert failed["error"]["type"] == "RuntimeError"
    assert report["censored"] is True
    assert report["qualified_delta_Q"] is None
    assert report["expected_release_undiscounted_return"] is None
    assert report["expected_release_suffix_undiscounted_reward"] is None
    assert report["expected_release_suffix_by_repeat"] is None
    assert report["partial_expected_release_accumulated_return"] is not None
    expected_partial = sum(
        float(row["undiscounted_return"]) * float(weight)
        for row, weight in zip(report["releases"], report["proposal_q"])
    )
    assert report["partial_expected_release_accumulated_return"] == pytest.approx(expected_partial)
    assert failed["termination_classification"] == "implementation_error"
    assert "unresolved" in failed
    assert "queue_count" in failed
    assert "service_count" in failed
    assert "remaining_soc" in failed
    assert "conflict_events" in failed


def test_paired_gate_natural_energy_failure_is_terminal_and_budget_is_censored(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)
    policy = _TinyPolicy(np.zeros(25, dtype=np.float32))
    natural = diagnostic.paired_gate_value(
        policy,
        _NaturalFailureEnv(),
        {"1": np.zeros(3, dtype=np.float32), "2": np.zeros(3, dtype=np.float32)},
        {},
        1,
        [107],
        2,
    )
    assert all(row["termination_classification"] == "natural_terminal" for row in natural["branches"])
    assert all(row["censored"] is False for row in natural["branches"])
    assert all(row["outcome_counts"]["failed"] == 1 for row in natural["branches"])
    budget = diagnostic.paired_gate_value(
        policy,
        _FakeEnv(),
        {"1": np.zeros(3, dtype=np.float32), "2": np.zeros(3, dtype=np.float32)},
        {},
        1,
        [108],
        1,
    )
    assert all(row["termination_classification"] == "computation_budget_censored" for row in budget["branches"])
    assert budget["expected_release_undiscounted_return"] is None
    assert budget["qualified_delta_Q"] is None


def test_paired_gate_coverage_uses_approved_decision_even_with_fallback_rank(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)
    policy = _TinyPolicy(np.zeros(25, dtype=np.float32))
    report = diagnostic.paired_gate_value(
        policy,
        _FallbackDecisionEnv(),
        {"1": np.zeros(3, dtype=np.float32), "2": np.zeros(3, dtype=np.float32)},
        {},
        1,
        [109],
        5,
    )
    coverage = report["release_coverage_by_repeat"][0]
    row4 = next(row for row in report["releases"] if row["proposal"] == 4)
    row5 = next(row for row in report["releases"] if row["proposal"] == 5)
    assert row4["first_step"]["runtime_shortlist_rank"] == 2
    assert row4["first_step"]["actual_immediate_approval"] is True
    assert coverage["actual_immediate_approval"][4] is True
    assert np.isscalar(coverage["p_release"])
    assert coverage["p_times_C"] == pytest.approx(report["p_release"] * coverage["C_i"])
    assert coverage["proposal_weighted_approval_terms"][4] == pytest.approx(coverage["proposal_probabilities"][4])
    assert row5["first_step"]["future_reservation"] is True
    assert coverage["actual_immediate_approval"][5] is False


def test_paired_gate_preserves_safety_invariant_diagnostic_and_metrics(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)
    from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError

    def fail_with_safety(self, actions, proposal_shortlists=None):
        del actions, proposal_shortlists
        self.telemetry["safety_interventions"] = self.telemetry.get("safety_interventions", 0.0) + 1.0
        self.telemetry["collisions"] += 1
        self.conflict_events.append({"kind": "collision", "time": self.now})
        raise SafetyInvariantError("blocked", diagnostic={"reason": "no certified action"})

    monkeypatch.setattr(_FakeEnv, "step", fail_with_safety)
    report = diagnostic.paired_gate_value(
        _TinyPolicy(np.zeros(25, dtype=np.float32)),
        _FakeEnv(),
        {"1": np.zeros(3, dtype=np.float32), "2": np.zeros(3, dtype=np.float32)},
        {},
        1,
        [110],
        2,
    )
    failed = report["releases"][0]
    assert failed["error"]["type"] == "SafetyInvariantError", failed
    assert failed["error"]["diagnostic"]["reason"] == "no certified action"
    assert failed["safety_counter_delta"]["collisions"] == pytest.approx(1.0)
    assert failed["safety_counter_delta"]["safety_interventions"] == pytest.approx(1.0)
    assert failed["safety_violation_status"] == "violation"
    assert failed["safety_invariant_error"] is True
    assert failed["conflict_events"][0]["kind"] == "collision"
    assert report["qualified_delta_Q"] is None


def test_paired_gate_rejects_nonfinite_rewards(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)

    def nan_reward(self, actions, proposal_shortlists=None):
        del actions, proposal_shortlists
        self.step_count += 1
        self.now = float(self.step_count)
        return self._observations(), {"1": float("nan")}, {"1": False}, {"1": False}, {}

    monkeypatch.setattr(_FakeEnv, "step", nan_reward)
    report = diagnostic.paired_gate_value(
        _TinyPolicy(np.zeros(25, dtype=np.float32)),
        _FakeEnv(),
        {"1": np.zeros(3, dtype=np.float32), "2": np.zeros(3, dtype=np.float32)},
        {},
        1,
        [111],
        2,
    )
    assert all(row["error"]["type"] == "ValueError" for row in report["branches"])
    assert all("nonfinite reward" in row["error"]["message"] for row in report["branches"])
    assert report["qualified_delta_Q"] is None


def test_paired_gate_branch_callback_isolated_and_complete_across_natural_and_error_branches(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)
    original = _FakeEnv.step

    def fail_selected(self, actions, proposal_shortlists=None):
        if self.step_count == 0:
            selected = hierarchical_from_normalized_v4_2(actions["1"])
            if selected.proposal == 3:
                raise RuntimeError("callback branch fixture failure")
        return original(self, actions, proposal_shortlists)

    monkeypatch.setattr(_FakeEnv, "step", fail_selected)
    callback_rows = []

    def callback(row, completed, total):
        row["callback_mutated"] = True
        row["first_step"]["callback_mutated"] = True
        callback_rows.append((row, completed, total))

    report = diagnostic.paired_gate_value(
        _TinyPolicy(np.zeros(25, dtype=np.float32)),
        _FakeEnv(),
        {"1": np.zeros(3, dtype=np.float32), "2": np.zeros(3, dtype=np.float32)},
        {},
        1,
        [114],
        5,
        branch_callback=callback,
    )
    assert len(callback_rows) == 26
    assert [completed for _, completed, _ in callback_rows] == list(range(1, 27))
    assert all(total == 26 for _, _, total in callback_rows)
    assert any(row.get("error", {}).get("type") == "RuntimeError" for row, _, _ in callback_rows)
    assert any(row["termination_classification"] == "natural_terminal" for row, _, _ in callback_rows)
    assert all("callback_mutated" not in row for row in report["branches"])
    assert all("callback_mutated" not in row["first_step"] for row in report["branches"])


def test_paired_gate_branch_callback_error_propagates_and_restores_rng(monkeypatch):
    monkeypatch.setattr(student_rollout_v4_2, "_student_request_details", _details)
    policy = _TinyPolicy(np.zeros(25, dtype=np.float32))
    before = _rng_state()

    def callback(_row, _completed, _total):
        raise RuntimeError("branch writer failed")

    with pytest.raises(RuntimeError, match="branch writer failed"):
        diagnostic.paired_gate_value(
            policy,
            _FakeEnv(),
            {"1": np.zeros(3, dtype=np.float32), "2": np.zeros(3, dtype=np.float32)},
            {},
            1,
            [115],
            2,
            branch_callback=callback,
        )
    assert _same_rng(before, _rng_state())


def test_gate_update_alignment_reports_finite_local_prediction_and_no_mutation():
    before = JointPPOV42(2)
    after = copy.deepcopy(before)
    with torch.no_grad():
        after.model.gate_head.bias.add_(torch.tensor([0.01, -0.01]))
    before_state = {key: value.detach().clone() for key, value in before.model.state_dict().items()}
    after_state = {key: value.detach().clone() for key, value in after.model.state_dict().items()}
    before_flags = {name: parameter.requires_grad for name, parameter in before.model.named_parameters()}
    for parameter in before.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before_grads = {name: parameter.grad.detach().clone() for name, parameter in before.model.named_parameters()}
    rng_before = _rng_state()
    report = diagnostic.gate_update_alignment(
        before,
        after,
        np.asarray([[0.0, 0.0], [1.0, 0.5]], dtype=np.float32),
        delta_q=0.25,
    )
    assert report["rows"] == 2
    assert all(np.isfinite(report["linear_approximation"]))
    assert len(report["sign_match"]) == 2
    assert report["delta_q"] == pytest.approx(0.25)
    assert len(report["exact_linear_residual"]) == 2
    assert report["local_m"] == pytest.approx([value * 0.25 for value in report["actual_probability_delta"]])
    assert report["proposal_unchanged"] is True
    assert report["proposal_probability_unchanged"] is True
    assert report["mask_schema_contract"]["paired_value_composition"] == "same_frozen_policy_for_first_q_and_all_continuations"
    assert report["model_unchanged"] is True
    assert report["rng_unchanged"] is True
    assert _same_rng(rng_before, _rng_state())
    assert all(torch.equal(before_state[key], before.model.state_dict()[key]) for key in before_state)
    assert all(torch.equal(after_state[key], after.model.state_dict()[key]) for key in after_state)
    assert all(parameter.requires_grad == before_flags[name] for name, parameter in before.model.named_parameters())
    assert all(torch.equal(parameter.grad, before_grads[name]) for name, parameter in before.model.named_parameters())


def test_gate_update_alignment_handles_frozen_before_parameters():
    before = JointPPOV42(2)
    after = copy.deepcopy(before)
    with torch.no_grad():
        after.model.gate_head.bias.add_(torch.tensor([0.01, -0.01]))
    for parameter in before.model.parameters():
        parameter.requires_grad_(False)
    report = diagnostic.gate_update_alignment(
        before,
        after,
        np.asarray([[0.0, 0.0]], dtype=np.float32),
    )
    assert report["model_unchanged"] is True
    assert np.isfinite(report["linear_approximation"]).all()
    assert np.isfinite(report["exact_linear_residual"]).all()
    assert all(not parameter.requires_grad for parameter in before.model.parameters())


def test_real_terminal_parallel_env_keyed_request_two_tick_smoke():
    config = replace(
        _config(),
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        teacher_mode="teacher_label_only",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        warm_start_uavs=1,
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        max_simulation_seconds=3.0,
        runtime_proposal_shortlist_k=3,
    )
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(104)
    policy = JointPPOV42(88)
    request = diagnostic.keyed_request(policy, env)
    following, rewards, terminated, truncated, info = env.step(
        request.actions,
        proposal_shortlists=request.shortlists,
    )
    assert isinstance(following, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminated, dict)
    assert isinstance(truncated, dict)
    json.dumps(diagnostic._jsonable({"request": request.details, "info": info}), allow_nan=False)


def test_restore_diagnostic_snapshot_rebinds_real_terminal_hooks_and_isolates_copy():
    from terminal_marl.core.uav import TerminalOutcome
    from terminal_marl.training import diagnose_p3_terminal_credit as terminal_credit
    from terminal_marl.training import terminal_lifecycle_metrics

    config = replace(
        _config(),
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        teacher_mode="teacher_label_only",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        warm_start_uavs=1,
        total_requested_uavs=3,
        fixed_arrival_schedule=(0.0, 10.0, 20.0),
        max_simulation_seconds=30.0,
        runtime_proposal_shortlist_k=3,
    )
    source = TerminalParallelEnv(config)
    source.reset(112)
    terminal_lifecycle_metrics._capture_terminal_rows(source)
    terminal_credit._request_capture(source)
    source_requested = set(source.ledger.requested)
    source_outcomes = copy.deepcopy(source.ledger.outcomes)
    source_active = set(source.active)
    source_generated = int(source.arrivals.generated)
    source_index = int(source.arrivals.index)
    source_next_time = float(source.arrivals.next_time)
    source_rng = pickle.dumps(source.arrivals.rng.bit_generator.state)

    branch = copy.deepcopy(source)
    diagnostic.restore_diagnostic_snapshot(branch)
    assert branch._step_observer is None
    assert branch._resolve.__self__ is branch
    assert branch.arrivals.requests_until.__self__ is branch.arrivals

    future = branch.arrivals.requests_until(20.0)
    signatures = [diagnostic._request_signature(request) for request in future]
    assert [row["arrival_id"] for row in signatures] == [1, 2]
    assert [row["request_time"] for row in signatures] == pytest.approx([10.0, 20.0])
    assert all({"arrival_id", "request_time", "initial_soc", "base_priority", "gate_id", "mission"} <= set(row) for row in signatures)
    assert source.arrivals.generated == source_generated
    assert source.arrivals.index == source_index
    assert source.arrivals.next_time == pytest.approx(source_next_time)
    assert pickle.dumps(source.arrivals.rng.bit_generator.state) == source_rng

    branch_uav = next(iter(branch.active.values()))
    branch._resolve(branch_uav, TerminalOutcome.FAILED_ENERGY, "diagnostic")
    assert source.ledger.requested == source_requested
    assert source.ledger.outcomes == source_outcomes
    assert set(source.active) == source_active
    assert branch.ledger.outcomes[branch_uav.arrival_id] is TerminalOutcome.FAILED_ENERGY


def test_real_paired_gate_respects_environment_time_limit():
    config = replace(
        _config(),
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        teacher_mode="teacher_label_only",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        warm_start_uavs=1,
        total_requested_uavs=1,
        fixed_arrival_schedule=(0.0,),
        max_simulation_seconds=2.0,
        runtime_proposal_shortlist_k=3,
    )
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(105)
    policy = JointPPOV42(88)
    request = diagnostic.keyed_request(policy, env)
    target_uid = next(iter(observations))
    report = diagnostic.paired_gate_value(
        policy,
        env,
        request.actions,
        request.shortlists,
        target_uid,
        [106],
        3,
    )
    assert all(branch["steps"] == 2 for branch in report["branches"])
    assert all(branch["censor_reason"] == "environment_time_limit" for branch in report["branches"])
    assert report["censored"] is True
    assert report["qualified_delta_Q"] is None
    json.dumps(report, allow_nan=False)
