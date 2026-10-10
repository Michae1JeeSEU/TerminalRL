from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from terminal_marl.algo.student_ppo_v4_2 import StudentStepV42
from terminal_marl.training import diagnose_p3_task_neutral_episode_gradients as diagnostic


def _step(reward: float, *, actor: bool | np.ndarray, terminal: bool = False, end: bool = False, value: float = 0.0) -> StudentStepV42:
    actor_mask = np.asarray([actor] if isinstance(actor, (bool, np.bool_)) else actor, dtype=bool)
    return StudentStepV42(
        world=np.zeros((len(actor_mask), 88), dtype=np.float32),
        requested_gate=np.zeros(len(actor_mask), dtype=np.int64),
        requested_proposal=np.full(len(actor_mask), -1, dtype=np.int64),
        gate_logprob=np.zeros(len(actor_mask), dtype=np.float32),
        proposal_logprob=np.full(len(actor_mask), np.nan, dtype=np.float32),
        actor_mask=actor_mask,
        value=float(value), next_value=0.0 if terminal else float(value), reward=float(reward),
        terminated=terminal, episode_end=end,
    )


def test_full_episode_mc_reaches_no_actor_tail_and_does_not_cross_boundary() -> None:
    first = [_step(1.0, actor=True), _step(2.0, actor=False), _step(3.0, actor=False, terminal=True, end=True)]
    second = [_step(10.0, actor=True), _step(20.0, actor=False, terminal=True, end=True)]
    result = diagnostic.compute_credit_batch([first, second], gamma=1.0, gae_lambda=1.0, reward_scale=1.0)
    assert result["episodes"][0]["mc_scaled"].tolist() == [6.0, 5.0, 3.0]
    assert result["episodes"][1]["mc_scaled"].tolist() == [30.0, 20.0]
    assert result["episodes"][0]["returns_shared"].tolist() == [6.0, 5.0, 3.0]
    assert result["episodes"][1]["returns_shared"].tolist() == [30.0, 20.0]
    assert result["actor_decisions"] == 2


def test_default_mc_unit_and_reward_scale() -> None:
    result = diagnostic.compute_credit_batch([[_step(1.0, actor=True), _step(2.0, actor=False, terminal=True, end=True)]])
    assert np.allclose(result["episodes"][0]["mc_raw"], [3.0, 2.0])
    assert np.allclose(result["episodes"][0]["mc_scaled"], [0.03, 0.02])


def test_episode_report_counts_actor_world_decisions_and_actor_steps() -> None:
    rows = [[
        _step(1.0, actor=np.asarray([True, True])),
        _step(2.0, actor=np.asarray([False, False])),
        _step(3.0, actor=np.asarray([True, False]), terminal=True, end=True),
    ]]
    batch = diagnostic.compute_credit_batch(rows, gamma=1.0, gae_lambda=1.0, reward_scale=1.0)
    report = diagnostic._episode_report(batch["episodes"][0], {
        "index": 1, "record_path": "record.json", "tape_path": "tape.json",
        "env_seed": 1, "policy_seed": 2, "requests": 1,
    })
    assert report["actor_steps"] == 2
    assert report["actor_decisions"] == 3


def test_gradient_statistics_cancellation_alignment_and_aggregate() -> None:
    cancelled = diagnostic.gradient_statistics([[1.0, 0.0], [-1.0, 0.0]], [0.0, 0.0])
    assert cancelled["coherence"] == 0.0
    assert cancelled["aggregate_check"]["allclose"] is True
    aligned = diagnostic.gradient_statistics([[1.0, 2.0], [2.0, 4.0]], [3.0, 6.0])
    assert aligned["coherence"] == pytest.approx(1.0)
    assert aligned["cosine_matrix"][0][1] == pytest.approx(1.0)
    assert aligned["aggregate_check"]["max_abs_error"] == 0.0


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_adapter = nn.Linear(88, 2, bias=False)

    def distributions(self, observations: torch.Tensor):
        gate = Categorical(logits=self.gate_adapter(observations))
        proposal = Categorical(logits=torch.zeros((len(observations), 25), device=observations.device))
        return gate, proposal


class _ToyPolicy:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = _ToyModel()


def test_gate_gradient_does_not_mutate_model_state() -> None:
    policy = _ToyPolicy()
    before = copy.deepcopy(policy.model.state_dict())
    rows = [[_step(1.0, actor=True, terminal=True, end=True)]]
    batch = diagnostic.compute_credit_batch(rows, gamma=1.0, gae_lambda=1.0, reward_scale=1.0)
    result = diagnostic._gradient_analysis(policy, batch)
    assert result["model_unchanged"] is True
    for key, value in before.items():
        assert torch.equal(value, policy.model.state_dict()[key])


def test_check_only_has_zero_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(diagnostic, "_preflight", lambda: {
        "cells": [], "total_steps": diagnostic.EXPECTED_TOTAL_STEPS,
        "total_requests": 144, "max_replay_steps": 6000,
    })
    report = diagnostic.run(tmp_path / "check", check_only=True)
    assert report["status"] == "CHECK_ONLY_OK"
    assert report["checks"]["runtime_replay_calls"] == 0
    assert report["checks"]["runtime_env_reset"] is False
    assert report["checks"]["runtime_optimizer_calls"] == 0
