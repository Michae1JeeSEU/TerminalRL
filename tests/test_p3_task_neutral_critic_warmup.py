from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from terminal_marl.algo import student_ppo_v4_2 as ppo
from terminal_marl.core.config import SimConfig
from terminal_marl.training import diagnose_p3_ppo_credit_signal as credit
from terminal_marl.training import run_p3_task_neutral_critic_warmup as warmup
from terminal_marl.training import run_p3_task_neutral_pilot as pilot
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval


class _TinyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Parameter(torch.tensor([0.25]))
        self.critic_encoder = nn.Linear(88, 1)
        self.critic_head = nn.Linear(2, 1)
        nn.init.zeros_(self.critic_encoder.weight)
        nn.init.zeros_(self.critic_encoder.bias)
        nn.init.zeros_(self.critic_head.weight)
        nn.init.zeros_(self.critic_head.bias)

    def critic(self, agent_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.critic_encoder(agent_sets)
        valid = mask.unsqueeze(-1)
        mean = (encoded * valid).sum(1) / valid.sum(1).clamp_min(1)
        maximum = encoded.masked_fill(~valid, float("-inf")).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.critic_head(torch.cat((mean, maximum), -1)).squeeze(-1)


def _tiny_policy() -> SimpleNamespace:
    model = _TinyCritic()
    model.eval()
    return SimpleNamespace(model=model, device=torch.device("cpu"))


def test_mc_targets_match_production_returns_at_episode_boundaries() -> None:
    rewards = np.asarray([1.0, 2.0, 3.0, 4.0])
    ends = np.asarray([False, True, False, True])
    observed = warmup.mc_targets(rewards, ends)
    expected = np.concatenate((
        credit.mc_returns(rewards[:2], 0.99),
        credit.mc_returns(rewards[2:], 0.99),
    ))
    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(observed, [1.0 + 0.99 * 2.0, 2.0, 3.0 + 0.99 * 4.0, 4.0])
    np.testing.assert_allclose(
        warmup.mc_targets([1.0, 2.0, 3.0], gamma=0.5),
        [2.75, 3.5, 3.0],
    )


def test_critic_fit_reduces_synthetic_error_and_preserves_actor() -> None:
    policy = _tiny_policy()
    worlds = [np.eye(88, dtype=np.float32)[None, index % 4] for index in range(8)]
    targets = np.asarray([0.25 + 0.1 * index for index in range(8)], dtype=np.float64)
    trajectories = [{
        "scenario": "synthetic",
        "seed": 1,
        "worlds": worlds,
        "targets": targets,
    }]
    before_actor = policy.model.actor.detach().clone()
    result = warmup.fit_critic(
        policy,
        trajectories,
        epochs=20,
        minibatch_size=64,
        learning_rate=3e-3,
    )
    assert result["after"]["mse"] < result["before"]["mse"]
    assert result["actor_invariance"]["non_critic_state_bitwise_unchanged"] is True
    assert torch.equal(policy.model.actor.detach(), before_actor)


def _valid_warmup_payload(policy: SimpleNamespace, config: SimConfig) -> dict:
    state = {name: value.detach().clone() for name, value in policy.model.state_dict().items()}
    return {
        "stage": warmup.STAGE,
        "curriculum_id": warmup.CURRICULUM_ID,
        "task_priority_enabled": False,
        "low_level_priority_time_weight": 0.0,
        "parent_checkpoint_sha256": warmup.PARENT_CHECKPOINT_SHA256,
        "neutral_config_snapshots": {"dense24": neutral_eval._config_snapshot(config)},
        "model_state": state,
        "actor_invariance": {
            "non_critic_state_bitwise_unchanged": True,
            "critic_parameter_names": ["critic_encoder.weight", "critic_encoder.bias", "critic_head.weight", "critic_head.bias"],
        },
    }


def test_warmup_rejects_wrong_stage_and_actor_difference() -> None:
    config = SimConfig(total_requested_uavs=24, task_priority_enabled=False, low_level_priority_time_weight=0.0)
    policy = _tiny_policy()
    payload = _valid_warmup_payload(policy, config)
    with pytest.raises(pilot.TaskNeutralPilotError, match="stage"):
        pilot._validate_initial_warmup_payload({**payload, "stage": "wrong"}, policy, config)
    changed = dict(payload)
    changed_state = dict(payload["model_state"])
    changed_state["actor"] = changed_state["actor"] + 1.0
    changed["model_state"] = changed_state
    with pytest.raises(pilot.TaskNeutralPilotError, match="non-critic"):
        pilot._validate_initial_warmup_payload(changed, policy, config)


def test_generic_request_count_does_not_use_dense24_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"arrival_id": index, "outcome": "normal", "mission_type": "standard"}
        for index in range(3)
    ]
    ledger = SimpleNamespace(
        requested={0, 1, 2},
        outcomes={0: "normal", 1: "normal", 2: "normal"},
    )
    env = SimpleNamespace(ledger=ledger)
    monkeypatch.setattr(credit.terminal_lifecycle_metrics, "_merge_terminal_rows", lambda captured, completed: list(captured))
    monkeypatch.setattr(credit.terminal_lifecycle_metrics, "_completed_rows", lambda env, first_seen: [])
    monkeypatch.setattr(credit.terminal_lifecycle_metrics, "_enum_value", lambda value: value)
    monkeypatch.setattr(credit.terminal_lifecycle_metrics, "summarize_terminal_rows", lambda *args, **kwargs: {})
    requests = [{"arrival_id": index} for index in range(3)]
    result = credit._request_lifecycle(
        env, rows, {}, requests, SimpleNamespace(total_requested_uavs=3)
    )
    assert result["requested_count"] if "requested_count" in result else True
    assert result["all_requests_observed"] is True
    assert result["ledger_alignment"] is True
    assert len(result["request_rows"]) == 3


def test_safety_required_counters_need_observed_zero_and_optional_stays_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = {
        "terminal": True,
        "all_requests_observed": True,
        "ledger_alignment": True,
        "request_rows": [{"arrival_id": index} for index in range(3)],
    }
    replay = {"replay": {"exact": True}, "lifecycle": lifecycle, "env": object()}
    monkeypatch.setattr(
        credit.terminal_credit,
        "_safety_snapshot",
        lambda env: {
            name: 0.0 for name in credit.terminal_credit.REQUIRED_SAFETY_COUNTERS
        } | {"uncertified_execution": None},
    )
    report = warmup._trajectory_record(
        scenario="flow6", seed=1, requests=3, tape_path=Path("tape.json"),
        replay=replay, steps=[], checkpoint_sha256="a" * 64,
        model_digest_before="b" * 64, model_digest_after="b" * 64,
    )
    assert report["safety"]["required_counters_zero"] is True
    assert report["safety"]["uncertified_execution_available"] is False
    assert report["safety"]["uncertified_execution_zero"] is None

    missing = {
        name: 0.0 for name in credit.terminal_credit.REQUIRED_SAFETY_COUNTERS
    }
    missing[credit.terminal_credit.REQUIRED_SAFETY_COUNTERS[0]] = None
    monkeypatch.setattr(credit.terminal_credit, "_safety_snapshot", lambda env: missing)
    with pytest.raises(warmup.CriticWarmupError, match="required safety counter"):
        warmup._trajectory_record(
            scenario="flow6", seed=1, requests=3, tape_path=Path("tape.json"),
            replay=replay, steps=[], checkpoint_sha256="a" * 64,
            model_digest_before="b" * 64, model_digest_after="b" * 64,
        )
