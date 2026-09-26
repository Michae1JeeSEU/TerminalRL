import copy

import numpy as np
import pytest
import torch

from terminal_marl.algo import student_ppo_v4_2
from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.algo.gate_reference_v4_2 import GateReference
from terminal_marl.algo.student_ppo_v4_2 import StudentStepV42
from terminal_marl.training import p3_kl_recovery_v1 as recovery
from terminal_marl.training import run_v4_2_p3_gate_anchored_v6 as v6


def _state_equal(left, right):
    return recovery._exact_equal(left, right)


def _make_case():
    torch.manual_seed(7)
    np.random.seed(7)
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88, learning_rate=3e-4)
    v6._trainable_names(policy)
    policy.optimizer = torch.optim.Adam(
        [parameter for parameter in policy.model.parameters() if parameter.requires_grad],
        lr=3e-4,
    )
    generator = np.random.default_rng(71)
    old_rows = [
        {
            "observation_v2_1": generator.normal(size=88).astype(np.float32),
            "teacher_gate": index,
            "decision_event_id": f"old-{index}",
            "seed": 101 + index,
        }
        for index in range(2)
    ]
    coverage_rows = [
        {
            "observation_v2_1": generator.normal(size=88).astype(np.float32),
            "teacher_gate": 1 - index,
            "decision_event_id": f"coverage-{index}",
            "seed": 201 + index,
        }
        for index in range(2)
    ]
    observations = np.asarray([
        row["observation_v2_1"] for row in [*old_rows, *coverage_rows]
    ], dtype=np.float32)
    reference = GateReference.from_policy(
        policy, observations, np.ones(len(observations), dtype=np.float32),
        coefficient=.1, max_kl=.05,
    )
    policy.gate_reference = reference
    steps = []
    for index in range(4):
        world = generator.normal(size=(2, 88)).astype(np.float32)
        batch = policy.act_details(world, world)
        steps.append(StudentStepV42(
            world,
            batch.requested_gate,
            np.asarray([-1 if value is None else value for value in batch.requested_proposal]),
            batch.gate_logprob,
            np.asarray([np.nan if value is None else value for value in batch.proposal_logprob]),
            np.ones(2, dtype=bool),
            batch.value,
            0.0,
            float(index + 1),
            True,
            True,
        ))
    protocol = {
        "optimizer": {
            "update_epochs": 1,
            "minibatch_size": 4,
            "gamma": .99,
            "gae_lambda": .95,
            "clip_ratio": .1,
            "value_coef": .5,
            "entropy_coef": .001,
            "target_kl": .01,
            "maximum_approx_kl": .05,
            "minimum_accepted_updates": 1,
            "maximum_rejected_updates": 1,
        },
    }
    return policy, steps, reference, old_rows, coverage_rows, protocol


@pytest.fixture
def real_case(monkeypatch):
    monkeypatch.setattr(v6.p3_legacy, "_policy_tape_shortlists_exact", lambda *args, **kwargs: True)
    return _make_case()


def _recover(case):
    policy, steps, reference, old_rows, coverage_rows, protocol = case
    return recovery.recover_update(
        policy, steps, None, "synthetic-tape.json", protocol, reference,
        old_rows, coverage_rows,
    )


def test_natural_real_calibrated_update_is_reported_as_updated(real_case):
    policy = real_case[0]
    result = _recover(real_case)

    assert result["status"] == "UPDATED"
    assert result["processed"] == 1
    assert result["updated"] == 1
    assert result["skipped"] == 0
    assert result["update"]["accepted_updates"] >= 1
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["status"] == "UPDATED"
    assert policy.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_lr_fallback_retries_real_optimizer_step_and_restores_base_lr(real_case, monkeypatch):
    calls = {"count": 0}

    def reject_first(_policy, _steps):
        calls["count"] += 1
        return .02 if calls["count"] == 1 else 0.0

    monkeypatch.setattr(student_ppo_v4_2, "_full_rollout_sampled_approx_kl", reject_first)
    result = _recover(real_case)

    assert result["status"] == "UPDATED"
    assert result["updated"] == 1
    assert result["skipped"] == 0
    assert calls["count"] == 2
    assert [attempt["status"] for attempt in result["attempts"]] == ["REJECTED_KL", "UPDATED"]
    assert result["attempts"][1]["effective_learning_rates"] == pytest.approx([1.5e-4])
    assert real_case[0].optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_accepted_update_with_later_pure_kl_stop_remains_updated(real_case, monkeypatch):
    calls = {"count": 0}

    def accept_then_reject(_policy, _steps):
        calls["count"] += 1
        return 0.0 if calls["count"] == 1 else .02

    monkeypatch.setattr(student_ppo_v4_2, "_full_rollout_sampled_approx_kl", accept_then_reject)
    case = list(real_case)
    case[-1] = dict(case[-1], optimizer=dict(case[-1]["optimizer"], minibatch_size=2))
    result = _recover(tuple(case))

    assert result["status"] == "UPDATED"
    assert result["updated"] == 1
    assert result["skipped"] == 0
    assert calls["count"] == 2
    assert result["update"]["accepted_updates"] == 1
    assert result["update"]["rejected_updates"] == 1
    assert result["update"]["rejection_reason"] == "rollout_kl_exceeds_target"
    assert len(result["attempts"]) == 1


def test_four_pure_kl_rejections_skip_and_restore_model_adam_but_keep_last_rng(real_case, monkeypatch):
    monkeypatch.setattr(student_ppo_v4_2, "_full_rollout_sampled_approx_kl", lambda *_: .02)
    policy = real_case[0]
    model_before = copy.deepcopy(policy.model.state_dict())
    optimizer_before = copy.deepcopy(policy.optimizer.state_dict())
    rng_before = recovery._rng_snapshot()

    result = _recover(real_case)

    assert result["status"] == "SKIPPED_KL"
    assert result["processed"] == 1
    assert result["updated"] == 0
    assert result["skipped"] == 1
    assert len(result["attempts"]) == 4
    assert all(attempt["status"] == "REJECTED_KL" for attempt in result["attempts"])
    assert result["update"]["accepted_updates"] == 0
    assert _state_equal(model_before, policy.model.state_dict())
    assert _state_equal(optimizer_before, policy.optimizer.state_dict())
    assert policy.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert not _state_equal(rng_before["numpy"], np.random.get_state())


def test_exception_is_hard_failure_without_retry_and_complete_rollback(real_case, monkeypatch):
    policy = real_case[0]
    model_before = copy.deepcopy(policy.model.state_dict())
    optimizer_before = copy.deepcopy(policy.optimizer.state_dict())
    rng_before = recovery._rng_snapshot()

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic update failure")

    monkeypatch.setattr(recovery.v6, "_update", fail)
    result = _recover(real_case)

    assert result["status"] == "HARD_FAILED"
    assert result["updated"] == 0
    assert result["skipped"] == 0
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["status"] == "HARD_FAILED"
    assert _state_equal(model_before, policy.model.state_dict())
    assert _state_equal(optimizer_before, policy.optimizer.state_dict())
    assert _state_equal(rng_before, recovery._rng_snapshot())


def test_reference_rebind_is_hard_failure_and_never_retried(real_case, monkeypatch):
    calls = {"count": 0}
    reference_state = recovery._reference_snapshot(real_case[2])

    def rebind(policy, *_args):
        calls["count"] += 1
        with torch.no_grad():
            real_case[2].reference_probs[0, 0].add_(.1)
        policy.gate_reference = object()
        return (
            {
                "accepted_updates": 0,
                "attempted_updates": 1,
                "rejected_updates": 1,
                "updates": 0,
                "stopped": True,
                "rejection_reason": "rollout_kl_exceeds_target",
            },
            {"accepted_updates": False, "finite_update": True, "finite_model": True,
             "finite_optimizer": True, "reference_source_fixed": True},
            {},
        )

    monkeypatch.setattr(recovery.v6, "_update", rebind)
    result = _recover(real_case)

    assert result["status"] == "HARD_FAILED"
    assert calls["count"] == 1
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["status"] == "HARD_FAILED"
    assert "reference_ok=False" in result["error"]
    assert recovery._reference_equal(real_case[2], reference_state)


def test_nonfinite_gate_is_hard_failure_and_never_retried(real_case, monkeypatch):
    calls = {"count": 0}

    def nonfinite(*_args):
        calls["count"] += 1
        return (
            {
                "accepted_updates": 0,
                "attempted_updates": 1,
                "rejected_updates": 1,
                "updates": 0,
                "stopped": True,
                "rejection_reason": "rollout_kl_exceeds_target",
            },
            {key: (key != "finite_model") for key in recovery.REQUIRED_UPDATE_GATES},
            {},
        )

    monkeypatch.setattr(recovery.v6, "_update", nonfinite)
    result = _recover(real_case)

    assert result["status"] == "HARD_FAILED"
    assert calls["count"] == 1
    assert result["updated"] == 0
    assert result["skipped"] == 0


def test_accepted_update_with_nonkl_rejection_reason_is_hard_failure(real_case, monkeypatch):
    def nonkl(*_args):
        return (
            {
                "accepted_updates": 1,
                "attempted_updates": 1,
                "rejected_updates": 0,
                "updates": 1,
                "stopped": False,
                "rejection_reason": "optimizer_step_error",
            },
            {"accepted_updates": True, "finite_update": True, "finite_model": True,
             "finite_optimizer": True, "reference_source_fixed": True,
             "reference_gate_kl_cap": True, "frozen_proposal_subgraph": True},
            {},
        )

    monkeypatch.setattr(recovery.v6, "_update", nonkl)
    result = _recover(real_case)

    assert result["status"] == "HARD_FAILED"
    assert result["updated"] == 0
    assert result["skipped"] == 0
    assert len(result["attempts"]) == 1
    assert result["update"]["accepted_updates"] == 1
    assert "optimizer_step_error" in result["error"]
