import pytest

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.training import run_v4_2_p2_runtime_shortlist_entry_v1 as entry


def test_entry_protocol_freezes_k3_fresh_discarded_updates():
    value = entry.protocol()
    assert value["bindings"]["runtime_proposal_shortlist_k"] == 3
    assert value["entry"]["seeds"] == [807001, 807002]
    assert value["entry"]["scenarios"] == ["flow6", "flow12"]
    assert value["bindings"]["entry_updates_discarded"]
    assert value["bindings"]["entry_checkpoints_formal_initialization"] == "forbidden"
    assert value["authorization"]["PPO_ALLOWED_on_pass"]
    assert value["optimizer"]["trainable_scope"] == [
        "gate_adapter", "critic_encoder", "critic_head",
    ]
    assert value["hard_gates"]["frozen_proposal_subgraph"]
    assert value["hard_gates"]["post_update_runtime_shortlist_stability"]


def test_entry_review_validation_is_fail_closed():
    protocol = entry.protocol()
    report = {
        "stage": "V4_2_P2_RUNTIME_SHORTLIST_ADMISSION_REVIEW_V1",
        "PASS": True, "gates": {"all": True},
        "ENTRY_PROTOCOL_PREREGISTRATION_ALLOWED": True,
        "ENTRY_VALIDATION_ALLOWED": False, "PPO_ALLOWED": False,
        "protocol_sha256": protocol["bindings"]["admission_review_protocol_sha256"],
        "candidate_checkpoint_sha256": protocol["bindings"]["candidate_checkpoint_sha256"],
    }
    entry.validate_review(report, protocol)
    report["PPO_ALLOWED"] = True
    with pytest.raises(PermissionError):
        entry.validate_review(report, protocol)


def test_entry_source_hashes_bind_runtime_replay_and_optimizer():
    hashes = entry.source_hashes()
    assert "terminal_marl/envs/parallel_env.py" in hashes
    assert "terminal_marl/core/safety_trace.py" in hashes
    assert "terminal_marl/algo/student_ppo_v4_2.py" in hashes
    assert "terminal_marl/algo/joint_ppo_v4_2.py" in hashes
    assert "terminal_marl/core/low_level_joint_action_v4_2.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_entry_optimizer_freezes_proposal_generating_subgraph():
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    trainable = entry._configure_entry_optimizer(policy, 3e-5)
    assert trainable
    for name, parameter in policy.model.named_parameters():
        expected = name.startswith(entry.ENTRY_TRAINABLE_PREFIXES)
        assert parameter.requires_grad is expected
    optimizer_parameters = {
        id(parameter)
        for group in policy.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameters == {
        id(parameter) for parameter in policy.model.parameters()
        if parameter.requires_grad
    }
