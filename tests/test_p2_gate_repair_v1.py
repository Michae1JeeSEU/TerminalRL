from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_adapter_v4_2 import (
    GateResidualAdapterStudentPPOV42,
    migrate_base_policy_to_adapter,
)
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
from terminal_marl.training import train_v4_2_p2_gate_repair_v1 as runner
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config


def _row(index: int, gate: int, *, seed: int = 778000, interface: str = "o2m") -> dict:
    phase = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}[interface]
    observation = np.zeros(88, dtype=np.float32)
    observation[index % 12] = 1.0
    return {
        "decision_event_id": f"repair-event-{index}",
        "seed": seed,
        "observation_v2_1": observation.tolist(),
        "teacher_gate": gate,
        "actor_mask": True,
        "support_split": "train",
        "scenario": "flow6",
        "interface": interface,
        "phase": phase,
    }


def test_zero_initialized_adapter_is_exact_identity_after_migration() -> None:
    torch.manual_seed(7)
    base = StudentPPOV42(88)
    base.optimizer.param_groups[0]["lr"] = 0.123
    adapter = migrate_base_policy_to_adapter(base, learning_rate=3e-4)
    assert all(group["lr"] == 3e-4 for group in adapter.optimizer.param_groups)
    assert adapter.migration["base_optimizer_state_copied"] is False
    assert adapter.migration["adapter_optimizer_state_initialized"] is False
    observations = torch.randn(8, 88)
    with torch.no_grad():
        base_gate, base_proposal = base.model.distributions(observations)
        adapter_gate, adapter_proposal = adapter.model.distributions(observations)
        assert torch.equal(base_gate.logits, adapter_gate.logits)
        assert torch.equal(base_proposal.logits, adapter_proposal.logits)
        agent_sets = torch.randn(2, 3, 88)
        mask = torch.ones(2, 3, dtype=torch.bool)
        assert torch.equal(base.model.critic(agent_sets, mask), adapter.model.critic(agent_sets, mask))
    assert all(torch.equal(parameter, torch.zeros_like(parameter))
               for name, parameter in adapter.model.gate_adapter[-1].named_parameters())


def test_small_repair_update_isolated_to_adapter_and_preserves_proposal_critic() -> None:
    torch.manual_seed(11)
    base = StudentPPOV42(88)
    adapter = migrate_base_policy_to_adapter(base)
    rows = [_row(index, index % 2, interface=("o2m", "m2i", "i2f")[index % 3]) for index in range(8)]
    before = copy.deepcopy(adapter.model)
    result = runner.train_repair(adapter, base.model, rows, rows, epochs=1, support_batch_size=4, d1_batch_size=4)
    assert result["finite_losses"] and result["finite_gradients"]
    changed = runner._changed_parameter_names(before.state_dict(), adapter.model.state_dict())
    assert changed and all(name.startswith("gate_adapter.") for name in changed)
    assert set(runner._changed_parameter_names(base.model.state_dict(), adapter.model.state_dict())) == set()
    with torch.no_grad():
        observations = torch.as_tensor(np.asarray([row["observation_v2_1"] for row in rows], dtype=np.float32))
        before_gate, before_proposal = before.distributions(observations)
        after_gate, after_proposal = adapter.model.distributions(observations)
        assert torch.equal(before_proposal.logits, after_proposal.logits)
        assert not torch.equal(before_gate.logits, after_gate.logits)
    assert runner._state_equal(runner._critic_state(before), runner._critic_state(adapter.model))


def test_repair_threshold_boundaries_are_inclusive() -> None:
    support = {
        "macro_F1": .80, "WAIT_recall": .70, "RELEASE_recall": .80, "brier": .15, "binary_nll": .50,
        "interface": {name: {"macro_F1": .70} for name in runner.audit.INTERFACES},
    }
    d1 = {"macro_F1": .90, "WAIT_recall": .90, "RELEASE_recall": .90, "brier": .08, "binary_nll": .30}
    invariants = {"proposal_invariant": True, "critic_invariant": True, "parameter_isolation": True,
                  "finite_losses": True, "finite_gradients": True}
    assert runner.repair_threshold_checks(support, d1, invariants)["all"]
    d1["binary_nll"] = .300001
    assert not runner.repair_threshold_checks(support, d1, invariants)["all"]


def test_audit_binding_rejection(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({
        "stage": "V4_2_P2_GATE_LEARNABILITY_AUDIT_V1",
        "AUDIT_COMPLETE": True,
        "audit_pass": False,
        "repair_preregistration_allowed": False,
    }), encoding="utf-8")
    with pytest.raises(runner.RepairInputError, match="did not pass"):
        runner.validate_audit_manifest(path, expected_sha256=None)


def test_checkpoint_round_trip_into_adapter_policy(tmp_path) -> None:
    base = StudentPPOV42(88)
    adapter = migrate_base_policy_to_adapter(base)
    rows = [_row(index, index % 2) for index in range(4)]
    runner.train_repair(adapter, base.model, rows, rows, epochs=1, support_batch_size=4, d1_batch_size=4)
    config = p2_config(6)
    checkpoint = tmp_path / "qualification_candidate.pt"
    save_checkpoint(checkpoint, adapter, config, stage="V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V1",
                    architecture_id=adapter.architecture_id, checkpoint_role="qualification_candidate",
                    PPO_ALLOWED=False, GATE_REPAIR_ALLOWED=False, curriculum_promotion_allowed=False,
                    checkpoint_promotable=False)
    restored = GateResidualAdapterStudentPPOV42(88)
    payload = load_checkpoint(checkpoint, restored, config, load_optimizer=True)
    assert payload["stage"] == "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V1"
    assert payload["checkpoint_promotable"] is False
    for name, parameter in adapter.model.state_dict().items():
        assert torch.equal(parameter, restored.model.state_dict()[name])
