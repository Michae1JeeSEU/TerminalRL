from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)

runner = importlib.import_module("terminal_marl.training.train_v4_2_p2_gate_repair_v3")


def _row(index: int, gate: int, *, split: str = "train", seed: int = 783001) -> dict:
    interfaces = ("o2m", "m2i", "i2f")
    interface = interfaces[index % len(interfaces)]
    phase = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}[interface]
    observation = np.zeros(88, dtype=np.float32)
    observation[index % 12] = 1.0
    return {
        "decision_event_id": f"repair-v3-event-{split}-{index}",
        "seed": seed,
        "observation_v2_1": observation.tolist(),
        "teacher_gate": gate,
        "actor_mask": True,
        "support_split": split,
        "scenario": "flow6",
        "interface": interface,
        "phase": phase,
    }


def _metric(*, support: bool = True) -> dict:
    values = {
        "macro_F1": 0.80 if support else 0.90,
        "WAIT_recall": 0.70 if support else 0.90,
        "RELEASE_recall": 0.80 if support else 0.90,
        "brier": 0.15 if support else 0.08,
        "binary_nll": 0.50 if support else 0.30,
        "interface": {name: {"macro_F1": 0.70} for name in ("o2m", "m2i", "i2f")},
    }
    return values


def test_protocol_binds_passing_v2_audit_and_calibrated_candidate() -> None:
    value = runner.protocol()
    assert value["protocol"] == "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3"
    assert value["predecessor"]["audit_manifest_sha256"] == runner.AUDIT_MANIFEST_SHA
    assert value["bindings"]["candidate_architecture_id"] == runner.CANDIDATE_ARCHITECTURE_ID
    assert value["bindings"]["candidate_temperature"] == 2.5
    assert value["training"]["objective"] == "0.5_support_CE_plus_0.5_D1_CE_plus_2.0_D1_anchor_KL"
    assert value["training"]["learning_rate"] == 0.0001


def test_stable_anchor_kl_extreme_logits_has_finite_value_and_gradient() -> None:
    candidate_logits = torch.tensor([[0.0, -110.0], [0.0, -1000.0]], dtype=torch.float32)
    repaired_logits = torch.tensor([[-0.25, -109.75], [0.25, -999.75]],
                                    dtype=torch.float32, requires_grad=True)
    float32_probs = torch.exp(F.log_softmax(candidate_logits, dim=-1))
    assert float32_probs[1, 1].item() == 0.0
    value = runner.stable_anchor_kl_from_logits(candidate_logits, repaired_logits)
    assert bool(torch.isfinite(value).item())
    value.backward()
    assert repaired_logits.grad is not None
    assert bool(torch.isfinite(repaired_logits.grad).all().item())


def test_one_epoch_calibrated_training_changes_only_adapter_and_writes_progress(tmp_path: Path) -> None:
    torch.manual_seed(19)
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88, temperature=2.5)
    frozen = copy.deepcopy(policy.model)
    before_model = copy.deepcopy(policy.model)
    before_state = {name: value.detach().clone() for name, value in policy.model.state_dict().items()}
    rows = [_row(index, index % 2, seed=783001) for index in range(8)]
    progress = tmp_path / "progress.json"
    result = runner.train_repair(
        policy, frozen, rows, rows, epochs=1, support_batch_size=4,
        d1_batch_size=4, progress_path=progress,
    )
    assert result["finite_losses"] is True
    assert result["finite_gradients"] is True
    assert result["final_epoch"] == 1
    changed_state = runner._changed_state_names(before_state, policy.model.state_dict())
    changed_parameters = runner._changed_parameter_names(before_model, policy.model)
    assert changed_state
    assert changed_parameters
    assert all(name.startswith("gate_adapter.") for name in changed_parameters)
    assert "gate_temperature" not in changed_state
    assert runner._state_equal(
        {"gate_temperature": before_state["gate_temperature"]},
        {"gate_temperature": policy.model.state_dict()["gate_temperature"]},
    )
    assert runner._state_equal(runner._critic_state(before_model), runner._critic_state(policy.model))
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert payload["stage"] == "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3"
    assert payload["latest_epoch"] == 1
    assert payload["completed_epochs"] == 1
    assert payload["latest"]["loss"] == result["history"][-1]["loss"]


def test_thresholds_are_inclusive_and_require_invariants() -> None:
    invariants = {
        "proposal_invariant": True,
        "critic_invariant": True,
        "temperature_state_invariant": True,
        "common_state_bitwise_equal": True,
        "parameter_isolation": True,
        "finite_losses": True,
        "finite_gradients": True,
    }
    checks = runner.repair_threshold_checks(_metric(support=True), _metric(support=False), invariants)
    assert checks["all"] is True
    invariants["temperature_state_invariant"] = False
    assert runner.repair_threshold_checks(
        _metric(support=True), _metric(support=False), invariants
    )["all"] is False


def test_failure_manifest_has_no_checkpoint_and_all_authorizations_false(tmp_path: Path) -> None:
    output = tmp_path / "repair-v3"
    error = FloatingPointError("non-finite repair component; epoch=2 step=1 finite={} values={}")
    manifest_path = runner.write_failure_manifest(output, error)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["repair_pass"] is False
    assert payload["checkpoint"] is None
    assert payload["checkpoint_saved"] is False
    assert payload["no_checkpoint"] is True
    assert payload["STUDENT_QUALIFICATION_ALLOWED"] is False
    assert payload["PPO_ALLOWED"] is False
    assert payload["GATE_REPAIR_ALLOWED"] is False
    assert payload["curriculum_promotion_allowed"] is False
    assert payload["checkpoint_promotable"] is False
