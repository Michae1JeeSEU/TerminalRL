from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_adapter_v4_2 import GateResidualAdapterStudentPPOV42
from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TEMPERATURE_BUFFER,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
    migrate_v2_policy_to_calibrated,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training import calibrate_v4_2_p2_gate_v3 as runner
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config


def _row(index: int, gate: int, *, interface: str = "o2m") -> dict:
    phase = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}[interface]
    observation = np.zeros(88, dtype=np.float32)
    observation[index % 12] = 1.0
    return {
        "decision_event_id": f"calibration-v3-event-{index}",
        "seed": 778000,
        "observation_v2_1": observation.tolist(),
        "teacher_gate": gate,
        "actor_mask": True,
        "support_split": "train",
        "scenario": "flow6",
        "interface": interface,
        "phase": phase,
    }


def test_temperature_scaling_preserves_argmax_proposal_and_critic() -> None:
    torch.manual_seed(23)
    source = GateResidualAdapterStudentPPOV42(88)
    calibrated = migrate_v2_policy_to_calibrated(source)
    rows = [_row(index, index % 2, interface=("o2m", "m2i", "i2f")[index % 3]) for index in range(9)]
    invariance = runner.decision_and_logit_invariance(
        source.model, calibrated.model, {"train": rows, "dev": rows, "d1": rows},
    )
    assert invariance["raw_gate_argmax_invariant"]
    assert invariance["proposal_logits_invariant"]
    assert runner._critic_state_equal(source.model, calibrated.model)
    assert torch.equal(calibrated.model.gate_temperature, torch.tensor(CALIBRATION_TEMPERATURE))
    assert calibrated.model.state_dict()[TEMPERATURE_BUFFER].shape == torch.Size([])


def test_strict_migration_and_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(29)
    source = GateResidualAdapterStudentPPOV42(88)
    calibrated = migrate_v2_policy_to_calibrated(source, temperature=2.5)
    assert calibrated.migration["new_buffer_names"] == [TEMPERATURE_BUFFER]
    assert calibrated.migration["source_optimizer_state_copied"] is False
    assert set(source.model.state_dict()).issubset(set(calibrated.model.state_dict()))
    checkpoint = tmp_path / "calibrated.pt"
    runner._save_calibration_checkpoint(
        checkpoint,
        calibrated,
        base_checkpoint_sha256="v2-checkpoint",
        source_hashes={"protocol": "v3-protocol"},
        qualification_pass=True,
    )
    restored = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    payload = load_checkpoint(checkpoint, restored, p2_config(6), load_optimizer=False)
    assert payload["stage"] == "V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3"
    assert payload["architecture_id"] == restored.architecture_id
    assert payload["temperature"] == 2.5
    assert payload["optimizer_saved"] is False
    for name, value in calibrated.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])


def test_v2_failure_shape_rejection(tmp_path) -> None:
    path = tmp_path / "v2-manifest.json"
    path.write_text(json.dumps({
        "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V2",
        "status": "PASS",
    }), encoding="utf-8")
    with pytest.raises(runner.RepairInputError, match="status"):
        runner.validate_v2_repair_manifest(path, expected_sha256=None)


def test_protocol_binds_v2_sources_and_fixed_grid() -> None:
    value = runner.protocol()
    assert value["predecessor"]["repair_manifest_sha256"] == runner.V2_REPAIR_MANIFEST_SHA
    assert value["predecessor"]["checkpoint_sha256"] == runner.V2_CHECKPOINT_SHA
    assert value["predecessor"]["protocol_sha256"] == runner.V2_PROTOCOL_SHA
    assert tuple(value["temperature_selection"]["preregistered_grid"]) == runner.TEMPERATURE_GRID


def test_calibration_thresholds_are_inclusive() -> None:
    support = {
        "macro_F1": .80, "WAIT_recall": .70, "RELEASE_recall": .80, "brier": .15,
        "binary_nll": .50,
        "interface": {name: {"macro_F1": .70} for name in runner.audit.INTERFACES},
    }
    d1 = {"macro_F1": .90, "WAIT_recall": .90, "RELEASE_recall": .90,
          "brier": .08, "binary_nll": .30}
    invariants = {
        "proposal_logits_invariant": True,
        "critic_state_invariant": True,
        "common_state_bitwise_equal": True,
        "only_new_temperature_buffer": True,
        "temperature_finite_positive": True,
        "finite_metrics": True,
        "no_training_or_optimizer": True,
    }
    checks = runner.calibration_threshold_checks(support, d1, invariants)
    assert checks["all"]
    assert set(checks["invariants"]) == {
        "proposal_logits_invariant",
        "critic_state_invariant",
        "common_state_bitwise_equal",
        "only_new_temperature_buffer",
        "temperature_finite_positive",
        "finite_metrics",
        "no_training_or_optimizer",
        "all",
    }
    assert "finite_losses" not in checks["invariants"]
    assert "finite_gradients" not in checks["invariants"]
    d1["binary_nll"] = .300001
    assert not runner.calibration_threshold_checks(support, d1, invariants)["all"]
