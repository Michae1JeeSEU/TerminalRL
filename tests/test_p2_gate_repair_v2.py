from __future__ import annotations

import copy
import json

import numpy as np
import torch
from torch.nn import functional as F

from terminal_marl.algo.gate_adapter_v4_2 import migrate_base_policy_to_adapter
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.training import train_v4_2_p2_gate_repair_v2 as runner


def _row(index: int, gate: int, *, interface: str = "o2m") -> dict:
    phase = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}[interface]
    observation = np.zeros(88, dtype=np.float32)
    observation[index % 12] = 1.0
    return {
        "decision_event_id": f"repair-v2-event-{index}",
        "seed": 778000,
        "observation_v2_1": observation.tolist(),
        "teacher_gate": gate,
        "actor_mask": True,
        "support_split": "train",
        "scenario": "flow6",
        "interface": interface,
        "phase": phase,
    }


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


def test_synthetic_training_is_finite_adapter_only_and_writes_progress(tmp_path) -> None:
    torch.manual_seed(19)
    base = StudentPPOV42(88)
    adapter = migrate_base_policy_to_adapter(base, learning_rate=3e-4)
    before = copy.deepcopy(adapter.model)
    rows = [_row(index, index % 2, interface=("o2m", "m2i", "i2f")[index % 3]) for index in range(8)]
    progress = tmp_path / "progress.json"
    result = runner.train_repair(
        adapter,
        base.model,
        rows,
        rows,
        epochs=10,
        support_batch_size=4,
        d1_batch_size=4,
        progress_path=progress,
    )
    assert result["finite_losses"] and result["finite_gradients"]
    assert result["final_epoch"] == 10
    changed = runner._changed_parameter_names(before.state_dict(), adapter.model.state_dict())
    assert changed and all(name.startswith("gate_adapter.") for name in changed)
    assert runner._changed_parameter_names(base.model.state_dict(), adapter.model.state_dict()) == []
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert payload["status"] == "RUNNING"
    assert payload["latest_epoch"] == 10
    assert payload["completed_epochs"] == 10
    assert payload["latest"]["loss"] == result["history"][-1]["loss"]


def test_protocol_binds_v1_protocol_and_failure_log() -> None:
    value = runner.protocol()
    assert value["predecessor"]["protocol_sha256"] == runner.V1_PROTOCOL_SHA256
    assert value["predecessor"]["failure_log_sha256"] == runner.V1_FAILURE_LOG_SHA256


def test_failure_manifest_has_no_checkpoint_and_all_authorizations_false(tmp_path) -> None:
    output = tmp_path / "repair-v2"
    output.mkdir()
    error = FloatingPointError("non-finite repair loss; epoch=2 step=1 finite={} values={}")
    manifest_path = runner.write_failure_manifest(output, error)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["checkpoint"] is None
    assert payload["checkpoint_saved"] is False
    assert payload["no_checkpoint"] is True
    assert payload["STUDENT_QUALIFICATION_ALLOWED"] is False
    assert payload["PPO_ALLOWED"] is False
    assert payload["GATE_REPAIR_ALLOWED"] is False
    assert payload["curriculum_promotion_allowed"] is False
    assert payload["checkpoint_promotable"] is False
