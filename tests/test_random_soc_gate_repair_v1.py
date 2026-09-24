from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.training import diagnose_trained_dense30_v2 as diagnostic
from terminal_marl.training import train_random_soc_gate_repair_v1 as runner


def _row(index: int, gate: int, *, seed: int, event_prefix: str = "event") -> dict:
    observation = np.zeros(88, dtype=np.float32)
    observation[0 if gate else 1] = 1.0
    observation[(index + 7) % 88] = float(index + 1) / 10.0
    return {
        "decision_event_id": f"{event_prefix}-{index}",
        "seed": seed,
        "observation_v2_1": observation.tolist(),
        "observation_schema": runner.OBSERVATION_SCHEMA,
        "action_schema": runner.ACTION_SCHEMA,
        "teacher_gate": gate,
        "actor_mask": True,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _patch_source(monkeypatch: pytest.MonkeyPatch, source_path: Path):
    source_policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88, temperature=2.5)
    source_before = {
        name: value.detach().clone()
        for name, value in source_policy.model.state_dict().items()
    }
    config = diagnostic.build_scenario_config("dense24")
    source_sha = runner._sha256(source_path)
    monkeypatch.setattr(runner, "SOURCE_CHECKPOINT_SHA256", source_sha)

    monkeypatch.setattr(runner.diagnostic, "build_scenario_config", lambda scenario: config)
    monkeypatch.setattr(
        runner.diagnostic,
        "load_trained_checkpoint",
        lambda path, loaded_config: (source_policy, {"checkpoint_sha256": source_sha}),
    )
    return source_policy, source_before, config, source_sha


def test_training_changes_only_gate_adapter_and_reports_final_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_path = tmp_path / "source.pt"
    source_path.write_bytes(b"frozen-source")
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_rows(train_path, [_row(index, index % 2, seed=846410 + index) for index in range(8)])
    _write_rows(dev_path, [_row(index, index % 2, seed=846420 + index, event_prefix="dev") for index in range(4)])
    source_policy, before, config, source_sha = _patch_source(monkeypatch, source_path)

    output = tmp_path / "repair.pt"
    result = runner.train_gate_repair(source_path, [train_path], [dev_path], output)

    assert result["status"] == "COMPLETED"
    assert result["stage"] == runner.STAGE
    assert result["invariants"]["dev_gradients"] is False
    assert result["invariants"]["checkpoint_selection"] == "final_epoch_only"
    assert result["invariants"]["non_adapter_state_bitwise_unchanged"] is True
    assert result["invariants"]["proposal_logits_bitwise_unchanged"] is True
    assert result["invariants"]["adapter_state_changed"] is True
    assert result["metadata"]["supervised_training_called"] is True
    assert result["metadata"]["ppo_called"] is False
    assert result["metadata"]["teacher_enabled"] is False
    assert result["metadata"]["PPO_ALLOWED"] is False
    assert result["metadata"]["checkpoint_promotable"] is False
    assert result["metadata"]["source_checkpoint_sha256"] == source_sha

    after = source_policy.model.state_dict()
    for name, value in before.items():
        if not name.startswith(runner.ADAPTER_PREFIX):
            assert torch.equal(value, after[name]), name
    assert any(not torch.equal(value, after[name]) for name, value in before.items()
               if name.startswith(runner.ADAPTER_PREFIX))

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert "optimizer" not in payload
    assert payload["metadata"]["stage"] == runner.STAGE
    assert len((tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()) == runner.EPOCHS
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "COMPLETED"
    assert progress["completed_epochs"] == runner.EPOCHS
    assert len(progress["history"]) == runner.EPOCHS

    restored, restored_payload = runner.load_repaired_checkpoint(
        output, config, runner._sha256(output)
    )
    assert isinstance(restored, TemperatureCalibratedGateResidualAdapterStudentPPOV42)
    assert restored_payload["metadata"]["architecture_id"] == runner.CALIBRATED_ARCHITECTURE_ID
    assert restored_payload["metadata"]["temperature"] == 2.5
    for name, value in payload["model"].items():
        assert torch.equal(value, restored.model.state_dict()[name]), name


def test_training_rejects_single_class_and_seed_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_path = tmp_path / "source.pt"
    source_path.write_bytes(b"source")
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_rows(train_path, [_row(index, 0, seed=846500 + index) for index in range(2)])
    _write_rows(dev_path, [_row(index, index % 2, seed=846600 + index, event_prefix="dev") for index in range(2)])
    monkeypatch.setattr(runner.diagnostic, "build_scenario_config", lambda scenario: object())
    with pytest.raises(runner.RepairInputError, match="both teacher_gate classes"):
        runner.train_gate_repair(source_path, [train_path], [dev_path], tmp_path / "repair.pt")

    _write_rows(train_path, [_row(index, index % 2, seed=846700 + index) for index in range(2)])
    _write_rows(dev_path, [_row(index, index % 2, seed=846700 + index, event_prefix="dev") for index in range(2)])
    with pytest.raises(runner.RepairInputError, match="seed overlap"):
        runner.train_gate_repair(source_path, [train_path], [dev_path], tmp_path / "repair2.pt")


def test_training_rejects_nonfinite_actor_row(tmp_path: Path):
    path = tmp_path / "labels.jsonl"
    row = _row(0, 0, seed=846800)
    row["observation_v2_1"][3] = float("nan")
    _write_rows(path, [row])
    with pytest.raises(runner.RepairInputError, match="not finite 88D"):
        runner._load_rows([path], split="train")


def test_schema_and_output_collision_are_rejected(tmp_path: Path):
    path = tmp_path / "labels.jsonl"
    row = _row(0, 0, seed=846850)
    row["action_schema"] = "wrong-action-schema"
    _write_rows(path, [row])
    with pytest.raises(runner.RepairInputError, match="action_schema"):
        runner._load_rows([path], split="train")

    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"existing")
    with pytest.raises(runner.RepairInputError, match="overwrite"):
        runner._resolve_output(existing)


def test_loader_rejects_optimizer_and_changed_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_path = tmp_path / "source.pt"
    source_path.write_bytes(b"source-loader")
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_rows(train_path, [_row(index, index % 2, seed=846900 + index) for index in range(4)])
    _write_rows(dev_path, [_row(index, index % 2, seed=846910 + index, event_prefix="dev") for index in range(2)])
    _, _, config, source_sha = _patch_source(monkeypatch, source_path)
    output = tmp_path / "repair.pt"
    runner.train_gate_repair(source_path, [train_path], [dev_path], output)

    payload = torch.load(output, map_location="cpu", weights_only=False)
    bad_optimizer = copy.deepcopy(payload)
    bad_optimizer["optimizer"] = {}
    optimizer_path = tmp_path / "optimizer.pt"
    torch.save(bad_optimizer, optimizer_path)
    with pytest.raises(runner.RepairInputError, match="optimizer"):
        runner.load_repaired_checkpoint(optimizer_path, config, runner._sha256(optimizer_path))

    bad_stage = copy.deepcopy(payload)
    bad_stage["metadata"]["stage"] = "V4_2_TRAINED_DENSE30_V2_DIAGNOSTIC"
    stage_path = tmp_path / "stage.pt"
    torch.save(bad_stage, stage_path)
    with pytest.raises(runner.RepairInputError, match="stage"):
        runner.load_repaired_checkpoint(stage_path, config, runner._sha256(stage_path))
