from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
from terminal_marl.training import train_p3_gate_coverage_repair_v2 as runner


def _row(index: int, gate: int, *, seed: int, prefix: str) -> dict:
    observation = np.zeros(88, dtype=np.float32)
    observation[index % 88] = float(index + 1) / 10.0
    observation[(index + 17) % 88] += float(gate)
    return {
        "decision_event_id": f"{prefix}-event-{index}",
        "seed": seed,
        "observation_v2_1": observation.tolist(),
        "observation_schema": runner.OBSERVATION_SCHEMA,
        "action_schema": runner.ACTION_SCHEMA,
        "teacher_gate": int(gate),
        "actor_mask": True,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, object]:
    config = runner._build_execution_config()
    source = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
    )
    path = tmp_path / "source.pt"
    save_checkpoint(path, source, config, stage="fixture_source", formal=False)
    monkeypatch.setattr(runner, "SOURCE_CHECKPOINT_SHA256", runner._sha256(path))
    return path, config


def test_group_weights_are_fixed_and_class_balanced() -> None:
    anchor = [
        _row(0, 0, seed=1, prefix="anchor"),
        _row(1, 0, seed=2, prefix="anchor"),
        _row(2, 1, seed=3, prefix="anchor"),
    ]
    coverage = [
        _row(3, 0, seed=4, prefix="coverage"),
        _row(4, 1, seed=5, prefix="coverage"),
    ]
    weights, report = runner.build_sample_weights(anchor, coverage)

    assert float(weights.sum()) == pytest.approx(1.0)
    assert float(weights[:3].sum()) == pytest.approx(0.5)
    assert float(weights[3:].sum()) == pytest.approx(0.5)
    assert report["anchor"]["class_total_weights"] == {"WAIT": pytest.approx(.25), "RELEASE": pytest.approx(.25)}
    assert report["coverage"]["class_total_weights"] == {"WAIT": pytest.approx(.25), "RELEASE": pytest.approx(.25)}


def test_weighted_batch_loss_keeps_global_weights_and_scales_short_batch() -> None:
    per_sample = torch.tensor([1.0, 3.0])
    weights = torch.tensor([0.1, 0.2])
    total_weight = torch.tensor(1.0)
    loss = runner._weighted_batch_loss(
        per_sample,
        weights,
        total_weight=total_weight,
        population_size=6,
    )
    assert loss.item() == pytest.approx((0.1 + 0.6) * 3.0)


def test_split_rejects_seed_overlap_and_global_event_duplicates() -> None:
    anchor = [_row(0, 0, seed=1, prefix="anchor"), _row(1, 1, seed=2, prefix="anchor")]
    coverage = [_row(2, 0, seed=3, prefix="coverage"), _row(3, 1, seed=4, prefix="coverage")]
    dev = [_row(4, 0, seed=3, prefix="dev"), _row(5, 1, seed=5, prefix="dev")]
    with pytest.raises(runner.GateCoverageRepairInputError, match="seed overlap"):
        runner._validate_splits(anchor, coverage, dev)

    dev = [_row(4, 0, seed=5, prefix="dev"), _row(5, 1, seed=6, prefix="dev")]
    dev[0]["decision_event_id"] = anchor[0]["decision_event_id"]
    with pytest.raises(runner.GateCoverageRepairInputError, match="globally unique"):
        runner._validate_splits(anchor, coverage, dev)


def test_fit_freezes_non_adapter_and_roundtrips_strict_calibrated_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, config = _source_fixture(tmp_path, monkeypatch)
    anchor_path = tmp_path / "anchor.jsonl"
    coverage_path = tmp_path / "coverage.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_rows(anchor_path, [_row(i, i % 2, seed=100 + i, prefix="anchor") for i in range(8)])
    _write_rows(coverage_path, [_row(i, i % 2, seed=200 + i, prefix="coverage") for i in range(8)])
    _write_rows(dev_path, [_row(i, i % 2, seed=300 + i, prefix="dev") for i in range(4)])

    result = runner.fit_gate_coverage_repair(
        source_checkpoint=source_path,
        anchor_train=[anchor_path],
        coverage_train=[coverage_path],
        dev=[dev_path],
        output=tmp_path / "fit",
    )

    assert result["status"] == "COMPLETED"
    assert result["stage"] == runner.STAGE
    assert result["checkpoint_roundtrip_model_exact"] is True
    assert result["invariants"]["non_adapter_state_bitwise_unchanged"] is True
    assert result["invariants"]["proposal_logits_bitwise_unchanged"] is True
    assert result["invariants"]["adapter_state_changed"] is True
    assert result["invariants"]["checkpoint_selection"] == "final_epoch_only"
    assert result["invariants"]["source_optimizer_state_copied"] is False

    checkpoint = Path(result["checkpoint"])
    report = Path(result["fit_report"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "optimizer" not in payload
    assert payload["metadata"]["stage"] == runner.STAGE
    assert payload["metadata"]["source_checkpoint_sha256"] == runner._sha256(source_path)
    assert payload["metadata"]["v4_2_rejected_release_hold_mode"] == "canonical_wait"
    assert payload["report"]["coverage_label_semantics"].startswith("state_only_gate_feasibility")
    assert len(payload["report"]["history"]) == runner.EPOCHS
    assert {"anchor", "coverage", "dev", "dev_by_source"} <= set(payload["report"]["metrics_before"])
    assert {"anchor", "coverage", "dev", "dev_by_source"} <= set(payload["report"]["metrics_after"])
    assert list(payload["report"]["metrics_before"]["dev_by_source"]) == [str(dev_path)]

    restored = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
    )
    load_checkpoint(checkpoint, restored, config, load_optimizer=False)
    for name, value in payload["model"].items():
        assert torch.equal(value, restored.model.state_dict()[name]), name
    assert json.loads(report.read_text(encoding="utf-8"))["checkpoint_roundtrip_model_exact"] is True


def test_fit_refuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path, _ = _source_fixture(tmp_path, monkeypatch)
    anchor_path = tmp_path / "anchor.jsonl"
    coverage_path = tmp_path / "coverage.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    rows = [_row(0, 0, seed=1000, prefix="x"), _row(1, 1, seed=1001, prefix="x")]
    _write_rows(anchor_path, rows)
    _write_rows(coverage_path, [_row(2, 0, seed=1002, prefix="coverage"), _row(3, 1, seed=1003, prefix="coverage")])
    _write_rows(dev_path, [_row(4, 0, seed=1004, prefix="dev"), _row(5, 1, seed=1005, prefix="dev")])
    output = tmp_path / "existing"
    output.mkdir()
    (output / "marker").write_text("keep", encoding="utf-8")

    with pytest.raises(runner.GateCoverageRepairInputError, match="overwrite"):
        runner.fit_gate_coverage_repair(source_path, [anchor_path], [coverage_path], [dev_path], output)
    assert (output / "marker").read_text(encoding="utf-8") == "keep"
