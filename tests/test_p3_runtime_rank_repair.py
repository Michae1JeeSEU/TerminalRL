from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import checkpoint_metadata
from terminal_marl.training import train_p3_runtime_rank_repair as runner


class ToyModel(nn.Module):
    architecture_id = "toy"

    def __init__(self) -> None:
        super().__init__()
        self.actor_encoder = nn.Linear(88, 4, bias=False)
        self.gate_head = nn.Linear(4, 2)
        self.proposal_head = nn.Linear(4, 25)

    def distributions(self, observations: torch.Tensor):
        encoded = self.actor_encoder(observations)
        return Categorical(logits=self.gate_head(encoded)), Categorical(logits=self.proposal_head(encoded))


def _rows(count: int, *, offset: float = 0.0) -> list[dict]:
    rows = []
    for index in range(count):
        observation = [0.0] * 88
        observation[index % 8] = 0.2 + offset + index * 0.03
        rows.append({
            "observation": observation,
            "certified_proposals": [24],
        })
    return rows


def _toy_policy() -> SimpleNamespace:
    model = ToyModel()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("proposal_head."))
    parameters = [parameter for name, parameter in model.named_parameters() if name.startswith("proposal_head.")]
    policy = SimpleNamespace(model=model, device=torch.device("cpu"))
    policy.optimizer = torch.optim.Adam(parameters, lr=runner.LEARNING_RATE)
    return policy


def _source_logits(policy: SimpleNamespace, rows: list[dict]) -> torch.Tensor:
    observations = runner._observations(rows, policy.device)
    return runner._source_raw_logits(policy.model, observations)


def test_raw_proposal_logits_bypass_distribution_normalization() -> None:
    policy = _toy_policy()
    rows = _rows(2)
    observations = runner._observations(rows, policy.device)
    raw = runner._raw_proposal_logits(policy.model, observations)
    _, distribution = policy.model.distributions(observations)
    assert torch.equal(raw, policy.model.proposal_head(policy.model.actor_encoder(observations)))
    assert not torch.equal(raw, distribution.logits)

    objective = runner.k3_coverage_objective(raw, [[24], [24]])
    objective["loss"].backward()
    assert policy.model.proposal_head.weight.grad is not None
    assert policy.model.proposal_head.bias.grad is not None
    assert policy.model.actor_encoder.weight.grad is None


def test_kl_cap_rolls_back_model_and_optimizer_state() -> None:
    policy = _toy_policy()
    train_rows = _rows(2)
    validation_rows = _rows(2, offset=0.4)
    probe = _source_logits(policy, train_rows)
    top3 = set(torch.argsort(probe[0], descending=True)[:3].tolist())
    train_rows[0]["certified_proposals"] = [next(index for index in range(25) if index not in top3)]
    train_source = _source_logits(policy, train_rows)
    validation_source = _source_logits(policy, validation_rows)
    before_model = runner._state_cpu(policy.model)
    before_optimizer = runner._optimizer_state_cpu(policy.optimizer)

    result = runner._fit_steps(
        policy,
        train_rows,
        validation_rows,
        source_train_logits=train_source,
        source_validation_logits=validation_source,
        trainable_parameters=[p for n, p in policy.model.named_parameters() if n.startswith("proposal_head.")],
        max_steps=1,
        retention_kl_max=0.0,
    )

    assert result["rejected_steps"] == 1
    assert result["accepted_steps"] == 0
    assert result["stopped_reason"] == "retention_kl_cap_exceeded"
    assert runner._state_equal(before_model, runner._state_cpu(policy.model))
    assert runner._optimizer_state_equal(before_optimizer, runner._optimizer_state_cpu(policy.optimizer))


def test_validation_rows_do_not_participate_in_update() -> None:
    first = _toy_policy()
    second = _toy_policy()
    second.model.load_state_dict(first.model.state_dict())
    train_rows = _rows(3)
    validation_a = _rows(2, offset=0.1)
    validation_b = _rows(2, offset=3.0)
    train_source = _source_logits(first, train_rows)
    validation_source_a = _source_logits(first, validation_a)
    validation_source_b = _source_logits(second, validation_b)
    train_source_b = _source_logits(second, train_rows)
    parameters_first = [p for n, p in first.model.named_parameters() if n.startswith("proposal_head.")]
    parameters_second = [p for n, p in second.model.named_parameters() if n.startswith("proposal_head.")]

    runner._fit_steps(
        first, train_rows, validation_a,
        source_train_logits=train_source,
        source_validation_logits=validation_source_a,
        trainable_parameters=parameters_first,
        max_steps=1,
        retention_kl_max=10.0,
    )
    runner._fit_steps(
        second, train_rows, validation_b,
        source_train_logits=train_source_b,
        source_validation_logits=validation_source_b,
        trainable_parameters=parameters_second,
        max_steps=1,
        retention_kl_max=10.0,
    )
    first_state = runner._state_cpu(first.model)
    second_state = runner._state_cpu(second.model)
    assert runner._state_equal(first_state, second_state)


def test_checkpoint_save_reload_preserves_all_state(tmp_path) -> None:
    config = runner._build_execution_config()
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
    )
    parameters, _ = runner._freeze_to_proposal_head(policy)
    policy.optimizer = torch.optim.Adam(parameters, lr=runner.LEARNING_RATE)
    rows = _rows(1)
    metadata = dict(checkpoint_metadata(config))
    metadata.update({"stage": runner.STAGE, "method": "toy-roundtrip"})
    checkpoint = tmp_path / "candidate.pt"
    model_exact, optimizer_exact, gate_exact, allstate_exact = runner._save_and_restore(
        checkpoint,
        policy,
        config,
        metadata,
        {"stage": runner.STAGE},
        rows,
        rows,
    )
    assert checkpoint.is_file()
    assert model_exact is True
    assert optimizer_exact is True
    assert gate_exact is True
    assert allstate_exact is True


def test_runtime_scope_requires_fixed_other_tape_scope_marker() -> None:
    row = {
        "uid": 7,
        "label_scope": "catalogue_only",
    }
    with pytest.raises(runner.RuntimeRankRepairInputError, match="scope is invalid"):
        runner._validate_runtime_scope(row, 0)


def _write_runtime_collector_fixture(tmp_path, *, bad_model_record: bool = False):
    rows = []
    for index in range(runner.TRAIN_ROWS_EXPECTED + runner.VALIDATION_ROWS_EXPECTED):
        if index < runner.TRAIN_ROWS_EXPECTED:
            split = "train"
            seeds = sorted(runner.TRAIN_SEEDS)
            local_index = index
        else:
            split = "validation"
            seeds = sorted(runner.VALIDATION_SEEDS)
            local_index = index - runner.TRAIN_ROWS_EXPECTED
        seed = seeds[local_index % len(seeds)]
        rows.append({
            "observation": [0.0] * 88,
            "source_proposal_logits": [0.0] * 25,
            "source_gate_logits": [0.0] * 2,
            "catalogue_certified_proposals": [],
            "certified_proposals": [],
            "label_scope": runner.RUNTIME_LABEL_SCOPE,
            "runtime_label_hash": "fixture",
            "seed": seed,
            "split": split,
            "step": local_index,
            "uid": index,
            "arrival_id": index,
        })
    records = []
    for index, (seed, scenario, split) in enumerate(runner.EXPECTED_SOURCE_RECORDS):
        records.append({
            "seed": seed,
            "scenario": scenario,
            "split": split,
            "replay_exact": True,
            "outcomes_match": True,
            "rng_unchanged": True,
            "model_weights_unchanged": not (bad_model_record and index == 2),
            "errors": [],
            "step_inputs": [],
        })
    manifest = {
        "stage": "P3_RUNTIME_RANK_REPAIR_COLLECTION",
        "status": "COMPLETED",
        "mode": "full",
        "runtime_labels": True,
        "source_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256,
        "records": records,
        "records_count": 6,
        "records_expected": 6,
        "rows_count": 164,
        "rows_expected": 164,
        "split_counts": {"train": 82, "validation": 82},
    }
    (tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_runtime_collector_manifest_schema_loads_without_duplicate_top_level_digest(tmp_path) -> None:
    _write_runtime_collector_fixture(tmp_path)
    rows, report = runner._load_collection(tmp_path)
    assert len(rows) == 164
    assert sum(row["split"] == "train" for row in rows) == 82
    assert sum(row["split"] == "validation" for row in rows) == 82
    assert rows[0]["label_scope"] == runner.RUNTIME_LABEL_SCOPE
    assert len(rows[0]["observation"]) == 88
    assert len(rows[0]["source_proposal_logits"]) == 25
    assert len(rows[0]["source_gate_logits"]) == 2
    assert report["runtime_labels"] is True


def test_runtime_collector_manifest_rejects_record_model_change(tmp_path) -> None:
    _write_runtime_collector_fixture(tmp_path, bad_model_record=True)
    with pytest.raises(runner.RuntimeRankRepairInputError, match="model_weights_unchanged"):
        runner._load_collection(tmp_path)
