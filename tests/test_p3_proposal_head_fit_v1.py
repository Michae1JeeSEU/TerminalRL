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
from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training import train_p3_proposal_head_probe_v1 as runner


def _source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = runner._build_execution_config()
    source = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
    )
    source_path = tmp_path / "source.pt"
    save_checkpoint(source_path, source, config, stage="fixture_source", formal=False)
    monkeypatch.setattr(runner, "SOURCE_CHECKPOINT_SHA256", runner._sha256(source_path))
    return source_path, config, source


def _rows(source, *, empty: bool = False) -> list[dict]:
    rows: list[dict] = []
    observations = []
    for index in range(6):
        values = np.zeros(88, dtype=np.float32)
        values[index] = 0.2 + index * 0.03
        observations.append(values)
    for index in range(6):
        # The collector records one actor state per forward call.  Keep the
        # fixture row-wise so the runner's source-logit check covers that
        # serialization boundary rather than only a batched forward.
        with torch.no_grad():
            tensor = torch.as_tensor(observations[index][None, :], dtype=torch.float32)
            gate, proposal = source.model.distributions(tensor)
        split = "train" if index < 4 else "validation"
        seed = (873101, 873102, 873103, 873101, 873104, 873105)[index]
        rows.append({
            "observation": observations[index].tolist(),
            "source_proposal_logits": proposal.logits[0].tolist(),
            "source_gate_logits": gate.logits[0].tolist(),
            "certified_proposals": [] if empty else ([0, 1] if index % 2 == 0 else [3]),
            "seed": seed,
            "split": split,
            "step": index * 10,
            "uid": 100 + index,
            "arrival_id": 200 + index,
        })
    return rows


def _collection(tmp_path: Path, rows: list[dict], *, exact: bool = True) -> Path:
    collection = tmp_path / "collection"
    collection.mkdir(parents=True)
    (collection / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    reports = [
        {
            "seed": seed,
            "scenario": scenario,
            "split": split,
            "replay_exact": exact,
            "outcomes_match": exact,
            "rng_unchanged": exact,
            "model_weights_unchanged": exact,
            "errors": [],
        }
        for seed, scenario, split in runner.EXPECTED_SOURCE_RECORDS
    ]
    (collection / "manifest.json").write_text(
        json.dumps({
            "status": "COMPLETED",
            "mode": "full",
            "source_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256,
            "source_model_digest_unchanged": True,
            "protocol": {
                "sha256": runner._sha256(runner.PROTOCOL),
                "source_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256,
            },
            "reports": reports,
            "records_expected": 6,
            "rows_expected": 192,
            "rows_observed": len(rows),
        }), encoding="utf-8",
    )
    return collection


def test_freeze_scope_is_exactly_proposal_head() -> None:
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    parameters, names = runner._freeze_to_proposal_head(policy)
    assert names == ["proposal_head.weight", "proposal_head.bias"]
    assert sum(parameter.numel() for parameter in parameters) == 2425
    assert all(
        parameter.requires_grad == name.startswith("proposal_head.")
        for name, parameter in policy.model.named_parameters()
    )


def test_rowwise_collection_logits_validate_against_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, _, source = _source_fixture(tmp_path, monkeypatch)
    del source_path
    rows = _rows(source)
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
    )
    # Loading is exercised by the fit tests; this test isolates the collector
    # row-wise validation seam with the same source model values.
    policy.model.load_state_dict(source.model.state_dict(), strict=True)
    runner._validate_source_rows(policy, rows)


def test_fit_updates_only_proposal_and_roundtrips_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, _, source = _source_fixture(tmp_path, monkeypatch)
    collection = _collection(tmp_path, _rows(source))

    result = runner.fit_proposal_head(collection, tmp_path / "fit", source_checkpoint=source_path)

    assert result["status"] in {"COMPLETED", "STOPPED_KL_CAP"}
    assert result["accepted_steps"] >= 1
    assert result["invariants"]["trainable_parameter_count"] == 2425
    assert result["invariants"]["nonproposal_state_bitwise_unchanged"] is True
    assert result["invariants"]["gate_logits_bitwise_unchanged"] is True
    assert result["invariants"]["all_nonproposal_parameters_frozen"] is True
    assert result["invariants"]["checkpoint_roundtrip_model_exact"] is True
    assert result["invariants"]["checkpoint_roundtrip_optimizer_exact"] is True
    payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert payload["metadata"]["stage"] == runner.STAGE
    assert payload["metadata"]["source_checkpoint_sha256"] == runner._sha256(source_path)
    assert payload["metadata"]["trainable_parameter_count"] == 2425


def test_retention_cap_rolls_back_real_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, _, source = _source_fixture(tmp_path, monkeypatch)
    collection = _collection(tmp_path, _rows(source))
    monkeypatch.setattr(runner, "RETENTION_KL_MAX", 0.0)

    result = runner.fit_proposal_head(collection, tmp_path / "rollback", source_checkpoint=source_path)

    assert result["status"] == "STOPPED_KL_CAP"
    assert result["accepted_steps"] == 0
    assert result["rejected_steps"] == 1
    assert result["effective_fit"] is False
    assert result["invariants"]["nonproposal_state_bitwise_unchanged"] is True
    assert result["invariants"]["checkpoint_roundtrip_model_exact"] is True


def test_all_empty_certified_rows_are_reported_without_effective_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, _, source = _source_fixture(tmp_path, monkeypatch)
    collection = _collection(tmp_path, _rows(source, empty=True))

    result = runner.fit_proposal_head(collection, tmp_path / "empty", source_checkpoint=source_path)

    assert result["status"] == "NO_EFFECTIVE_FIT"
    assert result["accepted_steps"] == 0
    assert result["rejected_steps"] == 0
    assert result["effective_fit"] is False
    assert result["invariants"]["nonproposal_state_bitwise_unchanged"] is True


def test_collection_integrity_and_split_validation_are_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, _, source = _source_fixture(tmp_path, monkeypatch)
    collection = _collection(tmp_path, _rows(source), exact=False)
    with pytest.raises(runner.ProposalHeadProbeInputError, match="exact"):
        runner.fit_proposal_head(collection, tmp_path / "bad_exact", source_checkpoint=source_path)

    rows = _rows(source)
    rows[0]["seed"] = 873104
    collection = _collection(tmp_path / "leak", rows)
    with pytest.raises(runner.ProposalHeadProbeInputError, match="wrong split"):
        runner.fit_proposal_head(collection, tmp_path / "bad_split", source_checkpoint=source_path)
