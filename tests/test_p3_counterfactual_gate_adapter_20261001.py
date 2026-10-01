"""Focused tests for the bounded P3 gate-adapter fit."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from terminal_marl.training import fit_p3_counterfactual_gate_adapter_20261001 as fit


def _calibrated_model() -> nn.Module:
    from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
        TemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
    )

    torch.manual_seed(17)
    model = TemperatureCalibratedGateResidualAdapterActorCentralCriticV42(88, temperature=2.5)
    model.eval()
    return model


def _source_checkpoint(tmp_path: Path, model: nn.Module) -> Path:
    path = tmp_path / "source.pt"
    torch.save(
        {
            "metadata": {
                "schema_version": "test",
                "observation_dimension": 88,
                "architecture_id": "v4_2_hierarchical_gate_temperature_calibrated_residual_adapter_v3",
                "temperature": 2.5,
                "execution_config": {"task_neutral": True},
                "training_called": False,
                "optimizer_saved": False,
            },
            "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            "report": {"status": "source"},
        },
        path,
    )
    return path


def _fit_inputs(model: nn.Module, *, target_shift: float = 0.0002) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(19)
    anchor = torch.randn(8, 88)
    target = anchor[:4].clone()
    with torch.no_grad():
        gate, _ = model.distributions(target)
        p0 = gate.probs[:, 1].clone()
    pstar = torch.clamp(p0 + target_shift, 1e-4, 1.0 - 1e-4)
    return anchor, target, pstar, p0


def _run_core(tmp_path: Path, *, target_shift: float = 0.0002, p0_override: torch.Tensor | None = None):
    model = _calibrated_model()
    anchor, target, pstar, p0 = _fit_inputs(model, target_shift=target_shift)
    source = _source_checkpoint(tmp_path, model)
    return fit._fit_core(
        model=model,
        source_metadata={"observation_dimension": 88, "temperature": 2.5},
        source_sha=fit._sha256(source),
        source_payload_report={},
        anchor_world=anchor,
        target_world=target,
        target_values=pstar,
        target_ids=[1, 2, 3, 4],
        target_p0_recorded=p0 if p0_override is None else p0_override,
        source_checkpoint_path=source,
        output_dir=tmp_path / "fit",
    )


def test_parameter_scope_only_adapter_and_proposal_is_frozen() -> None:
    model = _calibrated_model()
    trainable, names = fit._parameter_scope(model)
    assert trainable
    assert names
    assert all(name.startswith("gate_adapter.") for name in names)
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if name in names)
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if name not in names)
    before = {name: value.detach().clone() for name, value in model.named_parameters() if not name.startswith("gate_adapter.")}
    gate, proposal = model.distributions(torch.randn(3, 88))
    loss = gate.logits[:, 1].mean() + proposal.logits[:, 0].mean()
    loss.backward()
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in names)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())
    assert torch.equal(model.gate_temperature, torch.tensor(2.5))


def test_chunk_weighted_mean_equals_full_mean() -> None:
    values = torch.linspace(-2.0, 3.0, 1025)
    assert fit._mean_chunked(values, 512).item() == pytest.approx(values.mean().item(), abs=1e-12)
    full_input = values.clone().requires_grad_(True)
    full_input.square().mean().backward()
    chunk_input = values.clone().requires_grad_(True)
    for start in range(0, len(chunk_input), 512):
        chunk = chunk_input[start : start + 512]
        (chunk.square().mean() * (len(chunk) / len(chunk_input))).backward()
    assert torch.equal(full_input.grad, chunk_input.grad)


def test_stable_kl_has_zero_identity_and_no_floor() -> None:
    logits = torch.tensor([-80.0, 0.0, 80.0], dtype=torch.float64)
    result = fit._binary_kl_from_logits(logits, logits)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-15)


def test_baseline_p0_mismatch_stops_before_optimizer(tmp_path: Path) -> None:
    model = _calibrated_model()
    _, _, _, p0 = _fit_inputs(model)
    with pytest.raises(fit.FitError, match="initial target p0"):
        _run_core(tmp_path, p0_override=p0 + 0.1)
    assert not (tmp_path / "fit" / "candidate.pt").exists()


def test_intrinsic_failure_writes_report_without_candidate(tmp_path: Path) -> None:
    report = _run_core(tmp_path, target_shift=0.0)
    assert report["status"] == "FIT_SCREEN_FAIL"
    assert report["candidate_checkpoint"] is None
    assert (tmp_path / "fit" / "fit.json").is_file()
    assert not (tmp_path / "fit" / "candidate.pt").exists()


def test_candidate_checkpoint_is_strictly_reloadable_and_second_run_refuses(tmp_path: Path) -> None:
    report = _run_core(tmp_path, target_shift=0.0002)
    assert report["status"] == "FIT_OK"
    candidate = tmp_path / "fit" / "candidate.pt"
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    assert set(payload) == {"metadata", "model", "report"}
    assert "optimizer" not in payload
    reloaded = fit._instantiate_model_state(payload["model"])
    assert all(torch.equal(reloaded.state_dict()[name], value) for name, value in payload["model"].items())
    with pytest.raises(fit.FitError, match="output already exists"):
        _run_core(tmp_path, target_shift=0.0002)


def test_rollback_restores_adapter_and_stops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real = fit._evaluate_kl
    calls = {"count": 0}

    def forced(model, observations, reference_gate_logits, *, chunk_size=fit.ANCHOR_CHUNK_SIZE):
        calls["count"] += 1
        values, q = real(model, observations, reference_gate_logits, chunk_size=chunk_size)
        if calls["count"] == 2:
            values = values.clone()
            values[0] = fit.KL_CAP + 1.0
        return values, q

    monkeypatch.setattr(fit, "_evaluate_kl", forced)
    report = _run_core(tmp_path, target_shift=0.0002)
    assert report["stop_reason"] == "ROLLBACK_STOP"
    assert report["rejected_updates"] == 1
    assert report["accepted_updates"] == 0
    assert report["candidate_checkpoint"] is None


def test_unknown_input_is_rejected_without_fabricated_fit(tmp_path: Path) -> None:
    with pytest.raises(fit.FitError, match="fit preflight missing"):
        fit.run_fit(
            targets_path=tmp_path / "targets.json",
            readiness_path=tmp_path / "readiness.json",
            anchors_path=tmp_path / "anchors",
            source_checkpoint=tmp_path / "source.pt",
            output_dir=tmp_path / "fit",
        )


def test_candidate_metadata_does_not_copy_source_training_history() -> None:
    metadata = fit._checkpoint_metadata(
        {
            "schema_version": "v",
            "architecture_id": "a",
            "execution_config": {"task_neutral": True},
            "optimizer_saved": True,
            "training_called": False,
            "report": {"old": "history"},
        },
        source_sha=fit.BEFORE_CHECKPOINT_SHA256,
    )
    assert metadata["stage"] == fit.STAGE
    assert metadata["parent_checkpoint_sha256"] == fit.BEFORE_CHECKPOINT_SHA256
    assert metadata["candidate_promoted"] is False
    assert metadata["PPO_ALLOWED"] is False
    assert "report" not in metadata
    assert metadata["source_training_metadata_not_restored"] is True


def test_nonfinite_technical_error_preserves_partial_without_rollback_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "technical"

    def failing_fit(**kwargs):
        del kwargs
        output.mkdir(parents=True)
        (output / "progress.json").write_text(
            json.dumps({"status": "RUNNING", "update": 3, "accepted_updates": 2,
                        "rejected_updates": 0}), encoding="utf-8"
        )
        raise fit.FitError("non-finite anchor KL")

    monkeypatch.setattr(fit, "run_fit", failing_fit)
    assert fit.main(["--phase", "fit", "--output-dir", str(output)]) == 2
    partial = json.loads((output / "partial.json").read_text(encoding="utf-8"))
    error = json.loads((output / "error.json").read_text(encoding="utf-8"))
    assert partial["status"] == "PARTIAL"
    assert partial["latest_progress"]["accepted_updates"] == 2
    assert partial["latest_progress"]["rejected_updates"] == 0
    assert "ROLLBACK_STOP" not in json.dumps(partial)
    assert error["error_type"] == "FitError"
