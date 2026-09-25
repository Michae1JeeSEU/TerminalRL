"""Offline gate-coverage repair fit for the P3 holding diagnosis.

This runner fits only the calibrated model's ``gate_adapter`` from two
labelled groups.  The anchor group is the existing E04 gate data and the
coverage group is the state-only gate-feasibility collection.  The proposal
head, actor encoder, gate head, critic, and temperature remain frozen.  The
runner never creates an environment, invokes a Teacher, or performs PPO.

The output is diagnostic-only and is intentionally written as a normal
calibrated V4.2 checkpoint so that the existing strict checkpoint loader can
read its model state with the canonical-WAIT execution configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import checkpoint_metadata, load_checkpoint
from terminal_marl.training import evaluate_random_soc_gate_repair_v1 as base
from terminal_marl.training import train_random_soc_gate_repair_v1 as legacy
from terminal_marl.training import run_v4_2_p3_controlled_emergency_admission_v5 as admission


STAGE = "P3_GATE_COVERAGE_REPAIR_V2"
OBSERVATION_DIM = 88
OBSERVATION_SCHEMA = legacy.OBSERVATION_SCHEMA
ACTION_SCHEMA = legacy.ACTION_SCHEMA
ADAPTER_PREFIX = "gate_adapter."

SOURCE_CHECKPOINT_SHA256 = (
    "e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353"
)
TRAINING_SEED = 853401
SEED = TRAINING_SEED
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
GRADIENT_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.1
ANCHOR_GROUP_WEIGHT = 0.5
COVERAGE_GROUP_WEIGHT = 0.5


class GateCoverageRepairInputError(ValueError):
    """Raised when a fit input or checkpoint violates this fit contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateCoverageRepairInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    return admission._jsonable(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _paths(value: Sequence[str | Path] | str | Path) -> list[str | Path]:
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def _load_rows(paths: Sequence[str | Path] | str | Path, *, split: str) -> list[dict[str, Any]]:
    """Reuse the V1 row/schema checks while keeping this stage's error type."""

    values = _paths(paths)
    _require(bool(values), f"{split} label paths are required")
    try:
        return legacy._load_rows(values, split=split)
    except legacy.RepairInputError as exc:
        raise GateCoverageRepairInputError(str(exc)) from exc


def _validate_splits(
    anchor_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
) -> None:
    _require(bool(anchor_rows), "anchor train rows are empty")
    _require(bool(coverage_rows), "coverage train rows are empty")
    _require(bool(dev_rows), "dev rows are empty")

    all_rows = [*anchor_rows, *coverage_rows, *dev_rows]
    event_ids = [str(row["decision_event_id"]) for row in all_rows]
    seen_event_ids: set[str] = set()
    duplicate_event_ids: set[str] = set()
    for event in event_ids:
        if event in seen_event_ids:
            duplicate_event_ids.add(event)
        seen_event_ids.add(event)
    duplicates = sorted(duplicate_event_ids)
    _require(not duplicates, f"decision_event_id is not globally unique: {duplicates[:5]}")

    train_seeds = {int(row["seed"]) for row in [*anchor_rows, *coverage_rows]}
    dev_seeds = {int(row["seed"]) for row in dev_rows}
    overlap = sorted(train_seeds & dev_seeds)
    _require(not overlap, f"train/dev seed overlap: {overlap}")

    for name, rows in (("anchor", anchor_rows), ("coverage", coverage_rows)):
        labels = {int(row["teacher_gate"]) for row in rows}
        _require(labels == {0, 1}, f"{name} training labels must contain both teacher_gate classes")


def _class_balanced_group_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Give each class half of its group's fixed total weight."""

    _require(float(group_weight) > 0.0, "group weight must be positive")
    labels = np.asarray([int(row["teacher_gate"]) for row in rows], dtype=np.int64)
    counts = {label: int((labels == label).sum()) for label in (0, 1)}
    _require(all(counts[label] > 0 for label in (0, 1)), "both gate classes are required")
    weights = np.asarray(
        [float(group_weight) / (2.0 * counts[int(label)]) for label in labels],
        dtype=np.float64,
    )
    _require(bool(np.isfinite(weights).all()), "non-finite group sample weights")
    _require(abs(float(weights.sum()) - float(group_weight)) <= 1e-12,
             "group sample weights do not sum to the configured group weight")
    return weights, {
        "rows": int(len(rows)),
        "class_counts": {"WAIT": counts[0], "RELEASE": counts[1]},
        "class_total_weights": {
            "WAIT": float(weights[labels == 0].sum()),
            "RELEASE": float(weights[labels == 1].sum()),
        },
        "group_weight": float(group_weight),
        "row_weight_by_class": {
            "WAIT": float(group_weight) / (2.0 * counts[0]),
            "RELEASE": float(group_weight) / (2.0 * counts[1]),
        },
    }


def build_sample_weights(
    anchor_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return fixed, globally normalized weights in anchor+coverage order."""

    anchor, anchor_report = _class_balanced_group_weights(
        anchor_rows, group_weight=ANCHOR_GROUP_WEIGHT
    )
    coverage, coverage_report = _class_balanced_group_weights(
        coverage_rows, group_weight=COVERAGE_GROUP_WEIGHT
    )
    weights = np.concatenate((anchor, coverage)).astype(np.float64, copy=False)
    _require(abs(float(weights.sum()) - 1.0) <= 1e-12,
             "combined sample weights must sum to one")
    return weights, {"anchor": anchor_report, "coverage": coverage_report, "total_weight": 1.0}


def _rows_to_tensors(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    return legacy._rows_to_tensors(rows)


def _metrics(policy: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        return legacy._metrics(policy, rows)
    except legacy.RepairInputError as exc:
        raise GateCoverageRepairInputError(str(exc)) from exc


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _build_execution_config() -> Any:
    """Build the existing P3 dense24 config with opt-in canonical WAIT."""

    protocol = base._load_yaml_protocol()
    config = admission._build_config("dense24", protocol)
    _require(hasattr(config, "v4_2_rejected_release_hold_mode"),
             "execution config lacks v4_2_rejected_release_hold_mode")
    config = replace(config, v4_2_rejected_release_hold_mode="canonical_wait")
    _require(config.v4_2_rejected_release_hold_mode == "canonical_wait",
             "canonical_wait execution mode was not applied")
    _require(not bool(config.teacher_enabled), "fit config must disable Teacher")
    _require(str(config.scheduler_mode) == "execution", "fit config must use execution scheduler")
    return config


def _load_source_policy(
    source_checkpoint: str | Path,
    config: Any,
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any], str]:
    path = Path(source_checkpoint)
    _require(path.is_file(), f"source checkpoint not found: {path}")
    actual_sha = _sha256(path)
    _require(actual_sha == SOURCE_CHECKPOINT_SHA256,
             f"source checkpoint SHA256 mismatch: {actual_sha}")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        temperature=CALIBRATION_TEMPERATURE,
    )
    try:
        payload = load_checkpoint(path, policy, config, load_optimizer=False)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise GateCoverageRepairInputError(f"strict source checkpoint load failed: {exc}") from exc
    _require(getattr(policy.model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID,
             "source policy architecture id changed")
    _require(float(policy.model.gate_temperature.detach().cpu().item()) == CALIBRATION_TEMPERATURE,
             "source gate temperature changed")
    _require(len(policy.optimizer.state) == 0, "source optimizer state was restored")
    policy.model.eval()
    return policy, payload, actual_sha


def _freeze_to_adapter(policy: Any) -> tuple[list[torch.nn.Parameter], list[str]]:
    trainable: list[torch.nn.Parameter] = []
    names: list[str] = []
    for name, parameter in policy.model.named_parameters():
        enabled = name.startswith(ADAPTER_PREFIX)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
            names.append(name)
    _require(bool(trainable), "source policy has no gate_adapter parameters")
    _require(all(name.startswith(ADAPTER_PREFIX) for name in names),
             "trainable scope contains a non-gate_adapter parameter")
    return trainable, names


def _proposal_logits(policy: Any, rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    observations, _ = _rows_to_tensors(rows)
    _, proposal = legacy._gate_and_proposal_logits(policy, observations)
    return proposal


def _weighted_batch_loss(
    per_sample_loss: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    total_weight: torch.Tensor,
    population_size: int,
) -> torch.Tensor:
    """Keep global sample weights while preserving a stable per-batch scale."""

    _require(per_sample_loss.ndim == 1 and sample_weights.ndim == 1,
             "weighted loss inputs must be one-dimensional")
    _require(per_sample_loss.shape == sample_weights.shape and len(per_sample_loss) > 0,
             "weighted loss inputs must have the same non-empty shape")
    _require(int(population_size) >= len(per_sample_loss), "invalid weighted loss population size")
    _require(bool(torch.isfinite(per_sample_loss).all().item())
             and bool(torch.isfinite(sample_weights).all().item()),
             "weighted loss inputs must be finite")
    _require(bool(torch.isfinite(total_weight).item()) and float(total_weight.detach().cpu()) > 0.0,
             "total sample weight must be finite and positive")
    # The weights are normalized once over the complete training population.
    # Multiplying by N / batch_size keeps the optimizer-step scale stable for
    # the final short batch without renormalizing each batch's group mix.
    return (
        (per_sample_loss * sample_weights).sum()
        / total_weight
        * (float(population_size) / float(len(per_sample_loss)))
    )


def _output_paths(output: str | Path) -> tuple[Path, Path, Path, Path]:
    target = Path(output)
    if target.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        root = target.parent
        checkpoint = target
        _require(not checkpoint.exists(), f"refusing to overwrite checkpoint: {checkpoint}")
        report = root / "fit_report.json"
        progress = root / "progress.json"
        progress_jsonl = root / "progress.jsonl"
        for sidecar in (report, progress, progress_jsonl):
            _require(not sidecar.exists(), f"refusing to overwrite output: {sidecar}")
        root.mkdir(parents=True, exist_ok=True)
        return root, checkpoint, report, progress_jsonl
    _require(not target.exists(), f"refusing to overwrite output directory: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target, target / "p3_gate_coverage_repair_v2.pt", target / "fit_report.json", target / "progress.jsonl"


def _metadata(
    config: Any,
    *,
    source_checkpoint: Path,
    source_sha: str,
    source_payload: Mapping[str, Any],
    train_rows: int,
    dev_rows: int,
) -> dict[str, Any]:
    value = dict(checkpoint_metadata(config))
    execution = dict(vars(config))
    execution["v4_2_rejected_release_hold_mode"] = str(config.v4_2_rejected_release_hold_mode)
    source_meta = source_payload.get("metadata", {})
    source_stage = source_meta.get("stage") if isinstance(source_meta, Mapping) else None
    value.update({
        "stage": STAGE,
        "source_stage": source_stage,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "source_sha256": source_sha,
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": float(CALIBRATION_TEMPERATURE),
        "v4_2_rejected_release_hold_mode": "canonical_wait",
        "execution_config": _jsonable(execution),
        "current_execution_metadata": _jsonable(value),
        "training_called": True,
        "supervised_training_called": True,
        "teacher_called": False,
        "teacher_enabled": False,
        "ppo_called": False,
        "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
        "diagnostic_only": True,
        "heldout_consumed": False,
        "optimizer_restored": False,
        "optimizer_saved": False,
        "training_seed": TRAINING_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "label_smoothing": LABEL_SMOOTHING,
        "anchor_group_weight": ANCHOR_GROUP_WEIGHT,
        "coverage_group_weight": COVERAGE_GROUP_WEIGHT,
        "train_rows": int(train_rows),
        "dev_rows": int(dev_rows),
        "checkpoint_selection": "final_epoch_only",
        "dev_gradients": False,
        "early_stopping": False,
    })
    return value


def fit_gate_coverage_repair(
    source_checkpoint: str | Path,
    anchor_train: Sequence[str | Path] | str | Path,
    coverage_train: Sequence[str | Path] | str | Path,
    dev: Sequence[str | Path] | str | Path,
    output: str | Path,
    *,
    seed: int = TRAINING_SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    gradient_clip_norm: float = GRADIENT_CLIP_NORM,
    label_smoothing: float = LABEL_SMOOTHING,
) -> dict[str, Any]:
    """Fit the adapter and save only the final epoch."""

    _require(int(seed) == TRAINING_SEED, f"training seed is fixed at {TRAINING_SEED}")
    _require(int(epochs) == EPOCHS, f"epochs are fixed at {EPOCHS}")
    _require(int(batch_size) == BATCH_SIZE, f"batch size is fixed at {BATCH_SIZE}")
    _require(float(learning_rate) == LEARNING_RATE, f"learning rate is fixed at {LEARNING_RATE}")
    _require(float(gradient_clip_norm) == GRADIENT_CLIP_NORM,
             f"gradient clip norm is fixed at {GRADIENT_CLIP_NORM}")
    _require(float(label_smoothing) == LABEL_SMOOTHING,
             f"label smoothing is fixed at {LABEL_SMOOTHING}")

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    anchor_paths = _paths(anchor_train)
    coverage_paths = _paths(coverage_train)
    dev_paths = _paths(dev)
    anchor_rows = _load_rows(anchor_paths, split="train")
    coverage_rows = _load_rows(coverage_paths, split="train")
    dev_rows_by_path = [
        (Path(path), _load_rows([path], split="dev")) for path in dev_paths
    ]
    dev_rows = [row for _, rows in dev_rows_by_path for row in rows]
    _validate_splits(anchor_rows, coverage_rows, dev_rows)
    sample_weights, weight_report = build_sample_weights(anchor_rows, coverage_rows)
    train_rows = [*anchor_rows, *coverage_rows]
    train_x, train_y = _rows_to_tensors(train_rows)
    dev_x, _ = _rows_to_tensors(dev_rows)
    config = _build_execution_config()
    policy, source_payload, source_sha = _load_source_policy(source_checkpoint, config)
    model = policy.model
    model.eval()
    before_state = _state_cpu(model)
    before_proposal_train = _proposal_logits(policy, train_rows)
    before_proposal_dev = _proposal_logits(policy, dev_rows)
    metrics_before = {
        "anchor": _metrics(policy, anchor_rows),
        "coverage": _metrics(policy, coverage_rows),
        "dev": _metrics(policy, dev_rows),
        "dev_by_source": {
            str(path): _metrics(policy, rows) for path, rows in dev_rows_by_path
        },
    }

    adapter_parameters, trainable_names = _freeze_to_adapter(policy)
    policy.optimizer = torch.optim.Adam(adapter_parameters, lr=LEARNING_RATE)
    device = next(model.parameters()).device
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    weight_tensor = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    total_weight = weight_tensor.sum()
    _require(bool(torch.isfinite(weight_tensor).all().item()) and float(total_weight) > 0.0,
             "training sample weights are not finite and positive")
    generator = torch.Generator(device=device).manual_seed(TRAINING_SEED)
    root, checkpoint_path, report_path, progress_jsonl_path = _output_paths(output)
    progress_path = root / "progress.json"
    _write_json(progress_path, {
        "stage": STAGE,
        "status": "RUNNING",
        "epochs": EPOCHS,
        "completed_epochs": 0,
        "checkpoint_selection": "final_epoch_only",
    })
    progress_jsonl_path.write_text("", encoding="utf-8")

    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_y), generator=generator, device=device)
        weighted_loss_sum = 0.0
        for start in range(0, len(train_y), BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            policy.optimizer.zero_grad(set_to_none=True)
            gate_distribution, _ = model.distributions(train_x[indices])
            per_sample = F.cross_entropy(
                gate_distribution.logits,
                train_y[indices],
                reduction="none",
                label_smoothing=LABEL_SMOOTHING,
            )
            batch_weights = weight_tensor[indices]
            loss = _weighted_batch_loss(
                per_sample,
                batch_weights,
                total_weight=total_weight,
                population_size=len(train_y),
            )
            _require(bool(torch.isfinite(loss).item()), f"non-finite loss at epoch {epoch}")
            loss.backward()
            gradients = [parameter.grad for parameter in adapter_parameters if parameter.grad is not None]
            _require(bool(gradients) and all(bool(torch.isfinite(g).all().item()) for g in gradients),
                     f"non-finite adapter gradient at epoch {epoch}")
            norm = torch.nn.utils.clip_grad_norm_(adapter_parameters, GRADIENT_CLIP_NORM)
            _require(bool(torch.isfinite(norm).item()), f"non-finite gradient norm at epoch {epoch}")
            policy.optimizer.step()
            weighted_loss_sum += float((per_sample.detach() * batch_weights).sum().cpu().item())

        model.eval()
        epoch_metrics = {
            "anchor": _metrics(policy, anchor_rows),
            "coverage": _metrics(policy, coverage_rows),
            "dev": _metrics(policy, dev_rows),
        }
        record = {
            "epoch": epoch,
            "weighted_train_loss": weighted_loss_sum / float(total_weight.detach().cpu().item()),
            "metrics": epoch_metrics,
        }
        _require(np.isfinite(record["weighted_train_loss"]), f"non-finite epoch loss at epoch {epoch}")
        history.append(record)
        with progress_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        _write_json(progress_path, {
            "stage": STAGE,
            "status": "RUNNING",
            "epochs": EPOCHS,
            "completed_epochs": epoch,
            "latest": record,
            "checkpoint_selection": "final_epoch_only",
        })

    model.eval()
    after_state = _state_cpu(model)
    after_proposal_train = _proposal_logits(policy, train_rows)
    after_proposal_dev = _proposal_logits(policy, dev_rows)
    non_adapter_names = [name for name in before_state if not name.startswith(ADAPTER_PREFIX)]
    adapter_names = [name for name in before_state if name.startswith(ADAPTER_PREFIX)]
    non_adapter_state_bitwise_unchanged = all(
        torch.equal(before_state[name], after_state[name]) for name in non_adapter_names
    )
    adapter_state_changed = any(
        not torch.equal(before_state[name], after_state[name]) for name in adapter_names
    )
    proposal_logits_bitwise_unchanged = (
        torch.equal(before_proposal_train, after_proposal_train)
        and torch.equal(before_proposal_dev, after_proposal_dev)
    )
    _require(non_adapter_state_bitwise_unchanged, "non-adapter state changed during fit")
    _require(proposal_logits_bitwise_unchanged, "proposal logits changed during fit")
    _require(adapter_state_changed, "gate_adapter state did not change during fit")
    _require(all(not parameter.requires_grad for name, parameter in model.named_parameters()
                 if not name.startswith(ADAPTER_PREFIX)),
             "non-adapter parameters remain trainable")

    metrics_after = {
        "anchor": _metrics(policy, anchor_rows),
        "coverage": _metrics(policy, coverage_rows),
        "dev": _metrics(policy, dev_rows),
        "dev_by_source": {
            str(path): _metrics(policy, rows) for path, rows in dev_rows_by_path
        },
    }
    metadata = _metadata(
        config,
        source_checkpoint=Path(source_checkpoint),
        source_sha=source_sha,
        source_payload=source_payload,
        train_rows=len(train_rows),
        dev_rows=len(dev_rows),
    )
    metadata.update({
        "anchor_train_paths": [str(Path(path)) for path in anchor_paths],
        "coverage_train_paths": [str(Path(path)) for path in coverage_paths],
        "dev_paths": [str(Path(path)) for path in dev_paths],
        "anchor_train_sha256": [_sha256(path) for path in anchor_paths],
        "coverage_train_sha256": [_sha256(path) for path in coverage_paths],
        "dev_sha256": [_sha256(path) for path in dev_paths],
    })
    invariants = {
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": [name for name in before_state if name not in trainable_names],
        "non_adapter_state_bitwise_unchanged": bool(non_adapter_state_bitwise_unchanged),
        "proposal_logits_bitwise_unchanged": bool(proposal_logits_bitwise_unchanged),
        "adapter_state_changed": bool(adapter_state_changed),
        "source_optimizer_state_copied": False,
        "optimizer_saved": False,
        "dev_gradients": False,
        "early_stopping": False,
        "checkpoint_selection": "final_epoch_only",
    }
    report_payload = {
        "stage": STAGE,
        "diagnostic_only": True,
        "PPO_ALLOWED": False,
        "heldout_consumed": False,
        "source_checkpoint": str(Path(source_checkpoint)),
        "source_checkpoint_sha256": source_sha,
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": float(CALIBRATION_TEMPERATURE),
        "v4_2_rejected_release_hold_mode": "canonical_wait",
        "coverage_label_semantics": "state_only_gate_feasibility_over_all_26_candidates; not_neural_K3_success_guarantee",
        "training_seed": TRAINING_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "label_smoothing": LABEL_SMOOTHING,
        "group_weights": weight_report,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "history": history,
        "invariants": invariants,
        "metadata": metadata,
    }
    payload = {"metadata": metadata, "model": model.state_dict(), "report": report_payload}
    torch.save(payload, checkpoint_path)

    restored = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        temperature=CALIBRATION_TEMPERATURE,
    )
    try:
        load_checkpoint(checkpoint_path, restored, config, load_optimizer=False)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise GateCoverageRepairInputError(f"saved checkpoint strict load failed: {exc}") from exc
    restored_state = _state_cpu(restored.model)
    checkpoint_roundtrip_model_exact = _state_equal(after_state, restored_state)
    _require(checkpoint_roundtrip_model_exact, "saved checkpoint model roundtrip mismatch")
    report_payload["checkpoint_roundtrip_model_exact"] = True
    report_payload["checkpoint_sha256"] = _sha256(checkpoint_path)
    report_payload["checkpoint"] = str(checkpoint_path)
    _write_json(report_path, report_payload)
    _write_json(progress_path, {
        "stage": STAGE,
        "status": "COMPLETED",
        "epochs": EPOCHS,
        "completed_epochs": EPOCHS,
        "checkpoint_selection": "final_epoch_only",
        "checkpoint": str(checkpoint_path),
    })
    return {
        "status": "COMPLETED",
        "stage": STAGE,
        "checkpoint": str(checkpoint_path),
        "fit_report": str(report_path),
        "progress": str(progress_path),
        "progress_jsonl": str(progress_jsonl_path),
        "source_checkpoint_sha256": source_sha,
        "checkpoint_sha256": report_payload["checkpoint_sha256"],
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "group_weights": weight_report,
        "invariants": invariants,
        "checkpoint_roundtrip_model_exact": True,
    }


train_gate_coverage_repair = fit_gate_coverage_repair
train_gate_coverage_repair_v2 = fit_gate_coverage_repair
_build_sample_weights = build_sample_weights
RepairInputError = GateCoverageRepairInputError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-train", nargs="+", required=True)
    parser.add_argument("--coverage-train", nargs="+", required=True)
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = fit_gate_coverage_repair(
        source_checkpoint=args.source_checkpoint,
        anchor_train=args.anchor_train,
        coverage_train=args.coverage_train,
        dev=args.dev,
        output=args.output,
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
