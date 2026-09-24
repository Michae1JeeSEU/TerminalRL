"""Bounded supervised gate repair for the random-SOC curriculum stage.

This module deliberately has a small surface.  It loads the frozen dense24
diagnostic policy through the diagnostic loader, trains only its residual gate
adapter on labelled actor rows, and writes a separate, non-promotable
checkpoint.  It does not call an environment, a Teacher, or PPO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
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
from terminal_marl.core.checkpoints import checkpoint_metadata
from terminal_marl.training import diagnose_trained_dense30_v2 as diagnostic


STAGE = "RANDOM_SOC_GATE_REPAIR_V1"
OBSERVATION_DIM = 88
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"
ADAPTER_PREFIX = "gate_adapter."
SOURCE_CHECKPOINT_SHA256 = diagnostic.TRAINED_CHECKPOINT_SHA256
TRAINING_SEED = 846401
SEED = TRAINING_SEED
EPOCHS = 35
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
GRADIENT_CLIP_NORM = 1.0


class RepairInputError(ValueError):
    """Raised when an input or checkpoint violates the repair contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairInputError(message)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert small report values to deterministic JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_label_file(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    _require(source.is_file(), f"label file not found: {source}")
    if source.suffix.lower() == ".json":
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RepairInputError(f"invalid JSON label file: {source}") from exc
        if isinstance(value, Mapping):
            value = value.get("rows")
        _require(isinstance(value, list), f"JSON labels must be a list: {source}")
        rows = value
    else:
        rows = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RepairInputError(
                        f"invalid JSONL at {source}:{line_number}"
                    ) from exc
                rows.append(value)
    _require(bool(rows), f"label file is empty: {source}")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"label row is not an object: {source}:{index + 1}")
        result.append(dict(row))
    return result


def _normalize_rows(rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
    """Validate rows and retain only actor-mask rows used by the trainer."""

    _require(split in {"train", "dev"}, f"unknown split: {split}")
    normalized: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    all_seeds: set[int] = set()
    for index, source in enumerate(rows):
        _require(isinstance(source, Mapping), f"{split} row {index} is not a mapping")
        row = dict(source)
        actor_mask = row.get("actor_mask")
        _require(isinstance(actor_mask, (bool, np.bool_)),
                 f"{split} row {index} actor_mask must be boolean")
        seed = row.get("seed")
        _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool),
                 f"{split} row {index} seed is required")
        all_seeds.add(int(seed))
        if not bool(actor_mask):
            continue

        _require(row.get("observation_schema") == OBSERVATION_SCHEMA,
                 f"{split} row {index} observation_schema does not match {OBSERVATION_SCHEMA}")
        _require(row.get("action_schema") == ACTION_SCHEMA,
                 f"{split} row {index} action_schema does not match {ACTION_SCHEMA}")

        values = row.get("observation_v2_1")
        _require(isinstance(values, (list, tuple, np.ndarray)) and len(values) == OBSERVATION_DIM,
                 f"{split} row {index} must contain an 88D observation")
        try:
            observation = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RepairInputError(f"{split} row {index} observation is not numeric") from exc
        _require(observation.shape == (OBSERVATION_DIM,) and bool(np.isfinite(observation).all()),
                 f"{split} row {index} observation is not finite 88D")

        gate = row.get("teacher_gate")
        _require(isinstance(gate, (int, np.integer)) and not isinstance(gate, bool),
                 f"{split} row {index} teacher_gate is required")
        _require(int(gate) in (0, 1), f"{split} row {index} teacher_gate must be 0 or 1")

        event_id = row.get("decision_event_id")
        _require(isinstance(event_id, str) and bool(event_id),
                 f"{split} row {index} decision_event_id is required")
        _require(event_id not in event_ids,
                 f"{split} decision_event_id is not unique: {event_id}")
        event_ids.add(event_id)
        row["observation_v2_1"] = observation.astype(np.float32).tolist()
        row["teacher_gate"] = int(gate)
        row["seed"] = int(seed)
        row["actor_mask"] = True
        normalized.append(row)

    _require(bool(normalized), f"{split} has no actor_mask=true rows")
    # Keep the seed set on the returned collection for the caller without
    # changing row schemas.  The caller recomputes it from normalized rows.
    return normalized


def _load_rows(paths: Sequence[str | Path], *, split: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in paths:
        values.extend(_read_label_file(path))
    return _normalize_rows(values, split=split)


def _validate_splits(train_rows: Sequence[Mapping[str, Any]], dev_rows: Sequence[Mapping[str, Any]]) -> None:
    train_ids = {str(row["decision_event_id"]) for row in train_rows}
    dev_ids = {str(row["decision_event_id"]) for row in dev_rows}
    overlap = sorted(train_ids & dev_ids)
    _require(not overlap, f"train/dev decision_event_id overlap: {overlap[:5]}")
    train_seeds = {int(row["seed"]) for row in train_rows}
    dev_seeds = {int(row["seed"]) for row in dev_rows}
    overlap_seeds = sorted(train_seeds & dev_seeds)
    _require(not overlap_seeds, f"train/dev seed overlap: {overlap_seeds}")
    labels = {int(row["teacher_gate"]) for row in train_rows}
    _require(labels == {0, 1}, "training labels must contain both teacher_gate classes")


def _rows_to_tensors(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(
        np.asarray([row["observation_v2_1"] for row in rows], dtype=np.float32),
        dtype=torch.float32,
    )
    labels = torch.as_tensor(
        np.asarray([int(row["teacher_gate"]) for row in rows], dtype=np.int64),
        dtype=torch.long,
    )
    _require(tuple(observations.shape) == (len(rows), OBSERVATION_DIM), "invalid observation tensor shape")
    _require(bool(torch.isfinite(observations).all().item()), "observation tensor is not finite")
    return observations, labels


def _gate_and_proposal_logits(policy: Any, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model = policy.model
    device = next(model.parameters()).device
    with torch.no_grad():
        gate, proposal = model.distributions(observations.to(device))
        return gate.logits.detach().cpu(), proposal.logits.detach().cpu()


def _metrics(policy: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observations, labels = _rows_to_tensors(rows)
    gate_logits, _ = _gate_and_proposal_logits(policy, observations)
    predictions = gate_logits.argmax(dim=1)
    probabilities = gate_logits.softmax(dim=1)
    confusion = {
        "WAIT": {
            "WAIT": int(((labels == 0) & (predictions == 0)).sum().item()),
            "RELEASE": int(((labels == 0) & (predictions == 1)).sum().item()),
        },
        "RELEASE": {
            "WAIT": int(((labels == 1) & (predictions == 0)).sum().item()),
            "RELEASE": int(((labels == 1) & (predictions == 1)).sum().item()),
        },
    }
    per_class: dict[str, float | None] = {}
    recalls: dict[str, float | None] = {}
    for target, name in ((0, "WAIT"), (1, "RELEASE")):
        true_positive = float(((labels == target) & (predictions == target)).sum().item())
        actual = float((labels == target).sum().item())
        predicted = float((predictions == target).sum().item())
        recall = None if actual == 0.0 else true_positive / actual
        precision = None if predicted == 0.0 else true_positive / predicted
        if recall is None or precision is None or recall + precision == 0.0:
            f1 = None if recall is None else 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        recalls[name] = recall
        per_class[name] = f1
    present = [name for name, value in recalls.items() if value is not None]
    f1_values = [per_class[name] for name in present if per_class[name] is not None]
    macro_f1 = None if not f1_values else float(sum(f1_values) / len(f1_values))
    nll = float(F.nll_loss(gate_logits, labels, reduction="mean").item())
    values = [nll, macro_f1, recalls["WAIT"], recalls["RELEASE"]]
    _require(all(value is None or np.isfinite(value) for value in values), "non-finite classification metric")
    return {
        "rows": len(rows),
        "classes_present": [0 if name == "WAIT" else 1 for name in present],
        "class_names_present": present,
        "confusion": confusion,
        "WAIT_recall": recalls["WAIT"],
        "RELEASE_recall": recalls["RELEASE"],
        "per_class_F1": per_class,
        "macro_F1": macro_f1,
        "NLL": nll,
    }


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _resolve_output(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        _require(not target.exists(), f"refusing to overwrite checkpoint: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        for sidecar in (target.parent / "progress.json", target.parent / "progress.jsonl",
                        target.parent / "report.json"):
            _require(not sidecar.exists(), f"refusing to overwrite existing output: {sidecar}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "random_soc_gate_repair_v1.pt"
    _require(not checkpoint.exists(), f"refusing to overwrite checkpoint: {checkpoint}")
    for sidecar in (target / "progress.json", target / "progress.jsonl", target / "report.json"):
        _require(not sidecar.exists(), f"refusing to overwrite existing output: {sidecar}")
    return checkpoint


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_policy(checkpoint: str | Path):
    """Load the frozen source exactly through the dense30 diagnostic entrypoint."""

    source_path = Path(checkpoint)
    _require(source_path.is_file(), f"source checkpoint not found: {source_path}")
    config = diagnostic.build_scenario_config("dense24")
    loaded = diagnostic.load_trained_checkpoint(source_path, config)
    _require(isinstance(loaded, tuple) and len(loaded) == 2,
             "diagnostic source loader must return (policy, info)")
    policy, source_info = loaded
    _require(isinstance(policy, TemperatureCalibratedGateResidualAdapterStudentPPOV42),
             "source policy architecture is not the calibrated adapter")
    _require(int(policy.observation_dim) == OBSERVATION_DIM,
             "source policy observation dimension changed")
    model = policy.model
    _require(getattr(model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID,
             "source policy architecture id changed")
    temperature = getattr(model, "gate_temperature", None)
    _require(isinstance(temperature, torch.Tensor) and temperature.numel() == 1,
             "source policy temperature state is missing")
    _require(float(temperature.detach().cpu().item()) == float(CALIBRATION_TEMPERATURE),
             "source policy temperature changed")
    state = model.state_dict()
    _require(any(name.startswith(ADAPTER_PREFIX) for name in state),
             "source policy gate_adapter state is missing")
    actual_sha = _sha256(source_path)
    if isinstance(source_info, Mapping) and source_info.get("checkpoint_sha256") is not None:
        _require(str(source_info["checkpoint_sha256"]) == actual_sha,
                 "source loader hash does not match checkpoint bytes")
    _require(actual_sha == SOURCE_CHECKPOINT_SHA256,
             "source checkpoint SHA256 does not match the frozen dense30 source")
    return policy, config, actual_sha


def _trainable_adapter(policy: Any) -> tuple[list[torch.nn.Parameter], list[str]]:
    parameters: list[torch.nn.Parameter] = []
    names: list[str] = []
    for name, parameter in policy.model.named_parameters():
        trainable = name.startswith(ADAPTER_PREFIX)
        parameter.requires_grad_(trainable)
        if trainable:
            parameters.append(parameter)
            names.append(name)
    _require(bool(parameters), "source policy has no trainable gate_adapter parameters")
    _require(all(name.startswith(ADAPTER_PREFIX) for name in names),
             "trainable parameter scope is not gate_adapter only")
    return parameters, names


def _proposal_equal(policy: Any, rows: Sequence[Mapping[str, Any]], before: torch.Tensor) -> bool:
    observations, _ = _rows_to_tensors(rows)
    _, after = _gate_and_proposal_logits(policy, observations)
    return torch.equal(before, after)


def _metadata(config: Any, source_sha: str, policy: Any, *, train_rows: int, dev_rows: int) -> dict[str, Any]:
    current = checkpoint_metadata(config)
    _require(current.get("teacher_enabled") is False, "dense24 execution config must disable Teacher")
    model = policy.model
    temperature = float(model.gate_temperature.detach().cpu().item())
    _require(getattr(model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID,
             "source architecture id is not the known calibrated architecture")
    _require(temperature == float(CALIBRATION_TEMPERATURE), "source temperature is not fixed at 2.5")
    value = dict(current)
    value.update({
        "stage": STAGE,
        "source_checkpoint_sha256": source_sha,
        "source_sha256": source_sha,
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": temperature,
        "teacher_enabled": False,
        "training_called": True,
        "supervised_training_called": True,
        "teacher_called": False,
        "ppo_called": False,
        "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
        "optimizer_restored": False,
        "optimizer_saved": False,
        "training_seed": TRAINING_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "checkpoint_selection": "final_epoch_only",
        "dev_gradients": False,
        "early_stopping": False,
        "train_rows": int(train_rows),
        "dev_rows": int(dev_rows),
        "execution_config": diagnostic._config_payload(config),
        "current_execution_metadata": dict(current),
    })
    return value


def train_gate_repair(
    checkpoint: str | Path,
    train_labels: Sequence[str | Path],
    dev_labels: Sequence[str | Path],
    output: str | Path,
    *,
    seed: int = TRAINING_SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    gradient_clip_norm: float = GRADIENT_CLIP_NORM,
) -> dict[str, Any]:
    """Train and save the final epoch of the gate adapter repair."""

    _require(int(seed) == TRAINING_SEED, f"training seed is fixed at {TRAINING_SEED}")
    _require(int(epochs) == EPOCHS, f"epochs are fixed at {EPOCHS}")
    _require(int(batch_size) == BATCH_SIZE, f"batch size is fixed at {BATCH_SIZE}")
    _require(float(learning_rate) == LEARNING_RATE, f"learning rate is fixed at {LEARNING_RATE}")
    _require(float(gradient_clip_norm) == GRADIENT_CLIP_NORM,
             f"gradient clip norm is fixed at {GRADIENT_CLIP_NORM}")
    _require(bool(train_labels) and bool(dev_labels), "train and dev label paths are required")

    def _paths(value: Sequence[str | Path] | str | Path) -> list[str | Path]:
        if isinstance(value, (str, Path)):
            return [value]
        return list(value)

    train_label_paths = _paths(train_labels)
    dev_label_paths = _paths(dev_labels)
    _require(bool(train_label_paths) and bool(dev_label_paths), "train and dev label paths are required")

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    train_rows = _load_rows(train_label_paths, split="train")
    dev_rows = _load_rows(dev_label_paths, split="dev")
    _validate_splits(train_rows, dev_rows)
    train_x, train_y = _rows_to_tensors(train_rows)

    policy, config, source_sha = _source_policy(checkpoint)
    model = policy.model
    model.eval()
    before_state = _state_cpu(model)
    before_proposal_train = _gate_and_proposal_logits(policy, train_x)[1]
    dev_x, _ = _rows_to_tensors(dev_rows)
    before_proposal_dev = _gate_and_proposal_logits(policy, dev_x)[1]
    metrics_before = {"train": _metrics(policy, train_rows), "dev": _metrics(policy, dev_rows)}

    adapter_parameters, trainable_names = _trainable_adapter(policy)
    optimizer = torch.optim.Adam(adapter_parameters, lr=LEARNING_RATE)
    # Replace the source container's optimizer with the fresh adapter-only
    # optimizer.  Its state is intentionally not serialized.
    policy.optimizer = optimizer
    model.train()
    counts = torch.bincount(train_y, minlength=2).to(dtype=torch.float32)
    _require(bool(torch.all(counts > 0).item()), "training labels must contain both classes")
    weights = len(train_y) / (2.0 * counts)
    device = next(model.parameters()).device
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    weights = weights.to(device)
    generator = torch.Generator(device=device).manual_seed(TRAINING_SEED)

    output_path = _resolve_output(output)
    progress_path = output_path.parent / "progress.json"
    progress_lines_path = output_path.parent / "progress.jsonl"
    history: list[dict[str, Any]] = []
    _write_json(progress_path, {
        "stage": STAGE, "status": "RUNNING", "epochs": EPOCHS,
        "completed_epochs": 0, "history": [],
    })
    progress_lines_path.write_text("", encoding="utf-8")

    for epoch in range(1, EPOCHS + 1):
        permutation = torch.randperm(len(train_y), generator=generator, device=device)
        loss_sum = 0.0
        for start in range(0, len(train_y), BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            gate_distribution, _ = model.distributions(train_x[indices])
            loss = F.cross_entropy(gate_distribution.logits, train_y[indices], weight=weights)
            _require(bool(torch.isfinite(loss).item()), f"non-finite training loss at epoch {epoch}")
            loss.backward()
            gradients = [parameter.grad for parameter in adapter_parameters if parameter.grad is not None]
            _require(bool(gradients) and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients),
                     f"non-finite adapter gradient at epoch {epoch}")
            norm = torch.nn.utils.clip_grad_norm_(adapter_parameters, GRADIENT_CLIP_NORM)
            _require(bool(torch.isfinite(norm).item()), f"non-finite gradient norm at epoch {epoch}")
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item()) * len(indices)
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_y), "loss": loss_sum / len(train_y)}
        _require(np.isfinite(record["train_loss"]), f"non-finite epoch loss at epoch {epoch}")
        history.append(record)
        with progress_lines_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        _write_json(progress_path, {
            "stage": STAGE, "status": "RUNNING", "epochs": EPOCHS,
            "completed_epochs": epoch, "latest": record, "history": history,
        })

    model.eval()
    after_state = _state_cpu(model)
    metrics_after = {"train": _metrics(policy, train_rows), "dev": _metrics(policy, dev_rows)}
    after_proposal_train = _gate_and_proposal_logits(policy, train_x.cpu())[1]
    after_proposal_dev = _gate_and_proposal_logits(policy, dev_x.cpu())[1]
    non_adapter_names = [name for name in before_state if not name.startswith(ADAPTER_PREFIX)]
    non_adapter_state_bitwise_unchanged = all(
        torch.equal(before_state[name], after_state[name]) for name in non_adapter_names
    )
    adapter_names = [name for name in before_state if name.startswith(ADAPTER_PREFIX)]
    adapter_state_changed = any(
        not torch.equal(before_state[name], after_state[name]) for name in adapter_names
    )
    proposal_logits_bitwise_unchanged = (
        torch.equal(before_proposal_train, after_proposal_train)
        and torch.equal(before_proposal_dev, after_proposal_dev)
    )
    _require(non_adapter_state_bitwise_unchanged, "non-adapter state changed during repair")
    _require(proposal_logits_bitwise_unchanged, "proposal logits changed during repair")
    _require(adapter_state_changed, "gate_adapter state did not change during repair")
    _require(all(not parameter.requires_grad for name, parameter in model.named_parameters()
                 if not name.startswith(ADAPTER_PREFIX)),
             "non-adapter parameters are still trainable")

    metadata = _metadata(config, source_sha, policy, train_rows=len(train_rows), dev_rows=len(dev_rows))
    metadata.update({
        "train_label_paths": [str(Path(path)) for path in train_label_paths],
        "dev_label_paths": [str(Path(path)) for path in dev_label_paths],
        "train_label_sha256": [_sha256(path) for path in train_label_paths],
        "dev_label_sha256": [_sha256(path) for path in dev_label_paths],
    })
    payload = {
        "metadata": metadata,
        "model": model.state_dict(),
        "report": {
            "stage": STAGE,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "invariants": {
                "trainable_parameter_names": trainable_names,
                "non_adapter_state_bitwise_unchanged": non_adapter_state_bitwise_unchanged,
                "proposal_logits_bitwise_unchanged": proposal_logits_bitwise_unchanged,
                "adapter_state_changed": adapter_state_changed,
                "dev_gradients": False,
                "early_stopping": False,
                "checkpoint_selection": "final_epoch_only",
            },
            "history": history,
        },
    }
    torch.save(payload, output_path)
    _write_json(progress_path, {
        "stage": STAGE, "status": "COMPLETED", "epochs": EPOCHS,
        "completed_epochs": EPOCHS, "latest": history[-1], "history": history,
        "checkpoint": str(output_path),
    })
    report_path = output_path.parent / "report.json"
    result = {
        "status": "COMPLETED",
        "stage": STAGE,
        "checkpoint": str(output_path),
        "progress": str(progress_path),
        "progress_jsonl": str(progress_lines_path),
        "source_checkpoint_sha256": source_sha,
        "checkpoint_sha256": _sha256(output_path),
        "train_label_paths": [str(Path(path)) for path in train_label_paths],
        "dev_label_paths": [str(Path(path)) for path in dev_label_paths],
        "train_label_sha256": [_sha256(path) for path in train_label_paths],
        "dev_label_sha256": [_sha256(path) for path in dev_label_paths],
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "invariants": payload["report"]["invariants"],
        "metadata": metadata,
    }
    _write_json(report_path, result)
    return result


def load_repaired_checkpoint(
    path: str | Path,
    config: Any,
    expected_sha256: str,
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    """Strictly load this repair stage without restoring any optimizer."""

    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"repaired checkpoint not found: {checkpoint}")
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64,
             "expected repaired checkpoint SHA256 is required")
    _require(_sha256(checkpoint) == expected_sha256,
             "repaired checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "repaired checkpoint payload must be a mapping")
    _require("optimizer" not in payload, "repaired checkpoint must not contain optimizer state")
    metadata = payload.get("metadata")
    _require(isinstance(metadata, Mapping), "repaired checkpoint metadata is missing")
    current = checkpoint_metadata(config)
    _require(current.get("teacher_enabled") is False, "current execution config must disable Teacher")
    for key, value in current.items():
        _require(metadata.get(key) == value, f"current metadata mismatch: {key}")
    _require(metadata.get("current_execution_metadata") == current,
             "current execution metadata binding changed")
    required = {
        "stage": STAGE,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_sha256": SOURCE_CHECKPOINT_SHA256,
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": float(CALIBRATION_TEMPERATURE),
        "teacher_enabled": False,
        "training_called": True,
        "supervised_training_called": True,
        "teacher_called": False,
        "ppo_called": False,
        "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
        "optimizer_restored": False,
        "optimizer_saved": False,
    }
    for key, value in required.items():
        _require(metadata.get(key) == value, f"repair metadata mismatch: {key}")
    model_state = payload.get("model")
    _require(isinstance(model_state, Mapping), "repaired checkpoint model state is missing")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM, temperature=CALIBRATION_TEMPERATURE
    )
    _require(getattr(policy.model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID,
             "known calibrated architecture is unavailable")
    policy.model.load_state_dict(model_state, strict=True)
    temperature = policy.model.gate_temperature.detach().cpu().item()
    _require(float(temperature) == float(CALIBRATION_TEMPERATURE),
             "repaired checkpoint temperature state changed")
    policy.model.eval()
    return policy, dict(payload)


_load_repaired_checkpoint = load_repaired_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-labels", nargs="+", required=True)
    parser.add_argument("--dev-labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = train_gate_repair(
        args.checkpoint,
        args.train_labels,
        args.dev_labels,
        args.output,
    )
    print(json.dumps(_jsonable(result), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
