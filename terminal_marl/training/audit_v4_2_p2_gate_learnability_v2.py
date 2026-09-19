"""Offline P2 gate learnability audit for the frozen V5 support artifact.

This module is an offline diagnostic only.  It validates the final V5 support
artifact and the calibrated qualification candidate, scores the candidate,
and trains two small diagnostic probes on ``train_support`` only.  It never
resets an environment, calls the Teacher, tunes a threshold, or authorizes a
policy/checkpoint for promotion.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from terminal_marl.algo.joint_ppo_v4_2 import HierarchicalActorCentralCriticV42
from terminal_marl.core.checkpoints import checkpoint_metadata
from terminal_marl.training import audit_v4_2_p2_gate_learnability_v1 as v1
from terminal_marl.training import qualify_v4_2_p2_gate_student_v1 as qualification
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_learnability_audit_v2.yaml")
PROTOCOL_SHA = "c2cbc68f303e042acc0582f2e81ce20f54e6c9856a2790c5a2355b41c13c4486"

SUPPORT_MANIFEST_SHA = "100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960"
SUPPORT_TRAIN_SHA = "40002db70c2e0d5d181b53b2f23290c65954f6bbe6282be1066b6f558e6153d4"
SUPPORT_DEV_SHA = "da106e0d862c8e435748514abb10c4e0e5c500fb3bbd73f03cbfa6aa186a261b"
CANDIDATE_CHECKPOINT_SHA = "61e03e02e3fb204d2db4e814097543905f51e1a3772626fdac1d7bc02ab8ca6b"
CANDIDATE_ARCHITECTURE_ID = "v4_2_hierarchical_gate_temperature_calibrated_residual_adapter_v3"
CANDIDATE_TEMPERATURE = 2.5
D1_DATASET_SHA = "270e54ef6ed0c36f535e2a0f25cef7cf9fdd6c91016ee295e7e336fb800cacb0"
D1_MANIFEST_SHA = "a6319c291e5c91e155fd1c7fb20ae95dcdd4973733abfe83fc10c5e6006f472c"

# Aliases retained for small audit callers and parity with the V1 runner.
SUPPORT_MANIFEST_SHA256 = SUPPORT_MANIFEST_SHA
V5_MANIFEST_SHA = SUPPORT_MANIFEST_SHA
CANDIDATE_SHA = CANDIDATE_CHECKPOINT_SHA
CANDIDATE_CHECKPOINT_SHA256 = CANDIDATE_CHECKPOINT_SHA
D1_DATASET_SHA256 = D1_DATASET_SHA
D1_MANIFEST_SHA256 = D1_MANIFEST_SHA

OBSERVATION_DIM = 88
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"
INTERFACES = ("o2m", "m2i", "i2f")
PHASE_INTERFACE = {"ring_outer": "o2m", "ring_middle": "m2i", "ring_inner": "i2f"}
FUTURE_QUALIFICATION_SEEDS = tuple(range(782001, 782007))
SUPPORT_TRAIN_SEEDS = tuple(range(780001, 780010)) + tuple(range(783001, 783019))
SUPPORT_DEV_SEEDS = (tuple(range(781001, 781007)) + tuple(range(784001, 784010))
                     + tuple(range(785001, 785004)))
PROBE_SEEDS = (786101, 786102, 786103)
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
GRADIENT_CLIP_NORM = 1.0
PROBABILITY_THRESHOLD = 0.5
THRESHOLDS = {
    "dev_macro_F1_min": 0.80,
    "dev_WAIT_recall_min": 0.70,
    "dev_RELEASE_recall_min": 0.80,
    "dev_brier_max": 0.15,
    "dev_binary_nll_max": 0.50,
    "interface_macro_F1_min": 0.70,
}


class AuditInputError(ValueError):
    """Raised when an immutable audit input violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditInputError(message)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_sha = _sha256


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    _require(source.is_file(), f"dataset missing: {source}")
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditInputError(f"invalid JSONL at {source}:{line_number}") from exc
            _require(isinstance(value, dict), f"row is not an object at {source}:{line_number}")
            rows.append(value)
    _require(bool(rows), f"dataset is empty: {source}")
    return rows


def _resolve_artifact_path(raw: str | Path, manifest_path: Path) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    fallback = manifest_path.parent / path.name
    return fallback if fallback.is_file() else path


def protocol() -> dict[str, Any]:
    """Read and verify the byte-frozen V2 learnability protocol."""
    _require(PROTOCOL.is_file(), f"learnability audit protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA,
                 "learnability audit V2 protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "learnability audit V2 protocol must be a mapping")
    _require(value.get("protocol") == "V4_2_P2_GATE_LEARNABILITY_AUDIT_V2",
             "learnability audit V2 protocol id changed")
    bindings = value.get("bindings", {})
    expected_bindings = {
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "support_train_sha256": SUPPORT_TRAIN_SHA,
        "support_dev_sha256": SUPPORT_DEV_SHA,
        "support_stage": "V4_2_P2_GATE_SUPPORT_COLLECTION_V5_DEV_SUPPLEMENT",
        "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA,
        "candidate_architecture_id": CANDIDATE_ARCHITECTURE_ID,
        "candidate_temperature": CANDIDATE_TEMPERATURE,
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_dimension": OBSERVATION_DIM,
        "action_schema": ACTION_SCHEMA,
        "future_qualification_seeds": list(FUTURE_QUALIFICATION_SEEDS),
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"audit V2 binding changed: {key}")

    support = value.get("support_requirements", {})
    expected_support = {
        "SUPPORT_COLLECTION": "PASS",
        "SUPPORT_FROZEN": True,
        "LEARNABILITY_AUDIT_ALLOWED": True,
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "train_rows": 3786,
        "dev_rows": 2415,
        "train_all_gates": True,
        "dev_all_gates": True,
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "future_qualification_markers_mutated": False,
    }
    for key, expected in expected_support.items():
        _require(support.get(key) == expected, f"audit V2 support requirement changed: {key}")

    training = value.get("training", {})
    _require(tuple(training.get("seeds", ())) == PROBE_SEEDS, "probe seeds changed")
    _require(training.get("epochs") == EPOCHS, "probe epochs changed")
    _require(training.get("batch_size") == BATCH_SIZE, "probe batch size changed")
    _require(float(training.get("learning_rate")) == LEARNING_RATE, "probe learning rate changed")
    _require(float(training.get("gradient_clip_norm")) == GRADIENT_CLIP_NORM,
             "gradient clip changed")
    _require(training.get("loss") == "unweighted_two_class_cross_entropy",
             "probe loss definition changed")
    _require(training.get("final_epoch_only") is True
             and training.get("early_stopping") is False
             and training.get("selection_split") == "none_final_epoch_only",
             "probe selection semantics changed")
    evaluation = value.get("evaluation", {})
    _require(float(evaluation.get("probability_threshold")) == PROBABILITY_THRESHOLD,
             "evaluation probability threshold changed")
    thresholds = value.get("thresholds", {}).get("fresh_probe", {})
    for key, expected in THRESHOLDS.items():
        _require(float(thresholds.get(key)) == expected, f"threshold changed: {key}")
    _require(thresholds.get("all_three_seeds_required") is True,
             "all-three-seed gate changed")
    return value


def _interface(row: Mapping[str, Any]) -> str | None:
    direct = row.get("interface")
    if direct in INTERFACES:
        return str(direct)
    return PHASE_INTERFACE.get(str(row.get("phase")))


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    require_support_fields: bool = True,
    expected_seeds: Sequence[int] | None = None,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Validate finite 88D rows, identities, split fields, and seed leakage."""
    if expected_count is not None:
        _require(len(rows) == expected_count,
                 f"{split} row count changed: expected {expected_count}, got {len(rows)}")
    normalized: list[dict[str, Any]] = []
    events: list[str] = []
    observed_seeds: set[int] = set()
    for index, source in enumerate(rows):
        _require(isinstance(source, Mapping), f"{split} row {index} is not a mapping")
        row = dict(source)
        values = row.get("observation_v2_1")
        _require(isinstance(values, (list, tuple)) and len(values) == OBSERVATION_DIM,
                 f"{split} row {index} does not have finite 88D observation")
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise AuditInputError(f"{split} row {index} observation is not numeric") from exc
        _require(array.shape == (OBSERVATION_DIM,) and bool(np.isfinite(array).all()),
                 f"{split} row {index} observation is not finite 88D")
        try:
            gate = int(row["teacher_gate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditInputError(f"{split} row {index} has invalid teacher_gate") from exc
        _require(gate in (0, 1), f"{split} row {index} teacher_gate is not binary")
        row["teacher_gate"] = gate
        event = str(row.get("decision_event_id", ""))
        _require(bool(event) and event != "None", f"{split} row {index} missing decision_event_id")
        events.append(event)
        row["decision_event_id"] = event
        if "seed" in row:
            try:
                row["seed"] = int(row["seed"])
            except (TypeError, ValueError) as exc:
                raise AuditInputError(f"{split} row {index} has invalid seed") from exc
            observed_seeds.add(row["seed"])
            _require(row["seed"] not in FUTURE_QUALIFICATION_SEEDS,
                     f"{split} row {index} overlaps future qualification seed")
        if "support_split" in row:
            _require(str(row["support_split"]) == split,
                     f"{split} row {index} has support_split={row['support_split']!r}")
        if require_support_fields:
            _require(bool(row.get("actor_mask", True)), f"{split} row {index} is not an actor row")
            _require(_interface(row) in INTERFACES, f"{split} row {index} has no gate interface")
            _require("scenario" in row and bool(str(row["scenario"])),
                     f"{split} row {index} has no scenario")
        normalized.append(row)
    _require(events and len(events) == len(set(events)),
             f"{split} decision_event_id values are not unique")
    if expected_seeds is not None:
        _require(observed_seeds == set(int(seed) for seed in expected_seeds),
                 f"{split} seed coverage changed")
    return normalized


def _validate_support_status(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("stage") == "V4_2_P2_GATE_SUPPORT_COLLECTION_V5_DEV_SUPPLEMENT",
             "support manifest stage is not V5 dev supplement")
    for key, expected in {
        "SUPPORT_COLLECTION": "PASS",
        "SUPPORT_FROZEN": True,
        "LEARNABILITY_AUDIT_ALLOWED": True,
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "future_qualification_markers_mutated": False,
    }.items():
        _require(manifest.get(key) == expected, f"support manifest {key} changed")
    if "checkpoint_promotable" in manifest:
        _require(manifest.get("checkpoint_promotable") is False,
                 "support manifest unlocked checkpoint promotion")
    _require(manifest.get("training_called") is False
             and manifest.get("optimizer_called") is False,
             "support manifest contains training")
    bindings = manifest.get("bindings", {})
    for key, expected in {
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_markers_mutated": False,
    }.items():
        if key in bindings:
            _require(bindings.get(key) == expected, f"support binding {key} changed")
    _require(manifest.get("student_authoritative") is True
             and manifest.get("teacher_label_only") is True
             and manifest.get("teacher_replaced_behavior") is False,
             "support authority provenance changed")
    metrics = manifest.get("metrics", {})
    gates = manifest.get("gates", {})
    _require(isinstance(metrics, Mapping) and isinstance(gates, Mapping),
             "support metrics/gates missing")
    expected_gate_keys = {
        "cardinality", "teacher_class_minimums", "teacher_class_proportions",
        "interface_minimums", "interface_class_minimums", "schema_88d_finite",
        "decision_event_id_unique", "seed_coverage",
        "all_planned_episodes_execution_complete",
        "all_episode_shield_actor_fraction_within_inherited_limit",
        "all_episode_gates", "cross_split_seed_disjoint", "cross_split_event_disjoint",
        "all",
    }
    for split, expected_rows in (("train", 3786), ("dev", 2415)):
        _require(int(metrics.get(split, {}).get("rows", -1)) == expected_rows,
                 f"support {split} row count changed")
        split_gates = gates.get(split, {})
        _require(set(split_gates) == expected_gate_keys,
                 f"support {split} gate shape changed")
        _require(all(value is True for value in split_gates.values()),
                 f"support {split} gate changed")


def _support_file_metadata(files: Mapping[str, Any], split: str) -> tuple[str, str | Path | None]:
    """Return a combined file hash/path from both V5 manifest layouts."""
    combined = files.get("combined", {})
    raw_path = combined.get(split) if isinstance(combined, Mapping) else None
    hashes = files.get("sha256", {})
    expected_hash: str | None = None
    if isinstance(hashes, Mapping):
        for key in (f"combined_{split}", split):
            if hashes.get(key):
                expected_hash = str(hashes[key])
                break
    if expected_hash is None and isinstance(combined, Mapping):
        nested_hashes = combined.get("sha256", {})
        if isinstance(nested_hashes, Mapping) and nested_hashes.get(split):
            expected_hash = str(nested_hashes[split])
    return expected_hash or "", raw_path


def validate_support_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str | None = SUPPORT_MANIFEST_SHA,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Path]]:
    """Validate V5 status, hashes, row schema, seed identity, and split disjointness."""
    path = Path(manifest_path)
    _require(path.is_file(), f"support manifest missing: {path}")
    if expected_sha256 is not None:
        _require(_sha256(path) == expected_sha256, "support V5 manifest SHA mismatch")
    manifest = _read_json(path)
    _validate_support_status(manifest)
    _require(manifest.get("candidate_checkpoint_sha256") == CANDIDATE_CHECKPOINT_SHA,
             "support candidate binding mismatch")
    _require(manifest.get("candidate_architecture_id") == CANDIDATE_ARCHITECTURE_ID,
             "support candidate architecture binding mismatch")
    _require(float(manifest.get("candidate_temperature")) == CANDIDATE_TEMPERATURE,
             "support candidate temperature binding mismatch")
    _require(manifest.get("formal_episode_plan_complete") is True
             and manifest.get("formal_marker_created") is True,
             "support collection is not formally complete")
    files = manifest.get("support_files")
    _require(isinstance(files, Mapping), "support_files missing from V5 manifest")
    rows: dict[str, list[dict[str, Any]]] = {}
    paths: dict[str, Path] = {}
    event_sets: dict[str, set[str]] = {}
    expected = {
        "train": (SUPPORT_TRAIN_SHA, SUPPORT_TRAIN_SEEDS, 3786),
        "dev": (SUPPORT_DEV_SHA, SUPPORT_DEV_SEEDS, 2415),
    }
    for split, (fixed_hash, fixed_seeds, fixed_count) in expected.items():
        metadata_hash, raw_path = _support_file_metadata(files, split)
        _require(raw_path is not None, f"support_files.combined.{split} missing")
        data_path = _resolve_artifact_path(str(raw_path), path)
        _require(data_path.is_file(), f"support file missing: {data_path}")
        _require(_sha256(data_path) == fixed_hash, f"support {split} file SHA mismatch")
        if metadata_hash:
            _require(metadata_hash == fixed_hash, f"support {split} metadata SHA mismatch")
        values = _validate_rows(_read_jsonl(data_path), split=split,
                                require_support_fields=True,
                                expected_seeds=fixed_seeds,
                                expected_count=fixed_count)
        rows[split] = values
        paths[split] = data_path
        event_sets[split] = {str(value["decision_event_id"]) for value in values}
    _require(event_sets["train"].isdisjoint(event_sets["dev"]),
             "train/dev decision_event_id values overlap")
    return manifest, rows, paths


def validate_d1_anchor(
    dataset_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_dataset_sha256: str | None = D1_DATASET_SHA,
    expected_manifest_sha256: str | None = D1_MANIFEST_SHA,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate immutable D1 metadata for report-only scoring."""
    dataset = Path(dataset_path)
    manifest_file = Path(manifest_path)
    _require(dataset.is_file(), f"D1 dataset missing: {dataset}")
    _require(manifest_file.is_file(), f"D1 manifest missing: {manifest_file}")
    if expected_dataset_sha256 is not None:
        _require(_sha256(dataset) == expected_dataset_sha256, "D1 dataset SHA mismatch")
    if expected_manifest_sha256 is not None:
        _require(_sha256(manifest_file) == expected_manifest_sha256, "D1 manifest SHA mismatch")
    manifest = _read_json(manifest_file)
    _require(manifest.get("dataset_sha256") == D1_DATASET_SHA,
             "D1 manifest dataset binding mismatch")
    _require(int(manifest.get("aggregate_rows", -1)) > 0, "D1 manifest has no aggregate rows")
    rows = _validate_rows(_read_jsonl(dataset), split="d1", require_support_fields=False)
    _require(len(rows) == int(manifest["aggregate_rows"]),
             "D1 row count does not match manifest")
    return manifest, rows


def _row_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, np.ndarray]:
    observations = np.asarray([row["observation_v2_1"] for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["teacher_gate"]) for row in rows], dtype=np.int64)
    _require(observations.shape == (len(rows), OBSERVATION_DIM), "row arrays are not 88D")
    _require(bool(np.isfinite(observations).all()), "row arrays are not finite")
    _require(np.isin(labels, (0, 1)).all(), "row labels are not binary")
    return torch.as_tensor(observations, dtype=torch.float32), labels


def _core_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return v1._core_metrics(labels, probabilities)


def gate_metrics(rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Return V1-compatible metrics and scenario/interface breakdowns."""
    return v1.gate_metrics(rows, probabilities)


score_rows = gate_metrics
binary_gate_metrics = gate_metrics


def _probabilities(model: nn.Module, rows: Sequence[Mapping[str, Any]], *, batch_size: int = 2048) -> np.ndarray:
    observations, _ = _row_arrays(rows)
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            distribution, _ = model.distributions(observations[start:start + batch_size])
            values.append(distribution.probs[:, 1].detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)


def _proposal_logits(model: nn.Module, rows: Sequence[Mapping[str, Any]], *, batch_size: int = 2048) -> np.ndarray:
    observations, _ = _row_arrays(rows)
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            _, proposal = model.distributions(observations[start:start + batch_size])
            values.append(proposal.logits.detach().cpu().numpy().copy())
    return np.concatenate(values, axis=0)


def evaluate_model(model: nn.Module, datasets: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {name: gate_metrics(rows, _probabilities(model, rows))
            for name, rows in datasets.items()}


def _set_trainable(model: nn.Module, trainable_prefixes: Sequence[str]) -> tuple[list[str], list[str]]:
    prefixes = tuple(str(value) for value in trainable_prefixes)
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        parameter.requires_grad_(enabled)
        (trainable if enabled else frozen).append(name)
    _require(bool(trainable), "probe has no trainable parameters")
    return trainable, frozen


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def train_gate_probe(
    model: nn.Module,
    train_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    trainable_prefixes: Sequence[str],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    gradient_clip_norm: float = GRADIENT_CLIP_NORM,
) -> dict[str, Any]:
    """Train a probe on train_support only and retain the final epoch."""
    _require(bool(train_rows), "probe train rows are empty")
    _require(epochs >= 1 and batch_size >= 1 and learning_rate > 0 and gradient_clip_norm > 0,
             "invalid probe training settings")
    observations, labels = _row_arrays(train_rows)
    targets = torch.as_tensor(labels, dtype=torch.long)
    _seed_everything(seed)
    trainable, frozen = _set_trainable(model, trainable_prefixes)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
    )
    losses: list[float] = []
    rng = np.random.default_rng(int(seed))
    model.train()
    for epoch in range(1, int(epochs) + 1):
        epoch_losses: list[float] = []
        permutation = rng.permutation(len(train_rows))
        for start in range(0, len(train_rows), int(batch_size)):
            indices = permutation[start:start + int(batch_size)]
            logits, _ = model.distributions(observations[indices])
            loss = F.cross_entropy(logits.logits, targets[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite offline gate probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(gradient_clip_norm), error_if_nonfinite=True,
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    model.eval()
    return {
        "seed": int(seed),
        "epochs": int(epochs),
        "final_epoch": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "gradient_clip_norm": float(gradient_clip_norm),
        "loss_final_epoch": float(losses[-1]),
        "loss_first_epoch": float(losses[0]),
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
        "optimizer": "Adam",
        "loss": "unweighted_two_class_cross_entropy",
        "early_stopping": False,
        "selection_split": "none_final_epoch_only",
    }


def _state_changed(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> list[str]:
    _require(set(before) == set(after), "model state keys changed during probe")
    return [name for name in before
            if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu())]


def _parameter_changed(before_model: nn.Module, after_model: nn.Module) -> list[str]:
    before = dict(before_model.named_parameters())
    after = dict(after_model.named_parameters())
    _require(set(before) == set(after), "model parameter keys changed during probe")
    return [name for name in before
            if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu())]


def _state_invariance(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    excluded_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    _require(set(before) == set(after), "model state keys changed during invariance check")
    excluded = tuple(str(prefix) for prefix in excluded_prefixes)
    compared = [name for name in before
                if not any(name == prefix or name.startswith(prefix + ".") for prefix in excluded)]
    changed = [name for name in compared
               if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu())]
    return {
        "bitwise_equal": not changed,
        "changed_names": changed,
        "excluded_prefixes": list(excluded),
        "compared_names": compared,
    }


def _component_state_invariance(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor], component: str,
) -> dict[str, Any]:
    names = [name for name in before if name.startswith(component)]
    changed = [name for name in names
               if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu())]
    return {"bitwise_equal": not changed, "changed_names": changed, "state_names": names}


def proposal_logits_invariance(
    before_model: nn.Module,
    after_model: nn.Module,
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    overall_equal = True
    maximum = 0.0
    for name, rows in datasets.items():
        before = _proposal_logits(before_model, rows)
        after = _proposal_logits(after_model, rows)
        _require(before.shape == after.shape, f"proposal logits shape changed for {name}")
        delta = np.abs(after.astype(np.float64) - before.astype(np.float64))
        equal = bool(np.array_equal(before, after))
        max_delta = float(delta.max()) if delta.size else 0.0
        result[name] = {"bitwise_equal": equal, "max_abs_delta": max_delta,
                        "samples": int(len(rows))}
        overall_equal = overall_equal and equal
        maximum = max(maximum, max_delta)
    result["all_bitwise_equal"] = bool(overall_equal)
    result["all_max_abs_delta"] = float(maximum)
    result["invariant"] = bool(overall_equal or maximum == 0.0)
    return result


def threshold_checks(dev_metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "dev_macro_F1": float(dev_metrics.get("macro_F1", float("nan"))) >= THRESHOLDS["dev_macro_F1_min"],
        "dev_WAIT_recall": float(dev_metrics.get("WAIT_recall", float("nan"))) >= THRESHOLDS["dev_WAIT_recall_min"],
        "dev_RELEASE_recall": float(dev_metrics.get("RELEASE_recall", float("nan"))) >= THRESHOLDS["dev_RELEASE_recall_min"],
        "dev_brier": float(dev_metrics.get("brier", float("nan"))) <= THRESHOLDS["dev_brier_max"],
        "dev_binary_nll": float(dev_metrics.get("binary_nll", float("nan"))) <= THRESHOLDS["dev_binary_nll_max"],
    }
    interface_metrics = dev_metrics.get("interface", {})
    interface_checks: dict[str, bool] = {}
    for interface in INTERFACES:
        metric = interface_metrics.get(interface, {})
        interface_checks[interface] = float(metric.get("macro_F1", float("nan"))) >= THRESHOLDS["interface_macro_F1_min"]
    checks["every_interface_macro_F1"] = bool(interface_checks) and all(interface_checks.values())
    checks["interface"] = interface_checks
    checks["all"] = bool(all(value for key, value in checks.items()
                              if key not in {"interface"}))
    return checks


def fresh_probe_thresholds(results: Mapping[int | str, Mapping[str, Any]]) -> dict[str, Any]:
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in PROBE_SEEDS:
        value = results.get(seed, results.get(str(seed)))
        per_seed[str(seed)] = threshold_checks(
            value.get("evaluation", {}).get("dev", {}) if value else {}
        )
    present = all(seed in results or str(seed) in results for seed in PROBE_SEEDS)
    return {
        "per_seed": per_seed,
        "all_seeds_present": present,
        "all_seeds_pass": bool(present and all(item["all"] for item in per_seed.values())),
        "pass": bool(present and all(item["all"] for item in per_seed.values())),
        "thresholds": dict(THRESHOLDS),
    }


passes_threshold = lambda metrics: bool(threshold_checks(metrics)["all"])
learnability_passes = fresh_probe_thresholds


def _save_probe_checkpoint(path: Path, model: nn.Module, *, family: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": checkpoint_metadata(p2_config(6), policy_kind="offline_gate_probe"),
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_saved": False,
        "stage": "V4_2_P2_GATE_LEARNABILITY_AUDIT_V2",
        "architecture_id": getattr(model, "architecture_id", model.__class__.__name__),
        "probe_family": family,
        "seed": int(seed),
        "diagnostic_only": True,
        "checkpoint_promotable": False,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False,
    }
    torch.save(payload, path)


def _load_candidate(path: str | Path):
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"candidate checkpoint missing: {checkpoint}")
    _require(_sha256(checkpoint) == CANDIDATE_CHECKPOINT_SHA,
             "candidate checkpoint SHA mismatch")
    policy, payload = qualification._load_calibrated_checkpoint(
        checkpoint, expected_sha256=CANDIDATE_CHECKPOINT_SHA
    )
    for key, expected in {
        "architecture_id": CANDIDATE_ARCHITECTURE_ID,
        "temperature": CANDIDATE_TEMPERATURE,
        "qualification_only": True,
        "checkpoint_promotable": False,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False,
    }.items():
        actual = payload.get(key)
        if isinstance(expected, float):
            _require(float(actual) == expected, f"candidate checkpoint {key} changed")
        else:
            _require(actual == expected, f"candidate checkpoint {key} changed")
    policy.model.eval()
    return policy, payload


def _probe_result(
    model: nn.Module,
    train_rows: Sequence[Mapping[str, Any]],
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    training: Mapping[str, Any],
    *,
    family: str,
    checkpoint_path: Path,
    baseline: Mapping[str, Any],
    before_model: nn.Module,
    before_state: Mapping[str, torch.Tensor],
    trainable_prefixes: Sequence[str],
) -> dict[str, Any]:
    evaluation = evaluate_model(model, datasets)
    changed_state = _state_changed(before_state, model.state_dict())
    changed_parameters = _parameter_changed(before_model, model)
    allowed = tuple(str(prefix) for prefix in trainable_prefixes)
    unexpected = [name for name in changed_parameters
                  if not any(name == prefix or name.startswith(prefix + ".") for prefix in allowed)]
    result: dict[str, Any] = {
        "family": family,
        "seed": int(training["seed"]),
        "training": dict(training),
        "evaluation": evaluation,
        "checkpoint": str(checkpoint_path),
        "checkpoint_promotable": False,
        "diagnostic_only": True,
        "changed_state_names": changed_state,
        "changed_parameter_names": changed_parameters,
        "parameter_isolation": bool(changed_parameters and not unexpected),
        "unexpected_changed_parameter_names": unexpected,
        "baseline_comparison": {
            name: {
                "macro_F1_delta": float(evaluation[name]["macro_F1"] - baseline[name]["macro_F1"]),
                "binary_nll_delta": float(evaluation[name]["binary_nll"] - baseline[name]["binary_nll"]),
            }
            for name in datasets
        },
        "proposal_logits_invariance": proposal_logits_invariance(before_model, model, datasets),
        "critic_state_invariance": _component_state_invariance(
            before_state, model.state_dict(), "critic_"
        ),
        "common_state_invariance": _state_invariance(
            before_state, model.state_dict(), excluded_prefixes=allowed
        ),
    }
    result["informative"] = bool(
        result["parameter_isolation"]
        and evaluation["train"]["binary_nll"] <= baseline["train"]["binary_nll"]
    )
    # ``train_rows`` is deliberately accepted to keep this result helper
    # compatible with the V1 audit callers; final evaluation remains exactly
    # once per dataset and never uses dev/D1 rows for gradients.
    del train_rows
    return result


def run(
    output: str | Path,
    support_manifest: str | Path | None = None,
    candidate_checkpoint: str | Path | None = None,
    d1_dataset: str | Path | None = None,
    d1_manifest: str | Path | None = None,
    *,
    v5_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Run the offline V2 audit and write one immutable diagnostic manifest."""
    protocol_value = protocol()
    del protocol_value
    support_path = Path(support_manifest or v5_manifest or "")
    _require(bool(str(support_path)), "support_manifest is required")
    _require(candidate_checkpoint is not None and d1_dataset is not None and d1_manifest is not None,
             "candidate_checkpoint, d1_dataset and d1_manifest are required")
    output_path = Path(output)
    _require(not output_path.exists(), f"audit output already exists: {output_path}")

    # All frozen inputs are validated before creating output state.
    support_meta, support_rows, support_paths = validate_support_manifest(support_path)
    d1_meta, d1_rows = validate_d1_anchor(d1_dataset, d1_manifest)
    candidate, candidate_payload = _load_candidate(candidate_checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)

    datasets: dict[str, Sequence[Mapping[str, Any]]] = {
        "train": support_rows["train"],
        "dev": support_rows["dev"],
        "d1": d1_rows,
    }
    baseline = evaluate_model(candidate.model, datasets)
    baseline_proposal_logits = {
        name: _proposal_logits(candidate.model, rows) for name, rows in datasets.items()
    }
    probe_root = output_path / "probes"
    probes: dict[str, dict[str, Any]] = {
        "candidate_gate_adapter_only": {},
        "fresh_actor_encoder_gate_head": {},
    }

    for seed in PROBE_SEEDS:
        # Candidate adapter diagnostic: the calibrated candidate is copied
        # exactly; only the residual adapter parameters may update.
        adapter_model = copy.deepcopy(candidate.model)
        adapter_before_model = copy.deepcopy(adapter_model)
        adapter_before_state = {
            name: value.detach().clone() for name, value in adapter_model.state_dict().items()
        }
        training = train_gate_probe(
            adapter_model, support_rows["train"], seed=seed,
            trainable_prefixes=("gate_adapter",), epochs=EPOCHS,
            batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
            gradient_clip_norm=GRADIENT_CLIP_NORM,
        )
        adapter_path = probe_root / f"non_promotable_probe_candidate_gate_adapter_only_{seed}.pt"
        _save_probe_checkpoint(adapter_path, adapter_model,
                               family="candidate_gate_adapter_only", seed=seed)
        adapter_result = _probe_result(
            adapter_model, support_rows["train"], datasets, training,
            family="candidate_gate_adapter_only", checkpoint_path=adapter_path,
            baseline=baseline, before_model=adapter_before_model,
            before_state=adapter_before_state, trainable_prefixes=("gate_adapter",),
        )
        adapter_result["proposal_logits_snapshot_sha256"] = {
            name: hashlib.sha256(np.asarray(_proposal_logits(adapter_model, rows), dtype=np.float32).tobytes()).hexdigest()
            for name, rows in datasets.items()
        }
        adapter_result["proposal_logits_baseline_snapshot_sha256"] = {
            name: hashlib.sha256(np.asarray(baseline_proposal_logits[name], dtype=np.float32).tobytes()).hexdigest()
            for name, rows in datasets.items()
        }
        probes["candidate_gate_adapter_only"][str(seed)] = adapter_result

        # Fresh probe is intentionally independent of the candidate.  Its
        # proposal head and critic remain frozen while actor/gate learn.
        _seed_everything(seed)
        fresh_model = HierarchicalActorCentralCriticV42(OBSERVATION_DIM)
        fresh_before_model = copy.deepcopy(fresh_model)
        fresh_before_state = {
            name: value.detach().clone() for name, value in fresh_model.state_dict().items()
        }
        training = train_gate_probe(
            fresh_model, support_rows["train"], seed=seed,
            trainable_prefixes=("actor_encoder", "gate_head"), epochs=EPOCHS,
            batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
            gradient_clip_norm=GRADIENT_CLIP_NORM,
        )
        fresh_path = probe_root / f"non_promotable_probe_fresh_actor_encoder_gate_head_{seed}.pt"
        _save_probe_checkpoint(fresh_path, fresh_model,
                               family="fresh_actor_encoder_gate_head", seed=seed)
        fresh_result = _probe_result(
            fresh_model, support_rows["train"], datasets, training,
            family="fresh_actor_encoder_gate_head", checkpoint_path=fresh_path,
            baseline=baseline, before_model=fresh_before_model,
            before_state=fresh_before_state,
            trainable_prefixes=("actor_encoder", "gate_head"),
        )
        probes["fresh_actor_encoder_gate_head"][str(seed)] = fresh_result

    fresh_threshold = fresh_probe_thresholds(probes["fresh_actor_encoder_gate_head"])
    adapter_results = probes["candidate_gate_adapter_only"]
    adapter_informative = bool(adapter_results) and all(
        bool(value.get("informative")) for value in adapter_results.values()
    )
    audit_pass = bool(fresh_threshold["pass"])
    result: dict[str, Any] = {
        "stage": "V4_2_P2_GATE_LEARNABILITY_AUDIT_V2",
        "AUDIT_COMPLETE": True,
        "audit_pass": audit_pass,
        "learnability_pass": audit_pass,
        "repair_preregistration_allowed": audit_pass,
        "candidate_gate_adapter_only_informative": adapter_informative,
        "candidate_gate_adapter_informative": adapter_informative,
        "support_manifest": str(support_path),
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "support_stage": support_meta.get("stage"),
        "support_files": {
            name: {"path": str(support_paths[name]), "sha256": _sha256(support_paths[name])}
            for name in ("train", "dev")
        },
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA,
        "candidate_architecture_id": CANDIDATE_ARCHITECTURE_ID,
        "candidate_temperature": CANDIDATE_TEMPERATURE,
        "candidate_payload": {
            "stage": candidate_payload.get("stage"),
            "architecture_id": candidate_payload.get("architecture_id"),
            "temperature": candidate_payload.get("temperature"),
            "qualification_only": candidate_payload.get("qualification_only"),
            "checkpoint_promotable": candidate_payload.get("checkpoint_promotable"),
        },
        "d1_dataset": str(d1_dataset),
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest": str(d1_manifest),
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "d1_anchor": {"report_only": True, "manifest": d1_meta, "rows": len(d1_rows)},
        "baseline": baseline,
        "probes": probes,
        "fresh_probe_thresholds": fresh_threshold,
        "thresholds": dict(THRESHOLDS),
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
        "training_called": True,
        "probe_training_called": True,
        "formal_training_called": False,
        "environment_called": False,
        "teacher_called": False,
        "dev_gradients": False,
        "d1_gradients": False,
        "dev_selection": False,
        "d1_selection": False,
        "threshold_tuning": False,
        "early_stopping": False,
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "all_checkpoints_promotable": False,
    }
    output_manifest = output_path / "manifest.json"
    result["manifest"] = str(output_manifest)
    output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"AUDIT_COMPLETE": True, "audit_pass": audit_pass,
                      "GATE_REPAIR_ALLOWED": False, "PPO_ALLOWED": False}, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-manifest", "--v5-manifest", dest="support_manifest", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--d1-dataset", type=Path, required=True)
    parser.add_argument("--d1-manifest", type=Path, required=True)
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    _main()
