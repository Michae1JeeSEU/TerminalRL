"""Offline P2 gate learnability audit for the frozen V2 support artifact.

The audit is deliberately an offline diagnostic.  It reads the V2 support
manifest, the candidate checkpoint, and the immutable D1 anchor, then trains
two small gate probes on V2 ``train_support`` only.  It never resets an
environment, calls the Teacher, or promotes a checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.core.checkpoints import checkpoint_metadata, load_checkpoint
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_learnability_audit_v1.yaml")
PROTOCOL_SHA = "c45339541ae11decf484659cb2437e6186715480a7de7b8705e54ac7750b2840"

SUPPORT_MANIFEST_SHA = "09402508bc02155c1369de477dc320d5ca596768d4de6461f4a0526f2c7d9304"
CANDIDATE_CHECKPOINT_SHA = "5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f"
D1_DATASET_SHA = "270e54ef6ed0c36f535e2a0f25cef7cf9fdd6c91016ee295e7e336fb800cacb0"
D1_MANIFEST_SHA = "a6319c291e5c91e155fd1c7fb20ae95dcdd4973733abfe83fc10c5e6006f472c"
# Conventional names used by adjacent V4.2 runners.
V2_MANIFEST_SHA256 = SUPPORT_MANIFEST_SHA
CANDIDATE_SHA = CANDIDATE_CHECKPOINT_SHA
CANDIDATE_CHECKPOINT_SHA256 = CANDIDATE_CHECKPOINT_SHA
D1_DATASET_SHA256 = D1_DATASET_SHA
D1_MANIFEST_SHA256 = D1_MANIFEST_SHA
OBSERVATION_DIM = 88
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
INTERFACES = ("o2m", "m2i", "i2f")
PHASE_INTERFACE = {"ring_outer": "o2m", "ring_middle": "m2i", "ring_inner": "i2f"}
FUTURE_QUALIFICATION_SEEDS = tuple(range(774001, 774007))
PROBE_SEEDS = (778101, 778102, 778103)
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
    """Raised when a frozen input does not satisfy the audit contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditInputError(message)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Existing runners use the short name; keeping both makes this module easy to
# use from tests and from the command line without importing a collector.
_sha = _sha256


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def protocol() -> dict[str, Any]:
    """Read and verify the immutable audit protocol."""
    _require(PROTOCOL.is_file(), f"audit protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "learnability audit protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "learnability audit protocol must be a mapping")
    _require(value.get("protocol") == "V4_2_P2_GATE_LEARNABILITY_AUDIT_V1", "audit protocol id changed")
    bindings = value.get("bindings", {})
    _require(bindings.get("support_manifest_sha256") == SUPPORT_MANIFEST_SHA, "support manifest binding changed")
    _require(bindings.get("candidate_checkpoint_sha256") == CANDIDATE_CHECKPOINT_SHA,
             "candidate checkpoint binding changed")
    _require(bindings.get("d1_dataset_sha256") == D1_DATASET_SHA, "D1 dataset binding changed")
    _require(bindings.get("d1_manifest_sha256") == D1_MANIFEST_SHA, "D1 manifest binding changed")
    _require(bindings.get("observation_dimension") == OBSERVATION_DIM, "observation dimension changed")
    training = value.get("training", {})
    _require(tuple(training.get("seeds", ())) == PROBE_SEEDS, "probe seeds changed")
    _require(training.get("epochs") == EPOCHS, "probe epochs changed")
    _require(training.get("batch_size") == BATCH_SIZE, "probe batch size changed")
    _require(float(training.get("learning_rate")) == LEARNING_RATE, "probe learning rate changed")
    _require(float(training.get("gradient_clip_norm")) == GRADIENT_CLIP_NORM, "gradient clip changed")
    _require(training.get("loss") == "unweighted_two_class_cross_entropy", "probe loss definition changed")
    evaluation = value.get("evaluation", {})
    _require(float(evaluation.get("probability_threshold")) == PROBABILITY_THRESHOLD,
             "evaluation probability threshold changed")
    configured = value.get("thresholds", {}).get("fresh_probe", {})
    for key, expected in THRESHOLDS.items():
        _require(float(configured.get(key)) == expected, f"threshold changed: {key}")
    return value


def _resolve_artifact_path(raw: str | Path, manifest_path: Path) -> Path:
    """Resolve server paths in a synced manifest without changing its bytes."""
    path = Path(raw)
    if path.is_file():
        return path
    # The formal manifest stores the server-side absolute path.  On the local
    # Syncthing mirror the same artifact is next to the manifest.
    fallback = manifest_path.parent / path.name
    if fallback.is_file():
        return fallback
    return path


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


def _interface(row: Mapping[str, Any]) -> str | None:
    direct = row.get("interface")
    if direct in INTERFACES:
        return str(direct)
    phase = row.get("phase")
    return PHASE_INTERFACE.get(str(phase))


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    require_support_fields: bool = True,
) -> list[dict[str, Any]]:
    """Validate the immutable row contract and return shallow row copies."""
    normalized: list[dict[str, Any]] = []
    event_ids: list[str] = []
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
        event_ids.append(event)
        row["decision_event_id"] = event
        if "seed" in row:
            try:
                row["seed"] = int(row["seed"])
            except (TypeError, ValueError) as exc:
                raise AuditInputError(f"{split} row {index} has invalid seed") from exc
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
    _require(len(event_ids) == len(set(event_ids)), f"{split} decision_event_id values are not unique")
    return normalized


def _validate_support_status(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("stage") == "V4_2_P2_GATE_SUPPORT_COLLECTION_V2",
             "support manifest stage is not V2")
    _require(manifest.get("SUPPORT_COLLECTION") == "PASS", "support collection is not PASS")
    _require(manifest.get("SUPPORT_FROZEN") is True, "support manifest is not frozen")
    _require(manifest.get("LEARNABILITY_AUDIT_ALLOWED") is True,
             "learnability audit is not authorized by support manifest")
    _require(manifest.get("GATE_REPAIR_ALLOWED") is False, "support manifest unlocked gate repair")
    _require(manifest.get("PPO_ALLOWED") is False, "support manifest unlocked PPO")
    _require(manifest.get("curriculum_promotion_allowed") is False,
             "support manifest unlocked curriculum promotion")
    _require(manifest.get("d1_rows_mixed") is False, "support manifest mixes D1 rows")
    _require(manifest.get("diagnostic_rows_mixed") is False, "support manifest mixes diagnostic rows")
    _require(manifest.get("future_qualification_rollouts") == 0,
             "support manifest includes future qualification rollouts")
    _require(manifest.get("future_qualification_consumed_marker") is False,
             "support manifest consumed future qualification marker")


def validate_support_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str | None = SUPPORT_MANIFEST_SHA,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Path]]:
    """Verify V2 status, file hashes, row schema, and split disjointness."""
    path = Path(manifest_path)
    _require(path.is_file(), f"support manifest missing: {path}")
    if expected_sha256 is not None:
        _require(_sha256(path) == expected_sha256, "support manifest SHA mismatch")
    manifest = _read_json(path)
    _validate_support_status(manifest)
    bindings = manifest.get("bindings", {})
    _require(bindings.get("candidate_checkpoint_sha256") == CANDIDATE_CHECKPOINT_SHA,
             "support candidate binding mismatch")
    _require(bindings.get("d1_dataset_sha256") == D1_DATASET_SHA, "support D1 dataset binding mismatch")
    _require(bindings.get("d1_manifest_sha256") == D1_MANIFEST_SHA, "support D1 manifest binding mismatch")
    files = manifest.get("support_files")
    _require(isinstance(files, Mapping), "support_files missing from V2 manifest")
    hashes = files.get("sha256", {})
    _require(isinstance(hashes, Mapping), "support_files.sha256 missing")
    rows: dict[str, list[dict[str, Any]]] = {}
    paths: dict[str, Path] = {}
    event_sets: dict[str, set[str]] = {}
    for split in ("train", "dev"):
        raw = files.get(split)
        _require(raw is not None, f"support_files.{split} missing")
        data_path = _resolve_artifact_path(raw, path)
        _require(data_path.is_file(), f"support file missing: {data_path}")
        expected_file_sha = str(hashes.get(split, ""))
        _require(bool(expected_file_sha) and _sha256(data_path) == expected_file_sha,
                 f"support {split} file SHA mismatch")
        values = _validate_rows(_read_jsonl(data_path), split=split, require_support_fields=True)
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
    """Validate immutable D1 metadata and return rows for report-only scoring."""
    dataset = Path(dataset_path)
    manifest_file = Path(manifest_path)
    _require(dataset.is_file(), f"D1 dataset missing: {dataset}")
    _require(manifest_file.is_file(), f"D1 manifest missing: {manifest_file}")
    if expected_dataset_sha256 is not None:
        _require(_sha256(dataset) == expected_dataset_sha256, "D1 dataset SHA mismatch")
    if expected_manifest_sha256 is not None:
        _require(_sha256(manifest_file) == expected_manifest_sha256, "D1 manifest SHA mismatch")
    manifest = _read_json(manifest_file)
    _require(manifest.get("dataset_sha256") == D1_DATASET_SHA, "D1 manifest dataset binding mismatch")
    _require(int(manifest.get("aggregate_rows", -1)) > 0, "D1 manifest has no aggregate rows")
    rows = _validate_rows(_read_jsonl(dataset), split="d1", require_support_fields=False)
    _require(len(rows) == int(manifest["aggregate_rows"]), "D1 row count does not match manifest")
    return manifest, rows


def _row_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, np.ndarray]:
    observations = np.asarray([row["observation_v2_1"] for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["teacher_gate"]) for row in rows], dtype=np.int64)
    _require(observations.shape == (len(rows), OBSERVATION_DIM), "row arrays are not 88D")
    _require(bool(np.isfinite(observations).all()), "row arrays are not finite")
    _require(np.isin(labels, (0, 1)).all(), "row labels are not binary")
    return torch.as_tensor(observations, dtype=torch.float32), labels


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _core_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    _require(labels.ndim == 1 and probabilities.ndim == 1 and len(labels) == len(probabilities),
             "labels/probabilities must be aligned vectors")
    _require(len(labels) > 0, "cannot score an empty dataset")
    _require(bool(np.isfinite(probabilities).all()), "probabilities are not finite")
    _require(bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all()), "probabilities outside [0,1]")
    prediction_release = probabilities >= PROBABILITY_THRESHOLD
    teacher_release = labels == 1
    teacher_wait = ~teacher_release
    prediction_wait = ~prediction_release
    tp_release = int((teacher_release & prediction_release).sum())
    fp_release = int((teacher_wait & prediction_release).sum())
    fn_release = int((teacher_release & prediction_wait).sum())
    tp_wait = int((teacher_wait & prediction_wait).sum())
    fp_wait = int((teacher_release & prediction_wait).sum())
    fn_wait = int((teacher_wait & prediction_release).sum())
    wait_recall = tp_wait / max(1, int(teacher_wait.sum()))
    release_recall = tp_release / max(1, int(teacher_release.sum()))
    wait_precision = tp_wait / max(1, int(prediction_wait.sum()))
    release_precision = tp_release / max(1, int(prediction_release.sum()))
    wait_f1 = 2.0 * wait_precision * wait_recall / max(1e-12, wait_precision + wait_recall)
    release_f1 = 2.0 * release_precision * release_recall / max(1e-12, release_precision + release_recall)
    stable = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    nll = -(labels * np.log(stable) + (1 - labels) * np.log1p(-stable))
    wait_probability = probabilities[teacher_wait]
    release_probability = probabilities[teacher_release]
    return {
        "samples": int(len(labels)),
        "teacher": {"WAIT": int(teacher_wait.sum()), "RELEASE": int(teacher_release.sum())},
        "predicted": {"WAIT": int(prediction_wait.sum()), "RELEASE": int(prediction_release.sum())},
        "confusion": {
            "Teacher_WAIT_predicted_WAIT": tp_wait,
            "Teacher_WAIT_predicted_RELEASE": fn_wait,
            "Teacher_RELEASE_predicted_WAIT": fn_release,
            "Teacher_RELEASE_predicted_RELEASE": tp_release,
        },
        "WAIT_recall": float(wait_recall),
        "RELEASE_recall": float(release_recall),
        "WAIT_precision": float(wait_precision),
        "RELEASE_precision": float(release_precision),
        "macro_F1": float((wait_f1 + release_f1) / 2.0),
        "false_WAIT": fp_wait,
        "false_RELEASE": fp_release,
        "false_wait_rate": float(fp_wait / max(1, int(teacher_release.sum()))),
        "false_release_rate": float(fp_release / max(1, int(teacher_wait.sum()))),
        "brier": float(np.mean((stable - labels) ** 2)),
        "binary_nll": float(np.mean(nll)),
        "probabilities_by_teacher_label": {
            "WAIT": _stats(wait_probability),
            "RELEASE": _stats(release_probability),
        },
        # This spelling is retained for compatibility with the earlier gate
        # forgetting diagnostic and makes reports easy to compare.
        "release_probability": {
            "teacher_WAIT": _stats(wait_probability),
            "teacher_RELEASE": _stats(release_probability),
        },
    }
    result["probabilities_by_label"] = result["probabilities_by_teacher_label"]
    result["probability_by_label"] = result["probabilities_by_teacher_label"]
    result["confusion_matrix"] = result["confusion"]
    result["NLL"] = result["binary_nll"]
    result["Brier"] = result["brier"]
    return result


def gate_metrics(rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Return the preregistered binary gate metrics for one dataset."""
    labels = np.asarray([int(row["teacher_gate"]) for row in rows], dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    result = _core_metrics(labels, values)
    for dimension in ("scenario", "interface"):
        groups: dict[str, dict[str, Any]] = {}
        keys = [str(row.get(dimension, _interface(row) if dimension == "interface" else "unknown"))
                for row in rows]
        for key in sorted(set(keys)):
            mask = np.asarray([item == key for item in keys], dtype=bool)
            groups[key] = _core_metrics(labels[mask], values[mask])
        result[dimension] = groups
    result["breakdowns"] = {"scenario": result["scenario"], "interface": result["interface"]}
    result["scenarios"] = result["scenario"]
    result["interfaces"] = result["interface"]
    return result


# Explicit aliases make the scoring helper discoverable to small audit tests.
score_rows = gate_metrics
binary_gate_metrics = gate_metrics


def _probabilities(model: nn.Module, rows: Sequence[Mapping[str, Any]], *, batch_size: int = 2048) -> np.ndarray:
    observations, _ = _row_arrays(rows)
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = observations[start:start + batch_size]
            distribution, _ = model.distributions(batch)
            values.append(distribution.probs[:, 1].detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)


def _proposal_logits(model: nn.Module, rows: Sequence[Mapping[str, Any]], *, batch_size: int = 2048) -> np.ndarray:
    observations, _ = _row_arrays(rows)
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = observations[start:start + batch_size]
            _, proposal = model.distributions(batch)
            values.append(proposal.logits.detach().cpu().numpy().copy())
    return np.concatenate(values, axis=0)


def evaluate_model(model: nn.Module, datasets: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Evaluate a model once per named dataset, without changing its state."""
    return {name: gate_metrics(rows, _probabilities(model, rows)) for name, rows in datasets.items()}


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
    """Train a gate probe on train_support only and retain only final state."""
    _require(len(train_rows) > 0, "probe train rows are empty")
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
                float(gradient_clip_norm),
                error_if_nonfinite=True,
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
    changed: list[str] = []
    _require(set(before) == set(after), "model state keys changed during probe")
    for name in before:
        left, right = before[name].detach().cpu(), after[name].detach().cpu()
        if not torch.equal(left, right):
            changed.append(name)
    return changed


def proposal_logits_invariance(
    before_model: nn.Module,
    after_model: nn.Module,
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compare proposal logits before/after a frozen-encoder probe."""
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
        result[name] = {
            "bitwise_equal": equal,
            "max_abs_delta": max_delta,
            "samples": int(len(rows)),
        }
        overall_equal = overall_equal and equal
        maximum = max(maximum, max_delta)
    result["all_bitwise_equal"] = bool(overall_equal)
    result["all_max_abs_delta"] = float(maximum)
    result["invariant"] = bool(overall_equal or maximum == 0.0)
    return result


def threshold_checks(dev_metrics: Mapping[str, Any]) -> dict[str, bool]:
    """Apply inclusive preregistered dev and interface thresholds."""
    checks = {
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
    checks["all"] = all(checks.values())
    checks["interface"] = interface_checks
    return checks


def fresh_probe_thresholds(results: Mapping[int | str, Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate final-dev threshold checks across exactly the three seeds."""
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in PROBE_SEEDS:
        value = results.get(seed, results.get(str(seed)))
        checks = threshold_checks(value.get("evaluation", {}).get("dev", {}) if value else {})
        per_seed[str(seed)] = checks
    return {
        "per_seed": per_seed,
        "all_seeds_present": all(seed in results or str(seed) in results for seed in PROBE_SEEDS),
        "all_seeds_pass": all(item["all"] for item in per_seed.values()),
        "pass": all(
            all(seed in results or str(seed) in results for seed in PROBE_SEEDS)
            and item["all"] for item in per_seed.values()
        ),
        "thresholds": dict(THRESHOLDS),
    }


# Names used by callers that prefer a predicate over a detailed check report.
passes_threshold = lambda metrics: bool(threshold_checks(metrics)["all"])
learnability_passes = fresh_probe_thresholds


def _save_probe_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    family: str,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = p2_config(6)
    payload = {
        "metadata": checkpoint_metadata(config, policy_kind="offline_gate_probe"),
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_saved": False,
        "stage": "V4_2_P2_GATE_LEARNABILITY_AUDIT_V1",
        "probe_family": family,
        "seed": int(seed),
        "diagnostic_only": True,
        "checkpoint_promotable": False,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False,
    }
    torch.save(payload, path)


def _load_candidate(path: str | Path) -> StudentPPOV42:
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"candidate checkpoint missing: {checkpoint}")
    _require(_sha256(checkpoint) == CANDIDATE_CHECKPOINT_SHA, "candidate checkpoint SHA mismatch")
    policy = StudentPPOV42(OBSERVATION_DIM)
    load_checkpoint(checkpoint, policy, p2_config(6), load_optimizer=False)
    policy.model.eval()
    return policy


def _probe_result(
    model: nn.Module,
    train_rows: Sequence[Mapping[str, Any]],
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    training: Mapping[str, Any],
    *,
    family: str,
    checkpoint_path: Path,
    baseline: Mapping[str, Any],
    before_state: Mapping[str, torch.Tensor] | None = None,
    before_model: nn.Module | None = None,
) -> dict[str, Any]:
    evaluation = evaluate_model(model, datasets)
    result = {
        "family": family,
        "seed": int(training["seed"]),
        "training": dict(training),
        "evaluation": evaluation,
        "checkpoint": str(checkpoint_path),
        "checkpoint_promotable": False,
        "diagnostic_only": True,
        "baseline_comparison": {
            name: {
                "macro_F1_delta": float(evaluation[name]["macro_F1"] - baseline[name]["macro_F1"]),
                "binary_nll_delta": float(evaluation[name]["binary_nll"] - baseline[name]["binary_nll"]),
            }
            for name in datasets
        },
    }
    if before_state is not None:
        changed = _state_changed(before_state, model.state_dict())
        result["changed_parameter_names"] = changed
    if before_model is not None:
        result["proposal_logits_invariance"] = proposal_logits_invariance(before_model, model, datasets)
    train_improved = evaluation["train"]["binary_nll"] <= baseline["train"]["binary_nll"]
    changed = bool(result.get("changed_parameter_names"))
    result["informative"] = bool(changed and train_improved)
    return result


def run(
    output: str | Path,
    support_manifest: str | Path | None = None,
    candidate_checkpoint: str | Path | None = None,
    d1_dataset: str | Path | None = None,
    d1_manifest: str | Path | None = None,
    *,
    v2_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Run the offline audit and write one immutable diagnostic manifest."""
    p = protocol()
    support_path = Path(support_manifest or v2_manifest or "")
    _require(bool(str(support_path)), "support_manifest is required")
    _require(candidate_checkpoint is not None and d1_dataset is not None and d1_manifest is not None,
             "candidate_checkpoint, d1_dataset and d1_manifest are required")
    output_path = Path(output)
    _require(not output_path.exists(), f"audit output already exists: {output_path}")
    output_path.mkdir(parents=True, exist_ok=False)
    manifest, support_rows, support_paths = validate_support_manifest(support_path)
    d1_meta, d1_rows = validate_d1_anchor(d1_dataset, d1_manifest)
    candidate = _load_candidate(candidate_checkpoint)
    datasets: dict[str, Sequence[Mapping[str, Any]]] = {
        "train": support_rows["train"],
        "dev": support_rows["dev"],
        "d1": d1_rows,
    }
    baseline = evaluate_model(candidate.model, datasets)
    baseline_proposal_logits = {name: _proposal_logits(candidate.model, rows) for name, rows in datasets.items()}

    probe_root = output_path / "probes"
    probes: dict[str, dict[str, Any]] = {
        "candidate_frozen_encoder_gate_head": {},
        "fresh_actor_encoder_gate_head": {},
    }
    for seed in PROBE_SEEDS:
        # Candidate frozen encoder/head: only gate_head is mutable, and the
        # clone begins with the candidate's exact actor state.
        frozen_model = copy.deepcopy(candidate.model)
        before_state = {name: value.detach().clone() for name, value in frozen_model.state_dict().items()}
        before_model = copy.deepcopy(frozen_model)
        _set_trainable(frozen_model, ("gate_head",))
        training = train_gate_probe(
            frozen_model,
            support_rows["train"],
            seed=seed,
            trainable_prefixes=("gate_head",),
        )
        checkpoint_path = probe_root / f"non_promotable_probe_candidate_frozen_encoder_gate_head_{seed}.pt"
        _save_probe_checkpoint(checkpoint_path, frozen_model,
                               family="candidate_frozen_encoder_gate_head", seed=seed)
        result = _probe_result(
            frozen_model, support_rows["train"], datasets, training,
            family="candidate_frozen_encoder_gate_head", checkpoint_path=checkpoint_path,
            baseline=baseline, before_state=before_state, before_model=before_model,
        )
        # Keep the explicit all-dataset proposal snapshot in the report.  It
        # is redundant with the invariance helper but useful for audit review.
        result["proposal_logits_snapshot_sha256"] = {
            name: hashlib.sha256(np.asarray(_proposal_logits(frozen_model, rows), dtype=np.float32).tobytes()).hexdigest()
            for name, rows in datasets.items()
        }
        result["proposal_logits_baseline_snapshot_sha256"] = {
            name: hashlib.sha256(np.asarray(baseline_proposal_logits[name], dtype=np.float32).tobytes()).hexdigest()
            for name in datasets
        }
        probes["candidate_frozen_encoder_gate_head"][str(seed)] = result

        # Fresh actor encoder + gate head: its proposal head and critic are
        # frozen, but this diagnostic is independent of candidate parameters.
        _seed_everything(seed)
        fresh_model = HierarchicalActorCentralCriticV42(OBSERVATION_DIM)
        fresh_before_state = {name: value.detach().clone() for name, value in fresh_model.state_dict().items()}
        _set_trainable(fresh_model, ("actor_encoder", "gate_head"))
        training = train_gate_probe(
            fresh_model,
            support_rows["train"],
            seed=seed,
            trainable_prefixes=("actor_encoder", "gate_head"),
        )
        checkpoint_path = probe_root / f"non_promotable_probe_fresh_actor_encoder_gate_head_{seed}.pt"
        _save_probe_checkpoint(checkpoint_path, fresh_model,
                               family="fresh_actor_encoder_gate_head", seed=seed)
        result = _probe_result(
            fresh_model, support_rows["train"], datasets, training,
            family="fresh_actor_encoder_gate_head", checkpoint_path=checkpoint_path,
            baseline=baseline, before_state=fresh_before_state,
        )
        probes["fresh_actor_encoder_gate_head"][str(seed)] = result

    fresh_threshold = fresh_probe_thresholds(probes["fresh_actor_encoder_gate_head"])
    frozen_results = probes["candidate_frozen_encoder_gate_head"]
    frozen_informative = bool(frozen_results) and all(bool(value.get("informative")) for value in frozen_results.values())
    audit_pass = bool(fresh_threshold["pass"])
    result = {
        "stage": "V4_2_P2_GATE_LEARNABILITY_AUDIT_V1",
        "AUDIT_COMPLETE": True,
        "audit_pass": audit_pass,
        "learnability_pass": audit_pass,
        "repair_preregistration_allowed": audit_pass,
        "candidate_frozen_encoder_gate_head_informative": frozen_informative,
        "support_manifest": str(support_path),
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "support_files": {name: {"path": str(support_paths[name]), "sha256": _sha256(support_paths[name])}
                          for name in ("train", "dev")},
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA,
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
    parser.add_argument("--support-manifest", "--v2-manifest", dest="support_manifest", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--d1-dataset", type=Path, required=True)
    parser.add_argument("--d1-manifest", type=Path, required=True)
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    _main()
