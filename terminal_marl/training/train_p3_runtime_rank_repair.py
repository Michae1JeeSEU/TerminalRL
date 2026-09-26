"""One bounded runtime-label K=3 proposal-rank repair fit.

This stage is deliberately separate from ``train_p3_proposal_head_probe_v1``.
It consumes a completed runtime-label collection, keeps the calibrated 03c0
policy as the source, and updates only the proposal head.  The training loss
is evaluated on the raw proposal-head logits so that the differentiable K=3
rank surrogate has the intended graph.

The module does not collect tapes or run an environment.  Collection and
runtime-label provenance are checked before a fit is allowed to start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
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
from terminal_marl.core.checkpoints import checkpoint_metadata, load_checkpoint, save_checkpoint
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator
from terminal_marl.training import k3_gate_feasibility
from terminal_marl.training.k3_coverage_objective import k3_coverage_objective


STAGE = "P3_RUNTIME_RANK_REPAIR_V1"
OBSERVATION_DIM = 88
PROPOSAL_COUNT = 25
GATE_COUNT = 2
PROPOSAL_PREFIX = "proposal_head."

SOURCE_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_gate_coverage_repair_v2_20260925/fit853401/"
    "p3_gate_coverage_repair_v2.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba"
)

# This is the new configuration record for this stage.  The protocol used by
# the collector is retained only as collection lineage and is never treated as
# the fit protocol.
PROTOCOL = (
    Path(__file__).resolve().parents[2]
    / "docs/evidence/v4_2/p3_runtime_rank_repair_20260926.md"
)
SPARSE_PROTOCOL = (
    Path(__file__).resolve().parents[2]
    / "docs/evidence/v4_2/p3_runtime_rank_sparse_20260926.md"
)
TRAIN_SEEDS = frozenset({873101, 873102, 873103})
VALIDATION_SEEDS = frozenset({873104, 873105, 873106})
ALLOWED_SEEDS = TRAIN_SEEDS | VALIDATION_SEEDS
EXPECTED_SOURCE_RECORDS = (
    (873101, "flow6", "train"),
    (873102, "flow12", "train"),
    (873103, "dense24", "train"),
    (873104, "flow6", "validation"),
    (873105, "flow12", "validation"),
    (873106, "dense24", "validation"),
)

TRAIN_ROWS_EXPECTED = 82
VALIDATION_ROWS_EXPECTED = 82
TOTAL_ROWS_EXPECTED = TRAIN_ROWS_EXPECTED + VALIDATION_ROWS_EXPECTED
FULL_BATCH_SIZE = TRAIN_ROWS_EXPECTED
TRAINING_SEED = 890401
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0
KL_COEFFICIENT = 1.0
RETENTION_KL_MAX = 0.05
MAX_OPTIMIZER_STEPS = 120
TRAINABLE_PARAMETER_COUNT = 2425
RUNTIME_LABEL_SCOPE = "same_prestate_fixed_other_tape_actions_runtime_singleton"


class RuntimeRankRepairInputError(ValueError):
    """Raised when a runtime rank-repair input violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeRankRepairInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeRankRepairInputError(f"invalid JSON: {path}") from exc


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    result: dict[str, Any] = {"param_groups": [], "state": {}}
    for group in state["param_groups"]:
        result["param_groups"].append({key: value for key, value in group.items()})
    for key, values in state["state"].items():
        result["state"][key] = {
            name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for name, value in values.items()
        }
    return result


def _optimizer_state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.get("param_groups") != right.get("param_groups"):
        return False
    left_state = left.get("state", {})
    right_state = right.get("state", {})
    if set(left_state) != set(right_state):
        return False
    for key in left_state:
        if set(left_state[key]) != set(right_state[key]):
            return False
        for name in left_state[key]:
            a, b = left_state[key][name], right_state[key][name]
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                    return False
                if not torch.equal(a, b):
                    return False
            elif a != b:
                return False
    return True


def _as_finite_vector(value: Any, *, length: int, name: str) -> list[float]:
    _require(isinstance(value, (list, tuple, np.ndarray)), f"{name} must be a vector")
    _require(len(value) == length, f"{name} must have length {length}")
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeRankRepairInputError(f"{name} is not numeric") from exc
    _require(array.shape == (length,) and bool(np.isfinite(array).all()), f"{name} is not finite")
    return array.tolist()


def _as_certified(value: Any, *, name: str = "certified_proposals") -> list[int]:
    _require(isinstance(value, (list, tuple, set, frozenset)), f"{name} must be a list")
    result: list[int] = []
    for item in value:
        _require(
            isinstance(item, (int, np.integer)) and not isinstance(item, bool),
            f"{name} must contain integer indices",
        )
        index = int(item)
        _require(0 <= index < PROPOSAL_COUNT, f"{name} index out of range: {index}")
        _require(index not in result, f"{name} contains duplicate index: {index}")
        result.append(index)
    return sorted(result)


def _row_identity(row: Mapping[str, Any], index: int) -> tuple[Any, ...]:
    uid = row.get("uid", row.get("runtime_uid"))
    arrival = row.get("arrival_id", row.get("arrival"))
    _require(uid is not None and str(uid) != "", f"row {index} uid is required")
    _require(arrival is not None, f"row {index} arrival_id is required")
    return (str(row["split"]), int(row["seed"]), int(row["step"]), str(uid), int(arrival))


def _runtime_certified_value(row: Mapping[str, Any], index: int) -> Any:
    # Runtime labels take precedence over catalogue labels.  A catalogue-only
    # row is rejected instead of silently training on the stale static label.
    for key in (
        "runtime_certified_proposals",
        "runtime_certified_proposal_indices",
        "runtime_certified",
    ):
        if key in row:
            return row[key]
    labels = row.get("runtime_labels")
    if isinstance(labels, Mapping):
        for key in ("certified_proposals", "certified_proposal_indices"):
            if key in labels:
                return labels[key]
    label = row.get("runtime_label")
    if isinstance(label, Mapping):
        for key in ("certified_proposals", "certified_proposal_indices"):
            if key in label:
                return label[key]
    if "certified_proposals" in row:
        return row["certified_proposals"]
    if "catalogue_certified_proposals" in row or "catalogue_certified_proposal_indices" in row:
        raise RuntimeRankRepairInputError(f"row {index} has catalogue labels but no runtime label set")
    raise RuntimeRankRepairInputError(f"row {index} has no runtime certified proposal set")


def _validate_runtime_scope(row: Mapping[str, Any], index: int) -> None:
    scope = row.get("label_scope", row.get("scope"))
    _require(scope == RUNTIME_LABEL_SCOPE, f"row {index} runtime label scope is invalid")


def _normalize_rows(rows: Any) -> list[dict[str, Any]]:
    if isinstance(rows, Mapping):
        rows = rows.get("rows", rows.get("runtime_rows"))
    _require(isinstance(rows, list), "runtime collection rows must be a list")
    _require(len(rows) == TOTAL_ROWS_EXPECTED, f"runtime collection must contain {TOTAL_ROWS_EXPECTED} rows")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    split_seeds: dict[str, set[int]] = {"train": set(), "validation": set()}
    for index, source in enumerate(rows):
        _require(isinstance(source, Mapping), f"row {index} must be an object")
        row = dict(source)
        split = row.get("split")
        _require(split in {"train", "validation"}, f"row {index} split is invalid")
        seed = row.get("seed")
        _require(
            isinstance(seed, (int, np.integer)) and not isinstance(seed, bool),
            f"row {index} seed is required",
        )
        seed = int(seed)
        expected = TRAIN_SEEDS if split == "train" else VALIDATION_SEEDS
        _require(seed in expected, f"row {index} seed {seed} is assigned to the wrong split")
        step = row.get("step")
        _require(
            isinstance(step, (int, np.integer)) and not isinstance(step, bool) and int(step) >= 0,
            f"row {index} step is invalid",
        )
        _as_finite_vector(row.get("observation"), length=OBSERVATION_DIM, name=f"row {index} observation")
        if row.get("source_checkpoint_sha256") is not None:
            _require(
                str(row["source_checkpoint_sha256"]) == SOURCE_CHECKPOINT_SHA256,
                f"row {index} source checkpoint SHA mismatch",
            )
        if row.get("source_sha256") is not None:
            _require(str(row["source_sha256"]) == SOURCE_CHECKPOINT_SHA256, f"row {index} source SHA mismatch")
        if row.get("source_proposal_logits") is not None:
            row["source_proposal_logits"] = _as_finite_vector(
                row["source_proposal_logits"], length=PROPOSAL_COUNT,
                name=f"row {index} source_proposal_logits",
            )
        if row.get("source_gate_logits") is not None:
            row["source_gate_logits"] = _as_finite_vector(
                row["source_gate_logits"], length=GATE_COUNT,
                name=f"row {index} source_gate_logits",
            )
        row["certified_proposals"] = _as_certified(
            _runtime_certified_value(row, index), name=f"row {index} runtime_certified_proposals",
        )
        _validate_runtime_scope(row, index)
        for key in ("record_exact", "replay_exact", "outcomes_match", "rng_unchanged", "model_weights_unchanged"):
            if key in row:
                _require(row[key] is True, f"row {index} {key} is not true")
        row["seed"] = seed
        row["step"] = int(step)
        identity = _row_identity(row, index)
        _require(identity not in identities, f"duplicate runtime row identity: {identity}")
        identities.add(identity)
        split_seeds[str(split)].add(seed)
        normalized.append(row)
    _require(sum(row["split"] == "train" for row in normalized) == TRAIN_ROWS_EXPECTED, "train row count is not 82")
    _require(
        sum(row["split"] == "validation" for row in normalized) == VALIDATION_ROWS_EXPECTED,
        "validation row count is not 82",
    )
    _require(split_seeds["train"] == TRAIN_SEEDS, "runtime train seed set changed")
    _require(split_seeds["validation"] == VALIDATION_SEEDS, "runtime validation seed set changed")
    _require(not (split_seeds["train"] & split_seeds["validation"]), "train/validation seed leakage")
    return normalized


def _collection_manifest(collection_dir: Path) -> tuple[Path, Mapping[str, Any]]:
    manifest_path = collection_dir / "manifest.json"
    _require(manifest_path.is_file(), f"collection manifest missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    _require(isinstance(manifest, Mapping), "collection manifest must be an object")
    _require(str(manifest.get("status", "")) == "COMPLETED", "runtime collection is not COMPLETED")
    _require(manifest.get("runtime_labels") is True, "runtime collection runtime_labels is not true")
    source_hash = manifest.get("source_checkpoint_sha256", manifest.get("source_sha256"))
    _require(source_hash == SOURCE_CHECKPOINT_SHA256, "collection source checkpoint SHA mismatch")
    reports = manifest.get("reports", manifest.get("records"))
    _require(isinstance(reports, list) and len(reports) == len(EXPECTED_SOURCE_RECORDS), "collection must contain six records")
    observed: list[tuple[int, str, str]] = []
    for index, report in enumerate(reports):
        _require(isinstance(report, Mapping), f"collection record {index} must be an object")
        observed.append((int(report.get("seed", -1)), str(report.get("scenario")), str(report.get("split"))))
        _require(
            report.get("replay_exact", report.get("record_exact", report.get("exact"))) is True,
            f"collection record {index} is not exact",
        )
        for field in ("outcomes_match", "rng_unchanged", "model_weights_unchanged"):
            _require(report.get(field) is True, f"collection record {index} {field} is not true")
        if "runtime_labels" in report:
            _require(report.get("runtime_labels") is True, f"collection record {index} runtime_labels is not true")
        errors = report.get("errors", [])
        _require(isinstance(errors, list) and not errors, f"collection record {index} has errors")
    _require(tuple(observed) == EXPECTED_SOURCE_RECORDS, "collection seed/scenario/split binding changed")
    return manifest_path, manifest


def _load_collection(collection_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = Path(collection_dir)
    _require(directory.is_dir(), f"runtime collection directory missing: {directory}")
    manifest_path, manifest = _collection_manifest(directory)
    rows_path = directory / "rows.json"
    if not rows_path.is_file():
        rows_path = directory / "runtime_rows.json"
    _require(rows_path.is_file(), f"runtime collection rows missing: {directory / 'rows.json'}")
    rows = _normalize_rows(_read_json(rows_path))
    return rows, {
        "collection_dir": str(directory),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "rows": str(rows_path),
        "rows_sha256": _sha256(rows_path),
        "row_count": len(rows),
        "runtime_labels": True,
        "lineage_protocol": _jsonable(manifest.get("protocol")),
    }


def _build_execution_config() -> Any:
    protocol = evaluator._protocol()
    configs = evaluator._build_configs(protocol)
    config = configs["dense24"]
    _require(hasattr(config, "v4_2_rejected_release_hold_mode"), "execution config lacks canonical wait mode")
    _require(str(config.v4_2_rejected_release_hold_mode) == "canonical_wait", "canonical_wait execution mode was not applied")
    _require(not bool(config.teacher_enabled), "runtime rank repair must disable Teacher")
    _require(str(config.scheduler_mode) == "execution", "runtime rank repair must use execution scheduler")
    return config


def _load_source_policy(
    source_checkpoint: str | Path,
    config: Any,
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any], str]:
    path = Path(source_checkpoint)
    _require(path.is_file(), f"source checkpoint not found: {path}")
    actual_sha = _sha256(path)
    _require(actual_sha == SOURCE_CHECKPOINT_SHA256, f"source checkpoint SHA256 mismatch: {actual_sha}")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM, temperature=CALIBRATION_TEMPERATURE,
    )
    try:
        payload = load_checkpoint(path, policy, config, load_optimizer=False)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise RuntimeRankRepairInputError(f"strict source checkpoint load failed: {exc}") from exc
    _require(getattr(policy.model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID, "source architecture id changed")
    _require(float(policy.model.gate_temperature.detach().cpu().item()) == CALIBRATION_TEMPERATURE, "source temperature changed")
    policy.model.eval()
    return policy, payload, actual_sha


def _freeze_to_proposal_head(policy: Any) -> tuple[list[torch.nn.Parameter], list[str]]:
    trainable: list[torch.nn.Parameter] = []
    names: list[str] = []
    for name, parameter in policy.model.named_parameters():
        enabled = name.startswith(PROPOSAL_PREFIX)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
            names.append(name)
    _require(names == ["proposal_head.weight", "proposal_head.bias"], f"unexpected proposal trainable scope: {names}")
    _require(sum(parameter.numel() for parameter in trainable) == TRAINABLE_PARAMETER_COUNT, "proposal trainable parameter count is not 2425")
    return trainable, names


def _observations(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    values = torch.as_tensor([row["observation"] for row in rows], dtype=torch.float32, device=device)
    _require(values.ndim == 2 and values.shape[1] == OBSERVATION_DIM, "observation tensor is not 88D")
    _require(bool(torch.isfinite(values).all().item()), "observation tensor is not finite")
    return values


def _source_logits(rows: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    proposal = torch.as_tensor(
        [row.get("source_proposal_logits", [0.0] * PROPOSAL_COUNT) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    gate = torch.as_tensor(
        [row.get("source_gate_logits", [0.0] * GATE_COUNT) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    _require(proposal.shape == (len(rows), PROPOSAL_COUNT), "source proposal logits shape mismatch")
    _require(gate.shape == (len(rows), GATE_COUNT), "source gate logits shape mismatch")
    _require(bool(torch.isfinite(proposal).all().item()) and bool(torch.isfinite(gate).all().item()), "source logits are not finite")
    return proposal, gate


def _raw_proposal_logits(model: torch.nn.Module, observations: torch.Tensor) -> torch.Tensor:
    """Return the unnormalized proposal-head logits, with no distribution wrapper."""
    values = model.proposal_head(model.actor_encoder(observations))
    _require(values.ndim == 2 and values.shape[1] == PROPOSAL_COUNT, "raw proposal logits shape mismatch")
    _require(bool(torch.isfinite(values).all().item()), "raw proposal logits are not finite")
    return values


def _source_raw_logits(model: torch.nn.Module, observations: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        values = _raw_proposal_logits(model, observations).detach().clone()
    _require(bool(torch.isfinite(values).all().item()), "source raw proposal logits are not finite")
    return values


def _normalize_sparse_target_rows(value: Sequence[int] | None) -> list[int] | None:
    if value is None:
        return None
    rows = list(value)
    _require(
        all(isinstance(item, (int, np.integer)) and not isinstance(item, bool) for item in rows),
        "sparse target rows must contain integer output indices",
    )
    normalized = [int(item) for item in rows]
    _require(len(set(normalized)) == len(normalized), "sparse target rows contain duplicates")
    _require(all(0 <= item < PROPOSAL_COUNT for item in normalized), "sparse target row is out of range")
    return sorted(normalized)


def _derive_sparse_target_rows(
    source_train_logits: torch.Tensor,
    train_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Derive output rows from source train misses only.

    Each source K=3 miss contributes its highest-logit certified proposal row.
    Validation rows are intentionally not accepted by this seam.
    """
    _require(source_train_logits.ndim == 2, "source train logits must be a matrix")
    _require(source_train_logits.shape == (len(train_rows), PROPOSAL_COUNT), "source train logits shape mismatch")
    _require(bool(torch.isfinite(source_train_logits).all().item()), "source train logits are not finite")
    target_rows: set[int] = set()
    miss_rows: list[dict[str, Any]] = []
    for index, (logits, row) in enumerate(zip(source_train_logits, train_rows)):
        certified = set(int(item) for item in row["certified_proposals"])
        ranked = [int(item) for item in torch.argsort(logits, descending=True, stable=True).tolist()]
        if not certified or certified.intersection(ranked[:3]):
            continue
        certified_order = [item for item in ranked if item in certified]
        _require(bool(certified_order), f"source train miss {index} has no certified proposal")
        target = certified_order[0]
        target_rows.add(target)
        miss_rows.append({
            "train_row_index": index,
            "seed": int(row["seed"]) if "seed" in row else None,
            "step": int(row["step"]) if "step" in row else None,
            "uid": str(row.get("uid", row.get("runtime_uid"))) if row.get("uid", row.get("runtime_uid")) is not None else None,
            "certified_proposals": sorted(certified),
            "highest_certified_output_row": target,
        })
    return sorted(target_rows), miss_rows


def _effective_update_parameter_count(model: torch.nn.Module, target_rows: Sequence[int]) -> int:
    normalized = _normalize_sparse_target_rows(target_rows)
    _require(normalized is not None, "target rows are required")
    weight = getattr(model, "proposal_head", None)
    _require(weight is not None and hasattr(weight, "weight") and hasattr(weight, "bias"), "proposal head is missing")
    _require(weight.weight.ndim == 2 and weight.bias.ndim == 1, "proposal head dimensions are invalid")
    _require(weight.weight.shape[0] == PROPOSAL_COUNT and weight.bias.shape[0] == PROPOSAL_COUNT, "proposal head row count changed")
    return len(normalized) * (int(weight.weight.shape[1]) + 1)


def _mask_proposal_gradients(model: torch.nn.Module, target_rows: Sequence[int]) -> None:
    normalized = _normalize_sparse_target_rows(target_rows)
    _require(normalized is not None, "target rows are required")
    weight = getattr(model, "proposal_head", None)
    _require(weight is not None and hasattr(weight, "weight") and hasattr(weight, "bias"), "proposal head is missing")
    for parameter in (weight.weight, weight.bias):
        _require(parameter.grad is not None, "missing proposal gradient for sparse mask")
    weight_mask = torch.zeros_like(weight.weight.grad)
    bias_mask = torch.zeros_like(weight.bias.grad)
    if normalized:
        indices = torch.as_tensor(normalized, dtype=torch.long, device=weight_mask.device)
        weight_mask.index_fill_(0, indices, 1.0)
        bias_mask.index_fill_(0, indices, 1.0)
    weight.weight.grad.mul_(weight_mask)
    weight.bias.grad.mul_(bias_mask)


def _non_target_proposal_rows_bitwise_unchanged(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    target_rows: Sequence[int],
) -> bool:
    normalized = _normalize_sparse_target_rows(target_rows)
    _require(normalized is not None, "target rows are required")
    target = set(normalized)
    non_target = [index for index in range(PROPOSAL_COUNT) if index not in target]
    if not non_target:
        return True
    indices = torch.as_tensor(non_target, dtype=torch.long)
    for name in ("proposal_head.weight", "proposal_head.bias"):
        _require(name in before and name in after, f"missing proposal state: {name}")
        if not torch.equal(before[name].index_select(0, indices), after[name].index_select(0, indices)):
            return False
    return True


def _validate_source_rows(policy: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    device = policy.device
    for index, row in enumerate(rows):
        observations = _observations([row], device)
        with torch.no_grad():
            gate, proposal = policy.model.distributions(observations)
        if row.get("source_proposal_logits") is not None:
            expected = torch.as_tensor(row["source_proposal_logits"], dtype=torch.float32, device=device)
            _require(torch.equal(proposal.logits[0], expected), f"collection source proposal logits mismatch at row {index}")
        if row.get("source_gate_logits") is not None:
            expected = torch.as_tensor(row["source_gate_logits"], dtype=torch.float32, device=device)
            _require(torch.equal(gate.logits[0], expected), f"collection source gate logits mismatch at row {index}")


def _kl_from_logits(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    _require(reference_logits.shape == candidate_logits.shape, "KL logits shape mismatch")
    # Accumulate the retention metric in float64.  Float32 roundoff can make
    # the KL of two nearly identical distributions slightly negative and would
    # make the explicit zero-cap rollback dependent on that roundoff.
    reference_values = reference_logits.to(dtype=torch.float64)
    candidate_values = candidate_logits.to(dtype=torch.float64)
    reference_logp = F.log_softmax(reference_values, dim=-1)
    candidate_logp = F.log_softmax(candidate_values, dim=-1)
    values = (reference_logp.exp() * (reference_logp - candidate_logp)).sum(dim=-1)
    _require(bool(torch.isfinite(values).all().item()), "non-finite proposal KL")
    return values


def _rank_summary(logits: torch.Tensor, certified: Sequence[Sequence[int]]) -> dict[str, Any]:
    _require(not logits.requires_grad, "rank summary must not retain a training graph")
    objective = k3_coverage_objective(logits, certified)
    masses = [
        float(k3_gate_feasibility.k3_certified_mass(logits[index].cpu(), row, shortlist_k=3)["certified_mass"])
        for index, row in enumerate(certified)
    ]
    _require(bool(np.isfinite(np.asarray(masses, dtype=np.float64)).all()), "non-finite qK3 metrics")
    return {
        "top3_misses": int(objective["miss_count"]),
        "rank_loss": float(objective["loss"].detach().cpu().item()),
        "qK3_mean": float(np.mean(masses)),
        "qK3_min": float(np.min(masses)),
        "qK3_max": float(np.max(masses)),
    }


def _coverage_metrics(
    policy: Any,
    rows: Sequence[Mapping[str, Any]],
    source_logits: torch.Tensor | None = None,
) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "top3_misses": 0, "qK3_mean": None, "qK3_min": None, "qK3_max": None, "kl_mean": None, "kl_max": None}
    observations = _observations(rows, policy.device)
    certified = [row["certified_proposals"] for row in rows]
    with torch.no_grad():
        candidate = _raw_proposal_logits(policy.model, observations).detach()
        reference = candidate if source_logits is None else source_logits.detach()
        kl = _kl_from_logits(reference, candidate)
        rank = _rank_summary(candidate, certified)
    result = {
        "rows": len(rows),
        "nonempty_certified_rows": sum(bool(row) for row in certified),
        "top3_misses": rank["top3_misses"],
        "rank_loss": rank["rank_loss"],
        "qK3_mean": rank["qK3_mean"],
        "qK3_min": rank["qK3_min"],
        "qK3_max": rank["qK3_max"],
        "kl_mean": float(kl.mean().cpu().item()),
        "kl_max": float(kl.max().cpu().item()),
    }
    _require(all(np.isfinite(float(value)) for value in result.values() if isinstance(value, (int, float))), "non-finite coverage metrics")
    return result


def _all_metrics(
    policy: Any,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    source_train_logits: torch.Tensor,
    source_validation_logits: torch.Tensor,
) -> dict[str, Any]:
    return {
        "train": _coverage_metrics(policy, train_rows, source_train_logits),
        "validation": _coverage_metrics(policy, validation_rows, source_validation_logits),
    }


def _proposal_delta_norm(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    values = []
    for name in ("proposal_head.weight", "proposal_head.bias"):
        values.append((after[name].float() - before[name].float()).pow(2).sum())
    return float(torch.sqrt(torch.stack(values).sum()).item())


def _require_finite_state(model: torch.nn.Module) -> None:
    for name, value in model.state_dict().items():
        _require(bool(torch.isfinite(value).all().item()), f"non-finite model state: {name}")


def _gate_logits(model: torch.nn.Module, observations: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        distributions = model.distributions(observations)
        values = distributions[0].logits.detach().clone()
    _require(bool(torch.isfinite(values).all().item()), "non-finite gate logits")
    return values


def _fit_steps(
    policy: Any,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    source_train_logits: torch.Tensor,
    source_validation_logits: torch.Tensor,
    trainable_parameters: Sequence[torch.nn.Parameter] | None = None,
    trainable_names: Sequence[str] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int | None = None,
    retention_kl_max: float | None = None,
    progress_path: Path | None = None,
    sparse_target_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run the bounded fit on a policy; this seam is also used by toy tests."""
    device = policy.device
    train_x = _observations(train_rows, device)
    validation_x = _observations(validation_rows, device)
    train_certified = [row["certified_proposals"] for row in train_rows]
    validation_certified = [row["certified_proposals"] for row in validation_rows]
    if trainable_parameters is None:
        trainable_parameters = [parameter for name, parameter in policy.model.named_parameters() if name.startswith(PROPOSAL_PREFIX)]
    if trainable_names is None:
        trainable_names = [name for name, parameter in policy.model.named_parameters() if name.startswith(PROPOSAL_PREFIX)]
    if optimizer is None:
        optimizer = torch.optim.Adam(list(trainable_parameters), lr=LEARNING_RATE)
    if max_steps is None:
        max_steps = MAX_OPTIMIZER_STEPS
    if retention_kl_max is None:
        retention_kl_max = RETENTION_KL_MAX
    sparse_target_rows = _normalize_sparse_target_rows(sparse_target_rows)
    policy.optimizer = optimizer
    model = policy.model
    model.eval()
    before_state = _state_cpu(model)
    before_gate_train = _gate_logits(model, train_x)
    before_gate_validation = _gate_logits(model, validation_x)
    initial_metrics = _all_metrics(
        policy, train_rows, validation_rows,
        source_train_logits=source_train_logits,
        source_validation_logits=source_validation_logits,
    )
    history: list[dict[str, Any]] = []
    accepted_steps = 0
    rejected_steps = 0
    stopped_reason: str | None = None

    # A fresh source model has no rank-repair graph; the source tensors are
    # detached snapshots captured before this loop.
    for step in range(1, int(max_steps) + 1):
        with torch.no_grad():
            current_raw = _raw_proposal_logits(model, train_x).detach()
            current_rank = k3_coverage_objective(current_raw, train_certified)
        _require(bool(torch.isfinite(current_rank["loss"]).item()), f"non-finite rank loss before step {step}")
        if int(current_rank["miss_count"]) == 0:
            stopped_reason = "train_miss_count_zero"
            break

        model_before = _state_cpu(model)
        optimizer_before = _optimizer_state_cpu(optimizer)
        optimizer.zero_grad(set_to_none=True)
        raw_logits = _raw_proposal_logits(model, train_x)
        rank = k3_coverage_objective(raw_logits, train_certified)
        kl_values = _kl_from_logits(source_train_logits, raw_logits)
        kl_loss = kl_values.mean()
        loss = rank["loss"] + KL_COEFFICIENT * kl_loss
        _require(bool(torch.isfinite(loss).item()), f"non-finite loss at step {step}")
        loss.backward()
        gradients = [parameter.grad for parameter in trainable_parameters if parameter.grad is not None]
        _require(bool(gradients), f"missing proposal gradient at step {step}")
        _require(all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients), f"non-finite proposal gradient at step {step}")
        if sparse_target_rows is not None:
            _mask_proposal_gradients(model, sparse_target_rows)
            gradients = [parameter.grad for parameter in trainable_parameters if parameter.grad is not None]
            _require(all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients), f"non-finite masked proposal gradient at step {step}")
        norm = torch.nn.utils.clip_grad_norm_(list(trainable_parameters), GRADIENT_CLIP_NORM)
        _require(bool(torch.isfinite(norm).item()), f"non-finite gradient norm at step {step}")
        optimizer.step()
        _require_finite_state(model)

        with torch.no_grad():
            after_raw = _raw_proposal_logits(model, train_x).detach()
            after_kl_values = _kl_from_logits(source_train_logits, after_raw)
            after_kl = float(after_kl_values.mean().cpu().item())
        _require(np.isfinite(after_kl), f"non-finite train KL at step {step}")
        rejected = after_kl > float(retention_kl_max)
        if rejected:
            model.load_state_dict(model_before, strict=True)
            optimizer.load_state_dict(optimizer_before)
            rejected_steps += 1
            stopped_reason = "retention_kl_cap_exceeded"
        else:
            accepted_steps += 1

        model.eval()
        metrics = _all_metrics(
            policy, train_rows, validation_rows,
            source_train_logits=source_train_logits,
            source_validation_logits=source_validation_logits,
        )
        train_metrics = metrics["train"]
        step_state = _state_cpu(model)
        step_nonproposal_unchanged = all(
            torch.equal(before_state[name], step_state[name])
            for name in before_state
            if not name.startswith(PROPOSAL_PREFIX)
        )
        step_non_target_unchanged = (
            True
            if sparse_target_rows is None
            else _non_target_proposal_rows_bitwise_unchanged(before_state, step_state, sparse_target_rows)
        )
        _require(step_non_target_unchanged, f"non-target proposal row changed at step {step}")
        step_gate_unchanged = torch.equal(before_gate_train, _gate_logits(model, train_x)) and torch.equal(
            before_gate_validation, _gate_logits(model, validation_x),
        )
        record = {
            "step": step,
            "accepted": not rejected,
            "accepted_steps_total": accepted_steps,
            "rejected_steps_total": rejected_steps,
            "loss": float(loss.detach().cpu().item()),
            "rank_loss": float(rank["loss"].detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "train": train_metrics,
            "validation": metrics["validation"],
            "train_qK3": train_metrics["qK3_mean"],
            "validation_qK3": metrics["validation"]["qK3_mean"],
            "train_top3_misses": train_metrics["top3_misses"],
            "validation_top3_misses": metrics["validation"]["top3_misses"],
            "train_kl": train_metrics["kl_mean"],
            "validation_kl": metrics["validation"]["kl_mean"],
            "nonproposal_state_bitwise_unchanged": step_nonproposal_unchanged,
            "non_target_proposal_rows_bitwise_unchanged": step_non_target_unchanged,
            "gate_logits_bitwise_unchanged": step_gate_unchanged,
            "rejected_reason": stopped_reason if rejected else None,
        }
        history.append(record)
        _require(all(np.isfinite(float(value)) for value in (record["loss"], record["rank_loss"], record["kl_loss"])), f"non-finite metrics at step {step}")
        if progress_path is not None:
            _write_json(progress_path, {
                "stage": STAGE,
                "status": "RUNNING",
                "processed_rows": len(train_rows) + len(validation_rows),
                "steps_completed": step,
                "accepted_steps": accepted_steps,
                "rejected_steps": rejected_steps,
                "stopped_reason": stopped_reason,
                "latest": record,
            })
        if rejected:
            break
        if int(train_metrics["top3_misses"]) == 0:
            stopped_reason = "train_miss_count_zero"
            break

    model.eval()
    after_state = _state_cpu(model)
    after_gate_train = _gate_logits(model, train_x)
    after_gate_validation = _gate_logits(model, validation_x)
    nonproposal_state_bitwise_unchanged = all(
        torch.equal(before_state[name], after_state[name])
        for name in before_state
        if not name.startswith(PROPOSAL_PREFIX)
    )
    gate_logits_bitwise_unchanged = torch.equal(before_gate_train, after_gate_train) and torch.equal(
        before_gate_validation, after_gate_validation,
    )
    non_target_proposal_rows_bitwise_unchanged = (
        True
        if sparse_target_rows is None
        else _non_target_proposal_rows_bitwise_unchanged(before_state, after_state, sparse_target_rows)
    )
    _require(non_target_proposal_rows_bitwise_unchanged, "non-target proposal row changed during fit")
    frozen_parameters = all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith(PROPOSAL_PREFIX)
    )
    delta_norm = _proposal_delta_norm(before_state, after_state)
    effective_fit = bool(delta_norm > 0.0 and accepted_steps > 0)
    return {
        "before_state": before_state,
        "after_state": after_state,
        "before_metrics": initial_metrics,
        "after_metrics": _all_metrics(
            policy, train_rows, validation_rows,
            source_train_logits=source_train_logits,
            source_validation_logits=source_validation_logits,
        ),
        "history": history,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "stopped_reason": stopped_reason,
        "delta_norm": delta_norm,
        "effective_fit": effective_fit,
        "trainable_names": list(trainable_names),
        "trainable_count": sum(parameter.numel() for parameter in trainable_parameters),
        "requires_grad_parameter_count": sum(parameter.numel() for parameter in trainable_parameters),
        "sparse_target_rows": list(sparse_target_rows) if sparse_target_rows is not None else None,
        "effective_update_parameters": (
            _effective_update_parameter_count(model, sparse_target_rows)
            if sparse_target_rows is not None
            else sum(parameter.numel() for parameter in trainable_parameters)
        ),
        "nonproposal_state_bitwise_unchanged": nonproposal_state_bitwise_unchanged,
        "non_target_proposal_rows_bitwise_unchanged": non_target_proposal_rows_bitwise_unchanged,
        "gate_logits_bitwise_unchanged": gate_logits_bitwise_unchanged,
        "all_nonproposal_parameters_frozen": frozen_parameters,
        "train_x": train_x,
        "validation_x": validation_x,
    }


def _output_paths(output_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(output_dir)
    _require(not root.exists(), f"refusing to overwrite output directory: {root}")
    root.mkdir(parents=True, exist_ok=False)
    return root, root / "p3_runtime_rank_repair_v1.pt", root / "fit_report.json"


def _metadata(
    config: Any,
    *,
    source_checkpoint: Path,
    source_sha: str,
    collection_report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source_payload: Mapping[str, Any],
    sparse_target_rows: Sequence[int] | None = None,
    source_initial_train_miss_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    value = dict(checkpoint_metadata(config))
    sparse_target_rows = _normalize_sparse_target_rows(sparse_target_rows)
    fit_mode = "sparse_target_rows" if sparse_target_rows is not None else "full_proposal_head"
    protocol = SPARSE_PROTOCOL if sparse_target_rows is not None else PROTOCOL
    effective_update_parameters = (
        len(sparse_target_rows) * (TRAINABLE_PARAMETER_COUNT // PROPOSAL_COUNT)
        if sparse_target_rows is not None
        else TRAINABLE_PARAMETER_COUNT
    )
    source_metadata = source_payload.get("metadata", {})
    if isinstance(source_metadata, Mapping):
        # Preserve source declarations that are not part of the generic
        # checkpoint schema; stage/method fields below intentionally describe
        # this candidate and therefore override same-named source fields.
        for key, item in source_metadata.items():
            value.setdefault(str(key), _jsonable(item))
    value.update({
        "stage": STAGE,
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": float(CALIBRATION_TEMPERATURE),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "source_sha256": source_sha,
        "source_metadata": _jsonable(source_metadata),
        "source_architecture_id": source_metadata.get("architecture_id", CALIBRATED_ARCHITECTURE_ID)
        if isinstance(source_metadata, Mapping) else CALIBRATED_ARCHITECTURE_ID,
        "protocol": str(protocol),
        "protocol_sha256": _sha256(protocol),
        "fit_mode": fit_mode,
        "sparse_target_rows": list(sparse_target_rows) if sparse_target_rows is not None else None,
        "source_initial_train_miss_rows": _jsonable(source_initial_train_miss_rows),
        "collection_dir": collection_report["collection_dir"],
        "collection_rows_sha256": collection_report["rows_sha256"],
        "collection_manifest_sha256": collection_report["manifest_sha256"],
        "collection_lineage_protocol": collection_report.get("lineage_protocol"),
        "train_seeds": sorted({int(row["seed"]) for row in rows if row["split"] == "train"}),
        "validation_seeds": sorted({int(row["seed"]) for row in rows if row["split"] == "validation"}),
        "development_seed_reuse": TRAINING_SEED,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "full_batch_size": FULL_BATCH_SIZE,
        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "retention_kl_max": RETENTION_KL_MAX,
        "kl_coefficient": KL_COEFFICIENT,
        "rank_objective": "k3_coverage_objective(raw_proposal_logits, C)",
        "method": {
            "mode": fit_mode,
            "rank_objective": "k3_coverage_objective(raw_proposal_logits, C)",
            "kl_coefficient": KL_COEFFICIENT,
            "full_batch_size": FULL_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "retention_kl_max": RETENTION_KL_MAX,
            "validation_used_for_update": False,
            "validation_used_for_selection": False,
            "early_stop": "train_miss_count_zero",
            "sparse_target_rows": list(sparse_target_rows) if sparse_target_rows is not None else None,
            "source_initial_train_miss_rows": _jsonable(source_initial_train_miss_rows),
            "effective_update_parameters": effective_update_parameters,
        },
        "trainable_scope": ["proposal_head.weight", "proposal_head.bias"],
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "requires_grad_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "effective_update_parameters": effective_update_parameters,
        "source_optimizer_copied": False,
        "runtime_labels": True,
        "PPO_ALLOWED": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "heldout_consumed": False,
    })
    return value


def _save_and_restore(
    checkpoint: Path,
    policy: Any,
    config: Any,
    metadata: Mapping[str, Any],
    report: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, bool, bool]:
    save_checkpoint(checkpoint, policy, config, metadata=dict(metadata), report=dict(report))
    restored = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM, temperature=CALIBRATION_TEMPERATURE,
    )
    restored_parameters, _ = _freeze_to_proposal_head(restored)
    restored.optimizer = torch.optim.Adam(restored_parameters, lr=LEARNING_RATE)
    loaded = load_checkpoint(checkpoint, restored, config, load_optimizer=True)
    _require(loaded.get("policy_architecture_migrated") is False, "candidate architecture migrated")
    model_exact = _state_equal(_state_cpu(policy.model), _state_cpu(restored.model))
    optimizer_exact = _optimizer_state_equal(_optimizer_state_cpu(policy.optimizer), _optimizer_state_cpu(restored.optimizer))
    _require(model_exact, "candidate checkpoint model roundtrip mismatch")
    _require(optimizer_exact, "candidate checkpoint optimizer roundtrip mismatch")
    source_train = _observations(train_rows, restored.device)
    source_validation = _observations(validation_rows, restored.device)
    gate_exact = torch.equal(_gate_logits(policy.model, _observations(train_rows, policy.device)), _gate_logits(restored.model, source_train))
    gate_exact = gate_exact and torch.equal(
        _gate_logits(policy.model, _observations(validation_rows, policy.device)),
        _gate_logits(restored.model, source_validation),
    )
    restored_allstate_exact = model_exact and optimizer_exact and gate_exact
    _require(restored_allstate_exact, "candidate checkpoint allstate roundtrip mismatch")
    return model_exact, optimizer_exact, gate_exact, restored_allstate_exact


def fit_runtime_rank_repair(
    collection_dir: str | Path,
    output_dir: str | Path,
    *,
    source_checkpoint: str | Path = SOURCE_CHECKPOINT,
    sparse_target_rows: bool = False,
) -> dict[str, Any]:
    """Run the registered runtime-label rank-repair fit."""
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    rows, collection_report = _load_collection(collection_dir)
    config = _build_execution_config()
    policy, source_payload, source_sha = _load_source_policy(source_checkpoint, config)
    _validate_source_rows(policy, rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    root, checkpoint_path, report_path = _output_paths(output_dir)
    progress_path = root / "progress.json"
    _write_json(progress_path, {"stage": STAGE, "status": "RUNNING", "processed_rows": len(rows), "steps_completed": 0})

    trainable_parameters, trainable_names = _freeze_to_proposal_head(policy)
    policy.optimizer = torch.optim.Adam(trainable_parameters, lr=LEARNING_RATE)
    model = policy.model
    model.eval()
    train_x = _observations(train_rows, policy.device)
    validation_x = _observations(validation_rows, policy.device)
    source_train_logits = _source_raw_logits(model, train_x)
    source_validation_logits = _source_raw_logits(model, validation_x)
    derived_sparse_rows: list[int] | None = None
    source_initial_train_miss_rows: list[dict[str, Any]] = []
    if sparse_target_rows:
        derived_sparse_rows, source_initial_train_miss_rows = _derive_sparse_target_rows(
            source_train_logits, train_rows,
        )
    fit = _fit_steps(
        policy,
        train_rows,
        validation_rows,
        source_train_logits=source_train_logits,
        source_validation_logits=source_validation_logits,
        trainable_parameters=trainable_parameters,
        trainable_names=trainable_names,
        optimizer=policy.optimizer,
        progress_path=progress_path,
        sparse_target_rows=derived_sparse_rows,
    )
    if fit["stopped_reason"] == "retention_kl_cap_exceeded":
        status = "STOPPED_KL_CAP"
    elif fit["stopped_reason"] == "train_miss_count_zero" and fit["accepted_steps"] > 0:
        status = "COMPLETED"
    elif fit["accepted_steps"] == 0 and fit["rejected_steps"] == 0:
        status = "NO_EFFECTIVE_FIT"
    else:
        status = "COMPLETED"

    metadata = _metadata(
        config,
        source_checkpoint=Path(source_checkpoint),
        source_sha=source_sha,
        collection_report=collection_report,
        rows=rows,
        source_payload=source_payload,
        sparse_target_rows=derived_sparse_rows,
        source_initial_train_miss_rows=source_initial_train_miss_rows,
    )
    fit_mode = "sparse_target_rows" if derived_sparse_rows is not None else "full_proposal_head"
    protocol = SPARSE_PROTOCOL if derived_sparse_rows is not None else PROTOCOL
    invariants = {
        "trainable_parameter_names": fit["trainable_names"],
        "trainable_parameter_count": fit["trainable_count"],
        "requires_grad_parameter_count": fit["requires_grad_parameter_count"],
        "effective_update_parameters": fit["effective_update_parameters"],
        "sparse_target_rows": fit["sparse_target_rows"],
        "non_target_proposal_rows_bitwise_unchanged": fit["non_target_proposal_rows_bitwise_unchanged"],
        "nonproposal_state_bitwise_unchanged": fit["nonproposal_state_bitwise_unchanged"],
        "gate_logits_bitwise_unchanged": fit["gate_logits_bitwise_unchanged"],
        "all_nonproposal_parameters_frozen": fit["all_nonproposal_parameters_frozen"],
        "source_optimizer_copied": False,
        "retention_kl_max": RETENTION_KL_MAX,
    }
    _require(invariants["nonproposal_state_bitwise_unchanged"], "non-proposal state changed during fit")
    _require(invariants["non_target_proposal_rows_bitwise_unchanged"], "non-target proposal row changed during fit")
    _require(invariants["gate_logits_bitwise_unchanged"], "gate logits changed during fit")
    _require(invariants["all_nonproposal_parameters_frozen"], "non-proposal parameters remain trainable")
    report: dict[str, Any] = {
        "stage": STAGE,
        "status": status,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "collection": collection_report,
        "protocol": str(protocol),
        "protocol_sha256": _sha256(protocol),
        "fit_mode": fit_mode,
        "sparse_target_rows": fit["sparse_target_rows"],
        "source_initial_train_miss_rows": source_initial_train_miss_rows,
        "seed": TRAINING_SEED,
        "development_seed_reuse": True,
        "method": {
            "mode": fit_mode,
            "rank_objective": "k3_coverage_objective(raw_proposal_logits, C)",
            "kl_coefficient": KL_COEFFICIENT,
            "full_batch_size": FULL_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "retention_kl_max": RETENTION_KL_MAX,
            "validation_used_for_update": False,
            "validation_used_for_selection": False,
            "early_stop": "train_miss_count_zero",
            "sparse_target_rows": fit["sparse_target_rows"],
            "source_initial_train_miss_rows": source_initial_train_miss_rows,
            "effective_update_parameters": fit["effective_update_parameters"],
        },
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "accepted_steps": fit["accepted_steps"],
        "rejected_steps": fit["rejected_steps"],
        "stopped_reason": fit["stopped_reason"],
        "effective_fit": fit["effective_fit"],
        "metrics_before": fit["before_metrics"],
        "metrics_after": fit["after_metrics"],
        "history": fit["history"],
        "invariants": invariants,
        "metadata": metadata,
    }
    model_exact, optimizer_exact, gate_exact, allstate_exact = _save_and_restore(
        checkpoint_path, policy, config, metadata, report, train_rows, validation_rows,
    )
    report["checkpoint"] = str(checkpoint_path)
    report["checkpoint_sha256"] = _sha256(checkpoint_path)
    report["checkpoint_roundtrip_model_exact"] = model_exact
    report["checkpoint_roundtrip_optimizer_exact"] = optimizer_exact
    report["checkpoint_roundtrip_gate_exact"] = gate_exact
    report["checkpoint_roundtrip_allstate_exact"] = allstate_exact
    report["invariants"].update({
        "checkpoint_roundtrip_model_exact": model_exact,
        "checkpoint_roundtrip_optimizer_exact": optimizer_exact,
        "checkpoint_roundtrip_gate_exact": gate_exact,
        "checkpoint_roundtrip_allstate_exact": allstate_exact,
    })
    _write_json(report_path, report)
    _write_json(progress_path, {
        "stage": STAGE,
        "status": status,
        "processed_rows": len(rows),
        "accepted_steps": fit["accepted_steps"],
        "rejected_steps": fit["rejected_steps"],
        "stopped_reason": fit["stopped_reason"],
        "checkpoint": str(checkpoint_path),
    })
    return {
        "status": status,
        "stage": STAGE,
        "checkpoint": str(checkpoint_path),
        "fit_report": str(report_path),
        "progress": str(progress_path),
        "accepted_steps": fit["accepted_steps"],
        "rejected_steps": fit["rejected_steps"],
        "stopped_reason": fit["stopped_reason"],
        "source_checkpoint_sha256": source_sha,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "effective_fit": fit["effective_fit"],
        "invariants": report["invariants"],
    }


# Explicit aliases make the stage convenient to call without making it look
# like the old proposal-head probe in metadata or output names.
fit_runtime_rank_repair_v1 = fit_runtime_rank_repair
fit = fit_runtime_rank_repair


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--sparse-target-rows", action="store_true")
    args = parser.parse_args()
    result = fit_runtime_rank_repair(
        args.collection_dir,
        args.output_dir,
        source_checkpoint=args.source_checkpoint,
        sparse_target_rows=args.sparse_target_rows,
    )
    print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result["status"] in {"COMPLETED", "NO_EFFECTIVE_FIT", "STOPPED_KL_CAP"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
