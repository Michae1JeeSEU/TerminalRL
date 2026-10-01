"""Bounded, adapter-only preparation fit for the P3 gate targets.

The module has two deliberately separate phases.  ``preflight`` validates the
already produced JSON/NPZ inputs without constructing a model.  ``fit`` loads
the frozen BEFORE actor, trains only ``gate_adapter.*`` for at most the
registered number of full-batch updates, and writes a candidate checkpoint
only after the intrinsic screens pass.  It does not create an environment or
run a policy trajectory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import p3_counterfactual_gate_fit_preflight_20261001 as readiness_mod


SCHEMA = "p3_counterfactual_gate_adapter_fit_20261001_v1"
STAGE = "P3_COUNTERFACTUAL_GATE_ADAPTER_FIT_20261001_V1"
PROTOCOL_PATH = Path("docs/evidence/v4_2/p3_counterfactual_gate_adapter_fit_protocol_20261001.md")
TEST_PATH = Path("tests/test_p3_counterfactual_gate_adapter_20261001.py")
TARGETS_SHA256 = "4085299501758c55c476fac9676092ace9d9c69f4ea5d3a68b74f786eaaf1473"
READINESS_SCHEMA = "p3_counterfactual_gate_fit_preflight_20261001_v1"
ANCHOR_MANIFEST_SHA256 = "1129676804003dd5a569a7723015ef9ceddd8c1cc016cb1ffca02c3cd4318d35"
ANCHOR_SOURCE_FULL_SHA256 = "30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195"
EVENT_FULL_SHA256 = "fb89f62339755f7ebcd2bb0f9672438000fe8b754067cd2cdb28bf8407feb3e6"
BEFORE_CHECKPOINT_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
BEFORE_MODEL_DIGEST = "3434378719efdb02bf3d41febb52e66797a0d3928756fc0cbbfde056ddd337b0"
OBSERVATION_DIM = 88
RELEASE_INDEX = 1
FIT_SEED = 9303501
LEARNING_RATE = 1e-4
MAX_UPDATES = 80
ANCHOR_CHUNK_SIZE = 512
KL_CAP = 0.001
KL_TOLERANCE = 1e-10
TARGET_P_TOLERANCE = 1e-6


class FitError(ValueError):
    """Raised when a frozen fit input or intrinsic check is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FitError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> tuple[dict[str, Any], str]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    data = target.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FitError(f"invalid JSON input: {target}: {exc}") from exc
    _require(isinstance(payload, dict), f"JSON object required: {target}")
    return dict(payload), hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any], *, refuse_existing: bool = True) -> None:
    if refuse_existing and path.exists():
        raise FitError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_file(stored: str | Path, *, bases: Sequence[Path] = ()) -> Path:
    raw = Path(stored)
    candidates = [raw]
    normalized = str(stored).replace("\\", "/")
    prefix = "/home/user/training_outputs/terminal/"
    if normalized.startswith(prefix):
        mirror_root = Path(__file__).resolve().parents[3] / "training_outputs" / "terminal"
        candidates.append(mirror_root / normalized[len(prefix):])
    for base in bases:
        candidates.extend((base / raw, base / raw.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FitError(f"file missing: {stored}")


def _resolve_output_input(path: str | Path) -> Path:
    value = Path(path)
    if value.is_file() or value.is_dir():
        return value
    return _resolve_file(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _snapshot_rng() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        snapshot["torch_cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return snapshot


def _restore_rng(snapshot: Mapping[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if torch.cuda.is_available() and "torch_cuda" in snapshot:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    ln = left["numpy"]
    rn = right["numpy"]
    if ln[0] != rn[0] or not np.array_equal(ln[1], rn[1]) or ln[2:] != rn[2:]:
        return False
    if not torch.equal(left["torch"], right["torch"]):
        return False
    if "torch_cuda" in left or "torch_cuda" in right:
        if len(left.get("torch_cuda", ())) != len(right.get("torch_cuda", ())):
            return False
        if any(not torch.equal(a, b) for a, b in zip(left.get("torch_cuda", ()), right.get("torch_cuda", ()))):
            return False
    return True


def _binary_kl_from_logits(candidate_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    """Stable Bernoulli KL(candidate || reference), with no probability floor."""
    candidate = candidate_logits.to(dtype=torch.float64)
    reference = reference_logits.to(dtype=torch.float64)
    probability = torch.sigmoid(candidate)
    return (
        probability * (F.logsigmoid(candidate) - F.logsigmoid(reference))
        + (1.0 - probability)
        * (F.logsigmoid(-candidate) - F.logsigmoid(-reference))
    )


def _binary_bce_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits.to(dtype=torch.float64), target.to(dtype=torch.float64), reduction="mean"
    )


def _mean_chunked(values: torch.Tensor, chunk_size: int = ANCHOR_CHUNK_SIZE) -> torch.Tensor:
    _require(values.ndim == 1 and values.numel() > 0, "chunked mean requires non-empty vector")
    pieces = []
    total = int(values.numel())
    for start in range(0, total, int(chunk_size)):
        chunk = values[start : start + int(chunk_size)]
        pieces.append(chunk.mean() * (chunk.numel() / total))
    return torch.stack(pieces).sum()


def _gate_and_proposal_logits(model: nn.Module, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gate, proposal = model.distributions(observations)
    gate_logits = gate.logits
    proposal_logits = proposal.logits
    _require(gate_logits.ndim == 2 and gate_logits.shape[1] == 2, "gate distribution must have two logits")
    _require(proposal_logits.ndim == 2 and proposal_logits.shape[1] == 25, "proposal distribution must have 25 logits")
    return gate_logits[:, RELEASE_INDEX] - gate_logits[:, 0], proposal_logits


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _restore_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict({name: value.detach().clone() for name, value in state.items()}, strict=True)


def _parameter_scope(model: nn.Module) -> tuple[list[nn.Parameter], tuple[str, ...]]:
    trainable: list[nn.Parameter] = []
    names: list[str] = []
    for name, parameter in model.named_parameters():
        allowed = name == "gate_adapter" or name.startswith("gate_adapter.")
        parameter.requires_grad_(allowed)
        if allowed:
            trainable.append(parameter)
            names.append(name)
    _require(trainable and names, "gate_adapter parameters are missing")
    return trainable, tuple(names)


def _load_source_model(path: str | Path) -> tuple[nn.Module, dict[str, Any], dict[str, Any], str]:
    from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
        CALIBRATION_TEMPERATURE,
        TemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
    )

    source = _resolve_file(path)
    digest = _sha256(source)
    _require(digest == BEFORE_CHECKPOINT_SHA256, f"BEFORE checkpoint SHA mismatch: {digest}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "source checkpoint must be a mapping")
    model_state = payload.get("model")
    _require(isinstance(model_state, Mapping), "source checkpoint model state is missing")
    model = _instantiate_model_state(dict(model_state))
    model.eval()
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {}
    report = dict(payload.get("report", {})) if isinstance(payload.get("report"), Mapping) else {}
    _require(int(metadata.get("observation_dimension", OBSERVATION_DIM)) == OBSERVATION_DIM,
             "source observation dimension mismatch")
    _require(metadata.get("architecture_id") == "v4_2_hierarchical_gate_temperature_calibrated_residual_adapter_v3",
             "source calibrated architecture binding mismatch")
    _require(float(metadata.get("temperature", CALIBRATION_TEMPERATURE)) == CALIBRATION_TEMPERATURE,
             "source calibration temperature mismatch")
    _require(_state_digest(model.state_dict()) == BEFORE_MODEL_DIGEST,
             "source model digest mismatch")
    return model, metadata, report, digest


def _instantiate_model_state(model_state: Mapping[str, torch.Tensor]) -> nn.Module:
    from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
        CALIBRATION_TEMPERATURE,
        TemperatureCalibratedGateResidualAdapterActorCentralCriticV42,
    )

    model = TemperatureCalibratedGateResidualAdapterActorCentralCriticV42(
        observation_dim=OBSERVATION_DIM, temperature=CALIBRATION_TEMPERATURE
    )
    model.load_state_dict(dict(model_state), strict=True)
    model.eval()
    return model


def _load_anchor_data(readiness: Mapping[str, Any], *, bases: Sequence[Path] = ()) -> tuple[torch.Tensor, dict[int, int], dict[int, dict[str, Any]]]:
    anchors = readiness.get("anchors")
    _require(isinstance(anchors, Mapping), "readiness anchor section is missing")
    cells = anchors.get("cells")
    _require(isinstance(cells, Mapping), "readiness anchor cells are missing")
    worlds: list[torch.Tensor] = []
    target_indices: dict[int, int] = {}
    reports: dict[int, dict[str, Any]] = {}
    offset = 0
    for cell_id in (1, 2, 3, 4):
        cell = cells.get(str(cell_id), cells.get(cell_id))
        _require(isinstance(cell, Mapping), f"readiness cell {cell_id} missing")
        npz_path = _resolve_file(cell.get("npz_path"), bases=bases)
        arrays = readiness_mod._read_anchor_npz(npz_path)
        world = torch.from_numpy(np.asarray(arrays["world"], dtype=np.float32)).clone()
        worlds.append(world)
        reports[cell_id] = {
            "cell_id": cell_id,
            "npz_path": str(npz_path),
            "npz_sha256": _sha256(npz_path),
            "ticks": int(cell["ticks"]),
            "rows": int(cell["rows"]),
        }
        row = anchors.get("target_rows", {}).get(str(cell_id), anchors.get("target_rows", {}).get(cell_id))
        _require(isinstance(row, Mapping), f"readiness target row {cell_id} missing")
        target_indices[cell_id] = offset + int(row["world_row_index"])
        offset += int(world.shape[0])
    combined = torch.cat(worlds, dim=0)
    _require(int(combined.shape[1]) == OBSERVATION_DIM, "anchor observation dimension mismatch")
    _require(int(combined.shape[0]) == int(anchors.get("total_rows", -1)), "anchor row total mismatch")
    return combined, target_indices, reports


def _load_frozen_inputs(
    targets_path: str | Path,
    readiness_path: str | Path,
    anchors_path: str | Path | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], torch.Tensor, dict[int, int], dict[int, dict[str, Any]], str, str]:
    targets_file = _resolve_file(targets_path)
    targets, target_binding = readiness_mod._validate_targets(targets_file)
    _require(target_binding["sha256"] == TARGETS_SHA256, "targets binding mismatch")
    readiness_file = _resolve_file(readiness_path)
    readiness, readiness_sha = _read_json(readiness_file)
    _require(readiness.get("schema") == READINESS_SCHEMA, "fit readiness schema mismatch")
    _require(readiness.get("status") == "READY_FOR_ROOT_REVIEW", "fit readiness is not ready")
    _require(readiness.get("targets", {}).get("sha256") == TARGETS_SHA256, "readiness target SHA mismatch")
    anchor_info = readiness.get("anchors")
    _require(isinstance(anchor_info, Mapping), "fit readiness anchors are missing")
    _require(anchor_info.get("manifest_sha256") == ANCHOR_MANIFEST_SHA256,
             "anchor manifest binding mismatch")
    _require(anchor_info.get("source_full_sha256") == ANCHOR_SOURCE_FULL_SHA256,
             "anchor source full binding mismatch")
    _require(anchor_info.get("event_full_sha256") == EVENT_FULL_SHA256,
             "anchor event full binding mismatch")
    manifest_path = _resolve_file(anchor_info.get("manifest_path"))
    _require(_sha256(manifest_path) == ANCHOR_MANIFEST_SHA256,
             "anchor manifest changed after preflight")
    if anchors_path is not None:
        requested_anchor = Path(anchors_path)
        if not requested_anchor.exists():
            requested_anchor = _resolve_output_input(anchors_path)
        manifest_path = requested_anchor / "full.json" if requested_anchor.is_dir() else requested_anchor
        _require(_sha256(manifest_path) == ANCHOR_MANIFEST_SHA256, "provided anchor manifest SHA mismatch")
    world, target_indices, anchor_reports = _load_anchor_data(readiness)
    for cell_id, report in anchor_reports.items():
        declared = anchor_info["cells"][str(cell_id)]
        _require(report["npz_sha256"] == declared.get("npz_sha256"),
                 f"anchor cell {cell_id} NPZ changed after preflight")
    return targets, readiness, world, target_indices, anchor_reports, readiness_sha, target_binding["sha256"]


def _preflight_payload(
    targets_path: str | Path,
    readiness_path: str | Path,
    anchors_path: str | Path | None,
    source_checkpoint: str | Path,
) -> dict[str, Any]:
    """Validate all static bindings without loading torch or constructing a model."""
    targets_file = _resolve_file(targets_path)
    readiness_file = _resolve_file(readiness_path)
    targets, target_binding = readiness_mod._validate_targets(targets_file)
    readiness, readiness_sha = _read_json(readiness_file)
    _require(readiness.get("schema") == READINESS_SCHEMA, "fit readiness schema mismatch")
    _require(readiness.get("status") == "READY_FOR_ROOT_REVIEW", "fit readiness is not ready")
    _require(readiness.get("targets", {}).get("sha256") == TARGETS_SHA256, "readiness target SHA mismatch")
    _require(readiness.get("anchors", {}).get("manifest_sha256") == ANCHOR_MANIFEST_SHA256,
             "readiness anchor manifest SHA mismatch")
    if anchors_path is not None:
        requested_anchor = Path(anchors_path)
        if not requested_anchor.exists():
            requested_anchor = _resolve_output_input(anchors_path)
        manifest_path = requested_anchor / "full.json" if requested_anchor.is_dir() else requested_anchor
        _require(_sha256(manifest_path) == ANCHOR_MANIFEST_SHA256, "provided anchor manifest SHA mismatch")
    # This calls only the existing JSON/NPZ validator; it does not construct a model.
    validated = readiness_mod.build_readiness(targets_file, anchors_path or readiness["anchors"]["manifest_path"])
    _require(validated.get("status") == "READY_FOR_ROOT_REVIEW", "anchor validation failed")
    source_file = _resolve_file(source_checkpoint)
    source_sha = _sha256(source_file)
    _require(source_sha == BEFORE_CHECKPOINT_SHA256, f"source checkpoint SHA mismatch: {source_sha}")
    protocol_file = Path(__file__).resolve().parents[2] / PROTOCOL_PATH
    _require(protocol_file.is_file(), f"fit protocol missing: {protocol_file}")
    test_file = Path(__file__).resolve().parents[2] / TEST_PATH
    _require(test_file.is_file(), f"fit test file missing: {test_file}")
    runner_file = Path(__file__).resolve()
    return {
        "schema": SCHEMA,
        "stage": STAGE,
        "phase": "preflight",
        "status": "FIT_PREFLIGHT_OK",
        "targets": {"path": str(targets_file), "sha256": target_binding["sha256"], "schema": target_binding["schema"]},
        "readiness": {"path": str(readiness_file), "sha256": readiness_sha, "schema": readiness["schema"]},
        "anchors": {
            "manifest_path": readiness["anchors"]["manifest_path"],
            "manifest_sha256": ANCHOR_MANIFEST_SHA256,
            "source_full_sha256": ANCHOR_SOURCE_FULL_SHA256,
            "event_full_sha256": EVENT_FULL_SHA256,
            "total_ticks": int(readiness["anchors"]["total_ticks"]),
            "total_rows": int(readiness["anchors"]["total_rows"]),
        },
        "source_checkpoint": {"path": str(source_file), "sha256": source_sha, "model_digest": BEFORE_MODEL_DIGEST},
        "scope": {
            "fit_seed": FIT_SEED,
            "learning_rate": LEARNING_RATE,
            "max_updates": MAX_UPDATES,
            "anchor_chunk_size": ANCHOR_CHUNK_SIZE,
            "kl_cap": KL_CAP,
            "trainable_scope": "gate_adapter.* only",
            "critic_frozen": True,
            "proposal_and_temperature_frozen": True,
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "PPO_ALLOWED": False,
            "heldout_used": False,
            "teacher_used": False,
            "environment_created": False,
            "model_forward_used": False,
            "optimizer_created": False,
        },
        "code_bindings": {
            "runner": {"path": str(runner_file), "sha256": _sha256(runner_file)},
            "test": {"path": str(test_file), "sha256": _sha256(test_file)},
            "protocol": {"path": str(protocol_file), "sha256": _sha256(protocol_file)},
        },
        "target_cells": sorted(targets),
        "source_metadata_not_restored": ["optimizer", "report", "training_history", "admission_flags"],
    }


def build_preflight(
    *,
    targets_path: str | Path,
    readiness_path: str | Path,
    anchors_path: str | Path | None,
    source_checkpoint: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists() or output.is_dir(), "fit output path is not a directory")
    preflight_path = output / "preflight.json"
    _require(not preflight_path.exists(), f"refusing to overwrite existing output: {preflight_path}")
    for name in ("candidate.pt", "fit.json", "error.json", "progress.json", "partial.json"):
        _require(not (output / name).exists(), f"refusing to overwrite existing output: {output / name}")
    payload = _preflight_payload(targets_path, readiness_path, anchors_path, source_checkpoint)
    _write_json(preflight_path, payload)
    return payload


def _target_rows(
    targets: Mapping[int, Mapping[str, Any]],
    world: torch.Tensor,
    indices: Mapping[int, int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    ids = [1, 2, 3, 4]
    rows = torch.stack([world[int(indices[cell_id])] for cell_id in ids], dim=0)
    pstar = torch.tensor([float(targets[cell_id]["pstar"]) for cell_id in ids], dtype=torch.float64)
    p0_recorded = torch.tensor([float(targets[cell_id]["p0"]) for cell_id in ids], dtype=torch.float64)
    return rows, torch.stack((pstar, p0_recorded), dim=1), ids


def _checkpoint_metadata(source_metadata: Mapping[str, Any], *, source_sha: str) -> dict[str, Any]:
    volatile = {
        "stage", "source_checkpoint", "source_checkpoint_sha256", "training_called",
        "supervised_training_called", "teacher_called", "ppo_called", "PPO_ALLOWED",
        "checkpoint_promotable", "diagnostic_only", "heldout_consumed", "optimizer_restored",
        "optimizer_saved", "training_seed", "batch_size", "learning_rate", "gradient_clip_norm",
        "train_rows", "checkpoint_selection", "early_stopping", "terminal_credit_selection",
        "terminal_credit_summary", "source_manifest", "anchor_path", "max_updates",
        "training_scope", "bounded_backtracking", "normal_advantage", "objective",
        "teacher_gate_used_in_loss", "smoke", "report", "history",
    }
    retained = {key: copy.deepcopy(value) for key, value in source_metadata.items() if key not in volatile}
    retained.update({
        "stage": STAGE,
        "candidate_label": "P3_COUNTERFACTUAL_GATE_ADAPTER_CANDIDATE",
        "parent_checkpoint_role": "BEFORE3fe",
        "architecture_id": "v4_2_hierarchical_gate_temperature_calibrated_residual_adapter_v3",
        "temperature": 2.5,
        "source_checkpoint_sha256": source_sha,
        "source_model_digest": BEFORE_MODEL_DIGEST,
        "parent_checkpoint_sha256": source_sha,
        "candidate_promoted": False,
        "checkpoint_promotable": False,
        "diagnostic_only": True,
        "PPO_ALLOWED": False,
        "teacher_used": False,
        "heldout_used": False,
        "optimizer_restored": False,
        "optimizer_saved": False,
        "training_called": True,
        "supervised_training_called": True,
        "teacher_called": False,
        "ppo_called": False,
        "diagnostic_only": True,
        "fit_seed": FIT_SEED,
        "fit_trainable_scope": "gate_adapter.* only",
        "fit_optimizer": "Adam",
        "fit_learning_rate": LEARNING_RATE,
        "fit_weight_decay": 0.0,
        "optimizer_called": True,
        "weights_changed": True,
        "source_metadata_retained": True,
        "source_training_metadata_not_restored": True,
    })
    return retained


def _source_checkpoint_payload(
    source_path: str | Path, source_sha: str, model: nn.Module, report: Mapping[str, Any],
) -> dict[str, Any]:
    source = _resolve_file(source_path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "source checkpoint must be a mapping")
    source_metadata = payload.get("metadata", {})
    _require(isinstance(source_metadata, Mapping), "source checkpoint metadata is missing")
    return {
        "metadata": _checkpoint_metadata(source_metadata, source_sha=source_sha),
        "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "report": dict(report),
    }


def _compare_state(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _state_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(value.detach()).all()) for value in model.state_dict().values())


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _compute_reference_outputs(model: nn.Module, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gate_parts: list[torch.Tensor] = []
    proposal_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(observations.shape[0]), ANCHOR_CHUNK_SIZE):
            gate_logits, proposal_logits = _gate_and_proposal_logits(
                model, observations[start : start + ANCHOR_CHUNK_SIZE]
            )
            gate_parts.append(gate_logits.detach().clone())
            proposal_parts.append(proposal_logits.detach().clone())
    return torch.cat(gate_parts), torch.cat(proposal_parts)


def _evaluate_outputs(model: nn.Module, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gate_logits, proposal_logits = _gate_and_proposal_logits(model, observations)
    return gate_logits, proposal_logits


def _evaluate_kl(
    model: nn.Module,
    observations: torch.Tensor,
    reference_gate_logits: torch.Tensor,
    *,
    chunk_size: int = ANCHOR_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks: list[torch.Tensor] = []
    q_chunks: list[torch.Tensor] = []
    for start in range(0, int(observations.shape[0]), chunk_size):
        with torch.no_grad():
            gate, proposal = _evaluate_outputs(model, observations[start : start + chunk_size])
            ref = reference_gate_logits[start : start + chunk_size]
            chunks.append(_binary_kl_from_logits(gate, ref).detach())
            q_chunks.append(proposal.detach())
    return torch.cat(chunks), torch.cat(q_chunks)


def _fit_core(
    *,
    model: nn.Module,
    source_metadata: Mapping[str, Any],
    source_sha: str,
    source_payload_report: Mapping[str, Any],
    anchor_world: torch.Tensor,
    target_world: torch.Tensor,
    target_values: torch.Tensor,
    target_ids: Sequence[int],
    target_p0_recorded: torch.Tensor,
    source_checkpoint_path: str | Path,
    output_dir: Path,
    report_bindings: Mapping[str, Any] | None = None,
    caller_rng_snapshot: Mapping[str, Any] | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
) -> dict[str, Any]:
    del model_factory  # Kept as an explicit seam for test stubs; model is already loaded here.
    caller_rng = caller_rng_snapshot if caller_rng_snapshot is not None else _snapshot_rng()
    old_threads = torch.get_num_threads()
    fit_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate.pt"
    report_path = output_dir / "fit.json"
    error_path = output_dir / "error.json"
    progress_path = output_dir / "progress.json"
    partial_path = output_dir / "partial.json"
    _require(not candidate_path.exists() and not report_path.exists() and not error_path.exists()
             and not progress_path.exists() and not partial_path.exists(),
             "fit output already exists; refusing restart or overwrite")
    try:
        torch.set_num_threads(1)
        random.seed(FIT_SEED)
        np.random.seed(FIT_SEED)
        torch.manual_seed(FIT_SEED)
        model.eval()
        _require(_state_finite(model), "non-finite source model state")
        source_state = _clone_state(model)
        trainable, trainable_names = _parameter_scope(model)
        frozen_names = tuple(name for name in source_state if not (name == "gate_adapter" or name.startswith("gate_adapter.")))
        frozen_before = {name: source_state[name].clone() for name in frozen_names}
        anchor_reference_gate, anchor_reference_q = _compute_reference_outputs(model, anchor_world)
        target_reference_gate, target_reference_q = _compute_reference_outputs(model, target_world)
        initial_p = torch.sigmoid(target_reference_gate.to(dtype=torch.float64)).detach()
        target_values = target_values.to(dtype=torch.float64)
        target_p0_recorded = target_p0_recorded.to(dtype=torch.float64)
        _require(torch.all(torch.isfinite(initial_p)), "initial target probabilities are non-finite")
        _require(torch.all(torch.abs(initial_p - target_p0_recorded) <= TARGET_P_TOLERANCE),
                 "initial target p0 does not match frozen source")
        initial_anchor_kl = _evaluate_kl(model, anchor_world, anchor_reference_gate)[0]
        _require(float(initial_anchor_kl.max()) <= KL_CAP + KL_TOLERANCE,
                 "source anchor KL is outside the registered cap")
        initial_target_bce = _binary_bce_from_logits(target_reference_gate, target_values)
        history: list[dict[str, Any]] = []
        accepted = 0
        rejected = 0
        stop_reason = "MAX_UPDATES"
        optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE, weight_decay=0.0)
        _write_progress(progress_path, {
            "schema": SCHEMA, "stage": STAGE, "phase": "fit", "status": "RUNNING",
            "update": 0, "accepted_updates": 0, "rejected_updates": 0,
            "anchor_rows": int(anchor_world.shape[0]), "target_rows": int(target_world.shape[0]),
        })
        for update in range(1, MAX_UPDATES + 1):
            optimizer.zero_grad(set_to_none=True)
            target_gate, _ = _evaluate_outputs(model, target_world)
            target_loss = _binary_bce_from_logits(target_gate, target_values)
            _require(bool(torch.isfinite(target_loss)), f"non-finite target loss at update {update}")
            target_loss.backward()
            anchor_loss_value = 0.0
            for start in range(0, int(anchor_world.shape[0]), ANCHOR_CHUNK_SIZE):
                gate, _ = _evaluate_outputs(model, anchor_world[start : start + ANCHOR_CHUNK_SIZE])
                reference = anchor_reference_gate[start : start + ANCHOR_CHUNK_SIZE]
                chunk_kl = _binary_kl_from_logits(gate, reference)
                _require(bool(torch.isfinite(chunk_kl).all()), f"non-finite anchor KL at update {update}")
                weighted_chunk = chunk_kl.mean() * (chunk_kl.numel() / anchor_world.shape[0])
                _require(bool(torch.isfinite(weighted_chunk)), f"non-finite anchor loss at update {update}")
                weighted_chunk.backward()
                anchor_loss_value += float(weighted_chunk.detach())
            loss_value = float(target_loss.detach()) + anchor_loss_value
            _require(math.isfinite(loss_value), f"non-finite total loss at update {update}")
            before_step = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if name in trainable_names}
            for name, parameter in model.named_parameters():
                if name in trainable_names and (parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())):
                    raise FitError(f"non-finite or missing adapter gradient: {name}")
            optimizer.step()
            _require(_state_finite(model), f"non-finite model state at update {update}")
            candidate_kl, candidate_q = _evaluate_kl(model, anchor_world, anchor_reference_gate)
            with torch.no_grad():
                target_after, candidate_target_q = _evaluate_outputs(model, target_world)
                target_after_kl = _binary_kl_from_logits(target_after, target_reference_gate)
            _require(bool(torch.isfinite(candidate_kl).all()) and bool(torch.isfinite(target_after_kl).all()),
                     f"non-finite KL after update {update}")
            _require(torch.equal(candidate_q.detach().cpu(), anchor_reference_q.detach().cpu()),
                     f"proposal q changed at update {update}")
            _require(torch.equal(candidate_target_q.detach().cpu(), target_reference_q.detach().cpu()),
                     f"target proposal q changed at update {update}")
            _require(all(torch.equal(model.state_dict()[name].detach().cpu(), value)
                         for name, value in frozen_before.items()),
                     f"frozen state changed at update {update}")
            max_kl = float(candidate_kl.max().item())
            max_target_kl = float(target_after_kl.max().item())
            if max_kl > KL_CAP + KL_TOLERANCE or max_target_kl > KL_CAP + KL_TOLERANCE:
                for name, parameter in model.named_parameters():
                    if name in before_step:
                        parameter.data.copy_(before_step[name].to(parameter.device, dtype=parameter.dtype))
                rejected += 1
                stop_reason = "ROLLBACK_STOP"
                history.append({"update": update, "loss": loss_value, "anchor_kl": max_kl,
                                "target_kl": max_target_kl, "accepted": False, "rollback": True})
                _write_progress(progress_path, {
                    "schema": SCHEMA, "stage": STAGE, "phase": "fit", "status": "ROLLBACK_STOP",
                    "update": update, "accepted_updates": accepted, "rejected_updates": rejected,
                    "anchor_kl": max_kl, "target_kl": max_target_kl,
                })
                break
            accepted += 1
            history.append({"update": update, "loss": loss_value, "target_bce": float(target_loss.detach()),
                            "anchor_kl": max_kl, "target_kl": max_target_kl, "accepted": True, "rollback": False})
            _write_progress(progress_path, {
                "schema": SCHEMA, "stage": STAGE, "phase": "fit", "status": "RUNNING",
                "update": update, "accepted_updates": accepted, "rejected_updates": rejected,
                "anchor_kl": max_kl, "target_kl": max_target_kl,
            })
            del candidate_q, candidate_target_q

        final_gate, final_q = _compute_reference_outputs(model, target_world)
        final_anchor_kl, final_anchor_q = _evaluate_kl(model, anchor_world, anchor_reference_gate)
        final_target_kl = _binary_kl_from_logits(final_gate, target_reference_gate).detach()
        _require(_state_finite(model), "non-finite final model state")
        _require(bool(torch.isfinite(final_anchor_kl).all()) and bool(torch.isfinite(final_target_kl).all()),
                 "non-finite final KL")
        _require(_sha256(_resolve_file(source_checkpoint_path)) == source_sha,
                 "source checkpoint changed during fit")
        final_p = torch.sigmoid(final_gate.to(dtype=torch.float64)).detach()
        final_bce = float(_binary_bce_from_logits(final_gate, target_values).item())
        initial_bce = float(initial_target_bce.item())
        frozen_unchanged = all(torch.equal(model.state_dict()[name].detach().cpu(), value) for name, value in frozen_before.items())
        q_unchanged = torch.equal(final_q.detach().cpu(), target_reference_q.detach().cpu()) and torch.equal(final_anchor_q.detach().cpu(), anchor_reference_q.detach().cpu())
        source_state_unchanged_except_adapter = all(
            torch.equal(model.state_dict()[name].detach().cpu(), value) for name, value in frozen_before.items()
        )
        target_progress = bool(torch.all(torch.abs(final_p - target_values) <= torch.abs(initial_p - target_values) + 1e-6))
        intrinsic_pass = (
            final_bce < initial_bce - 1e-10
            and target_progress
            and float(final_anchor_kl.max()) <= KL_CAP + KL_TOLERANCE
            and float(final_target_kl.max()) <= KL_CAP + KL_TOLERANCE
            and frozen_unchanged and q_unchanged and source_state_unchanged_except_adapter
        )
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "stage": STAGE,
            "phase": "fit",
            "status": "FIT_OK" if intrinsic_pass else "FIT_SCREEN_FAIL",
            "source_checkpoint_sha256": source_sha,
            "source_model_digest": _state_digest(source_state),
            "target_ids": list(target_ids),
            "anchor_rows": int(anchor_world.shape[0]),
            "target_rows": int(target_world.shape[0]),
            "accepted_updates": accepted,
            "rejected_updates": rejected,
            "max_updates": MAX_UPDATES,
            "stop_reason": stop_reason,
            "initial_target_bce": initial_bce,
            "final_target_bce": final_bce,
            "initial_target_p": [float(value) for value in initial_p],
            "target_pstar": [float(value) for value in target_values],
            "final_target_p": [float(value) for value in final_p],
            "target_delta_p": [float(value) for value in (final_p - initial_p)],
            "max_anchor_kl": float(final_anchor_kl.max().item()),
            "max_target_kl": float(final_target_kl.max().item()),
            "intrinsic_screen": {
                "target_bce_decreased": final_bce < initial_bce - 1e-10,
                "each_target_progressed": target_progress,
                "anchor_kl_within_cap": float(final_anchor_kl.max()) <= KL_CAP + KL_TOLERANCE,
                "target_kl_within_cap": float(final_target_kl.max()) <= KL_CAP + KL_TOLERANCE,
                "frozen_state_unchanged": frozen_unchanged,
                "proposal_q_unchanged": q_unchanged,
                "source_state_unchanged_except_adapter": source_state_unchanged_except_adapter,
            },
            "trainable_scope": list(trainable_names),
            "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": 0.0},
            "optimizer_called": True,
            "weights_changed": any(
                not torch.equal(model.state_dict()[name].detach().cpu(), value)
                for name, value in source_state.items()
                if name == "gate_adapter" or name.startswith("gate_adapter.")
            ),
            "scope": {"PPO_ALLOWED": False, "teacher_used": False, "heldout_used": False,
                      "environment_created": False, "source_optimizer_restored": False},
            "history": history,
            "elapsed_seconds": time.perf_counter() - fit_started,
            "candidate_checkpoint": None,
            "caller_rng_unchanged": None,
        }
        if report_bindings:
            report.update(_jsonable(dict(report_bindings)))
        if intrinsic_pass:
            payload = _source_checkpoint_payload(source_checkpoint_path, source_sha, model, report)
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            serialized_candidate = buffer.getvalue()
            reload_payload = torch.load(io.BytesIO(serialized_candidate), map_location="cpu", weights_only=False)
            reloaded_model = _instantiate_model_state(reload_payload["model"])
            _require(_compare_state(reloaded_model.state_dict(), payload["model"]), "candidate strict reload state mismatch")
            reload_target_gate, reload_target_q = _compute_reference_outputs(reloaded_model, target_world)
            reload_anchor_gate, reload_anchor_q = _compute_reference_outputs(reloaded_model, anchor_world)
            reload_metadata = reload_payload.get("metadata", {})
            _require(reload_metadata.get("stage") == STAGE, "candidate metadata stage mismatch")
            _require(reload_metadata.get("parent_checkpoint_sha256") == source_sha,
                     "candidate metadata parent SHA mismatch")
            _require(reload_metadata.get("PPO_ALLOWED") is False and reload_metadata.get("candidate_promoted") is False,
                     "candidate metadata admission flags mismatch")
            _require(torch.equal(reload_target_gate, final_gate), "candidate target logits changed on reload")
            _require(torch.equal(torch.sigmoid(reload_target_gate.to(dtype=torch.float64)), final_p),
                     "candidate target probabilities changed on reload")
            _require(torch.equal(reload_target_q, final_q), "candidate target proposal q changed on reload")
            _require(torch.equal(reload_anchor_q, final_anchor_q), "candidate anchor proposal q changed on reload")
            _require(torch.equal(reload_anchor_gate, _compute_reference_outputs(model, anchor_world)[0]),
                     "candidate anchor gate logits changed on reload")
            _require(not candidate_path.exists(), f"refusing to overwrite existing output: {candidate_path}")
            with candidate_path.open("xb") as handle:
                handle.write(serialized_candidate)
            report["candidate_checkpoint"] = {
                "path": str(candidate_path),
                "sha256": _sha256(candidate_path),
                "model_digest": _state_digest(reloaded_model.state_dict()),
                "strict_reload": True,
                "optimizer_state_saved": "optimizer" in reload_payload,
            }
            _require(report["candidate_checkpoint"]["optimizer_state_saved"] is False,
                     "candidate checkpoint contains source optimizer state")
        _restore_rng(caller_rng)
        report["caller_rng_unchanged"] = _rng_equal(caller_rng, _snapshot_rng())
        _require(report["caller_rng_unchanged"] is True, "caller RNG was not restored")
        _write_progress(progress_path, {
            "schema": SCHEMA, "stage": STAGE, "phase": "fit", "status": report["status"],
            "update": accepted + rejected, "accepted_updates": accepted,
            "rejected_updates": rejected, "stop_reason": stop_reason,
        })
        _write_json(report_path, _jsonable(report))
        return report
    finally:
        _restore_rng(caller_rng)
        try:
            torch.set_num_threads(old_threads)
        except RuntimeError:
            pass


def run_fit(
    *,
    targets_path: str | Path,
    readiness_path: str | Path,
    anchors_path: str | Path | None,
    source_checkpoint: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    preflight_path = output / "preflight.json"
    _require(preflight_path.is_file(), f"fit preflight missing: {preflight_path}")
    for name in ("candidate.pt", "fit.json", "error.json", "progress.json", "partial.json"):
        _require(not (output / name).exists(), f"fit output already exists: {output / name}")
    preflight_payload, preflight_sha = _read_json(preflight_path)
    _require(preflight_payload.get("status") == "FIT_PREFLIGHT_OK", "fit preflight is not FIT_PREFLIGHT_OK")
    targets, readiness, anchor_world, target_indices, anchor_reports, readiness_sha, target_sha = _load_frozen_inputs(
        targets_path, readiness_path, anchors_path
    )
    _require(preflight_payload.get("targets", {}).get("sha256") == target_sha, "preflight target binding changed")
    _require(preflight_payload.get("readiness", {}).get("sha256") == readiness_sha, "preflight readiness binding changed")
    source_file = _resolve_file(source_checkpoint)
    source_sha = _sha256(source_file)
    _require(source_sha == BEFORE_CHECKPOINT_SHA256, "source checkpoint changed")
    _require(preflight_payload.get("source_checkpoint", {}).get("sha256") == source_sha,
             "preflight source checkpoint binding changed")
    scope = preflight_payload.get("scope", {})
    _require(scope.get("PPO_ALLOWED") is False and scope.get("critic_frozen") is True
             and scope.get("proposal_and_temperature_frozen") is True
             and scope.get("trainable_scope") == "gate_adapter.* only",
             "preflight fit scope binding changed")
    current_code = {
        "runner": _sha256(Path(__file__).resolve()),
        "test": _sha256(Path(__file__).resolve().parents[2] / TEST_PATH),
        "protocol": _sha256(Path(__file__).resolve().parents[2] / PROTOCOL_PATH),
    }
    declared_code = preflight_payload.get("code_bindings", {})
    _require(all(declared_code.get(name, {}).get("sha256") == digest for name, digest in current_code.items()),
             "fit code binding changed after preflight")
    caller_rng = _snapshot_rng()
    try:
        random.seed(FIT_SEED)
        np.random.seed(FIT_SEED)
        torch.manual_seed(FIT_SEED)
        model, metadata, source_report, loaded_sha = _load_source_model(source_file)
        _require(loaded_sha == source_sha, "source load hash changed")
        target_world, target_pack, target_ids = _target_rows(targets, anchor_world, target_indices)
        pstar = target_pack[:, 0]
        p0 = target_pack[:, 1]
        return _fit_core(
            model=model,
            source_metadata=metadata,
            source_sha=source_sha,
            source_payload_report=source_report,
            anchor_world=anchor_world,
            target_world=target_world,
            target_values=pstar,
            target_ids=target_ids,
            target_p0_recorded=p0,
            source_checkpoint_path=source_file,
            output_dir=output,
            report_bindings={
                "preflight_sha256": preflight_sha,
                "readiness_sha256": readiness_sha,
                "target_sha256": target_sha,
                "anchor_cells": anchor_reports,
            },
            caller_rng_snapshot=caller_rng,
        )
    finally:
        _restore_rng(caller_rng)


def _default_path(relative: str) -> Path:
    if os.name == "nt":
        root = Path(__file__).resolve().parents[3] / "training_outputs" / "terminal"
    else:
        root = Path("/home/user/training_outputs/terminal")
    return root / relative


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "fit"), required=True)
    parser.add_argument("--targets", type=Path, default=_default_path("p3_followup_20260930/counterfactual_gate_targets_20261001/targets_v1.json"))
    parser.add_argument("--readiness", type=Path, default=_default_path("p3_followup_20260930/counterfactual_gate_targets_20261001/fit_readiness_v1.json"))
    parser.add_argument("--anchors", type=Path, default=_default_path("p3_followup_20260930/gate_anchor_states_20261001"))
    parser.add_argument("--source-checkpoint", type=Path, default=Path("/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.phase == "preflight":
            result = build_preflight(
                targets_path=args.targets, readiness_path=args.readiness, anchors_path=args.anchors,
                source_checkpoint=args.source_checkpoint, output_dir=args.output_dir,
            )
        else:
            result = run_fit(
                targets_path=args.targets, readiness_path=args.readiness, anchors_path=args.anchors,
                source_checkpoint=args.source_checkpoint, output_dir=args.output_dir,
            )
    except Exception as exc:
        error = {"schema": SCHEMA, "stage": STAGE, "phase": args.phase, "status": "ERROR",
                 "error_type": type(exc).__name__, "message": str(exc)}
        if args.phase == "fit":
            progress_payload = None
            progress_path = args.output_dir / "progress.json"
            if progress_path.is_file():
                try:
                    progress_payload, _ = _read_json(progress_path)
                except Exception:
                    progress_payload = None
            partial = {
                "schema": SCHEMA, "stage": STAGE, "phase": "fit", "status": "PARTIAL",
                "error_type": type(exc).__name__, "message": str(exc),
                "latest_progress": progress_payload,
            }
            try:
                _write_json(args.output_dir / "partial.json", partial, refuse_existing=True)
            except Exception:
                pass
        try:
            _write_json(args.output_dir / "error.json", error)
        except Exception:
            pass
        print(json.dumps(error, sort_keys=True))
        return 2
    print(json.dumps({"status": result.get("status"), "phase": args.phase,
                      "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
