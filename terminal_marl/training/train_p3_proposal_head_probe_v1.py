"""Single proposal-head offline probe for the registered P3 experiment.

The runner consumes a completed, exact state-only collection.  It loads the
fixed calibrated 03c0 policy, freezes every parameter except the 25-way
``proposal_head``, and performs the one registered fit.  It never creates an
environment, calls a Teacher, samples an action, or runs PPO.
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
from terminal_marl.core.checkpoints import checkpoint_metadata, load_checkpoint
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator
from terminal_marl.training import k3_gate_feasibility


STAGE = "P3_PROPOSAL_HEAD_PROBE_V1"
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
PROTOCOL = (
    Path(__file__).resolve().parents[2]
    / "docs/evidence/v4_2/p3_proposal_head_probe_v1_20260926.md"
)
TRAIN_SEEDS = frozenset({873101, 873102, 873103})
VALIDATION_SEEDS = frozenset({873104, 873105, 873106})
ALLOWED_SEEDS = TRAIN_SEEDS | VALIDATION_SEEDS
TRAINING_SEED = 890401
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 40
GRADIENT_CLIP_NORM = 1.0
KL_COEFFICIENT = 1.0
RETENTION_KL_MAX = 0.05
MAX_ROWS = 192


class ProposalHeadProbeInputError(ValueError):
    """Raised when the proposal probe input violates its registered contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProposalHeadProbeInputError(message)


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
                if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not torch.equal(a, b):
                    return False
            elif a != b:
                return False
    return True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProposalHeadProbeInputError(f"invalid JSON: {path}") from exc


EXPECTED_SOURCE_RECORDS = (
    (873101, "flow6", "train"),
    (873102, "flow12", "train"),
    (873103, "dense24", "train"),
    (873104, "flow6", "validation"),
    (873105, "flow12", "validation"),
    (873106, "dense24", "validation"),
)


def _collection_manifest(collection_dir: Path) -> tuple[Path, Mapping[str, Any]]:
    manifest_path = collection_dir / "manifest.json"
    _require(manifest_path.is_file(), f"collection manifest missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    _require(isinstance(manifest, Mapping), "collection manifest must be an object")
    _require(str(manifest.get("status", "")) == "COMPLETED",
             f"collection is not COMPLETED: {manifest.get('status')!r}")
    _require(str(manifest.get("mode", "")) == "full", "proposal collection must be full mode")
    _require(manifest.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
             "collection source checkpoint SHA mismatch")
    protocol = manifest.get("protocol")
    _require(isinstance(protocol, Mapping), "collection protocol binding is missing")
    protocol_sha = _sha256(PROTOCOL)
    _require(protocol.get("sha256") == protocol_sha, "collection protocol SHA mismatch")
    _require(protocol.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
             "collection protocol source checkpoint SHA mismatch")
    _require(manifest.get("source_model_digest_unchanged") is True,
             "collection source model digest was not unchanged")
    reports = manifest.get("reports")
    _require(isinstance(reports, list) and len(reports) == len(EXPECTED_SOURCE_RECORDS),
             "collection must contain six completed reports")
    observed_records: list[tuple[int, str, str]] = []
    for index, report in enumerate(reports):
        _require(isinstance(report, Mapping), f"collection report {index} must be an object")
        seed = report.get("seed")
        scenario = report.get("scenario")
        split = report.get("split")
        _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool),
                 f"collection report {index} seed is invalid")
        observed_records.append((int(seed), str(scenario), str(split)))
        for field in ("replay_exact", "outcomes_match", "rng_unchanged", "model_weights_unchanged"):
            _require(report.get(field) is True, f"collection report {index} {field} is not true")
        errors = report.get("errors", [])
        _require(isinstance(errors, list) and not errors, f"collection report {index} has errors")
    _require(tuple(observed_records) == EXPECTED_SOURCE_RECORDS,
             f"collection report seed/scenario binding changed: {observed_records}")
    return manifest_path, manifest


def _as_finite_vector(value: Any, *, length: int, name: str) -> list[float]:
    _require(isinstance(value, (list, tuple, np.ndarray)), f"{name} must be a vector")
    _require(len(value) == length, f"{name} must have length {length}")
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ProposalHeadProbeInputError(f"{name} is not numeric") from exc
    _require(array.shape == (length,) and bool(np.isfinite(array).all()), f"{name} is not finite")
    return array.tolist()


def _as_certified(value: Any) -> list[int]:
    _require(isinstance(value, (list, tuple, set, frozenset)), "certified_proposals must be a list")
    values: list[int] = []
    for item in value:
        _require(isinstance(item, (int, np.integer)) and not isinstance(item, bool),
                 "certified_proposals must contain integer indices")
        index = int(item)
        _require(0 <= index < PROPOSAL_COUNT, f"certified proposal out of range: {index}")
        _require(index not in values, f"certified proposal duplicated: {index}")
        values.append(index)
    return sorted(values)


def _row_identity(row: Mapping[str, Any], index: int) -> tuple[Any, ...]:
    uid = row.get("uid", row.get("runtime_uid"))
    arrival = row.get("arrival_id", row.get("arrival"))
    _require(uid is not None and str(uid) != "", f"row {index} uid is required")
    _require(arrival is not None, f"row {index} arrival_id is required")
    return (str(row["split"]), int(row["seed"]), int(row["step"]), str(uid), int(arrival))


def _normalize_rows(rows: Any) -> list[dict[str, Any]]:
    _require(isinstance(rows, list), "collection rows.json must contain a list")
    _require(bool(rows), "collection rows.json is empty")
    _require(len(rows) <= MAX_ROWS, f"collection contains more than {MAX_ROWS} rows")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    split_seeds: dict[str, set[int]] = {"train": set(), "validation": set()}
    for index, source in enumerate(rows):
        _require(isinstance(source, Mapping), f"row {index} must be an object")
        row = dict(source)
        split = row.get("split")
        _require(split in {"train", "validation"}, f"row {index} split is invalid")
        seed = row.get("seed")
        _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool),
                 f"row {index} seed is required")
        seed = int(seed)
        _require(seed in ALLOWED_SEEDS, f"row {index} seed is outside the registered development set")
        expected = TRAIN_SEEDS if split == "train" else VALIDATION_SEEDS
        _require(seed in expected, f"row {index} seed {seed} is assigned to the wrong split")
        step = row.get("step")
        _require(isinstance(step, (int, np.integer)) and not isinstance(step, bool) and int(step) >= 0,
                 f"row {index} step is invalid")
        _as_finite_vector(row.get("observation"), length=OBSERVATION_DIM, name=f"row {index} observation")
        row["source_proposal_logits"] = _as_finite_vector(
            row.get("source_proposal_logits"), length=PROPOSAL_COUNT,
            name=f"row {index} source_proposal_logits",
        )
        row["source_gate_logits"] = _as_finite_vector(
            row.get("source_gate_logits"), length=GATE_COUNT,
            name=f"row {index} source_gate_logits",
        )
        row["certified_proposals"] = _as_certified(row.get("certified_proposals", []))
        row["seed"] = seed
        row["step"] = int(step)
        identity = _row_identity(row, index)
        _require(identity not in identities, f"duplicate row identity: {identity}")
        identities.add(identity)
        split_seeds[str(split)].add(seed)
        normalized.append(row)
    _require(not (split_seeds["train"] & split_seeds["validation"]), "train/validation seed leakage")
    _require(any(row["split"] == "train" for row in normalized), "collection has no train rows")
    _require(any(row["split"] == "validation" for row in normalized), "collection has no validation rows")
    return normalized


def _load_collection(collection_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = Path(collection_dir)
    _require(directory.is_dir(), f"collection directory missing: {directory}")
    manifest_path, manifest = _collection_manifest(directory)
    rows_path = directory / "rows.json"
    _require(rows_path.is_file(), f"collection rows missing: {rows_path}")
    rows = _normalize_rows(_read_json(rows_path))
    report = {
        "collection_dir": str(directory),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "rows": str(rows_path),
        "rows_sha256": _sha256(rows_path),
        "row_count": len(rows),
    }
    return rows, report


def _build_execution_config() -> Any:
    # Use the same V7 formal protocol/config construction as the collector.
    protocol = evaluator._protocol()
    configs = evaluator._build_configs(protocol)
    config = configs["dense24"]
    _require(hasattr(config, "v4_2_rejected_release_hold_mode"),
             "execution config lacks v4_2_rejected_release_hold_mode")
    _require(str(config.v4_2_rejected_release_hold_mode) == "canonical_wait",
             "canonical_wait execution mode was not applied")
    _require(not bool(config.teacher_enabled), "proposal fit config must disable Teacher")
    _require(str(config.scheduler_mode) == "execution", "proposal fit config must use execution scheduler")
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
        OBSERVATION_DIM, temperature=CALIBRATION_TEMPERATURE,
    )
    try:
        payload = load_checkpoint(path, policy, config, load_optimizer=False)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise ProposalHeadProbeInputError(f"strict source checkpoint load failed: {exc}") from exc
    _require(getattr(policy.model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID,
             "source architecture id changed")
    _require(float(policy.model.gate_temperature.detach().cpu().item()) == CALIBRATION_TEMPERATURE,
             "source temperature changed")
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
    _require(names == ["proposal_head.weight", "proposal_head.bias"],
             f"unexpected proposal trainable scope: {names}")
    _require(sum(parameter.numel() for parameter in trainable) == 2425,
             "proposal trainable parameter count is not 2425")
    return trainable, names


def _observations(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    values = torch.as_tensor([row["observation"] for row in rows], dtype=torch.float32, device=device)
    _require(values.ndim == 2 and values.shape[1] == OBSERVATION_DIM, "observation tensor is not 88D")
    _require(bool(torch.isfinite(values).all().item()), "observation tensor is not finite")
    return values


def _source_logits(rows: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    proposal = torch.as_tensor([row["source_proposal_logits"] for row in rows], dtype=torch.float32, device=device)
    gate = torch.as_tensor([row["source_gate_logits"] for row in rows], dtype=torch.float32, device=device)
    _require(proposal.shape == (len(rows), PROPOSAL_COUNT), "source proposal logits shape mismatch")
    _require(gate.shape == (len(rows), GATE_COUNT), "source gate logits shape mismatch")
    return proposal, gate


def _validate_source_rows(policy: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    # The collector may have forwarded one actor state at a time.  Validate
    # row-wise so a later batched GEMM implementation cannot create a false
    # bitwise mismatch through a different accumulation order.
    device = policy.device
    for index, row in enumerate(rows):
        observations = _observations([row], device)
        expected_proposal, expected_gate = _source_logits([row], device)
        with torch.no_grad():
            gate, proposal = policy.model.distributions(observations)
        _require(torch.equal(proposal.logits[0], expected_proposal[0]),
                 f"collection source proposal logits do not match source checkpoint at row {index}")
        _require(torch.equal(gate.logits[0], expected_gate[0]),
                 f"collection source gate logits do not match source checkpoint at row {index}")


def _kl_from_logits(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    _require(reference_logits.shape == candidate_logits.shape, "KL logits shape mismatch")
    reference_logp = F.log_softmax(reference_logits, dim=-1)
    candidate_logp = F.log_softmax(candidate_logits, dim=-1)
    reference_p = reference_logp.exp()
    values = (reference_p * (reference_logp - candidate_logp)).sum(dim=-1)
    _require(bool(torch.isfinite(values).all().item()), "non-finite proposal KL")
    return values


def _set_loss(candidate_logits: torch.Tensor, rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    logp = F.log_softmax(candidate_logits, dim=-1)
    terms: list[torch.Tensor] = []
    for index, row in enumerate(rows):
        certified = row["certified_proposals"]
        if certified:
            indices = torch.as_tensor(certified, dtype=torch.long, device=logp.device)
            terms.append(-torch.logsumexp(logp[index, indices], dim=0))
    if terms:
        return torch.stack(terms).mean()
    return candidate_logits.sum() * 0.0


def _coverage_metrics(policy: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "qK3_mean": None, "qK3_min": None, "qK3_max": None, "kl_mean": None}
    observations = _observations(rows, policy.device)
    reference, _ = _source_logits(rows, policy.device)
    with torch.no_grad():
        _, distribution = policy.model.distributions(observations)
        candidate = distribution.logits.detach()
        kl = _kl_from_logits(reference, candidate)
    masses = [
        float(k3_gate_feasibility.k3_certified_mass(candidate[index].cpu(), row["certified_proposals"])["certified_mass"])
        for index, row in enumerate(rows)
    ]
    return {
        "rows": len(rows),
        "nonempty_certified_rows": sum(bool(row["certified_proposals"]) for row in rows),
        "qK3_mean": float(np.mean(masses)),
        "qK3_min": float(np.min(masses)),
        "qK3_max": float(np.max(masses)),
        "kl_mean": float(kl.mean().cpu().item()),
        "kl_max": float(kl.max().cpu().item()),
    }


def _all_metrics(policy: Any, train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "train": _coverage_metrics(policy, train_rows),
        "validation": _coverage_metrics(policy, validation_rows),
    }


def _proposal_delta_norm(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    values = []
    for name in ("proposal_head.weight", "proposal_head.bias"):
        values.append((after[name].float() - before[name].float()).pow(2).sum())
    return float(torch.sqrt(torch.stack(values).sum()).item())


def _output_paths(output_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    root = Path(output_dir)
    _require(not root.exists(), f"refusing to overwrite output directory: {root}")
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = root / "p3_proposal_head_probe_v1.pt"
    report = root / "fit_report.json"
    progress = root / "progress.json"
    progress_jsonl = root / "progress.jsonl"
    return root, checkpoint, report, progress_jsonl


def _metadata(
    config: Any,
    *,
    source_checkpoint: Path,
    source_sha: str,
    collection_report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source_payload: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(checkpoint_metadata(config))
    source_metadata = source_payload.get("metadata", {})
    value.update({
        "stage": STAGE,
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": float(CALIBRATION_TEMPERATURE),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "source_sha256": source_sha,
        "source_architecture_id": source_metadata.get("architecture_id", CALIBRATED_ARCHITECTURE_ID)
        if isinstance(source_metadata, Mapping) else CALIBRATED_ARCHITECTURE_ID,
        "protocol": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "collection_dir": collection_report["collection_dir"],
        "collection_rows_sha256": collection_report["rows_sha256"],
        "collection_manifest_sha256": collection_report["manifest_sha256"],
        "train_seeds": sorted({int(row["seed"]) for row in rows if row["split"] == "train"}),
        "validation_seeds": sorted({int(row["seed"]) for row in rows if row["split"] == "validation"}),
        "proposal_probe_seed": TRAINING_SEED,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "retention_kl_max": RETENTION_KL_MAX,
        "trainable_scope": ["proposal_head.weight", "proposal_head.bias"],
        "trainable_parameter_count": 2425,
        "source_optimizer_copied": False,
        "diagnostic_only": True,
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
    trainable_parameters: Sequence[torch.nn.Parameter],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool]:
    payload = {
        "metadata": dict(metadata),
        "model": policy.model.state_dict(),
        "optimizer": policy.optimizer.state_dict(),
        "report": dict(report),
    }
    torch.save(payload, checkpoint)
    restored = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM, temperature=CALIBRATION_TEMPERATURE,
    )
    restored_parameters, _ = _freeze_to_proposal_head(restored)
    restored.optimizer = torch.optim.Adam(restored_parameters, lr=LEARNING_RATE)
    loaded = load_checkpoint(checkpoint, restored, config, load_optimizer=True)
    _require(loaded.get("policy_architecture_migrated") is False, "candidate architecture migrated")
    model_exact = _state_equal(_state_cpu(policy.model), _state_cpu(restored.model))
    optimizer_exact = _optimizer_state_equal(
        _optimizer_state_cpu(policy.optimizer), _optimizer_state_cpu(restored.optimizer)
    )
    _require(model_exact, "candidate checkpoint model roundtrip mismatch")
    _require(optimizer_exact, "candidate checkpoint optimizer roundtrip mismatch")
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        observations = _observations(rows, restored.device)
        with torch.no_grad():
            restored_gate, restored_proposal = restored.model.distributions(observations)
            source_gate, source_proposal = policy.model.distributions(observations)
        _require(torch.equal(restored_gate.logits, source_gate.logits),
                 f"{split} gate logits changed on checkpoint roundtrip")
        _require(torch.equal(restored_proposal.logits, source_proposal.logits),
                 f"{split} proposal logits changed on checkpoint roundtrip")
    return model_exact, optimizer_exact


def fit_proposal_head(
    collection_dir: str | Path,
    output_dir: str | Path,
    *,
    source_checkpoint: str | Path = SOURCE_CHECKPOINT,
) -> dict[str, Any]:
    """Run the single registered proposal-head fit and write its report."""

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    rows, collection_report = _load_collection(collection_dir)
    config = _build_execution_config()
    policy, source_payload, source_sha = _load_source_policy(source_checkpoint, config)
    _validate_source_rows(policy, rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    root, checkpoint_path, report_path, progress_jsonl_path = _output_paths(output_dir)
    progress_path = root / "progress.json"
    _write_json(progress_path, {"stage": STAGE, "status": "RUNNING", "processed_rows": len(rows), "epochs_completed": 0})
    progress_jsonl_path.write_text("", encoding="utf-8")

    trainable_parameters, trainable_names = _freeze_to_proposal_head(policy)
    policy.optimizer = torch.optim.Adam(trainable_parameters, lr=LEARNING_RATE)
    model = policy.model
    model.eval()
    before_state = _state_cpu(model)
    before_gate: dict[str, torch.Tensor] = {}
    device = policy.device
    train_x = _observations(train_rows, device)
    train_reference, _ = _source_logits(train_rows, device)
    validation_x = _observations(validation_rows, device)
    with torch.no_grad():
        before_gate["train"] = model.distributions(train_x)[0].logits.detach().cpu().clone()
        before_gate["validation"] = model.distributions(validation_x)[0].logits.detach().cpu().clone()
    before_metrics = _all_metrics(policy, train_rows, validation_rows)
    nonempty_train = sum(bool(row["certified_proposals"]) for row in train_rows)
    history: list[dict[str, Any]] = []
    accepted_steps = 0
    rejected_steps = 0
    stopped_reason: str | None = None

    if nonempty_train == 0:
        stopped_reason = "no_nonempty_certified_train_rows"
    else:
        generator = torch.Generator(device=device).manual_seed(TRAINING_SEED)
        for epoch in range(1, EPOCHS + 1):
            epoch_before_state = _state_cpu(model)
            model.train()
            permutation = torch.randperm(len(train_rows), generator=generator, device=device)
            epoch_accepted = 0
            epoch_rejected = 0
            for start in range(0, len(train_rows), BATCH_SIZE):
                indices = permutation[start:start + BATCH_SIZE]
                model_before = _state_cpu(model)
                optimizer_before = _optimizer_state_cpu(policy.optimizer)
                policy.optimizer.zero_grad(set_to_none=True)
                _, batch_distribution = model.distributions(train_x[indices])
                batch_rows = [train_rows[int(index)] for index in indices.detach().cpu().tolist()]
                set_loss = _set_loss(batch_distribution.logits, batch_rows)
                _, full_distribution = model.distributions(train_x)
                kl_values = _kl_from_logits(train_reference, full_distribution.logits)
                kl_loss = kl_values.mean()
                loss = set_loss + KL_COEFFICIENT * kl_loss
                _require(bool(torch.isfinite(loss).item()), f"non-finite loss at epoch {epoch}")
                loss.backward()
                gradients = [parameter.grad for parameter in trainable_parameters if parameter.grad is not None]
                _require(bool(gradients) and all(bool(torch.isfinite(g).all().item()) for g in gradients),
                         f"non-finite proposal gradient at epoch {epoch}")
                norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, GRADIENT_CLIP_NORM)
                _require(bool(torch.isfinite(norm).item()), f"non-finite gradient norm at epoch {epoch}")
                policy.optimizer.step()
                with torch.no_grad():
                    _, after_distribution = model.distributions(train_x)
                    after_kl = float(_kl_from_logits(train_reference, after_distribution.logits).mean().cpu().item())
                if after_kl > RETENTION_KL_MAX:
                    model.load_state_dict(model_before, strict=True)
                    policy.optimizer.load_state_dict(optimizer_before)
                    rejected_steps += 1
                    epoch_rejected += 1
                    stopped_reason = "retention_kl_cap_exceeded"
                    break
                accepted_steps += 1
                epoch_accepted += 1
            model.eval()
            metrics = _all_metrics(policy, train_rows, validation_rows)
            record = {
                "epoch": epoch,
                "accepted_steps": epoch_accepted,
                "rejected_steps": epoch_rejected,
                "accepted_steps_total": accepted_steps,
                "rejected_steps_total": rejected_steps,
                "train_set_rows": nonempty_train,
                "train_kl": metrics["train"]["kl_mean"],
                "train_qK3": metrics["train"]["qK3_mean"],
                "validation_kl": metrics["validation"]["kl_mean"],
                "validation_qK3": metrics["validation"]["qK3_mean"],
                "proposal_parameter_delta_norm": _proposal_delta_norm(before_state, _state_cpu(model)),
                "epoch_proposal_parameter_delta_norm": _proposal_delta_norm(
                    epoch_before_state, _state_cpu(model),
                ),
                "metrics": metrics,
            }
            history.append(record)
            with progress_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            _write_json(progress_path, {
                "stage": STAGE,
                "status": "RUNNING",
                "processed_rows": len(rows),
                "epochs_completed": epoch,
                "accepted_steps": accepted_steps,
                "rejected_steps": rejected_steps,
                "stopped_reason": stopped_reason,
                "latest": record,
            })
            if stopped_reason is not None:
                break

    model.eval()
    after_state = _state_cpu(model)
    delta_norm = _proposal_delta_norm(before_state, after_state)
    with torch.no_grad():
        after_gate_train = model.distributions(train_x)[0].logits.detach().cpu().clone()
        after_gate_validation = model.distributions(validation_x)[0].logits.detach().cpu().clone()
    gate_logits_bitwise_unchanged = (
        torch.equal(before_gate["train"], after_gate_train)
        and torch.equal(before_gate["validation"], after_gate_validation)
    )
    nonproposal_state_bitwise_unchanged = all(
        torch.equal(before_state[name], after_state[name])
        for name in before_state if not name.startswith(PROPOSAL_PREFIX)
    )
    frozen_parameters = all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith(PROPOSAL_PREFIX)
    )
    effective_fit = bool(delta_norm > 0.0 and accepted_steps > 0)
    if stopped_reason == "retention_kl_cap_exceeded":
        status = "STOPPED_KL_CAP"
    elif nonempty_train == 0 or not effective_fit:
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
    )
    report: dict[str, Any] = {
        "stage": STAGE,
        "status": status,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "collection": collection_report,
        "protocol": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "seed": TRAINING_SEED,
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
                      "epochs": EPOCHS, "gradient_clip_norm": GRADIENT_CLIP_NORM},
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_nonempty_certified_rows": nonempty_train,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "stopped_reason": stopped_reason,
        "effective_fit": effective_fit,
        "metrics_before": before_metrics,
        "metrics_after": _all_metrics(policy, train_rows, validation_rows),
        "history": history,
        "invariants": {
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_parameters),
            "nonproposal_state_bitwise_unchanged": nonproposal_state_bitwise_unchanged,
            "gate_logits_bitwise_unchanged": gate_logits_bitwise_unchanged,
            "all_nonproposal_parameters_frozen": frozen_parameters,
            "source_optimizer_copied": False,
            "retention_kl_max": RETENTION_KL_MAX,
        },
        "metadata": metadata,
    }
    _require(nonproposal_state_bitwise_unchanged, "non-proposal state changed during fit")
    _require(gate_logits_bitwise_unchanged, "gate logits changed during fit")
    _require(frozen_parameters, "non-proposal parameters remain trainable")
    model_exact, optimizer_exact = _save_and_restore(
        checkpoint_path, policy, config, metadata, report, trainable_parameters,
        train_rows, validation_rows,
    )
    report["checkpoint"] = str(checkpoint_path)
    report["checkpoint_sha256"] = _sha256(checkpoint_path)
    report["checkpoint_roundtrip_model_exact"] = model_exact
    report["checkpoint_roundtrip_optimizer_exact"] = optimizer_exact
    report["invariants"]["checkpoint_roundtrip_model_exact"] = model_exact
    report["invariants"]["checkpoint_roundtrip_optimizer_exact"] = optimizer_exact
    _write_json(report_path, report)
    _write_json(progress_path, {
        "stage": STAGE, "status": status, "processed_rows": len(rows),
        "accepted_steps": accepted_steps, "rejected_steps": rejected_steps,
        "checkpoint": str(checkpoint_path), "stopped_reason": stopped_reason,
    })
    return {
        "status": status,
        "stage": STAGE,
        "checkpoint": str(checkpoint_path),
        "fit_report": str(report_path),
        "progress": str(progress_path),
        "progress_jsonl": str(progress_jsonl_path),
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "source_checkpoint_sha256": source_sha,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "effective_fit": effective_fit,
        "invariants": report["invariants"],
    }


fit_proposal_head_probe = fit_proposal_head
train_proposal_head_probe_v1 = fit_proposal_head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    args = parser.parse_args()
    result = fit_proposal_head(
        args.collection_dir, args.output_dir, source_checkpoint=args.source_checkpoint,
    )
    print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result["status"] in {"COMPLETED", "NO_EFFECTIVE_FIT", "STOPPED_KL_CAP"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
