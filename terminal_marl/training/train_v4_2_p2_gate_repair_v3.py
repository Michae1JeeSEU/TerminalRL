"""Conservative anchored P2 gate residual-adapter repair V3.

The runner consumes the frozen V5 on-policy support artifact, the passing V2
learnability audit, the calibrated qualification candidate, and the report-
only D1 anchor.  It fine-tunes only the candidate's existing ``gate_adapter``
parameters.  No environment, Teacher, development gradient, threshold
tuning, policy migration, or PPO authorization is performed here.
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

from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training import audit_v4_2_p2_gate_learnability_v2 as audit


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_repair_v3.yaml")
PROTOCOL_SHA = "f8a84632fea21e6789c8a40c0a22938cb5f0bdfcdb4ec9efd5e48304867778e8"

AUDIT_MANIFEST_SHA = "5a72549d05c4733c2163512bf30cfe28d601c38e3ddfe23dc8291ec2f17b3679"
SUPPORT_MANIFEST_SHA = "100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960"
SUPPORT_TRAIN_SHA = "40002db70c2e0d5d181b53b2f23290c65954f6bbe6282be1066b6f558e6153d4"
SUPPORT_DEV_SHA = "da106e0d862c8e435748514abb10c4e0e5c500fb3bbd73f03cbfa6aa186a261b"
CANDIDATE_CHECKPOINT_SHA = "61e03e02e3fb204d2db4e814097543905f51e1a3772626fdac1d7bc02ab8ca6b"
CANDIDATE_ARCHITECTURE_ID = "v4_2_hierarchical_gate_temperature_calibrated_residual_adapter_v3"
CANDIDATE_TEMPERATURE = 2.5
D1_DATASET_SHA = "270e54ef6ed0c36f535e2a0f25cef7cf9fdd6c91016ee295e7e336fb800cacb0"
D1_MANIFEST_SHA = "a6319c291e5c91e155fd1c7fb20ae95dcdd4973733abfe83fc10c5e6006f472c"
OBSERVATION_DIM = 88
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"
FUTURE_QUALIFICATION_SEEDS = tuple(range(782001, 782007))
ADAPTER_PREFIX = "gate_adapter"

TRAINING_SEED = 787101
EPOCHS = 20
SUPPORT_BATCH_SIZE = 128
D1_BATCH_SIZE = 128
LEARNING_RATE = 1e-4
GRADIENT_CLIP_NORM = 1.0
ANCHOR_KL_WEIGHT = 2.0

SUPPORT_THRESHOLDS = {
    "macro_F1_min": 0.80,
    "WAIT_recall_min": 0.70,
    "RELEASE_recall_min": 0.80,
    "brier_max": 0.15,
    "binary_nll_max": 0.50,
    "interface_macro_F1_min": 0.70,
}
D1_THRESHOLDS = {
    "macro_F1_min": 0.90,
    "WAIT_recall_min": 0.90,
    "RELEASE_recall_min": 0.90,
    "brier_max": 0.08,
    "binary_nll_max": 0.30,
}


class RepairInputError(ValueError):
    """Raised when a repair input violates the preregistered contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairInputError(message)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_sha = _sha256


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def protocol() -> dict[str, Any]:
    """Read and mechanically validate the byte-frozen V3 protocol."""
    _require(PROTOCOL.is_file(), f"repair protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA,
                 "V3 repair protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "V3 repair protocol must be a mapping")
    _require(value.get("protocol") == "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
             "V3 repair protocol id changed")

    predecessor = value.get("predecessor", {})
    for key, expected in {
        "audit_protocol": "V4_2_P2_GATE_LEARNABILITY_AUDIT_V2",
        "audit_stage": "V4_2_P2_GATE_LEARNABILITY_AUDIT_V2",
        "audit_manifest_sha256": AUDIT_MANIFEST_SHA,
        "audit_pass": True,
        "repair_preregistration_allowed": True,
    }.items():
        _require(predecessor.get(key) == expected, f"V3 predecessor binding changed: {key}")

    bindings = value.get("bindings", {})
    expected_bindings = {
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "support_train_sha256": SUPPORT_TRAIN_SHA,
        "support_dev_sha256": SUPPORT_DEV_SHA,
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
        _require(bindings.get(key) == expected, f"V3 repair binding changed: {key}")

    audit_prerequisite = value.get("audit_prerequisite", {})
    for key, expected in {
        "stage": "V4_2_P2_GATE_LEARNABILITY_AUDIT_V2",
        "AUDIT_COMPLETE": True,
        "audit_pass": True,
        "learnability_pass": True,
        "repair_preregistration_allowed": True,
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
    }.items():
        _require(audit_prerequisite.get(key) == expected,
                 f"V3 audit prerequisite changed: {key}")

    architecture = value.get("architecture", {})
    for key, expected in {
        "id": CANDIDATE_ARCHITECTURE_ID,
        "base": "TemperatureCalibratedGateResidualAdapterStudentPPOV42",
        "initialized_from": "exact_deepcopy_of_calibrated_candidate",
        "trainable_scope": "gate_adapter_only",
        "proposal_unchanged": True,
        "critic_unchanged": True,
        "temperature": CANDIDATE_TEMPERATURE,
        "migration": "none",
    }.items():
        _require(architecture.get(key) == expected,
                 f"V3 architecture binding changed: {key}")

    training = value.get("training", {})
    expected_training = {
        "seed": TRAINING_SEED,
        "epochs": EPOCHS,
        "support_batch_size": SUPPORT_BATCH_SIZE,
        "d1_batch_size": D1_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
    }
    for key, expected in expected_training.items():
        actual = training.get(key)
        if isinstance(expected, float):
            _require(float(actual) == expected, f"V3 training binding changed: {key}")
        else:
            _require(int(actual) == expected, f"V3 training binding changed: {key}")
    for key, expected in {
        "steps_per_epoch": "max_ceil_support_or_d1_batches",
        "optimizer": "Adam",
        "objective": "0.5_support_CE_plus_0.5_D1_CE_plus_2.0_D1_anchor_KL",
        "anchor_KL": "categorical_KL_frozen_candidate_to_repaired_float64_logits",
        "checkpoint_selection": "final_epoch_only",
        "progress": "after_each_completed_epoch",
    }.items():
        _require(training.get(key) == expected, f"V3 training binding changed: {key}")
    for key in ("reshuffle_each_epoch", "cycle_shorter_source"):
        _require(training.get(key) is True, f"V3 training binding changed: {key}")
    for key in ("dev_gradients", "d1_selection", "early_stopping", "threshold_tuning"):
        _require(training.get(key) is False, f"V3 training restriction changed: {key}")

    stability = value.get("numerical_stability", {})
    for key, expected in {
        "candidate_log_probability_dtype": "float64",
        "repaired_log_probability_dtype": "float64",
        "probability_source": "exp_candidate_log_softmax",
        "categorical_from_float32_probabilities": False,
        "anchor_weight": ANCHOR_KL_WEIGHT,
        "finite_component_checks": True,
        "finite_gradient_checks": True,
        "failure_context": "epoch_step_finite_flags_component_values",
    }.items():
        _require(stability.get(key) == expected,
                 f"V3 numerical-stability binding changed: {key}")
    _require(stability.get("stable_formula") ==
             "sum(exp(candidate_logp) * (candidate_logp - repaired_logp), dim=-1).mean()",
             "V3 stable-KL formula changed")

    thresholds = value.get("thresholds", {})
    for name, configured in (("support_dev", SUPPORT_THRESHOLDS), ("d1_anchor", D1_THRESHOLDS)):
        section = thresholds.get(name, {})
        for key, expected in configured.items():
            _require(float(section.get(key)) == expected,
                     f"V3 threshold changed: {name}.{key}")
    return value


def validate_audit_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = AUDIT_MANIFEST_SHA,
) -> dict[str, Any]:
    """Require the completed passing V2 learnability audit before repair."""
    manifest_path = Path(path)
    _require(manifest_path.is_file(), f"audit manifest missing: {manifest_path}")
    if expected_sha256 is not None:
        _require(_sha256(manifest_path) == expected_sha256, "audit manifest SHA mismatch")
    manifest = _read_json(manifest_path)
    for key, expected in {
        "stage": "V4_2_P2_GATE_LEARNABILITY_AUDIT_V2",
        "AUDIT_COMPLETE": True,
        "audit_pass": True,
        "learnability_pass": True,
        "repair_preregistration_allowed": True,
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA,
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "environment_called": False,
        "teacher_called": False,
        "dev_gradients": False,
        "d1_gradients": False,
        "dev_selection": False,
        "d1_selection": False,
        "threshold_tuning": False,
    }.items():
        _require(manifest.get(key) == expected, f"audit manifest binding changed: {key}")
    return manifest


def _load_candidate(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load the calibrated candidate through the V2 audit loader."""
    loaded = audit._load_candidate(path)
    if isinstance(loaded, tuple):
        candidate, payload = loaded
    else:
        candidate, payload = loaded, {}
    _require(getattr(candidate, "architecture_id", None) == CANDIDATE_ARCHITECTURE_ID,
             "candidate architecture mismatch")
    _require(float(candidate.model.gate_temperature.detach().cpu()) == CANDIDATE_TEMPERATURE,
             "candidate temperature mismatch")
    return candidate, payload


validate_support_manifest = audit.validate_support_manifest
validate_d1_anchor = audit.validate_d1_anchor
evaluate_model = audit.evaluate_model


def _set_adapter_only_trainable(policy: Any) -> tuple[list[str], list[str]]:
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in policy.model.named_parameters():
        enabled = name == ADAPTER_PREFIX or name.startswith(ADAPTER_PREFIX + ".")
        parameter.requires_grad_(enabled)
        (trainable if enabled else frozen).append(name)
    _require(bool(trainable), "gate adapter has no trainable parameters")
    _require(all(name.startswith(ADAPTER_PREFIX + ".") for name in trainable),
             "repair trainable scope is not gate_adapter only")
    return trainable, frozen


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _observations(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    values, _ = audit._row_arrays(rows)
    return values


def _distribution_logits(model: nn.Module, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    observations = _observations(rows)
    model.eval()
    with torch.no_grad():
        gate, proposal = model.distributions(observations)
    return gate.logits.detach().cpu().numpy().copy(), proposal.logits.detach().cpu().numpy().copy()


def _logit_invariance(
    before_model: nn.Module,
    after_model: nn.Module,
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    all_equal = True
    maximum = 0.0
    for name, rows in datasets.items():
        before_gate, before_proposal = _distribution_logits(before_model, rows)
        after_gate, after_proposal = _distribution_logits(after_model, rows)
        gate_delta = np.abs(after_gate.astype(np.float64) - before_gate.astype(np.float64))
        proposal_delta = np.abs(after_proposal.astype(np.float64) - before_proposal.astype(np.float64))
        result[name] = {
            "gate_bitwise_equal": bool(np.array_equal(before_gate, after_gate)),
            "proposal_bitwise_equal": bool(np.array_equal(before_proposal, after_proposal)),
            "gate_max_abs_delta": float(gate_delta.max()) if gate_delta.size else 0.0,
            "proposal_max_abs_delta": float(proposal_delta.max()) if proposal_delta.size else 0.0,
            "samples": int(len(rows)),
        }
        all_equal = all_equal and result[name]["proposal_bitwise_equal"]
        maximum = max(maximum, result[name]["proposal_max_abs_delta"])
    result["proposal_all_bitwise_equal"] = bool(all_equal)
    result["proposal_all_max_abs_delta"] = float(maximum)
    result["proposal_invariant"] = bool(all_equal or maximum == 0.0)
    return result


def _critic_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("critic_encoder.") or name.startswith("critic_head.")
    }


def _state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def _changed_state_names(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor],
) -> list[str]:
    _require(set(before) == set(after), "model state keys changed unexpectedly")
    return sorted(name for name in before
                  if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu()))


def _changed_parameter_names(before_model: nn.Module, after_model: nn.Module) -> list[str]:
    before = dict(before_model.named_parameters())
    after = dict(after_model.named_parameters())
    _require(set(before) == set(after), "model parameter keys changed unexpectedly")
    return sorted(name for name in before
                  if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu()))


def _batch_indices(permutation: np.ndarray, step: int, batch_size: int) -> np.ndarray:
    if len(permutation) == 0:
        raise RepairInputError("repair source is empty")
    start = (int(step) * int(batch_size)) % len(permutation)
    offsets = np.arange(int(batch_size), dtype=np.int64)
    return permutation[(start + offsets) % len(permutation)]


def _finite_gradients(parameters: Sequence[nn.Parameter]) -> tuple[bool, float]:
    norms: list[float] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            return False, float("nan")
        norms.append(float(parameter.grad.detach().norm().cpu()))
    return True, float(max(norms, default=0.0))


def stable_anchor_kl_from_logits(
    candidate_logits: torch.Tensor,
    repaired_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute candidate-to-repaired categorical KL in float64 from logits."""
    _require(tuple(candidate_logits.shape) == tuple(repaired_logits.shape),
             "candidate and repaired logits must have identical shapes")
    candidate_logp = F.log_softmax(candidate_logits.double(), dim=-1)
    repaired_logp = F.log_softmax(repaired_logits.double(), dim=-1)
    candidate_p = torch.exp(candidate_logp)
    return torch.sum(candidate_p * (candidate_logp - repaired_logp), dim=-1).mean()


stable_anchor_kl = stable_anchor_kl_from_logits
_stable_anchor_kl = stable_anchor_kl_from_logits


def _step_context(
    epoch: int,
    step: int,
    components: Mapping[str, torch.Tensor],
) -> tuple[dict[str, bool], dict[str, float]]:
    del epoch, step
    finite: dict[str, bool] = {}
    values: dict[str, float] = {}
    for name, value in components.items():
        finite[name] = bool(torch.isfinite(value.detach()).all().item())
        values[name] = float(value.detach().cpu())
    return finite, values


def _nonfinite_error(
    epoch: int,
    step: int,
    components: Mapping[str, torch.Tensor],
    reason: str = "non-finite repair component or loss",
) -> FloatingPointError:
    finite, values = _step_context(epoch, step, components)
    return FloatingPointError(
        f"{reason}; epoch={epoch} step={step} finite={finite} values={values}"
    )


def _write_progress(
    path: str | Path,
    *,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    steps_per_epoch: int,
) -> Path:
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps({
        "status": "RUNNING",
        "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
        "latest_epoch": int(epoch),
        "completed_epochs": int(len(history)),
        "steps_per_epoch": int(steps_per_epoch),
        "latest": dict(history[-1]) if history else None,
        "history": [dict(item) for item in history],
    }, indent=2, sort_keys=True), encoding="utf-8")
    return progress_path


def train_repair(
    policy: Any,
    frozen_candidate: nn.Module,
    support_rows: Sequence[Mapping[str, Any]],
    d1_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = TRAINING_SEED,
    epochs: int = EPOCHS,
    support_batch_size: int = SUPPORT_BATCH_SIZE,
    d1_batch_size: int = D1_BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    gradient_clip_norm: float = GRADIENT_CLIP_NORM,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train only gate_adapter with the paired anchored objective."""
    _require(support_rows and d1_rows, "repair sources must be nonempty")
    _require(epochs >= 1 and support_batch_size >= 1 and d1_batch_size >= 1
             and learning_rate > 0.0 and gradient_clip_norm > 0.0,
             "invalid repair training budget")
    _seed_everything(seed)
    support_x = _observations(support_rows)
    support_y = torch.as_tensor([int(row["teacher_gate"]) for row in support_rows], dtype=torch.long)
    d1_x = _observations(d1_rows)
    d1_y = torch.as_tensor([int(row["teacher_gate"]) for row in d1_rows], dtype=torch.long)
    trainable_names, frozen_names = _set_adapter_only_trainable(policy)
    trainable_parameters = [parameter for parameter in policy.model.parameters()
                            if parameter.requires_grad]
    for group in policy.optimizer.param_groups:
        group["lr"] = float(learning_rate)
    support_steps = int(math.ceil(len(support_rows) / support_batch_size))
    d1_steps = int(math.ceil(len(d1_rows) / d1_batch_size))
    steps_per_epoch = max(support_steps, d1_steps)
    history: list[dict[str, float]] = []
    finite_losses = True
    finite_gradients = True
    max_gradient_norm = 0.0
    frozen_candidate.eval()
    policy.model.train()
    for epoch in range(1, int(epochs) + 1):
        support_permutation = np.random.default_rng(int(seed) + epoch).permutation(len(support_rows))
        d1_permutation = np.random.default_rng(int(seed) + 100000 + epoch).permutation(len(d1_rows))
        epoch_values: list[dict[str, float]] = []
        for step in range(steps_per_epoch):
            components: dict[str, torch.Tensor] = {}
            try:
                support_indices = _batch_indices(support_permutation, step, support_batch_size)
                d1_indices = _batch_indices(d1_permutation, step, d1_batch_size)
                support_gate, _ = policy.model.distributions(support_x[support_indices])
                d1_gate, _ = policy.model.distributions(d1_x[d1_indices])
                support_ce = F.cross_entropy(support_gate.logits, support_y[support_indices])
                d1_ce = F.cross_entropy(d1_gate.logits, d1_y[d1_indices])
                components["support_CE"] = support_ce
                components["d1_CE"] = d1_ce
                with torch.no_grad():
                    candidate_gate, _ = frozen_candidate.distributions(d1_x[d1_indices])
                    candidate_logits = candidate_gate.logits.detach()
                d1_anchor_kl = stable_anchor_kl_from_logits(candidate_logits, d1_gate.logits)
                components["d1_anchor_KL"] = d1_anchor_kl
                loss = 0.5 * support_ce + 0.5 * d1_ce + ANCHOR_KL_WEIGHT * d1_anchor_kl
                components["loss"] = loss
                finite, _ = _step_context(epoch, step, components)
                if not all(finite.values()):
                    finite_losses = False
                    raise _nonfinite_error(epoch, step, components)
                policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients_ok, gradient_norm = _finite_gradients(trainable_parameters)
                finite_gradients = finite_gradients and gradients_ok
                if not gradients_ok:
                    raise _nonfinite_error(epoch, step, components, "non-finite repair gradients")
                try:
                    clipped = nn.utils.clip_grad_norm_(
                        trainable_parameters, float(gradient_clip_norm), error_if_nonfinite=True
                    )
                except Exception as exc:
                    raise _nonfinite_error(epoch, step, components,
                                           "gradient clipping failed") from exc
                if not bool(torch.isfinite(clipped.detach()).all().item()):
                    raise _nonfinite_error(epoch, step, components,
                                           "non-finite clipped gradient norm")
                max_gradient_norm = max(max_gradient_norm, float(clipped.detach().cpu()))
                policy.optimizer.step()
                epoch_values.append({
                    "support_CE": float(support_ce.detach().cpu()),
                    "d1_CE": float(d1_ce.detach().cpu()),
                    "d1_anchor_KL": float(d1_anchor_kl.detach().cpu()),
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": gradient_norm,
                })
            except FloatingPointError:
                raise
            except Exception as exc:
                finite, values = _step_context(epoch, step, components)
                raise FloatingPointError(
                    f"repair step failed; epoch={epoch} step={step} "
                    f"finite={finite} values={values}: {exc}"
                ) from exc
        aggregate = {
            "epoch": float(epoch),
            "support_CE": float(np.mean([item["support_CE"] for item in epoch_values])),
            "d1_CE": float(np.mean([item["d1_CE"] for item in epoch_values])),
            "d1_anchor_KL": float(np.mean([item["d1_anchor_KL"] for item in epoch_values])),
            "loss": float(np.mean([item["loss"] for item in epoch_values])),
            "gradient_norm": float(np.mean([item["gradient_norm"] for item in epoch_values])),
        }
        history.append(aggregate)
        if progress_path is not None:
            _write_progress(progress_path, epoch=epoch, history=history,
                            steps_per_epoch=steps_per_epoch)
    policy.model.eval()
    return {
        "seed": int(seed),
        "epochs": int(epochs),
        "final_epoch": int(epochs),
        "support_batch_size": int(support_batch_size),
        "d1_batch_size": int(d1_batch_size),
        "steps_per_epoch": int(steps_per_epoch),
        "support_steps_per_epoch": int(support_steps),
        "d1_steps_per_epoch": int(d1_steps),
        "learning_rate": float(learning_rate),
        "gradient_clip_norm": float(gradient_clip_norm),
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "objective": "0.5_support_CE+0.5_D1_CE+2.0_D1_anchor_KL",
        "anchor_KL": "categorical_KL_frozen_candidate_to_repaired_float64_logits",
        "finite_losses": bool(finite_losses and all(np.isfinite(item["loss"]) for item in history)),
        "finite_gradients": bool(finite_gradients),
        "max_gradient_norm_before_clip": float(max_gradient_norm),
        "early_stopping": False,
        "selection_split": "none_final_epoch_only",
        "progress_path": str(progress_path) if progress_path is not None else None,
        "history": history,
    }


def repair_threshold_checks(
    support_dev: Mapping[str, Any],
    d1_anchor: Mapping[str, Any],
    invariants: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply inclusive support/D1 metric gates and hard invariants."""
    support_checks = {
        "macro_F1": float(support_dev.get("macro_F1", float("nan"))) >= SUPPORT_THRESHOLDS["macro_F1_min"],
        "WAIT_recall": float(support_dev.get("WAIT_recall", float("nan"))) >= SUPPORT_THRESHOLDS["WAIT_recall_min"],
        "RELEASE_recall": float(support_dev.get("RELEASE_recall", float("nan"))) >= SUPPORT_THRESHOLDS["RELEASE_recall_min"],
        "brier": float(support_dev.get("brier", float("nan"))) <= SUPPORT_THRESHOLDS["brier_max"],
        "binary_nll": float(support_dev.get("binary_nll", float("nan"))) <= SUPPORT_THRESHOLDS["binary_nll_max"],
    }
    interface_checks = {
        interface: float(support_dev.get("interface", {}).get(interface, {}).get("macro_F1", float("nan")))
        >= SUPPORT_THRESHOLDS["interface_macro_F1_min"]
        for interface in audit.INTERFACES
    }
    support_checks["every_interface_macro_F1"] = bool(interface_checks) and all(interface_checks.values())
    d1_checks = {
        "macro_F1": float(d1_anchor.get("macro_F1", float("nan"))) >= D1_THRESHOLDS["macro_F1_min"],
        "WAIT_recall": float(d1_anchor.get("WAIT_recall", float("nan"))) >= D1_THRESHOLDS["WAIT_recall_min"],
        "RELEASE_recall": float(d1_anchor.get("RELEASE_recall", float("nan"))) >= D1_THRESHOLDS["RELEASE_recall_min"],
        "brier": float(d1_anchor.get("brier", float("nan"))) <= D1_THRESHOLDS["brier_max"],
        "binary_nll": float(d1_anchor.get("binary_nll", float("nan"))) <= D1_THRESHOLDS["binary_nll_max"],
    }
    invariant_checks = {
        "proposal_invariant": bool(invariants.get("proposal_invariant", False)),
        "critic_invariant": bool(invariants.get("critic_invariant", False)),
        "temperature_state_invariant": bool(invariants.get("temperature_state_invariant", False)),
        "common_state_bitwise_equal": bool(invariants.get("common_state_bitwise_equal", False)),
        "parameter_isolation": bool(invariants.get("parameter_isolation", False)),
        "finite_losses": bool(invariants.get("finite_losses", False)),
        "finite_gradients": bool(invariants.get("finite_gradients", False)),
    }
    return {
        "support_dev": {**support_checks, "interface": interface_checks,
                        "all": all(support_checks.values())},
        "d1_anchor": {**d1_checks, "all": all(d1_checks.values())},
        "invariants": {**invariant_checks, "all": all(invariant_checks.values())},
        "all": bool(all(support_checks.values()) and all(d1_checks.values())
                    and all(invariant_checks.values())),
    }


def _load_inputs(
    support_manifest: str | Path,
    audit_manifest: str | Path,
    candidate_checkpoint: str | Path,
    d1_dataset: str | Path,
    d1_manifest: str | Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Path],
           list[dict[str, Any]], Any, dict[str, Any]]:
    audit_manifest_value = validate_audit_manifest(audit_manifest)
    _, support_rows, support_paths = validate_support_manifest(support_manifest)
    _, d1_rows = validate_d1_anchor(d1_dataset, d1_manifest)
    candidate, candidate_payload = _load_candidate(candidate_checkpoint)
    return (audit_manifest_value, support_rows, support_paths, d1_rows,
            candidate, candidate_payload)


def write_failure_manifest(
    output: str | Path,
    exception: BaseException,
    *,
    progress_path: str | Path | None = None,
) -> Path:
    """Write terminal failure evidence without serializing a partial checkpoint."""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    payload = {
        "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
        "status": "FAIL",
        "repair_pass": False,
        "evidence_type": "newly_run",
        "exception": {"type": type(exception).__name__, "message": str(exception)},
        "checkpoint": None,
        "checkpoint_saved": False,
        "no_checkpoint": True,
        "checkpoint_role": None,
        "qualification_only": True,
        "STUDENT_QUALIFICATION_ALLOWED": False,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "environment_called": False,
        "teacher_called": False,
        "dev_gradients": False,
        "d1_selection": False,
        "early_stopping": False,
        "threshold_tuning": False,
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "audit_manifest_sha256": AUDIT_MANIFEST_SHA,
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA,
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "progress": str(progress_path) if progress_path is not None else None,
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def run(
    output: str | Path,
    support_manifest: str | Path,
    audit_manifest: str | Path,
    candidate_checkpoint: str | Path,
    d1_dataset: str | Path,
    d1_manifest: str | Path,
) -> dict[str, Any]:
    """Run the fixed conservative anchored residual-adapter repair."""
    protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"repair output already exists: {output_path}")
    (audit_manifest_value, support_rows, support_paths, d1_rows,
     candidate, candidate_payload) = _load_inputs(
        support_manifest, audit_manifest, candidate_checkpoint, d1_dataset, d1_manifest,
    )
    output_path.mkdir(parents=True, exist_ok=False)
    datasets: dict[str, Sequence[Mapping[str, Any]]] = {
        "train": support_rows["train"], "dev": support_rows["dev"], "d1": d1_rows,
    }
    baseline = evaluate_model(candidate.model, datasets)
    repair_policy = copy.deepcopy(candidate)
    frozen_candidate = copy.deepcopy(candidate.model)
    before_policy_model = copy.deepcopy(repair_policy.model)
    before_policy_state = {
        name: value.detach().cpu().clone() for name, value in repair_policy.model.state_dict().items()
    }
    before_candidate_state = {
        name: value.detach().cpu().clone() for name, value in candidate.model.state_dict().items()
    }
    before_identity = _logit_invariance(candidate.model, repair_policy.model, datasets)
    _require(all(item["gate_bitwise_equal"] for name, item in before_identity.items()
                 if name in datasets), "candidate deepcopy changed gate logits")
    _require(before_identity["proposal_invariant"], "candidate deepcopy changed proposal logits")
    progress_path = output_path / "progress.json"
    try:
        training = train_repair(
            repair_policy, frozen_candidate, support_rows["train"], d1_rows,
            progress_path=progress_path,
        )
    except Exception as exc:
        write_failure_manifest(output_path, exc, progress_path=progress_path)
        raise

    final = evaluate_model(repair_policy.model, datasets)
    after_invariance = _logit_invariance(candidate.model, repair_policy.model, datasets)
    changed_state = _changed_state_names(before_policy_state, repair_policy.model.state_dict())
    changed_parameters = _changed_parameter_names(before_policy_model, repair_policy.model)
    # The parameter comparison above uses an exact candidate model copy; the
    # state-level comparison additionally catches an accidental buffer update.
    common_changed = [name for name in changed_state
                      if not (name == ADAPTER_PREFIX or name.startswith(ADAPTER_PREFIX + "."))]
    temperature_equal = torch.equal(
        before_policy_state["gate_temperature"].detach().cpu(),
        repair_policy.model.state_dict()["gate_temperature"].detach().cpu(),
    )
    critic_invariant = _state_equal(_critic_state(candidate.model), _critic_state(repair_policy.model))
    parameter_isolation = bool(changed_parameters) and all(
        name.startswith(ADAPTER_PREFIX + ".") for name in changed_parameters
    )
    invariants = {
        "before_identity": before_identity,
        "after": after_invariance,
        "proposal_invariant": bool(after_invariance["proposal_invariant"]),
        "critic_invariant": bool(critic_invariant),
        "temperature_state_invariant": bool(temperature_equal),
        "changed_state_names": changed_state,
        "changed_parameter_names": changed_parameters,
        "common_changed_state_names": common_changed,
        "parameter_isolation": bool(parameter_isolation and not common_changed),
        "common_state_bitwise_equal": not bool(common_changed),
        "finite_losses": bool(training["finite_losses"]),
        "finite_gradients": bool(training["finite_gradients"]),
        "base_state_unchanged_except_adapter": not bool(common_changed),
        "temperature": float(repair_policy.model.gate_temperature.detach().cpu()),
    }
    thresholds = repair_threshold_checks(final["dev"], final["d1"], invariants)
    repair_pass = bool(thresholds["all"])
    checkpoint_path = output_path / "gate_repair_v3_qualification_candidate.pt"
    save_checkpoint(
        checkpoint_path,
        repair_policy,
        audit.p2_config(6),
        stage="V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
        architecture_id=CANDIDATE_ARCHITECTURE_ID,
        base_checkpoint_sha256=CANDIDATE_CHECKPOINT_SHA,
        training_seed=TRAINING_SEED,
        qualification_only=True,
        checkpoint_role="qualification_candidate",
        qualification_pass=repair_pass,
        STUDENT_QUALIFICATION_ALLOWED=repair_pass,
        PPO_ALLOWED=False,
        GATE_REPAIR_ALLOWED=False,
        curriculum_promotion_allowed=False,
        checkpoint_promotable=False,
        environment_called=False,
        teacher_called=False,
    )
    result: dict[str, Any] = {
        "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
        "status": "PASS" if repair_pass else "FAIL",
        "repair_pass": repair_pass,
        "evidence_type": "newly_run",
        "repair_training": "formal_offline_gate_repair",
        "support_manifest": str(support_manifest),
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "support_files": {
            name: {"path": str(support_paths[name]), "sha256": _sha256(support_paths[name])}
            for name in ("train", "dev")
        },
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": AUDIT_MANIFEST_SHA,
        "audit_binding_status": {
            key: audit_manifest_value.get(key)
            for key in ("stage", "AUDIT_COMPLETE", "audit_pass", "repair_preregistration_allowed")
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
        "baseline": baseline,
        "final": final,
        "training": training,
        "invariants": invariants,
        "thresholds": thresholds,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_saved": True,
        "checkpoint_role": "qualification_candidate",
        "qualification_only": True,
        "STUDENT_QUALIFICATION_ALLOWED": repair_pass,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "environment_called": False,
        "teacher_called": False,
        "dev_gradients": False,
        "d1_selection": False,
        "early_stopping": False,
        "threshold_tuning": False,
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    output_manifest = output_path / "manifest.json"
    result["manifest"] = str(output_manifest)
    output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "stage": result["stage"], "status": result["status"],
        "repair_pass": repair_pass,
        "STUDENT_QUALIFICATION_ALLOWED": repair_pass,
        "PPO_ALLOWED": False, "GATE_REPAIR_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--d1-dataset", type=Path, required=True)
    parser.add_argument("--d1-manifest", type=Path, required=True)
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    _main()
