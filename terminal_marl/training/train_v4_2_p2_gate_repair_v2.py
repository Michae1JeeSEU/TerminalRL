"""Numerically stable offline P2 V4.2 gate residual-adapter repair.

This V2 runner is a numerical-stability-only successor to the failed V1
repair attempt.  It preserves the V1 architecture, data bindings, training
budget, objective weights, thresholds, and authorization gates.  The D1
anchor KL is evaluated directly from logits in float64 so float32 probability
underflow cannot create a non-finite repair loss.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from terminal_marl.algo.gate_adapter_v4_2 import (
    ADAPTER_PREFIX,
    GateResidualAdapterStudentPPOV42,
    migrate_base_policy_to_adapter,
)
from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training import audit_v4_2_p2_gate_learnability_v1 as audit
from terminal_marl.training import train_v4_2_p2_gate_repair_v1 as v1


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_repair_v2.yaml")
PROTOCOL_SHA = "5b56a47f0d816f59f537836ab504d8c369d77ec5bd54b9679bdb6aeeb7b1719f"
V1_PROTOCOL_SHA256 = "88bdc0578d021a6c8782ebc57f0b8bda241b4b37ac02b99ffc58dcfddea09061"
V1_FAILURE_LOG_SHA256 = "67b28de48a8cde96c48f285fdcd680ccb22bf7212cd9727305ed620845a417b9"
AUDIT_MANIFEST_SHA = v1.AUDIT_MANIFEST_SHA
CANDIDATE_CHECKPOINT_SHA = v1.CANDIDATE_CHECKPOINT_SHA
SUPPORT_MANIFEST_SHA = v1.SUPPORT_MANIFEST_SHA
D1_DATASET_SHA = v1.D1_DATASET_SHA
D1_MANIFEST_SHA = v1.D1_MANIFEST_SHA
OBSERVATION_DIM = v1.OBSERVATION_DIM
TRAINING_SEED = v1.TRAINING_SEED
EPOCHS = v1.EPOCHS
SUPPORT_BATCH_SIZE = v1.SUPPORT_BATCH_SIZE
D1_BATCH_SIZE = v1.D1_BATCH_SIZE
LEARNING_RATE = v1.LEARNING_RATE
GRADIENT_CLIP_NORM = v1.GRADIENT_CLIP_NORM
SUPPORT_THRESHOLDS = dict(v1.SUPPORT_THRESHOLDS)
D1_THRESHOLDS = dict(v1.D1_THRESHOLDS)


RepairInputError = v1.RepairInputError
_require = v1._require
_sha256 = v1._sha256
_sha = _sha256
validate_audit_manifest = v1.validate_audit_manifest
_set_adapter_only_trainable = v1._set_adapter_only_trainable
_seed_everything = v1._seed_everything
_observations = v1._observations
_distribution_logits = v1._distribution_logits
_logit_invariance = v1._logit_invariance
_critic_state = v1._critic_state
_state_equal = v1._state_equal
_changed_parameter_names = v1._changed_parameter_names
_batch_indices = v1._batch_indices
_finite_gradients = v1._finite_gradients
repair_threshold_checks = v1.repair_threshold_checks
_load_inputs = v1._load_inputs


def protocol() -> dict[str, Any]:
    """Load and mechanically validate the V2 protocol bindings."""
    _require(PROTOCOL.is_file(), f"repair protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "V2 repair protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "V2 repair protocol must be a mapping")
    _require(value.get("protocol") == "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V2",
             "V2 repair protocol id changed")
    predecessor = value.get("predecessor", {})
    _require(predecessor.get("protocol") == "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V1",
             "V1 predecessor protocol binding changed")
    _require(predecessor.get("protocol_sha256") == V1_PROTOCOL_SHA256,
             "V1 predecessor protocol SHA binding changed")
    _require(predecessor.get("failure_log_sha256") == V1_FAILURE_LOG_SHA256,
             "V1 failure-log SHA binding changed")
    bindings = value.get("bindings", {})
    expected_bindings = {
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "audit_manifest_sha256": AUDIT_MANIFEST_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA,
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "observation_dimension": OBSERVATION_DIM,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"V2 repair binding changed: {key}")
    architecture = value.get("architecture", {})
    expected_architecture = {
        "id": "v4_2_hierarchical_gate_residual_adapter_v1",
        "base": "HierarchicalActorCentralCriticV42",
        "adapter": "mlp_88_to_2_hidden_96_zero_final_affine",
        "trainable_scope": "gate_adapter_only",
        "proposal_unchanged": True,
        "critic_unchanged": True,
    }
    for key, expected in expected_architecture.items():
        _require(architecture.get(key) == expected, f"V2 architecture binding changed: {key}")
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
            _require(float(actual) == expected, f"V2 training binding changed: {key}")
        else:
            _require(int(actual) == expected, f"V2 training binding changed: {key}")
    _require(training.get("steps_per_epoch") == "max_ceil_support_or_d1_batches",
             "V2 steps-per-epoch binding changed")
    _require(training.get("reshuffle_each_epoch") is True, "V2 reshuffle binding changed")
    _require(training.get("cycle_shorter_source") is True, "V2 source-cycling binding changed")
    _require(training.get("optimizer") == "Adam", "V2 optimizer binding changed")
    _require(training.get("objective") == "0.5_support_CE_plus_0.5_D1_CE_plus_0.5_D1_anchor_KL",
             "V2 objective changed")
    _require(training.get("anchor_KL") == "categorical_KL_frozen_candidate_to_repaired",
             "V2 anchor-KL binding changed")
    for key in ("dev_gradients", "d1_selection", "early_stopping"):
        _require(training.get(key) is False, f"V2 training restriction changed: {key}")
    _require(training.get("checkpoint_selection") == "final_epoch_only",
             "V2 checkpoint selection changed")
    stability = value.get("numerical_stability", {})
    expected_stability = {
        "changed_training_component": "d1_anchor_KL",
        "candidate_log_probability_dtype": "float64",
        "repaired_log_probability_dtype": "float64",
        "probability_source": "exp_candidate_log_softmax",
        "categorical_from_float32_probabilities": False,
        "finite_component_checks": True,
        "failure_context": "epoch_step_finite_flags_component_values",
    }
    for key, expected in expected_stability.items():
        _require(stability.get(key) == expected, f"V2 numerical-stability binding changed: {key}")
    _require(stability.get("stable_formula") ==
             "sum(exp(candidate_logp) * (candidate_logp - repaired_logp), dim=-1).mean()",
             "V2 stable-KL formula binding changed")
    thresholds = value.get("thresholds", {})
    for name, configured in (("support_dev", SUPPORT_THRESHOLDS), ("d1_anchor", D1_THRESHOLDS)):
        section = thresholds.get(name, {})
        for key, expected in configured.items():
            _require(float(section.get(key)) == expected, f"V2 threshold changed: {name}.{key}")
    return value


def stable_anchor_kl_from_logits(
    candidate_logits: torch.Tensor,
    repaired_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute candidate-to-repaired categorical KL stably in float64."""
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
    """Persist the latest completed epoch aggregate for terminal monitoring."""
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    latest = dict(history[-1]) if history else None
    payload = {
        "status": "RUNNING",
        "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V2",
        "latest_epoch": int(epoch),
        "completed_epochs": int(len(history)),
        "steps_per_epoch": int(steps_per_epoch),
        "latest": latest,
        "history": [dict(item) for item in history],
    }
    progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return progress_path


def train_repair(
    policy: GateResidualAdapterStudentPPOV42,
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
    """Run the V1-matched paired objective with stable float64 anchor KL."""
    _require(support_rows and d1_rows, "repair sources must be nonempty")
    _require(epochs >= 1 and support_batch_size >= 1 and d1_batch_size >= 1,
             "invalid repair training budget")
    _seed_everything(seed)
    support_x = _observations(support_rows)
    support_y = torch.as_tensor([int(row["teacher_gate"]) for row in support_rows], dtype=torch.long)
    d1_x = _observations(d1_rows)
    d1_y = torch.as_tensor([int(row["teacher_gate"]) for row in d1_rows], dtype=torch.long)
    trainable_names, frozen_names = _set_adapter_only_trainable(policy)
    trainable_parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
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
                loss = 0.5 * support_ce + 0.5 * d1_ce + 0.5 * d1_anchor_kl
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
                        trainable_parameters,
                        float(gradient_clip_norm),
                        error_if_nonfinite=True,
                    )
                except Exception as exc:
                    raise _nonfinite_error(epoch, step, components, "gradient clipping failed") from exc
                if not bool(torch.isfinite(clipped.detach()).all().item()):
                    raise _nonfinite_error(epoch, step, components, "non-finite clipped gradient norm")
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
                raise FloatingPointError(
                    f"repair step failed; epoch={epoch} step={step} "
                    f"finite/context={_step_context(epoch, step, components)[0]} "
                    f"values={_step_context(epoch, step, components)[1]}: {exc}"
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
            try:
                _write_progress(progress_path, epoch=epoch, history=history, steps_per_epoch=steps_per_epoch)
            except Exception as exc:
                components = {name: torch.as_tensor(value) for name, value in aggregate.items()
                              if name != "epoch"}
                raise FloatingPointError(
                    f"progress write failed; epoch={epoch} step={steps_per_epoch - 1} "
                    f"finite/context={_step_context(epoch, steps_per_epoch - 1, components)[0]} "
                    f"values={_step_context(epoch, steps_per_epoch - 1, components)[1]}: {exc}"
                ) from exc
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
        "objective": "0.5_support_CE+0.5_D1_CE+0.5_D1_anchor_KL",
        "anchor_KL": "categorical_KL_frozen_candidate_to_repaired_float64_logits",
        "finite_losses": bool(finite_losses and all(np.isfinite(item["loss"]) for item in history)),
        "finite_gradients": bool(finite_gradients),
        "max_gradient_norm_before_clip": float(max_gradient_norm),
        "early_stopping": False,
        "selection_split": "none_final_epoch_only",
        "progress_path": str(progress_path) if progress_path is not None else None,
        "history": history,
    }


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
        "status": "FAIL",
        "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V2",
        "evidence_type": "newly_run",
        "exception": {
            "type": type(exception).__name__,
            "message": str(exception),
        },
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
        "v1_protocol_sha256": V1_PROTOCOL_SHA256,
        "v1_failure_log_sha256": V1_FAILURE_LOG_SHA256,
        "progress": str(progress_path) if progress_path is not None else None,
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
    """Run the fixed offline V2 residual-adapter repair qualification."""
    p = protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"repair output already exists: {output_path}")
    audit_manifest_value, support_rows, support_paths, d1_rows, candidate = _load_inputs(
        support_manifest, audit_manifest, candidate_checkpoint, d1_dataset, d1_manifest,
    )
    output_path.mkdir(parents=True, exist_ok=False)
    datasets: dict[str, Sequence[Mapping[str, Any]]] = {
        "train": support_rows["train"],
        "dev": support_rows["dev"],
        "d1": d1_rows,
    }
    baseline = audit.evaluate_model(candidate.model, datasets)
    adapter = migrate_base_policy_to_adapter(candidate, learning_rate=LEARNING_RATE)
    before_adapter_state = {
        name: value.detach().cpu().clone() for name, value in adapter.model.state_dict().items()
    }
    before_candidate_state = {
        name: value.detach().cpu().clone() for name, value in candidate.model.state_dict().items()
    }
    before_identity = _logit_invariance(candidate.model, adapter.model, datasets)
    _require(all(item["gate_bitwise_equal"] for name, item in before_identity.items() if name in datasets),
             "zero adapter does not preserve candidate gate logits")
    _require(before_identity["proposal_invariant"], "zero adapter does not preserve proposal logits")
    progress_path = output_path / "progress.json"
    try:
        training = train_repair(
            adapter,
            candidate.model,
            support_rows["train"],
            d1_rows,
            progress_path=progress_path,
        )
    except Exception as exc:
        write_failure_manifest(output_path, exc, progress_path=progress_path)
        raise
    final = audit.evaluate_model(adapter.model, datasets)
    after_invariance = _logit_invariance(candidate.model, adapter.model, datasets)
    changed = _changed_parameter_names(before_adapter_state, adapter.model.state_dict())
    base_common_changed = _changed_parameter_names(before_candidate_state, adapter.model.state_dict())
    adapter_state = {
        name: value.detach().cpu().clone() for name, value in adapter.model.state_dict().items()
    }
    critic_invariant = _state_equal(_critic_state(candidate.model), _critic_state(adapter.model))
    parameter_isolation = bool(changed) and all(name.startswith(ADAPTER_PREFIX + ".") for name in changed)
    invariants = {
        "before_identity": before_identity,
        "after": after_invariance,
        "proposal_invariant": bool(after_invariance["proposal_invariant"]),
        "critic_invariant": bool(critic_invariant),
        "changed_parameter_names": changed,
        "base_common_changed_parameter_names": base_common_changed,
        "parameter_isolation": parameter_isolation and not base_common_changed,
        "finite_losses": bool(training["finite_losses"]),
        "finite_gradients": bool(training["finite_gradients"]),
        "base_state_unchanged_except_adapter": not bool(base_common_changed),
        "adapter_state_parameter_count": len(adapter_state),
    }
    thresholds = repair_threshold_checks(final["dev"], final["d1"], invariants)
    repair_pass = bool(thresholds["all"])
    checkpoint_path = output_path / "gate_repair_v2_qualification_candidate.pt"
    save_checkpoint(
        checkpoint_path,
        adapter,
        audit.p2_config(6),
        stage="V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V2",
        architecture_id=adapter.architecture_id,
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
    result = {
        "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V2",
        "status": "PASS" if repair_pass else "FAIL",
        "evidence_type": "newly_run",
        "repair_training": "formal_offline_gate_repair",
        "repair_pass": repair_pass,
        "support_manifest": str(support_manifest),
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "support_files": {
            name: {"path": str(support_paths[name]), "sha256": _sha256(support_paths[name])}
            for name in ("train", "dev")
        },
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": AUDIT_MANIFEST_SHA,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA,
        "d1_dataset": str(d1_dataset),
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest": str(d1_manifest),
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "audit_binding_status": {
            "AUDIT_COMPLETE": audit_manifest_value["AUDIT_COMPLETE"],
            "audit_pass": audit_manifest_value["audit_pass"],
            "repair_preregistration_allowed": audit_manifest_value["repair_preregistration_allowed"],
        },
        "architecture_id": adapter.architecture_id,
        "baseline": baseline,
        "final": final,
        "training": training,
        "invariants": invariants,
        "thresholds": thresholds,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
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
        "v1_protocol_sha256": V1_PROTOCOL_SHA256,
        "v1_failure_log_sha256": V1_FAILURE_LOG_SHA256,
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    output_manifest = output_path / "manifest.json"
    result["manifest"] = str(output_manifest)
    output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "stage": result["stage"],
        "repair_pass": repair_pass,
        "STUDENT_QUALIFICATION_ALLOWED": repair_pass,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
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
