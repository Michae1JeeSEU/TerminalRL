"""Offline P2 V4.2 gate residual-adapter repair qualification.

This runner consumes only the frozen V2 support artifact, the completed
learnability audit, the candidate checkpoint, and the immutable D1 anchor.  It
does not reset an environment or invoke the Teacher.  The only optimized
parameters are the newly introduced ``gate_adapter.*`` parameters.
"""
from __future__ import annotations

import argparse
import hashlib
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
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import yaml

from terminal_marl.algo.gate_adapter_v4_2 import (
    ADAPTER_PREFIX,
    GateResidualAdapterStudentPPOV42,
    migrate_base_policy_to_adapter,
)
from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training import audit_v4_2_p2_gate_learnability_v1 as audit


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_repair_v1.yaml")
PROTOCOL_SHA = "88bdc0578d021a6c8782ebc57f0b8bda241b4b37ac02b99ffc58dcfddea09061"
AUDIT_MANIFEST_SHA = "e2affdddb4628908c2fc7a766f6ea47fc1a7263b44456fde08d8ce8875b3ed22"
CANDIDATE_CHECKPOINT_SHA = audit.CANDIDATE_CHECKPOINT_SHA
SUPPORT_MANIFEST_SHA = audit.SUPPORT_MANIFEST_SHA
D1_DATASET_SHA = audit.D1_DATASET_SHA
D1_MANIFEST_SHA = audit.D1_MANIFEST_SHA
OBSERVATION_DIM = audit.OBSERVATION_DIM
TRAINING_SEED = 779101
EPOCHS = 60
SUPPORT_BATCH_SIZE = 128
D1_BATCH_SIZE = 128
LEARNING_RATE = 3e-4
GRADIENT_CLIP_NORM = 1.0
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
    _require(PROTOCOL.is_file(), f"repair protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "repair protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "repair protocol must be a mapping")
    _require(value.get("protocol") == "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V1", "repair protocol id changed")
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
        _require(bindings.get(key) == expected, f"repair binding changed: {key}")
    training = value.get("training", {})
    _require(int(training.get("seed")) == TRAINING_SEED, "repair seed changed")
    _require(int(training.get("epochs")) == EPOCHS, "repair epochs changed")
    _require(int(training.get("support_batch_size")) == SUPPORT_BATCH_SIZE, "support batch size changed")
    _require(int(training.get("d1_batch_size")) == D1_BATCH_SIZE, "D1 batch size changed")
    _require(float(training.get("learning_rate")) == LEARNING_RATE, "repair learning rate changed")
    _require(float(training.get("gradient_clip_norm")) == GRADIENT_CLIP_NORM,
             "repair gradient clip changed")
    thresholds = value.get("thresholds", {})
    for name, configured in (("support_dev", SUPPORT_THRESHOLDS), ("d1_anchor", D1_THRESHOLDS)):
        section = thresholds.get(name, {})
        for key, expected in configured.items():
            _require(float(section.get(key)) == expected, f"repair threshold changed: {name}.{key}")
    return value


def validate_audit_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = AUDIT_MANIFEST_SHA,
) -> dict[str, Any]:
    """Require the completed passing learnability audit before repair."""
    manifest_path = Path(path)
    _require(manifest_path.is_file(), f"audit manifest missing: {manifest_path}")
    if expected_sha256 is not None:
        _require(_sha256(manifest_path) == expected_sha256, "audit manifest SHA mismatch")
    manifest = _read_json(manifest_path)
    _require(manifest.get("stage") == "V4_2_P2_GATE_LEARNABILITY_AUDIT_V1",
             "audit manifest stage mismatch")
    _require(manifest.get("AUDIT_COMPLETE") is True, "learnability audit is incomplete")
    _require(manifest.get("audit_pass") is True, "learnability audit did not pass")
    _require(manifest.get("repair_preregistration_allowed") is True,
             "repair was not preregistered by the audit")
    _require(manifest.get("support_manifest_sha256") == SUPPORT_MANIFEST_SHA,
             "audit support binding mismatch")
    _require(manifest.get("candidate_checkpoint_sha256") == CANDIDATE_CHECKPOINT_SHA,
             "audit candidate binding mismatch")
    _require(manifest.get("d1_dataset_sha256") == D1_DATASET_SHA,
             "audit D1 dataset binding mismatch")
    _require(manifest.get("d1_manifest_sha256") == D1_MANIFEST_SHA,
             "audit D1 manifest binding mismatch")
    _require(manifest.get("GATE_REPAIR_ALLOWED") is False, "audit unlocked gate repair")
    _require(manifest.get("PPO_ALLOWED") is False, "audit unlocked PPO")
    return manifest


def _set_adapter_only_trainable(policy: GateResidualAdapterStudentPPOV42) -> tuple[list[str], list[str]]:
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
    max_delta = 0.0
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
        max_delta = max(max_delta, result[name]["proposal_max_abs_delta"])
    result["proposal_all_bitwise_equal"] = bool(all_equal)
    result["proposal_all_max_abs_delta"] = float(max_delta)
    result["proposal_invariant"] = bool(all_equal or max_delta == 0.0)
    return result


def _critic_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("critic_encoder.") or name.startswith("critic_head.")
    }


def _state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def _changed_parameter_names(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> list[str]:
    common = set(before) & set(after)
    _require(common == set(before) or set(before).issubset(set(after)), "state keys changed unexpectedly")
    return sorted(name for name in common if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu()))


def _batch_indices(permutation: np.ndarray, step: int, batch_size: int) -> np.ndarray:
    if len(permutation) == 0:
        raise RepairInputError("repair source is empty")
    starts = (int(step) * int(batch_size)) % len(permutation)
    offsets = np.arange(int(batch_size), dtype=np.int64)
    return permutation[(starts + offsets) % len(permutation)]


def _finite_gradients(parameters: Sequence[nn.Parameter]) -> tuple[bool, float]:
    norms: list[float] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            return False, float("nan")
        norms.append(float(parameter.grad.detach().norm().cpu()))
    return True, float(max(norms, default=0.0))


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
) -> dict[str, Any]:
    """Run the fixed paired support/D1 residual objective for all epochs."""
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
    # The policy owns the formal Adam optimizer.  Its frozen base parameters
    # carry no gradients and therefore acquire no optimizer state.
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
            support_indices = _batch_indices(support_permutation, step, support_batch_size)
            d1_indices = _batch_indices(d1_permutation, step, d1_batch_size)
            support_gate, _ = policy.model.distributions(support_x[support_indices])
            d1_gate, _ = policy.model.distributions(d1_x[d1_indices])
            support_ce = F.cross_entropy(support_gate.logits, support_y[support_indices])
            d1_ce = F.cross_entropy(d1_gate.logits, d1_y[d1_indices])
            # The target is always the frozen candidate distribution evaluated
            # on the exact same D1 batch as the repaired distribution.
            with torch.no_grad():
                candidate_gate, _ = frozen_candidate.distributions(d1_x[d1_indices])
                candidate_probs = candidate_gate.probs.detach()
            repaired_gate = d1_gate
            d1_anchor_kl = kl_divergence(
                Categorical(probs=candidate_probs),
                repaired_gate,
            ).mean()
            loss = 0.5 * support_ce + 0.5 * d1_ce + 0.5 * d1_anchor_kl
            if not bool(torch.isfinite(loss)):
                finite_losses = False
                raise FloatingPointError("non-finite repair loss")
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients_ok, gradient_norm = _finite_gradients(trainable_parameters)
            finite_gradients = finite_gradients and gradients_ok
            if not gradients_ok:
                raise FloatingPointError("non-finite repair gradients")
            clipped = nn.utils.clip_grad_norm_(trainable_parameters, float(gradient_clip_norm), error_if_nonfinite=True)
            max_gradient_norm = max(max_gradient_norm, float(clipped.detach().cpu()))
            policy.optimizer.step()
            epoch_values.append({
                "support_CE": float(support_ce.detach().cpu()),
                "d1_CE": float(d1_ce.detach().cpu()),
                "d1_anchor_KL": float(d1_anchor_kl.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "gradient_norm": gradient_norm,
            })
        history.append({
            "epoch": float(epoch),
            "support_CE": float(np.mean([item["support_CE"] for item in epoch_values])),
            "d1_CE": float(np.mean([item["d1_CE"] for item in epoch_values])),
            "d1_anchor_KL": float(np.mean([item["d1_anchor_KL"] for item in epoch_values])),
            "loss": float(np.mean([item["loss"] for item in epoch_values])),
            "gradient_norm": float(np.mean([item["gradient_norm"] for item in epoch_values])),
        })
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
        "anchor_KL": "categorical_KL_frozen_candidate_to_repaired",
        "finite_losses": bool(finite_losses and all(np.isfinite(item["loss"]) for item in history)),
        "finite_gradients": bool(finite_gradients),
        "max_gradient_norm_before_clip": float(max_gradient_norm),
        "early_stopping": False,
        "selection_split": "none_final_epoch_only",
        "history": history,
    }


def repair_threshold_checks(
    support_dev: Mapping[str, Any],
    d1_anchor: Mapping[str, Any],
    invariants: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply inclusive support/D1 gates and required invariants."""
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
        "parameter_isolation": bool(invariants.get("parameter_isolation", False)),
        "finite_losses": bool(invariants.get("finite_losses", False)),
        "finite_gradients": bool(invariants.get("finite_gradients", False)),
    }
    return {
        "support_dev": {**support_checks, "interface": interface_checks, "all": all(support_checks.values())},
        "d1_anchor": {**d1_checks, "all": all(d1_checks.values())},
        "invariants": {**invariant_checks, "all": all(invariant_checks.values())},
        "all": bool(all(support_checks.values()) and all(d1_checks.values()) and all(invariant_checks.values())),
    }


def _load_inputs(
    support_manifest: str | Path,
    audit_manifest: str | Path,
    candidate_checkpoint: str | Path,
    d1_dataset: str | Path,
    d1_manifest: str | Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Path], list[dict[str, Any]], Any]:
    audit_manifest_value = validate_audit_manifest(audit_manifest)
    _, support_rows, support_paths = audit.validate_support_manifest(support_manifest)
    _, d1_rows = audit.validate_d1_anchor(d1_dataset, d1_manifest)
    candidate = audit._load_candidate(candidate_checkpoint)
    return audit_manifest_value, support_rows, support_paths, d1_rows, candidate


def run(
    output: str | Path,
    support_manifest: str | Path,
    audit_manifest: str | Path,
    candidate_checkpoint: str | Path,
    d1_dataset: str | Path,
    d1_manifest: str | Path,
) -> dict[str, Any]:
    """Run the fixed offline residual-adapter repair qualification."""
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
    before_adapter_state = {name: value.detach().cpu().clone() for name, value in adapter.model.state_dict().items()}
    before_candidate_state = {name: value.detach().cpu().clone() for name, value in candidate.model.state_dict().items()}
    before_identity = _logit_invariance(candidate.model, adapter.model, datasets)
    _require(all(item["gate_bitwise_equal"] for name, item in before_identity.items() if name in datasets),
             "zero adapter does not preserve candidate gate logits")
    _require(before_identity["proposal_invariant"], "zero adapter does not preserve proposal logits")
    training = train_repair(adapter, candidate.model, support_rows["train"], d1_rows)
    final = audit.evaluate_model(adapter.model, datasets)
    after_invariance = _logit_invariance(candidate.model, adapter.model, datasets)
    changed = _changed_parameter_names(before_adapter_state, adapter.model.state_dict())
    base_common_changed = _changed_parameter_names(before_candidate_state, adapter.model.state_dict())
    adapter_state = {name: value.detach().cpu().clone() for name, value in adapter.model.state_dict().items()}
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
    checkpoint_path = output_path / "gate_repair_v1_qualification_candidate.pt"
    save_checkpoint(
        checkpoint_path,
        adapter,
        audit.p2_config(6),
        stage="V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V1",
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
        "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V1",
        "evidence_type": "newly_run",
        "repair_training": "formal_offline_gate_repair",
        "repair_pass": repair_pass,
        "support_manifest": str(support_manifest),
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "support_files": {name: {"path": str(support_paths[name]), "sha256": _sha256(support_paths[name])}
                          for name in ("train", "dev")},
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
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    output_manifest = output_path / "manifest.json"
    result["manifest"] = str(output_manifest)
    output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"repair_pass": repair_pass, "STUDENT_QUALIFICATION_ALLOWED": repair_pass,
                      "PPO_ALLOWED": False, "GATE_REPAIR_ALLOWED": False}, sort_keys=True), flush=True)
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
