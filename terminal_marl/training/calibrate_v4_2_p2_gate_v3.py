"""Offline fixed-temperature calibration for the V4.2 gate adapter.

This runner consumes the failed-but-qualified V2 repair evidence and applies a
fixed post-V2 development temperature diagnostic.  It performs no optimizer
update, policy training, environment rollout, or Teacher call.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.algo.gate_adapter_v4_2 import (
    GateResidualAdapterStudentPPOV42,
)
from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    CALIBRATION_TEMPERATURE,
    TEMPERATURE_BUFFER,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
    migrate_v2_policy_to_calibrated,
)
from terminal_marl.core.checkpoints import checkpoint_metadata, load_checkpoint
from terminal_marl.training import audit_v4_2_p2_gate_learnability_v1 as audit
from terminal_marl.training import train_v4_2_p2_gate_repair_v2 as v2


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_calibration_v3.yaml")
PROTOCOL_SHA = "c5e0b50f645f488cf3d36784cd37ba97fa016cae11decb9ac149bf3b932ba946"
V2_PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_repair_v2.yaml")
V2_REPAIR_MANIFEST_SHA = "6586154aa8caa111d335f0f1f0658e264372242764e9adb8e874817b435064ce"
V2_CHECKPOINT_SHA = "91509526f9d26dc7560336c376891a7746be79adb5099229d58ccd3bc8a587e5"
V2_PROTOCOL_SHA = "5b56a47f0d816f59f537836ab504d8c369d77ec5bd54b9679bdb6aeeb7b1719f"
SUPPORT_MANIFEST_SHA = v2.SUPPORT_MANIFEST_SHA
D1_DATASET_SHA = v2.D1_DATASET_SHA
D1_MANIFEST_SHA = v2.D1_MANIFEST_SHA
OBSERVATION_DIM = v2.OBSERVATION_DIM
TEMPERATURE_GRID = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


RepairInputError = v2.RepairInputError
_require = v2._require
_sha256 = v2._sha256
_sha = _sha256


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return True


def protocol() -> dict[str, Any]:
    """Load and validate the fixed V3 calibration protocol."""
    _require(PROTOCOL.is_file(), f"calibration protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "V3 calibration protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "V3 calibration protocol must be a mapping")
    _require(value.get("protocol") == "V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3",
             "V3 calibration protocol id changed")
    predecessor = value.get("predecessor", {})
    expected_predecessor = {
        "protocol": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V2",
        "repair_manifest_sha256": V2_REPAIR_MANIFEST_SHA,
        "checkpoint_sha256": V2_CHECKPOINT_SHA,
        "protocol_sha256": V2_PROTOCOL_SHA,
    }
    for key, expected in expected_predecessor.items():
        _require(predecessor.get(key) == expected, f"V2 predecessor binding changed: {key}")
    _require(V2_PROTOCOL.is_file(), f"V2 protocol missing: {V2_PROTOCOL}")
    _require(_sha256(V2_PROTOCOL) == V2_PROTOCOL_SHA, "V2 protocol SHA mismatch")
    bindings = value.get("bindings", {})
    expected_bindings = {
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "observation_dimension": OBSERVATION_DIM,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"V3 binding changed: {key}")
    architecture = value.get("architecture", {})
    expected_architecture = {
        "id": CALIBRATED_ARCHITECTURE_ID,
        "base": "v4_2_hierarchical_gate_residual_adapter_v1",
        "calibration": "scalar_gate_temperature_buffer",
        "temperature_buffer": TEMPERATURE_BUFFER,
        "temperature": CALIBRATION_TEMPERATURE,
        "proposal_unchanged": True,
        "critic_unchanged": True,
        "raw_gate_argmax_unchanged": True,
        "common_state_strict_copy": True,
    }
    for key, expected in expected_architecture.items():
        actual = architecture.get(key)
        if isinstance(expected, float):
            _require(float(actual) == expected, f"V3 architecture changed: {key}")
        else:
            _require(actual == expected, f"V3 architecture changed: {key}")
    selection = value.get("temperature_selection", {})
    _require(float(selection.get("temperature")) == CALIBRATION_TEMPERATURE,
             "V3 selected temperature changed")
    _require(tuple(float(item) for item in selection.get("preregistered_grid", ())) == TEMPERATURE_GRID,
             "V3 temperature grid changed")
    _require(selection.get("selection_source") == "post_v2_development_diagnostic_on_frozen_dev",
             "V3 temperature selection source changed")
    _require(selection.get("qualification_seeds_untouched") is True,
             "V3 qualification-seed protection changed")
    training = value.get("training", {})
    for key in ("training_called", "optimizer_called", "environment_rollout", "teacher_calls"):
        _require(training.get(key) is False, f"V3 training restriction changed: {key}")
    thresholds = value.get("thresholds", {})
    for name, configured in (("support_dev", v2.SUPPORT_THRESHOLDS), ("d1_anchor", v2.D1_THRESHOLDS)):
        section = thresholds.get(name, {})
        for key, expected in configured.items():
            _require(float(section.get(key)) == expected, f"V3 threshold changed: {name}.{key}")
    return value


def _validate_v2_threshold_shape(manifest: Mapping[str, Any]) -> None:
    thresholds = manifest.get("thresholds", {})
    support = thresholds.get("support_dev", {})
    support_names = ("macro_F1", "WAIT_recall", "RELEASE_recall", "brier", "binary_nll",
                     "every_interface_macro_F1")
    failed = [name for name in support_names if support.get(name) is not True]
    _require(failed == ["binary_nll"],
             f"V2 support-dev failure shape changed: failed={failed}")
    _require(support.get("binary_nll") is False, "V2 binary-NLL failure is not explicit")
    _require(support.get("all") is False, "V2 support-dev gate was not failed")
    interfaces = support.get("interface", {})
    _require(set(interfaces) == set(audit.INTERFACES), "V2 interface threshold shape changed")
    _require(all(value.get("macro_F1") is True for value in interfaces.values()),
             "V2 interface threshold failure detected")
    d1 = thresholds.get("d1_anchor", {})
    d1_names = ("macro_F1", "WAIT_recall", "RELEASE_recall", "brier", "binary_nll")
    _require(all(d1.get(name) is True for name in d1_names), "V2 D1 threshold failure detected")
    _require(d1.get("all") is True, "V2 D1 thresholds did not all pass")
    invariants = thresholds.get("invariants", {})
    invariant_names = ("proposal_invariant", "critic_invariant", "parameter_isolation",
                       "finite_losses", "finite_gradients")
    _require(all(invariants.get(name) is True for name in invariant_names),
             "V2 invariant threshold failure detected")
    _require(invariants.get("all") is True, "V2 invariant thresholds did not all pass")


def validate_v2_repair_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = V2_REPAIR_MANIFEST_SHA,
) -> dict[str, Any]:
    """Require the precise failed V2 repair shape eligible for calibration."""
    manifest_path = Path(path)
    _require(manifest_path.is_file(), f"V2 repair manifest missing: {manifest_path}")
    if expected_sha256 is not None:
        _require(_sha256(manifest_path) == expected_sha256, "V2 repair manifest SHA mismatch")
    manifest = _read_json(manifest_path)
    _require(manifest.get("stage") == "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V2",
             "V2 repair manifest stage mismatch")
    _require(manifest.get("status") == "FAIL", "V2 repair manifest status is not FAIL")
    _require(manifest.get("repair_pass") is False, "V2 repair unexpectedly passed")
    _require(manifest.get("STUDENT_QUALIFICATION_ALLOWED") is False,
             "V2 repair unexpectedly authorized qualification")
    _require(manifest.get("PPO_ALLOWED") is False, "V2 repair unlocked PPO")
    _require(manifest.get("GATE_REPAIR_ALLOWED") is False, "V2 repair unlocked gate repair")
    _require(manifest.get("curriculum_promotion_allowed") is False,
             "V2 repair unlocked curriculum promotion")
    _require(manifest.get("checkpoint_promotable") is False, "V2 repair checkpoint is promotable")
    _require(manifest.get("qualification_only") is True, "V2 repair was not qualification-only")
    _require(manifest.get("checkpoint_role") == "qualification_candidate",
             "V2 repair checkpoint role changed")
    _require(manifest.get("architecture_id") == v2.GateResidualAdapterStudentPPOV42.architecture_id,
             "V2 repair architecture changed")
    for key, expected in {
        "protocol_sha256": V2_PROTOCOL_SHA,
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "checkpoint_sha256": V2_CHECKPOINT_SHA,
    }.items():
        _require(manifest.get(key) == expected, f"V2 repair binding changed: {key}")
    _validate_v2_threshold_shape(manifest)
    return manifest


def _load_v2_checkpoint(path: str | Path) -> tuple[GateResidualAdapterStudentPPOV42, dict[str, Any]]:
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"V2 checkpoint missing: {checkpoint}")
    _require(_sha256(checkpoint) == V2_CHECKPOINT_SHA, "V2 checkpoint SHA mismatch")
    policy = GateResidualAdapterStudentPPOV42(OBSERVATION_DIM)
    payload = load_checkpoint(checkpoint, policy, audit.p2_config(6), load_optimizer=False)
    _require(payload.get("stage") == "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V2",
             "V2 checkpoint stage mismatch")
    _require(payload.get("architecture_id") == policy.architecture_id,
             "V2 checkpoint architecture mismatch")
    _require(payload.get("qualification_only") is True, "V2 checkpoint was not qualification-only")
    _require(payload.get("checkpoint_role") == "qualification_candidate",
             "V2 checkpoint role mismatch")
    _require(payload.get("checkpoint_promotable") is False, "V2 checkpoint is promotable")
    _require(payload.get("PPO_ALLOWED") is False, "V2 checkpoint unlocked PPO")
    _require(payload.get("GATE_REPAIR_ALLOWED") is False, "V2 checkpoint unlocked gate repair")
    _require(payload.get("curriculum_promotion_allowed") is False,
             "V2 checkpoint unlocked curriculum promotion")
    _require(payload.get("STUDENT_QUALIFICATION_ALLOWED") is False,
             "V2 checkpoint unexpectedly authorized qualification")
    policy.model.eval()
    return policy, payload


def _load_inputs(
    v2_repair_manifest: str | Path,
    v2_checkpoint: str | Path,
    support_manifest: str | Path,
    d1_dataset: str | Path,
    d1_manifest: str | Path,
) -> tuple[dict[str, Any], GateResidualAdapterStudentPPOV42, dict[str, Any],
           dict[str, list[dict[str, Any]]], dict[str, Path], list[dict[str, Any]]]:
    v2_manifest_value = validate_v2_repair_manifest(v2_repair_manifest)
    v2_policy, checkpoint_payload = _load_v2_checkpoint(v2_checkpoint)
    _, support_rows, support_paths = audit.validate_support_manifest(support_manifest)
    _, d1_rows = audit.validate_d1_anchor(d1_dataset, d1_manifest)
    return v2_manifest_value, v2_policy, checkpoint_payload, support_rows, support_paths, d1_rows


def _distribution_arrays(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations, _ = audit._row_arrays(rows)
    model.eval()
    with torch.no_grad():
        gate, proposal = model.distributions(observations)
    return (
        gate.logits.detach().cpu().numpy().copy(),
        gate.logits.argmax(dim=-1).detach().cpu().numpy().copy(),
        proposal.logits.detach().cpu().numpy().copy(),
    )


def decision_and_logit_invariance(
    before_model: torch.nn.Module,
    after_model: torch.nn.Module,
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Verify raw gate argmax, proposal logits, and calibration finiteness."""
    result: dict[str, Any] = {}
    raw_argmax_all = True
    proposal_all = True
    for name, rows in datasets.items():
        before_gate, before_argmax, before_proposal = _distribution_arrays(before_model, rows)
        after_gate, after_argmax, after_proposal = _distribution_arrays(after_model, rows)
        _require(before_gate.shape == after_gate.shape, f"gate logits shape changed for {name}")
        _require(before_proposal.shape == after_proposal.shape, f"proposal logits shape changed for {name}")
        gate_argmax_equal = bool(np.array_equal(before_argmax, after_argmax))
        proposal_equal = bool(np.array_equal(before_proposal, after_proposal))
        raw_argmax_all = raw_argmax_all and gate_argmax_equal
        proposal_all = proposal_all and proposal_equal
        result[name] = {
            "raw_gate_argmax_equal": gate_argmax_equal,
            "proposal_bitwise_equal": proposal_equal,
            "raw_gate_argmax_samples": int(len(before_argmax)),
            "proposal_max_abs_delta": float(np.max(np.abs(
                after_proposal.astype(np.float64) - before_proposal.astype(np.float64)
            ))) if before_proposal.size else 0.0,
        }
    result["raw_gate_argmax_invariant"] = bool(raw_argmax_all)
    result["proposal_logits_invariant"] = bool(proposal_all)
    return result


def common_state_invariance(
    source_model: torch.nn.Module,
    calibrated_model: torch.nn.Module,
) -> dict[str, Any]:
    """Check exact common state copy and the single new temperature buffer."""
    source_state = source_model.state_dict()
    calibrated_state = calibrated_model.state_dict()
    source_keys = set(source_state)
    calibrated_keys = set(calibrated_state)
    common = source_keys & calibrated_keys
    target_only = sorted(calibrated_keys - source_keys)
    source_only = sorted(source_keys - calibrated_keys)
    common_equal = bool(not source_only) and all(
        torch.equal(source_state[key].detach().cpu(), calibrated_state[key].detach().cpu())
        for key in common
    )
    parameter_names_equal = {
        name for name, _ in source_model.named_parameters()
    } == {name for name, _ in calibrated_model.named_parameters()}
    return {
        "common_state_bitwise_equal": bool(common_equal),
        "common_state_names": sorted(common),
        "new_buffer_names": target_only,
        "unexpected_source_state_names": source_only,
        "only_new_temperature_buffer": bool(target_only == [TEMPERATURE_BUFFER] and parameter_names_equal),
        "parameter_names_equal": bool(parameter_names_equal),
    }


def _critic_state_equal(source_model: torch.nn.Module, calibrated_model: torch.nn.Module) -> bool:
    return v2._state_equal(v2._critic_state(source_model), v2._critic_state(calibrated_model))


def calibration_threshold_checks(
    support_dev: Mapping[str, Any],
    d1_anchor: Mapping[str, Any],
    invariants: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply V2 metric gates with calibration-only invariant evidence."""
    support_checks = {
        "macro_F1": float(support_dev.get("macro_F1", float("nan"))) >= v2.SUPPORT_THRESHOLDS["macro_F1_min"],
        "WAIT_recall": float(support_dev.get("WAIT_recall", float("nan"))) >= v2.SUPPORT_THRESHOLDS["WAIT_recall_min"],
        "RELEASE_recall": float(support_dev.get("RELEASE_recall", float("nan"))) >= v2.SUPPORT_THRESHOLDS["RELEASE_recall_min"],
        "brier": float(support_dev.get("brier", float("nan"))) <= v2.SUPPORT_THRESHOLDS["brier_max"],
        "binary_nll": float(support_dev.get("binary_nll", float("nan"))) <= v2.SUPPORT_THRESHOLDS["binary_nll_max"],
    }
    interface_checks = {
        interface: float(support_dev.get("interface", {}).get(interface, {}).get("macro_F1", float("nan")))
        >= v2.SUPPORT_THRESHOLDS["interface_macro_F1_min"]
        for interface in audit.INTERFACES
    }
    support_checks["every_interface_macro_F1"] = bool(interface_checks) and all(interface_checks.values())
    d1_checks = {
        "macro_F1": float(d1_anchor.get("macro_F1", float("nan"))) >= v2.D1_THRESHOLDS["macro_F1_min"],
        "WAIT_recall": float(d1_anchor.get("WAIT_recall", float("nan"))) >= v2.D1_THRESHOLDS["WAIT_recall_min"],
        "RELEASE_recall": float(d1_anchor.get("RELEASE_recall", float("nan"))) >= v2.D1_THRESHOLDS["RELEASE_recall_min"],
        "brier": float(d1_anchor.get("brier", float("nan"))) <= v2.D1_THRESHOLDS["brier_max"],
        "binary_nll": float(d1_anchor.get("binary_nll", float("nan"))) <= v2.D1_THRESHOLDS["binary_nll_max"],
    }
    invariant_checks = {
        "proposal_logits_invariant": bool(invariants.get("proposal_logits_invariant", False)),
        "critic_state_invariant": bool(invariants.get("critic_state_invariant", False)),
        "common_state_bitwise_equal": bool(invariants.get("common_state_bitwise_equal", False)),
        "only_new_temperature_buffer": bool(invariants.get("only_new_temperature_buffer", False)),
        "temperature_finite_positive": bool(invariants.get("temperature_finite_positive", False)),
        "finite_metrics": bool(invariants.get("finite_metrics", False)),
        "no_training_or_optimizer": bool(invariants.get("no_training_or_optimizer", False)),
    }
    return {
        "support_dev": {
            **support_checks,
            "interface": interface_checks,
            "all": all(support_checks.values()),
        },
        "d1_anchor": {**d1_checks, "all": all(d1_checks.values())},
        "invariants": {**invariant_checks, "all": all(invariant_checks.values())},
        "all": bool(
            all(support_checks.values())
            and all(d1_checks.values())
            and all(invariant_checks.values())
        ),
    }


threshold_checks = calibration_threshold_checks
repair_threshold_checks = calibration_threshold_checks


def _save_calibration_checkpoint(
    path: Path,
    policy: TemperatureCalibratedGateResidualAdapterStudentPPOV42,
    *,
    base_checkpoint_sha256: str,
    source_hashes: Mapping[str, str],
    qualification_pass: bool,
) -> None:
    """Save model metadata only; calibration performs no optimizer update."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": checkpoint_metadata(audit.p2_config(6), policy_kind="offline_gate_calibration"),
        "model": {name: value.detach().cpu() for name, value in policy.model.state_dict().items()},
        "optimizer_saved": False,
        "stage": "V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3",
        "architecture_id": policy.architecture_id,
        "temperature": float(policy.model.gate_temperature.detach().cpu()),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "source_hashes": dict(source_hashes),
        "qualification_only": True,
        "checkpoint_role": "qualification_candidate",
        "qualification_pass": bool(qualification_pass),
        "STUDENT_QUALIFICATION_ALLOWED": bool(qualification_pass),
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "training_called": False,
        "optimizer_called": False,
        "environment_called": False,
        "teacher_called": False,
    }
    torch.save(payload, path)


def run(
    output: str | Path,
    v2_repair_manifest: str | Path,
    v2_checkpoint: str | Path,
    support_manifest: str | Path,
    d1_dataset: str | Path,
    d1_manifest: str | Path,
) -> dict[str, Any]:
    """Run fixed-T V3 calibration and write one qualification candidate on pass."""
    protocol_value = protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"calibration output already exists: {output_path}")
    v2_manifest_value, v2_policy, checkpoint_payload, support_rows, support_paths, d1_rows = _load_inputs(
        v2_repair_manifest,
        v2_checkpoint,
        support_manifest,
        d1_dataset,
        d1_manifest,
    )
    output_path.mkdir(parents=True, exist_ok=False)
    datasets: dict[str, Sequence[Mapping[str, Any]]] = {
        "train": support_rows["train"],
        "dev": support_rows["dev"],
        "d1": d1_rows,
    }
    calibrated = migrate_v2_policy_to_calibrated(v2_policy, temperature=CALIBRATION_TEMPERATURE)
    invariance = decision_and_logit_invariance(v2_policy.model, calibrated.model, datasets)
    common = common_state_invariance(v2_policy.model, calibrated.model)
    critic_invariant = _critic_state_equal(v2_policy.model, calibrated.model)
    temperature_finite_positive = bool(
        torch.isfinite(calibrated.model.gate_temperature).all().item()
        and bool((calibrated.model.gate_temperature > 0).all().item())
        and float(calibrated.model.gate_temperature.detach().cpu()) == CALIBRATION_TEMPERATURE
    )
    baseline = audit.evaluate_model(v2_policy.model, datasets)
    calibrated_metrics = audit.evaluate_model(calibrated.model, datasets)
    finite_metrics = _all_finite(calibrated_metrics)
    invariants = {
        **invariance,
        **common,
        "critic_state_invariant": bool(critic_invariant),
        "temperature_finite_positive": temperature_finite_positive,
        "finite_metrics": bool(finite_metrics),
        "training_called": False,
        "optimizer_called": False,
        "environment_called": False,
        "teacher_called": False,
    }
    invariants["no_training_or_optimizer"] = bool(
        not invariants["training_called"] and not invariants["optimizer_called"]
    )
    thresholds = calibration_threshold_checks(
        calibrated_metrics["dev"], calibrated_metrics["d1"], invariants,
    )
    calibration_pass = bool(thresholds["all"])
    source_hashes = {
        "runner": _sha256(Path(__file__)),
        "protocol": _sha256(PROTOCOL),
        "v2_repair_manifest": V2_REPAIR_MANIFEST_SHA,
        "v2_checkpoint": V2_CHECKPOINT_SHA,
        "v2_protocol": V2_PROTOCOL_SHA,
    }
    checkpoint_path: Path | None = None
    if calibration_pass:
        checkpoint_path = output_path / "gate_temperature_calibration_v3_qualification_candidate.pt"
        _save_calibration_checkpoint(
            checkpoint_path,
            calibrated,
            base_checkpoint_sha256=V2_CHECKPOINT_SHA,
            source_hashes=source_hashes,
            qualification_pass=calibration_pass,
        )
    result = {
        "stage": "V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3",
        "status": "PASS" if calibration_pass else "FAIL",
        "calibration_pass": calibration_pass,
        "evidence_type": "newly_run",
        "temperature": CALIBRATION_TEMPERATURE,
        "temperature_grid": list(TEMPERATURE_GRID),
        "temperature_selection_source": "post_v2_development_diagnostic_on_frozen_dev",
        "temperature_diagnostic": protocol_value["temperature_selection"]["diagnostic_metrics"],
        "qualification_seeds_untouched": True,
        "v2_repair_manifest": str(v2_repair_manifest),
        "v2_repair_manifest_sha256": V2_REPAIR_MANIFEST_SHA,
        "v2_checkpoint": str(v2_checkpoint),
        "v2_checkpoint_sha256": V2_CHECKPOINT_SHA,
        "v2_protocol_sha256": V2_PROTOCOL_SHA,
        "support_manifest": str(support_manifest),
        "support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "support_files": {
            name: {"path": str(support_paths[name]), "sha256": _sha256(support_paths[name])}
            for name in ("train", "dev")
        },
        "d1_dataset": str(d1_dataset),
        "d1_dataset_sha256": D1_DATASET_SHA,
        "d1_manifest": str(d1_manifest),
        "d1_manifest_sha256": D1_MANIFEST_SHA,
        "v2_status": {
            "status": v2_manifest_value["status"],
            "repair_pass": v2_manifest_value["repair_pass"],
            "STUDENT_QUALIFICATION_ALLOWED": v2_manifest_value["STUDENT_QUALIFICATION_ALLOWED"],
        },
        "checkpoint_payload": {
            "stage": checkpoint_payload["stage"],
            "architecture_id": checkpoint_payload["architecture_id"],
            "qualification_only": checkpoint_payload["qualification_only"],
            "checkpoint_promotable": checkpoint_payload["checkpoint_promotable"],
        },
        "architecture_id": calibrated.architecture_id,
        "baseline": baseline,
        "calibrated": calibrated_metrics,
        "invariants": invariants,
        "thresholds": thresholds,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_role": "qualification_candidate" if checkpoint_path is not None else None,
        "qualification_only": True,
        "STUDENT_QUALIFICATION_ALLOWED": calibration_pass,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "training_called": False,
        "optimizer_called": False,
        "environment_called": False,
        "teacher_called": False,
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "source_hashes": source_hashes,
    }
    output_manifest = output_path / "manifest.json"
    result["manifest"] = str(output_manifest)
    output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "stage": result["stage"],
        "calibration_pass": calibration_pass,
        "STUDENT_QUALIFICATION_ALLOWED": calibration_pass,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v2-repair-manifest", type=Path, required=True)
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--d1-dataset", type=Path, required=True)
    parser.add_argument("--d1-manifest", type=Path, required=True)
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    _main()
