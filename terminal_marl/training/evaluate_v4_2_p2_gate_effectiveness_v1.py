"""Once-only paired P2 gate effectiveness evaluation V1.

This evaluator is deliberately descriptive: it loads the repaired candidate
and frozen calibrated baseline without optimizer state, evaluates both beside
the nominal rule controller, and writes immutable evidence.  It contains no
training, update, Teacher, label, or checkpoint-promotion path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training import evaluate_v4_2_p2_v4 as generic
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_effectiveness_v1.yaml")
PROTOCOL_SHA = "5882361adcabaff4c0017cae73a8b95dc8029e204942bf4a83879981bd321d84"

EFFECTIVENESS_STAGE = "V4_2_P2_GATE_EFFECTIVENESS_EVALUATION_V1"
QUALIFICATION_MANIFEST_SHA = "7cd4f02c88dc953fe1545c613424e8c961c295d4f8c099456d6697f300088feb"
QUALIFICATION_PROTOCOL_SHA = "7ab9e2a84b9aea8da2ea975d98b4867de4a45bd5725bc72019b2e95ddbb75094"
REPAIR_MANIFEST_SHA = "e1e0883e355676593ab4d561aeb46fb37656629ec514ad6f01e12c3bbd17d410"
REPAIR_PROTOCOL_SHA = "f8a84632fea21e6789c8a40c0a22938cb5f0bdfcdb4ec9efd5e48304867778e8"
REPAIRED_CHECKPOINT_SHA = "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
BASELINE_MANIFEST_SHA = "a784ffa753d4588d9a0b997a8ee913010c128abfbf49c96b5e790f5bf6ae7272"
BASELINE_CHECKPOINT_SHA = "61e03e02e3fb204d2db4e814097543905f51e1a3772626fdac1d7bc02ab8ca6b"
ARCHITECTURE_ID = CALIBRATED_ARCHITECTURE_ID
TEMPERATURE = CALIBRATION_TEMPERATURE
OBSERVATION_DIM = 88
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"

FORMAL_SEEDS = (788001, 788002, 788003, 788004, 788005, 788006)
FORMAL_SCENARIOS = ("flow6", "flow8", "flow12", "flow6", "flow8", "flow12")
CONTROLLERS = ("candidate", "baseline", "rule")
SMOKE_SEED = 788000
SMOKE_SCENARIO = "flow6"
SMOKE_STEPS = 160
EXPECTED_FORMAL_RECORDS = 18
EXPECTED_FORMAL_REQUESTED = 52
SHIELD_ACTOR_FRACTION_MAX = 0.05
GAP_CLOSURE_MIN = 0.50
RESERVATION_MARKER = "p2_gate_effectiveness_v1_consumed.json"
OLD_QUALIFICATION_SEEDS = (782001, 782002, 782003, 782004, 782005, 782006)

# Reuse the reviewed geometry, replay, likelihood, and nominal-rule
# mechanics.  These aliases are intentionally exposed for focused tests.
_geometry_replay = generic._geometry_replay
_wait_metrics = generic._wait_metrics
_policy_tape_logprob_exact = generic._policy_tape_logprob_exact


class EffectivenessInputError(ValueError):
    """Raised when a frozen input binding is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EffectivenessInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_sha = _sha256


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        values = [_jsonable(item) for item in value]
        try:
            return sorted(values)
        except TypeError:
            return sorted(values, key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ))
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _print_result(value: Mapping[str, Any]) -> None:
    print(json.dumps(_jsonable(value), indent=2, sort_keys=True), flush=True)


def protocol() -> dict[str, Any]:
    """Load the byte-frozen effectiveness protocol and verify its bindings."""
    _require(PROTOCOL.is_file(), f"effectiveness protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA,
                 "effectiveness protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "effectiveness protocol must be a mapping")
    _require(value.get("protocol") == EFFECTIVENESS_STAGE,
             "effectiveness protocol id changed")
    predecessor = value.get("predecessor", {})
    for key, expected in {
        "qualification_manifest_sha256": QUALIFICATION_MANIFEST_SHA,
        "qualification_protocol_sha256": QUALIFICATION_PROTOCOL_SHA,
        "qualification_status": "PASS",
        "P2_GATE_STUDENT_QUALIFICATION": "PASS",
        "P2_EFFECTIVENESS_EVALUATION_ALLOWED": True,
        "repair_manifest_sha256": REPAIR_MANIFEST_SHA,
        "repair_protocol_sha256": REPAIR_PROTOCOL_SHA,
        "repaired_checkpoint_sha256": REPAIRED_CHECKPOINT_SHA,
        "repaired_checkpoint_stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
    }.items():
        _require(predecessor.get(key) == expected,
                 f"predecessor binding changed: {key}")
    bindings = value.get("bindings", {})
    for key, expected in {
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_dimension": OBSERVATION_DIM,
        "action_schema": ACTION_SCHEMA,
        "architecture_id": ARCHITECTURE_ID,
        "temperature": TEMPERATURE,
        "deterministic": True,
        "fixed_geometry": True,
        "natural_flow": True,
        "no_teacher": True,
        "no_labels": True,
        "no_updates": True,
    }.items():
        actual = bindings.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"evaluation binding changed: {key}")
    candidate = value.get("candidate", {})
    baseline = value.get("baseline", {})
    for section, expected in ((candidate, {
        "controller": "candidate", "checkpoint_sha256": REPAIRED_CHECKPOINT_SHA,
        "checkpoint_stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
        "checkpoint_role": "qualification_candidate", "architecture_id": ARCHITECTURE_ID,
        "temperature": TEMPERATURE, "load_optimizer": False,
    }), (baseline, {
        "controller": "baseline", "calibration_manifest_sha256": BASELINE_MANIFEST_SHA,
        "checkpoint_sha256": BASELINE_CHECKPOINT_SHA,
        "checkpoint_stage": "V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3",
        "checkpoint_role": "qualification_candidate", "architecture_id": ARCHITECTURE_ID,
        "temperature": TEMPERATURE, "load_optimizer": False,
    })):
        for key, expected in expected.items():
            actual = section.get(key)
            _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                     f"controller binding changed: {key}")
    formal = value.get("formal", {})
    for key, expected in {
        "seeds": list(FORMAL_SEEDS), "scenarios": list(FORMAL_SCENARIOS),
        "controllers": list(CONTROLLERS), "deterministic": True,
        "expected_records": EXPECTED_FORMAL_RECORDS,
        "expected_requests_per_controller": EXPECTED_FORMAL_REQUESTED,
        "historical_registry_before_first_reset": True,
        "reservation_marker": RESERVATION_MARKER, "reserve_once": True,
        "old_qualification_seeds_forbidden": list(OLD_QUALIFICATION_SEEDS),
    }.items():
        _require(formal.get(key) == expected, f"formal binding changed: {key}")
    smoke = value.get("smoke", {})
    for key, expected in {
        "seed": SMOKE_SEED, "scenario": SMOKE_SCENARIO, "steps": SMOKE_STEPS,
        "formal_marker": False, "authorization": False,
    }.items():
        _require(smoke.get(key) == expected, f"smoke binding changed: {key}")
    _require(float(value.get("effectiveness", {}).get("gap_closure_min")) == GAP_CLOSURE_MIN,
             "gap-closure threshold changed")
    return value


def _validate_manifest_sha(path: str | Path, expected: str | None) -> dict[str, Any]:
    manifest_path = Path(path)
    _require(manifest_path.is_file(), f"manifest missing: {manifest_path}")
    if expected is not None:
        _require(_sha256(manifest_path) == expected, f"manifest SHA mismatch: {manifest_path}")
    return _read_json(manifest_path)


def validate_qualification_manifest(
    path: str | Path, *, expected_sha256: str | None = QUALIFICATION_MANIFEST_SHA,
) -> dict[str, Any]:
    manifest = _validate_manifest_sha(path, expected_sha256)
    for key, expected in {
        "status": "PASS",
        "P2_GATE_STUDENT_QUALIFICATION": "PASS",
        "P2_EFFECTIVENESS_EVALUATION_ALLOWED": True,
        "protocol_sha256": QUALIFICATION_PROTOCOL_SHA,
        "repaired_checkpoint_sha256": REPAIRED_CHECKPOINT_SHA,
        "qualification_only": True,
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
    }.items():
        _require(manifest.get(key) == expected,
                 f"qualification manifest binding changed: {key}")
    return manifest


def validate_repair_manifest(
    path: str | Path, *, expected_sha256: str | None = REPAIR_MANIFEST_SHA,
) -> dict[str, Any]:
    manifest = _validate_manifest_sha(path, expected_sha256)
    for key, expected in {
        "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
        "status": "PASS", "repair_pass": True,
        "protocol_sha256": REPAIR_PROTOCOL_SHA,
        "checkpoint_sha256": REPAIRED_CHECKPOINT_SHA,
        "candidate_checkpoint_sha256": BASELINE_CHECKPOINT_SHA,
        "candidate_architecture_id": ARCHITECTURE_ID,
        "candidate_temperature": TEMPERATURE,
        "qualification_only": True,
        "STUDENT_QUALIFICATION_ALLOWED": True,
        "PPO_ALLOWED": False, "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False, "checkpoint_promotable": False,
    }.items():
        actual = manifest.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"repair manifest binding changed: {key}")
    return manifest


def validate_baseline_manifest(
    path: str | Path, *, expected_sha256: str | None = BASELINE_MANIFEST_SHA,
) -> dict[str, Any]:
    manifest = _validate_manifest_sha(path, expected_sha256)
    for key, expected in {
        "stage": "V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3",
        "status": "PASS", "calibration_pass": True,
        "checkpoint_sha256": BASELINE_CHECKPOINT_SHA,
        "architecture_id": ARCHITECTURE_ID,
        "temperature": TEMPERATURE,
        "qualification_only": True,
        "STUDENT_QUALIFICATION_ALLOWED": True,
        "PPO_ALLOWED": False, "checkpoint_promotable": False,
    }.items():
        actual = manifest.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"baseline manifest binding changed: {key}")
    return manifest


def _load_calibrated_checkpoint(
    path: str | Path, *, expected_sha256: str | None, expected_stage: str,
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
    if expected_sha256 is not None:
        _require(_sha256(checkpoint) == expected_sha256,
                 f"checkpoint SHA mismatch: {checkpoint}")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM, temperature=TEMPERATURE,
    )
    payload = load_checkpoint(checkpoint, policy, p2_config(6), load_optimizer=False)
    checkpoint_temperature = payload.get("temperature")
    if checkpoint_temperature is None:
        checkpoint_temperature = float(policy.model.gate_temperature.detach().cpu())
        payload = dict(payload)
        payload["temperature"] = checkpoint_temperature
    for key, expected in {
        "stage": expected_stage, "architecture_id": ARCHITECTURE_ID,
        "temperature": TEMPERATURE, "qualification_only": True,
        "checkpoint_role": "qualification_candidate", "checkpoint_promotable": False,
        "PPO_ALLOWED": False, "curriculum_promotion_allowed": False,
    }.items():
        actual = payload.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"checkpoint binding changed: {key}")
    _require(len(policy.optimizer.state) == 0, "checkpoint optimizer state was loaded")
    _require(float(policy.model.gate_temperature.detach().cpu()) == TEMPERATURE,
             "checkpoint temperature changed")
    policy.model.eval()
    return policy, payload


def load_candidate_checkpoint(
    path: str | Path, *, expected_sha256: str | None = REPAIRED_CHECKPOINT_SHA,
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    policy, payload = _load_calibrated_checkpoint(
        path, expected_sha256=expected_sha256,
        expected_stage="V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
    )
    _require(payload.get("qualification_pass") is True,
             "candidate checkpoint is not qualification-passing")
    return policy, payload


def load_baseline_checkpoint(
    path: str | Path, *, expected_sha256: str | None = BASELINE_CHECKPOINT_SHA,
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    policy, payload = _load_calibrated_checkpoint(
        path, expected_sha256=expected_sha256,
        expected_stage="V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3",
    )
    _require(payload.get("qualification_pass") is True,
             "baseline checkpoint is not calibration-passing")
    return policy, payload


def _load_inputs(
    qualification_manifest: str | Path,
    repair_manifest: str | Path,
    candidate_checkpoint: str | Path,
    baseline_manifest: str | Path,
    baseline_checkpoint: str | Path,
    *,
    expected_qualification_sha256: str | None = QUALIFICATION_MANIFEST_SHA,
    expected_repair_sha256: str | None = REPAIR_MANIFEST_SHA,
    expected_candidate_sha256: str | None = REPAIRED_CHECKPOINT_SHA,
    expected_baseline_manifest_sha256: str | None = BASELINE_MANIFEST_SHA,
    expected_baseline_sha256: str | None = BASELINE_CHECKPOINT_SHA,
) -> dict[str, Any]:
    qualification = validate_qualification_manifest(
        qualification_manifest, expected_sha256=expected_qualification_sha256,
    )
    repair = validate_repair_manifest(
        repair_manifest, expected_sha256=expected_repair_sha256,
    )
    baseline_manifest_value = validate_baseline_manifest(
        baseline_manifest, expected_sha256=expected_baseline_manifest_sha256,
    )
    candidate, candidate_payload = load_candidate_checkpoint(
        candidate_checkpoint, expected_sha256=expected_candidate_sha256,
    )
    baseline, baseline_payload = load_baseline_checkpoint(
        baseline_checkpoint, expected_sha256=expected_baseline_sha256,
    )
    _require(repair.get("checkpoint_sha256") == REPAIRED_CHECKPOINT_SHA,
             "repair manifest/candidate checkpoint binding mismatch")
    _require(qualification.get("repaired_checkpoint_sha256") == REPAIRED_CHECKPOINT_SHA,
             "qualification manifest/candidate checkpoint binding mismatch")
    _require(baseline_manifest_value.get("checkpoint_sha256") == BASELINE_CHECKPOINT_SHA,
             "baseline manifest/checkpoint binding mismatch")
    return {
        "qualification_manifest": qualification,
        "repair_manifest": repair,
        "baseline_manifest": baseline_manifest_value,
        "candidate": candidate,
        "candidate_payload": candidate_payload,
        "baseline": baseline,
        "baseline_payload": baseline_payload,
    }


def reservation_marker_path(candidate_checkpoint: str | Path) -> Path:
    return Path(candidate_checkpoint).parent / RESERVATION_MARKER


def reserve_formal_attempt(
    marker: str | Path, history_root: str | Path, output: str | Path,
    *, protocol_sha256: str = PROTOCOL_SHA,
) -> dict[str, Any]:
    """Build the historical registry and reserve the formal seeds once."""
    registry = historical_registry(Path(history_root), Path(output))
    overlap = sorted(set(FORMAL_SEEDS) & set(registry.get("seeds", set())))
    _require(not overlap, f"formal effectiveness seeds already consumed: {overlap}")
    payload = {
        "protocol": EFFECTIVENESS_STAGE,
        "protocol_sha256": protocol_sha256 or _sha256(PROTOCOL),
        "qualification_manifest_sha256": QUALIFICATION_MANIFEST_SHA,
        "repair_manifest_sha256": REPAIR_MANIFEST_SHA,
        "candidate_checkpoint_sha256": REPAIRED_CHECKPOINT_SHA,
        "baseline_checkpoint_sha256": BASELINE_CHECKPOINT_SHA,
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_scenarios": list(FORMAL_SCENARIOS),
        "controllers": list(CONTROLLERS),
        "old_qualification_seeds_forbidden": list(OLD_QUALIFICATION_SEEDS),
        "output": str(output),
        "reserved_before_first_reset": True,
    }
    reserve_once(Path(marker), payload)
    return {"historical": registry, "payload": payload, "marker": str(marker)}


def _student_result(policy: Any, config: Any, seed: int, output: Path, *, smoke: bool) -> dict[str, Any]:
    result = dict(generic.evaluate(policy, config, seed, output, smoke=smoke))
    geometry = result.get("geometry", {})
    result["controller_kind"] = "student"
    result["no_reverse_transfer"] = int(geometry.get("reverse_samples", 0)) == 0
    result["all_requests_complete"] = bool(
        result.get("terminal") and result.get("completed") == result.get("requests")
    )
    result["no_teacher"] = not bool(result.get("teacher_enabled"))
    result["no_labels"] = int(result.get("label_rows", 0)) == 0
    result["no_updates"] = result.get("training_called") is False
    result["shield_actor_fraction_gate"] = float(
        result.get("shield_actor_fraction", float("inf"))
    ) <= SHIELD_ACTOR_FRACTION_MAX
    result["gates"] = {
        **dict(result.get("gates", {})),
        "all_requests_complete": result["all_requests_complete"],
        "safety_all_zero": not any(bool(value) for value in (result.get("safety") or {}).values()),
        "exact_tape_replay": bool(result.get("replay_exact")),
        "exact_requested_likelihood": bool(result.get("requested_logprob_exact")),
        "no_reverse_transfer": result["no_reverse_transfer"],
        "shield_actor_fraction": result["shield_actor_fraction_gate"],
        "no_teacher": result["no_teacher"],
        "no_labels": result["no_labels"],
        "no_updates": result["no_updates"],
    }
    result["gates"]["all"] = bool(result["gates"]) and all(
        bool(value) for key, value in result["gates"].items() if key != "all"
    )
    _write_json(output / "evaluation.json", result)
    return result


def _rule_result(config: Any, seed: int, output: Path, *, smoke: bool) -> dict[str, Any]:
    result = dict(generic.evaluate_rule(config, seed, output, smoke=smoke))
    geometry = result.get("geometry", {})
    result["controller_kind"] = "rule"
    result["all_requests_complete"] = bool(
        result.get("terminal") and result.get("completed") == result.get("requests")
    )
    result["no_reverse_transfer"] = int(geometry.get("reverse_samples", 0)) == 0
    result["no_teacher"] = not bool(result.get("teacher_enabled"))
    result["no_updates"] = result.get("training_called") is False
    result["rule_shield_zero"] = int(result.get("shield", 0)) == 0
    result["gates"] = {
        **dict(result.get("gates", {})),
        "all_requests_complete": result["all_requests_complete"],
        "safety_all_zero": not any(bool(value) for value in (result.get("safety") or {}).values()),
        "exact_tape_replay": bool(result.get("replay_exact")),
        "no_reverse_transfer": result["no_reverse_transfer"],
        "rule_shield_zero": result["rule_shield_zero"],
        "no_teacher": result["no_teacher"],
        "no_updates": result["no_updates"],
    }
    result["gates"]["all"] = bool(result["gates"]) and all(
        bool(value) for key, value in result["gates"].items() if key != "all"
    )
    _write_json(output / "evaluation.json", result)
    return result


def _weighted_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            values.append((value_f, int(row.get("completed", 0))))
    total = sum(weight for _, weight in values)
    if not values or total <= 0:
        return None
    return float(sum(value * weight for value, weight in values) / total)


def aggregate_records(records: Sequence[Mapping[str, Any]], controller: str) -> dict[str, Any]:
    rows = [dict(row.get("result", {})) for row in records if row.get("controller") == controller]
    requests = sum(int(row.get("requests", 0)) for row in rows)
    completed = sum(int(row.get("completed", 0)) for row in rows)
    all_gates = bool(rows) and all(
        isinstance(row.get("gates"), Mapping)
        and bool(row["gates"])
        and all(bool(value) for value in row["gates"].values())
        for row in rows
    )
    mean_seconds = float(np.mean([float(row["seconds"]) for row in rows])) if rows else None
    wait = _weighted_mean(rows, "weighted_wait_seconds")
    request_to_touchdown = _weighted_mean(rows, "request_to_touchdown_seconds")
    safety_keys = sorted({key for row in rows for key in (row.get("safety") or {})})
    safety = {
        key: sum(int(bool((row.get("safety") or {}).get(key, 0))) for row in rows)
        for key in safety_keys
    }
    actor = sum(int(row.get("actor_decisions", 0)) for row in rows)
    shield_actor = sum(int(row.get("shield_actor", 0)) for row in rows)
    aggregate = {
        "controller": controller,
        "records": len(rows),
        "requests": requests,
        "completed": completed,
        "completion_fraction": completed / max(1, requests),
        "all_gates": all_gates,
        "mean_seconds": mean_seconds,
        "completion_seconds": mean_seconds,
        "weighted_wait_seconds": wait,
        "mean_wait_seconds": wait,
        "request_to_touchdown_seconds": request_to_touchdown,
        "mean_request_to_touchdown_seconds": request_to_touchdown,
        "weighted_mean_wait_seconds": wait,
        "weighted_mean_request_to_touchdown_seconds": request_to_touchdown,
        "safety": safety,
        "safety_all_zero": not any(safety.values()),
        "shield": sum(int(row.get("shield", 0)) for row in rows),
        "shield_actor": shield_actor,
        "actor_decisions": actor,
        "shield_actor_fraction": shield_actor / max(1, actor),
    }
    return aggregate


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        value = float(left) - float(right)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def pair_deltas(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = {}
    for row in records:
        result = row.get("result", {})
        key = (int(result.get("seed", row.get("seed", -1))), str(row.get("scenario", result.get("scenario", ""))))
        grouped.setdefault(key, {})[str(row.get("controller"))] = result
    deltas = []
    # Keep the manifest's standardized metric names while reading the raw
    # per-episode names emitted by the generic evaluator.
    metrics = (
        ("completion_seconds", "seconds"),
        ("weighted_mean_wait_seconds", "weighted_wait_seconds"),
        ("weighted_mean_request_to_touchdown_seconds", "request_to_touchdown_seconds"),
    )
    for (seed, scenario), values in sorted(grouped.items()):
        candidate, baseline, rule = (values.get(name) for name in CONTROLLERS)
        if candidate is None or baseline is None or rule is None:
            continue
        deltas.append({
            "seed": seed,
            "scenario": scenario,
            "candidate_minus_baseline": {
                output_metric: _delta(candidate.get(raw_metric), baseline.get(raw_metric))
                for output_metric, raw_metric in metrics
            },
            "candidate_minus_rule": {
                output_metric: _delta(candidate.get(raw_metric), rule.get(raw_metric))
                for output_metric, raw_metric in metrics
            },
        })
    return deltas


def gap_closure(candidate: Any, baseline: Any, rule: Any) -> float | None:
    """Return robust closure of the baseline-minus-rule gap."""
    try:
        candidate_f, baseline_f, rule_f = float(candidate), float(baseline), float(rule)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (candidate_f, baseline_f, rule_f)):
        return None
    if baseline_f > rule_f:
        return float((baseline_f - candidate_f) / (baseline_f - rule_f))
    if candidate_f <= baseline_f:
        return 1.0
    if baseline_f < rule_f:
        return float((baseline_f - candidate_f) / (rule_f - baseline_f))
    return float(-(candidate_f - baseline_f))


def effectiveness_gate_checks(aggregates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    candidate = aggregates.get("candidate", {})
    baseline = aggregates.get("baseline", {})
    rule = aggregates.get("rule", {})
    candidate_wait = candidate.get("weighted_mean_wait_seconds")
    baseline_wait = baseline.get("weighted_mean_wait_seconds")
    rule_wait = rule.get("weighted_mean_wait_seconds")
    candidate_touchdown = candidate.get("weighted_mean_request_to_touchdown_seconds")
    baseline_touchdown = baseline.get("weighted_mean_request_to_touchdown_seconds")
    rule_touchdown = rule.get("weighted_mean_request_to_touchdown_seconds")
    wait_closure = gap_closure(candidate_wait, baseline_wait, rule_wait)
    touchdown_closure = gap_closure(candidate_touchdown, baseline_touchdown, rule_touchdown)
    checks = {
        "candidate_wait_le_baseline": (
            candidate_wait is not None and baseline_wait is not None
            and float(candidate_wait) <= float(baseline_wait)
        ),
        "candidate_request_to_touchdown_le_baseline": (
            candidate_touchdown is not None and baseline_touchdown is not None
            and float(candidate_touchdown) <= float(baseline_touchdown)
        ),
        "wait_gap_closure": wait_closure,
        "request_to_touchdown_gap_closure": touchdown_closure,
        "wait_gap_closure_ge_50pct": (
            wait_closure is not None and math.isfinite(wait_closure)
            and wait_closure >= GAP_CLOSURE_MIN
        ),
        "request_to_touchdown_gap_closure_ge_50pct": (
            touchdown_closure is not None and math.isfinite(touchdown_closure)
            and touchdown_closure >= GAP_CLOSURE_MIN
        ),
    }
    checks["all"] = all(bool(checks[name]) for name in (
        "candidate_wait_le_baseline",
        "candidate_request_to_touchdown_le_baseline",
        "wait_gap_closure_ge_50pct",
        "request_to_touchdown_gap_closure_ge_50pct",
    ))
    return checks


def _formal_shape(records: Sequence[Mapping[str, Any]]) -> bool:
    if len(records) != EXPECTED_FORMAL_RECORDS:
        return False
    for controller in CONTROLLERS:
        rows = [row for row in records if row.get("controller") == controller]
        if len(rows) != len(FORMAL_SEEDS):
            return False
        ordered = sorted(rows, key=lambda row: int(row.get("seed", row.get("result", {}).get("seed", -1))))
        if [int(row.get("seed", row.get("result", {}).get("seed", -1))) for row in ordered] != list(FORMAL_SEEDS):
            return False
        if [str(row.get("scenario")) for row in ordered] != list(FORMAL_SCENARIOS):
            return False
        if any(not isinstance(row.get("result", {}).get("gates"), Mapping)
               or not all(bool(value) for value in row["result"]["gates"].values())
               for row in rows):
            return False
    return True


def smoke_semantics(*, smoke: bool, seeds: Sequence[int], scenarios: Sequence[str], steps: int) -> dict[str, bool]:
    return {
        "single_seed": bool(smoke and list(seeds) == [SMOKE_SEED]),
        "flow6": bool(smoke and list(scenarios) == [SMOKE_SCENARIO]),
        "steps": bool(smoke and int(steps) == SMOKE_STEPS),
        "no_reservation": bool(smoke),
        "no_authorization": bool(smoke),
    }


def source_hashes() -> dict[str, str]:
    return {
        "runner": _sha256(Path(__file__)),
        "protocol": _sha256(PROTOCOL),
        "generic_evaluator": _sha256(Path(generic.__file__)),
    }


def _authorizations(*, passed: bool, smoke: bool, marker_created: bool) -> dict[str, Any]:
    allowed = bool(passed and not smoke)
    return {
        "P2_EFFECTIVENESS_EVALUATION": "PASS" if allowed else "FAIL",
        "P2_PPO_ADMISSION_REVIEW_ALLOWED": allowed,
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "checkpoint_rewrite": False,
        "formal_marker_created": bool(marker_created),
    }


def run(
    output: str | Path,
    qualification_manifest: str | Path,
    repair_manifest: str | Path,
    candidate_checkpoint: str | Path,
    baseline_manifest: str | Path,
    baseline_checkpoint: str | Path,
    history_root: str | Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute one smoke probe or one reserved formal paired evaluation."""
    protocol_value = protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"effectiveness output already exists: {output_path}")
    inputs = _load_inputs(
        qualification_manifest, repair_manifest, candidate_checkpoint,
        baseline_manifest, baseline_checkpoint,
    )
    output_path.mkdir(parents=True, exist_ok=False)
    seeds = [SMOKE_SEED] if smoke else list(FORMAL_SEEDS)
    scenarios = [SMOKE_SCENARIO] if smoke else list(FORMAL_SCENARIOS)
    reservation = None
    if not smoke:
        marker = reservation_marker_path(candidate_checkpoint)
        _require(not marker.exists(), f"effectiveness reservation marker already exists: {marker}")
        reservation = reserve_formal_attempt(marker, history_root, output_path)
    _write_json(output_path / "pre_reset_registry.json", {
        "protocol": EFFECTIVENESS_STAGE,
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "seeds": seeds,
        "scenarios": scenarios,
        "controllers": list(CONTROLLERS),
        "smoke": bool(smoke),
        "historical_registry_checked": bool(not smoke),
        "reserved_before_first_reset": bool(reservation is not None),
        "reservation_marker": None if reservation is None else reservation["marker"],
    })
    records: list[dict[str, Any]] = []
    exception_text: str | None = None
    policy_pairs = (
        ("candidate", inputs["candidate"]),
        ("baseline", inputs["baseline"]),
    )
    for index, seed in enumerate(seeds):
        scenario = scenarios[index % len(scenarios)]
        config = p2_config(int(scenario.removeprefix("flow")))
        planned = list(policy_pairs) + [("rule", None)]
        for controller, policy in planned:
            folder = output_path / f"{controller}_{seed}"
            try:
                if controller == "rule":
                    result = _rule_result(config, int(seed), folder, smoke=smoke)
                else:
                    result = _student_result(policy, config, int(seed), folder, smoke=smoke)
                result["scenario"] = scenario
                records.append({
                    "controller": controller, "seed": int(seed),
                    "scenario": scenario, "result": result,
                })
            except Exception as exc:  # fail closed and retain direct evidence
                exception_text = f"{type(exc).__name__}: {exc}"
                records.append({
                    "controller": controller, "seed": int(seed),
                    "scenario": scenario,
                    "result": {
                        "seed": int(seed), "scenario": scenario,
                        "exception": exception_text,
                        "gates": {"all": False},
                    },
                    "exception": exception_text,
                })
                _write_json(output_path / "progress.json", {
                    "stage": EFFECTIVENESS_STAGE, "smoke": bool(smoke),
                    "records_completed": len(records),
                    "records_planned": len(CONTROLLERS) * len(seeds),
                    "exception": exception_text,
                })
                break
            _write_json(output_path / "progress.json", {
                "stage": EFFECTIVENESS_STAGE, "smoke": bool(smoke),
                "records_completed": len(records),
                "records_planned": len(CONTROLLERS) * len(seeds),
                "last_controller": controller, "last_seed": int(seed),
                "last_scenario": scenario,
                "last_gates": records[-1]["result"].get("gates", {}),
            })
        if exception_text is not None:
            break
    aggregates = {name: aggregate_records(records, name) for name in CONTROLLERS}
    hard_gates = {
        "formal_shape": (not smoke) and _formal_shape(records),
        "all_controller_episode_gates": all(
            aggregates[name]["all_gates"] for name in CONTROLLERS
        ),
        "requested_per_controller": all(
            aggregates[name]["requests"] == EXPECTED_FORMAL_REQUESTED
            for name in CONTROLLERS
        ) if not smoke else False,
        "no_exception": exception_text is None,
    }
    effectiveness = effectiveness_gate_checks(aggregates)
    passed = bool(
        not smoke and exception_text is None and all(hard_gates.values())
        and bool(effectiveness.get("all"))
    )
    result = {
        "stage": EFFECTIVENESS_STAGE,
        "status": "PASS" if passed else "FAIL",
        "evidence_type": "newly_run",
        "smoke": bool(smoke),
        "protocol": str(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "qualification_manifest": str(qualification_manifest),
        "qualification_manifest_sha256": QUALIFICATION_MANIFEST_SHA,
        "qualification_protocol_sha256": QUALIFICATION_PROTOCOL_SHA,
        "repair_manifest": str(repair_manifest),
        "repair_manifest_sha256": REPAIR_MANIFEST_SHA,
        "repair_protocol_sha256": REPAIR_PROTOCOL_SHA,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": REPAIRED_CHECKPOINT_SHA,
        "baseline_manifest": str(baseline_manifest),
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA,
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_checkpoint_sha256": BASELINE_CHECKPOINT_SHA,
        "input_hashes": {
            "qualification_manifest": _sha256(qualification_manifest),
            "repair_manifest": _sha256(repair_manifest),
            "candidate_checkpoint": _sha256(candidate_checkpoint),
            "baseline_manifest": _sha256(baseline_manifest),
            "baseline_checkpoint": _sha256(baseline_checkpoint),
        },
        "candidate_payload": {
            "stage": inputs["candidate_payload"].get("stage"),
            "architecture_id": inputs["candidate_payload"].get("architecture_id"),
            "temperature": inputs["candidate_payload"].get("temperature"),
            "optimizer_present_but_not_loaded": "optimizer" in inputs["candidate_payload"],
        },
        "baseline_payload": {
            "stage": inputs["baseline_payload"].get("stage"),
            "architecture_id": inputs["baseline_payload"].get("architecture_id"),
            "temperature": inputs["baseline_payload"].get("temperature"),
            "optimizer_present_but_not_loaded": "optimizer" in inputs["baseline_payload"],
        },
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_scenarios": list(FORMAL_SCENARIOS),
        "controllers": list(CONTROLLERS),
        "records": records,
        "aggregates": aggregates,
        "paired_deltas": pair_deltas(records),
        "hard_gates": hard_gates,
        "effectiveness_gates": effectiveness,
        "reservation": reservation,
        "formal_marker_created": bool(reservation is not None),
        "exception": exception_text,
        "training_called": False,
        "optimizer_called": False,
        "policy_update": False,
        "teacher_called": False,
        "labels_collected": False,
        "deterministic": True,
        "smoke_semantics": smoke_semantics(
            smoke=smoke, seeds=seeds, scenarios=scenarios,
            steps=SMOKE_STEPS if smoke else 0,
        ),
        "authorizations": _authorizations(
            passed=passed, smoke=smoke, marker_created=reservation is not None,
        ),
        "P2_EFFECTIVENESS_EVALUATION": "PASS" if passed else "FAIL",
        "P2_PPO_ADMISSION_REVIEW_ALLOWED": bool(passed),
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "checkpoint_rewrite": False,
        "source_hashes": source_hashes(),
    }
    manifest_path = output_path / "manifest.json"
    result["manifest"] = str(manifest_path)
    _write_json(manifest_path, result)
    _print_result({
        "stage": EFFECTIVENESS_STAGE, "status": result["status"],
        "smoke": bool(smoke), "records": len(records),
        "P2_EFFECTIVENESS_EVALUATION": result["P2_EFFECTIVENESS_EVALUATION"],
        "P2_PPO_ADMISSION_REVIEW_ALLOWED": result["P2_PPO_ADMISSION_REVIEW_ALLOWED"],
        "PPO_ALLOWED": False,
    })
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-manifest", "--calibration-manifest", dest="baseline_manifest", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    _main()
