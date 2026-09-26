"""Inference-only paired evaluation for the frozen V7 formal checkpoints.

The evaluator loads the source checkpoint and all three completed formal
checkpoints before the paired evaluation.  Its optional smoke path loads only
the bound source checkpoint.  It never constructs an optimizer update,
consumes an evaluation seed, or promotes a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
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
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.training import (
    diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as lifecycle,
)
from terminal_marl.training import (
    run_v4_2_p3_controlled_emergency_admission_v5 as admission,
)
from terminal_marl.training import (
    run_v4_2_p3_gate_anchored_v6 as v6,
)
from terminal_marl.training import (
    run_p3_adaptive_kl_formal_v7 as formal_v7,
)
from terminal_marl.training import terminal_lifecycle_metrics as terminal_metrics


STAGE = "P3_ADAPTIVE_KL_FINAL_PAIRED_EVALUATION_V7"
RUNNER_MODULE = "terminal_marl.training.evaluate_p3_adaptive_kl_v7"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/p3_adaptive_kl_v7.yaml")
SOURCE_CHECKPOINT_SHA256 = (
    "03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba"
)
OBSERVATION_DIM = 88
ARM_LABELS = ("source_03c0", "final_run0", "final_run1", "final_run2")
IN_DISTRIBUTION_SEEDS = tuple(range(880101, 880113))
IN_DISTRIBUTION_SCENARIOS = ("flow6", "flow12", "dense24") * 4
DENSE30_SEEDS = (880201, 880202)
RANDOM_DENSE24_SEEDS = (880301, 880302)
EVALUATION_SCENARIOS = ("flow6", "flow12", "dense24", "dense30", "random_dense24")
SMOKE_SEED = 0
SMOKE_MAX_STEPS = 2


class EvaluationInputError(ValueError):
    """Raised when a frozen V7 evaluation input does not match its binding."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    return admission._jsonable(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return value


def _protocol() -> dict[str, Any]:
    _require(PROTOCOL.is_file(), f"V7 protocol missing: {PROTOCOL}")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, Mapping), "V7 protocol must be a mapping")
    value = dict(value)
    _require(value.get("protocol") == "V4_2_P3_ADAPTIVE_KL_V7", "V7 protocol id changed")
    lineage = value.get("lineage") or {}
    _require(lineage.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
             "V7 source checkpoint SHA changed")
    bindings = value.get("bindings") or {}
    for key, expected in {
        "action_schema_version": "low_action_schema_v4_2_hierarchical",
        "observation_schema_version": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": OBSERVATION_DIM,
        "vehicle_profile_version": "eh216s_research_v2",
        "runtime_shortlist_k": 3,
        "reconcile_late_merge_slots_enabled": False,
        "v4_2_rejected_release_hold_mode": "canonical_wait",
        "teacher": False,
        "scheduler_mode": "execution",
        "high_level_enabled": False,
        "shield_enabled": True,
    }.items():
        _require(bindings.get(key) == expected, f"V7 binding changed: {key}")
    environment = value.get("environment") or {}
    _require(environment.get("full_scenario_horizon_seconds") ==
             {"flow6": 1210, "flow12": 1570, "dense24": 1800},
             "V7 scenario horizons changed")
    evaluation = value.get("post_training_evaluation") or {}
    in_distribution = evaluation.get("in_distribution") or {}
    _require(in_distribution.get("seeds") == list(IN_DISTRIBUTION_SEEDS),
             "V7 in-distribution evaluation seeds changed")
    _require(in_distribution.get("scenarios") == list(IN_DISTRIBUTION_SCENARIOS),
             "V7 in-distribution evaluation scenarios changed")
    _require(in_distribution.get("same_seed_for_baseline_and_final") is True,
             "V7 evaluation must pair baseline and final on the same seeds")
    _require(in_distribution.get("baseline") == "source_03c0_and_each_final_run_checkpoint",
             "V7 evaluation baseline binding changed")
    stress = evaluation.get("stress") or {}
    dense30 = stress.get("dense30_no_inflow") or {}
    _require(dense30.get("seeds") == list(DENSE30_SEEDS) and
             int(dense30.get("initial_vehicle_count")) == 30 and
             dense30.get("inflow") == "none",
             "V7 dense30 evaluation binding changed")
    random_dense24 = stress.get("random_dense24_same_load") or {}
    _require(random_dense24.get("seeds") == list(RANDOM_DENSE24_SEEDS) and
             int(random_dense24.get("initial_vehicle_count")) == 12 and
             int(random_dense24.get("inflow_vehicle_count")) == 12,
             "V7 randomized dense24 evaluation binding changed")
    _require(evaluation.get("heldout_consumed") is False,
             "V7 evaluation must not consume heldout seeds")
    _require(evaluation.get("evaluation_results_do_not_flow_back_to_tuning") is True,
             "V7 evaluation feedback binding changed")
    return value


def _protocol_digest(protocol: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(protocol), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _config_snapshot(config: Any) -> dict[str, Any]:
    return _jsonable(config) if hasattr(config, "__dataclass_fields__") else _jsonable(dict(vars(config)))


def _build_configs(protocol: Mapping[str, Any]) -> dict[str, Any]:
    base_configs, _ = formal_v7._build_configs(protocol)
    configs = dict(base_configs)
    for scenario, config in configs.items():
        _require(str(config.action_schema_version) == "low_action_schema_v4_2_hierarchical",
                 f"{scenario} action schema changed")
        _require(str(config.observation_schema_version) == "phaseA_low_obs_v2_1_retry_state",
                 f"{scenario} observation schema changed")
        _require(int(config.runtime_proposal_shortlist_k) == 3,
                 f"{scenario} runtime shortlist changed")
        _require(not bool(config.teacher_enabled), f"{scenario} Teacher enabled")
        _require(str(config.scheduler_mode) == "execution", f"{scenario} scheduler changed")
        _require(not bool(config.high_level_enabled), f"{scenario} high-level geometry enabled")
        _require(not bool(config.reconcile_late_merge_slots_enabled),
                 f"{scenario} late-merge reconciliation enabled")
    dense24 = configs["dense24"]
    configs["dense30"] = replace(
        dense24,
        reset_mode="warm_start_rings",
        warm_start_uavs=30,
        fixed_arrival_schedule=(0.0,) * 30,
        total_requested_uavs=30,
        max_simulation_seconds=1800.0,
        warm_start_layout_randomized=False,
    )
    configs["random_dense24"] = replace(
        dense24,
        warm_start_layout_randomized=True,
    )
    _require(configs["dense30"].warm_start_uavs == 30 and
             configs["dense30"].total_requested_uavs == 30 and
             len(configs["dense30"].fixed_arrival_schedule) == 30,
             "dense30 warm-start binding changed")
    _require(configs["random_dense24"].warm_start_uavs == 12 and
             configs["random_dense24"].total_requested_uavs == 24 and
             configs["random_dense24"].warm_start_layout_randomized is True,
             "random dense24 warm-start binding changed")
    return configs


def _evaluation_plan() -> list[dict[str, Any]]:
    plan = [
        {"scenario": scenario, "seed": int(seed)}
        for seed, scenario in zip(IN_DISTRIBUTION_SEEDS, IN_DISTRIBUTION_SCENARIOS)
    ]
    plan.extend({"scenario": "dense30", "seed": int(seed)} for seed in DENSE30_SEEDS)
    plan.extend({"scenario": "random_dense24", "seed": int(seed)} for seed in RANDOM_DENSE24_SEEDS)
    _require(len(plan) == 16, "V7 evaluation must contain exactly 16 scenario/seed cells")
    return plan


def _resolve_bound_path(value: Any, *, relative_to: Path) -> Path:
    target = Path(str(value))
    return target if target.is_absolute() else relative_to / target


def _validate_formal_manifests(
    formal_root: Path,
    *,
    protocol_sha256: str,
) -> list[dict[str, Any]]:
    _require(formal_root.is_dir(), f"formal root missing: {formal_root}")
    bindings: list[dict[str, Any]] = []
    for run_index in range(3):
        manifest_path = formal_root / f"run{run_index}" / "manifest.json"
        manifest = _read_json(manifest_path)
        _require(manifest.get("status") == "COMPLETED",
                 f"formal run{run_index} is not COMPLETED")
        processed = manifest.get("processed_episodes", manifest.get("episodes_processed"))
        _require(int(processed) == 240, f"formal run{run_index} processed episode count changed")
        _require(manifest.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
                 f"formal run{run_index} source checkpoint SHA changed")
        _require(manifest.get("protocol_sha256") == protocol_sha256,
                 f"formal run{run_index} protocol SHA changed")
        checkpoint_value = manifest.get("final_checkpoint")
        checkpoint_sha = manifest.get("final_checkpoint_sha256")
        _require(checkpoint_value and checkpoint_sha,
                 f"formal run{run_index} final checkpoint binding is incomplete")
        checkpoint_path = _resolve_bound_path(checkpoint_value, relative_to=manifest_path.parent)
        _require(checkpoint_path.is_file(), f"formal run{run_index} final checkpoint missing: {checkpoint_path}")
        actual_checkpoint_sha = _sha256(checkpoint_path)
        _require(actual_checkpoint_sha == str(checkpoint_sha),
                 f"formal run{run_index} final checkpoint SHA mismatch")
        bindings.append({
            "run_index": run_index,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": actual_checkpoint_sha,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "protocol_sha256": protocol_sha256,
        })
    return bindings


def _model_digest(policy: Any) -> str:
    return v6._tensor_digest(policy.model.state_dict())


def _load_eval_policy(
    checkpoint: Path,
    *,
    config: Any,
    expected_checkpoint_sha256: str,
    protocol_sha256: str,
    final_checkpoint: bool,
    learning_rate: float,
) -> tuple[Any, dict[str, Any], str]:
    actual_sha = _sha256(checkpoint)
    _require(actual_sha == expected_checkpoint_sha256,
             f"checkpoint SHA mismatch: {checkpoint}")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        temperature=CALIBRATION_TEMPERATURE,
        learning_rate=float(learning_rate),
    )
    payload = load_checkpoint(checkpoint, policy, config, load_optimizer=False)
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    _require(isinstance(metadata, Mapping), "checkpoint metadata is missing")
    _require(getattr(policy.model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID,
             "checkpoint architecture changed")
    _require(float(policy.model.gate_temperature.detach().cpu().item()) ==
             float(CALIBRATION_TEMPERATURE), "checkpoint temperature changed")
    _require(int(policy.observation_dim) == OBSERVATION_DIM, "checkpoint observation dimension changed")
    _require(len(policy.optimizer.state) == 0, "checkpoint optimizer state was restored")
    declared_architecture = metadata.get("architecture_id")
    if declared_architecture is not None:
        _require(declared_architecture == CALIBRATED_ARCHITECTURE_ID,
                 "checkpoint metadata architecture changed")
    declared_temperature = metadata.get("temperature")
    if declared_temperature is not None:
        _require(float(declared_temperature) == float(CALIBRATION_TEMPERATURE),
                 "checkpoint metadata temperature changed")
    if final_checkpoint:
        _require(payload.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
                 "final checkpoint source SHA binding changed")
        _require(payload.get("protocol_sha256") == protocol_sha256,
                 "final checkpoint protocol SHA binding changed")
    policy.model.eval()
    return policy, dict(payload), actual_sha


def _failure_record(
    *,
    label: str,
    scenario: str,
    seed: int,
    checkpoint: Path,
    config: Any,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "label": label,
        "scenario": scenario,
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "status": "failed",
        "requests": int(config.total_requested_uavs),
        "completed": 0,
        "normal": 0,
        "emergency": 0,
        "failed_energy": 0,
        "unresolved": 0,
        "other_failures": {},
        "summary": None,
        "action_tape": None,
        "lifecycle": None,
        "dev_pass": False,
        "safety_invariant_error": isinstance(error, SafetyInvariantError),
        "error": {"type": type(error).__name__, "message": str(error)},
        "weights_unchanged": None,
        "teacher_enabled": bool(getattr(config, "teacher_enabled", False)),
        "PPO_ALLOWED": False,
        "label_rows": 0,
    }


def _episode_record(
    *,
    policy: Any,
    label: str,
    checkpoint: Path,
    config: Any,
    protocol: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
    output: Path,
    max_steps: int | None = None,
    require_full_resolution: bool = True,
) -> dict[str, Any]:
    horizon = int(round(float(config.max_simulation_seconds) / float(config.dt)))
    requested_steps = horizon if max_steps is None else int(max_steps)
    try:
        record = admission._record_rollout(
            policy=policy,
            config=config,
            p=runtime_protocol,
            scenario=scenario,
            seed=int(seed),
            output=output,
            label=label,
            deterministic=False,
            max_steps=requested_steps,
            require_full_resolution=bool(require_full_resolution),
            checkpoint=checkpoint,
        )
        if not isinstance(record, Mapping):
            raise EvaluationInputError("rollout record must be a mapping")
        record = dict(record)
        _normalize_outcome_accounting(record, requests=int(config.total_requested_uavs))
        record["success_lifecycle"] = record.get("lifecycle")
        tape = record.get("action_tape")
        low_soc_threshold = float(
            (runtime_protocol.get("attributes") or {}).get("low_soc_analysis_threshold", .65)
            if isinstance(runtime_protocol.get("attributes"), Mapping) else .65
        )
        if isinstance(tape, (str, Path)) and Path(str(tape)).is_file():
            try:
                record["lifecycle"] = terminal_metrics.terminal_lifecycle_metrics(
                    config,
                    tape,
                    low_soc_threshold,
                )
                record["lifecycle_scope"] = "all_requested_terminal_rows"
            except Exception as exc:
                # Keep the rollout's original outcome/gate fields intact when
                # a secondary metrics replay cannot be accepted.
                record["lifecycle_scope"] = "success_only_legacy"
                record["lifecycle_full_request_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        else:
            record["lifecycle_scope"] = "success_only_legacy"
        try:
            record["dev_pass"] = bool(v6._dev_pass(record))
        except Exception as exc:
            record["dev_pass"] = False
            record["dev_pass_error"] = {"type": type(exc).__name__, "message": str(exc)}
        record["evaluation_arm"] = label
        record["evaluation_scenario"] = scenario
        record["evaluation_seed"] = int(seed)
        record["checkpoint_sha256"] = _sha256(checkpoint)
        record["steps_recorded"] = len(record.get("_steps", ()))
        record.pop("_steps", None)
        return record
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        failure = _failure_record(
            label=label, scenario=scenario, seed=seed, checkpoint=checkpoint,
            config=config, error=exc,
        )
        _normalize_outcome_accounting(failure, requests=int(config.total_requested_uavs))
        return failure


def _normalize_outcome_accounting(record: dict[str, Any], *, requests: int) -> None:
    """Keep missing terminal outcomes in the request denominator explicitly."""
    raw_other = record.get("other_failures") or {}
    other_count = int(sum(int(value) for value in raw_other.values())) if isinstance(raw_other, Mapping) else 0
    raw_counts = {
        "normal": int(record.get("normal", 0) or 0),
        "emergency": int(record.get("emergency", 0) or 0),
        "failed_energy": int(record.get("failed_energy", 0) or 0),
        "other_failures": other_count,
        "unresolved": int(record.get("unresolved", 0) or 0),
    }
    observed_total = int(sum(raw_counts.values()))
    record["raw_outcome_counts"] = dict(raw_counts)
    record["outcome_accounting_error"] = bool(observed_total > requests)
    record["remaining_requests_not_observed"] = max(0, int(requests) - observed_total)
    record["unknown_outcomes"] = int(record["remaining_requests_not_observed"])
    record["unresolved_includes_unknown"] = bool(record["unknown_outcomes"])
    if observed_total > requests:
        record["outcome_accounting_error_message"] = (
            f"observed outcome counts {observed_total} exceed requests {requests}"
        )
    if record["remaining_requests_not_observed"]:
        record["unresolved"] = int(record.get("unresolved", 0) or 0) + int(
            record["remaining_requests_not_observed"]
        )
    record["outcome_accounting_complete"] = bool(
        observed_total == requests and isinstance(record.get("summary"), Mapping)
    )


def _record_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("requests", "normal", "emergency", "failed_energy", "unresolved")
    result = {key: int(sum(int(record.get(key, 0) or 0) for record in records)) for key in keys}
    other: Counter[str] = Counter()
    for record in records:
        values = record.get("other_failures") or {}
        if isinstance(values, Mapping):
            other.update({str(key): int(value) for key, value in values.items()})
    result["other_failures"] = dict(sorted(other.items()))
    result["other_failure_count"] = int(sum(other.values()))
    result["episodes"] = len(records)
    result["remaining_requests_not_observed"] = int(
        sum(int(record.get("remaining_requests_not_observed", 0) or 0) for record in records)
    )
    result["unknown_outcomes"] = int(
        sum(int(record.get("unknown_outcomes", 0) or 0) for record in records)
    )
    result["outcome_accounting_errors"] = int(
        sum(bool(record.get("outcome_accounting_error")) for record in records)
    )
    result["safety_invariant_error_episodes"] = int(
        sum(bool(record.get("safety_invariant_error")) for record in records)
    )
    result["dev_pass_episodes"] = int(sum(bool(record.get("dev_pass")) for record in records))
    return result


def _lifecycle_summary(records: Sequence[Mapping[str, Any]], *, threshold: float) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    legacy_rows: list[Mapping[str, Any]] = []
    requested = 0
    missing_requests = 0
    full_records = 0
    incomplete_records = 0
    for record in records:
        value = record.get("lifecycle")
        fallback_requests = value.get("requested_count", 0) if isinstance(value, Mapping) else 0
        record_requests = int(record.get("requests", fallback_requests) or 0)
        requested += record_requests
        if not isinstance(value, Mapping) or value.get("scope") != "all_requested_terminal_rows":
            incomplete_records += 1
            missing_requests += record_requests
            if isinstance(value, Mapping) and isinstance(value.get("rows"), list):
                legacy_rows.extend(row for row in value["rows"] if isinstance(row, Mapping))
            continue
        record_rows = value.get("rows")
        selected = [row for row in record_rows if isinstance(row, Mapping)] if isinstance(record_rows, list) else []
        rows.extend(selected)
        full_records += 1
        record_missing = int(value.get("missing_request_count", 0) or 0)
        missing_requests += max(record_missing, record_requests - len(selected))
    if rows or full_records:
        summary = terminal_metrics.summarize_terminal_rows(
            rows,
            low_soc_threshold=float(threshold),
            requested_count=requested,
            enforce_unique=False,
        )
        summary.update({
            "scope": "mixed_incomplete" if incomplete_records else "all_requested_terminal_rows_aggregate",
            "episodes": full_records,
            "incomplete_metrics_episodes": incomplete_records,
            "missing_request_count": missing_requests,
            "all_requests_observed": missing_requests == 0,
            "requested_denominator": requested,
            "legacy_rows_excluded_from_observed_metrics": len(legacy_rows),
        })
        return {"available": True, **summary}
    if not legacy_rows:
        return {
            "available": False,
            "rows": [],
            "completed": 0,
            "by_priority": {},
            "low_soc": {"threshold": float(threshold), "count": 0, "mean_wait_seconds": None},
            "scope": "incomplete_no_lifecycle_rows",
            "requested_denominator": requested,
            "missing_request_count": missing_requests,
            "all_requests_observed": False if requested else True,
        }
    summary = lifecycle.summarize_lifecycle_rows(list(legacy_rows), low_soc_threshold=float(threshold))
    return {
        "available": True,
        "scope": "incomplete_success_only_legacy",
        "requested_denominator": requested,
        "missing_request_count": missing_requests,
        "all_requests_observed": False if missing_requests else True,
        **summary,
    }


def _wait_metric(lifecycle_value: Any, name: str) -> float | None:
    if not isinstance(lifecycle_value, Mapping):
        return None
    if name == "priority_weighted_wait_seconds":
        value = lifecycle_value.get(name)
    else:
        low_soc = lifecycle_value.get("low_soc")
        value = low_soc.get("mean_wait_seconds") if isinstance(low_soc, Mapping) else lifecycle_value.get(name)
    return None if value is None else float(value)


def _arm_summary(
    *,
    label: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    model_digest_before: str,
    policy: Any,
    records: list[dict[str, Any]],
    configs: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    model_digest_after = _model_digest(policy)
    scenario_summary: dict[str, Any] = {}
    for scenario in EVALUATION_SCENARIOS:
        selected = [row for row in records if row.get("scenario") == scenario]
        scenario_summary[scenario] = {
            "counts": _record_counts(selected),
            "lifecycle": _lifecycle_summary(
                selected,
                threshold=float((protocol.get("reward") or {}).get("low_soc_urgency_threshold", .65)),
            ),
            "config": _config_snapshot(configs[scenario]),
        }
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "model_weight_digest_before": model_digest_before,
        "model_weight_digest_after": model_digest_after,
        "weights_unchanged": bool(model_digest_before == model_digest_after and
                                   all(record.get("weights_unchanged") is True for record in records)),
        "teacher_disabled": all(record.get("teacher_enabled") is False for record in records),
        "PPO_ALLOWED": False,
        "records": records,
        "counts": _record_counts(records),
        "by_scenario": scenario_summary,
        "lifecycle": _lifecycle_summary(
            records,
            threshold=float((protocol.get("reward") or {}).get("low_soc_urgency_threshold", .65)),
        ),
        "episodes_expected": 16,
        "episodes_processed": len(records),
        "evaluation_complete": len(records) == 16,
        "development_only": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
    }


def _baseline_wait_deltas(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    source = arms.get("source_03c0") or {}
    source_records = {
        (str(row.get("scenario")), int(row.get("seed"))): row
        for row in source.get("records", ())
        if isinstance(row, Mapping) and row.get("seed") is not None
    }
    result: dict[str, Any] = {}
    for label, arm in arms.items():
        if label == "source_03c0":
            continue
        deltas: list[dict[str, Any]] = []
        for candidate in arm.get("records", ()):
            key = (str(candidate.get("scenario")), int(candidate.get("seed")))
            baseline = source_records.get(key)
            if baseline is None:
                deltas.append({"scenario": key[0], "seed": key[1], "missing_baseline": True})
                continue
            baseline_lifecycle = baseline.get("lifecycle")
            candidate_lifecycle = candidate.get("lifecycle")
            baseline_priority = _wait_metric(baseline_lifecycle, "priority_weighted_wait_seconds")
            candidate_priority = _wait_metric(candidate_lifecycle, "priority_weighted_wait_seconds")
            baseline_low = _wait_metric(baseline_lifecycle, "low_soc_mean_wait_seconds")
            candidate_low = _wait_metric(candidate_lifecycle, "low_soc_mean_wait_seconds")
            deltas.append({
                "scenario": key[0],
                "seed": key[1],
                "baseline_priority_weighted_wait_seconds": baseline_priority,
                "arm_priority_weighted_wait_seconds": candidate_priority,
                "priority_weighted_wait_delta_seconds": (
                    None if baseline_priority is None or candidate_priority is None
                    else candidate_priority - baseline_priority
                ),
                "baseline_low_soc_mean_wait_seconds": baseline_low,
                "arm_low_soc_mean_wait_seconds": candidate_low,
                "low_soc_wait_delta_seconds": (
                    None if baseline_low is None or candidate_low is None
                    else candidate_low - baseline_low
                ),
            })
        result[label] = deltas
    return result


def _load_source_binding(
    *,
    protocol: Mapping[str, Any],
    configs: Mapping[str, Any],
    protocol_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lineage = protocol.get("lineage") or {}
    source_value = lineage.get("source_checkpoint_path")
    _require(source_value, "V7 source checkpoint path missing")
    source_path = Path(str(source_value))
    _require(source_path.is_file(), f"V7 source checkpoint missing: {source_path}")
    source_actual_sha = _sha256(source_path)
    _require(source_actual_sha == SOURCE_CHECKPOINT_SHA256, "V7 source checkpoint SHA mismatch")
    optimizer = protocol.get("optimizer") or {}
    learning_rate = float(optimizer.get("learning_rate", 3e-4))
    source_policy, source_payload, source_sha = _load_eval_policy(
        source_path,
        config=configs["dense24"],
        expected_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        protocol_sha256=protocol_sha,
        final_checkpoint=False,
        learning_rate=learning_rate,
    )
    item = {
        "label": "source_03c0",
        "checkpoint": source_path,
        "checkpoint_sha256": source_sha,
        "policy": source_policy,
        "payload": source_payload,
        "model_weight_digest": _model_digest(source_policy),
    }
    return item, {
        "protocol_sha256": protocol_sha,
        "source_checkpoint_sha256": source_sha,
    }, {
        "source_checkpoint": str(source_path),
    }


def _load_bindings(
    *,
    formal_root: Path,
    protocol: Mapping[str, Any],
    configs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    protocol_sha = _sha256(PROTOCOL)
    final_bindings = _validate_formal_manifests(formal_root, protocol_sha256=protocol_sha)
    source_item, checkpoint_binding, source_paths = _load_source_binding(
        protocol=protocol, configs=configs, protocol_sha=protocol_sha,
    )
    loaded: list[dict[str, Any]] = [source_item]
    for binding in final_bindings:
        optimizer = protocol.get("optimizer") or {}
        learning_rate = float(optimizer.get("learning_rate", 3e-4))
        policy, payload, checkpoint_sha = _load_eval_policy(
            binding["checkpoint"],
            config=configs["dense24"],
            expected_checkpoint_sha256=binding["checkpoint_sha256"],
            protocol_sha256=protocol_sha,
            final_checkpoint=True,
            learning_rate=learning_rate,
        )
        loaded.append({
            "label": f"final_run{binding['run_index']}",
            "checkpoint": binding["checkpoint"],
            "checkpoint_sha256": checkpoint_sha,
            "policy": policy,
            "payload": payload,
            "model_weight_digest": _model_digest(policy),
            "formal_manifest": binding,
        })
    _require(tuple(item["label"] for item in loaded) == ARM_LABELS,
             "V7 checkpoint arm order changed")
    return loaded, checkpoint_binding, {
        "final_manifests": final_bindings,
        **source_paths,
    }


def _smoke(
    *,
    output: Path,
    policy: Any,
    checkpoint: Path,
    config: Any,
    protocol: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    _require(not output.exists(), f"smoke output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    record = _episode_record(
        policy=policy,
        label="smoke_source",
        checkpoint=checkpoint,
        config=config,
        protocol=protocol,
        runtime_protocol=runtime_protocol,
        scenario="dense24",
        seed=SMOKE_SEED,
        output=output / "dense24_0",
        max_steps=SMOKE_MAX_STEPS,
        require_full_resolution=False,
    )
    checks = {
        "weights_unchanged": record.get("weights_unchanged") is True,
        "teacher_disabled": record.get("teacher_enabled") is False,
        "PPO_ALLOWED_false": record.get("PPO_ALLOWED") is False,
        "action_tape_present": isinstance(record.get("action_tape"), str),
        "exact_step_count": int(record.get("steps_recorded", 0)) == SMOKE_MAX_STEPS,
        "partial_smoke": True,
    }
    result = {
        "stage": f"{STAGE}_SMOKE",
        "status": "SMOKE_COMPLETED" if all(checks.values()) else "SMOKE_FAILED",
        "seed": SMOKE_SEED,
        "max_steps": SMOKE_MAX_STEPS,
        "checks": checks,
        "record": record,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "PPO_ALLOWED": False,
    }
    _write_json(output / "manifest.json", result)
    return result


def run(
    *,
    formal_root: str | Path,
    output: str | Path,
    check_only: bool = False,
    smoke_output: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output)
    _require(not output_path.exists(), f"evaluation output must be fresh: {output_path}")
    protocol = _protocol()
    protocol_sha = _sha256(PROTOCOL)
    configs = _build_configs(protocol)
    runtime_protocol = v6._runtime_protocol(protocol)
    if smoke_output is not None:
        source_item, checkpoint_binding, binding_paths = _load_source_binding(
            protocol=protocol, configs=configs, protocol_sha=protocol_sha,
        )
        smoke_result = _smoke(
            output=Path(smoke_output),
            policy=source_item["policy"],
            checkpoint=source_item["checkpoint"],
            config=configs["dense24"],
            protocol=protocol,
            runtime_protocol=runtime_protocol,
        )
        return {
            "stage": STAGE,
            "status": smoke_result["status"],
            "check_only": bool(check_only),
            "output_created": False,
            "protocol_sha256": protocol_sha,
            "protocol_digest": _protocol_digest(protocol),
            "evaluation_cells": [],
            "checkpoint_binding": checkpoint_binding,
            "binding_paths": binding_paths,
            "arms": [{
                "label": source_item["label"],
                "checkpoint": str(source_item["checkpoint"]),
                "checkpoint_sha256": source_item["checkpoint_sha256"],
                "model_weight_digest": source_item["model_weight_digest"],
                "optimizer_state_empty": len(source_item["policy"].optimizer.state) == 0,
            }],
            "smoke": smoke_result,
            "development_only": True,
            "P3_QUALIFIED": False,
            "P4_ALLOWED": False,
            "PPO_ALLOWED": False,
        }
    plan = _evaluation_plan()
    loaded, checkpoint_binding, binding_paths = _load_bindings(
        formal_root=Path(formal_root), protocol=protocol, configs=configs,
    )
    if check_only:
        return {
            "stage": STAGE,
            "status": "CHECK_ONLY_OK",
            "check_only": True,
            "output_created": False,
            "protocol_sha256": protocol_sha,
            "protocol_digest": _protocol_digest(protocol),
            "evaluation_cells": plan,
            "checkpoint_binding": checkpoint_binding,
            "binding_paths": binding_paths,
            "arms": [
                {
                    "label": item["label"],
                    "checkpoint": str(item["checkpoint"]),
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "model_weight_digest": item["model_weight_digest"],
                    "optimizer_state_empty": len(item["policy"].optimizer.state) == 0,
                }
                for item in loaded
            ],
            "smoke": None,
            "development_only": True,
            "P3_QUALIFIED": False,
            "P4_ALLOWED": False,
            "PPO_ALLOWED": False,
        }

    output_path.mkdir(parents=True, exist_ok=False)
    binding_manifest = {
        "stage": STAGE,
        "status": "RUNNING",
        "protocol_sha256": protocol_sha,
        "protocol_digest": _protocol_digest(protocol),
        "formal_root": str(formal_root),
        "arms": [
            {
                "label": item["label"],
                "checkpoint": str(item["checkpoint"]),
                "checkpoint_sha256": item["checkpoint_sha256"],
                "model_weight_digest_before": item["model_weight_digest"],
            }
            for item in loaded
        ],
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "development_only": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "PPO_ALLOWED": False,
    }
    _write_json(output_path / "binding_manifest.json", binding_manifest)
    arms: dict[str, dict[str, Any]] = {}
    for item in loaded:
        records: list[dict[str, Any]] = []
        arm_dir = output_path / item["label"]
        arm_dir.mkdir(parents=True, exist_ok=False)
        for index, cell in enumerate(plan, start=1):
            config = configs[cell["scenario"]]
            record = _episode_record(
                policy=item["policy"],
                label=item["label"],
                checkpoint=item["checkpoint"],
                config=config,
                protocol=protocol,
                runtime_protocol=runtime_protocol,
                scenario=cell["scenario"],
                seed=int(cell["seed"]),
                output=arm_dir / f"episode_{index:02d}_{cell['scenario']}_{cell['seed']}",
            )
            records.append(record)
            _write_json(output_path / "progress.json", {
                "stage": STAGE,
                "status": "RUNNING",
                "current_arm": item["label"],
                "processed_episodes": sum(len(value.get("records", ())) for value in arms.values()) + len(records),
                "arms_completed": list(arms),
                "current_arm_processed": len(records),
                "records": records,
                "P3_QUALIFIED": False,
                "P4_ALLOWED": False,
                "PPO_ALLOWED": False,
            })
        arms[item["label"]] = _arm_summary(
            label=item["label"], checkpoint=item["checkpoint"],
            checkpoint_sha256=item["checkpoint_sha256"],
            model_digest_before=item["model_weight_digest"], policy=item["policy"],
            records=records, configs=configs, protocol=protocol,
        )
        _write_json(output_path / "progress.json", {
            "stage": STAGE,
            "status": "RUNNING",
            "current_arm": item["label"],
            "processed_episodes": sum(len(value.get("records", ())) for value in arms.values()),
            "arms_completed": list(arms),
            "arm_summary": arms[item["label"]],
            "P3_QUALIFIED": False,
            "P4_ALLOWED": False,
            "PPO_ALLOWED": False,
        })

    binding_manifest["status"] = "COMPLETED"
    binding_manifest["arms"] = [
        {
            "label": label,
            "checkpoint": arm["checkpoint"],
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "model_weight_digest_before": arm["model_weight_digest_before"],
            "model_weight_digest_after": arm["model_weight_digest_after"],
            "weights_unchanged": arm["weights_unchanged"],
        }
        for label, arm in arms.items()
    ]
    _write_json(output_path / "binding_manifest.json", binding_manifest)
    result = {
        "stage": STAGE,
        "status": "COMPLETED",
        "protocol_sha256": protocol_sha,
        "protocol_digest": _protocol_digest(protocol),
        "formal_root": str(formal_root),
        "evaluation_cells": plan,
        "arms": arms,
        "baseline_wait_deltas": _baseline_wait_deltas(arms),
        "checkpoint_binding": checkpoint_binding,
        "binding_paths": binding_paths,
        "records_total": sum(len(arm["records"]) for arm in arms.values()),
        "episodes_expected": 64,
        "episodes_processed": sum(len(arm["records"]) for arm in arms.values()),
        "development_only": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
        "training_or_update_called": False,
        "teacher_called": False,
    }
    _write_json(output_path / "pair_summary.json", result)
    _write_json(output_path / "manifest.json", result)
    _write_json(output_path / "progress.json", {
        "stage": STAGE,
        "status": "COMPLETED",
        "processed_episodes": result["episodes_processed"],
        "episodes_expected": result["episodes_expected"],
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "PPO_ALLOWED": False,
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(
            formal_root=args.formal_root,
            output=args.output,
            check_only=args.check_only,
            smoke_output=args.smoke_output,
        )
    except (EvaluationInputError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(_jsonable({
        "stage": result.get("stage"),
        "status": result.get("status"),
        "episodes_processed": result.get("episodes_processed"),
        "episodes_expected": result.get("episodes_expected"),
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    }), sort_keys=True), flush=True)
    return 0 if result.get("status") in {
        "CHECK_ONLY_OK", "COMPLETED", "SMOKE_COMPLETED",
    } else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
