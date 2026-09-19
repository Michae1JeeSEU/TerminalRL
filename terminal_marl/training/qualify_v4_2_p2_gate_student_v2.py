"""Fresh stochastic student-only qualification for the anchored repair V3.

This entry point deliberately has no training path.  It loads the repaired
temperature-calibrated student, reserves the six fresh V2 seeds once, and
evaluates sampled requested gates against the Teacher's label-only sidecar.
The Teacher never supplies the executed action.  A completed formal attempt
may be finalized from immutable episode artifacts exactly once without a
second rollout or reset.
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
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import audit_v4_2_p2_gate_learnability_v2 as audit
from terminal_marl.training import collect_v4_2_p2_gate_support_v2 as support_v2
from terminal_marl.training import qualify_v4_2_p2_gate_student_v1 as v1
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_student_qualification_v2.yaml")
PROTOCOL_SHA = "7ab9e2a84b9aea8da2ea975d98b4867de4a45bd5725bc72019b2e95ddbb75094"

REPAIR_MANIFEST_SHA = "e1e0883e355676593ab4d561aeb46fb37656629ec514ad6f01e12c3bbd17d410"
REPAIR_PROTOCOL_SHA = "f8a84632fea21e6789c8a40c0a22938cb5f0bdfcdb4ec9efd5e48304867778e8"
REPAIRED_CHECKPOINT_SHA = "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
BASE_CALIBRATED_CHECKPOINT_SHA = "61e03e02e3fb204d2db4e814097543905f51e1a3772626fdac1d7bc02ab8ca6b"
REPAIR_STAGE = "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3"
REPAIR_ARCHITECTURE_ID = CALIBRATED_ARCHITECTURE_ID
REPAIR_TEMPERATURE = CALIBRATION_TEMPERATURE
OBSERVATION_DIM = 88
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"

QUALIFICATION_STAGE = "V4_2_P2_GATE_STUDENT_QUALIFICATION_V2"
FORMAL_SEEDS = (782001, 782002, 782003, 782004, 782005, 782006)
FORMAL_SCENARIOS = ("flow6", "flow8", "flow12", "flow6", "flow8", "flow12")
OLD_QUALIFICATION_SEEDS = (774001, 774002, 774003, 774004, 774005, 774006)
SMOKE_SEED = 782000
SMOKE_SCENARIO = "flow6"
SMOKE_MAX_STEPS = 160
RESERVATION_MARKER = "p2_gate_student_qualification_v2_consumed.json"
SHIELD_ACTOR_FRACTION_MAX = 0.05
EXPECTED_FORMAL_REQUESTED = 52
INTERFACES = tuple(audit.INTERFACES)

# Behavioral helpers inherited from the reviewed V1 qualification path.  They
# do not carry V1 seed/binding constants into the V2 protocol or aggregation.
RepairInputError = v1.RepairInputError
_policy_tape_logprob_exact = support_v2._v1._policy_tape_logprob_exact
_summary_value = v1._summary_value
episode_gate_checks = v1.episode_gate_checks
_episode_gate_checks = episode_gate_checks
sampled_action_metrics = v1.sampled_action_metrics
sampled_gate_error_metrics = sampled_action_metrics
sampled_false_wait_release = sampled_action_metrics
sampled_action_threshold_checks = v1.sampled_action_threshold_checks
row_integrity_checks = v1.row_integrity_checks
_finite_leq = v1._finite_leq
_finite_geq = v1._finite_geq
_read_jsonl_rows = v1._read_jsonl_rows
_resolve_existing_artifact = v1._resolve_existing_artifact
_reconstruct_episode_record = v1._reconstruct_episode_record


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairInputError(message)


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
    """Convert nested runtime values deterministically for JSON manifests."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        try:
            return sorted(converted)
        except TypeError:
            return sorted(converted, key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ))
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
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
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def _all_boolean_flags_true(value: Any) -> bool:
    """Return true only when every boolean threshold flag is true."""
    if isinstance(value, Mapping):
        return all(_all_boolean_flags_true(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_boolean_flags_true(item) for item in value)
    if isinstance(value, bool):
        return value
    return True


def protocol() -> dict[str, Any]:
    """Load the byte-frozen V2 qualification protocol."""
    _require(PROTOCOL.is_file(), f"qualification protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "qualification protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "qualification protocol must be a mapping")
    _require(value.get("protocol") == QUALIFICATION_STAGE,
             "qualification protocol id changed")
    predecessor = value.get("predecessor", {})
    expected_predecessor = {
        "protocol": REPAIR_STAGE,
        "repair_manifest_sha256": REPAIR_MANIFEST_SHA,
        "repair_protocol_sha256": REPAIR_PROTOCOL_SHA,
        "repaired_checkpoint_sha256": REPAIRED_CHECKPOINT_SHA,
        "repaired_checkpoint_stage": REPAIR_STAGE,
        "repaired_architecture_id": REPAIR_ARCHITECTURE_ID,
        "repaired_temperature": REPAIR_TEMPERATURE,
        "repair_pass": True,
        "STUDENT_QUALIFICATION_ALLOWED": True,
    }
    for key, expected in expected_predecessor.items():
        _require(predecessor.get(key) == expected,
                 f"repair predecessor binding changed: {key}")
    bindings = value.get("bindings", {})
    for key, expected in {
        "observation_schema": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": 88,
        "action_schema": "low_action_schema_v4_2_hierarchical",
        "repaired_architecture_id": REPAIR_ARCHITECTURE_ID,
        "temperature": REPAIR_TEMPERATURE,
    }.items():
        actual = bindings.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"qualification binding changed: {key}")
    formal = value.get("formal", {})
    _require(tuple(formal.get("seeds", ())) == FORMAL_SEEDS,
             "formal qualification seeds changed")
    _require(tuple(formal.get("scenarios", ())) == FORMAL_SCENARIOS,
             "formal scenario assignment changed")
    _require(formal.get("stochastic") is True, "formal qualification is not stochastic")
    _require(formal.get("episodes") == len(FORMAL_SEEDS), "formal episode count changed")
    _require(formal.get("expected_requested") == EXPECTED_FORMAL_REQUESTED,
             "formal requested cardinality changed")
    _require(formal.get("max_steps_from_support_config") is True,
             "formal horizon binding changed")
    _require(formal.get("historical_registry_before_first_reset") is True,
             "historical registry ordering changed")
    _require(formal.get("reservation_marker") == RESERVATION_MARKER,
             "reservation marker changed")
    _require(formal.get("reserve_once") is True, "reservation is not one-shot")
    _require(tuple(formal.get("old_qualification_seeds_forbidden", ())) == OLD_QUALIFICATION_SEEDS,
             "old qualification seed boundary changed")
    smoke = value.get("smoke", {})
    _require(smoke.get("seed") == SMOKE_SEED and smoke.get("scenario") == SMOKE_SCENARIO,
             "smoke seed/scenario changed")
    _require(smoke.get("max_steps") == SMOKE_MAX_STEPS,
             "smoke horizon changed")
    _require(smoke.get("stochastic") is True and smoke.get("formal_marker") is False
             and smoke.get("authorization") is False,
             "smoke authorization boundary changed")
    training = value.get("training", {})
    for key in ("training_called", "optimizer_called", "policy_update",
                "repaired_checkpoint_optimizer_loaded"):
        _require(training.get(key) is False, f"training restriction changed: {key}")
    _require(training.get("teacher_label_only") is True
             and training.get("student_authoritative") is True,
             "Teacher/student provenance changed")
    thresholds = value.get("thresholds", {})
    actions = thresholds.get("sampled_actions", {})
    for key, expected in {
        "overall_false_wait_rate_max": 0.30,
        "overall_false_release_rate_max": 0.30,
        "interface_false_wait_rate_max": 0.40,
        "interface_false_release_rate_max": 0.40,
    }.items():
        _require(float(actions.get(key)) == expected,
                 f"sampled action threshold changed: {key}")
    probabilities = thresholds.get("probabilities", {})
    for key, expected in {
        "macro_F1_min": 0.75,
        "WAIT_recall_min": 0.65,
        "RELEASE_recall_min": 0.75,
        "brier_max": 0.20,
        "binary_nll_max": 0.70,
        "interface_macro_F1_min": 0.65,
    }.items():
        _require(float(probabilities.get(key)) == expected,
                 f"probability threshold changed: {key}")
    completion = thresholds.get("completeness", {})
    _require(completion.get("episodes") == len(FORMAL_SEEDS)
             and completion.get("requested") == EXPECTED_FORMAL_REQUESTED,
             "completion threshold changed")
    return value


def validate_repair_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = REPAIR_MANIFEST_SHA,
) -> dict[str, Any]:
    """Require a passing, qualification-authorized anchored repair manifest."""
    manifest_path = Path(path)
    _require(manifest_path.is_file(), f"repair manifest missing: {manifest_path}")
    if expected_sha256 is not None:
        _require(_sha256(manifest_path) == expected_sha256, "repair manifest SHA mismatch")
    manifest = _read_json(manifest_path)
    for key, expected in {
        "stage": REPAIR_STAGE,
        "status": "PASS",
        "repair_pass": True,
        "repair_training": "formal_offline_gate_repair",
        "support_manifest_sha256": audit.SUPPORT_MANIFEST_SHA,
        "audit_manifest_sha256": "5a72549d05c4733c2163512bf30cfe28d601c38e3ddfe23dc8291ec2f17b3679",
        "candidate_checkpoint_sha256": BASE_CALIBRATED_CHECKPOINT_SHA,
        "candidate_architecture_id": REPAIR_ARCHITECTURE_ID,
        "candidate_temperature": REPAIR_TEMPERATURE,
        "d1_dataset_sha256": audit.D1_DATASET_SHA,
        "d1_manifest_sha256": audit.D1_MANIFEST_SHA,
        "qualification_only": True,
        "STUDENT_QUALIFICATION_ALLOWED": True,
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
    }.items():
        actual = manifest.get(key)
        if isinstance(expected, float):
            _require(float(actual) == expected, f"repair manifest binding changed: {key}")
        else:
            _require(actual == expected, f"repair manifest binding changed: {key}")
    _require(manifest.get("protocol_sha256") == REPAIR_PROTOCOL_SHA,
             "repair protocol binding changed")
    _require(manifest.get("checkpoint_sha256") == REPAIRED_CHECKPOINT_SHA,
             "repair checkpoint binding changed")
    _require(manifest.get("checkpoint_role") == "qualification_candidate",
             "repair checkpoint role changed")
    thresholds = manifest.get("thresholds", {})
    for section in ("support_dev", "d1_anchor", "invariants"):
        _require(thresholds.get(section, {}).get("all") is True,
                 f"repair threshold section failed: {section}")
        _require(_all_boolean_flags_true(thresholds.get(section, {})),
                 f"repair threshold flag failed: {section}")
    _require(thresholds.get("all") is True, "repair threshold aggregate failed")
    return manifest


def _load_repaired_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = REPAIRED_CHECKPOINT_SHA,
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    """Load V3 repair weights while deliberately not loading its optimizer."""
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"repaired checkpoint missing: {checkpoint}")
    if expected_sha256 is not None:
        _require(_sha256(checkpoint) == expected_sha256,
                 "repaired checkpoint SHA mismatch")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM, temperature=REPAIR_TEMPERATURE,
    )
    payload = load_checkpoint(checkpoint, policy, p2_config(6), load_optimizer=False)
    checkpoint_temperature = payload.get("temperature")
    if checkpoint_temperature is None:
        checkpoint_temperature = float(policy.model.gate_temperature.detach().cpu())
        payload = dict(payload)
        payload["temperature"] = checkpoint_temperature
    for key, expected in {
        "stage": REPAIR_STAGE,
        "architecture_id": REPAIR_ARCHITECTURE_ID,
        "temperature": REPAIR_TEMPERATURE,
        "qualification_only": True,
        "qualification_pass": True,
        "STUDENT_QUALIFICATION_ALLOWED": True,
        "checkpoint_role": "qualification_candidate",
        "checkpoint_promotable": False,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False,
    }.items():
        actual = checkpoint_temperature if key == "temperature" else payload.get(key)
        if isinstance(expected, float):
            _require(float(actual) == expected, f"repaired checkpoint binding changed: {key}")
        else:
            _require(actual == expected, f"repaired checkpoint binding changed: {key}")
    _require("optimizer" in payload,
             "repaired checkpoint must retain formal repair optimizer state")
    _require(len(policy.optimizer.state) == 0,
             "qualification loaded repaired optimizer state")
    _require(float(policy.model.gate_temperature.detach().cpu()) == REPAIR_TEMPERATURE,
             "repaired checkpoint temperature buffer changed")
    policy.model.eval()
    return policy, payload


def _load_inputs(
    repair_manifest: str | Path,
    checkpoint: str | Path,
) -> tuple[dict[str, Any], TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    manifest = validate_repair_manifest(repair_manifest)
    policy, payload = _load_repaired_checkpoint(checkpoint)
    _require(manifest.get("checkpoint_sha256") == REPAIRED_CHECKPOINT_SHA,
             "repair manifest/checkpoint binding mismatch")
    return manifest, policy, payload


def formal_horizons() -> tuple[int, ...]:
    return tuple(int(support_v2.support_config(int(name.removeprefix("flow"))).max_simulation_seconds /
                     support_v2.support_config(int(name.removeprefix("flow"))).dt)
                 for name in FORMAL_SCENARIOS)


def reservation_marker_path(checkpoint: str | Path) -> Path:
    return Path(checkpoint).parent / RESERVATION_MARKER


def reserve_formal_attempt(
    marker: str | Path,
    history_root: str | Path,
    output: str | Path,
    *,
    protocol_sha256: str = PROTOCOL_SHA,
    seeds: Sequence[int] = FORMAL_SEEDS,
    checkpoint_sha256: str = REPAIRED_CHECKPOINT_SHA,
) -> dict[str, Any]:
    """Check history, then reserve the V2 formal seed set exactly once."""
    marker_path = Path(marker)
    _require(not marker_path.exists(), f"formal qualification marker already exists: {marker_path}")
    history_path = Path(history_root)
    _require(history_path.is_dir(), f"qualification history root missing: {history_path}")
    historical = support_v2.historical_registry(history_path, Path(output))
    consumed = set(int(value) for value in historical.get("seeds", set()))
    overlap = sorted(consumed.intersection(int(seed) for seed in seeds))
    _require(not overlap, f"formal qualification seeds already consumed: {overlap}")
    old_overlap = sorted(consumed.intersection(OLD_QUALIFICATION_SEEDS))
    payload = {
        "protocol": QUALIFICATION_STAGE,
        "protocol_sha256": protocol_sha256,
        "formal_seeds": [int(seed) for seed in seeds],
        "old_qualification_seeds": list(OLD_QUALIFICATION_SEEDS),
        "output": str(output),
        "repaired_checkpoint_sha256": checkpoint_sha256,
        "historical_seed_count": len(consumed),
        "old_qualification_seed_overlap": old_overlap,
        "reserved_before_first_reset": True,
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    support_v2.reserve_once(marker_path, payload)
    return {"marker": str(marker_path), "historical": historical, "payload": payload}


def probability_metrics(model: torch.nn.Module, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probabilities = audit._probabilities(model, rows)
    metrics = audit.gate_metrics(rows, probabilities)
    metrics["probabilities"] = probabilities.tolist()
    return metrics


def probability_threshold_checks(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "macro_F1": _finite_geq(metrics.get("macro_F1"), 0.75),
        "WAIT_recall": _finite_geq(metrics.get("WAIT_recall"), 0.65),
        "RELEASE_recall": _finite_geq(metrics.get("RELEASE_recall"), 0.75),
        "brier": _finite_leq(metrics.get("brier"), 0.20),
        "binary_nll": _finite_leq(metrics.get("binary_nll"), 0.70),
    }
    interface_checks = {
        interface: _finite_geq(metrics.get("interface", {}).get(interface, {}).get("macro_F1"), 0.65)
        for interface in INTERFACES
    }
    checks["interface"] = interface_checks
    checks["every_interface_macro_F1"] = bool(interface_checks) and all(interface_checks.values())
    checks["all"] = bool(all(value for key, value in checks.items() if key != "interface"))
    return checks


def _empty_threshold_result() -> dict[str, Any]:
    return {"all": False}


def _aggregate_records(
    policy: TemperatureCalibratedGateResidualAdapterStudentPPOV42,
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
    exception_text: str | None = None,
) -> dict[str, Any]:
    if rows:
        row_checks = row_integrity_checks(rows)
        sampled = sampled_action_metrics(rows)
        sampled_checks = sampled_action_threshold_checks(sampled)
        probabilities = probability_metrics(policy.model, rows)
        probability_checks = probability_threshold_checks(probabilities)
    else:
        row_checks = {"finite_88d_rows": False, "unique_decision_ids": False,
                      "both_requested_gates": False, "all": False}
        sampled = {}
        sampled_checks = _empty_threshold_result()
        probabilities = {}
        probability_checks = _empty_threshold_result()
    requested = sum(int(record.get("requested", 0)) for record in records)
    completed = sum(int(record.get("completed", 0)) for record in records)
    complete = {
        "episodes": len(records) == len(FORMAL_SEEDS)
        and {int(record.get("seed", -1)) for record in records} == set(FORMAL_SEEDS),
        "requested": requested == EXPECTED_FORMAL_REQUESTED
        and completed == EXPECTED_FORMAL_REQUESTED,
        "all_episode_gates": bool(records) and len(records) == len(FORMAL_SEEDS)
        and all(bool(record.get("qualification_gates", {}).get("all")) for record in records),
    }
    complete["all"] = all(complete.values())
    qualification_pass = False if smoke else bool(
        complete["all"] and row_checks["all"] and sampled_checks["all"]
        and probability_checks["all"] and exception_text is None
    )
    return {
        "qualification_pass": qualification_pass,
        "row_checks": row_checks,
        "sampled": sampled,
        "sampled_checks": sampled_checks,
        "probabilities": probabilities,
        "probability_checks": probability_checks,
        "completeness": {
            "episodes": len(records), "expected_episodes": len(FORMAL_SEEDS),
            "requested": requested, "completed": completed,
            "expected_requested": EXPECTED_FORMAL_REQUESTED,
        },
        "exception": exception_text,
    }


def _episode_plan(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [{"seed": SMOKE_SEED, "scenario": SMOKE_SCENARIO, "max_steps": SMOKE_MAX_STEPS}]
    return [{"seed": int(seed), "scenario": scenario, "max_steps": horizon}
            for seed, scenario, horizon in zip(FORMAL_SEEDS, FORMAL_SCENARIOS, formal_horizons())]


def _write_pre_reset_registry(output: Path, marker: Path) -> None:
    _write_json(output / "pre_reset_registry.json", {
        "built_before_first_reset": True,
        "historical_registry_checked": True,
        "formal_seeds": list(FORMAL_SEEDS),
        "old_qualification_seeds_forbidden": list(OLD_QUALIFICATION_SEEDS),
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "reservation_marker": str(marker),
    })


def run(
    output: str | Path,
    repair_manifest: str | Path,
    checkpoint: str | Path,
    history_root: str | Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run one smoke or formal stochastic V2 qualification attempt."""
    protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"qualification output already exists: {output_path}")
    repair_manifest_value, policy, checkpoint_payload = _load_inputs(repair_manifest, checkpoint)
    history_path = Path(history_root)
    if not smoke:
        _require(history_path.is_dir(), f"qualification history root missing: {history_path}")
    output_path.mkdir(parents=True, exist_ok=False)
    reservation: dict[str, Any] | None = None
    marker_path = reservation_marker_path(checkpoint)
    if not smoke:
        reservation = reserve_formal_attempt(
            marker_path, history_path, output_path,
            protocol_sha256=PROTOCOL_SHA or _sha256(PROTOCOL),
        )
        _write_pre_reset_registry(output_path, marker_path)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    exception_text: str | None = None
    plan = _episode_plan(smoke=smoke)
    for index, item in enumerate(plan):
        episode_output = output_path / f"seed{item['seed']}_{item['scenario']}"
        try:
            summary, episode_rows = support_v2._episode_record(
                policy, split="qualification", seed=int(item["seed"]),
                scenario=str(item["scenario"]), output=episode_output,
                max_steps=int(item["max_steps"]),
            )
            summary = dict(summary)
            summary["rows"] = len(episode_rows)
            summary["qualification_gates"] = episode_gate_checks(summary, int(item["max_steps"]))
            records.append(summary)
            rows.extend(dict(row) for row in episode_rows)
            _write_json(output_path / "progress.json", {
                "stage": QUALIFICATION_STAGE, "smoke": smoke,
                "episodes_completed": index + 1, "episodes_planned": len(plan),
                "seeds_completed": [record.get("seed") for record in records],
                "rows": len(rows), "last_episode_gates": summary["qualification_gates"],
            })
            if not summary["qualification_gates"]["all"]:
                exception_text = "episode gate failure"
                break
        except Exception as exc:
            exception_text = f"{type(exc).__name__}: {exc}"
            records.append({"seed": int(item["seed"]), "scenario": str(item["scenario"]),
                            "max_steps": int(item["max_steps"]), "exception": exception_text,
                            "qualification_gates": {"all": False}})
            _write_json(output_path / "progress.json", {
                "stage": QUALIFICATION_STAGE, "smoke": smoke,
                "episodes_completed": index, "episodes_planned": len(plan),
                "exception": exception_text,
            })
            break
    aggregate = _aggregate_records(policy, records, rows, smoke=smoke,
                                   exception_text=exception_text)
    qualification_pass = bool(aggregate["qualification_pass"])
    result = {
        "stage": QUALIFICATION_STAGE,
        "status": "PASS" if qualification_pass else "FAIL",
        "evidence_type": "newly_run",
        "smoke": bool(smoke),
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "protocol": str(PROTOCOL),
        "repair_manifest": str(repair_manifest),
        "repair_manifest_sha256": REPAIR_MANIFEST_SHA,
        "repair_protocol_sha256": REPAIR_PROTOCOL_SHA,
        "repaired_checkpoint": str(checkpoint),
        "repaired_checkpoint_sha256": REPAIRED_CHECKPOINT_SHA,
        "repair_status": {
            "status": repair_manifest_value.get("status"),
            "repair_pass": repair_manifest_value.get("repair_pass"),
            "STUDENT_QUALIFICATION_ALLOWED": repair_manifest_value.get("STUDENT_QUALIFICATION_ALLOWED"),
        },
        "checkpoint_payload": {
            "stage": checkpoint_payload.get("stage"),
            "architecture_id": checkpoint_payload.get("architecture_id"),
            "temperature": checkpoint_payload.get("temperature"),
            "qualification_pass": checkpoint_payload.get("qualification_pass"),
            "qualification_only": checkpoint_payload.get("qualification_only"),
            "checkpoint_role": checkpoint_payload.get("checkpoint_role"),
            "checkpoint_promotable": checkpoint_payload.get("checkpoint_promotable"),
            "optimizer_present_but_not_loaded": "optimizer" in checkpoint_payload,
        },
        "architecture_id": REPAIR_ARCHITECTURE_ID,
        "temperature": REPAIR_TEMPERATURE,
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_scenarios": list(FORMAL_SCENARIOS),
        "smoke_seed": SMOKE_SEED,
        "records": records,
        "rows": len(rows),
        "row_checks": aggregate["row_checks"],
        "sampled_action_metrics": aggregate["sampled"],
        "sampled_action_checks": aggregate["sampled_checks"],
        "probability_metrics": aggregate["probabilities"],
        "probability_checks": aggregate["probability_checks"],
        "completeness": aggregate["completeness"],
        "reservation": reservation,
        "exception": exception_text,
        "P2_GATE_STUDENT_QUALIFICATION": "PASS" if qualification_pass else "FAIL",
        "P2_EFFECTIVENESS_EVALUATION_ALLOWED": bool(qualification_pass),
        "checkpoint_role": "effectiveness_candidate" if qualification_pass else None,
        "qualification_only": True,
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "training_called": False,
        "optimizer_called": False,
        "policy_update": False,
        "repaired_checkpoint_optimizer_loaded": False,
        "environment_rollout": True,
        "teacher_called": True,
        "teacher_label_only": True,
        "student_authoritative": True,
        "formal_marker_created": bool(reservation is not None),
        "qualification_seed_selection": False,
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    manifest_path = output_path / "manifest.json"
    result["manifest"] = str(manifest_path)
    _write_json(manifest_path, result)
    print(json.dumps({"stage": QUALIFICATION_STAGE, "status": result["status"],
                      "P2_GATE_STUDENT_QUALIFICATION": result["P2_GATE_STUDENT_QUALIFICATION"],
                      "P2_EFFECTIVENESS_EVALUATION_ALLOWED": result["P2_EFFECTIVENESS_EVALUATION_ALLOWED"],
                      "PPO_ALLOWED": False}, sort_keys=True), flush=True)
    return result


def _validate_completed_artifacts(
    output: Path, checkpoint: Path, marker_payload: Mapping[str, Any],
) -> dict[str, Any]:
    progress_path = output / "progress.json"
    registry_path = output / "pre_reset_registry.json"
    manifest_path = output / "manifest.json"
    _require(not manifest_path.exists(), f"qualification manifest already exists: {manifest_path}")
    _require(progress_path.is_file(), f"qualification progress missing: {progress_path}")
    _require(registry_path.is_file(), f"pre-reset registry missing: {registry_path}")
    progress = _read_json(progress_path)
    _require(progress.get("stage") == QUALIFICATION_STAGE, "qualification progress stage mismatch")
    _require(progress.get("smoke") is False, "completed artifacts are not formal")
    _require(progress.get("episodes_completed") == len(FORMAL_SEEDS)
             and progress.get("episodes_planned") == len(FORMAL_SEEDS),
             "qualification progress is not exactly 6/6")
    _require(list(progress.get("seeds_completed", ())) == list(FORMAL_SEEDS),
             "qualification progress seed registry mismatch")
    registry = _read_json(registry_path)
    marker_path = checkpoint.parent / RESERVATION_MARKER
    _require(registry.get("built_before_first_reset") is True,
             "pre-reset registry was not built before first reset")
    _require(registry.get("historical_registry_checked") is True,
             "pre-reset historical registry check missing")
    _require(registry.get("formal_seeds") == list(FORMAL_SEEDS),
             "pre-reset formal seed registry mismatch")
    _require(registry.get("protocol_sha256") == (PROTOCOL_SHA or _sha256(PROTOCOL)),
             "pre-reset protocol binding mismatch")
    _require(registry.get("reservation_marker") == str(marker_path),
             "pre-reset reservation marker mismatch")
    for key, expected in {
        "protocol": QUALIFICATION_STAGE,
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "formal_seeds": list(FORMAL_SEEDS),
        "repaired_checkpoint_sha256": REPAIRED_CHECKPOINT_SHA,
        "reserved_before_first_reset": True,
    }.items():
        _require(marker_payload.get(key) == expected, f"reservation marker binding changed: {key}")
    _require(marker_payload.get("output") == str(output),
             "reservation marker output mismatch")
    _require(marker_payload.get("old_qualification_seeds") == list(OLD_QUALIFICATION_SEEDS),
             "reservation marker old-seed boundary changed")
    return {"progress": progress, "pre_reset_registry": registry, "marker": marker_payload}


def finalize_existing(
    output: str | Path,
    repair_manifest: str | Path,
    checkpoint: str | Path,
    history_root: str | Path,
) -> dict[str, Any]:
    """Finalize completed V2 artifacts without rollout, reset, or reservation."""
    protocol()
    output_path = Path(output)
    _require(output_path.is_dir(), f"completed qualification output missing: {output_path}")
    _require(not (output_path / "manifest.json").exists(),
             f"qualification manifest already exists: {output_path / 'manifest.json'}")
    history_path = Path(history_root)
    _require(history_path.is_dir(), f"qualification history root missing: {history_path}")
    repair_manifest_value, policy, checkpoint_payload = _load_inputs(repair_manifest, checkpoint)
    checkpoint_path = Path(checkpoint)
    marker_path = reservation_marker_path(checkpoint_path)
    _require(marker_path.is_file(), f"formal reservation marker missing: {marker_path}")
    marker_payload = _read_json(marker_path)
    top_level = _validate_completed_artifacts(output_path, checkpoint_path, marker_payload)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for seed, scenario, max_steps in zip(FORMAL_SEEDS, FORMAL_SCENARIOS, formal_horizons()):
        summary, episode_rows = _reconstruct_episode_record(
            policy, output=output_path, seed=int(seed), scenario=scenario,
            max_steps=int(max_steps),
        )
        records.append(summary)
        rows.extend(episode_rows)
    aggregate = _aggregate_records(policy, records, rows, smoke=False, exception_text=None)
    qualification_pass = bool(aggregate["qualification_pass"])
    result = {
        "stage": QUALIFICATION_STAGE,
        "status": "PASS" if qualification_pass else "FAIL",
        "evidence_type": "recovered_completed_artifacts",
        "smoke": False,
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "protocol": str(PROTOCOL),
        "repair_manifest": str(repair_manifest),
        "repair_manifest_sha256": REPAIR_MANIFEST_SHA,
        "repair_protocol_sha256": REPAIR_PROTOCOL_SHA,
        "repaired_checkpoint": str(checkpoint),
        "repaired_checkpoint_sha256": REPAIRED_CHECKPOINT_SHA,
        "repair_status": {
            "status": repair_manifest_value.get("status"),
            "repair_pass": repair_manifest_value.get("repair_pass"),
            "STUDENT_QUALIFICATION_ALLOWED": repair_manifest_value.get("STUDENT_QUALIFICATION_ALLOWED"),
        },
        "checkpoint_payload": {
            "stage": checkpoint_payload.get("stage"),
            "architecture_id": checkpoint_payload.get("architecture_id"),
            "temperature": checkpoint_payload.get("temperature"),
            "qualification_pass": checkpoint_payload.get("qualification_pass"),
            "qualification_only": checkpoint_payload.get("qualification_only"),
            "checkpoint_role": checkpoint_payload.get("checkpoint_role"),
            "checkpoint_promotable": checkpoint_payload.get("checkpoint_promotable"),
            "optimizer_present_but_not_loaded": "optimizer" in checkpoint_payload,
        },
        "architecture_id": REPAIR_ARCHITECTURE_ID,
        "temperature": REPAIR_TEMPERATURE,
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_scenarios": list(FORMAL_SCENARIOS),
        "records": records,
        "rows": len(rows),
        "row_checks": aggregate["row_checks"],
        "sampled_action_metrics": aggregate["sampled"],
        "sampled_action_checks": aggregate["sampled_checks"],
        "probability_metrics": aggregate["probabilities"],
        "probability_checks": aggregate["probability_checks"],
        "completeness": aggregate["completeness"],
        "reservation": {"marker": str(marker_path), "reused_existing": True,
                        "payload": marker_payload},
        "pre_reset_registry": top_level["pre_reset_registry"],
        "exception": None,
        "recovery_note": "recovered_from_completed_artifacts",
        "recovered_from_completed_artifacts": True,
        "formal_rollouts_repeated": False,
        "formal_seed_resets": 0,
        "P2_GATE_STUDENT_QUALIFICATION": "PASS" if qualification_pass else "FAIL",
        "P2_EFFECTIVENESS_EVALUATION_ALLOWED": bool(qualification_pass),
        "checkpoint_role": "effectiveness_candidate" if qualification_pass else None,
        "qualification_only": True,
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "checkpoint_promotable": False,
        "training_called": False,
        "optimizer_called": False,
        "policy_update": False,
        "repaired_checkpoint_optimizer_loaded": False,
        "environment_rollout": True,
        "teacher_called": True,
        "teacher_label_only": True,
        "student_authoritative": True,
        "formal_marker_created": False,
        "formal_marker_present": True,
        "qualification_seed_selection": False,
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    manifest_path = output_path / "manifest.json"
    result["manifest"] = str(manifest_path)
    _write_json(manifest_path, result)
    print(json.dumps({"stage": QUALIFICATION_STAGE, "status": result["status"],
                      "recovered_from_completed_artifacts": True,
                      "formal_rollouts_repeated": False, "formal_seed_resets": 0,
                      "P2_EFFECTIVENESS_EVALUATION_ALLOWED": result["P2_EFFECTIVENESS_EVALUATION_ALLOWED"],
                      "PPO_ALLOWED": False}, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    torch.set_num_threads(1)
    args = parser.parse_args()
    if args.finalize_existing:
        if args.smoke:
            raise SystemExit("--finalize-existing cannot be combined with --smoke")
        print(json.dumps(finalize_existing(
            args.output, args.repair_manifest, args.checkpoint, args.history_root,
        ), indent=2, sort_keys=True), flush=True)
    else:
        print(json.dumps(run(
            args.output, args.repair_manifest, args.checkpoint, args.history_root,
            smoke=args.smoke,
        ), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    _main()
