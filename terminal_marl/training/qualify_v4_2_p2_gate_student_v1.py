"""Fresh stochastic, student-authoritative P2 gate qualification.

This entry point deliberately has no training path.  It loads the fixed
temperature-calibrated candidate, reserves the six preregistered future
seeds once, and evaluates the sampled requested gate against the Teacher's
label-only sidecar.  The Teacher never supplies the executed action.
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
from terminal_marl.training import audit_v4_2_p2_gate_learnability_v1 as audit
from terminal_marl.training import calibrate_v4_2_p2_gate_v3 as calibration_v3
from terminal_marl.training import collect_v4_2_p2_gate_support_v2 as support_v2
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_student_qualification_v1.yaml")
PROTOCOL_SHA = "8064bebd115cad253e22ea019f678bb0a5c9085bb8a47989319fd9956a54aa65"
CALIBRATION_MANIFEST_SHA = "a784ffa753d4588d9a0b997a8ee913010c128abfbf49c96b5e790f5bf6ae7272"
CALIBRATED_CHECKPOINT_SHA = "61e03e02e3fb204d2db4e814097543905f51e1a3772626fdac1d7bc02ab8ca6b"
CALIBRATION_PROTOCOL_SHA = "c5e0b50f645f488cf3d36784cd37ba97fa016cae11decb9ac149bf3b932ba946"
CALIBRATION_STAGE = "V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3"
QUALIFICATION_STAGE = "V4_2_P2_GATE_STUDENT_QUALIFICATION_V1"
FORMAL_SEEDS = (774001, 774002, 774003, 774004, 774005, 774006)
FORMAL_SCENARIOS = ("flow6", "flow8", "flow12", "flow6", "flow8", "flow12")
SMOKE_SEED = 774000
SMOKE_SCENARIO = "flow6"
SMOKE_MAX_STEPS = 160
RESERVATION_MARKER = "p2_gate_student_qualification_v1_consumed.json"
SHIELD_ACTOR_FRACTION_MAX = 0.05
EXPECTED_FORMAL_REQUESTED = 52
INTERFACES = tuple(audit.INTERFACES)
RepairInputError = calibration_v3.RepairInputError


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
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
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


def protocol() -> dict[str, Any]:
    """Load the frozen qualification protocol after checking its byte hash."""
    _require(PROTOCOL.is_file(), f"qualification protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "qualification protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "qualification protocol must be a mapping")
    _require(value.get("protocol") == QUALIFICATION_STAGE, "qualification protocol id changed")
    predecessor = value.get("predecessor", {})
    expected_predecessor = {
        "protocol": CALIBRATION_STAGE,
        "calibration_manifest_sha256": CALIBRATION_MANIFEST_SHA,
        "calibrated_checkpoint_sha256": CALIBRATED_CHECKPOINT_SHA,
        "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA,
    }
    for key, expected in expected_predecessor.items():
        _require(predecessor.get(key) == expected, f"calibration binding changed: {key}")
    bindings = value.get("bindings", {})
    for key, expected in {
        "observation_schema": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": 88,
        "action_schema": "low_action_schema_v4_2_hierarchical",
        "calibrated_architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": CALIBRATION_TEMPERATURE,
    }.items():
        actual = bindings.get(key)
        _require(float(actual) == expected if isinstance(expected, float) else actual == expected,
                 f"qualification binding changed: {key}")
    formal = value.get("formal", {})
    _require(tuple(formal.get("seeds", ())) == FORMAL_SEEDS, "formal qualification seeds changed")
    _require(tuple(formal.get("scenarios", ())) == FORMAL_SCENARIOS, "formal scenario assignment changed")
    _require(formal.get("stochastic") is True, "formal qualification is not stochastic")
    _require(formal.get("episodes") == len(FORMAL_SEEDS), "formal episode count changed")
    _require(formal.get("expected_requested") == EXPECTED_FORMAL_REQUESTED,
             "formal requested cardinality changed")
    _require(formal.get("max_steps_from_support_config") is True, "formal horizon binding changed")
    _require(formal.get("historical_registry_before_first_reset") is True,
             "historical registry ordering changed")
    _require(formal.get("reservation_marker") == RESERVATION_MARKER, "reservation marker changed")
    _require(formal.get("reserve_once") is True, "reservation is not one-shot")
    smoke = value.get("smoke", {})
    _require(smoke.get("seed") == SMOKE_SEED and smoke.get("scenario") == SMOKE_SCENARIO,
             "smoke seed/scenario changed")
    _require(smoke.get("max_steps") == SMOKE_MAX_STEPS, "smoke horizon changed")
    _require(smoke.get("stochastic") is True and smoke.get("formal_marker") is False,
             "smoke authorization boundary changed")
    training = value.get("training", {})
    for key in ("training_called", "optimizer_called", "policy_update"):
        _require(training.get(key) is False, f"training restriction changed: {key}")
    _require(training.get("teacher_label_only") is True and training.get("student_authoritative") is True,
             "Teacher/student provenance changed")
    thresholds = value.get("thresholds", {})
    actions = thresholds.get("sampled_actions", {})
    expected_action_thresholds = {
        "overall_false_wait_rate_max": 0.30,
        "overall_false_release_rate_max": 0.30,
        "interface_false_wait_rate_max": 0.40,
        "interface_false_release_rate_max": 0.40,
    }
    for key, expected in expected_action_thresholds.items():
        _require(float(actions.get(key)) == expected, f"sampled action threshold changed: {key}")
    probabilities = thresholds.get("probabilities", {})
    expected_probability_thresholds = {
        "macro_F1_min": 0.75,
        "WAIT_recall_min": 0.65,
        "RELEASE_recall_min": 0.75,
        "brier_max": 0.20,
        "binary_nll_max": 0.70,
        "interface_macro_F1_min": 0.65,
    }
    for key, expected in expected_probability_thresholds.items():
        _require(float(probabilities.get(key)) == expected, f"probability threshold changed: {key}")
    completion = thresholds.get("completeness", {})
    _require(completion.get("episodes") == len(FORMAL_SEEDS), "completion episode gate changed")
    _require(completion.get("requested") == EXPECTED_FORMAL_REQUESTED, "completion requested gate changed")
    return value


def validate_calibration_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = CALIBRATION_MANIFEST_SHA,
) -> dict[str, Any]:
    """Require a passing, qualification-authorized V3 calibration manifest."""
    manifest_path = Path(path)
    _require(manifest_path.is_file(), f"calibration manifest missing: {manifest_path}")
    if expected_sha256 is not None:
        _require(_sha256(manifest_path) == expected_sha256, "calibration manifest SHA mismatch")
    manifest = _read_json(manifest_path)
    expected = {
        "stage": CALIBRATION_STAGE,
        "status": "PASS",
        "calibration_pass": True,
        "STUDENT_QUALIFICATION_ALLOWED": True,
        "qualification_only": True,
        "checkpoint_role": "qualification_candidate",
        "checkpoint_promotable": False,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": CALIBRATION_TEMPERATURE,
        "protocol_sha256": CALIBRATION_PROTOCOL_SHA,
        "checkpoint_sha256": CALIBRATED_CHECKPOINT_SHA,
    }
    for key, wanted in expected.items():
        actual = manifest.get(key)
        if isinstance(wanted, float):
            _require(float(actual) == wanted, f"calibration manifest binding changed: {key}")
        else:
            _require(actual == wanted, f"calibration manifest binding changed: {key}")
    _require(manifest.get("training_called") is False, "calibration used training")
    _require(manifest.get("optimizer_called") is False, "calibration used optimizer")
    _require(manifest.get("environment_called") is False, "calibration used environment")
    _require(manifest.get("teacher_called") is False, "calibration used Teacher")
    return manifest


def _load_calibrated_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = CALIBRATED_CHECKPOINT_SHA,
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"calibrated checkpoint missing: {checkpoint}")
    if expected_sha256 is not None:
        _require(_sha256(checkpoint) == expected_sha256, "calibrated checkpoint SHA mismatch")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88, temperature=CALIBRATION_TEMPERATURE)
    payload = load_checkpoint(checkpoint, policy, p2_config(6), load_optimizer=False)
    expected = {
        "stage": CALIBRATION_STAGE,
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": CALIBRATION_TEMPERATURE,
        "qualification_only": True,
        "checkpoint_role": "qualification_candidate",
        "checkpoint_promotable": False,
        "PPO_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "curriculum_promotion_allowed": False,
    }
    for key, wanted in expected.items():
        actual = payload.get(key)
        if isinstance(wanted, float):
            _require(float(actual) == wanted, f"calibrated checkpoint binding changed: {key}")
        else:
            _require(actual == wanted, f"calibrated checkpoint binding changed: {key}")
    _require(payload.get("optimizer_saved") is False and "optimizer" not in payload,
             "calibrated checkpoint contains optimizer state")
    policy.model.eval()
    return policy, payload


def _load_inputs(
    calibration_manifest: str | Path,
    checkpoint: str | Path,
) -> tuple[dict[str, Any], TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    manifest = validate_calibration_manifest(calibration_manifest)
    policy, payload = _load_calibrated_checkpoint(checkpoint)
    _require(manifest.get("checkpoint_sha256") == CALIBRATED_CHECKPOINT_SHA,
             "calibration manifest/checkpoint binding mismatch")
    return manifest, policy, payload


def formal_horizons() -> tuple[int, ...]:
    """Return the fixed simulator horizon for each formal scenario."""
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
    checkpoint_sha256: str = CALIBRATED_CHECKPOINT_SHA,
) -> dict[str, Any]:
    """Check historical identities, then reserve the formal seed set once."""
    marker_path = Path(marker)
    _require(not marker_path.exists(), f"formal qualification marker already exists: {marker_path}")
    history_path = Path(history_root)
    _require(history_path.is_dir(), f"qualification history root missing: {history_path}")
    historical = support_v2.historical_registry(history_path, Path(output))
    consumed = set(int(value) for value in historical.get("seeds", set()))
    overlap = sorted(consumed.intersection(int(seed) for seed in seeds))
    _require(not overlap, f"formal qualification seeds already consumed: {overlap}")
    payload = {
        "protocol": QUALIFICATION_STAGE,
        "protocol_sha256": protocol_sha256,
        "formal_seeds": [int(seed) for seed in seeds],
        "output": str(output),
        "calibrated_checkpoint_sha256": checkpoint_sha256,
        "historical_seed_count": len(consumed),
        "reserved_before_first_reset": True,
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    support_v2.reserve_once(marker_path, payload)
    return {"marker": str(marker_path), "historical": historical, "payload": payload}


def _summary_value(summary: Mapping[str, Any], *keys: str, default: Any = False) -> Any:
    for key in keys:
        if key in summary:
            return summary[key]
    return default


def episode_gate_checks(summary: Mapping[str, Any], max_steps: int) -> dict[str, bool]:
    """Apply the per-episode safety/provenance gates to one V2 record."""
    requested = int(summary.get("requested", summary.get("config_requested", 0)))
    completed = int(summary.get("completed", 0))
    queue = summary.get("queue", ())
    if isinstance(queue, Sequence) and not isinstance(queue, (str, bytes)):
        queue_final = queue[-1] if queue else summary.get("queue_final")
    else:
        queue_final = summary.get("queue_final", queue)
    safety = summary.get("safety", {})
    safety_zero = isinstance(safety, Mapping) and bool(safety) and all(int(value) == 0 for value in safety.values())
    counts = summary.get("counts", {})
    actor_decisions = int(summary.get("actor_decisions", counts.get("actor_decisions", 0)))
    shield_actor_count = int(summary.get("shield_actor_count", counts.get("shield_actor", 0)))
    fraction = summary.get("shield_actor_fraction")
    if fraction is None:
        fraction = shield_actor_count / max(1, actor_decisions)
    finite_fraction = bool(np.isfinite(float(fraction)))
    checks = {
        "completed_equals_requested": requested > 0 and completed == requested,
        "terminal": bool(summary.get("terminal", False)),
        "queue_final_zero": queue_final == 0,
        "within_horizon": 0 <= int(summary.get("steps", 0)) <= int(max_steps),
        "safety_all_zero": safety_zero,
        "shield_actor_fraction_within_limit": finite_fraction and float(fraction) <= SHIELD_ACTOR_FRACTION_MAX,
        "exact_replay": bool(_summary_value(summary, "replay_exact", default=False)),
        "requested_logprob_exact": bool(_summary_value(summary, "requested_logprob_exact", default=False)),
        "teacher_label_only": bool(_summary_value(summary, "teacher_label_only", default=False)),
        "student_authoritative": bool(_summary_value(summary, "student_authoritative", default=False)),
        "actor_rows_positive": int(summary.get("actor_rows", 0)) > 0,
        "no_training_or_optimizer": summary.get("training_called", False) is False
        and summary.get("optimizer_called", False) is False
        and summary.get("policy_update", False) is False,
    }
    checks["all"] = all(checks.values())
    return checks


_episode_gate_checks = episode_gate_checks


def _class_counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return {
        "WAIT": sum(int(row[key]) == 0 for row in rows),
        "RELEASE": sum(int(row[key]) == 1 for row in rows),
    }


def _error_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    teacher = _class_counts(rows, "teacher_gate")
    requested = _class_counts(rows, "requested_gate")
    false_wait = sum(int(row["teacher_gate"]) == 1 and int(row["requested_gate"]) == 0 for row in rows)
    false_release = sum(int(row["teacher_gate"]) == 0 and int(row["requested_gate"]) == 1 for row in rows)
    false_wait_rate = float(false_wait / teacher["RELEASE"]) if teacher["RELEASE"] else float("nan")
    false_release_rate = float(false_release / teacher["WAIT"]) if teacher["WAIT"] else float("nan")
    return {
        "samples": len(rows),
        "teacher": teacher,
        "requested": requested,
        "false_WAIT": int(false_wait),
        "false_RELEASE": int(false_release),
        "false_wait_rate": false_wait_rate,
        "false_release_rate": false_release_rate,
        "false_WAIT_rate": false_wait_rate,
        "false_RELEASE_rate": false_release_rate,
        "both_teacher_classes": teacher["WAIT"] > 0 and teacher["RELEASE"] > 0,
        "both_requested_gates": requested["WAIT"] > 0 and requested["RELEASE"] > 0,
    }


def sampled_action_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score sampled requested gates against label-only Teacher gates."""
    _require(bool(rows), "cannot score an empty sampled qualification set")
    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        for key in ("teacher_gate", "requested_gate"):
            try:
                row[key] = int(row[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"row {index} has invalid {key}") from exc
            _require(row[key] in (0, 1), f"row {index} has non-binary {key}")
        normalized.append(row)
    result = _error_rates(normalized)
    interface_metrics: dict[str, dict[str, Any]] = {}
    for interface in INTERFACES:
        interface_metrics[interface] = _error_rates(
            [row for row in normalized if str(row.get("interface", "")) == interface]
        )
    result["interface"] = interface_metrics
    result["interfaces"] = interface_metrics
    return result


sampled_gate_error_metrics = sampled_action_metrics
sampled_false_wait_release = sampled_action_metrics


def _finite_leq(value: Any, limit: float) -> bool:
    try:
        return bool(np.isfinite(float(value)) and float(value) <= float(limit))
    except (TypeError, ValueError):
        return False


def sampled_action_threshold_checks(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply inclusive sampled-action disagreement thresholds."""
    overall = {
        "both_teacher_classes": bool(metrics.get("both_teacher_classes", False)),
        "both_requested_gates": bool(metrics.get("both_requested_gates", False)),
        "false_WAIT_rate": _finite_leq(
            metrics.get("false_wait_rate", metrics.get("false_WAIT_rate")), 0.30
        ),
        "false_RELEASE_rate": _finite_leq(
            metrics.get("false_release_rate", metrics.get("false_RELEASE_rate")), 0.30
        ),
    }
    interface_checks: dict[str, Any] = {}
    for interface in INTERFACES:
        item = metrics.get("interface", {}).get(interface, {})
        interface_checks[interface] = {
            "both_teacher_classes": bool(item.get("both_teacher_classes", False)),
            "false_WAIT_rate": _finite_leq(
                item.get("false_wait_rate", item.get("false_WAIT_rate")), 0.40
            ),
            "false_RELEASE_rate": _finite_leq(
                item.get("false_release_rate", item.get("false_RELEASE_rate")), 0.40
            ),
        }
        interface_checks[interface]["all"] = all(interface_checks[interface].values())
    return {
        **overall,
        "interface": interface_checks,
        "all": bool(all(overall.values()) and all(item["all"] for item in interface_checks.values())),
    }


def probability_metrics(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate calibrated probabilities on the same fresh actor rows."""
    probabilities = audit._probabilities(model, rows)
    metrics = audit.gate_metrics(rows, probabilities)
    metrics["probabilities"] = probabilities.tolist()
    return metrics


def probability_threshold_checks(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply inclusive probability metric thresholds."""
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
    checks["all"] = bool(all(checks.values()))
    return checks


def _finite_geq(value: Any, minimum: float) -> bool:
    try:
        return bool(np.isfinite(float(value)) and float(value) >= float(minimum))
    except (TypeError, ValueError):
        return False


def row_integrity_checks(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    """Check finite 88D observations, unique event IDs, and gate coverage."""
    observations_ok = True
    event_ids: list[str] = []
    requested_values: list[int] = []
    for row in rows:
        values = row.get("observation_v2_1")
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            observations_ok = False
            continue
        observations_ok = observations_ok and array.shape == (88,) and bool(np.isfinite(array).all())
        event_ids.append(str(row.get("decision_event_id", "")))
        try:
            requested_values.append(int(row["requested_gate"]))
        except (KeyError, TypeError, ValueError):
            requested_values.append(-1)
    checks = {
        "finite_88d_rows": bool(rows) and observations_ok,
        "unique_decision_ids": bool(event_ids)
        and all(event not in {"", "None"} for event in event_ids)
        and len(event_ids) == len(set(event_ids)),
        "both_requested_gates": 0 in requested_values and 1 in requested_values,
    }
    checks["all"] = all(checks.values())
    return checks


def _empty_threshold_result() -> dict[str, Any]:
    return {"all": False}


def _episode_plan(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [{"seed": SMOKE_SEED, "scenario": SMOKE_SCENARIO, "max_steps": SMOKE_MAX_STEPS}]
    return [
        {"seed": int(seed), "scenario": scenario, "max_steps": horizon}
        for seed, scenario, horizon in zip(FORMAL_SEEDS, FORMAL_SCENARIOS, formal_horizons())
    ]


def run(
    output: str | Path,
    calibration_manifest: str | Path,
    checkpoint: str | Path,
    history_root: str | Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run one smoke or formal stochastic qualification attempt."""
    protocol_value = protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"qualification output already exists: {output_path}")
    calibration_manifest_value, policy, checkpoint_payload = _load_inputs(calibration_manifest, checkpoint)
    history_path = Path(history_root)
    if not smoke:
        _require(history_path.is_dir(), f"qualification history root missing: {history_path}")
    output_path.mkdir(parents=True, exist_ok=False)

    reservation: dict[str, Any] | None = None
    if not smoke:
        reservation = reserve_formal_attempt(
            reservation_marker_path(checkpoint),
            history_path,
            output_path,
            protocol_sha256=PROTOCOL_SHA or _sha256(PROTOCOL),
        )
        _write_json(output_path / "pre_reset_registry.json", {
            "built_before_first_reset": True,
            "historical_registry_checked": True,
            "formal_seeds": list(FORMAL_SEEDS),
            "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
            "reservation_marker": str(reservation_marker_path(checkpoint)),
        })

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    exception_text: str | None = None
    for index, item in enumerate(_episode_plan(smoke=smoke)):
        episode_output = output_path / f"seed{item['seed']}_{item['scenario']}"
        try:
            summary, episode_rows = support_v2._episode_record(
                policy,
                split="qualification",
                seed=int(item["seed"]),
                scenario=str(item["scenario"]),
                output=episode_output,
                max_steps=int(item["max_steps"]),
            )
            summary = dict(summary)
            summary["rows"] = len(episode_rows)
            gates = episode_gate_checks(summary, int(item["max_steps"]))
            summary["qualification_gates"] = gates
            records.append(summary)
            rows.extend(dict(row) for row in episode_rows)
            _write_json(output_path / "progress.json", {
                "stage": QUALIFICATION_STAGE,
                "smoke": smoke,
                "episodes_completed": index + 1,
                "episodes_planned": len(_episode_plan(smoke=smoke)),
                "seeds_completed": [record.get("seed") for record in records],
                "rows": len(rows),
                "last_episode_gates": gates,
            })
            if not gates["all"]:
                exception_text = "episode gate failure"
                break
        except Exception as exc:  # preserve terminal evidence and stop immediately
            exception_text = f"{type(exc).__name__}: {exc}"
            records.append({
                "seed": int(item["seed"]),
                "scenario": str(item["scenario"]),
                "max_steps": int(item["max_steps"]),
                "exception": exception_text,
                "qualification_gates": {"all": False},
            })
            _write_json(output_path / "progress.json", {
                "stage": QUALIFICATION_STAGE,
                "smoke": smoke,
                "episodes_completed": index,
                "episodes_planned": len(_episode_plan(smoke=smoke)),
                "exception": exception_text,
            })
            break

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

    if smoke:
        qualification_pass = False
    else:
        requested = sum(int(record.get("requested", 0)) for record in records)
        completed = sum(int(record.get("completed", 0)) for record in records)
        complete = {
            "episodes": len(records) == len(FORMAL_SEEDS)
            and {int(record.get("seed", -1)) for record in records} == set(FORMAL_SEEDS),
            "requested": requested == EXPECTED_FORMAL_REQUESTED and completed == EXPECTED_FORMAL_REQUESTED,
            "all_episode_gates": bool(records) and len(records) == len(FORMAL_SEEDS)
            and all(bool(record.get("qualification_gates", {}).get("all")) for record in records),
        }
        complete["all"] = all(complete.values())
        qualification_pass = bool(
            complete["all"] and row_checks["all"] and sampled_checks["all"] and probability_checks["all"]
            and exception_text is None
        )

    result = {
        "stage": QUALIFICATION_STAGE,
        "status": "PASS" if qualification_pass else "FAIL",
        "evidence_type": "newly_run",
        "smoke": bool(smoke),
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "protocol": str(PROTOCOL),
        "calibration_manifest": str(calibration_manifest),
        "calibration_manifest_sha256": CALIBRATION_MANIFEST_SHA,
        "calibrated_checkpoint": str(checkpoint),
        "calibrated_checkpoint_sha256": CALIBRATED_CHECKPOINT_SHA,
        "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA,
        "calibration_status": {
            "status": calibration_manifest_value.get("status"),
            "calibration_pass": calibration_manifest_value.get("calibration_pass"),
            "STUDENT_QUALIFICATION_ALLOWED": calibration_manifest_value.get("STUDENT_QUALIFICATION_ALLOWED"),
        },
        "checkpoint_payload": {
            "stage": checkpoint_payload.get("stage"),
            "architecture_id": checkpoint_payload.get("architecture_id"),
            "temperature": checkpoint_payload.get("temperature"),
            "qualification_only": checkpoint_payload.get("qualification_only"),
            "checkpoint_promotable": checkpoint_payload.get("checkpoint_promotable"),
        },
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": CALIBRATION_TEMPERATURE,
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_scenarios": list(FORMAL_SCENARIOS),
        "smoke_seed": SMOKE_SEED,
        "records": records,
        "rows": len(rows),
        "row_checks": row_checks,
        "sampled_action_metrics": sampled,
        "sampled_action_checks": sampled_checks,
        "probability_metrics": probabilities,
        "probability_checks": probability_checks,
        "completeness": {
            "episodes": len(records),
            "expected_episodes": len(FORMAL_SEEDS),
            "requested": sum(int(record.get("requested", 0)) for record in records),
            "completed": sum(int(record.get("completed", 0)) for record in records),
            "expected_requested": EXPECTED_FORMAL_REQUESTED,
        },
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
        "environment_rollout": True,
        "teacher_called": True,
        "teacher_label_only": True,
        "student_authoritative": True,
        "formal_marker_created": bool(reservation is not None),
        "qualification_seed_selection": False,
    }
    manifest_path = output_path / "manifest.json"
    result["manifest"] = str(manifest_path)
    _write_json(manifest_path, result)
    print(json.dumps({
        "stage": QUALIFICATION_STAGE,
        "status": result["status"],
        "P2_GATE_STUDENT_QUALIFICATION": result["P2_GATE_STUDENT_QUALIFICATION"],
        "P2_EFFECTIVENESS_EVALUATION_ALLOWED": result["P2_EFFECTIVENESS_EVALUATION_ALLOWED"],
        "PPO_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    _main()
