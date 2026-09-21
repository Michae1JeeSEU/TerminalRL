"""Paired held-out evaluation for the P3 mixed priority/SOC PPO repair.

The runner is deliberately evaluation-only.  It validates the immutable audit,
training reservation, and checkpoint lineage before reserving the held-out
attempt, then evaluates trained, initial, and rule controllers on the same
seed/scenario pairs.  Completion is retained as a performance diagnostic; it
is never part of an episode integrity gate.
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
import yaml

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training import (
    diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as pilot_v2,
)
from terminal_marl.training import (
    evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness,
)
from terminal_marl.training import evaluate_v4_2_p2_v4 as generic_evaluate
from terminal_marl.training.evaluate_v4_2_p2_gate_runtime_shortlist_v1 import (
    _tape_shortlist_checks,
)
from terminal_marl.training.run_v4_2_p2_runtime_shortlist_entry_v1 import (
    _frozen_state_equal,
)
from terminal_marl.training.run_v4_2_p3_mixed_priority_soc_training_v2 import (
    scenario_config,
)
from terminal_marl.training.run_v4_2_p3_mixed_priority_soc_training_v2 import (
    protocol as training_protocol,
)
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_HELDOUT_V1"
PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_heldout_v1.yaml"
)
PROTOCOL_SHA = "acbc92492edf0d65ced142a5069d0b1386f742105d8ad7a616aa3390949442d3"

AUDIT_STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V2_FORMAL_AUDIT_V1"
TRAINING_STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V2"
AUDIT_PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v2_formal_audit_v1.yaml"
)
TRAINING_PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v2.yaml"
)
AUDIT_MANIFEST_SHA = (
    "09e920a3945ae3b81a82b5c8a3b9e567785a8c205b8fe7774ff68b0fff33ba3d"
)
AUDIT_PROTOCOL_SHA = (
    "fe2422c9af9ff6b3f0e234cb66987a0ff54a051c3975b60b8846e05a8f0d1ce6"
)
TRAINING_PROTOCOL_SHA = (
    "4a445d885f0e7662c9f1592f44464d970ece8322aa8a7cf81012d59e7fd916ed"
)
TRAINED_CHECKPOINT_SHA = (
    "bdff62ac94fbbd42ebb0fc571c09b6b2cd7b182058b0bf1a2ef496ab863ca9c6"
)
INITIAL_CHECKPOINT_SHA = (
    "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
)
SCHEDULER_SOURCE = Path("terminal_marl/core/merge_scheduler.py")
SCHEDULER_SOURCE_SHA = (
    "58bf3f3a29b86f00855610000a819424e5c90b0c1f1433a780bcb42f86309ce1"
)
TRAINING_RESERVATION_MARKER = "p3_mixed_priority_soc_training_v2_consumed.json"
TRAINING_RESERVATION_MARKER_SHA = (
    "800e0dba20891485ce5a221f95003b3ce61d061993a1ada34f21ce6bddbd989d"
)
HELDOUT_RESERVATION_MARKER = "p3_mixed_priority_soc_heldout_v1_consumed.json"

SEEDS = (822001, 822002, 822003, 822004, 822005, 822006)
SCENARIOS = ("flow6", "flow12", "dense24", "flow6", "flow12", "dense24")
CONTROLLERS = ("trained", "initial", "rule")
LOW_SOC_THRESHOLD = 0.65
GEOMETRY_INNER_EXCURSION_MAX_M = 0.05
RUNTIME_SHORTLIST_K = 3
EXPECTED_RECORDS = len(SEEDS) * len(CONTROLLERS)

# These fields are written by the bound V2 training runner.  They are checked
# here in addition to the marker SHA so a copied marker cannot be relabeled.
TRAINING_ENTRY_SEEDS = [820001, 820002]
TRAINING_FORMAL_SEEDS = [821001, 821002, 821003, 821004, 821005, 821006, 821007, 821008]
TRAINING_HELDOUT_SEEDS = list(SEEDS)
PILOT_MANIFEST_SHA = (
    "c094528791475d8748174c2b3c0408d22dc53d0f8e0a1b64029d54e8bb012d0e"
)
SUPERSEDED_TRAINING_MANIFEST_SHA = (
    "7b6420139c53f269aa48c25e75e6f1bd9205ea91fd38e2fdf8d0088b59619c59"
)


class HeldoutInputError(ValueError):
    """Raised when an immutable held-out input violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldoutInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(path: str | Path) -> str:
    """Hash indirection used by the fail-closed tests and audit harness."""
    return _sha256(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    _require(source.is_file(), f"JSON input missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeldoutInputError(f"cannot read JSON input: {source}") from exc
    _require(isinstance(value, dict), f"JSON object required: {source}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def protocol() -> dict[str, Any]:
    """Load the byte-frozen held-out protocol and validate all bindings."""
    _require(PROTOCOL.is_file(), f"held-out protocol missing: {PROTOCOL}")
    _require(
        PROTOCOL_SHA != "TO_BE_FROZEN" and _sha(PROTOCOL) == PROTOCOL_SHA,
        "P3 held-out protocol changed after preregistration",
    )
    _require(_sha(AUDIT_PROTOCOL) == AUDIT_PROTOCOL_SHA, "audit protocol changed")
    _require(_sha(TRAINING_PROTOCOL) == TRAINING_PROTOCOL_SHA, "training protocol changed")
    _require(_sha(SCHEDULER_SOURCE) == SCHEDULER_SOURCE_SHA, "scheduler source changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "held-out protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "held-out protocol id changed")
    _require(value.get("status") == "PREREGISTERED_HELDOUT", "held-out status changed")

    bindings = value.get("bindings")
    _require(isinstance(bindings, Mapping), "held-out bindings are missing")
    expected_bindings = {
        "audit_stage": AUDIT_STAGE,
        "audit_manifest_sha256": AUDIT_MANIFEST_SHA,
        "audit_protocol_sha256": AUDIT_PROTOCOL_SHA,
        "training_stage": TRAINING_STAGE,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
        "trained_checkpoint_sha256": TRAINED_CHECKPOINT_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "scheduler_source": SCHEDULER_SOURCE.as_posix(),
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
        "training_reservation_marker": TRAINING_RESERVATION_MARKER,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"held-out binding changed: {key}")
    _require(
        bindings.get("training_reservation_marker_sha256")
        == TRAINING_RESERVATION_MARKER_SHA,
        "training reservation marker SHA binding changed",
    )

    heldout = value.get("heldout")
    _require(isinstance(heldout, Mapping), "held-out section is missing")
    _require(list(heldout.get("seeds", [])) == list(SEEDS), "held-out seed order changed")
    _require(list(heldout.get("scenarios", [])) == list(SCENARIOS), "held-out scenario order changed")
    _require(list(heldout.get("controllers", [])) == list(CONTROLLERS), "controller order changed")
    for key in ("deterministic", "no_teacher", "no_labels", "no_updates",
                "paired_same_seed_and_scenario", "full_scenario_horizon"):
        _require(heldout.get(key) is True, f"held-out {key} contract changed")
    _require(heldout.get("dense_horizon_seconds") == 1080, "dense held-out horizon changed")
    _require(heldout.get("expected_records") == EXPECTED_RECORDS, "held-out record count changed")

    episode_gates = value.get("episode_gates")
    _require(isinstance(episode_gates, Mapping), "episode gates are missing")
    expected_episode_gates = {
        "safety_all_zero": True,
        "exact_replay": True,
        "student_requested_likelihood_exact": True,
        "runtime_shortlist_k": RUNTIME_SHORTLIST_K,
        "geometry_reverse_samples_zero": True,
        "geometry_inner_excursion_max_m": GEOMETRY_INNER_EXCURSION_MAX_M,
        "no_teacher": True,
        "no_labels": True,
        "no_updates": True,
        "actor_decisions_nonzero": True,
        "rule_shield_zero": True,
        "lifecycle_replay_exact": True,
        "frozen_proposal_subgraph": True,
        "completion_report_only": True,
    }
    for key, expected in expected_episode_gates.items():
        _require(episode_gates.get(key) == expected, f"episode gate changed: {key}")

    decision = value.get("decision_gates")
    _require(isinstance(decision, Mapping), "decision gates are missing")
    for key in (
        "all_records", "all_episode_integrity", "trained_dense_completion_aggregate_nonworse_than_initial",
        "trained_dense_strict_improvement", "trained_sparse_priority_weighted_wait_nonworse_than_initial",
        "trained_dense_priority_weighted_wait_nonworse_than_initial",
        "trained_overall_low_soc_mean_wait_nonworse_than_initial", "none_metrics_fail_closed",
        "rule_diagnostic_only",
    ):
        _require(decision.get(key) is True, f"decision gate changed: {key}")
    _require(decision.get("trained_sparse_completion") == 1.0, "sparse completion gate changed")

    coverage = value.get("coverage")
    _require(isinstance(coverage, Mapping), "coverage section is missing")
    _require(coverage.get("low_soc_threshold") == LOW_SOC_THRESHOLD, "low-SOC threshold changed")
    _require(coverage.get("priority_weighted_wait_weight") == "base_priority",
             "priority-weighted wait contract changed")
    reservation = value.get("reservation")
    _require(isinstance(reservation, Mapping), "held-out reservation section is missing")
    _require(reservation.get("marker") == HELDOUT_RESERVATION_MARKER,
             "held-out reservation marker changed")
    _require(reservation.get("reserve_before_first_reset") is True, "reservation order changed")
    _require(reservation.get("one_time") is True, "reservation one-time contract changed")
    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "authorization section is missing")
    _require(authorization.get("heldout_evaluation_allowed_on_pass") is True,
             "held-out authorization changed")
    _require(authorization.get("heldout_untouched") is True, "held-out immutability changed")
    _require(authorization.get("P3_QUALIFIED_on_pass") is True, "P3 qualification authorization changed")
    _require(authorization.get("checkpoint_promotable_on_pass") is True,
             "checkpoint promotion authorization changed")
    _require(authorization.get("P4_ALLOWED") is False, "P4 authorization changed")
    return value


def source_hashes() -> dict[str, str]:
    """Hash the relative source snapshot that executes held-out evaluation."""
    paths = (
        Path("terminal_marl/training/evaluate_v4_2_p3_mixed_priority_soc_heldout_v1.py"),
        PROTOCOL,
        AUDIT_PROTOCOL,
        TRAINING_PROTOCOL,
        SCHEDULER_SOURCE,
        Path("terminal_marl/training/evaluate_v4_2_p2_v4.py"),
        Path("terminal_marl/training/diagnose_v4_2_p3_mixed_priority_soc_pilot_v2.py"),
        Path("terminal_marl/training/evaluate_v4_2_p2_gate_runtime_shortlist_v1.py"),
        Path("terminal_marl/training/evaluate_v4_2_p2_gate_effectiveness_v1.py"),
        Path("terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v2.py"),
        Path("terminal_marl/training/student_rollout_v4_2.py"),
        Path("terminal_marl/training/run_v4_2_p2_runtime_shortlist_entry_v1.py"),
        Path("terminal_marl/envs/parallel_env.py"),
        Path("terminal_marl/core/safety_trace.py"),
    )
    return {path.as_posix(): _sha(path) for path in paths}


def validate_audit_manifest(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact offline audit authorization and final checkpoint."""
    bindings = p["bindings"]
    _require(path.is_file(), f"audit manifest missing: {path}")
    _require(_sha(path) == bindings["audit_manifest_sha256"], "audit manifest SHA mismatch")
    report = _read_json(path)
    expected = {
        "stage": bindings["audit_stage"],
        "PASS": True,
        "HELDOUT_EVALUATION_ALLOWED": True,
        "heldout_untouched": True,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "protocol_sha256": bindings["audit_protocol_sha256"],
    }
    for key, expected_value in expected.items():
        _require(report.get(key) == expected_value, f"audit manifest binding changed: {key}")
    coverage = report.get("corrected_coverage")
    _require(isinstance(coverage, Mapping), "audit corrected coverage is missing")
    for key in ("priority_coverage", "soc_in_bounds", "soc_range_observed"):
        _require(coverage.get(key) is True, f"audit corrected coverage gate failed: {key}")
    final = report.get("final_checkpoint")
    _require(isinstance(final, Mapping), "audit final checkpoint evidence is missing")
    _require(final.get("sha256") == bindings["trained_checkpoint_sha256"],
             "audit final checkpoint SHA binding changed")
    _require(isinstance(final.get("path"), str) and bool(final["path"]),
             "audit final checkpoint path is missing")
    _require(report.get("training_protocol_sha256") == TRAINING_PROTOCOL_SHA,
             "audit training protocol binding changed")
    _require(report.get("scheduler_source") == SCHEDULER_SOURCE.as_posix(),
             "audit scheduler source binding changed")
    _require(report.get("scheduler_source_sha256") == SCHEDULER_SOURCE_SHA,
             "audit scheduler SHA binding changed")
    return report


def validate_training_reservation_marker(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable V2 reservation that kept held-out seeds untouched."""
    bindings = p["bindings"]
    marker = Path(path)
    _require(marker.is_file(), f"training reservation marker missing: {marker}")
    _require(marker.name == bindings["training_reservation_marker"],
             "training reservation marker filename changed")
    _require(_sha(marker) == bindings["training_reservation_marker_sha256"],
             "training reservation marker SHA mismatch")
    payload = _read_json(marker)
    expected = {
        "stage": bindings["training_stage"],
        "protocol_sha256": bindings["training_protocol_sha256"],
        "pilot_manifest_sha256": PILOT_MANIFEST_SHA,
        "superseded_training_manifest_sha256": SUPERSEDED_TRAINING_MANIFEST_SHA,
        "candidate_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "entry_seeds": TRAINING_ENTRY_SEEDS,
        "formal_seeds": TRAINING_FORMAL_SEEDS,
        "heldout_seeds": TRAINING_HELDOUT_SEEDS,
        "heldout_untouched": True,
        "reserved_before_first_reset": True,
    }
    for key, expected_value in expected.items():
        _require(payload.get(key) == expected_value,
                 f"training reservation marker field changed: {key}")
    return payload


validate_formal = validate_audit_manifest
validate_training_marker = validate_training_reservation_marker


def validate_trained_checkpoint_payload(
    payload: Mapping[str, Any], p: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate lineage fields after the checkpoint schema loader returns."""
    bindings = p["bindings"]
    expected = {
        "stage": bindings["training_stage"],
        "formal": True,
        "ppo_called": True,
        "protocol_sha256": bindings["training_protocol_sha256"],
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "checkpoint_promotable": False,
    }
    for key, expected_value in expected.items():
        _require(payload.get(key) == expected_value,
                 f"trained checkpoint payload binding changed: {key}")
    return dict(payload)


def load_trained_checkpoint(
    path: Path, p: Mapping[str, Any]
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    bindings = p["bindings"]
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"trained checkpoint missing: {checkpoint}")
    _require(_sha(checkpoint) == bindings["trained_checkpoint_sha256"],
             "trained checkpoint SHA mismatch")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
    )
    payload = load_checkpoint(checkpoint, policy, generic_evaluate.p2_config(6), load_optimizer=False)
    validate_trained_checkpoint_payload(payload, p)
    _require(len(policy.optimizer.state) == 0, "trained checkpoint optimizer state was loaded")
    policy.model.eval()
    return policy, payload


# Compatibility aliases make the fail-closed loader easy to exercise from
# review harnesses that use the shorter names from earlier held-out runners.
_load_trained = load_trained_checkpoint


def load_initial_checkpoint(
    path: Path, p: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    bindings = p["bindings"]
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"initial checkpoint missing: {checkpoint}")
    _require(_sha(checkpoint) == bindings["initial_checkpoint_sha256"],
             "initial checkpoint SHA mismatch")
    try:
        policy, payload = effectiveness.load_candidate_checkpoint(
            checkpoint, expected_sha256=bindings["initial_checkpoint_sha256"]
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise HeldoutInputError("initial qualified checkpoint validation failed") from exc
    _require(len(policy.optimizer.state) == 0, "initial checkpoint optimizer state was loaded")
    policy.model.eval()
    return policy, payload


_load_initial = load_initial_checkpoint


def _action_tape_path(result: Mapping[str, Any]) -> str:
    summary = result.get("summary")
    candidate = summary.get("action_tape") if isinstance(summary, Mapping) else None
    if not isinstance(candidate, str) or not candidate:
        candidate = result.get("action_tape")
    if not isinstance(candidate, str) or not candidate:
        raise HeldoutInputError("evaluation result has no action tape")
    return candidate


def _zero_safety(result: Mapping[str, Any]) -> bool:
    safety = result.get("safety")
    return isinstance(safety, Mapping) and not any(bool(value) for value in safety.values())


def _geometry_gate(result: Mapping[str, Any]) -> bool:
    geometry = result.get("geometry")
    if not isinstance(geometry, Mapping):
        return False
    try:
        return (
            int(geometry.get("samples", 0)) > 0
            and int(geometry.get("reverse_samples", -1)) == 0
            and float(geometry.get("max_inner_excursion_m", math.inf))
            <= GEOMETRY_INNER_EXCURSION_MAX_M
        )
    except (TypeError, ValueError):
        return False


def _student_shortlist_checks(result: Mapping[str, Any]) -> dict[str, Any]:
    return _tape_shortlist_checks(_action_tape_path(result), expected_k=RUNTIME_SHORTLIST_K)


def _rule_shortlist_checks() -> dict[str, Any]:
    return {
        "shortlists_supplied": 0,
        "fallback_selected": 0,
        "exhausted": 0,
        "max_candidates_checked": 0,
        "requested_is_first": True,
        "unique": True,
        "bounded": True,
        "teacher_free": True,
        "all": True,
    }


def _episode_gates(
    result: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    shortlist: Mapping[str, Any],
    *,
    controller: str,
    frozen_proposal_subgraph: bool,
) -> dict[str, bool]:
    is_student = controller in {"trained", "initial"}
    student_likelihood = bool(result.get("student_likelihood_exact")) if is_student else True
    actor_decisions = result.get("actor_decisions")
    actor_gate = isinstance(actor_decisions, (int, float)) and not isinstance(actor_decisions, bool) and actor_decisions > 0
    no_teacher = result.get("teacher_enabled", False) is False
    no_labels = result.get("label_rows", 0) == 0
    no_updates = result.get("training_called", False) is False
    replay = result.get("replay_exact") is True and lifecycle.get("replay_exact") is True
    gates = {
        "safety_all_zero": _zero_safety(result),
        "exact_replay": bool(replay),
        "student_requested_likelihood_exact": student_likelihood,
        "runtime_shortlist_k3": bool(shortlist.get("all")),
        "geometry": _geometry_gate(result),
        "no_teacher": no_teacher,
        "no_labels": no_labels,
        "no_updates": no_updates,
        "actor_decisions_nonzero": bool(actor_gate if is_student else True),
        "rule_shield_zero": bool(result.get("shield", 0) == 0 if controller == "rule" else True),
        "lifecycle_replay_exact": lifecycle.get("replay_exact") is True,
        "frozen_proposal_subgraph": bool(frozen_proposal_subgraph),
        "deterministic": result.get("deterministic") is True,
    }
    # Explicitly keep completion out of the integrity gates.
    return gates


def _completion_fraction(result: Mapping[str, Any]) -> float | None:
    try:
        requests = int(result.get("requests"))
        completed = int(result.get("completed"))
    except (TypeError, ValueError):
        return None
    if requests <= 0:
        return None
    return float(completed / requests)


def evaluate_episode(
    *,
    controller: str,
    policy: Any,
    config: Any,
    seed: int,
    output: Path,
    frozen_proposal_subgraph: bool,
    low_soc_threshold: float = LOW_SOC_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate one controller and fail closed on integrity-gate failure."""
    if controller == "rule":
        result = generic_evaluate.evaluate_rule(config, seed, output, smoke=False)
        shortlist = _rule_shortlist_checks()
    elif controller in {"trained", "initial"}:
        result = generic_evaluate.evaluate(policy, config, seed, output, smoke=False)
        shortlist = _student_shortlist_checks(result)
    else:
        raise HeldoutInputError(f"unknown held-out controller: {controller}")
    tape_path = _action_tape_path(result)
    lifecycle = pilot_v2.lifecycle_metrics(
        config, tape_path, low_soc_threshold=low_soc_threshold,
    )
    gates = _episode_gates(
        result,
        lifecycle,
        shortlist,
        controller=controller,
        frozen_proposal_subgraph=frozen_proposal_subgraph,
    )
    enriched = dict(result)
    enriched["runtime_shortlist_checks"] = dict(shortlist)
    enriched["gates"] = gates
    enriched["completion_fraction"] = _completion_fraction(result)
    enriched["completion_report_only"] = True
    enriched["episode_integrity"] = bool(gates) and all(gates.values())
    _write(output / "evaluation.json", enriched)
    return {
        "controller": controller,
        "seed": int(seed),
        "result": enriched,
        "runtime_shortlist_checks": dict(shortlist),
        "lifecycle": lifecycle,
        "completion_fraction": enriched["completion_fraction"],
        "completion_report_only": True,
        "episode_integrity": enriched["episode_integrity"],
        "frozen_proposal_subgraph": bool(frozen_proposal_subgraph),
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _aggregate_controller(
    records: Sequence[Mapping[str, Any]],
    controller: str,
    scenario_group: str,
    *,
    low_soc_threshold: float = LOW_SOC_THRESHOLD,
) -> dict[str, Any]:
    scenarios = {"sparse": {"flow6", "flow12"}, "dense": {"dense24"}, "overall": set(SCENARIOS)}
    allowed = scenarios.get(scenario_group)
    if allowed is None:
        raise ValueError(f"unknown scenario group: {scenario_group}")
    selected = [
        row for row in records
        if row.get("controller") == controller and str(row.get("scenario", "")) in allowed
    ]
    requests_values: list[int] = []
    completed_values: list[int] = []
    lifecycle_rows: list[Mapping[str, Any]] = []
    integrity = bool(selected)
    for row in selected:
        result = row.get("result")
        life = row.get("lifecycle")
        integrity = integrity and row.get("episode_integrity") is True
        if not isinstance(result, Mapping) or not isinstance(life, Mapping):
            integrity = False
            continue
        try:
            requests = int(result.get("requests"))
            completed = int(result.get("completed"))
        except (TypeError, ValueError):
            integrity = False
            continue
        if requests < 0 or completed < 0 or completed > requests:
            integrity = False
            continue
        requests_values.append(requests)
        completed_values.append(completed)
        rows = life.get("rows")
        if not isinstance(rows, list):
            integrity = False
            continue
        if not all(isinstance(item, Mapping) for item in rows):
            integrity = False
        lifecycle_rows.extend(item for item in rows if isinstance(item, Mapping))

    expected_row_count = 0
    for row in selected:
        life = row.get("lifecycle")
        if isinstance(life, Mapping) and isinstance(life.get("rows"), list):
            expected_row_count += len(life["rows"])
    if len(lifecycle_rows) != expected_row_count:
        integrity = False

    requests_total = sum(requests_values) if len(requests_values) == len(selected) else None
    completed_total = sum(completed_values) if len(completed_values) == len(selected) else None
    completion = (
        None if requests_total is None or requests_total <= 0
        else float(completed_total / requests_total)
    )
    weighted_numer = 0.0
    weighted_denom = 0.0
    low_soc_waits: list[float] = []
    rows_valid = True
    for index, item in enumerate(lifecycle_rows):
        if not isinstance(item, Mapping):
            rows_valid = False
            continue
        priority = _finite(item.get("base_priority"))
        wait = _finite(item.get("wait_seconds"))
        initial_soc = _finite(item.get("initial_soc"))
        if priority is None or wait is None or initial_soc is None or priority <= 0 or wait < 0:
            rows_valid = False
            continue
        weighted_numer += priority * wait
        weighted_denom += priority
        if initial_soc <= low_soc_threshold:
            low_soc_waits.append(wait)
    if not rows_valid or not lifecycle_rows or weighted_denom <= 0:
        priority_wait = None
    else:
        priority_wait = float(weighted_numer / weighted_denom)
    low_soc_mean = None if not low_soc_waits else float(np.mean(low_soc_waits))
    return {
        "controller": controller,
        "scenario_group": scenario_group,
        "records": len(selected),
        "requests": requests_total,
        "completed": completed_total,
        "completion_fraction": completion,
        "priority_weighted_wait_seconds": priority_wait,
        "low_soc_mean_wait_seconds": low_soc_mean,
        "low_soc_threshold": float(low_soc_threshold),
        "lifecycle_rows": len(lifecycle_rows),
        "episode_integrity": bool(integrity),
        "metrics_complete": all(
            value is not None and _finite(value) is not None
            for value in (completion, priority_wait, low_soc_mean)
        ),
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    controller: str,
    scenario_group: str = "overall",
    *,
    low_soc_threshold: float = LOW_SOC_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate completion, priority-weighted wait, and low-SOC wait."""
    return _aggregate_controller(
        records, controller, scenario_group, low_soc_threshold=low_soc_threshold,
    )


def aggregate_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    low_soc_threshold: float = LOW_SOC_THRESHOLD,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        group: {
            controller: aggregate_records(
                records, controller, group, low_soc_threshold=low_soc_threshold,
            )
            for controller in CONTROLLERS
        }
        for group in ("sparse", "dense", "overall")
    }


def _metric(aggregates: Mapping[str, Any], group: str, controller: str, name: str) -> float | None:
    group_value = aggregates.get(group)
    if not isinstance(group_value, Mapping):
        return None
    controller_value = group_value.get(controller)
    if not isinstance(controller_value, Mapping):
        return None
    value = controller_value.get(name)
    return _finite(value)


def _paired_records(records: Sequence[Mapping[str, Any]]) -> bool:
    expected = [
        (seed, scenario, controller)
        for seed, scenario in zip(SEEDS, SCENARIOS)
        for controller in CONTROLLERS
    ]
    actual = []
    try:
        for row in records:
            if not isinstance(row, Mapping) or row.get("seed") is None:
                return False
            actual.append((int(row["seed"]), str(row["scenario"]), str(row["controller"])))
    except (KeyError, TypeError, ValueError):
        return False
    return actual == expected


def decision_gates(
    records: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    *,
    frozen_proposal_subgraph: bool,
) -> dict[str, bool]:
    """Evaluate only preregistered integrity/performance gates."""
    metric_values = [
        _metric(aggregates, "sparse", "trained", "completion_fraction"),
        _metric(aggregates, "dense", "trained", "completion_fraction"),
        _metric(aggregates, "dense", "initial", "completion_fraction"),
        _metric(aggregates, "sparse", "trained", "priority_weighted_wait_seconds"),
        _metric(aggregates, "sparse", "initial", "priority_weighted_wait_seconds"),
        _metric(aggregates, "dense", "trained", "priority_weighted_wait_seconds"),
        _metric(aggregates, "dense", "initial", "priority_weighted_wait_seconds"),
        _metric(aggregates, "overall", "trained", "low_soc_mean_wait_seconds"),
        _metric(aggregates, "overall", "initial", "low_soc_mean_wait_seconds"),
    ]
    none_metrics_fail_closed = all(value is not None for value in metric_values)
    trained_sparse_completion = _metric(aggregates, "sparse", "trained", "completion_fraction")
    trained_dense_completion = _metric(aggregates, "dense", "trained", "completion_fraction")
    initial_dense_completion = _metric(aggregates, "dense", "initial", "completion_fraction")
    trained_dense_wait = _metric(aggregates, "dense", "trained", "priority_weighted_wait_seconds")
    initial_dense_wait = _metric(aggregates, "dense", "initial", "priority_weighted_wait_seconds")
    trained_sparse_wait = _metric(aggregates, "sparse", "trained", "priority_weighted_wait_seconds")
    initial_sparse_wait = _metric(aggregates, "sparse", "initial", "priority_weighted_wait_seconds")
    trained_low_soc = _metric(aggregates, "overall", "trained", "low_soc_mean_wait_seconds")
    initial_low_soc = _metric(aggregates, "overall", "initial", "low_soc_mean_wait_seconds")
    records_integrity = (
        len(records) == EXPECTED_RECORDS
        and bool(records)
        and all(
            isinstance(row, Mapping) and row.get("episode_integrity") is True
            for row in records
        )
    )
    dense_strict = False
    if none_metrics_fail_closed:
        dense_strict = bool(
            trained_dense_completion > initial_dense_completion
            or (
                trained_dense_completion == initial_dense_completion
                and trained_dense_wait < initial_dense_wait
            )
        )
    gates = {
        "all_records": len(records) == EXPECTED_RECORDS and _paired_records(records),
        "all_episode_integrity": records_integrity,
        "frozen_proposal_subgraph": bool(frozen_proposal_subgraph),
        "none_metrics_fail_closed": none_metrics_fail_closed,
        "trained_sparse_completion": trained_sparse_completion == 1.0
        if trained_sparse_completion is not None else False,
        "trained_dense_completion_aggregate_nonworse_than_initial": (
            trained_dense_completion >= initial_dense_completion
            if trained_dense_completion is not None and initial_dense_completion is not None else False
        ),
        "trained_dense_strict_improvement": dense_strict,
        "trained_sparse_priority_weighted_wait_nonworse_than_initial": (
            trained_sparse_wait <= initial_sparse_wait
            if trained_sparse_wait is not None and initial_sparse_wait is not None else False
        ),
        "trained_dense_priority_weighted_wait_nonworse_than_initial": (
            trained_dense_wait <= initial_dense_wait
            if trained_dense_wait is not None and initial_dense_wait is not None else False
        ),
        "trained_overall_low_soc_mean_wait_nonworse_than_initial": (
            trained_low_soc <= initial_low_soc
            if trained_low_soc is not None and initial_low_soc is not None else False
        ),
        "rule_diagnostic_only": True,
    }
    gates["all"] = all(gates.values())
    return gates


def compute_deltas(aggregates: Mapping[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for group in ("sparse", "dense", "overall"):
        deltas[group] = {}
        for baseline in ("initial", "rule"):
            deltas[group][f"trained_minus_{baseline}"] = {
                metric: (
                    None
                    if _metric(aggregates, group, "trained", metric) is None
                    or _metric(aggregates, group, baseline, metric) is None
                    else _metric(aggregates, group, "trained", metric)
                    - _metric(aggregates, group, baseline, metric)
                )
                for metric in (
                    "completion_fraction",
                    "priority_weighted_wait_seconds",
                    "low_soc_mean_wait_seconds",
                )
            }
    return deltas


def run(
    *,
    output: Path,
    audit_manifest: Path,
    trained_checkpoint: Path,
    initial_checkpoint: Path,
    training_reservation_marker: Path | None = None,
) -> dict[str, Any]:
    """Run the once-only deterministic held-out evaluation."""
    p = protocol()
    bindings, heldout = p["bindings"], p["heldout"]
    output = Path(output)
    _require(not output.exists(), f"output already exists: {output}")
    audit_report = validate_audit_manifest(Path(audit_manifest), p)
    if training_reservation_marker is not None:
        marker = Path(training_reservation_marker)
    else:
        # The V2 training runner places its reservation beside the original
        # checkpoint it received, while the final checkpoint lives in the
        # formal output directory.  Prefer the original checkpoint directory;
        # retain the formal directory as a compatibility fallback for synced
        # layouts that co-locate both artifacts.
        marker = Path(initial_checkpoint).parent / bindings["training_reservation_marker"]
        if not marker.is_file():
            colocated = Path(trained_checkpoint).parent / bindings["training_reservation_marker"]
            if colocated.is_file():
                marker = colocated
    training_marker = validate_training_reservation_marker(marker, p)
    trained, trained_payload = load_trained_checkpoint(Path(trained_checkpoint), p)
    initial, initial_payload = load_initial_checkpoint(Path(initial_checkpoint), p)
    frozen_proposal_subgraph = _frozen_state_equal(
        initial.model.state_dict(), trained.model.state_dict(),
    )

    output.mkdir(parents=True, exist_ok=False)
    heldout_marker = Path(trained_checkpoint).parent / str(
        p["reservation"]["marker"]
    )
    _require(not heldout_marker.exists(), f"held-out consumption marker already exists: {heldout_marker}")
    reserve_once(heldout_marker, {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "audit_manifest_sha256": bindings["audit_manifest_sha256"],
        "trained_checkpoint_sha256": bindings["trained_checkpoint_sha256"],
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "heldout_seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "controllers": list(CONTROLLERS),
        "heldout_untouched": True,
        "output": str(output),
        "reserved_before_first_reset": True,
        "one_time": True,
    })
    _write(output / "pre_reset_registry.json", {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "heldout_seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "controllers": list(CONTROLLERS),
        "reservation_marker": str(heldout_marker),
        "reserved_before_first_reset": True,
    })

    scenario_protocol = training_protocol()
    records: list[dict[str, Any]] = []
    for seed, scenario in zip(SEEDS, SCENARIOS):
        config = scenario_config(str(scenario), scenario_protocol)
        for controller, policy in (
            ("trained", trained), ("initial", initial), ("rule", None),
        ):
            folder = output / f"{controller}_{scenario}_{seed}"
            record = evaluate_episode(
                controller=controller,
                policy=policy,
                config=config,
                seed=int(seed),
                output=folder,
                frozen_proposal_subgraph=frozen_proposal_subgraph,
                low_soc_threshold=float(p["coverage"]["low_soc_threshold"]),
            )
            record["scenario"] = str(scenario)
            records.append(record)
            _write(output / "progress.json", {
                "completed_records": len(records),
                "expected_records": EXPECTED_RECORDS,
                "last_seed": int(seed),
                "last_scenario": str(scenario),
                "last_controller": controller,
                "last_episode_integrity": record["episode_integrity"],
            })
            if not record["episode_integrity"]:
                raise RuntimeError(
                    f"held-out episode integrity gate failed: {controller}/{scenario}/{seed}"
                )

    aggregates = aggregate_groups(
        records, low_soc_threshold=float(p["coverage"]["low_soc_threshold"]),
    )
    decision = decision_gates(
        records, aggregates, frozen_proposal_subgraph=frozen_proposal_subgraph,
    )
    passed = bool(decision["all"])
    result = {
        "stage": STAGE,
        "PASS": passed,
        "P3_QUALIFIED": passed,
        "HELDOUT_EVALUATION_ALLOWED": True,
        "P4_ALLOWED": False,
        "checkpoint_promotable": passed,
        "records": records,
        "aggregates": aggregates,
        "paired_deltas": compute_deltas(aggregates),
        "decision": decision,
        "source_hashes": source_hashes(),
        "protocol_sha256": PROTOCOL_SHA,
        "audit_manifest_sha256": bindings["audit_manifest_sha256"],
        "audit_protocol_sha256": bindings["audit_protocol_sha256"],
        "training_protocol_sha256": bindings["training_protocol_sha256"],
        "trained_checkpoint_sha256": bindings["trained_checkpoint_sha256"],
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "scheduler_source": bindings["scheduler_source"],
        "scheduler_source_sha256": bindings["scheduler_source_sha256"],
        "training_reservation_marker": str(marker),
        "training_reservation_marker_sha256": bindings["training_reservation_marker_sha256"],
        "heldout_reservation_marker": str(heldout_marker),
        "heldout_reservation_marker_sha256": _sha(heldout_marker),
        "heldout_untouched_before_run": training_marker["heldout_untouched"],
        "trained_checkpoint_payload": {
            "stage": trained_payload.get("stage"),
            "formal": trained_payload.get("formal"),
            "ppo_called": trained_payload.get("ppo_called"),
            "checkpoint_promotable": trained_payload.get("checkpoint_promotable"),
        },
        "initial_checkpoint_payload": {
            "stage": initial_payload.get("stage"),
            "qualification_pass": initial_payload.get("qualification_pass"),
        },
        "frozen_proposal_subgraph": bool(frozen_proposal_subgraph),
        "deterministic": True,
        "no_teacher": True,
        "no_labels": True,
        "no_updates": True,
        "completion_report_only": True,
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-manifest", "--formal-manifest", dest="audit_manifest", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--training-reservation-marker", type=Path)
    parser.add_argument("--history-root", type=Path)
    args = parser.parse_args()
    result = run(
        output=args.output,
        audit_manifest=args.audit_manifest,
        trained_checkpoint=args.trained_checkpoint,
        initial_checkpoint=args.initial_checkpoint,
        training_reservation_marker=args.training_reservation_marker,
    )
    print(json.dumps({
        "stage": result["stage"],
        "PASS": result["PASS"],
        "P3_QUALIFIED": result["P3_QUALIFIED"],
        "checkpoint_promotable": result["checkpoint_promotable"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
