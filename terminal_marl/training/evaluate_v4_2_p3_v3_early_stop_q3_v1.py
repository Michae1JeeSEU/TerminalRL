"""Fresh paired Q3 confirmation for the exposed P3 V3 checkpoint frontier.

The checkpoint in this protocol was selected after the diagnostic frontier had
completed on an exposed seed.  This module therefore treats that frontier as
selection evidence only and consumes only the fresh Q3 seed schedule.  All
episode-integrity gates remain hard gates; completion and wait values are
reported performance metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
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
    evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness,
)
from terminal_marl.training import evaluate_v4_2_p2_v4 as generic_evaluate
from terminal_marl.training import (
    evaluate_v4_2_p3_mixed_priority_soc_heldout_v2 as heldout_v2,
)
from terminal_marl.training import (
    diagnose_v4_2_p3_v3_checkpoint_safety_frontier_v1 as frontier_runner,
)
from terminal_marl.training.run_v4_2_p2_runtime_shortlist_entry_v1 import (
    _frozen_state_equal,
)
from terminal_marl.training.run_v4_2_p3_mixed_priority_soc_training_v3 import (
    REWARD_CONFIG,
    REWARD_SOURCE_HASHES,
    scenario_config as v3_scenario_config,
)
from terminal_marl.training.run_v4_2_p3_mixed_priority_soc_training_v3 import (
    protocol as training_protocol,
)
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


STAGE = "V4_2_P3_V3_EARLY_STOP_Q3_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/p3_v3_early_stop_q3_v1.yaml")
# Frozen after the YAML bytes were finalized.
PROTOCOL_SHA = "9dad9f8287d02ef1ac584553fab35c23089a08f2c0d940472778cb1ccb367dc8"

TRAINING_STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V3"
TRAINING_PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v3.yaml"
)
TRAINING_PROTOCOL_SHA = (
    "c69dc4b308e97652936c8da5bb6a47ef61a73a4f3cb599de8f39cb703f3c4d25"
)
TRAINING_ROOT_MANIFEST_SHA = (
    "38703fa2af89a56f2ac46551f3c6006bf750547a7768f707990c7873e8209b76"
)
TRAINING_FINAL_CHECKPOINT_SHA = (
    "0fb4acf9b05ba2fa8f5700e64eca6c2b9fc6174bb091c4e12ee4c1ea3942fb6b"
)
TRAINED_CHECKPOINT_SHA = (
    "19c31195dd881e7e4fde91530aff00375f2787754569491749474699b7f06b2d"
)
INITIAL_CHECKPOINT_SHA = (
    "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
)
FRONTIER_MANIFEST_SHA = (
    "e2be6b10f9cad30262bf68c3b9b0d57e73d9d5408f08cb7fe77f9e9f0ed40dfa"
)
FRONTIER_PROTOCOL_SHA = (
    "1bd3af56afc3b7db5199125a2ef5ca6c1205913966d19c653e4cfd901a118269"
)
EXPOSED_CONSUMPTION_MARKER_SHA = (
    "13e4811e25637273bd8f2f91defd2affc87f343cb6dc7a5d1512fc7be7762946"
)
SCHEDULER_SOURCE = Path("terminal_marl/core/merge_scheduler.py")
SCHEDULER_SOURCE_SHA = (
    "58bf3f3a29b86f00855610000a819424e5c90b0c1f1433a780bcb42f86309ce1"
)
TRAINING_RESERVATION_MARKER = "p3_mixed_priority_soc_training_v3_consumed.json"
TRAINING_RESERVATION_MARKER_SHA = (
    "225ccec9068389aeb2d8876443ed94913bc0c9a5f2e11497de789cd5b0bc4517"
)
Q3_RESERVATION_MARKER = "p3_v3_early_stop_q3_v1_consumed.json"

SEEDS = (826001, 826002, 826003, 826004, 826005, 826006)
SCENARIOS = ("flow6", "flow12", "dense24", "flow6", "flow12", "dense24")
CONTROLLERS = ("trained", "initial", "rule")
LOW_SOC_THRESHOLD = 0.65
GEOMETRY_INNER_EXCURSION_MAX_M = 0.05
RUNTIME_SHORTLIST_K = 3
EXPECTED_RECORDS = len(SEEDS) * len(CONTROLLERS)
TRAINING_ENTRY_SEEDS = [823001, 823002]
TRAINING_FORMAL_SEEDS = list(range(824001, 824013))
TRAINING_HELDOUT_SEEDS = [825001, 825002, 825003, 825004, 825005, 825006]
FORBIDDEN_SEEDS = [
    816001, 816002, 816003, 817001, 817002,
    818001, 818002, 818003, 818004, 818005, 818006, 818007, 818008,
    819001, 819002, 819003, 819004, 819005, 819006,
    820001, 820002,
    821001, 821002, 821003, 821004, 821005, 821006, 821007, 821008,
    822001, 822002, 822003, 822004, 822005, 822006,
    823001, 823002,
    824001, 824002, 824003, 824004, 824005, 824006,
    824007, 824008, 824009, 824010, 824011, 824012,
    825001, 825002, 825003, 825004, 825005, 825006,
]


class Q3InputError(ValueError):
    """Raised when a frozen Q3 input violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Q3InputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(path: str | Path) -> str:
    return _sha256(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    _require(source.is_file(), f"JSON input missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Q3InputError(f"cannot read JSON input: {source}") from exc
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _expected_reward() -> dict[str, float]:
    return {
        "low_level_priority_time_weight": 1.0,
        "low_level_soc_urgency_time_weight": 2.0,
        "low_level_soc_urgency_threshold": 0.65,
    }


def _reward_exact(value: Mapping[str, Any]) -> bool:
    for key, expected in _expected_reward().items():
        try:
            actual = float(value[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(actual) or actual != expected:
            return False
    return value.get("urgency_reward_config_exact") is True


def protocol() -> dict[str, Any]:
    """Load and validate the frozen fresh-Q3 protocol."""
    _require(PROTOCOL.is_file(), f"Q3 protocol missing: {PROTOCOL}")
    _require(PROTOCOL_SHA != "TO_BE_FROZEN", "Q3 protocol SHA is not frozen")
    _require(_sha(PROTOCOL) == PROTOCOL_SHA, "Q3 protocol changed after freeze")
    _require(_sha(TRAINING_PROTOCOL) == TRAINING_PROTOCOL_SHA,
             "V3 training protocol changed")
    _require(_sha(SCHEDULER_SOURCE) == SCHEDULER_SOURCE_SHA,
             "scheduler source changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "Q3 protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "Q3 protocol id changed")
    _require(value.get("status") == "PREREGISTERED_FRESH_Q3_CONFIRMATION",
             "Q3 protocol status changed")

    bindings = value.get("bindings")
    _require(isinstance(bindings, Mapping), "Q3 bindings are missing")
    expected_bindings = {
        "training_root_stage": TRAINING_STAGE,
        "training_root_manifest_sha256": TRAINING_ROOT_MANIFEST_SHA,
        "training_root_formal_final_checkpoint_sha256": TRAINING_FINAL_CHECKPOINT_SHA,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
        "trained_checkpoint_sha256": TRAINED_CHECKPOINT_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "frontier_manifest_sha256": FRONTIER_MANIFEST_SHA,
        "frontier_protocol_sha256": FRONTIER_PROTOCOL_SHA,
        "scheduler_source": SCHEDULER_SOURCE.as_posix(),
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
        "training_reservation_marker": TRAINING_RESERVATION_MARKER,
        "training_reservation_marker_sha256": TRAINING_RESERVATION_MARKER_SHA,
        "reward_sources": REWARD_SOURCE_HASHES,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"Q3 binding changed: {key}")

    provenance = value.get("provenance")
    _require(isinstance(provenance, Mapping), "Q3 provenance is missing")
    for key, expected in {
        "fresh_q3": True,
        "posthoc_selected_from_exposed_frontier": True,
        "exposed_seed_diagnostic_only": True,
        "exposed_seed": 825006,
        "exposed_scenario": "dense24",
        "posthoc_selected_checkpoint": "checkpoint_5",
        "frontier_failure_checkpoint": 12,
        "frontier_candidate_count": 13,
        "frontier_ckpt5_safety_zero": True,
        "exposed_consumption_marker_sha256": EXPOSED_CONSUMPTION_MARKER_SHA,
    }.items():
        _require(provenance.get(key) == expected, f"Q3 provenance changed: {key}")
    _require(provenance.get("frontier_safe_checkpoint_indices") == list(range(12)),
             "Q3 frontier safe-index evidence changed")

    reward = value.get("reward")
    _require(isinstance(reward, Mapping) and _reward_exact(reward),
             "Q3 urgency reward binding changed")
    q3 = value.get("q3")
    _require(isinstance(q3, Mapping), "Q3 schedule is missing")
    for key, expected in {
        "seeds": list(SEEDS), "scenarios": list(SCENARIOS),
        "controllers": list(CONTROLLERS), "deterministic": True,
        "no_teacher": True, "no_labels": True, "no_updates": True,
        "paired_same_seed_and_scenario": True, "full_scenario_horizon": True,
        "flow6_horizon_seconds": 1210, "flow12_horizon_seconds": 1570,
        "dense24_horizon_seconds": 1080, "expected_records": EXPECTED_RECORDS,
        "completion_report_only": True,
    }.items():
        _require(q3.get(key) == expected, f"Q3 schedule changed: {key}")

    episode_gates = value.get("episode_gates")
    _require(isinstance(episode_gates, Mapping), "Q3 episode gates are missing")
    expected_episode = {
        "safety_all_zero": True, "exact_replay": True,
        "student_requested_likelihood_exact": True,
        "runtime_shortlist_k": RUNTIME_SHORTLIST_K,
        "geometry_reverse_samples_zero": True,
        "geometry_inner_excursion_max_m": GEOMETRY_INNER_EXCURSION_MAX_M,
        "no_teacher": True, "no_labels": True, "no_updates": True,
        "actor_decisions_nonzero": True, "rule_shield_zero": True,
        "lifecycle_replay_exact": True, "frozen_proposal_subgraph": True,
        "deterministic": True, "urgency_reward_config_exact": True,
        "completion_report_only": True,
    }
    for key, expected in expected_episode.items():
        _require(episode_gates.get(key) == expected,
                 f"Q3 episode gate changed: {key}")

    decision = value.get("decision_gates")
    _require(isinstance(decision, Mapping), "Q3 decision gates are missing")
    for key, expected in {
        "all_records": True, "all_episode_integrity": True,
        "trained_sparse_completion": 1.0,
        "trained_dense_completion_aggregate_nonworse_than_initial": True,
        "trained_dense_strict_improvement": True,
        "trained_sparse_priority_weighted_wait_nonworse_than_initial": True,
        "trained_dense_priority_weighted_wait_nonworse_than_initial": True,
        "trained_overall_low_soc_mean_wait_nonworse_than_initial": True,
        "none_metrics_fail_closed": True, "rule_diagnostic_only": True,
    }.items():
        _require(decision.get(key) == expected, f"Q3 decision gate changed: {key}")

    coverage = value.get("coverage")
    _require(isinstance(coverage, Mapping), "Q3 coverage is missing")
    _require(coverage.get("low_soc_threshold") == LOW_SOC_THRESHOLD,
             "Q3 low-SOC threshold changed")
    _require(coverage.get("priority_weighted_wait_weight") == "base_priority",
             "Q3 priority-weighted wait contract changed")

    history = value.get("history")
    _require(isinstance(history, Mapping), "Q3 history contract is missing")
    _require(history.get("fresh_seed_disjoint") is True,
             "Q3 fresh-seed contract changed")
    _require(list(history.get("forbidden_seeds", [])) == FORBIDDEN_SEEDS,
             "Q3 forbidden seed schedule changed")

    reservation = value.get("reservation")
    _require(isinstance(reservation, Mapping), "Q3 reservation is missing")
    _require(reservation.get("marker") == Q3_RESERVATION_MARKER,
             "Q3 marker name changed")
    _require(reservation.get("reserve_before_first_reset") is True,
             "Q3 reservation order changed")
    _require(reservation.get("one_time") is True, "Q3 reservation is not one-time")

    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "Q3 authorization is missing")
    for key, expected in {
        "q3_confirmation_allowed_on_pass": True,
        "P3_QUALIFIED_on_pass": True,
        "checkpoint_promotable_on_pass": True,
        "P4_ALLOWED": False,
    }.items():
        _require(authorization.get(key) == expected,
                 f"Q3 authorization changed: {key}")
    return value


def validate_frontier_manifest(path: Path, p: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the diagnostic-only frontier and the selected checkpoint evidence."""
    report_path = Path(path)
    expected_sha = FRONTIER_MANIFEST_SHA
    if p is not None:
        expected_sha = str(p["bindings"]["frontier_manifest_sha256"])
    _require(report_path.is_file(), f"frontier manifest missing: {report_path}")
    _require(_sha(report_path) == expected_sha, "frontier manifest SHA mismatch")
    report = _read_json(report_path)
    expected = {
        "stage": "V4_2_P3_V3_CHECKPOINT_SAFETY_FRONTIER_V1",
        "PASS": True, "diagnostic_only": True, "diagnostic_integrity": True,
        "exposed_seed_diagnostic_only": True, "all_candidates_completed": True,
        "candidate_count": 13, "expected_candidate_count": 13,
        "earliest_failed_checkpoint": 12, "last_safe_before_first_failure": 11,
        "failure_checkpoint_indices": [12],
        "safe_checkpoint_indices": list(range(12)),
        "seed": 825006, "scenario": "dense24", "horizon_seconds": 1080.0,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "no_seed_consumption": True, "no_seed_reservation": True,
        "P3_QUALIFIED": False, "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "protocol_sha256": FRONTIER_PROTOCOL_SHA,
    }
    for key, expected_value in expected.items():
        _require(report.get(key) == expected_value,
                 f"frontier evidence changed: {key}")
    failed_evidence = report.get("failed_evidence")
    _require(isinstance(failed_evidence, Mapping),
             "frontier failure evidence is missing")
    _require(
        failed_evidence.get("heldout_consumption_marker_sha256")
        == EXPOSED_CONSUMPTION_MARKER_SHA,
        "exposed held-out consumption marker evidence changed",
    )
    hashes = report.get("checkpoint_sha256s")
    _require(isinstance(hashes, list) and len(hashes) == 12,
             "frontier checkpoint list is invalid")
    _require(hashes[4] == TRAINED_CHECKPOINT_SHA,
             "frontier checkpoint_5 SHA changed")
    records = report.get("records")
    _require(isinstance(records, list) and len(records) == 13,
             "frontier records are incomplete")
    selected = records[5]
    _require(isinstance(selected, Mapping), "frontier checkpoint_5 record is invalid")
    for key, expected_value in {
        "index": 5, "label": "checkpoint_5", "seed": 825006,
        "scenario": "dense24", "checkpoint_sha256": TRAINED_CHECKPOINT_SHA,
        "safety_all_zero": True, "completion_fraction": 1.0,
        "completion_report_only": True, "exact_replay": True,
        "frozen_proposal_subgraph": True, "deterministic": True,
        "no_teacher": True, "no_labels": True, "no_updates": True,
        "urgency_reward_config_exact": True,
    }.items():
        _require(selected.get(key) == expected_value,
                 f"frontier checkpoint_5 evidence changed: {key}")
    safety = selected.get("safety")
    _require(isinstance(safety, Mapping) and all(value == 0 for value in safety.values()),
             "frontier checkpoint_5 safety evidence is not zero")
    return report


def validate_training_root_manifest(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    bindings = p["bindings"]
    root = Path(path)
    _require(root.is_file(), f"training root manifest missing: {root}")
    _require(_sha(root) == bindings["training_root_manifest_sha256"],
             "training root manifest SHA mismatch")
    report = _read_json(root)
    expected = {
        "stage": TRAINING_STAGE, "PASS": True,
        "HELDOUT_EVALUATION_ALLOWED": True, "heldout_untouched": True,
        "P3_QUALIFIED": False, "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "formal_pass_authorizes_heldout_only": True,
        "protocol_sha256": TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "urgency_reward_config": dict(REWARD_CONFIG),
    }
    for key, expected_value in expected.items():
        _require(report.get(key) == expected_value,
                 f"training root evidence changed: {key}")
    _require(isinstance(report.get("entry"), Mapping)
             and report["entry"].get("PASS") is True,
             "training entry did not pass")
    formal = report.get("formal")
    _require(isinstance(formal, Mapping) and formal.get("PASS") is True,
             "training formal did not pass")
    _require(formal.get("final_checkpoint_sha256") ==
             bindings["training_root_formal_final_checkpoint_sha256"],
             "training formal final checkpoint changed")
    _require(isinstance(formal.get("final_checkpoint"), str)
             and bool(formal["final_checkpoint"]),
             "training formal checkpoint path missing")
    return report


def validate_training_reservation_marker(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    bindings = p["bindings"]
    marker = Path(path)
    _require(marker.is_file(), f"training reservation marker missing: {marker}")
    _require(marker.name == bindings["training_reservation_marker"],
             "training reservation marker filename changed")
    _require(_sha(marker) == bindings["training_reservation_marker_sha256"],
             "training reservation marker SHA mismatch")
    payload = _read_json(marker)
    expected = {
        "stage": TRAINING_STAGE, "protocol_sha256": TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "entry_seeds": TRAINING_ENTRY_SEEDS,
        "formal_seeds": TRAINING_FORMAL_SEEDS,
        "heldout_seeds": TRAINING_HELDOUT_SEEDS,
        "heldout_untouched": True, "reserved_before_first_reset": True,
        "urgency_reward_config": dict(REWARD_CONFIG),
    }
    for key, expected_value in expected.items():
        _require(payload.get(key) == expected_value,
                 f"training reservation evidence changed: {key}")
    return payload


def validate_trained_checkpoint_payload(
    payload: Mapping[str, Any], p: Mapping[str, Any]
) -> dict[str, Any]:
    bindings = p["bindings"]
    expected = {
        "stage": TRAINING_STAGE, "formal": True, "ppo_called": True,
        "protocol_sha256": TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "checkpoint_promotable": False,
        "urgency_reward_config": dict(REWARD_CONFIG),
    }
    for key, expected_value in expected.items():
        _require(payload.get(key) == expected_value,
                 f"selected checkpoint lineage changed: {key}")
    return dict(payload)


def load_trained_checkpoint(
    path: Path, p: Mapping[str, Any]
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"selected checkpoint missing: {checkpoint}")
    _require(_sha(checkpoint) == p["bindings"]["trained_checkpoint_sha256"],
             "selected checkpoint SHA mismatch")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
    )
    try:
        payload = load_checkpoint(
            checkpoint, policy, generic_evaluate.p2_config(6), load_optimizer=False,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise Q3InputError("selected checkpoint loading failed") from exc
    _require(isinstance(payload, Mapping), "selected checkpoint payload is invalid")
    validate_trained_checkpoint_payload(payload, p)
    _require(len(policy.optimizer.state) == 0,
             "selected checkpoint optimizer state was loaded")
    policy.model.eval()
    return policy, dict(payload)


def load_initial_checkpoint(
    path: Path, p: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"initial checkpoint missing: {checkpoint}")
    _require(_sha(checkpoint) == p["bindings"]["initial_checkpoint_sha256"],
             "initial checkpoint SHA mismatch")
    try:
        policy, payload = effectiveness.load_candidate_checkpoint(
            checkpoint, expected_sha256=p["bindings"]["initial_checkpoint_sha256"]
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise Q3InputError("initial qualified checkpoint validation failed") from exc
    _require(len(policy.optimizer.state) == 0,
             "initial checkpoint optimizer state was loaded")
    policy.model.eval()
    return policy, dict(payload)


def scenario_config(name: str, training_p: Mapping[str, Any] | None = None):
    p = training_protocol() if training_p is None else training_p
    config = v3_scenario_config(name, p)
    _require(_reward_exact({key: getattr(config, key) for key in REWARD_CONFIG}
                           | {"urgency_reward_config_exact": True}),
             "scenario urgency reward configuration changed")
    expected = {"flow6": 1210.0, "flow12": 1570.0, "dense24": 1080.0}
    _require(name in expected and float(config.max_simulation_seconds) == expected[name],
             f"{name} full horizon changed")
    return config


def evaluate_episode(**kwargs: Any) -> dict[str, Any]:
    """Reuse the validated V2 mechanics without changing completion semantics."""
    record = heldout_v2.evaluate_episode(**kwargs)
    _require(record.get("completion_report_only") is True,
             "completion was not report-only")
    return record


def aggregate_records(
    records: Sequence[Mapping[str, Any]], controller: str,
    scenario_group: str = "overall", *, low_soc_threshold: float = LOW_SOC_THRESHOLD,
) -> dict[str, Any]:
    return heldout_v2.aggregate_records(
        records, controller, scenario_group, low_soc_threshold=low_soc_threshold,
    )


def aggregate_groups(
    records: Sequence[Mapping[str, Any]], *, low_soc_threshold: float = LOW_SOC_THRESHOLD,
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


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric(
    aggregates: Mapping[str, Any], group: str, controller: str, name: str,
) -> float | None:
    group_value = aggregates.get(group)
    if not isinstance(group_value, Mapping):
        return None
    controller_value = group_value.get(controller)
    if not isinstance(controller_value, Mapping):
        return None
    return _finite(controller_value.get(name))


def _paired_records(records: Sequence[Mapping[str, Any]]) -> bool:
    expected = [
        (seed, scenario, controller)
        for seed, scenario in zip(SEEDS, SCENARIOS)
        for controller in CONTROLLERS
    ]
    actual: list[tuple[int, str, str]] = []
    try:
        for row in records:
            if not isinstance(row, Mapping) or row.get("seed") is None:
                return False
            actual.append((int(row["seed"]), str(row["scenario"]), str(row["controller"])))
    except (KeyError, TypeError, ValueError):
        return False
    return actual == expected


def decision_gates(
    records: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any], *,
    frozen_proposal_subgraph: bool,
) -> dict[str, bool]:
    """Apply the frozen paired Q3 performance gates fail-closed."""
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
    sparse_completion = _metric(aggregates, "sparse", "trained", "completion_fraction")
    trained_dense_completion = _metric(aggregates, "dense", "trained", "completion_fraction")
    initial_dense_completion = _metric(aggregates, "dense", "initial", "completion_fraction")
    trained_dense_wait = _metric(aggregates, "dense", "trained", "priority_weighted_wait_seconds")
    initial_dense_wait = _metric(aggregates, "dense", "initial", "priority_weighted_wait_seconds")
    trained_sparse_wait = _metric(aggregates, "sparse", "trained", "priority_weighted_wait_seconds")
    initial_sparse_wait = _metric(aggregates, "sparse", "initial", "priority_weighted_wait_seconds")
    trained_low_soc = _metric(aggregates, "overall", "trained", "low_soc_mean_wait_seconds")
    initial_low_soc = _metric(aggregates, "overall", "initial", "low_soc_mean_wait_seconds")
    records_integrity = (
        len(records) == EXPECTED_RECORDS and bool(records)
        and all(isinstance(row, Mapping) and row.get("episode_integrity") is True for row in records)
    )
    dense_strict = False
    if none_metrics_fail_closed:
        dense_strict = bool(
            trained_dense_completion > initial_dense_completion
            or (trained_dense_completion == initial_dense_completion
                and trained_dense_wait < initial_dense_wait)
        )
    gates = {
        "all_records": len(records) == EXPECTED_RECORDS and _paired_records(records),
        "all_episode_integrity": records_integrity,
        "frozen_proposal_subgraph": bool(frozen_proposal_subgraph),
        "none_metrics_fail_closed": none_metrics_fail_closed,
        "trained_sparse_completion": sparse_completion == 1.0 if sparse_completion is not None else False,
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
    return heldout_v2.compute_deltas(aggregates)


def validate_fresh_seed_schedule(
    identities: Mapping[str, Any], p: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject Q3 if any fresh seed or forbidden historical seed is present."""
    historical = identities.get("seeds", set())
    try:
        historical_seeds = {int(value) for value in historical}
    except (TypeError, ValueError) as exc:
        raise Q3InputError("historical seed registry is invalid") from exc
    forbidden = set(FORBIDDEN_SEEDS)
    if p is not None:
        history = p.get("history", {})
        if isinstance(history, Mapping):
            forbidden = {int(value) for value in history.get("forbidden_seeds", FORBIDDEN_SEEDS)}
    fresh = set(SEEDS)
    _require(not fresh & historical_seeds,
             "fresh Q3 seed already appears in historical registry")
    _require(not fresh & forbidden,
             "fresh Q3 seed overlaps forbidden historical schedule")
    return {
        "fresh_seeds": list(SEEDS),
        "historical_seed_count": len(historical_seeds),
        "historical_overlap": sorted(fresh & historical_seeds),
        "forbidden_overlap": sorted(fresh & forbidden),
        "fresh_seed_disjoint": True,
    }


def source_hashes() -> dict[str, str]:
    paths = (
        PROTOCOL,
        Path("terminal_marl/training/evaluate_v4_2_p3_v3_early_stop_q3_v1.py"),
        Path("terminal_marl/training/evaluate_v4_2_p3_mixed_priority_soc_heldout_v2.py"),
        Path("terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_heldout_v2.yaml"),
        Path("terminal_marl/training/evaluate_v4_2_p3_mixed_priority_soc_heldout_v1.py"),
        Path("terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_heldout_v1.yaml"),
        Path("terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v2.py"),
        Path("terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v2.yaml"),
        Path("terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v3.py"),
        TRAINING_PROTOCOL,
        Path("terminal_marl/configs/low_v4_2/p3_v3_checkpoint_safety_frontier_v1.yaml"),
        Path("terminal_marl/training/diagnose_v4_2_p3_v3_checkpoint_safety_frontier_v1.py"),
        SCHEDULER_SOURCE,
        Path("terminal_marl/core/rewards.py"),
        Path("terminal_marl/core/config.py"),
        Path("terminal_marl/envs/parallel_env.py"),
        Path("terminal_marl/training/evaluate_v4_2_p2_v4.py"),
        Path("terminal_marl/training/evaluate_v4_2_p2_gate_effectiveness_v1.py"),
        Path("terminal_marl/training/student_rollout_v4_2.py"),
        Path("terminal_marl/training/run_v4_2_p2_runtime_shortlist_entry_v1.py"),
        Path("terminal_marl/core/safety_trace.py"),
    )
    hashes = {path.as_posix(): _sha(path) for path in paths}
    for path, expected in REWARD_SOURCE_HASHES.items():
        _require(hashes.get(path) == expected, f"reward source changed: {path}")
    _require(hashes[SCHEDULER_SOURCE.as_posix()] == SCHEDULER_SOURCE_SHA,
             "scheduler source changed")
    return hashes


def reservation_payload(output: Path) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "training_root_manifest_sha256": TRAINING_ROOT_MANIFEST_SHA,
        "frontier_manifest_sha256": FRONTIER_MANIFEST_SHA,
        "trained_checkpoint_sha256": TRAINED_CHECKPOINT_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "q3_seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "controllers": list(CONTROLLERS),
        "fresh_q3": True,
        "posthoc_selected_from_exposed_frontier": True,
        "exposed_seed_diagnostic_only": True,
        "heldout_untouched": True,
        "output": str(output),
        "reserved_before_first_reset": True,
        "one_time": True,
        "urgency_reward_config": dict(REWARD_CONFIG),
    }


def _scenario_horizon_exact(config: Any, scenario: str) -> bool:
    expected = {"flow6": 1210.0, "flow12": 1570.0, "dense24": 1080.0}
    try:
        return scenario in expected and float(config.max_simulation_seconds) == expected[scenario]
    except (AttributeError, TypeError, ValueError):
        return False


def run(
    *, output: Path, training_root_manifest: Path, frontier_manifest: Path,
    trained_checkpoint: Path, initial_checkpoint: Path, history_root: Path,
) -> dict[str, Any]:
    """Run the fresh paired Q3 confirmation once."""
    p = protocol()
    bindings = p["bindings"]
    output = Path(output)
    _require(not output.exists(), f"Q3 output already exists: {output}")
    root = validate_training_root_manifest(Path(training_root_manifest), p)
    frontier = validate_frontier_manifest(Path(frontier_manifest), p)

    candidate = Path(trained_checkpoint)
    initial_path = Path(initial_checkpoint)
    training_marker = initial_path.parent / bindings["training_reservation_marker"]
    if not training_marker.is_file():
        training_marker = candidate.parent / bindings["training_reservation_marker"]
    training_marker_payload = validate_training_reservation_marker(training_marker, p)
    candidate, trained_payload = load_trained_checkpoint(candidate, p)
    initial, initial_payload = load_initial_checkpoint(initial_path, p)
    frozen_proposal_subgraph = _frozen_state_equal(
        initial.model.state_dict(), candidate.model.state_dict(),
    )
    _require(frozen_proposal_subgraph, "frozen proposal subgraph changed")

    marker = Path(trained_checkpoint).parent / Q3_RESERVATION_MARKER
    _require(not marker.exists(), f"Q3 consumption marker already exists: {marker}")
    frozen_source_hashes = source_hashes()
    # The historical helper writes only registry progress into a disposable
    # preflight directory; the requested output remains absent until the scan
    # and fresh-seed checks have passed.
    with tempfile.TemporaryDirectory(prefix="p3_v3_q3_registry_") as registry_dir:
        identities = historical_registry(Path(history_root), Path(registry_dir))
    registry_scan = validate_fresh_seed_schedule(identities, p)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "pre_reset_registry.json", {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "history_root": str(history_root),
        "registry_scan": registry_scan,
        "q3_seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "controllers": list(CONTROLLERS),
        "output_absent_before_run": True,
        "marker_absent_before_reservation": True,
        "reserved_before_first_reset": True,
    })
    reserve_once(marker, reservation_payload(output))

    scenario_protocol = training_protocol()
    records: list[dict[str, Any]] = []
    for seed, scenario in zip(SEEDS, SCENARIOS):
        config = scenario_config(str(scenario), scenario_protocol)
        _require(_scenario_horizon_exact(config, str(scenario)),
                 f"{scenario} did not use its full frozen horizon")
        for controller, policy in (
            ("trained", candidate), ("initial", initial), ("rule", None),
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
                training_p=scenario_protocol,
            )
            record["scenario"] = str(scenario)
            records.append(record)
            _write(output / "progress.json", {
                "completed_records": len(records),
                "expected_records": EXPECTED_RECORDS,
                "last_seed": int(seed),
                "last_scenario": str(scenario),
                "last_controller": controller,
                "last_episode_integrity": record.get("episode_integrity"),
            })
            if record.get("episode_integrity") is not True:
                raise RuntimeError(
                    f"Q3 episode integrity gate failed: {controller}/{scenario}/{seed}"
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
        "checkpoint_promotable": passed,
        "P4_ALLOWED": False,
        "fresh_q3": True,
        "posthoc_selected_from_exposed_frontier": True,
        "exposed_seed_diagnostic_only": True,
        "frontier_manifest_sha256": bindings["frontier_manifest_sha256"],
        "records": records,
        "aggregates": aggregates,
        "paired_deltas": compute_deltas(aggregates),
        "decision": decision,
        "registry_scan": registry_scan,
        "source_hashes": frozen_source_hashes,
        "protocol_sha256": PROTOCOL_SHA,
        "training_root_manifest_sha256": bindings["training_root_manifest_sha256"],
        "training_protocol_sha256": bindings["training_protocol_sha256"],
        "trained_checkpoint_sha256": bindings["trained_checkpoint_sha256"],
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "training_reservation_marker": str(
            training_marker
        ),
        "training_reservation_marker_sha256": bindings["training_reservation_marker_sha256"],
        "q3_consumption_marker": str(marker),
        "q3_consumption_marker_sha256": _sha(marker),
        "training_root_formal_pass": root["formal"]["PASS"],
        "training_marker_heldout_untouched": training_marker_payload["heldout_untouched"],
        "frontier_pass": frontier["PASS"],
        "trained_checkpoint_payload": {
            "stage": trained_payload.get("stage"),
            "formal": trained_payload.get("formal"),
            "ppo_called": trained_payload.get("ppo_called"),
            "protocol_sha256": trained_payload.get("protocol_sha256"),
            "initial_checkpoint_sha256": trained_payload.get("initial_checkpoint_sha256"),
            "checkpoint_promotable": trained_payload.get("checkpoint_promotable"),
            "urgency_reward_config": trained_payload.get("urgency_reward_config"),
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
        "urgency_reward_config": dict(REWARD_CONFIG),
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root-manifest", type=Path, required=True)
    parser.add_argument("--frontier-manifest", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        output=args.output,
        training_root_manifest=args.training_root_manifest,
        frontier_manifest=args.frontier_manifest,
        trained_checkpoint=args.trained_checkpoint,
        initial_checkpoint=args.initial_checkpoint,
        history_root=args.history_root,
    )
    print(json.dumps({
        "stage": result["stage"],
        "PASS": result["PASS"],
        "P3_QUALIFIED": result["P3_QUALIFIED"],
        "checkpoint_promotable": result["checkpoint_promotable"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
