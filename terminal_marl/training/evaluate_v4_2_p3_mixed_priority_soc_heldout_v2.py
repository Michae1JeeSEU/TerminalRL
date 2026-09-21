"""Paired held-out evaluation for the P3 V3 priority/SOC urgency repair.

This module is evaluation-only.  It validates the immutable V3 training root,
reservation marker, checkpoint lineage, and frozen proposal subgraph before
running the six deterministic seed/scenario pairs for trained, initial, and
rule controllers.  Completion and wait metrics are performance evidence; they
are never substituted for episode-integrity gates.
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
from terminal_marl.training import (
    evaluate_v4_2_p3_mixed_priority_soc_heldout_v1 as heldout_v1,
)
from terminal_marl.training.evaluate_v4_2_p2_gate_runtime_shortlist_v1 import (
    _tape_shortlist_checks,
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
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_HELDOUT_V2"
PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_heldout_v2.yaml"
)
# Filled from the final YAML bytes before any held-out reset.
PROTOCOL_SHA = "bdcd4be571555b0f512af30e8bf548cc11948b3347e18b552093ae1997e42e7a"

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
TRAINED_CHECKPOINT_SHA = (
    "0fb4acf9b05ba2fa8f5700e64eca6c2b9fc6174bb091c4e12ee4c1ea3942fb6b"
)
INITIAL_CHECKPOINT_SHA = (
    "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
)
SCHEDULER_SOURCE = Path("terminal_marl/core/merge_scheduler.py")
SCHEDULER_SOURCE_SHA = (
    "58bf3f3a29b86f00855610000a819424e5c90b0c1f1433a780bcb42f86309ce1"
)
TRAINING_RESERVATION_MARKER = "p3_mixed_priority_soc_training_v3_consumed.json"
TRAINING_RESERVATION_MARKER_SHA = (
    "225ccec9068389aeb2d8876443ed94913bc0c9a5f2e11497de789cd5b0bc4517"
)
HELDOUT_RESERVATION_MARKER = "p3_mixed_priority_soc_heldout_v2_consumed.json"

SEEDS = (825001, 825002, 825003, 825004, 825005, 825006)
SCENARIOS = ("flow6", "flow12", "dense24", "flow6", "flow12", "dense24")
CONTROLLERS = ("trained", "initial", "rule")
LOW_SOC_THRESHOLD = 0.65
GEOMETRY_INNER_EXCURSION_MAX_M = 0.05
RUNTIME_SHORTLIST_K = 3
EXPECTED_RECORDS = len(SEEDS) * len(CONTROLLERS)
TRAINING_ENTRY_SEEDS = [823001, 823002]
TRAINING_FORMAL_SEEDS = list(range(824001, 824013))
TRAINING_HELDOUT_SEEDS = list(SEEDS)


class HeldoutV2InputError(ValueError):
    """Raised when a frozen V3 held-out input violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldoutV2InputError(message)


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
        raise HeldoutV2InputError(f"cannot read JSON input: {source}") from exc
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


def protocol() -> dict[str, Any]:
    """Load the byte-frozen V2 held-out protocol and validate all bindings."""
    _require(PROTOCOL.is_file(), f"held-out protocol missing: {PROTOCOL}")
    _require(
        PROTOCOL_SHA != "TO_BE_FROZEN" and _sha(PROTOCOL) == PROTOCOL_SHA,
        "P3 held-out V2 protocol changed after preregistration",
    )
    _require(
        _sha(TRAINING_PROTOCOL) == TRAINING_PROTOCOL_SHA,
        "V3 training protocol changed",
    )
    _require(_sha(SCHEDULER_SOURCE) == SCHEDULER_SOURCE_SHA, "scheduler source changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "held-out protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "held-out protocol id changed")
    _require(value.get("status") == "PREREGISTERED_HELDOUT", "held-out status changed")

    bindings = value.get("bindings")
    _require(isinstance(bindings, Mapping), "held-out bindings are missing")
    expected_bindings = {
        "training_root_stage": TRAINING_STAGE,
        "training_root_manifest_sha256": TRAINING_ROOT_MANIFEST_SHA,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
        "trained_checkpoint_sha256": TRAINED_CHECKPOINT_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "scheduler_source": SCHEDULER_SOURCE.as_posix(),
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
        "training_reservation_marker": TRAINING_RESERVATION_MARKER,
        "training_reservation_marker_sha256": TRAINING_RESERVATION_MARKER_SHA,
        "reward_sources": REWARD_SOURCE_HASHES,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"held-out binding changed: {key}")

    reward = value.get("reward")
    _require(isinstance(reward, Mapping), "held-out reward section is missing")
    for key, expected in REWARD_CONFIG.items():
        try:
            actual = float(reward.get(key))
        except (TypeError, ValueError):
            raise HeldoutV2InputError(f"reward binding is invalid: {key}") from None
        _require(math.isfinite(actual) and actual == float(expected),
                 f"reward binding changed: {key}")
    _require(reward.get("urgency_reward_config_exact") is True,
             "urgency reward exact binding changed")

    heldout = value.get("heldout")
    _require(isinstance(heldout, Mapping), "held-out section is missing")
    _require(list(heldout.get("seeds", [])) == list(SEEDS), "held-out seed order changed")
    _require(list(heldout.get("scenarios", [])) == list(SCENARIOS),
             "held-out scenario order changed")
    _require(list(heldout.get("controllers", [])) == list(CONTROLLERS),
             "controller order changed")
    for key in (
        "deterministic", "no_teacher", "no_labels", "no_updates",
        "paired_same_seed_and_scenario", "full_scenario_horizon",
    ):
        _require(heldout.get(key) is True, f"held-out {key} contract changed")
    for key, expected in {
        "flow6_horizon_seconds": 1210,
        "flow12_horizon_seconds": 1570,
        "dense_horizon_seconds": 1080,
        "expected_records": EXPECTED_RECORDS,
    }.items():
        _require(heldout.get(key) == expected, f"held-out horizon/count changed: {key}")

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
        "deterministic": True,
        "urgency_reward_config_exact": True,
        "completion_report_only": True,
    }
    for key, expected in expected_episode_gates.items():
        _require(episode_gates.get(key) == expected, f"episode gate changed: {key}")

    decision = value.get("decision_gates")
    _require(isinstance(decision, Mapping), "decision gates are missing")
    for key in (
        "all_records", "all_episode_integrity",
        "trained_dense_completion_aggregate_nonworse_than_initial",
        "trained_dense_strict_improvement",
        "trained_sparse_priority_weighted_wait_nonworse_than_initial",
        "trained_dense_priority_weighted_wait_nonworse_than_initial",
        "trained_overall_low_soc_mean_wait_nonworse_than_initial",
        "none_metrics_fail_closed", "rule_diagnostic_only",
    ):
        _require(decision.get(key) is True, f"decision gate changed: {key}")
    _require(decision.get("trained_sparse_completion") == 1.0,
             "sparse completion gate changed")

    coverage = value.get("coverage")
    _require(isinstance(coverage, Mapping), "coverage section is missing")
    _require(coverage.get("low_soc_threshold") == LOW_SOC_THRESHOLD,
             "low-SOC threshold changed")
    _require(coverage.get("priority_weighted_wait_weight") == "base_priority",
             "priority-weighted wait contract changed")

    reservation = value.get("reservation")
    _require(isinstance(reservation, Mapping), "reservation section is missing")
    _require(reservation.get("marker") == HELDOUT_RESERVATION_MARKER,
             "held-out reservation marker changed")
    _require(reservation.get("reserve_before_first_reset") is True,
             "reservation order changed")
    _require(reservation.get("one_time") is True, "reservation one-time contract changed")

    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "authorization section is missing")
    _require(authorization.get("heldout_evaluation_allowed_on_pass") is True,
             "held-out authorization changed")
    _require(authorization.get("heldout_untouched") is True,
             "held-out immutability changed")
    _require(authorization.get("P3_QUALIFIED_on_pass") is True,
             "P3 qualification authorization changed")
    _require(authorization.get("checkpoint_promotable_on_pass") is True,
             "checkpoint promotion authorization changed")
    _require(authorization.get("P4_ALLOWED") is False, "P4 authorization changed")
    return value


def source_hashes() -> dict[str, str]:
    """Hash all code and frozen inputs used by this held-out evaluation."""
    paths = (
        PROTOCOL,
        Path("terminal_marl/training/evaluate_v4_2_p3_mixed_priority_soc_heldout_v2.py"),
        Path("terminal_marl/training/evaluate_v4_2_p3_mixed_priority_soc_heldout_v1.py"),
        Path("terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_heldout_v1.yaml"),
        Path("terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v2.py"),
        Path("terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v2.yaml"),
        Path("terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v3.py"),
        TRAINING_PROTOCOL,
        SCHEDULER_SOURCE,
        Path("terminal_marl/core/rewards.py"),
        Path("terminal_marl/core/config.py"),
        Path("terminal_marl/envs/parallel_env.py"),
        Path("terminal_marl/training/evaluate_v4_2_p2_v4.py"),
        Path("terminal_marl/training/diagnose_v4_2_p3_mixed_priority_soc_pilot_v2.py"),
        Path("terminal_marl/training/evaluate_v4_2_p2_gate_runtime_shortlist_v1.py"),
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


def validate_training_root_manifest(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact V3 root PASS and formal checkpoint evidence."""
    bindings = p["bindings"]
    root = Path(path)
    _require(root.is_file(), f"training root manifest missing: {root}")
    _require(_sha(root) == bindings["training_root_manifest_sha256"],
             "training root manifest SHA mismatch")
    report = _read_json(root)
    expected = {
        "stage": TRAINING_STAGE,
        "PASS": True,
        "HELDOUT_EVALUATION_ALLOWED": True,
        "heldout_untouched": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "formal_pass_authorizes_heldout_only": True,
        "protocol_sha256": TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "urgency_reward_config": dict(REWARD_CONFIG),
    }
    for key, expected_value in expected.items():
        _require(report.get(key) == expected_value,
                 f"training root binding changed: {key}")
    entry = report.get("entry")
    _require(isinstance(entry, Mapping), "training entry result missing")
    _require(entry.get("PASS") is True, "training entry did not pass")
    formal = report.get("formal")
    _require(isinstance(formal, Mapping), "training formal result missing")
    _require(formal.get("PASS") is True, "training formal result did not pass")
    _require(formal.get("final_checkpoint_sha256") == bindings["trained_checkpoint_sha256"],
             "training final checkpoint SHA changed")
    _require(isinstance(formal.get("final_checkpoint"), str)
             and bool(formal["final_checkpoint"]),
             "training final checkpoint path missing")
    return report


def validate_training_reservation_marker(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable V3 marker that reserves held-out seeds."""
    bindings = p["bindings"]
    marker = Path(path)
    _require(marker.is_file(), f"training reservation marker missing: {marker}")
    _require(marker.name == bindings["training_reservation_marker"],
             "training reservation marker filename changed")
    _require(_sha(marker) == bindings["training_reservation_marker_sha256"],
             "training reservation marker SHA mismatch")
    payload = _read_json(marker)
    expected = {
        "stage": TRAINING_STAGE,
        "protocol_sha256": TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "entry_seeds": TRAINING_ENTRY_SEEDS,
        "formal_seeds": TRAINING_FORMAL_SEEDS,
        "heldout_seeds": TRAINING_HELDOUT_SEEDS,
        "heldout_untouched": True,
        "reserved_before_first_reset": True,
        "urgency_reward_config": dict(REWARD_CONFIG),
    }
    for key, expected_value in expected.items():
        _require(payload.get(key) == expected_value,
                 f"training reservation marker field changed: {key}")
    return payload


validate_training_marker = validate_training_reservation_marker


def validate_trained_checkpoint_payload(
    payload: Mapping[str, Any], p: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate V3 lineage and reward metadata returned by the checkpoint loader."""
    bindings = p["bindings"]
    expected = {
        "stage": TRAINING_STAGE,
        "formal": True,
        "ppo_called": True,
        "protocol_sha256": TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "checkpoint_promotable": False,
        "urgency_reward_config": dict(REWARD_CONFIG),
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
    try:
        payload = load_checkpoint(
            checkpoint, policy, generic_evaluate.p2_config(6), load_optimizer=False,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise HeldoutV2InputError("trained checkpoint loading failed") from exc
    _require(isinstance(payload, Mapping), "trained checkpoint payload is invalid")
    validate_trained_checkpoint_payload(payload, p)
    _require(len(policy.optimizer.state) == 0,
             "trained checkpoint optimizer state was loaded")
    policy.model.eval()
    return policy, dict(payload)


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
        raise HeldoutV2InputError("initial qualified checkpoint validation failed") from exc
    _require(len(policy.optimizer.state) == 0,
             "initial checkpoint optimizer state was loaded")
    policy.model.eval()
    return policy, dict(payload)


_load_initial = load_initial_checkpoint


def scenario_config(name: str, training_p: Mapping[str, Any] | None = None):
    """Use the V3 builder, including full horizons and urgency reward values."""
    p = training_protocol() if training_p is None else training_p
    config = v3_scenario_config(name, p)
    _require(_urgency_reward_config_exact(config, p),
             "scenario urgency reward configuration changed")
    expected = {"flow6": 1210.0, "flow12": 1570.0, "dense24": 1080.0}
    _require(float(config.max_simulation_seconds) == expected[name],
             f"{name} full horizon changed")
    return config


def _urgency_reward_config_exact(config: Any, p: Mapping[str, Any] | None = None) -> bool:
    reward = REWARD_CONFIG if p is None else p.get("reward", REWARD_CONFIG)
    try:
        return all(
            math.isfinite(float(getattr(config, key)))
            and float(getattr(config, key)) == float(reward[key])
            for key in REWARD_CONFIG
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def evaluate_episode(
    *,
    controller: str,
    policy: Any,
    config: Any,
    seed: int,
    output: Path,
    frozen_proposal_subgraph: bool,
    low_soc_threshold: float = LOW_SOC_THRESHOLD,
    training_p: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reuse V1 episode mechanics and add the frozen V3 reward gate."""
    record = heldout_v1.evaluate_episode(
        controller=controller,
        policy=policy,
        config=config,
        seed=seed,
        output=output,
        frozen_proposal_subgraph=frozen_proposal_subgraph,
        low_soc_threshold=low_soc_threshold,
    )
    result = record.get("result")
    _require(isinstance(result, Mapping), "episode result is invalid")
    gates = result.get("gates")
    _require(isinstance(gates, dict), "episode gates are invalid")
    exact = _urgency_reward_config_exact(config, training_p)
    gates["urgency_reward_config_exact"] = bool(exact)
    result["urgency_reward_config_exact"] = bool(exact)
    result["completion_report_only"] = True
    record["completion_report_only"] = True
    record["episode_integrity"] = bool(gates) and all(bool(value) for value in gates.values())
    _write(Path(output) / "evaluation.json", result)
    return record


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    controller: str,
    scenario_group: str = "overall",
    *,
    low_soc_threshold: float = LOW_SOC_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate completion and lifecycle wait metrics using the V1 helper."""
    return heldout_v1.aggregate_records(
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


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric(
    aggregates: Mapping[str, Any], group: str, controller: str, name: str
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
    records: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    *,
    frozen_proposal_subgraph: bool,
) -> dict[str, bool]:
    """Apply the V1 decision gates to the V3 held-out seed pairing."""
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
        "trained_sparse_completion": sparse_completion == 1.0
        if sparse_completion is not None else False,
        "trained_dense_completion_aggregate_nonworse_than_initial": (
            trained_dense_completion >= initial_dense_completion
            if trained_dense_completion is not None and initial_dense_completion is not None
            else False
        ),
        "trained_dense_strict_improvement": dense_strict,
        "trained_sparse_priority_weighted_wait_nonworse_than_initial": (
            trained_sparse_wait <= initial_sparse_wait
            if trained_sparse_wait is not None and initial_sparse_wait is not None
            else False
        ),
        "trained_dense_priority_weighted_wait_nonworse_than_initial": (
            trained_dense_wait <= initial_dense_wait
            if trained_dense_wait is not None and initial_dense_wait is not None
            else False
        ),
        "trained_overall_low_soc_mean_wait_nonworse_than_initial": (
            trained_low_soc <= initial_low_soc
            if trained_low_soc is not None and initial_low_soc is not None
            else False
        ),
        "rule_diagnostic_only": True,
    }
    gates["all"] = all(gates.values())
    return gates


def compute_deltas(aggregates: Mapping[str, Any]) -> dict[str, Any]:
    """Use the V1 paired metric delta representation."""
    return heldout_v1.compute_deltas(aggregates)


def _scenario_horizon_exact(config: Any, scenario: str) -> bool:
    expected = {"flow6": 1210.0, "flow12": 1570.0, "dense24": 1080.0}
    return scenario in expected and float(config.max_simulation_seconds) == expected[scenario]


def run(
    *,
    output: Path,
    training_root_manifest: Path,
    trained_checkpoint: Path,
    initial_checkpoint: Path,
    training_reservation_marker: Path | None = None,
) -> dict[str, Any]:
    """Run the once-only deterministic V3 held-out evaluation."""
    p = protocol()
    bindings = p["bindings"]
    output = Path(output)
    _require(not output.exists(), f"output already exists: {output}")
    training_root = validate_training_root_manifest(Path(training_root_manifest), p)
    if training_reservation_marker is not None:
        marker = Path(training_reservation_marker)
    else:
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
    _require(frozen_proposal_subgraph, "frozen proposal subgraph changed")

    heldout_marker = Path(trained_checkpoint).parent / str(
        p["reservation"]["marker"]
    )
    _require(not heldout_marker.exists(),
             f"held-out consumption marker already exists: {heldout_marker}")
    output.mkdir(parents=True, exist_ok=False)
    reserve_once(heldout_marker, {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "training_root_manifest_sha256": bindings["training_root_manifest_sha256"],
        "trained_checkpoint_sha256": bindings["trained_checkpoint_sha256"],
        "initial_checkpoint_sha256": bindings["initial_checkpoint_sha256"],
        "heldout_seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "controllers": list(CONTROLLERS),
        "heldout_untouched": True,
        "output": str(output),
        "reserved_before_first_reset": True,
        "one_time": True,
        "urgency_reward_config": dict(REWARD_CONFIG),
    })
    _write(output / "pre_reset_registry.json", {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "heldout_seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "controllers": list(CONTROLLERS),
        "reservation_marker": str(heldout_marker),
        "reserved_before_first_reset": True,
        "heldout_untouched": True,
        "urgency_reward_config": dict(REWARD_CONFIG),
    })

    scenario_protocol = training_protocol()
    records: list[dict[str, Any]] = []
    for seed, scenario in zip(SEEDS, SCENARIOS):
        config = scenario_config(str(scenario), scenario_protocol)
        _require(_scenario_horizon_exact(config, str(scenario)),
                 f"{scenario} did not use its full frozen horizon")
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
        "training_root_manifest_sha256": bindings["training_root_manifest_sha256"],
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
        "training_root_formal_pass": training_root["formal"]["PASS"],
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
        "urgency_reward_config": dict(REWARD_CONFIG),
        "completion_report_only": True,
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root-manifest", "--root-manifest",
                        dest="training_root_manifest", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--training-reservation-marker", type=Path)
    parser.add_argument("--history-root", type=Path)
    args = parser.parse_args()
    result = run(
        output=args.output,
        training_root_manifest=args.training_root_manifest,
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
