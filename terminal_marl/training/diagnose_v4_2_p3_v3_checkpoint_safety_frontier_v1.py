"""One-time safety-frontier diagnostic for the exposed P3 V3 checkpoint chain.

The diagnostic evaluates the original K=3 checkpoint and checkpoint_1 through
checkpoint_12 on the already-consumed dense24 seed 825006.  It is deliberately
not a qualification or promotion protocol: no seed is reserved, no optimizer
is loaded, and every candidate is evaluated even after a SafetyInvariantError.
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
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.training import (
    diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as pilot_v2,
)
from terminal_marl.training import (
    evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness,
)
from terminal_marl.training.run_v4_2_p2_runtime_shortlist_entry_v1 import (
    _frozen_state_equal,
)
from terminal_marl.training.run_v4_2_p3_mixed_priority_soc_training_v3 import (
    REWARD_CONFIG,
    REWARD_SOURCE_HASHES,
    scenario_config,
)
from terminal_marl.training.run_v4_2_p3_mixed_priority_soc_training_v3 import (
    protocol as training_protocol,
)
from terminal_marl.training.run_v4_2_ppo_block2 import replay_geometry
from terminal_marl.training import student_rollout_v4_2


STAGE = "V4_2_P3_V3_CHECKPOINT_SAFETY_FRONTIER_V1"
PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_v3_checkpoint_safety_frontier_v1.yaml"
)
PROTOCOL_SHA = "1bd3af56afc3b7db5199125a2ef5ca6c1205913966d19c653e4cfd901a118269"
HELDOUT_PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_heldout_v2.yaml"
)
HELDOUT_PROTOCOL_SHA = (
    "bdcd4be571555b0f512af30e8bf548cc11948b3347e18b552093ae1997e42e7a"
)
HELDOUT_MARKER_NAME = "p3_mixed_priority_soc_heldout_v2_consumed.json"
HELDOUT_MARKER_SHA = (
    "13e4811e25637273bd8f2f91defd2affc87f343cb6dc7a5d1512fc7be7762946"
)
FAILED_SEED = 825006
FAILED_SCENARIO = "dense24"
FAILED_SUMMARY_SHA = (
    "080c7799a235fba1ca1c4b79d29f5bd0e00e44eec616cca1c5c3d6fb75dff3ec"
)
FAILED_TAPE_SHA = (
    "97bfb66b974cfd4e11012733c71cb8c16936a20b359d9af3bbd43d4970a50d6f"
)
FAILED_SAFETY_ERROR_TYPE = "SafetyInvariantError"
FAILED_SAFETY_MESSAGE = (
    "No certified bounded joint backup action; physics execution blocked for "
    "components ((35, 118),)"
)
FINAL_CHECKPOINT_SHA = (
    "0fb4acf9b05ba2fa8f5700e64eca6c2b9fc6174bb091c4e12ee4c1ea3942fb6b"
)
INITIAL_CHECKPOINT_SHA = (
    "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
)
TRAINING_PROTOCOL_SHA = (
    "c69dc4b308e97652936c8da5bb6a47ef61a73a4f3cb599de8f39cb703f3c4d25"
)
SCHEDULER_SOURCE = Path("terminal_marl/core/merge_scheduler.py")
SCHEDULER_SOURCE_SHA = (
    "58bf3f3a29b86f00855610000a819424e5c90b0c1f1433a780bcb42f86309ce1"
)
SEED = FAILED_SEED
SCENARIO = FAILED_SCENARIO
FULL_HORIZON_SECONDS = 1080.0
LOW_SOC_THRESHOLD = 0.65
CHECKPOINT_SHA256S = (
    "b68a91bb4bedc3731c9fd57a361e8743ee59da1e82a10cad9cf37082b5e20cd1",
    "e02f51537f5ec85b6d496d987384a432508748d395c354465dda9a927624aae9",
    "a562207301725375099608485b02db8ba11eda50fbe9651e66e0930e5d34d9dd",
    "7524498ecfcb7b1d15bea775a297f803d693dcf72fcb980d3426075c2ddd728f",
    "19c31195dd881e7e4fde91530aff00375f2787754569491749474699b7f06b2d",
    "905582829eae0d93c9352498ea7ed42f334f1597f9d22af42980b67179189cd3",
    "361654b5fae00a9da5dccaf186bc7b227069de496f20c96dd061e230e9693bb8",
    "1c1c59dd4ea511ea7b9198372087819999055d16158cabeb9bd9aa8a27daf5a9",
    "16a2be5b3606932530fd684473210faee87bc46dad4b6a06427c251129a9c592",
    "9b0fef1c898b8b4aeac206fe8d410749f36bbefb677245a2cb7021c81d646582",
    "8535db9e89adcfc31cecff86f5883b151f7837b1e00b22bc657c08fb5ddbb962",
    FINAL_CHECKPOINT_SHA,
)
CHECKPOINT_INDICES = tuple(range(1, 13))


class SafetyFrontierInputError(ValueError):
    """Raised when a frozen diagnostic input violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SafetyFrontierInputError(message)


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
        raise SafetyFrontierInputError(f"cannot read JSON input: {source}") from exc
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
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def protocol() -> dict[str, Any]:
    """Read and validate the frozen diagnostic protocol."""
    _require(PROTOCOL.is_file(), f"diagnostic protocol missing: {PROTOCOL}")
    _require(PROTOCOL_SHA != "TO_BE_FROZEN" and _sha(PROTOCOL) == PROTOCOL_SHA,
             "safety-frontier protocol changed after freeze")
    _require(_sha(HELDOUT_PROTOCOL) == HELDOUT_PROTOCOL_SHA,
             "held-out V2 protocol changed")
    _require(_sha(SCHEDULER_SOURCE) == SCHEDULER_SOURCE_SHA,
             "scheduler source changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "diagnostic protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "diagnostic protocol id changed")
    _require(value.get("status") == "PREREGISTERED_EXPOSED_SEED_DIAGNOSTIC",
             "diagnostic status changed")
    _require(value.get("exposed_seed_diagnostic_only") is True,
             "diagnostic seed authorization changed")

    bindings = value.get("bindings")
    _require(isinstance(bindings, Mapping), "diagnostic bindings are missing")
    expected_bindings = {
        "heldout_protocol": HELDOUT_PROTOCOL.as_posix(),
        "heldout_protocol_sha256": HELDOUT_PROTOCOL_SHA,
        "heldout_consumption_marker": HELDOUT_MARKER_NAME,
        "heldout_consumption_marker_sha256": HELDOUT_MARKER_SHA,
        "failed_seed": FAILED_SEED,
        "failed_scenario": FAILED_SCENARIO,
        "failed_summary_sha256": FAILED_SUMMARY_SHA,
        "failed_action_tape_sha256": FAILED_TAPE_SHA,
        "failed_safety_error_type": FAILED_SAFETY_ERROR_TYPE,
        "failed_safety_component_message": FAILED_SAFETY_MESSAGE,
        "final_checkpoint_sha256": FINAL_CHECKPOINT_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"diagnostic binding changed: {key}")

    scenario = value.get("scenario")
    _require(isinstance(scenario, Mapping), "diagnostic scenario is missing")
    for key, expected in {
        "name": SCENARIO, "seed": SEED, "deterministic": True,
        "no_updates": True, "no_labels": True, "no_teacher": True,
        "full_horizon_seconds": 1080, "total_requested_uavs": 24,
        "urgency_reward_config_exact": True,
        "low_level_priority_time_weight": 1.0,
        "low_level_soc_urgency_time_weight": 2.0,
        "low_level_soc_urgency_threshold": 0.65,
    }.items():
        _require(scenario.get(key) == expected, f"diagnostic scenario changed: {key}")

    candidates = value.get("candidates")
    _require(isinstance(candidates, Mapping), "candidate section is missing")
    _require(candidates.get("original_label") == "original_k3",
             "original candidate label changed")
    _require(candidates.get("checkpoint_pattern") == "checkpoint_{index}.pt",
             "checkpoint path pattern changed")
    _require(list(candidates.get("indices", [])) == list(CHECKPOINT_INDICES),
             "checkpoint index order changed")
    _require(list(candidates.get("sha256", [])) == list(CHECKPOINT_SHA256S),
             "checkpoint SHA order changed")

    diagnostic = value.get("diagnostic")
    _require(isinstance(diagnostic, Mapping), "diagnostic section is missing")
    for key, expected in {
        "evaluate_all_candidates_after_failure": True,
        "continue_after_safety_invariant": True,
        "direct_student_rollout": "terminal_marl/training/student_rollout_v4_2.py",
        "no_generic_evaluator": True,
        "replay_only_when_safety_zero": True,
        "priority_weighted_wait_weight": "base_priority",
        "low_soc_threshold": LOW_SOC_THRESHOLD,
        "output_independent": True,
    }.items():
        _require(diagnostic.get(key) == expected, f"diagnostic execution contract changed: {key}")

    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "diagnostic authorization is missing")
    for key, expected in {
        "diagnostic_only": True,
        "PASS_used_for_promotion": False,
        "checkpoint_promotable": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    }.items():
        _require(authorization.get(key) == expected, f"diagnostic authorization changed: {key}")
    return value


def source_hashes() -> dict[str, str]:
    paths = (
        PROTOCOL,
        HELDOUT_PROTOCOL,
        Path("terminal_marl/training/diagnose_v4_2_p3_v3_checkpoint_safety_frontier_v1.py"),
        Path("terminal_marl/training/evaluate_v4_2_p3_mixed_priority_soc_heldout_v2.py"),
        Path("terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v3.py"),
        Path("terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v3.yaml"),
        Path("terminal_marl/training/student_rollout_v4_2.py"),
        Path("terminal_marl/training/diagnose_v4_2_p3_mixed_priority_soc_pilot_v2.py"),
        Path("terminal_marl/training/run_v4_2_ppo_block2.py"),
        Path("terminal_marl/training/run_v4_2_p2_runtime_shortlist_entry_v1.py"),
        Path("terminal_marl/core/checkpoints.py"),
        Path("terminal_marl/core/runtime_traffic_safety.py"),
        Path("terminal_marl/core/safety_trace.py"),
        SCHEDULER_SOURCE,
        Path("terminal_marl/core/rewards.py"),
        Path("terminal_marl/core/config.py"),
        Path("terminal_marl/envs/parallel_env.py"),
    )
    hashes = {path.as_posix(): _sha(path) for path in paths}
    for path, expected in REWARD_SOURCE_HASHES.items():
        _require(hashes.get(path) == expected, f"reward source changed: {path}")
    _require(hashes[SCHEDULER_SOURCE.as_posix()] == SCHEDULER_SOURCE_SHA,
             "scheduler source changed")
    return hashes


def _validate_consumption_marker(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    bindings = p["bindings"]
    marker = Path(path)
    _require(marker.is_file(), f"held-out consumption marker missing: {marker}")
    _require(marker.name == bindings["heldout_consumption_marker"],
             "held-out marker filename changed")
    _require(_sha(marker) == bindings["heldout_consumption_marker_sha256"],
             "held-out consumption marker SHA mismatch")
    payload = _read_json(marker)
    expected = {
        "stage": "V4_2_P3_MIXED_PRIORITY_SOC_HELDOUT_V2",
        "protocol_sha256": bindings["heldout_protocol_sha256"],
        "heldout_seeds": [825001, 825002, 825003, 825004, 825005, 825006],
        "scenarios": ["flow6", "flow12", "dense24", "flow6", "flow12", "dense24"],
        "controllers": ["trained", "initial", "rule"],
        "heldout_untouched": True,
        "reserved_before_first_reset": True,
        "one_time": True,
        "urgency_reward_config": dict(REWARD_CONFIG),
    }
    for key, expected_value in expected.items():
        _require(payload.get(key) == expected_value,
                 f"held-out consumption marker field changed: {key}")
    return payload


def _validate_failed_summary(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    bindings = p["bindings"]
    summary_path = Path(path)
    _require(summary_path.is_file(), f"failed summary missing: {summary_path}")
    _require(_sha(summary_path) == bindings["failed_summary_sha256"],
             "failed summary SHA mismatch")
    summary = _read_json(summary_path)
    _require(summary.get("seed") == FAILED_SEED, "failed summary seed changed")
    safety = summary.get("safety")
    _require(isinstance(safety, Mapping), "failed summary safety is missing")
    _require(int(safety.get("SafetyInvariantError", 0)) == 1,
             "failed summary SafetyInvariantError count changed")
    _require(summary.get("safety_error") == bindings["failed_safety_component_message"],
             "failed summary SafetyInvariant message changed")
    return summary


def _validate_failed_tape(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    bindings = p["bindings"]
    tape_path = Path(path)
    _require(tape_path.is_file(), f"failed action tape missing: {tape_path}")
    _require(_sha(tape_path) == bindings["failed_action_tape_sha256"],
             "failed action tape SHA mismatch")
    tape = _read_json(tape_path)
    _require(tape.get("episode_seed") == FAILED_SEED, "failed tape seed changed")
    records = tape.get("action_tape")
    _require(isinstance(records, list), "failed tape action records are missing")
    safety_events = [
        item.get("safety_event") for item in records
        if isinstance(item, Mapping) and isinstance(item.get("safety_event"), Mapping)
    ]
    _require(any(
        event.get("kind") == "safety_invariant_error"
        and event.get("message") == bindings["failed_safety_component_message"]
        for event in safety_events
    ), "failed tape SafetyInvariant evidence changed")
    return tape


def validate_failed_evidence(
    *, marker: Path, summary: Path, action_tape: Path, p: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the immutable held-out failure before any diagnostic reset."""
    value = protocol() if p is None else p
    marker_value = _validate_consumption_marker(marker, value)
    summary_value = _validate_failed_summary(summary, value)
    tape_value = _validate_failed_tape(action_tape, value)
    return {"marker": marker_value, "summary": summary_value, "tape": tape_value}


def resolve_candidate_paths(
    checkpoint_dir: Path, initial_checkpoint: Path, p: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    value = protocol() if p is None else p
    candidates = value["candidates"]
    paths = [{
        "index": 0,
        "label": str(candidates["original_label"]),
        "path": Path(initial_checkpoint),
        "sha256": value["bindings"]["initial_checkpoint_sha256"],
    }]
    for index, expected_sha in zip(candidates["indices"], candidates["sha256"]):
        path = Path(checkpoint_dir) / str(candidates["checkpoint_pattern"]).format(index=int(index))
        paths.append({"index": int(index), "label": f"checkpoint_{int(index)}",
                      "path": path, "sha256": str(expected_sha)})
    for candidate in paths:
        path = candidate["path"]
        _require(path.is_file(), f"candidate checkpoint missing: {path}")
        _require(_sha(path) == candidate["sha256"],
                 f"candidate checkpoint SHA mismatch: {candidate['label']}")
    return paths


def validate_trained_checkpoint_payload(
    payload: Mapping[str, Any], p: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = protocol() if p is None else p
    expected = {
        "stage": "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V3",
        "formal": True,
        "ppo_called": True,
        "protocol_sha256": TRAINING_PROTOCOL_SHA,
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "checkpoint_promotable": False,
        "urgency_reward_config": dict(REWARD_CONFIG),
    }
    for key, expected_value in expected.items():
        _require(payload.get(key) == expected_value,
                 f"trained checkpoint lineage changed: {key}")
    return dict(payload)


def load_candidate(
    candidate: Mapping[str, Any], config: Any, p: Mapping[str, Any] | None = None,
) -> tuple[TemperatureCalibratedGateResidualAdapterStudentPPOV42, dict[str, Any]]:
    value = protocol() if p is None else p
    path = Path(candidate["path"])
    if int(candidate["index"]) == 0:
        try:
            policy, payload = effectiveness.load_candidate_checkpoint(
                path, expected_sha256=value["bindings"]["initial_checkpoint_sha256"]
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise SafetyFrontierInputError("original K3 checkpoint validation failed") from exc
        _require(len(policy.optimizer.state) == 0,
                 "original K3 checkpoint optimizer state was loaded")
        policy.model.eval()
        return policy, dict(payload)
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
    )
    try:
        payload = load_checkpoint(path, policy, config, load_optimizer=False)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise SafetyFrontierInputError(
            f"trained checkpoint loading failed: {candidate['label']}"
        ) from exc
    _require(isinstance(payload, Mapping), "trained checkpoint payload is invalid")
    validate_trained_checkpoint_payload(payload, value)
    _require(len(policy.optimizer.state) == 0,
             "trained checkpoint optimizer state was loaded")
    policy.model.eval()
    return policy, dict(payload)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _urgency_reward_config_exact(config: Any) -> bool:
    try:
        return all(float(getattr(config, key)) == float(value)
                   for key, value in REWARD_CONFIG.items())
    except (AttributeError, TypeError, ValueError):
        return False


def classify_summary(summary: Mapping[str, Any], config: Any) -> dict[str, Any]:
    """Classify one raw student rollout without rerunning likelihood checks."""
    safety_value = summary.get("safety")
    safety = dict(safety_value) if isinstance(safety_value, Mapping) else {}
    safety_error = summary.get("safety_error")
    safety_error_present = (
        int(safety.get("SafetyInvariantError", 0)) > 0
        or isinstance(safety_error, str) and bool(safety_error)
    )
    counts = summary.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}
    actor_decisions = int(counts.get("actor_decisions", 0))
    shield_actor = int(counts.get("shield_actor", 0))
    requests = int(getattr(config, "total_requested_uavs", 0))
    completed = int(summary.get("completed", 0))
    shortlist_value = summary.get("runtime_shortlist")
    shortlist = dict(shortlist_value) if isinstance(shortlist_value, Mapping) else {}
    safety_all_zero = bool(safety) and all(int(value) == 0 for value in safety.values())
    return {
        "safety": safety,
        "safety_error": safety_error,
        "safety_error_type": FAILED_SAFETY_ERROR_TYPE if safety_error_present else None,
        "safety_error_present": bool(safety_error_present),
        "safety_all_zero": bool(safety_all_zero),
        "completed": completed,
        "requests": requests,
        "completion_fraction": None if requests <= 0 else float(completed / requests),
        "seconds": _finite(summary.get("seconds")),
        "steps": int(summary.get("steps", 0)),
        "shield_fraction": float(shield_actor / max(1, actor_decisions)),
        "actor_decisions": actor_decisions,
        "runtime_shortlist": shortlist,
        "shortlist_exhausted": int(shortlist.get("runtime_shortlist_exhausted", 0)),
        "shortlist_fallback_selected": int(shortlist.get("runtime_shortlist_fallback_selected", 0)),
        "action_tape": summary.get("action_tape"),
        "terminal": bool(summary.get("terminal", False)),
        "urgency_reward_config_exact": _urgency_reward_config_exact(config),
    }


def classify_result(summary: Mapping[str, Any], config: Any) -> dict[str, Any]:
    """Compatibility alias for the raw-summary classifier."""
    return classify_summary(summary, config)


def _lifecycle_summary(config: Any, tape_path: str | Path) -> dict[str, Any]:
    lifecycle = pilot_v2.lifecycle_metrics(
        config, tape_path, low_soc_threshold=LOW_SOC_THRESHOLD,
    )
    low_soc = lifecycle.get("low_soc")
    return {
        "replay_exact": lifecycle.get("replay_exact") is True,
        "priority_weighted_wait_seconds": lifecycle.get("priority_weighted_wait_seconds"),
        "low_soc_mean_wait_seconds": (
            low_soc.get("mean_wait_seconds")
            if isinstance(low_soc, Mapping) else None
        ),
        "completed": lifecycle.get("completed"),
        "rows": len(lifecycle.get("rows", []))
        if isinstance(lifecycle.get("rows"), list) else None,
    }


def _analyze_safe_rollout(config: Any, record: dict[str, Any]) -> None:
    tape_path = record.get("action_tape")
    if not isinstance(tape_path, str) or not tape_path:
        record["analysis_error"] = "safe rollout did not provide an action tape"
        record["diagnostic_analysis_error"] = True
        record["exact_replay"] = False
        return
    try:
        exact, geometry = replay_geometry(config, tape_path)
        record["exact_replay"] = bool(exact)
        record["geometry"] = geometry
        record["lifecycle"] = _lifecycle_summary(config, tape_path)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
        record["analysis_error"] = f"{type(exc).__name__}: {exc}"
        record["diagnostic_analysis_error"] = True
        record["exact_replay"] = False


def frontier_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize safety failures without treating them as a stop condition."""
    failure_indices = [
        int(record["index"]) for record in records
        if bool(record.get("safety_error_present"))
    ]
    safe_indices = [
        int(record["index"]) for record in records
        if bool(record.get("safety_all_zero"))
    ]
    earliest_failed = min(failure_indices) if failure_indices else None
    safe_before = [index for index in safe_indices
                   if earliest_failed is None or index < earliest_failed]
    return {
        "earliest_failed_checkpoint": earliest_failed,
        "safe_checkpoint_indices": safe_indices,
        "failure_checkpoint_indices": failure_indices,
        "last_safe_before_first_failure": max(safe_before) if safe_before else None,
    }


def run(
    *,
    output: Path,
    checkpoint_dir: Path,
    initial_checkpoint: Path,
    heldout_consumption_marker: Path,
    failed_summary: Path,
    failed_action_tape: Path,
) -> dict[str, Any]:
    """Run all 13 candidates without reserving or consuming any seed."""
    p = protocol()
    _require(not Path(output).exists(), f"diagnostic output already exists: {output}")
    evidence = validate_failed_evidence(
        marker=Path(heldout_consumption_marker), summary=Path(failed_summary),
        action_tape=Path(failed_action_tape), p=p,
    )
    training_p = training_protocol()
    config = scenario_config(SCENARIO, training_p)
    _require(float(config.max_simulation_seconds) == FULL_HORIZON_SECONDS,
             "dense24 diagnostic horizon changed")
    _require(_urgency_reward_config_exact(config),
             "diagnostic urgency reward configuration changed")
    candidates = resolve_candidate_paths(checkpoint_dir, initial_checkpoint, p)
    frozen_source_hashes = source_hashes()
    loaded: list[dict[str, Any]] = []
    initial_policy: Any = None
    initial_state: Mapping[str, Any] | None = None
    for candidate in candidates:
        policy, payload = load_candidate(candidate, config, p)
        if initial_policy is None:
            initial_policy = policy
            initial_state = policy.model.state_dict()
            frozen = True
        else:
            frozen = _frozen_state_equal(initial_state, policy.model.state_dict())
        loaded.append({"candidate": candidate, "policy": policy, "payload": payload,
                       "frozen_proposal_subgraph": bool(frozen)})

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    max_steps = int(FULL_HORIZON_SECONDS / config.dt)
    for item in loaded:
        candidate = item["candidate"]
        folder = output / str(candidate["label"])
        record: dict[str, Any] = {
            "index": int(candidate["index"]),
            "label": str(candidate["label"]),
            "checkpoint": str(candidate["path"]),
            "checkpoint_sha256": str(candidate["sha256"]),
            "frozen_proposal_subgraph": item["frozen_proposal_subgraph"],
            "seed": SEED,
            "scenario": SCENARIO,
            "deterministic": True,
            "no_updates": True,
            "no_labels": True,
            "no_teacher": True,
            "completion_report_only": True,
        }
        try:
            _, summary = student_rollout_v4_2.rollout(
                item["policy"], config, seed=SEED, max_steps=max_steps,
                output=folder, deterministic=True, collect_labels=False,
            )
            record.update(classify_summary(summary, config))
            if record["safety_all_zero"]:
                _analyze_safe_rollout(config, record)
            else:
                record["exact_replay"] = None
                record["lifecycle"] = None
        except SafetyInvariantError as exc:
            record.update({
                "safety": {"SafetyInvariantError": 1},
                "safety_error": str(exc),
                "safety_error_type": type(exc).__name__,
                "safety_error_present": True,
                "safety_all_zero": False,
                "exact_replay": None,
                "lifecycle": None,
                "analysis_error": "rollout raised SafetyInvariantError",
                "diagnostic_execution_error": False,
            })
        except Exception as exc:  # continue the frontier sweep after one candidate
            record.update({
                "safety": {},
                "safety_error": None,
                "safety_error_type": type(exc).__name__,
                "safety_error_present": False,
                "safety_all_zero": False,
                "exact_replay": None,
                "lifecycle": None,
                "analysis_error": f"{type(exc).__name__}: {exc}",
                "diagnostic_execution_error": True,
            })
        records.append(record)
        _write(output / f"{candidate['label']}.json", record)
        _write(output / "progress.json", {
            "completed_candidates": len(records),
            "expected_candidates": len(loaded),
            "last_index": candidate["index"],
            "last_label": candidate["label"],
            "last_safety_all_zero": record.get("safety_all_zero"),
            "last_safety_error": record.get("safety_error"),
        })

    frontier = frontier_summary(records)
    complete = len(records) == len(loaded)
    diagnostic_integrity = complete and not any(
        bool(record.get("diagnostic_execution_error"))
        or bool(record.get("diagnostic_analysis_error"))
        for record in records
    )
    result = {
        "stage": STAGE,
        "PASS": bool(diagnostic_integrity),
        "diagnostic_only": True,
        "exposed_seed_diagnostic_only": True,
        "checkpoint_promotable": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "seed": SEED,
        "scenario": SCENARIO,
        "horizon_seconds": FULL_HORIZON_SECONDS,
        "all_candidates_completed": complete,
        "diagnostic_integrity": bool(diagnostic_integrity),
        "candidate_count": len(records),
        "expected_candidate_count": len(loaded),
        **frontier,
        "records": records,
        "failed_evidence": {
            "heldout_protocol_sha256": p["bindings"]["heldout_protocol_sha256"],
            "heldout_consumption_marker_sha256": p["bindings"]["heldout_consumption_marker_sha256"],
            "failed_summary_sha256": p["bindings"]["failed_summary_sha256"],
            "failed_action_tape_sha256": p["bindings"]["failed_action_tape_sha256"],
            "safety_error_type": p["bindings"]["failed_safety_error_type"],
            "safety_component_message": p["bindings"]["failed_safety_component_message"],
        },
        "checkpoint_sha256s": list(CHECKPOINT_SHA256S),
        "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "source_hashes": frozen_source_hashes,
        "no_seed_reservation": True,
        "no_seed_consumption": True,
        "urgency_reward_config": dict(REWARD_CONFIG),
    }
    _write(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-consumption-marker", type=Path, required=True)
    parser.add_argument("--failed-summary", type=Path, required=True)
    parser.add_argument("--failed-action-tape", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(
        output=args.output,
        checkpoint_dir=args.checkpoint_dir,
        initial_checkpoint=args.initial_checkpoint,
        heldout_consumption_marker=args.heldout_consumption_marker,
        failed_summary=args.failed_summary,
        failed_action_tape=args.failed_action_tape,
    )
    print(json.dumps({
        "stage": result["stage"],
        "PASS": result["PASS"],
        "diagnostic_only": result["diagnostic_only"],
        "earliest_failed_checkpoint": result["earliest_failed_checkpoint"],
        "safe_checkpoint_indices": result["safe_checkpoint_indices"],
        "failure_checkpoint_indices": result["failure_checkpoint_indices"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
