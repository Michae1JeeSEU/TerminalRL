"""Exact full-tape timeline replay for the exposed P3 V3 Q3 dense stall.

This module is deliberately a diagnostic-only reader of one already-consumed
Q3 action tape.  It does not load a policy, reserve a seed, alter actions, or
change any environment parameter.  Every recorded action and runtime
shortlist is replayed and every recorded observation/post-state hash is
checked before a timeline artifact is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from terminal_marl.core.low_level_joint_action_v4_2 import (
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest, _json
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.run_v4_2_p3_mixed_priority_soc_training_v3 import (
    REWARD_CONFIG,
    scenario_config,
)
from terminal_marl.training.run_v4_2_p3_mixed_priority_soc_training_v3 import (
    protocol as training_protocol,
)


STAGE = "V4_2_P3_Q3_DENSE_STALL_TIMELINE_V1"
PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_q3_dense_stall_timeline_v1.yaml"
)
PROTOCOL_SHA = (
    "49d43d565127e9cc01a6fa35f8ead1613ff8a3e9e7c02c6f9b38ca318a8e747c"
)
Q3_MANIFEST_SHA = (
    "2ad3c6e52d52a2431f86284cd604911aed99329bbc9af897fbbb723548309e8c"
)
Q3_PROTOCOL_SHA = (
    "9dad9f8287d02ef1ac584553fab35c23089a08f2c0d940472778cb1ccb367dc8"
)
ACTION_TAPE_SHA = (
    "52754eb3ca225a8ddd0b4de424b9e57405d241c6a1098370f48f048234162d25"
)
TRAINED_CHECKPOINT_SHA = (
    "19c31195dd881e7e4fde91530aff00375f2787754569491749474699b7f06b2d"
)
TRAINING_PROTOCOL_SHA = (
    "c69dc4b308e97652936c8da5bb6a47ef61a73a4f3cb599de8f39cb703f3c4d25"
)
SEED = 826006
SCENARIO = "dense24"
CONTROLLER = "trained"
HORIZON_SECONDS = 1080.0
EXPECTED_STEPS = 1080
TOTAL_REQUESTED_UAVS = 24
LOW_SOC_THRESHOLD = 0.65

SOURCE_EXPECTED = {
    "terminal_marl/training/run_v4_2_p3_mixed_priority_soc_training_v3.py": (
        "42344154369c8cdbb2bb065be2088d99e12b1f894aa7d33355f616a6b9c45249"
    ),
    "terminal_marl/training/diagnose_v4_2_p3_mixed_priority_soc_pilot_v2.py": (
        "7ec050733ef5139e9b003db6b1ee8594b87d114a84bbbec7b1a57996a71b4d15"
    ),
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v3.yaml": (
        TRAINING_PROTOCOL_SHA
    ),
    "terminal_marl/configs/low_v4_2/p3_v3_early_stop_q3_v1.yaml": (
        Q3_PROTOCOL_SHA
    ),
    "terminal_marl/core/safety_trace.py": (
        "cd66cf735630793f23214c11770448d6e9fb0ae7d877b2bf7a4f11dcbf0518b3"
    ),
    "terminal_marl/envs/parallel_env.py": (
        "925ac9a03e56327b98ef29d0b5c408720795cfa61fde6e8ff3ef5036f8255316"
    ),
    "terminal_marl/core/config.py": (
        "5fbadde0a690b02db05fcaafdca76c814502a33e382f57a5700aed6ce86f9dd7"
    ),
    "terminal_marl/core/rewards.py": (
        "e39503bb2e6ed021d03268c0894fa123efb39ef30b85a225f721b4a1f9dfb28b"
    ),
    "terminal_marl/core/merge_scheduler.py": (
        "58bf3f3a29b86f00855610000a819424e5c90b0c1f1433a780bcb42f86309ce1"
    ),
}


class TimelineInputError(ValueError):
    """Raised when a frozen diagnostic input or protocol is not exact."""


class TimelineReplayError(TimelineInputError):
    """Raised when a recorded replay transition does not match."""


def _require(condition: bool, message: str, *, error_type: type[ValueError] = TimelineInputError) -> None:
    if not condition:
        raise error_type(message)


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
        raise TimelineInputError(f"cannot read JSON input: {source}") from exc
    _require(isinstance(value, dict), f"JSON object required: {source}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Counter):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def protocol() -> dict[str, Any]:
    """Read and validate the byte-frozen read-only replay protocol."""
    _require(PROTOCOL.is_file(), f"timeline protocol missing: {PROTOCOL}")
    _require(PROTOCOL_SHA != "TO_BE_FROZEN", "timeline protocol SHA is not frozen")
    _require(_sha(PROTOCOL) == PROTOCOL_SHA, "timeline protocol changed after freeze")
    try:
        value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TimelineInputError("timeline protocol cannot be read") from exc
    _require(isinstance(value, dict), "timeline protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "timeline protocol id changed")
    _require(
        value.get("status") == "PREREGISTERED_READ_ONLY_REPLAY_DIAGNOSTIC",
        "timeline protocol status changed",
    )

    bindings = value.get("bindings")
    _require(isinstance(bindings, Mapping), "timeline bindings are missing")
    expected_bindings = {
        "q3_stage": "V4_2_P3_V3_EARLY_STOP_Q3_V1",
        "q3_manifest_sha256": Q3_MANIFEST_SHA,
        "q3_protocol_sha256": Q3_PROTOCOL_SHA,
        "action_tape_sha256": ACTION_TAPE_SHA,
        "trained_checkpoint_sha256": TRAINED_CHECKPOINT_SHA,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
        "training_stage": "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V3",
        "seed": SEED,
        "scenario": SCENARIO,
        "controller": CONTROLLER,
        "horizon_seconds": int(HORIZON_SECONDS),
        "expected_steps": EXPECTED_STEPS,
        "total_requested_uavs": TOTAL_REQUESTED_UAVS,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"timeline binding changed: {key}")

    source_section = value.get("source_hashes")
    _require(source_section == SOURCE_EXPECTED, "timeline source hash binding changed")
    scenario = value.get("scenario")
    _require(isinstance(scenario, Mapping), "timeline scenario is missing")
    for key, expected in {
        "name": SCENARIO,
        "seed": SEED,
        "deterministic": True,
        "action_schema_version": "low_action_schema_v4_2_hierarchical",
        "runtime_proposal_shortlist_k": 3,
        "reset_mode": "warm_start_rings",
        "warm_start_uavs": 12,
        "warm_start_steady_motion": True,
        "full_horizon_seconds": int(HORIZON_SECONDS),
        "total_requested_uavs": TOTAL_REQUESTED_UAVS,
        "no_policy_optimizer": True,
        "no_teacher": True,
        "no_labels": True,
        "no_updates": True,
        "low_level_priority_time_weight": 1.0,
        "low_level_soc_urgency_time_weight": 2.0,
        "low_level_soc_urgency_threshold": LOW_SOC_THRESHOLD,
        "urgency_reward_config_exact": True,
    }.items():
        _require(scenario.get(key) == expected, f"timeline scenario changed: {key}")

    replay = value.get("replay")
    _require(isinstance(replay, Mapping), "timeline replay section is missing")
    for key in (
        "verify_reset_state_hash",
        "verify_reset_recipe_hash",
        "verify_environment_config_hash",
        "verify_observation_hash",
        "verify_requested_category",
        "verify_runtime_shortlist",
        "verify_executed_mapped_action",
        "verify_post_state_hash",
        "replay_runtime_shortlists",
    ):
        _require(replay.get(key) is True, f"timeline replay gate changed: {key}")
    _require(replay.get("exact_steps_required") == EXPECTED_STEPS,
             "timeline replay step count changed")

    expected_bins = [
        {"name": "0-180", "start": 0, "end": 180},
        {"name": "180-380", "start": 180, "end": 380},
        {"name": "380-600", "start": 380, "end": 600},
        {"name": "600-800", "start": 600, "end": 800},
        {"name": "800-960", "start": 800, "end": 960},
        {"name": "960-1080", "start": 960, "end": 1080},
    ]
    _require(value.get("time_bins") == expected_bins, "timeline bins changed")

    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "timeline authorization is missing")
    for key, expected in {
        "diagnostic_only": True,
        "PASS_used_for_qualification": False,
        "checkpoint_promotable": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "no_seed_reservation": True,
        "no_seed_consumption": True,
    }.items():
        _require(authorization.get(key) == expected, f"timeline authorization changed: {key}")
    return value


def source_hashes() -> dict[str, str]:
    """Hash every source file that can change the exact replay semantics."""
    hashes = {path: _sha(path) for path in SOURCE_EXPECTED}
    _require(hashes == SOURCE_EXPECTED, "timeline source hash mismatch")
    return hashes


def _validate_q3_manifest(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.is_file(), f"Q3 manifest missing: {path}")
    _require(_sha(path) == p["bindings"]["q3_manifest_sha256"],
             "Q3 manifest SHA mismatch")
    report = _read_json(path)
    for key, expected in {
        "stage": p["bindings"]["q3_stage"],
        "protocol_sha256": p["bindings"]["q3_protocol_sha256"],
        "PASS": False,
        "P3_QUALIFIED": False,
        "checkpoint_promotable": False,
        "P4_ALLOWED": False,
        "fresh_q3": True,
        "exposed_seed_diagnostic_only": True,
        "trained_checkpoint_sha256": p["bindings"]["trained_checkpoint_sha256"],
    }.items():
        _require(report.get(key) == expected, f"Q3 manifest field changed: {key}")
    records = report.get("records")
    _require(isinstance(records, list), "Q3 manifest records are missing")
    matches = [
        item for item in records
        if isinstance(item, Mapping)
        and item.get("seed") == SEED
        and item.get("scenario") == SCENARIO
        and item.get("controller") == CONTROLLER
    ]
    _require(len(matches) == 1, "Q3 trained dense24 record is not unique")
    result = matches[0].get("result")
    _require(isinstance(result, Mapping), "Q3 selected result is missing")
    summary = result.get("summary")
    _require(isinstance(summary, Mapping), "Q3 selected summary is missing")
    _require(summary.get("seed") == SEED, "Q3 selected summary seed changed")
    _require(summary.get("steps") == EXPECTED_STEPS, "Q3 selected step count changed")
    _require(summary.get("action_tape"), "Q3 selected action tape path is missing")
    _require(result.get("completion_report_only") is True,
             "Q3 completion mode changed")
    return report


def _validate_action_tape(
    path: Path, p: Mapping[str, Any], q3_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    _require(path.is_file(), f"action tape missing: {path}")
    _require(_sha(path) == p["bindings"]["action_tape_sha256"],
             "action tape SHA mismatch")
    payload = _read_json(path)
    _require(payload.get("episode_seed") == SEED, "action tape seed changed")
    _require(payload.get("schema_version") == "A5V2-L1-S4-SafetyTrace-v4",
             "action tape schema changed")
    rows = payload.get("action_tape")
    _require(isinstance(rows, list), "full action tape is missing")
    _require(len(rows) == EXPECTED_STEPS, "full action tape length changed")
    _require(len(payload.get("records", [])) == 120,
             "bounded diagnostic record window changed")
    _require(payload.get("reset_state_hash"), "reset state hash missing")
    _require(payload.get("environment_config_hash"), "environment config hash missing")
    _require(payload.get("reset_recipe_hash"), "reset recipe hash missing")
    _require(isinstance(payload.get("reset_recipe"), Mapping),
             "reset recipe missing")
    _require(payload.get("dump_reason") == "student_only_v2_1",
             "action tape dump reason changed")

    q3_rows = q3_manifest["records"]
    selected = next(
        item for item in q3_rows
        if item.get("seed") == SEED
        and item.get("scenario") == SCENARIO
        and item.get("controller") == CONTROLLER
    )
    q3_tape_ref = Path(str(selected["result"]["summary"]["action_tape"]))
    _require(q3_tape_ref.name == path.name, "action tape path is not the Q3 selected tape")

    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"action tape row {index} is invalid")
        _require(row.get("step") == index, f"action tape step index mismatch: {index}")
        _require(float(row.get("time_before")) == float(index),
                 f"action tape time_before mismatch: {index}")
        _require(float(row.get("time_after")) == float(index + 1),
                 f"action tape time_after mismatch: {index}")
        _require(row.get("action_schema_version") ==
                 "low_action_schema_v4_2_hierarchical",
                 f"action schema mismatch at step {index}")
        _require(isinstance(row.get("a_requested"), Mapping),
                 f"requested actions missing at step {index}")
        _require(isinstance(row.get("runtime_proposal_shortlists"), Mapping),
                 f"runtime shortlists missing at step {index}")
        _require(isinstance(row.get("post_state_hash"), str),
                 f"post-state hash missing at step {index}")
        for key in ("teacher_joint", "teacher_gate", "teacher_proposal"):
            value = row.get(key, {})
            _require(all(item is None for item in value.values()),
                     f"teacher data present at step {index}")
        _require("safety_event" not in row,
                 f"safety event present at step {index}")
    return payload


def _build_config(p: Mapping[str, Any]):
    training = training_protocol()
    config = scenario_config(SCENARIO, training)
    checks = {
        "dt": 1.0,
        "max_simulation_seconds": HORIZON_SECONDS,
        "total_requested_uavs": TOTAL_REQUESTED_UAVS,
        "runtime_proposal_shortlist_k": 3,
        "teacher_enabled": False,
        "teacher_release_label_enabled": False,
        "low_level_priority_time_weight": REWARD_CONFIG["low_level_priority_time_weight"],
        "low_level_soc_urgency_time_weight": REWARD_CONFIG["low_level_soc_urgency_time_weight"],
        "low_level_soc_urgency_threshold": REWARD_CONFIG["low_level_soc_urgency_threshold"],
        "reset_mode": "warm_start_rings",
        "warm_start_uavs": 12,
        "warm_start_steady_motion": True,
    }
    for key, expected in checks.items():
        _require(getattr(config, key) == expected, f"scenario config changed: {key}")
    return config


def _gate_is_release(value: Any) -> bool:
    return value == 1 or value == "RELEASE" or value is True


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def _phase_counts(env: TerminalParallelEnv) -> dict[str, int]:
    return _counter_dict(Counter(uav.phase.value for uav in env.active.values()))


def _lz_snapshot(env: TerminalParallelEnv) -> dict[str, Any]:
    statuses = Counter(str(zone.status.value) for zone in env.lz_manager.zones)
    return {
        "total": len(env.lz_manager.zones),
        "status_counts": _counter_dict(statuses),
        "free": int(statuses.get("free", 0)),
        "reserved": int(sum(zone.reserved_by is not None for zone in env.lz_manager.zones)),
        "occupied": int(sum(zone.occupied_by is not None for zone in env.lz_manager.zones)),
    }


def _initial_soc(uav: Any) -> float:
    return float((uav.energy_remaining + uav.energy_consumed) / uav.energy_capacity)


def _release_telemetry_snapshot(
    env: TerminalParallelEnv, decisions: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for uid, row in sorted(env.last_release_telemetry.items(), key=lambda item: int(item[0])):
        result[str(uid)] = {
            "decision": decisions.get(str(uid), env.last_release_decisions.get(int(uid))),
            "interface": row.get("interface"),
            "primary_blocker": row.get("primary_blocker"),
            "secondary_constraints": list(row.get("secondary_constraints", ())),
            "attempt_count": row.get("attempt_count"),
            "retry_count": row.get("retry_count"),
            "runtime_shortlist_size": row.get("runtime_shortlist_size"),
            "runtime_shortlist_candidates_checked": row.get(
                "runtime_shortlist_candidates_checked"
            ),
            "runtime_shortlist_rank": row.get("runtime_shortlist_rank"),
            "decision_history_tail": list(row.get("decision_history", ()))[-3:],
        }
    return result


def _empty_uid_stat(uid: int) -> dict[str, Any]:
    return {
        "uid": int(uid),
        "requested_release_count": 0,
        "approved_release_count": 0,
        "first_approved_release_time": None,
        "last_approved_release_time": None,
        "last_touchdown_time": None,
        "phase_transitions": [],
        "zero_approval_periods": [],
        "_open_zero_approval": None,
        "_last_phase": None,
    }


def _update_uid_stats(
    uid_stats: dict[int, dict[str, Any]],
    env: TerminalParallelEnv,
    row: Mapping[str, Any],
    decisions: Mapping[str, Any],
    events: Sequence[Sequence[Any]],
) -> None:
    time_before = float(row["time_before"])
    requested = row.get("requested_gate", {})
    for uid_text, gate in requested.items():
        uid = int(uid_text)
        stat = uid_stats.setdefault(uid, _empty_uid_stat(uid))
        if not _gate_is_release(gate):
            continue
        decision = decisions.get(str(uid))
        stat["requested_release_count"] += 1
        if decision == "approved":
            stat["approved_release_count"] += 1
            if stat["first_approved_release_time"] is None:
                stat["first_approved_release_time"] = time_before
            stat["last_approved_release_time"] = time_before
            if stat["_open_zero_approval"] is not None:
                period = stat["_open_zero_approval"]
                period["end_time"] = time_before
                period["end_step"] = int(row["step"])
                stat["zero_approval_periods"].append(period)
                stat["_open_zero_approval"] = None
        else:
            if stat["_open_zero_approval"] is None:
                stat["_open_zero_approval"] = {
                    "start_time": time_before,
                    "start_step": int(row["step"]),
                    "end_time": None,
                    "end_step": None,
                    "steps": 0,
                }
            stat["_open_zero_approval"]["steps"] += 1

    for event in events:
        if len(event) < 3:
            continue
        kind, uid_value, event_time = event[0], int(event[1]), float(event[2])
        stat = uid_stats.setdefault(uid_value, _empty_uid_stat(uid_value))
        if kind == "touchdown":
            stat["last_touchdown_time"] = event_time

    for uid, uav in env.active.items():
        stat = uid_stats.setdefault(int(uid), _empty_uid_stat(int(uid)))
        phase = uav.phase.value
        previous = stat["_last_phase"]
        if previous is not None and previous != phase:
            stat["phase_transitions"].append({
                "time": float(env.now),
                "from": previous,
                "to": phase,
            })
        stat["_last_phase"] = phase


def _finalize_uid_stats(uid_stats: Mapping[int, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for uid in sorted(uid_stats):
        stat = dict(uid_stats[uid])
        open_period = stat.pop("_open_zero_approval")
        if open_period is not None:
            stat["zero_approval_periods"] = list(stat["zero_approval_periods"]) + [open_period]
        stat.pop("_last_phase", None)
        stat["first_zero_approval_period"] = (
            stat["zero_approval_periods"][0]
            if stat["zero_approval_periods"] else None
        )
        result.append(stat)
    return result


def _state_snapshot(
    env: TerminalParallelEnv,
    row: Mapping[str, Any],
    info: Mapping[str, Any],
    cumulative: Mapping[str, Any],
    new_events: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    # last_release_decisions persists across steps; info is the authoritative
    # per-step decision set, including UAVs that departed during this step.
    decisions = {str(uid): decision for uid, decision in info["release_decisions"].items()}
    requested_gate = row.get("requested_gate", {})
    requested_release = sum(_gate_is_release(value) for value in requested_gate.values())
    requested_wait = len(requested_gate) - requested_release
    decision_counts = Counter(str(value) for value in decisions.values())
    traces = info.get("action_traces", {})
    shield_uids = sorted(
        int(uid) for uid, trace in traces.items()
        if bool(trace.get("shield_used")) or trace.get("shield_applied_action") is not None
    )
    active = sorted(env.active)
    step = int(row["step"])
    time_before = float(row["time_before"])
    time_after = float(env.now)
    primary = Counter()
    secondary = Counter()
    for item in env.last_release_telemetry.values():
        blocker = item.get("primary_blocker")
        if blocker:
            primary[str(blocker)] += 1
        secondary.update(str(value) for value in item.get("secondary_constraints", ()))

    snapshot = {
        "step": step,
        "time_before": time_before,
        "time_after": time_after,
        "active_count": len(active),
        "active_uids": active,
        "phase_counts": _phase_counts(env),
        "completed": len(env.completed_uavs),
        "arrivals_generated": int(env.arrivals.generated),
        "queue_size": len(env.admission.queue.items),
        "landing_zones": _lz_snapshot(env),
        "new_events": _jsonable(list(new_events)),
        "requested_release": int(requested_release),
        "requested_wait": int(requested_wait),
        "release_decision_counts": _counter_dict(decision_counts),
        "release_approved": int(decision_counts.get("approved", 0)),
        "release_safety_reject": int(decision_counts.get("safety_reject", 0)),
        "release_safety_wait": int(decision_counts.get("safety_wait", 0)),
        "release_telemetry": _release_telemetry_snapshot(env, decisions),
        "primary_blockers": _counter_dict(primary),
        "secondary_constraints": _counter_dict(secondary),
        "shield_count": len(shield_uids),
        "shield_uids": shield_uids,
        "conflict_events": _jsonable(info.get("conflict_events", [])),
        "telemetry": _jsonable(dict(env.telemetry)),
        "cumulative": _jsonable(cumulative),
    }
    return snapshot


def _update_cumulative(cumulative: dict[str, Any], snapshot: Mapping[str, Any]) -> None:
    cumulative["steps"] += 1
    for key in (
        "requested_release",
        "requested_wait",
        "release_approved",
        "release_safety_reject",
        "release_safety_wait",
        "shield_count",
    ):
        cumulative[key] += int(snapshot[key])
    for key in ("release_decision_counts", "primary_blockers", "secondary_constraints"):
        counter = cumulative[key]
        counter.update(snapshot[key])
    cumulative["conflict_events"] += len(snapshot["conflict_events"])


def _time_series_snapshot(snapshot: Mapping[str, Any], cumulative: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "time": float(snapshot["time_after"]),
        "step": int(snapshot["step"]),
        "active_count": int(snapshot["active_count"]),
        "phase_counts": dict(snapshot["phase_counts"]),
        "completed": int(snapshot["completed"]),
        "arrivals_generated": int(snapshot["arrivals_generated"]),
        "queue_size": int(snapshot["queue_size"]),
        "landing_zones": dict(snapshot["landing_zones"]),
        "cumulative": _jsonable(cumulative),
    }


def _aggregate_bins(
    timeline: Sequence[Mapping[str, Any]], bins: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for spec in bins:
        start, end = float(spec["start"]), float(spec["end"])
        rows = [
            row for row in timeline
            if start <= float(row["time_before"]) < end
        ]
        decisions = Counter()
        primary = Counter()
        secondary = Counter()
        phases = Counter()
        for row in rows:
            decisions.update(row["release_decision_counts"])
            primary.update(row["primary_blockers"])
            secondary.update(row["secondary_constraints"])
            phases.update(row["phase_counts"])
        aggregates.append({
            "name": spec["name"],
            "start": int(start),
            "end": int(end),
            "steps": len(rows),
            "active_peak": max((int(row["active_count"]) for row in rows), default=0),
            "active_mean": (
                None if not rows else float(np.mean([row["active_count"] for row in rows]))
            ),
            "completed_start": None if not rows else int(rows[0]["completed"]),
            "completed_end": None if not rows else int(rows[-1]["completed"]),
            "arrivals_generated_end": None if not rows else int(rows[-1]["arrivals_generated"]),
            "queue_peak": max((int(row["queue_size"]) for row in rows), default=0),
            "queue_end": None if not rows else int(rows[-1]["queue_size"]),
            "requested_release": sum(int(row["requested_release"]) for row in rows),
            "requested_wait": sum(int(row["requested_wait"]) for row in rows),
            "release_approved": sum(int(row["release_approved"]) for row in rows),
            "release_safety_reject": sum(int(row["release_safety_reject"]) for row in rows),
            "release_safety_wait": sum(int(row["release_safety_wait"]) for row in rows),
            "shield_count": sum(int(row["shield_count"]) for row in rows),
            "decision_counts": _counter_dict(decisions),
            "primary_blockers": _counter_dict(primary),
            "secondary_constraints": _counter_dict(secondary),
            "phase_observation_counts": _counter_dict(phases),
            "landing_zones_end": None if not rows else rows[-1]["landing_zones"],
        })
    return aggregates


def _end_state(env: TerminalParallelEnv) -> dict[str, Any]:
    outstanding = []
    for uid, uav in sorted(env.active.items()):
        outstanding.append({
            "uid": int(uid),
            "arrival_id": int(uav.arrival_id),
            "base_priority": int(uav.base_priority),
            "initial_soc": _initial_soc(uav),
            "soc": float(uav.soc),
            "phase": uav.phase.value,
            "ring_role": None if uav.ring_role is None else uav.ring_role.value,
            "committed": bool(uav.committed_plan is not None),
            "reservation_ids": (
                [] if uav.committed_plan is None
                else list(uav.committed_plan.reservation_ids)
            ),
            "last_release_decision": env.last_release_decisions.get(int(uid)),
        })
    queue = []
    for item in env.admission.queue.items:
        request = item.request
        queue.append({
            "arrival_id": int(request.arrival_id),
            "base_priority": int(request.base_priority),
            "initial_soc": float(request.initial_soc),
            "soc": float(item.soc),
        })
    return {
        "time": float(env.now),
        "step_count": int(env.step_count),
        "outstanding_active": outstanding,
        "outstanding_queue": queue,
        "active_count": len(env.active),
        "completed": len(env.completed_uavs),
        "arrivals_generated": int(env.arrivals.generated),
        "queue_size": len(env.admission.queue.items),
        "landing_zones": _lz_snapshot(env),
        "telemetry": _jsonable(dict(env.telemetry)),
    }


def _replay_timeline(
    config: Any, payload: Mapping[str, Any], p: Mapping[str, Any]
) -> dict[str, Any]:
    rows = payload["action_tape"]
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(SEED)
    recipe = payload.get("reset_recipe", {"kind": "generic_reset", "steps": []})
    recipe_match = _digest(recipe) == payload.get("reset_recipe_hash")
    base_match = (
        True
        if recipe.get("base_reset_state_hash") is None
        else _digest(SafetyTraceRecorder._state(env)) == recipe["base_reset_state_hash"]
    )
    observations = SafetyTraceRecorder._apply_reset_recipe(env, recipe)
    reset_match = _digest(SafetyTraceRecorder._state(env)) == payload["reset_state_hash"]
    config_match = _digest(vars(env.config)) == payload["environment_config_hash"]
    _require(recipe_match and base_match and reset_match and config_match,
             "replay preflight hash mismatch", error_type=TimelineReplayError)

    timeline: list[dict[str, Any]] = []
    time_series: list[dict[str, Any]] = []
    cumulative: dict[str, Any] = {
        "steps": 0,
        "requested_release": 0,
        "requested_wait": 0,
        "release_approved": 0,
        "release_safety_reject": 0,
        "release_safety_wait": 0,
        "shield_count": 0,
        "release_decision_counts": Counter(),
        "primary_blockers": Counter(),
        "secondary_constraints": Counter(),
        "conflict_events": 0,
    }
    uid_stats: dict[int, dict[str, Any]] = {}
    previous_event_count = 0

    for expected_step, row in enumerate(rows):
        _require(env.step_count == expected_step,
                 f"environment step mismatch before step {expected_step}",
                 error_type=TimelineReplayError)
        _require(float(env.now) == float(row["time_before"]),
                 f"environment time mismatch before step {expected_step}",
                 error_type=TimelineReplayError)
        observation_match = _digest({
            key: value for key, value in sorted(observations.items())
        }) == row["observation_hash"]
        _require(observation_match, f"observation hash mismatch at step {expected_step}",
                 error_type=TimelineReplayError)
        actions = {
            str(uid): np.asarray(action, dtype=np.float32)
            for uid, action in row["a_requested"].items()
        }
        for uid, action in actions.items():
            try:
                decoded = hierarchical_from_normalized_v4_2(action)
            except ValueError as exc:
                raise TimelineReplayError(
                    f"requested action decode failed at step {expected_step}, uid {uid}"
                ) from exc
            _require(
                decoded.gate == row.get("requested_gate", {}).get(uid)
                and decoded.proposal == row.get("requested_proposal", {}).get(uid),
                f"requested category mismatch at step {expected_step}, uid {uid}",
                error_type=TimelineReplayError,
            )
        shortlists = {
            str(uid): tuple(np.asarray(action, dtype=np.float32) for action in values)
            for uid, values in row.get("runtime_proposal_shortlists", {}).items()
        }
        try:
            following, _, _, _, info = env.step(actions, shortlists)
        except Exception as exc:
            raise TimelineReplayError(
                f"environment replay failed at step {expected_step}: {type(exc).__name__}: {exc}"
            ) from exc

        post_hash_match = _digest(SafetyTraceRecorder._state(env)) == row["post_state_hash"]
        _require(post_hash_match, f"post-state hash mismatch at step {expected_step}",
                 error_type=TimelineReplayError)
        _require(float(env.now) == float(row["time_after"]),
                 f"environment time mismatch after step {expected_step}",
                 error_type=TimelineReplayError)
        actual_executed = _json({
            str(uid): trace.get("execution_applied_action")
            for uid, trace in info.get("action_traces", {}).items()
        })
        _require(actual_executed == row.get("executed_mapped_action"),
                 f"executed action mismatch at step {expected_step}",
                 error_type=TimelineReplayError)
        actual_shortlists = _json({
            str(uid): {
                "used": bool(trace.get("runtime_shortlist_used")),
                "rank": trace.get("runtime_shortlist_rank"),
                "candidates_checked": int(trace.get("runtime_shortlist_candidates_checked", 0)),
            }
            for uid, trace in info.get("action_traces", {}).items()
        })
        _require(actual_shortlists == row.get("runtime_shortlist_result"),
                 f"runtime shortlist mismatch at step {expected_step}",
                 error_type=TimelineReplayError)

        new_events = env.events[previous_event_count:]
        previous_event_count = len(env.events)
        decisions = info["release_decisions"]
        snapshot = _state_snapshot(env, row, info, cumulative, new_events)
        _update_cumulative(cumulative, snapshot)
        snapshot["cumulative"] = _jsonable(cumulative)
        timeline.append(snapshot)
        _update_uid_stats(uid_stats, env, row, decisions, new_events)
        if int(round(snapshot["time_after"])) % 60 == 0:
            time_series.append(_time_series_snapshot(snapshot, cumulative))
        observations = following

    _require(len(timeline) == EXPECTED_STEPS, "replay did not consume all steps",
             error_type=TimelineReplayError)
    _require(env.step_count == EXPECTED_STEPS and float(env.now) == HORIZON_SECONDS,
             "replay horizon mismatch", error_type=TimelineReplayError)
    finalized_uid_stats = _finalize_uid_stats(uid_stats)
    bins = _aggregate_bins(timeline, p["time_bins"])
    cumulative_json = _jsonable(cumulative)
    return {
        "exact": True,
        "steps_replayed": len(timeline),
        "steps_expected": EXPECTED_STEPS,
        "reset_state_match": bool(reset_match),
        "reset_recipe_match": bool(recipe_match),
        "base_reset_state_match": bool(base_match),
        "config_match": bool(config_match),
        "timeline": timeline,
        "time_bins": bins,
        "time_series_60s": time_series,
        "uid_stats": finalized_uid_stats,
        "end_state": _end_state(env),
        "cumulative": cumulative_json,
    }


def run(*, q3_manifest: Path, action_tape: Path, output: Path) -> dict[str, Any]:
    """Validate inputs and write the exact replay timeline only on success."""
    p = protocol()
    _require(not output.exists(), f"diagnostic output already exists: {output}")
    q3 = _validate_q3_manifest(q3_manifest, p)
    tape = _validate_action_tape(action_tape, p, q3)
    config = _build_config(p)
    hashes = source_hashes()

    replay = _replay_timeline(config, tape, p)
    _require(replay["exact"] and replay["steps_replayed"] == EXPECTED_STEPS,
             "exact replay did not pass", error_type=TimelineReplayError)
    output.mkdir(parents=True, exist_ok=False)
    timeline_path = output / "timeline.json"
    manifest_path = output / "manifest.json"
    _write(timeline_path, {
        "stage": STAGE,
        "diagnostic_only": True,
        "seed": SEED,
        "scenario": SCENARIO,
        "controller": CONTROLLER,
        "action_tape_sha256": ACTION_TAPE_SHA,
        "timeline": replay["timeline"],
        "time_bins": replay["time_bins"],
        "time_series_60s": replay["time_series_60s"],
        "uid_stats": replay["uid_stats"],
        "end_state": replay["end_state"],
        "cumulative": replay["cumulative"],
    })
    result = {
        "stage": STAGE,
        "PASS": True,
        "diagnostic_only": True,
        "exposed_seed_diagnostic_only": True,
        "q3_manifest_sha256": Q3_MANIFEST_SHA,
        "q3_protocol_sha256": Q3_PROTOCOL_SHA,
        "action_tape_sha256": ACTION_TAPE_SHA,
        "trained_checkpoint_sha256": TRAINED_CHECKPOINT_SHA,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
        "seed": SEED,
        "scenario": SCENARIO,
        "controller": CONTROLLER,
        "horizon_seconds": HORIZON_SECONDS,
        "steps_replayed": replay["steps_replayed"],
        "steps_expected": replay["steps_expected"],
        "exact_replay": replay["exact"],
        "reset_state_match": replay["reset_state_match"],
        "reset_recipe_match": replay["reset_recipe_match"],
        "base_reset_state_match": replay["base_reset_state_match"],
        "config_match": replay["config_match"],
        "timeline_path": str(timeline_path),
        "checkpoint_promotable": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "no_policy_optimizer": True,
        "no_teacher": True,
        "no_labels": True,
        "no_updates": True,
        "no_seed_reservation": True,
        "no_seed_consumption": True,
        "urgency_reward_config": dict(REWARD_CONFIG),
        "source_hashes": hashes,
        "time_bins": replay["time_bins"],
        "time_series_60s": replay["time_series_60s"],
        "uid_stats": replay["uid_stats"],
        "end_state": replay["end_state"],
        "cumulative": replay["cumulative"],
    }
    _write(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q3-manifest", type=Path, required=True)
    parser.add_argument("--action-tape", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        q3_manifest=args.q3_manifest,
        action_tape=args.action_tape,
        output=args.output,
    )
    print(json.dumps({
        "stage": result["stage"],
        "PASS": result["PASS"],
        "diagnostic_only": result["diagnostic_only"],
        "steps_replayed": result["steps_replayed"],
        "output": result["timeline_path"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
