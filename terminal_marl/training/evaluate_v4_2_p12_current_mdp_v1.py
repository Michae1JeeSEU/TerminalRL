"""Fresh read-only P1/P2 checkpoint requalification under the current MDP.

The formal protocol is intentionally descriptive.  It loads two frozen
checkpoints, runs each beside the nominal rule controller, replays every tape,
and reports lifecycle and paired effectiveness metrics.  There is no update,
Teacher, label, or checkpoint-promotion path in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.run_v4_2_curriculum_pilot import (
    config_for as curriculum_config_for,
    episode as curriculum_episode,
)
from terminal_marl.training.run_v4_2_ppo_block2 import (
    replay_geometry,
    restore as restore_p1,
)
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.student_rollout_v4_2 import rollout as student_rollout


PROTOCOL = Path("terminal_marl/configs/low_v4_2/p12_current_mdp_requalification_v1.yaml")
PROTOCOL_SHA = "44724cf715be202dbbccc83e802ea27cba16ae78fb6f1052a7a0a7c758a2d56f"
STAGE = "V4_2_P12_CURRENT_MDP_REQUALIFICATION_V1"

P1_CHECKPOINT_SHA = "6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f"
P2_CHECKPOINT_SHA = "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
P1_CHECKPOINT_PATH = Path(
    "D:/7_codespace/training_outputs/terminal/"
    "ppo_block2_transfer_repair_20260917/train/checkpoint_24.pt"
)
P2_CHECKPOINT_PATH = Path(
    "D:/7_codespace/training_outputs/terminal/"
    "p2_gate_repair_v3_anchored_20260919/gate_repair_v3_qualification_candidate.pt"
)

P1_FORMAL_SEEDS = (834101, 834102, 834103, 834104)
P1_COUNTS = (3, 6, 3, 6)
P2_FORMAL_SEEDS = (834201, 834202, 834203, 834204, 834205, 834206)
P2_COUNTS = (6, 8, 12, 6, 8, 12)
FORMAL_SEEDS = P1_FORMAL_SEEDS + P2_FORMAL_SEEDS
SMOKE_SEED = 832001
SMOKE_COUNT = 6
SMOKE_MAX_STEPS = 20
SHIELD_ACTOR_FRACTION_MAX = 0.05
P2_SHORTLIST_K = 3
P1_SHORTLIST_K = 1

REQUIRED_GATE_NAMES = (
    "exact_replay",
    "safety_counts_valid",
    "safety_zero",
    "completed_equals_requests",
    "terminal",
    "shield_burden",
    "actor_decisions",
    "geometry_observed",
    "no_reverse_transfer",
    "no_inner_excursion",
    "student_only",
)


class CurrentMDPInputError(ValueError):
    """Raised when the frozen current-MDP inputs or records are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentMDPInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_sha = _sha256


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def protocol() -> dict[str, Any]:
    """Read and validate the byte-frozen P1/P2 protocol."""
    _require(PROTOCOL.is_file(), f"protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "P1/P2 protocol changed after freeze")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "P1/P2 protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "P1/P2 protocol id changed")
    bindings = value.get("bindings", {})
    expected_bindings = {
        "observation_schema": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": 88,
        "action_schema": "low_action_schema_v4_2_hierarchical",
        "deterministic": True,
        "natural_flow": True,
        "first_arrival_seconds": 10.0,
        "arrival_interval_seconds": 60.0,
        "fixed_initial_soc": 0.8,
        "energy_capacity": 1500.0,
        "lz_assignment_mode": "lowest_id",
        "release_arbitration_mode": "legacy_insertion",
        "final_queue_capacity": 1,
        "safe_distance": 80.0,
        "nmac_distance": 40.0,
        "collision_distance": 10.0,
        "teacher_enabled": False,
        "labels_collected": False,
        "updates": False,
        "fixed_geometry": True,
    }
    for key, expected in expected_bindings.items():
        actual = bindings.get(key)
        _require(actual == expected, f"protocol binding changed: {key}")
    checkpoints = value.get("checkpoints", {})
    for key, expected in (
        ("p1", P1_CHECKPOINT_SHA),
        ("p2", P2_CHECKPOINT_SHA),
    ):
        _require(checkpoints.get(key, {}).get("sha256") == expected,
                 f"checkpoint binding changed: {key}")
    formal = value.get("formal", {})
    _require(formal.get("p1", {}).get("seeds") == list(P1_FORMAL_SEEDS),
             "P1 formal seeds changed")
    _require(formal.get("p1", {}).get("counts") == list(P1_COUNTS),
             "P1 formal counts changed")
    _require(formal.get("p2", {}).get("seeds") == list(P2_FORMAL_SEEDS),
             "P2 formal seeds changed")
    _require(formal.get("p2", {}).get("counts") == list(P2_COUNTS),
             "P2 formal counts changed")
    smoke = value.get("smoke", {})
    _require(smoke.get("seed") == SMOKE_SEED, "smoke seed changed")
    _require(smoke.get("count") == SMOKE_COUNT, "smoke count changed")
    _require(smoke.get("max_steps") == SMOKE_MAX_STEPS, "smoke horizon changed")
    gates = value.get("gates", {})
    _require(float(gates.get("shield_actor_fraction_max")) == SHIELD_ACTOR_FRACTION_MAX,
             "shield gate changed")
    authorization = value.get("authorization", {})
    for key in ("PPO_ALLOWED", "P3_ALLOWED", "training_called", "checkpoint_saved"):
        _require(authorization.get(key) is False, f"authorization changed: {key}")
    _require(authorization.get("effectiveness_threshold") is None,
             "descriptive effectiveness protocol gained a threshold")
    return value


def config_for(n: int, *, shortlist_k: int = P1_SHORTLIST_K):
    """Build the current-MDP natural-flow config from the existing pilot base."""
    if int(n) < 1:
        raise ValueError("n must be positive")
    if shortlist_k not in (1, 3):
        raise ValueError("shortlist_k must be 1 or 3")
    schedule = tuple(10.0 + 60.0 * index for index in range(int(n)))
    base = curriculum_config_for(int(n))
    return replace(
        base,
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        teacher_release_label_enabled=False,
        scheduler_mode="execution",
        high_level_enabled=False,
        reset_mode="flow_from_empty",
        warm_start_uavs=0,
        total_requested_uavs=int(n),
        fixed_arrival_schedule=schedule,
        max_simulation_seconds=schedule[-1] + 900.0,
        initial_soc_mode="fixed",
        initial_soc=0.8,
        initial_soc_min=None,
        initial_soc_max=None,
        energy_capacity=1500.0,
        safe_distance=80.0,
        nmac_distance=40.0,
        collision_distance=10.0,
        lz_assignment_mode="lowest_id",
        release_arbitration_mode="legacy_insertion",
        final_queue_capacity=1,
        runtime_proposal_shortlist_k=int(shortlist_k),
    )


def _validate_checkpoint(path: Path, expected_sha: str, label: str) -> None:
    _require(path.is_file(), f"{label} checkpoint missing: {path}")
    _require(_sha256(path) == expected_sha, f"{label} checkpoint SHA mismatch: {path}")


def load_p1_checkpoint(path: Path):
    _validate_checkpoint(path, P1_CHECKPOINT_SHA, "P1")
    return restore_p1(path, {
        "initial_checkpoint_sha256": P1_CHECKPOINT_SHA,
        "learning_rate": 0.00003,
    })


def load_p2_checkpoint(path: Path):
    _validate_checkpoint(path, P2_CHECKPOINT_SHA, "P2")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, learning_rate=0.0003,
    )
    payload = load_checkpoint(path, policy, config_for(6, shortlist_k=P2_SHORTLIST_K), load_optimizer=False)
    for key, expected in {
        "stage": "V4_2_P2_GATE_RESIDUAL_ADAPTER_REPAIR_V3",
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "qualification_only": True,
        "checkpoint_role": "qualification_candidate",
        "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
    }.items():
        _require(payload.get(key) == expected, f"P2 checkpoint binding changed: {key}")
    _require(len(policy.optimizer.state) == 0, "P2 optimizer state was loaded")
    policy.model.eval()
    return policy


def _completed_values(env: TerminalParallelEnv) -> list[Any]:
    completed = env.completed_uavs
    if isinstance(completed, Mapping):
        return list(completed.values())
    return list(completed)


def _terminal(env: TerminalParallelEnv) -> bool:
    return bool(
        env.arrivals.exhausted()
        and not env.active
        and not env.admission.queue.items
        and not env.service_uavs
    )


def _lifecycle_metrics(env: TerminalParallelEnv) -> dict[str, Any]:
    completed = _completed_values(env)
    waits = [float(u.external_wait_time + u.internal_wait_time) for u in completed]
    request_to_touchdown = [
        float(u.touchdown_time - u.request_time)
        for u in completed
        if u.touchdown_time is not None and u.request_time is not None
    ]
    return {
        "completed": len(completed),
        "terminal": _terminal(env),
        "mean_wait_seconds": float(np.mean(waits)) if waits else None,
        "mean_request_to_touchdown_seconds": (
            float(np.mean(request_to_touchdown)) if request_to_touchdown else None
        ),
    }


def _replay_lifecycle(config, tape: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay once for lifecycle state while using the canonical geometry observer."""
    replay_exact, geometry = replay_geometry(config, tape)
    env = TerminalParallelEnv(config)
    replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    _require(bool(replay.get("exact")) == bool(replay_exact), "replay APIs disagreed")
    lifecycle = _lifecycle_metrics(env)
    lifecycle["replay_exact"] = bool(replay_exact)
    return lifecycle, geometry


def _tape_payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and isinstance(value.get("action_tape"), list),
             f"invalid action tape: {path}")
    return value


def _tape_execution_checks(path: Path, *, expected_shortlist_k: int | None) -> dict[str, Any]:
    payload = _tape_payload(path)
    rows = payload["action_tape"]
    actor_decisions = 0
    release_rows = 0
    shortlist_rows = 0
    shortlist_valid = True
    teacher_free = True
    for row in rows:
        requested_gate = row.get("requested_gate", {})
        shortlists = row.get("runtime_proposal_shortlists", {})
        teacher = row.get("teacher_joint", {})
        if isinstance(teacher, Mapping):
            teacher_free = teacher_free and all(value is None for value in teacher.values())
        if not isinstance(requested_gate, Mapping):
            shortlist_valid = False
            continue
        actor_decisions += len(requested_gate)
        for uid, gate in requested_gate.items():
            if int(gate) != 1:
                continue
            release_rows += 1
            if expected_shortlist_k is None:
                continue
            values = shortlists.get(str(uid), shortlists.get(uid, ()))
            shortlist_rows += int(bool(values))
            shortlist_valid = shortlist_valid and isinstance(values, list) and len(values) == expected_shortlist_k
            if isinstance(values, list):
                shortlist_valid = shortlist_valid and len({json.dumps(value, sort_keys=True) for value in values}) == len(values)
    return {
        "actor_decisions": int(actor_decisions),
        "release_rows": int(release_rows),
        "shortlist_rows": int(shortlist_rows),
        "shortlist_valid": bool(shortlist_valid),
        "teacher_free": bool(teacher_free),
        "shortlist_contract": bool(
            expected_shortlist_k is None
            or (shortlist_valid and (release_rows == 0 or shortlist_rows == release_rows))
        ),
    }


def _safety_counts(raw: Mapping[str, Any]) -> tuple[dict[str, int], bool, str | None]:
    """Normalize the simulator safety counters and fail closed on bad input."""
    value = raw.get("safety")
    if not isinstance(value, Mapping) or not value:
        return {}, False, "missing safety counter mapping"
    counters: dict[str, int] = {}
    invalid: str | None = None

    def visit(mapping: Mapping[str, Any], prefix: str = "") -> None:
        nonlocal invalid
        if not mapping:
            invalid = invalid or (prefix or "safety") + " is empty"
            return
        for key, item in mapping.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, Mapping):
                visit(item, name)
                continue
            if isinstance(item, bool) or not isinstance(item, Real):
                invalid = invalid or f"{name} is not a numeric counter"
                continue
            number = float(item)
            if not np.isfinite(number) or number < 0.0 or number != round(number):
                invalid = invalid or f"{name} is not a finite non-negative integer"
                continue
            counters[name] = int(number)

    visit(value)
    if raw.get("safety_error") is not None and "SafetyInvariantError" not in counters:
        counters["SafetyInvariantError"] = 1
    if not counters and invalid is None:
        invalid = "safety counter mapping contained no counters"
    if invalid is not None:
        return counters, False, invalid
    return counters, True, None


def _safety_zero(value: Mapping[str, Any]) -> bool:
    """Return true only for a present, valid, all-zero counter mapping."""
    if not isinstance(value, Mapping) or not value:
        return False
    for item in value.values():
        if isinstance(item, Mapping):
            if not _safety_zero(item):
                return False
            continue
        if isinstance(item, bool) or not isinstance(item, Real):
            return False
        number = float(item)
        if not np.isfinite(number) or number < 0.0 or number != round(number) or int(number) != 0:
            return False
    return True


def _record_from_raw(
    *, stage: str, controller: str, seed: int, count: int, config,
    raw: Mapping[str, Any], lifecycle: Mapping[str, Any], geometry: Mapping[str, Any],
    execution: Mapping[str, Any], shortlist_k: int | None,
) -> dict[str, Any]:
    safety, safety_counts_valid, safety_error = _safety_counts(raw)
    safety_zero = safety_counts_valid and _safety_zero(safety)
    shield = float(raw.get("shield", raw.get("counts", {}).get("shield", 0)))
    actor_decisions = int(execution.get("actor_decisions", raw.get("counts", {}).get("actor_decisions", 0)))
    shield_fraction = shield / max(1, actor_decisions)
    completed = int(lifecycle.get("completed", raw.get("completed", 0)))
    terminal = bool(lifecycle.get("terminal", raw.get("terminal", False)))
    geometry_observed = int(geometry.get("samples", 0)) > 0
    controller_actor_gate = controller == "rule" or actor_decisions > 0
    gates = {
        "exact_replay": bool(lifecycle.get("replay_exact", raw.get("replay_exact", False))),
        "safety_counts_valid": safety_counts_valid,
        "safety_zero": safety_zero,
        "completed_equals_requests": completed == int(count),
        "terminal": terminal,
        "shield_burden": shield_fraction <= SHIELD_ACTOR_FRACTION_MAX,
        "actor_decisions": controller_actor_gate,
        "geometry_observed": geometry_observed,
        "no_reverse_transfer": int(geometry.get("reverse_samples", 0)) == 0,
        "no_inner_excursion": float(geometry.get("max_inner_excursion_m", float("inf"))) <= 0.05,
        "student_only": bool(execution.get("teacher_free", True)) and not bool(raw.get("label_rows", 0)),
    }
    if shortlist_k is not None:
        gates["shortlist_contract"] = bool(execution.get("shortlist_contract", False))
    result = {
        "stage": stage,
        "controller": controller,
        "seed": int(seed),
        "count": int(count),
        "requests": int(config.total_requested_uavs),
        "seed_kind": "diagnostic" if seed == SMOKE_SEED else "formal",
        "completed": completed,
        "terminal": terminal,
        "seconds": float(raw.get("seconds", 0.0)),
        "mean_wait_seconds": lifecycle.get("mean_wait_seconds"),
        "mean_request_to_touchdown_seconds": lifecycle.get("mean_request_to_touchdown_seconds"),
        "shield": shield,
        "actor_decisions": actor_decisions,
        "shield_actor_fraction": float(shield_fraction),
        "safety": safety,
        "geometry": dict(geometry),
        "execution": dict(execution),
        "safety_counts_valid": bool(safety_counts_valid),
        "safety_error": safety_error,
        "action_tape": str(raw.get("action_tape")),
        "gates": gates,
        "all_gates": bool(all(gates.values())),
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
    }
    return result


def _run_pilot_episode(stage: str, controller: str, policy: Any, seed: int, count: int,
                       output: Path, *, max_steps: int | None) -> dict[str, Any]:
    config = config_for(count, shortlist_k=P1_SHORTLIST_K)
    raw = curriculum_episode(config, seed, output, policy, max_steps)
    lifecycle, geometry = _replay_lifecycle(config, Path(raw["action_tape"]))
    execution = _tape_execution_checks(Path(raw["action_tape"]), expected_shortlist_k=None)
    return _record_from_raw(
        stage=stage, controller=controller, seed=seed, count=count, config=config,
        raw=raw, lifecycle=lifecycle, geometry=geometry, execution=execution,
        shortlist_k=None,
    )


def _run_p2_student(policy: Any, seed: int, count: int, output: Path, *, max_steps: int | None) -> dict[str, Any]:
    config = config_for(count, shortlist_k=P2_SHORTLIST_K)
    horizon = max_steps or int(config.max_simulation_seconds / config.dt)
    _, raw = student_rollout(
        policy, config, seed=seed, max_steps=horizon, output=output,
        deterministic=True, collect_labels=False,
    )
    lifecycle, geometry = _replay_lifecycle(config, Path(raw["action_tape"]))
    execution = _tape_execution_checks(Path(raw["action_tape"]), expected_shortlist_k=P2_SHORTLIST_K)
    return _record_from_raw(
        stage="P2", controller="candidate", seed=seed, count=count, config=config,
        raw=raw, lifecycle=lifecycle, geometry=geometry, execution=execution,
        shortlist_k=P2_SHORTLIST_K,
    )


def _run_rule(stage: str, seed: int, count: int, output: Path, *, max_steps: int | None) -> dict[str, Any]:
    return _run_pilot_episode(stage, "rule", None, seed, count, output, max_steps=max_steps)


def _validate_record(record: Mapping[str, Any]) -> None:
    _require(isinstance(record.get("gates"), Mapping), "record gates missing")
    for key in REQUIRED_GATE_NAMES:
        _require(bool(record["gates"].get(key)), f"record gate failed: {key}")
    if "shortlist_contract" in record["gates"]:
        _require(bool(record["gates"]["shortlist_contract"]), "record gate failed: shortlist_contract")
    _require(_safety_zero(record.get("safety", {})), "record safety is nonzero")
    _require(bool(record.get("action_tape")), "record action tape missing")


def aggregate_records(records: Sequence[Mapping[str, Any]], stage: str | None = None) -> dict[str, Any]:
    """Aggregate complete paired records; incomplete evidence fails closed."""
    rows = [dict(item) for item in records]
    _require(rows, "cannot aggregate empty records")
    if stage is not None:
        _require(all(row.get("stage") == stage for row in rows), "mixed stage records")
    for row in rows:
        _validate_record(row)
    controllers = {str(row.get("controller")) for row in rows}
    _require(controllers == {"candidate", "rule"}, "paired candidate/rule records required")
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        seed = int(row["seed"])
        controller = str(row["controller"])
        _require(controller not in by_seed.setdefault(seed, {}), f"duplicate paired record: {seed}/{controller}")
        by_seed[seed][controller] = row
    _require(all(set(pair) == {"candidate", "rule"} for pair in by_seed.values()),
             "every seed requires candidate and rule")
    paired = []
    for seed in sorted(by_seed):
        candidate = by_seed[seed]["candidate"]
        rule = by_seed[seed]["rule"]
        _require(candidate["count"] == rule["count"], f"paired count mismatch: {seed}")
        # The strict record validation above establishes completion for both
        # controllers.  Keep the explicit conditional so this report format
        # never emits a paired metric for an incomplete episode.
        if candidate.get("completed") == candidate.get("requests") and rule.get("completed") == rule.get("requests"):
            paired.append({
                "seed": seed,
                "count": int(candidate["count"]),
                "candidate_mean_wait_seconds": candidate.get("mean_wait_seconds"),
                "rule_mean_wait_seconds": rule.get("mean_wait_seconds"),
                "candidate_mean_request_to_touchdown_seconds": candidate.get("mean_request_to_touchdown_seconds"),
                "rule_mean_request_to_touchdown_seconds": rule.get("mean_request_to_touchdown_seconds"),
            })
    _require(len(paired) == len(by_seed), "paired metrics require complete candidate and rule episodes")

    def _mean(name: str, controller: str) -> float | None:
        values = [float(row[name]) for row in rows if row["controller"] == controller and row.get(name) is not None]
        return float(np.mean(values)) if values else None

    def _paired_mean(candidate_name: str, rule_name: str) -> dict[str, float | None]:
        candidate_values = [float(row[candidate_name]) for row in paired if row.get(candidate_name) is not None]
        rule_values = [float(row[rule_name]) for row in paired if row.get(rule_name) is not None]
        candidate_mean = float(np.mean(candidate_values)) if candidate_values else None
        rule_mean = float(np.mean(rule_values)) if rule_values else None
        delta = None if candidate_mean is None or rule_mean is None else candidate_mean - rule_mean
        return {"candidate": candidate_mean, "rule": rule_mean, "candidate_minus_rule": delta}

    return {
        "stage": stage,
        "records": len(rows),
        "seeds": sorted(by_seed),
        "controllers": ["candidate", "rule"],
        "candidate": {
            "mean_wait_seconds": _mean("mean_wait_seconds", "candidate"),
            "mean_request_to_touchdown_seconds": _mean("mean_request_to_touchdown_seconds", "candidate"),
            "completion_fraction": 1.0,
        },
        "rule": {
            "mean_wait_seconds": _mean("mean_wait_seconds", "rule"),
            "mean_request_to_touchdown_seconds": _mean("mean_request_to_touchdown_seconds", "rule"),
            "completion_fraction": 1.0,
        },
        "paired_complete_seeds": [int(row["seed"]) for row in paired],
        "paired_mean_wait_seconds": _paired_mean(
            "candidate_mean_wait_seconds", "rule_mean_wait_seconds"
        ),
        "paired_mean_request_to_touchdown_seconds": _paired_mean(
            "candidate_mean_request_to_touchdown_seconds",
            "rule_mean_request_to_touchdown_seconds",
        ),
        "descriptive_only": True,
        "improvement_threshold": None,
    }


def _preflight(p1_checkpoint: Path, p2_checkpoint: Path) -> tuple[Any, Any]:
    # Hashes are checked before any output directory creation or simulator reset.
    _validate_checkpoint(p1_checkpoint, P1_CHECKPOINT_SHA, "P1")
    _validate_checkpoint(p2_checkpoint, P2_CHECKPOINT_SHA, "P2")
    p1_policy = load_p1_checkpoint(p1_checkpoint)
    p2_policy = load_p2_checkpoint(p2_checkpoint)
    return p1_policy, p2_policy


def _seed_list(values: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _persist_episode_state(
    output: Path, records: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]],
    consumed_seeds: Sequence[int], *, status: str, last_attempt: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> None:
    """Persist records and progress after every episode boundary."""
    _write_json(output / "records.json", {"records": list(records)})
    progress = {
        "stage": "P1/P2",
        "status": status,
        "record_count": len(records),
        "attempted_episode_count": len(attempts),
        "attempted_seeds": _seed_list([int(item["seed"]) for item in attempts]),
        "consumed_seeds": _seed_list(consumed_seeds),
        "attempted_episodes": list(attempts),
        "last_attempt": dict(last_attempt) if last_attempt is not None else None,
        "error": dict(error) if error is not None else None,
    }
    _write_json(output / "progress.json", progress)


def _failed_gates(record: Mapping[str, Any]) -> list[str]:
    gates = record.get("gates")
    if not isinstance(gates, Mapping):
        return ["gates_missing"]
    return [str(name) for name, passed in gates.items() if not bool(passed)]


def _record_failure_category(record: Mapping[str, Any], failed_gates: Sequence[str]) -> str:
    if "safety_counts_valid" in failed_gates or "safety_zero" in failed_gates:
        return "safety"
    if "exact_replay" in failed_gates:
        return "replay"
    return "gate"


def _exception_failure_category(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".lower()
    if any(token in text for token in ("safety", "collision", "nmac", "separation")):
        return "safety"
    if any(token in text for token in ("replay", "action tape", "tape")):
        return "replay"
    return "exception"


def _failure_payload(*, stage: str, controller: str, seed: int, count: int,
                     category: str, failed_gates: Sequence[str] = (),
                     error: BaseException | str | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "controller": controller,
        "seed": int(seed),
        "count": int(count),
        "category": category,
        "failed_gates": [str(value) for value in failed_gates],
        "error_type": type(error).__name__ if isinstance(error, BaseException) else None,
        "error": str(error) if error is not None else None,
    }


def _manifest(
    *, output: Path, smoke: bool, status: str, records: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]], consumed_seeds: Sequence[int],
    failure: Mapping[str, Any] | None = None, aggregates: Mapping[str, Any] | None = None,
    p1_checkpoint: Path = P1_CHECKPOINT_PATH,
    p2_checkpoint: Path = P2_CHECKPOINT_PATH,
) -> dict[str, Any]:
    formal_seed_values = [] if smoke else _seed_list(consumed_seeds)
    diagnostic_seed_values = _seed_list(consumed_seeds) if smoke else []
    result = {
        "stage": STAGE,
        "status": status,
        "formal": not smoke,
        "diagnostic": smoke,
        "smoke": smoke,
        "protocol": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "p1_checkpoint": str(p1_checkpoint),
        "p1_checkpoint_sha256": P1_CHECKPOINT_SHA,
        "p2_checkpoint": str(p2_checkpoint),
        "p2_checkpoint_sha256": P2_CHECKPOINT_SHA,
        "records": list(records),
        "record_count": len(records),
        "aggregates": dict(aggregates) if aggregates is not None else None,
        "attempted_episodes": list(attempts),
        "attempted_seeds": _seed_list([int(item["seed"]) for item in attempts]),
        "consumed_seeds": _seed_list(consumed_seeds),
        "formal_seeds_consumed": formal_seed_values,
        "diagnostic_seeds_consumed": diagnostic_seed_values,
        "formal_seed_consumption": False if smoke else bool(formal_seed_values),
        "failure": dict(failure) if failure is not None else None,
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
        "training_called": False,
        "checkpoint_saved": False,
        "descriptive_effectiveness": True,
        "improvement_threshold": None,
    }
    _write_json(output / "manifest.json", result)
    return result


def run(*, output: Path, p1_checkpoint: Path = P1_CHECKPOINT_PATH,
        p2_checkpoint: Path = P2_CHECKPOINT_PATH, history_root: Path,
        smoke: bool = False) -> dict[str, Any]:
    """Run the diagnostic smoke case or the preregistered formal matrix."""
    protocol()
    _require(not output.exists(), f"output already exists: {output}")
    p1_checkpoint = Path(p1_checkpoint)
    p2_checkpoint = Path(p2_checkpoint)
    p1_policy, p2_policy = _preflight(p1_checkpoint, p2_checkpoint)

    if smoke:
        cases_p1 = ((SMOKE_SEED, SMOKE_COUNT),)
        cases_p2 = ((SMOKE_SEED, SMOKE_COUNT),)
        historical = {"seeds": set()}
    else:
        cases_p1 = tuple(zip(P1_FORMAL_SEEDS, P1_COUNTS))
        cases_p2 = tuple(zip(P2_FORMAL_SEEDS, P2_COUNTS))
        output.mkdir(parents=True, exist_ok=False)
        historical = historical_registry(Path(history_root), output)
        collisions = set(FORMAL_SEEDS) & set(historical["seeds"])
        if collisions:
            failure = {
                "category": "seed_collision",
                "error": f"formal seeds already consumed: {sorted(collisions)}",
                "failed_gates": ["fresh_seed_collision"],
            }
            _persist_episode_state(output, [], [], [], status="FORMAL_FAIL", error=failure)
            return _manifest(
                output=output, smoke=False, status="FORMAL_FAIL", records=[], attempts=[],
                consumed_seeds=[], failure=failure,
                p1_checkpoint=p1_checkpoint, p2_checkpoint=p2_checkpoint,
            )

    if smoke:
        output.mkdir(parents=True, exist_ok=False)
    planned = {
        "stage": STAGE,
        "formal": not smoke,
        "diagnostic": smoke,
        "p1": [{"seed": int(seed), "count": int(count)} for seed, count in cases_p1],
        "p2": [{"seed": int(seed), "count": int(count)} for seed, count in cases_p2],
        "formal_seeds": [] if smoke else list(FORMAL_SEEDS),
        "diagnostic_seeds": [SMOKE_SEED] if smoke else [],
        "historical_seed_count": len(historical.get("seeds", set())),
        "formal_seed_consumption": False,
        "consumed_seeds": [],
        "attempted_seeds": [],
        "PPO_ALLOWED": False,
        "P3_ALLOWED": False,
    }
    _write_json(output / "pre_reset_registry.json", planned)

    records: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    consumed_seeds: list[int] = []
    total = 2 * (len(cases_p1) + len(cases_p2))
    failure: dict[str, Any] | None = None

    def execute_episode(stage: str, controller: str, seed: int, count: int,
                        runner, episode_output: Path) -> bool:
        nonlocal failure
        attempt = {
            "stage": stage, "controller": controller,
            "seed": int(seed), "count": int(count),
        }
        attempts.append(attempt)
        if int(seed) not in consumed_seeds:
            consumed_seeds.append(int(seed))
        _persist_episode_state(
            output, records, attempts, consumed_seeds, status="RUNNING",
            last_attempt=attempt,
        )
        try:
            record = runner(episode_output)
        except Exception as error:
            failure = _failure_payload(
                stage=stage, controller=controller, seed=seed, count=count,
                category=_exception_failure_category(error), error=error,
            )
            _persist_episode_state(
                output, records, attempts, consumed_seeds, status="FORMAL_FAIL" if not smoke else "DIAGNOSTIC_FAIL",
                last_attempt=attempt, error=failure,
            )
            return False
        records.append(record)
        failed = _failed_gates(record)
        _persist_episode_state(
            output, records, attempts, consumed_seeds,
            status="RECORD_PERSISTED" if not failed else "FORMAL_FAIL" if not smoke else "DIAGNOSTIC_FAIL",
            last_attempt=attempt,
            error=(failure if failure is not None else None),
        )
        if failed:
            episode_failure = _failure_payload(
                stage=stage, controller=controller, seed=seed, count=count,
                category=_record_failure_category(record, failed), failed_gates=failed,
                error="episode gate failure",
            )
            critical_smoke_gates = {
                "safety_counts_valid", "safety_zero", "exact_replay", "shortlist_contract",
            }
            smoke_must_stop = bool(critical_smoke_gates.intersection(failed))
            if smoke and not smoke_must_stop:
                _persist_episode_state(
                    output, records, attempts, consumed_seeds,
                    status="DIAGNOSTIC_GATE_FAIL_CONTINUE",
                    last_attempt=attempt, error=episode_failure,
                )
                return True
            failure = episode_failure
            _persist_episode_state(
                output, records, attempts, consumed_seeds,
                status="FORMAL_FAIL" if not smoke else "DIAGNOSTIC_FAIL",
                last_attempt=attempt, error=failure,
            )
            return False
        return True

    def stop_manifest() -> dict[str, Any]:
        status = "FORMAL_FAIL" if not smoke else "DIAGNOSTIC_FAIL"
        return _manifest(
            output=output, smoke=smoke, status=status, records=records,
            attempts=attempts, consumed_seeds=consumed_seeds, failure=failure,
            p1_checkpoint=p1_checkpoint, p2_checkpoint=p2_checkpoint,
        )

    for seed, count in cases_p1:
        if not execute_episode(
            "P1", "candidate", seed, count,
            lambda episode_output: _run_pilot_episode(
                "P1", "candidate", p1_policy, seed, count, episode_output,
                max_steps=SMOKE_MAX_STEPS if smoke else None,
            ),
            output / f"p1_candidate_{seed}",
        ):
            result = stop_manifest()
            print(json.dumps({"stage": STAGE, "status": result["status"], "records": len(records)}, sort_keys=True), flush=True)
            return result
        if not execute_episode(
            "P1", "rule", seed, count,
            lambda episode_output: _run_rule(
                "P1", seed, count, episode_output,
                max_steps=SMOKE_MAX_STEPS if smoke else None,
            ),
            output / f"p1_rule_{seed}",
        ):
            result = stop_manifest()
            print(json.dumps({"stage": STAGE, "status": result["status"], "records": len(records)}, sort_keys=True), flush=True)
            return result
        _write_json(output / "progress.json", {
            "stage": "P1", "status": "PAIR_COMPLETE", "completed_records": len(records),
            "planned_records": total, "seed": int(seed), "count": int(count),
            "formal": not smoke, "attempted_seeds": _seed_list([item["seed"] for item in attempts]),
            "consumed_seeds": _seed_list(consumed_seeds),
        })

    for seed, count in cases_p2:
        if not execute_episode(
            "P2", "candidate", seed, count,
            lambda episode_output: _run_p2_student(
                p2_policy, seed, count, episode_output,
                max_steps=SMOKE_MAX_STEPS if smoke else None,
            ),
            output / f"p2_candidate_{seed}",
        ):
            result = stop_manifest()
            print(json.dumps({"stage": STAGE, "status": result["status"], "records": len(records)}, sort_keys=True), flush=True)
            return result
        if not execute_episode(
            "P2", "rule", seed, count,
            lambda episode_output: _run_rule(
                "P2", seed, count, episode_output,
                max_steps=SMOKE_MAX_STEPS if smoke else None,
            ),
            output / f"p2_rule_{seed}",
        ):
            result = stop_manifest()
            print(json.dumps({"stage": STAGE, "status": result["status"], "records": len(records)}, sort_keys=True), flush=True)
            return result
        _write_json(output / "progress.json", {
            "stage": "P2", "status": "PAIR_COMPLETE", "completed_records": len(records),
            "planned_records": total, "seed": int(seed), "count": int(count),
            "formal": not smoke, "attempted_seeds": _seed_list([item["seed"] for item in attempts]),
            "consumed_seeds": _seed_list(consumed_seeds),
        })

    if smoke:
        result = _manifest(
            output=output, smoke=True, status="DIAGNOSTIC_SMOKE", records=records,
            attempts=attempts, consumed_seeds=consumed_seeds,
            p1_checkpoint=p1_checkpoint, p2_checkpoint=p2_checkpoint,
        )
    else:
        try:
            aggregates = {
                "P1": aggregate_records([row for row in records if row["stage"] == "P1"], "P1"),
                "P2": aggregate_records([row for row in records if row["stage"] == "P2"], "P2"),
            }
        except Exception as error:
            failure = {
                "category": "aggregate",
                "failed_gates": ["aggregation"],
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _persist_episode_state(
                output, records, attempts, consumed_seeds,
                status="FORMAL_FAIL", error=failure,
            )
            result = _manifest(
                output=output, smoke=False, status="FORMAL_FAIL", records=records,
                attempts=attempts, consumed_seeds=consumed_seeds, failure=failure,
                p1_checkpoint=p1_checkpoint, p2_checkpoint=p2_checkpoint,
            )
            print(json.dumps({"stage": STAGE, "status": result["status"], "records": len(records)}, sort_keys=True), flush=True)
            return result
        result = _manifest(
            output=output, smoke=False, status="FORMAL_COMPLETE", records=records,
            attempts=attempts, consumed_seeds=consumed_seeds, aggregates=aggregates,
            p1_checkpoint=p1_checkpoint, p2_checkpoint=p2_checkpoint,
        )
    print(json.dumps({"stage": STAGE, "status": result["status"], "records": len(records)}, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p1-checkpoint", type=Path, default=P1_CHECKPOINT_PATH)
    parser.add_argument("--p2-checkpoint", type=Path, default=P2_CHECKPOINT_PATH)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    args = parser.parse_args()
    result = run(**vars(args))
    if str(result.get("status", "")).endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
