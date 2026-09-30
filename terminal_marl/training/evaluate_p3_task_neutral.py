"""Paired development evaluation with task-priority features neutralized.

This entry point is a fresh, read-only two-arm evaluation.  Both the fixed
source checkpoint and the fixed gate-repair checkpoint are rolled out on the
same six development cells after the task-priority input/reward path is
neutralized.  Existing runtime records are used only as an execution helper;
no historical baseline or admission decision is consumed here.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from terminal_marl.core.arrival_process import ArrivalProcess
from terminal_marl.core.uav import MISSION_SERVICE_SECONDS
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import run_p3_adaptive_kl_formal_v7 as formal_v7


STAGE = "P3_TASK_NEUTRAL_PAIRED_EVAL_V1"
LOW_SOC_THRESHOLD = rank_eval.LOW_SOC_THRESHOLD
OBSERVATION_DIM = rank_eval.OBSERVATION_DIM

# These are deliberately fixed bindings.  The source digest is the original
# 03c0 checkpoint; the candidate is the normal-advantage gate-repair fit.
SOURCE_CHECKPOINT = rank_eval.SOURCE_CHECKPOINT
SOURCE_CHECKPOINT_SHA256 = "03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba"
CANDIDATE_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt"
)
CANDIDATE_CHECKPOINT_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"

ARMS = ("source", "candidate")
SEED_SCENARIO = dict(rank_eval.SEED_SCENARIO)
EXPECTED_REQUESTS = dict(rank_eval.EXPECTED_REQUESTS)
EXPECTED_TOTAL_REQUESTS = sum(EXPECTED_REQUESTS[str(item["scenario"])] for item in rank_eval._evaluation_cells())


class TaskNeutralEvaluationError(ValueError):
    """Raised when the fixed paired evaluation binding is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaskNeutralEvaluationError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError, RuntimeError):
            pass
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskNeutralEvaluationError(f"invalid JSON: {target}") from exc
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _synced_path(path: Path) -> Path:
    """Resolve a server output through the documented local sync mirror."""

    if path.is_file():
        return path
    value = str(path).replace("\\", "/")
    prefix = "/home/user/training_outputs/terminal/"
    if value.startswith(prefix):
        local_root = Path(__file__).resolve().parents[3] / "training_outputs" / "terminal"
        local = local_root / value[len(prefix):]
        if local.is_file():
            return local
    return path


def _checkpoint_from_report(declared: Path, expected_sha: str) -> tuple[Path, str | None]:
    """Use the fixed checkpoint or its documented local sync mirror."""

    direct = _synced_path(declared)
    if direct.is_file() and _sha256(direct) == expected_sha:
        return direct, None
    if direct.is_file():
        raise TaskNeutralEvaluationError(
            f"{declared} SHA mismatch: expected {expected_sha}, got {_sha256(direct)}"
        )
    raise TaskNeutralEvaluationError(f"checkpoint missing: {declared}")


def _config_snapshot(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config):
        raw = dataclasses.asdict(config)
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = dict(vars(config))
    # Preserve the historical snapshot shape for the default mask mode.  The
    # opt-in mode remains explicit in reports so its protocol cannot be
    # mistaken for a legacy evaluation.
    if raw.get("student_decision_mask_mode") == "legacy":
        raw.pop("student_decision_mask_mode", None)
    value = _jsonable(raw)
    _require(isinstance(value, dict), "config snapshot must be a mapping")
    return value


def _neutralize_config(config: Any) -> Any:
    """Return a copy with exactly the two registered task-neutral changes."""

    if not hasattr(config, "task_priority_enabled"):
        raise TaskNeutralEvaluationError("evaluation config has no task_priority_enabled field")
    if not hasattr(config, "low_level_priority_time_weight"):
        raise TaskNeutralEvaluationError("evaluation config has no low_level_priority_time_weight field")
    try:
        neutral = dataclasses.replace(
            config,
            task_priority_enabled=False,
            low_level_priority_time_weight=0.0,
        )
    except TypeError:
        # Small test doubles may not be dataclasses.  Keep the production path
        # on dataclasses.replace while preserving the same copy semantics for
        # local input/output tests.
        neutral = copy.copy(config)
        setattr(neutral, "task_priority_enabled", False)
        setattr(neutral, "low_level_priority_time_weight", 0.0)
    _require(getattr(neutral, "task_priority_enabled") is False,
             "neutral config task_priority_enabled must be false")
    _require(float(getattr(neutral, "low_level_priority_time_weight")) == 0.0,
             "neutral config low_level_priority_time_weight must be zero")
    return neutral


def _config_delta(old: Any, new: Any) -> dict[str, Any]:
    old_snapshot = _config_snapshot(old)
    new_snapshot = _config_snapshot(new)
    keys = sorted(set(old_snapshot) | set(new_snapshot))
    changed = {
        key: {"old": old_snapshot.get(key), "new": new_snapshot.get(key)}
        for key in keys
        if old_snapshot.get(key) != new_snapshot.get(key)
    }
    _require(set(changed) <= {"task_priority_enabled", "low_level_priority_time_weight"},
             f"neutral config changed unexpected fields: {sorted(changed)}")
    return changed


def _evaluation_cells() -> list[dict[str, Any]]:
    cells = [dict(cell) for cell in rank_eval._evaluation_cells()]
    expected = {
        (873101, "flow6", "train"),
        (873102, "flow12", "train"),
        (873103, "dense24", "train"),
        (873104, "flow6", "validation"),
        (873105, "flow12", "validation"),
        (873106, "dense24", "validation"),
    }
    observed = {(int(item["seed"]), str(item["scenario"]), str(item["split"])) for item in cells}
    _require(observed == expected and len(cells) == 6, "registered development cells changed")
    return cells


def _build_neutral_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = formal_v7._protocol()
    configs, runtime_protocol = rank_eval._build_eval_inputs(protocol)
    neutral_configs: dict[str, Any] = {}
    for scenario, config in configs.items():
        neutral_configs[str(scenario)] = _neutralize_config(config)
        _config_delta(config, neutral_configs[str(scenario)])
    for scenario in {str(item["scenario"]) for item in _evaluation_cells()}:
        _require(scenario in neutral_configs, f"missing config for scenario {scenario}")
    return dict(configs), neutral_configs, (
        dict(runtime_protocol) if isinstance(runtime_protocol, Mapping) else {}
    )


def _arrival_rows(config: Any, *, seed: int, expected_requests: int) -> list[dict[str, Any]]:
    process = ArrivalProcess(config, seed=int(seed))
    rows: list[dict[str, Any]] = []
    max_rounds = max(2, int(expected_requests) + 2)
    for _ in range(max_rounds):
        if process.exhausted():
            break
        due = process.requests_until(process.next_time)
        if not due:
            raise TaskNeutralEvaluationError(
                f"ArrivalProcess produced no requests at seed {seed} time {process.next_time}"
            )
        for request in due:
            mission = getattr(request.mission_type, "value", str(request.mission_type))
            try:
                service = float(MISSION_SERVICE_SECONDS[request.mission_type])
            except (KeyError, TypeError):
                service = None
            rows.append({
                "arrival_id": int(request.arrival_id),
                "request_time": float(request.request_time),
                "gate_id": int(request.gate_id),
                "initial_soc": float(request.initial_soc),
                "mission_type": str(mission),
                "service_seconds": service,
                "energy_capacity": float(request.energy_capacity),
                "base_priority": int(request.base_priority),
            })
            if len(rows) > int(expected_requests):
                raise TaskNeutralEvaluationError(
                    f"ArrivalProcess exceeded {expected_requests} requests for seed {seed}"
                )
        if len(rows) == int(expected_requests):
            break
    _require(len(rows) == int(expected_requests),
             f"ArrivalProcess returned {len(rows)} requests, expected {expected_requests}")
    _require([row["arrival_id"] for row in rows] == list(range(int(expected_requests))),
             f"ArrivalProcess arrival IDs are incomplete for seed {seed}")
    return rows


def _same_physical_arrival(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    for key in ("arrival_id", "gate_id", "mission_type", "service_seconds", "energy_capacity"):
        if old.get(key) != new.get(key):
            return False
    for key in ("request_time", "initial_soc"):
        try:
            if not math.isclose(float(old[key]), float(new[key]), rel_tol=0.0, abs_tol=1e-12):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def validate_arrival_preflight(
    config: Any,
    neutral_config: Any,
    *,
    seed: int,
    expected_requests: int,
) -> dict[str, Any]:
    """Check same-seed physical arrivals and neutral base priorities."""

    old_rows = _arrival_rows(config, seed=seed, expected_requests=expected_requests)
    new_rows = _arrival_rows(neutral_config, seed=seed, expected_requests=expected_requests)
    physical_equal = all(_same_physical_arrival(old, new) for old, new in zip(old_rows, new_rows))
    base_all_one = all(int(row["base_priority"]) == 1 for row in new_rows)
    _require(physical_equal, f"ArrivalProcess physical attributes differ for seed {seed}")
    _require(base_all_one, f"neutral ArrivalProcess base priorities are not all one for seed {seed}")
    return {
        "seed": int(seed),
        "requests": int(expected_requests),
        "physical_equal": bool(physical_equal),
        "new_base_all_one": bool(base_all_one),
        "old": old_rows,
        "new": new_rows,
    }


def preflight_arrival_process(
    configs: Mapping[str, Any],
    neutral_configs: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected = list(_evaluation_cells() if cells is None else cells)
    reports: list[dict[str, Any]] = []
    for cell in selected:
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        _require(scenario in configs and scenario in neutral_configs,
                 f"missing arrival config for {scenario}")
        result = validate_arrival_preflight(
            configs[scenario], neutral_configs[scenario],
            seed=seed, expected_requests=int(EXPECTED_REQUESTS[scenario]),
        )
        result["scenario"] = scenario
        result["split"] = str(cell.get("split", "development"))
        reports.append(result)
    return reports


# Compatibility aliases make the small preflight useful to local tests and
# review scripts without introducing another framework.
_arrival_preflight = validate_arrival_preflight
_preflight_arrival_process = preflight_arrival_process


def _record_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    lifecycle = record.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or not isinstance(lifecycle.get("rows"), list):
        return []
    return [dict(row) for row in lifecycle["rows"] if isinstance(row, Mapping)]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _wait_values(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    return [value for row in rows if (value := _finite(row.get("wait_seconds"))) is not None]


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _safety_snapshot(record: Mapping[str, Any]) -> dict[str, Any]:
    for value in (record.get("safety"), (record.get("summary") or {}).get("safety")
                  if isinstance(record.get("summary"), Mapping) else None):
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _runtime_technical_ok(record: Mapping[str, Any], *, smoke: bool) -> bool:
    lifecycle = record.get("lifecycle")
    lifecycle_complete = bool(
        record.get("lifecycle_valid") is True
        and isinstance(lifecycle, Mapping)
        and lifecycle.get("scope") == "all_requested_terminal_rows"
        and lifecycle.get("replay_exact") is True
        and lifecycle.get("all_requests_observed") is True
        and lifecycle.get("ledger_alignment") is True
        and int(lifecycle.get("missing_request_count", 0) or 0) == 0
        and int(lifecycle.get("row_count", 0) or 0) == int(record.get("requests", 0) or 0)
    )
    safety_nonzero = record.get("safety_nonzero")
    if safety_nonzero is None:
        safety_nonzero = rank_eval.legacy_eval._safety_nonzero(record)
    no_error = not bool(record.get("error"))
    return bool(
        (True if smoke else lifecycle_complete)
        and record.get("weights_unchanged") is True
        and not bool(safety_nonzero)
        and no_error
        and record.get("teacher_called", False) is False
        and record.get("training_or_update_called", False) is False
    )


def _record_metrics(record: Mapping[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    counts = record.get("outcome_counts")
    if not isinstance(counts, Mapping):
        counts = {}
    requests = int(counts.get("requests", record.get("requests", 0)) or 0)
    normal = int(counts.get("normal", record.get("normal", 0)) or 0)
    emergency = int(counts.get("emergency", record.get("emergency", 0)) or 0)
    failed_energy = int(counts.get("failed_energy", record.get("failed_energy", 0)) or 0)
    other = int(counts.get("other", 0) or 0)
    unresolved = int(counts.get("unresolved", record.get("unresolved", 0)) or 0)
    rows = _record_rows(record)
    waits = _wait_values(rows)
    low_soc_waits = [
        float(row["wait_seconds"])
        for row in rows
        if _finite(row.get("initial_soc")) is not None
        and float(row["initial_soc"]) <= float(LOW_SOC_THRESHOLD)
        and _finite(row.get("wait_seconds")) is not None
    ]
    priority_wait = None
    if rows:
        weighted_rows = [
            row for row in rows
            if _finite(row.get("wait_seconds")) is not None
            and _finite(row.get("base_priority")) is not None
        ]
        if weighted_rows:
            weights = sum(float(row["base_priority"]) for row in weighted_rows)
            priority_wait = (
                sum(float(row["wait_seconds"]) * float(row["base_priority"]) for row in weighted_rows) / weights
                if weights else None
            )
    summary = record.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    reward_components = summary.get("reward_component_totals")
    if not isinstance(reward_components, Mapping):
        reward_components = {}
    reward_total = _finite(summary.get("reward_total"))
    safety = _safety_snapshot(record)
    safety_nonzero = record.get("safety_nonzero")
    if safety_nonzero is None:
        safety_nonzero = rank_eval.legacy_eval._safety_nonzero(record)
    mean_wait = None if not waits else sum(waits) / len(waits)
    p95_wait = _p95(waits)
    max_wait = None if not waits else max(waits)
    low_soc_wait = None if not low_soc_waits else sum(low_soc_waits) / len(low_soc_waits)
    return {
        "requests": requests,
        "normal": normal,
        "emergency": emergency,
        "failed_energy": failed_energy,
        "failed": failed_energy + other,
        "other": other,
        "unresolved": unresolved,
        "wait_rows": len(waits),
        "mean_wait": mean_wait,
        "mean_wait_seconds": mean_wait,
        "p95_wait": p95_wait,
        "p95_wait_seconds": p95_wait,
        "max_wait": max_wait,
        "max_wait_seconds": max_wait,
        "lowSOC_wait": low_soc_wait,
        "lowSOC_wait_seconds": low_soc_wait,
        "low_soc_wait": low_soc_wait,
        "low_soc_wait_seconds": low_soc_wait,
        "priority_wait": priority_wait,
        "reward": {"total": reward_total, "components": dict(reward_components),
                   "components_consistent": summary.get("reward_components_consistent")},
        "safety": safety,
        "safety_nonzero": bool(safety_nonzero),
        "weights_unchanged": record.get("weights_unchanged"),
        "lifecycle_valid": record.get("lifecycle_valid"),
        "technical_ok": _runtime_technical_ok(record, smoke=smoke),
        "error": record.get("error"),
    }


def _aggregate_arm(records: Sequence[Mapping[str, Any]], *, smoke: bool = False) -> dict[str, Any]:
    """Aggregate from all available lifecycle rows, retaining failure counts."""

    # The registered helper intentionally requires all six development cells
    # for its full split partition.  A check-only report and a terminal
    # failure may contain no/partial records, so retain a small partial marker
    # while still using the helper for every complete arm result.
    complete_keys = {
        (int(item["seed"]), str(item["scenario"]))
        for item in records
        if "seed" in item and "scenario" in item
    }
    expected_keys = {(int(item["seed"]), str(item["scenario"])) for item in _evaluation_cells()}
    if records and (smoke or complete_keys == expected_keys):
        runtime = rank_eval._candidate_summaries(records, smoke=smoke)
    else:
        runtime = {"train": {}, "validation": {}, "overall": {}}
    metrics = [_record_metrics(record, smoke=smoke) for record in records]
    rows = [row for record in records for row in _record_rows(record)]
    counts = {key: 0 for key in ("requests", "normal", "emergency", "failed_energy", "failed", "other", "unresolved")}
    for item in metrics:
        for key in counts:
            counts[key] += int(item.get(key, 0) or 0)
    waits = _wait_values(rows)
    low_soc_waits = [
        float(row["wait_seconds"])
        for row in rows
        if _finite(row.get("initial_soc")) is not None
        and float(row["initial_soc"]) <= float(LOW_SOC_THRESHOLD)
        and _finite(row.get("wait_seconds")) is not None
    ]
    priority_rows = [
        row for row in rows
        if _finite(row.get("wait_seconds")) is not None
        and _finite(row.get("base_priority")) is not None
    ]
    weight_sum = sum(float(row["base_priority"]) for row in priority_rows)
    priority_wait = (
        sum(float(row["wait_seconds"]) * float(row["base_priority"]) for row in priority_rows) / weight_sum
        if priority_rows and weight_sum else None
    )
    reward_components: dict[str, float] = {}
    reward_total = 0.0
    reward_seen = False
    for item in metrics:
        reward = item["reward"]
        if reward["total"] is not None:
            reward_total += float(reward["total"])
            reward_seen = True
        for key, value in reward["components"].items():
            numeric = _finite(value)
            if numeric is not None:
                reward_components[str(key)] = reward_components.get(str(key), 0.0) + numeric
    safety_nonzero = sum(bool(item["safety_nonzero"]) for item in metrics)
    mean_wait = None if not waits else sum(waits) / len(waits)
    p95_wait = _p95(waits)
    max_wait = None if not waits else max(waits)
    low_soc_wait = None if not low_soc_waits else sum(low_soc_waits) / len(low_soc_waits)
    total_reward = reward_total if reward_seen else None
    return {
        "records": len(records),
        "counts": counts,
        **{key: counts[key] for key in ("requests", "normal", "emergency", "failed", "unresolved")},
        "failed_energy": counts["failed_energy"],
        "other": counts["other"],
        "mean_wait": mean_wait,
        "mean_wait_seconds": mean_wait,
        "p95_wait": p95_wait,
        "p95_wait_seconds": p95_wait,
        "max_wait": max_wait,
        "max_wait_seconds": max_wait,
        "lowSOC_wait": low_soc_wait,
        "lowSOC_wait_seconds": low_soc_wait,
        "low_soc_wait": low_soc_wait,
        "low_soc_wait_seconds": low_soc_wait,
        "priority_wait": priority_wait,
        "wait_rows": len(waits),
        "lifecycle_rows": len(rows),
        "reward": {"total": total_reward, "components": reward_components},
        "reward_total": total_reward,
        "reward_components": reward_components,
        "safety_nonzero_cells": int(safety_nonzero),
        "technical_ok": bool(metrics) and all(bool(item["technical_ok"]) for item in metrics),
        "full_request_complete": bool(metrics) and all(
            item["lifecycle_valid"] is True for item in metrics
        ),
        "runtime_summaries": runtime,
    }


def aggregate_arm(records: Sequence[Mapping[str, Any]], *, smoke: bool = False) -> dict[str, Any]:
    """Public spelling for the all-rows arm aggregation used by tests."""

    return _aggregate_arm(records, smoke=smoke)


def _pair_key(record: Mapping[str, Any]) -> tuple[int, str]:
    try:
        return int(record["seed"]), str(record["scenario"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskNeutralEvaluationError("evaluation record seed/scenario is missing") from exc


def pair_records(
    source_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    cells: Sequence[Mapping[str, Any]] | None = None,
) -> list[tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    selected = list(_evaluation_cells() if cells is None else cells)
    expected = {(int(cell["seed"]), str(cell["scenario"])) for cell in selected}

    def index(records: Sequence[Mapping[str, Any]], label: str) -> dict[tuple[int, str], Mapping[str, Any]]:
        result: dict[tuple[int, str], Mapping[str, Any]] = {}
        for record in records:
            key = _pair_key(record)
            _require(key in expected, f"{label} has unexpected cell: {key}")
            _require(key not in result, f"{label} has duplicate cell: {key}")
            result[key] = record
        _require(set(result) == expected, f"{label} does not contain exactly the selected cells")
        return result

    source = index(source_records, "source")
    candidate = index(candidate_records, "candidate")
    return [
        (dict(cell), source[(int(cell["seed"]), str(cell["scenario"]))],
         candidate[(int(cell["seed"]), str(cell["scenario"]))])
        for cell in selected
    ]


_pair_records = pair_records


def _weight_digest(policy: Any) -> str | None:
    try:
        return rank_eval._model_digest(policy)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _weight_preflight(checkpoint: Path, expected_sha: str, config: Any, *, label: str) -> dict[str, Any]:
    resolved, report_path = _checkpoint_from_report(checkpoint, expected_sha)
    _require(_sha256(resolved) == expected_sha, f"{label} checkpoint SHA mismatch")
    policy, _payload = rank_eval._load_policy(resolved, config)
    optimizer = getattr(policy, "optimizer", None)
    state = getattr(optimizer, "state", None)
    optimizer_empty = isinstance(state, Mapping) and len(state) == 0
    model = getattr(policy, "model", None)
    model_eval = getattr(model, "training", None) is False
    return {
        "label": label,
        "declared_checkpoint": str(checkpoint),
        "resolved_checkpoint": str(resolved),
        "checkpoint_sha256": _sha256(resolved),
        "locator_report": report_path,
        "model_digest": _weight_digest(policy),
        "model_eval": model_eval,
        "optimizer_state_empty": optimizer_empty,
    }


def _record_fatal_reason(record: Mapping[str, Any], *, smoke: bool) -> str | None:
    if record.get("error"):
        error = record.get("error")
        return str(error.get("message", "record error")) if isinstance(error, Mapping) else str(error)
    if record.get("weights_unchanged") is not True:
        return "weights_unchanged is not true"
    safety_nonzero = record.get("safety_nonzero")
    if safety_nonzero is None:
        safety_nonzero = rank_eval.legacy_eval._safety_nonzero(record)
    if bool(safety_nonzero):
        return "runtime safety field indicates a violation"
    if not smoke and not _runtime_technical_ok(record, smoke=False):
        return "runtime technical fields are incomplete or invalid"
    if smoke and record.get("status") == "failed":
        return "smoke runtime record failed"
    return None


def _progress(output: Path, *, status: str, processed: int, expected: int,
              smoke: bool, check_only: bool = False, error: str | None = None) -> None:
    value: dict[str, Any] = {
        "stage": STAGE,
        "status": status,
        "processed": int(processed),
        "expected": int(expected),
        "smoke": bool(smoke),
        "check_only": bool(check_only),
        "full_evaluation": bool(not smoke and not check_only),
        "PPO_ALLOWED": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    }
    if error:
        value["error"] = str(error)
    _write_json(output / "progress.json", value)


def _report(
    *,
    status: str,
    smoke: bool,
    check_only: bool = False,
    cells: Sequence[Mapping[str, Any]],
    arrival_preflight: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    stop_reason: str | None = None,
) -> dict[str, Any]:
    source_records = list(records.get("source", ()))
    candidate_records = list(records.get("candidate", ()))
    summaries = {
        "source": _aggregate_arm(source_records, smoke=smoke) if source_records else _aggregate_arm([], smoke=smoke),
        "candidate": _aggregate_arm(candidate_records, smoke=smoke) if candidate_records else _aggregate_arm([], smoke=smoke),
    }
    cell_rows: list[dict[str, Any]] = []
    source_index = {_pair_key(record): record for record in source_records if "seed" in record and "scenario" in record}
    candidate_index = {_pair_key(record): record for record in candidate_records if "seed" in record and "scenario" in record}
    for cell in cells:
        key = (int(cell["seed"]), str(cell["scenario"]))
        cell_rows.append({
            "seed": key[0],
            "scenario": key[1],
            "split": str(cell.get("split", "development")),
            "requested": int(EXPECTED_REQUESTS[key[1]]),
            "source": _record_metrics(source_index[key], smoke=smoke) if key in source_index else None,
            "candidate": _record_metrics(candidate_index[key], smoke=smoke) if key in candidate_index else None,
        })
    result: dict[str, Any] = {
        "stage": STAGE,
        "status": status,
        "smoke": bool(smoke),
        "check_only": bool(check_only),
        "full_evaluation": bool(not smoke and not check_only),
        "records_expected": len(cells) * len(ARMS),
        "records_processed": len(source_records) + len(candidate_records),
        "requests_per_arm_expected": 6 if smoke else EXPECTED_TOTAL_REQUESTS,
        "requests_expected_total": (6 if smoke else EXPECTED_TOTAL_REQUESTS) * len(ARMS),
        "binding": dict(binding),
        "arrival_preflight": list(arrival_preflight),
        "arms": {
            "source": {"records": source_records, "summary": summaries["source"]},
            "candidate": {"records": candidate_records, "summary": summaries["candidate"]},
        },
        "overall": summaries,
        "cells": cell_rows,
        "pairing": {"expected_cells": len(cells), "paired_cells": sum(
            1 for row in cell_rows if row["source"] is not None and row["candidate"] is not None
        )},
        "technical_ok": (
            None if check_only else bool(
                summaries["source"]["technical_ok"] and summaries["candidate"]["technical_ok"]
            )
        ),
        "auto_promoted": False,
        "candidate_auto_promoted": False,
        "candidate_retained": None,
        "retention_decision": "pending_primary_review",
        "teacher_called": False,
        "training_or_update_called": False,
        "heldout_consumed": False,
        "PPO_ALLOWED": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    }
    if stop_reason:
        result["stop_reason"] = str(stop_reason)
    return result


def run(*, output_dir: str | Path, check_only: bool = False, smoke: bool = False) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"evaluation output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    cells = _evaluation_cells()
    selected_cells = cells[:1] if smoke else cells
    expected_records = len(selected_cells) * len(ARMS)
    _progress(output, status="STARTING", processed=0, expected=expected_records,
              smoke=smoke, check_only=check_only)

    preflight: list[dict[str, Any]] = []
    binding: dict[str, Any] = {
        "stage": STAGE,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
        "registered_cells": [dict(cell) for cell in cells],
        "selected_cells": [dict(cell) for cell in selected_cells],
        "arrival_requests_per_arm": 6 if smoke else EXPECTED_TOTAL_REQUESTS,
        "teacher_called": False,
        "training_or_update_called": False,
        "heldout_consumed": False,
    }
    try:
        _require(SOURCE_CHECKPOINT_SHA256 == rank_eval.SOURCE_CHECKPOINT_SHA256,
                 "rank evaluation source SHA binding changed")
        configs, neutral_configs, runtime_protocol = _build_neutral_inputs()
        del runtime_protocol
        preflight = preflight_arrival_process(configs, neutral_configs, selected_cells)
        source_weight = _weight_preflight(
            SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, neutral_configs["dense24"], label="source"
        )
        candidate_weight = _weight_preflight(
            CANDIDATE_CHECKPOINT, CANDIDATE_CHECKPOINT_SHA256, neutral_configs["dense24"], label="candidate"
        )
        binding.update({
            "arms": {"source": source_weight, "candidate": candidate_weight},
            "config_snapshots": {
                scenario: {
                    "original": _config_snapshot(configs[scenario]),
                    "neutral": _config_snapshot(neutral_configs[scenario]),
                    "changed_fields": _config_delta(configs[scenario], neutral_configs[scenario]),
                }
                for scenario in sorted(neutral_configs)
            },
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        result = _report(
            status="FAILED", smoke=smoke, check_only=check_only, cells=selected_cells,
            arrival_preflight=preflight, binding={**binding, "preflight_error": str(exc)},
            records={"source": [], "candidate": []}, stop_reason=str(exc),
        )
        _write_json(output / "report.json", result)
        _progress(output, status="FAILED", processed=0, expected=expected_records,
                  smoke=smoke, check_only=check_only, error=str(exc))
        return result

    if check_only:
        result = _report(
            status="CHECK_ONLY_OK", smoke=smoke, check_only=True, cells=selected_cells,
            arrival_preflight=preflight, binding=binding,
            records={"source": [], "candidate": []},
        )
        _write_json(output / "report.json", result)
        _progress(output, status="CHECK_ONLY_OK", processed=0, expected=expected_records,
                  smoke=smoke, check_only=True)
        return result

    records: dict[str, list[dict[str, Any]]] = {"source": [], "candidate": []}
    stop_reason: str | None = None
    processed = 0
    for cell in selected_cells:
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        for arm, checkpoint, expected_sha in (
            ("source", SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256),
            ("candidate", CANDIDATE_CHECKPOINT, CANDIDATE_CHECKPOINT_SHA256),
        ):
            record_output = output / "rollouts" / arm / f"rollout_{scenario}_{seed}"
            try:
                resolved, _report_path = _checkpoint_from_report(checkpoint, expected_sha)
                policy, _payload = rank_eval._load_policy(resolved, neutral_configs["dense24"])
                record = rank_eval._runtime_record(
                    policy=policy,
                    checkpoint=resolved,
                    config=neutral_configs[scenario],
                    scenario=scenario,
                    seed=seed,
                    output=record_output,
                    smoke=smoke,
                )
                record = dict(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                record = rank_eval._failure_record(
                    scenario=scenario, seed=seed, checkpoint=_synced_path(checkpoint),
                    requests=int(EXPECTED_REQUESTS[scenario]), error=exc,
                )
            record.update({
                "arm": arm,
                "checkpoint_declared": str(checkpoint),
                "checkpoint_sha256_expected": expected_sha,
                "config": _config_snapshot(neutral_configs[scenario]),
                "task_priority_enabled": False,
                "low_level_priority_time_weight": 0.0,
            })
            record_metrics = _record_metrics(record, smoke=smoke)
            record.update({
                "model_unchanged": record.get("weights_unchanged"),
                "safety": dict(record_metrics["safety"]),
                "reward": dict(record_metrics["reward"]),
                "wait": {
                    key: record_metrics[key]
                    for key in ("mean_wait", "p95_wait", "max_wait", "lowSOC_wait", "priority_wait")
                },
                "technical_ok": record_metrics["technical_ok"],
            })
            records[arm].append(record)
            _write_json(output / "records" / arm / f"{scenario}_{seed}.json", record)
            processed += 1
            reason = _record_fatal_reason(record, smoke=smoke)
            _progress(output, status="RUNNING" if reason is None else "FAILED",
                      processed=processed, expected=expected_records, smoke=smoke,
                      check_only=False, error=reason)
            if reason is not None:
                stop_reason = f"{arm}/{scenario}/{seed}: {reason}"
                break
        if stop_reason is not None:
            break

    status = "FAILED" if stop_reason is not None else ("SMOKE_COMPLETED" if smoke else "COMPLETED")
    result = _report(
        status=status, smoke=smoke, check_only=False, cells=selected_cells,
        arrival_preflight=preflight, binding=binding, records=records,
        stop_reason=stop_reason,
    )
    result["check_only"] = False
    _write_json(output / "report.json", result)
    _progress(output, status=status, processed=processed, expected=expected_records,
              smoke=smoke, check_only=False, error=stop_reason)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(output_dir=args.output_dir, check_only=args.check_only, smoke=args.smoke)
    except (TaskNeutralEvaluationError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({
        "stage": STAGE,
        "status": result.get("status"),
        "records_processed": result.get("records_processed", 0),
        "technical_ok": result.get("technical_ok", False),
        "PPO_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"CHECK_ONLY_OK", "COMPLETED", "SMOKE_COMPLETED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
