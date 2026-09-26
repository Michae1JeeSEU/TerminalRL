"""Failure-inclusive lifecycle metrics for one exact action-tape replay.

The older lifecycle helper only visits ``completed_uavs``.  This module keeps
the replay and the terminal-row collection together so failed ``_resolve``
rows remain in the request denominator while rows with unavailable fields are
reported as missing instead of receiving synthetic zero values.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv


SUCCESS_OUTCOMES = frozenset({"completed", "emergency_landed"})


class LifecycleMetricsError(ValueError):
    """Raised when an exact replay or its terminal ledger is inconsistent."""


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _initial_soc(uav: Any) -> float | None:
    capacity = _number(getattr(uav, "energy_capacity", None))
    remaining = _number(getattr(uav, "energy_remaining", None))
    consumed = _number(getattr(uav, "energy_consumed", None))
    if capacity is None or capacity <= 0.0 or remaining is None or consumed is None:
        return None
    return (remaining + consumed) / capacity


def _remember(values: Sequence[Any], first_seen: dict[int, dict[str, Any]]) -> None:
    for uav in values:
        arrival_id = getattr(uav, "arrival_id", None)
        if arrival_id is None:
            continue
        arrival_id = int(arrival_id)
        if arrival_id not in first_seen:
            first_seen[arrival_id] = {
                "initial_soc": _initial_soc(uav),
                "initial_soc_source": "first_seen_uav",
            }


def _row_from_uav(
    uav: Any,
    *,
    first_seen: Mapping[int, Mapping[str, Any]],
    terminal_time: Any,
    outcome: Any,
    reason: Any,
    source: str,
    phase_before_terminal: Any | None = None,
    emergency_landing: Any | None = None,
) -> dict[str, Any]:
    arrival_id = int(uav.arrival_id)
    cached = first_seen.get(arrival_id) or {}
    initial_soc = cached.get("initial_soc")
    if initial_soc is None:
        initial_soc = _initial_soc(uav)
        initial_source = "terminal_state_sum" if initial_soc is not None else "unavailable"
    else:
        initial_source = str(cached.get("initial_soc_source", "first_seen_uav"))
    external = _number(getattr(uav, "external_wait_time", None))
    internal = _number(getattr(uav, "internal_wait_time", None))
    wait = external + internal if external is not None and internal is not None else None
    touchdown = _number(getattr(uav, "touchdown_time", None))
    request_time = _number(getattr(uav, "request_time", None))
    terminal = _number(terminal_time)
    return {
        "arrival_id": arrival_id,
        "uid": getattr(uav, "uid", None),
        "base_priority": getattr(uav, "base_priority", None),
        "initial_soc": initial_soc,
        "initial_soc_source": initial_source,
        "request_time": request_time,
        "terminal_time": terminal,
        "phase_before_terminal": (
            _enum_value(phase_before_terminal)
            if phase_before_terminal is not None
            else _enum_value(getattr(uav, "phase", None))
        ),
        "external_wait_time": external,
        "internal_wait_time": internal,
        "wait_seconds": wait,
        "energy_remaining": _number(getattr(uav, "energy_remaining", None)),
        "energy_consumed": _number(getattr(uav, "energy_consumed", None)),
        "energy_capacity": _number(getattr(uav, "energy_capacity", None)),
        "final_soc": _number(getattr(uav, "soc", None)),
        "outcome": _enum_value(outcome),
        "reason": None if reason is None else str(reason),
        "touchdown_time": touchdown,
        "completion_time": _number(getattr(uav, "completion_time", None)),
        "emergency_landing": (
            bool(emergency_landing)
            if emergency_landing is not None
            else bool(getattr(uav, "emergency_landing", False))
        ),
        "source": source,
    }


def _capture_terminal_rows(env: TerminalParallelEnv) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Install passive hooks and return captured rows plus first-seen SOC."""
    captured: list[dict[str, Any]] = []
    first_seen: dict[int, dict[str, Any]] = {}
    original_step = env.step
    original_resolve = env._resolve

    def stepped(self: TerminalParallelEnv, *args: Any, **kwargs: Any):
        _remember(
            list(self.active.values())
            + list(self.service_uavs.values())
            + list(self.completed_uavs),
            first_seen,
        )
        result = original_step(*args, **kwargs)
        _remember(
            list(self.active.values())
            + list(self.service_uavs.values())
            + list(self.completed_uavs),
            first_seen,
        )
        return result

    def resolved(self: TerminalParallelEnv, uav: Any, outcome: Any, reason: Any):
        _remember([uav], first_seen)
        row = _row_from_uav(
            uav,
            first_seen=first_seen,
            terminal_time=self.now,
            outcome=outcome,
            reason=reason,
            source="_resolve",
            phase_before_terminal=getattr(uav, "phase", None),
        )
        result = original_resolve(uav, outcome, reason)
        row["post_terminal_outcome"] = _enum_value(getattr(uav, "terminal_outcome", None))
        row["post_terminal_reason"] = getattr(uav, "terminal_reason", None)
        captured.append(row)
        return result

    env.step = MethodType(stepped, env)
    env._resolve = MethodType(resolved, env)
    return captured, first_seen


def _completed_rows(
    env: TerminalParallelEnv,
    first_seen: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for uav in list(getattr(env, "completed_uavs", ())):
        rows.append(
            _row_from_uav(
                uav,
                first_seen=first_seen,
                terminal_time=getattr(uav, "completion_time", None),
                outcome=getattr(uav, "terminal_outcome", None),
                reason="ground_service_complete",
                source="completed_uavs",
            )
        )
    return rows


def _merge_terminal_rows(
    captured: Sequence[Mapping[str, Any]],
    completed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_arrival: dict[int, dict[str, Any]] = {}
    for row in list(captured) + list(completed):
        if row.get("arrival_id") is None:
            raise LifecycleMetricsError("terminal row has no arrival_id")
        arrival_id = int(row["arrival_id"])
        if arrival_id in by_arrival:
            raise LifecycleMetricsError(f"duplicate terminal row for arrival_id {arrival_id}")
        by_arrival[arrival_id] = dict(row)
    return [by_arrival[key] for key in sorted(by_arrival)]


def _wait_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if _number(row.get("wait_seconds")) is not None
        and _number(row.get("base_priority")) is not None
    ]


def _scope_summary(rows: Sequence[Mapping[str, Any]], *, low_soc_threshold: float) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    waits = _wait_rows(rows)
    by_priority: dict[str, dict[str, Any]] = {}
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        priority = _number(row.get("base_priority"))
        if priority is not None:
            grouped[int(priority)].append(row)
    for priority, group in sorted(grouped.items()):
        observed = _wait_rows(group)
        by_priority[str(priority)] = {
            "terminal_count": len(group),
            "count": len(observed),
            "mean_wait_seconds": (
                None if not observed else sum(float(row["wait_seconds"]) for row in observed) / len(observed)
            ),
            "outcomes": dict(sorted(Counter(str(row.get("outcome")) for row in group).items())),
        }
    low_soc_rows = [
        row for row in rows
        if _number(row.get("initial_soc")) is not None
        and float(row["initial_soc"]) <= float(low_soc_threshold)
    ]
    low_soc_waits = _wait_rows(low_soc_rows)
    landing_rows = [
        row for row in low_soc_rows
        if str(row.get("outcome")) in SUCCESS_OUTCOMES
        and _number(row.get("request_time")) is not None
        and _number(row.get("touchdown_time")) is not None
        and _number(row.get("final_soc")) is not None
    ]
    terminal_soc_rows = [
        row for row in low_soc_rows if _number(row.get("final_soc")) is not None
    ]
    initial_soc = [_number(row.get("initial_soc")) for row in rows]
    initial_soc = [value for value in initial_soc if value is not None]
    priorities = [_number(row.get("base_priority")) for row in waits]
    weighted = None
    if waits and all(value is not None for value in priorities):
        weight_sum = sum(float(row["base_priority"]) for row in waits)
        weighted = (
            sum(float(row["wait_seconds"]) * float(row["base_priority"]) for row in waits) / weight_sum
            if weight_sum else None
        )
    touchdown_rows = [
        row for row in rows
        if _number(row.get("request_time")) is not None
        and _number(row.get("touchdown_time")) is not None
    ]
    return {
        "row_count": len(rows),
        "observed_wait_count": len(waits),
        "rows_missing_wait_or_priority": len(rows) - len(waits),
        "outcomes": dict(sorted(Counter(str(row.get("outcome")) for row in rows).items())),
        "mean_wait_seconds": (
            None if not waits else sum(float(row["wait_seconds"]) for row in waits) / len(waits)
        ),
        "priority_weighted_wait_seconds": weighted,
        "by_priority": by_priority,
        "initial_soc_min": None if not initial_soc else min(initial_soc),
        "initial_soc_max": None if not initial_soc else max(initial_soc),
        "low_soc": {
            "threshold": float(low_soc_threshold),
            "count": len(low_soc_rows),
            "observed_wait_count": len(low_soc_waits),
            "mean_wait_seconds": (
                None if not low_soc_waits
                else sum(float(row["wait_seconds"]) for row in low_soc_waits) / len(low_soc_waits)
            ),
            "landing_soc_observed_count": len(landing_rows),
            "mean_landing_soc": (
                None if not landing_rows
                else sum(float(row["final_soc"]) for row in landing_rows) / len(landing_rows)
            ),
            "terminal_soc_observed_count": len(terminal_soc_rows),
            "mean_terminal_soc": (
                None if not terminal_soc_rows
                else sum(float(row["final_soc"]) for row in terminal_soc_rows) / len(terminal_soc_rows)
            ),
        },
        "request_to_touchdown_observed_count": len(touchdown_rows),
        "mean_request_to_touchdown_seconds": (
            None if not touchdown_rows else sum(
                float(row["touchdown_time"]) - float(row["request_time"])
                for row in touchdown_rows
            ) / len(touchdown_rows)
        ),
    }


def summarize_terminal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    low_soc_threshold: float,
    requested_count: int | None = None,
    missing_arrival_ids: Sequence[int] = (),
    enforce_unique: bool = True,
) -> dict[str, Any]:
    """Summarize terminal rows without replacing missing fields with zeros."""
    normalized = [dict(row) for row in rows]
    if enforce_unique:
        ids = [row.get("arrival_id") for row in normalized]
        if any(value is None for value in ids) or len(set(ids)) != len(ids):
            raise LifecycleMetricsError("terminal rows must have unique arrival_id values")
    requested = len(normalized) if requested_count is None else int(requested_count)
    missing = sorted({int(value) for value in missing_arrival_ids})
    success_rows = [row for row in normalized if str(row.get("outcome")) in SUCCESS_OUTCOMES]
    summary = _scope_summary(normalized, low_soc_threshold=low_soc_threshold)
    summary.update({
        "scope": "all_requested_terminal_rows",
        "requested_count": requested,
        "observed_terminal_count": len(normalized),
        "missing_request_count": len(missing),
        "missing_arrival_ids": missing,
        "requested_denominator": requested,
        "observed_metrics_denominator": summary["observed_wait_count"],
        "success_only": {
            "scope": "successful_terminal_rows",
            "success_outcomes": sorted(SUCCESS_OUTCOMES),
            **_scope_summary(success_rows, low_soc_threshold=low_soc_threshold),
        },
        "terminal_time_field": "terminal_time",
        "failure_terminal_time_not_touchdown": all(
            str(row.get("outcome")) in SUCCESS_OUTCOMES
            or row.get("terminal_time") is None
            or row.get("touchdown_time") is None
            or float(row["terminal_time"]) != float(row["touchdown_time"])
            for row in normalized
        ),
        "rows": normalized,
    })
    return summary


def terminal_lifecycle_metrics(
    config: Any,
    action_tape: str | Path,
    low_soc_threshold: float,
) -> dict[str, Any]:
    """Replay one tape and return failure-inclusive lifecycle metrics."""
    env = TerminalParallelEnv(config)
    captured, first_seen = _capture_terminal_rows(env)
    replay = SafetyTraceRecorder.replay_action_tape(env, action_tape)
    if not isinstance(replay, Mapping) or replay.get("exact") is not True:
        raise LifecycleMetricsError("action-tape replay was not exact")
    if (
        replay.get("steps_expected") is not None
        and replay.get("steps_replayed") is not None
        and int(replay["steps_expected"]) != int(replay["steps_replayed"])
    ):
        raise LifecycleMetricsError("action-tape replay step count was not exact")
    rows = _merge_terminal_rows(captured, _completed_rows(env, first_seen))

    ledger = getattr(env, "ledger", None)
    ledger_requested = {
        int(value) for value in getattr(ledger, "requested", set())
    }
    requested_count = int(getattr(config, "total_requested_uavs"))
    expected_ids = set(range(requested_count))
    unexpected_ledger_requests = sorted(ledger_requested - expected_ids)
    if unexpected_ledger_requests:
        raise LifecycleMetricsError(
            f"ledger contains unexpected requested arrival IDs: {unexpected_ledger_requests}"
        )
    observed_ids = {int(row["arrival_id"]) for row in rows}
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    ledger_outcomes = getattr(ledger, "outcomes", {})
    ledger_values = {
        int(key): str(_enum_value(value)) for key, value in ledger_outcomes.items()
    }
    row_values = {int(row["arrival_id"]): str(row.get("outcome")) for row in rows}
    if unexpected_ids:
        raise LifecycleMetricsError(f"terminal rows contain unexpected arrival IDs: {unexpected_ids}")
    for arrival_id, outcome in ledger_values.items():
        if arrival_id in row_values and row_values[arrival_id] != outcome:
            raise LifecycleMetricsError(f"ledger outcome mismatch for arrival_id {arrival_id}")
    rows_without_ledger_outcome = sorted(observed_ids - set(ledger_values))
    if rows_without_ledger_outcome:
        raise LifecycleMetricsError(
            f"ledger is missing terminal outcomes for arrival IDs: {rows_without_ledger_outcome}"
        )
    ledger_without_rows = sorted(set(ledger_values) - observed_ids)
    ledger_unresolved_requested = sorted(ledger_requested - set(ledger_values))
    summary = summarize_terminal_rows(
        rows,
        low_soc_threshold=float(low_soc_threshold),
        requested_count=requested_count,
        missing_arrival_ids=missing_ids,
        enforce_unique=True,
    )
    summary.update({
        "replay_exact": True,
        "steps_expected": replay.get("steps_expected"),
        "steps_replayed": replay.get("steps_replayed"),
        "requested_arrival_ids": sorted(expected_ids),
        "observed_arrival_ids": sorted(observed_ids),
        "unexpected_arrival_ids": unexpected_ids,
        "ledger_requested_ids": sorted(ledger_requested),
        "ledger_outcomes_without_rows": ledger_without_rows,
        "ledger_unresolved_requested_ids": ledger_unresolved_requested,
        "rows_without_ledger_outcome": rows_without_ledger_outcome,
        "all_requests_observed": not missing_ids,
        "ledger_alignment": not unexpected_ids and not ledger_without_rows and
        not ledger_unresolved_requested and not rows_without_ledger_outcome and not any(
            arrival_id in row_values and row_values[arrival_id] != outcome
            for arrival_id, outcome in ledger_values.items()
        ),
    })
    return summary


__all__ = [
    "LifecycleMetricsError",
    "SUCCESS_OUTCOMES",
    "summarize_terminal_rows",
    "terminal_lifecycle_metrics",
]
