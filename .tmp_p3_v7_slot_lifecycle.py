"""Failure-inclusive lifecycle evidence from four existing V7 action tapes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import tempfile
import types
from typing import Any, Mapping
from dataclasses import replace

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator


SEED = 873106
SCENARIO = "dense24"
LOW_SOC_THRESHOLD = 0.65
CLOSED_ROOT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_run0_slot_closed_loop_20260926")
OUTPUT_ROOT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_run0_slot_lifecycle_20260926")


class LifecycleInputError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleInputError(message)


def _jsonable(value: Any) -> Any:
    return evaluator._jsonable(value)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _cell_inputs() -> list[dict[str, Any]]:
    manifest = _read_json(CLOSED_ROOT / "manifest.json")
    cells = []
    for cell in manifest.get("cells", []):
        record = cell.get("record") or {}
        tape = Path(str(record.get("action_tape", "")))
        _require(tape.is_file(), f"cell action tape missing: {tape}")
        _require(cell.get("scenario") == SCENARIO and int(cell.get("seed", -1)) == SEED, "unexpected cell binding")
        cells.append({
            "label": str(cell["label"]),
            "reconcile": bool(cell["reconcile_late_merge_slots_enabled"]),
            "tape": tape,
            "record": record,
            "cell_manifest": cell,
        })
    _require(len(cells) == 4, f"expected 4 closed-loop cells, got {len(cells)}")
    return cells


def _capture_resolve(
    env: TerminalParallelEnv,
    rows: list[dict[str, Any]],
    first_seen: dict[int, dict[str, Any]],
) -> None:
    original = env._resolve

    def remember(values) -> None:
        for uav in values:
            arrival_id = int(uav.arrival_id)
            if arrival_id in first_seen:
                continue
            first_seen[arrival_id] = {
                "initial_soc": float(
                    (float(uav.energy_remaining) + float(uav.energy_consumed)) /
                    float(uav.energy_capacity)
                ),
                "initial_soc_source": "first_seen_uav",
            }

    original_step = env.step

    def stepped(self, *args, **kwargs):
        remember(list(self.active.values()) + list(self.service_uavs.values()))
        result = original_step(*args, **kwargs)
        remember(list(self.active.values()) + list(self.service_uavs.values()) + list(self.completed_uavs))
        return result

    env.step = types.MethodType(stepped, env)

    def wrapped(self, uav, outcome, reason):
        remember([uav])
        initial = first_seen.get(int(uav.arrival_id))
        initial_soc = (
            float(initial["initial_soc"])
            if initial is not None else
            (float(uav.energy_remaining) + float(uav.energy_consumed)) / float(uav.energy_capacity)
        )
        captured = {
            "arrival_id": int(uav.arrival_id),
            "uid": int(uav.uid),
            "base_priority": int(uav.base_priority),
            "initial_soc": float(initial_soc),
            "initial_soc_source": "first_seen_uav" if initial is not None else "terminal_state_sum",
            "request_time": float(uav.request_time),
            "terminal_time": float(self.now),
            "phase_before_terminal": getattr(uav.phase, "value", uav.phase),
            "external_wait_time": float(uav.external_wait_time),
            "internal_wait_time": float(uav.internal_wait_time),
            "wait_seconds": float(uav.external_wait_time + uav.internal_wait_time),
            "energy_remaining": float(uav.energy_remaining),
            "energy_consumed": float(uav.energy_consumed),
            "energy_capacity": float(uav.energy_capacity),
            "final_soc": float(uav.soc),
            "outcome": getattr(outcome, "value", outcome),
            "reason": str(reason),
            "touchdown_time": None if uav.touchdown_time is None else float(uav.touchdown_time),
            "completion_time": None if uav.completion_time is None else float(uav.completion_time),
            "emergency_landing": bool(uav.emergency_landing),
            "source": "_resolve",
        }
        result = original(uav, outcome, reason)
        captured["post_terminal_outcome"] = getattr(getattr(uav, "terminal_outcome", None), "value", uav.terminal_outcome)
        captured["post_terminal_reason"] = uav.terminal_reason
        rows.append(captured)
        return result

    env._resolve = types.MethodType(wrapped, env)


def _completed_rows(env: TerminalParallelEnv, first_seen: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for uav in env.completed_uavs:
        initial = first_seen.get(int(uav.arrival_id))
        initial_soc = (
            float(initial["initial_soc"])
            if initial is not None else
            (float(uav.energy_remaining) + float(uav.energy_consumed)) / float(uav.energy_capacity)
        )
        rows.append({
            "arrival_id": int(uav.arrival_id),
            "uid": int(uav.uid),
            "base_priority": int(uav.base_priority),
            "initial_soc": float(initial_soc),
            "initial_soc_source": "first_seen_uav" if initial is not None else "terminal_state_sum",
            "request_time": float(uav.request_time),
            "terminal_time": None if uav.completion_time is None else float(uav.completion_time),
            "phase_before_terminal": getattr(uav.phase, "value", uav.phase),
            "external_wait_time": float(uav.external_wait_time),
            "internal_wait_time": float(uav.internal_wait_time),
            "wait_seconds": float(uav.external_wait_time + uav.internal_wait_time),
            "energy_remaining": float(uav.energy_remaining),
            "energy_consumed": float(uav.energy_consumed),
            "energy_capacity": float(uav.energy_capacity),
            "final_soc": float(uav.soc),
            "outcome": getattr(uav.terminal_outcome, "value", uav.terminal_outcome),
            "reason": "ground_service_complete",
            "touchdown_time": None if uav.touchdown_time is None else float(uav.touchdown_time),
            "completion_time": None if uav.completion_time is None else float(uav.completion_time),
            "emergency_landing": bool(uav.emergency_landing),
            "source": "completed_uavs",
        })
    return rows


def _merge_terminal_rows(
    env: TerminalParallelEnv,
    captured: list[dict[str, Any]],
    first_seen: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_arrival = {int(row["arrival_id"]): row for row in captured}
    for row in _completed_rows(env, first_seen):
        by_arrival.setdefault(int(row["arrival_id"]), row)
    for outcome in env.energy_outcomes:
        arrival_id = outcome.get("arrival_id")
        if arrival_id is None or int(arrival_id) in by_arrival:
            continue
        by_arrival[int(arrival_id)] = {
            "arrival_id": int(arrival_id),
            "uid": outcome.get("uid"),
            "base_priority": None,
            "initial_soc": None,
            "initial_soc_source": "unavailable",
            "request_time": None,
            "terminal_time": outcome.get("time"),
            "phase_before_terminal": outcome.get("phase"),
            "external_wait_time": None,
            "internal_wait_time": None,
            "wait_seconds": None,
            "energy_remaining": outcome.get("energy_remaining"),
            "energy_consumed": outcome.get("energy_consumed"),
            "energy_capacity": outcome.get("energy_capacity"),
            "final_soc": outcome.get("soc"),
            "outcome": outcome.get("outcome"),
            "reason": outcome.get("reason"),
            "touchdown_time": None,
            "completion_time": None,
            "emergency_landing": outcome.get("emergency_landing"),
            "source": "energy_outcomes_only",
            "missing_fields": ["base_priority", "initial_soc", "request_time", "wait_seconds"],
        }
    return [by_arrival[key] for key in sorted(by_arrival)]


def _replay_outcomes(env: TerminalParallelEnv) -> dict[str, str]:
    return {str(key): getattr(value, "value", value) for key, value in env.ledger.outcomes.items()}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["arrival_id"]))
    complete_wait_rows = [
        row for row in ordered
        if row.get("wait_seconds") is not None and row.get("base_priority") is not None
    ]
    waits = [float(row["wait_seconds"]) for row in complete_wait_rows]
    by_priority: dict[str, dict[str, Any]] = {}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in complete_wait_rows:
        grouped[int(row["base_priority"])].append(row)
    for priority, group in sorted(grouped.items()):
        by_priority[str(priority)] = {
            "count": len(group),
            "mean_wait_seconds": float(statistics.fmean(row["wait_seconds"] for row in group)),
            "outcomes": dict(sorted(Counter(str(row["outcome"]) for row in group).items())),
        }
    low_soc = [row for row in complete_wait_rows if float(row["initial_soc"]) <= LOW_SOC_THRESHOLD]
    return {
        "rows": ordered,
        "row_count": len(ordered),
        "outcomes": dict(sorted(Counter(str(row["outcome"]) for row in ordered).items())),
        "priority_weighted_wait_seconds": None if not complete_wait_rows else float(
            sum(float(row["wait_seconds"]) * float(row["base_priority"]) for row in complete_wait_rows) /
            sum(float(row["base_priority"]) for row in complete_wait_rows)
        ),
        "mean_wait_seconds": None if not waits else float(statistics.fmean(waits)),
        "by_priority": by_priority,
        "low_soc": {
            "threshold": LOW_SOC_THRESHOLD,
            "count": len(low_soc),
            "mean_wait_seconds": None if not low_soc else float(statistics.fmean(row["wait_seconds"] for row in low_soc)),
        },
        "terminal_time_field": "terminal_time",
        "failure_touchdown_time_is_null_when_not_recorded": all(
            row["outcome"] in {"completed", "emergency_landed"} or row["touchdown_time"] is None
            for row in ordered
        ),
    }


def _prefix_cell(cell: Mapping[str, Any], config: Any) -> dict[str, Any]:
    payload = _read_json(Path(cell["tape"]))
    rows = payload.get("action_tape", [])
    _require(len(rows) >= 2, "tape shorter than prefix")
    prefix_payload = {key: value for key, value in payload.items() if key not in {"records", "action_tape"}}
    prefix_payload["action_tape"] = rows[:2]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(prefix_payload, handle, allow_nan=False)
        prefix_path = Path(handle.name)
    try:
        env = TerminalParallelEnv(config)
        captured: list[dict[str, Any]] = []
        _capture_resolve(env, captured, {})
        replay = SafetyTraceRecorder.replay_action_tape(env, prefix_path)
        return {
            "label": cell["label"],
            "reconcile": cell["reconcile"],
            "hook_bound": True,
            "captured_rows": len(captured),
            "replay": {key: replay.get(key) for key in ("exact", "steps_expected", "steps_replayed", "reset_state_match", "config_match")},
            "json_serializable": json.dumps(_jsonable({"captured_rows": captured, "replay": replay}), allow_nan=False) is not None,
        }
    finally:
        prefix_path.unlink(missing_ok=True)


def _run_full(cells: list[dict[str, Any]], config_off: Any, output_root: Path) -> dict[str, Any]:
    _require(not output_root.exists(), f"output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []
    try:
        for cell in cells:
            config = config_off
            if cell["reconcile"]:
                config = replace(config_off, reconcile_late_merge_slots_enabled=True)
            env = TerminalParallelEnv(config)
            captured: list[dict[str, Any]] = []
            first_seen: dict[int, dict[str, Any]] = {}
            _capture_resolve(env, captured, first_seen)
            replay = SafetyTraceRecorder.replay_action_tape(env, cell["tape"])
            replay_outcomes = _replay_outcomes(env)
            original_raw = ((cell["record"].get("outcomes") or {}).get("raw_outcomes") or {})
            terminal_rows = _merge_terminal_rows(env, captured, first_seen)
            missing = sorted(set(replay_outcomes) - {str(row["arrival_id"]) for row in terminal_rows})
            row_ids = [str(row["arrival_id"]) for row in terminal_rows]
            row_outcomes = {str(row["arrival_id"]): str(row.get("outcome")) for row in terminal_rows}
            cell_checks = {
                "full_tape_exact": bool(replay.get("exact")) and replay.get("steps_replayed") == replay.get("steps_expected"),
                "unique_arrival_rows_24": len(row_ids) == 24 and len(set(row_ids)) == 24,
                "rows_have_lifecycle_fields": len(terminal_rows) == 24 and all(
                    row.get("base_priority") is not None
                    and row.get("initial_soc") is not None
                    and row.get("request_time") is not None
                    and row.get("wait_seconds") is not None
                    and row.get("terminal_time") is not None
                    for row in terminal_rows
                ),
                "failed_terminal_time_not_touchdown": all(
                    row.get("outcome") in {"completed", "emergency_landed"}
                    or row.get("touchdown_time") is None
                    for row in terminal_rows
                ),
                "ledger_arrival_ids_match": set(row_ids) == set(replay_outcomes),
                "ledger_outcomes_match_rows": row_outcomes == replay_outcomes,
                "closed_loop_outcomes_match": original_raw == replay_outcomes,
            }
            report = {
                "label": cell["label"],
                "reconcile_late_merge_slots_enabled": cell["reconcile"],
                "tape": str(cell["tape"]),
                "replay": {
                    "exact": bool(replay.get("exact")),
                    "steps_expected": replay.get("steps_expected"),
                    "steps_replayed": replay.get("steps_replayed"),
                    "outcomes_match_closed_loop": original_raw == replay_outcomes,
                    "replay_outcomes": replay_outcomes,
                },
                "checks": cell_checks,
                "terminal_rows": terminal_rows,
                "missing_resolve_rows": missing,
                "resolve_hook_rows": len(captured),
                "completed_uavs_rows": sum(row.get("source") == "completed_uavs" for row in terminal_rows),
                "failure_inclusive_lifecycle": _summary(terminal_rows),
                "completed_only_lifecycle": cell["cell_manifest"].get("lifecycle"),
                "closed_loop_outcomes": cell["record"].get("outcomes"),
                "safety": (cell["record"].get("summary") or {}).get("safety"),
            }
            reports.append(report)
            _write_json(output_root / "progress.json", {"processed_cells": len(reports), "expected_cells": 4, "cells": reports})
            if not all(cell_checks.values()):
                raise LifecycleInputError(
                    f"lifecycle cell checks failed: {cell['label']} late={cell['reconcile']} {cell_checks}"
                )
        result = {
            "stage": "P3_V7_RUN0_SLOT_LIFECYCLE_V1",
            "status": "COMPLETED",
            "cells": reports,
            "all_replay_exact": all(item["replay"]["exact"] for item in reports),
            "all_outcomes_match": all(item["replay"]["outcomes_match_closed_loop"] for item in reports),
            "all_rows_24": all(item["failure_inclusive_lifecycle"]["row_count"] == 24 for item in reports),
            "all_missing_resolve_rows_empty": all(not item["missing_resolve_rows"] for item in reports),
            "source_root": str(CLOSED_ROOT),
            "output": str(output_root),
        }
        _write_json(output_root / "manifest.json", result)
        return result
    except BaseException as exc:
        _write_json(output_root / "manifest.json", {
            "stage": "P3_V7_RUN0_SLOT_LIFECYCLE_V1", "status": "FAILED",
            "error": {"type": type(exc).__name__, "message": str(exc)}, "cells": reports,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    protocol = evaluator._protocol()
    config_off = evaluator._build_configs(protocol)[SCENARIO]
    cells = _cell_inputs()
    if not args.run_full:
        prefix = [_prefix_cell(
            cell,
            replace(config_off, reconcile_late_merge_slots_enabled=True) if cell["reconcile"] else config_off,
        ) for cell in cells]
        result = {"stage": "P3_V7_RUN0_SLOT_LIFECYCLE_V1", "status": "PREFIX_OK", "cells": prefix, "output_created": False}
    else:
        result = _run_full(cells, config_off, args.output)
    print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"PREFIX_OK", "COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
