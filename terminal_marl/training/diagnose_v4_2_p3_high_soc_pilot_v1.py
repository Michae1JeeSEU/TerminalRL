"""Diagnostic-only high-SOC dense24 paired pilot.

The pilot reuses the established P3 safety/energy diagnostic runner.  It
changes only the initial SOC sampling range: the V4 protocol is loaded into a
deep copy and its ``initial_soc_min``/``initial_soc_max`` attributes are set to
``0.90`` and ``1.00``.  The source YAML, policy, and underlying runner remain
unchanged.  This is a bounded paired screen for the rule controller and the
frozen original K=3 checkpoint; it is not a training or qualification run.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from terminal_marl.training import (
    diagnose_v4_2_p3_safety_energy_sensitivity_v1 as safety_energy,
)
from terminal_marl.training import (
    evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness,
)


STAGE = "V4_2_P3_HIGH_SOC_PILOT_V1"
SCENARIO = safety_energy.SCENARIO
SCENARIO_PROTOCOL = safety_energy.SCENARIO_PROTOCOL
HIGH_SOC_MIN = 0.90
HIGH_SOC_MAX = 1.00
SAFE_DISTANCE_M = 80.0
PROTECTED_FLOOR_M = safety_energy.PROTECTED_FLOOR_M
COLLISION_DISTANCE_M = safety_energy.COLLISION_DISTANCE_M
ENERGY_CAPACITY = safety_energy.ENERGY_CAPACITY
FULL_HORIZON_SECONDS = safety_energy.FULL_HORIZON_SECONDS
ORIGINAL_K3_SHA256 = safety_energy.ORIGINAL_K3_SHA256
DEFAULT_CONTROLLERS = ("rule", "original_k3")
CONTROLLERS = frozenset(DEFAULT_CONTROLLERS)


class DiagnosticInputError(ValueError):
    """Raised for invalid high-SOC pilot inputs."""


def _jsonable(value: Any) -> Any:
    """Use the established diagnostic JSON conversion when available."""
    return safety_energy._jsonable(value)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write progress and manifest files atomically."""
    safety_energy._write_json_atomic(path, value)


def override_high_soc_protocol(
    raw_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an isolated V4 protocol copy with the high-SOC range applied."""
    protocol = copy.deepcopy(dict(raw_protocol))
    attributes = protocol.get("attributes")
    if not isinstance(attributes, dict):
        raise DiagnosticInputError("scenario protocol attributes must be a mapping")
    attributes["initial_soc_min"] = HIGH_SOC_MIN
    attributes["initial_soc_max"] = HIGH_SOC_MAX
    return protocol


def load_scenario_protocol(
    path: str | Path = SCENARIO_PROTOCOL,
) -> dict[str, Any]:
    """Load and isolate the V4 YAML, then apply only the SOC override."""
    source = safety_energy.load_scenario_protocol(path)
    return override_high_soc_protocol(source)


# Explicit aliases make the isolation boundary easy for small callers to use.
load_high_soc_protocol = load_scenario_protocol
high_soc_protocol = override_high_soc_protocol


def scenario_config(
    *,
    raw_protocol: Mapping[str, Any] | None = None,
):
    """Build the fixed 80 m dense24 config using the isolated SOC protocol."""
    protocol = (
        load_scenario_protocol()
        if raw_protocol is None
        else override_high_soc_protocol(raw_protocol)
    )
    config = safety_energy.scenario_config(
        SAFE_DISTANCE_M,
        raw_protocol=protocol,
    )
    reward = protocol.get("reward")
    if not isinstance(reward, Mapping):
        raise DiagnosticInputError("scenario protocol reward must be a mapping")
    # The safety/energy runner owns the P3 environment construction.  Restore
    # the V4 protocol's explicit urgency-reward values on the resulting config
    # so the high-SOC screen changes only the SOC sampling range.
    return replace(
        config,
        low_level_priority_time_weight=float(
            reward["low_level_priority_time_weight"]
        ),
        low_level_soc_urgency_time_weight=float(
            reward["low_level_soc_urgency_time_weight"]
        ),
        low_level_soc_urgency_threshold=float(
            reward["low_level_soc_urgency_threshold"]
        ),
    )


def build_case_matrix(
    seeds: Sequence[int],
    controllers: Sequence[str] = DEFAULT_CONTROLLERS,
) -> list[dict[str, Any]]:
    """Return one fixed-80 m rule/original-K3 pair for every seed."""
    values = tuple(str(controller) for controller in controllers)
    if values != DEFAULT_CONTROLLERS:
        raise DiagnosticInputError(
            "the high-SOC pilot requires the paired rule/original_k3 controllers"
        )
    return safety_energy.build_case_matrix(
        seeds,
        safe_distances=(SAFE_DISTANCE_M,),
        controllers=values,
    )


matrix_cases = build_case_matrix
construct_case_matrix = build_case_matrix


def run_case(
    case: Mapping[str, Any],
    *,
    output: Path,
    policy: Any | None = None,
    raw_protocol: Mapping[str, Any] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run one case with the high-SOC config and established helpers."""
    case_dir = output / str(case["case_id"])
    record = safety_energy._base_record(case, case_dir=case_dir)
    try:
        config = scenario_config(raw_protocol=raw_protocol)
        record["requests"] = int(config.total_requested_uavs)
        if case_dir.exists():
            raise FileExistsError(f"case output already exists: {case_dir}")
        max_steps_value = (
            int(max_steps)
            if max_steps is not None
            else int(round(FULL_HORIZON_SECONDS / float(config.dt)))
        )
        if max_steps_value < 1:
            raise DiagnosticInputError("max_steps must be positive")
        if str(case["controller"]) == "original_k3":
            if policy is None:
                raise DiagnosticInputError("original_k3 requires a loaded checkpoint")
            _, summary = safety_energy.student_rollout_v4_2.rollout(
                policy,
                config,
                seed=int(case["seed"]),
                max_steps=max_steps_value,
                output=case_dir,
                deterministic=True,
                collect_labels=False,
            )
        elif str(case["controller"]) == "rule":
            summary = safety_energy.curriculum.episode(
                config,
                int(case["seed"]),
                case_dir,
                policy=None,
                max_steps=max_steps_value,
            )
        else:
            raise DiagnosticInputError(f"unsupported controller: {case['controller']}")
        tape_path = summary.get("action_tape")
        record["artifacts"]["action_tape"] = (
            None if tape_path is None else str(tape_path)
        )
        telemetry, replay, replay_error, lifecycle = safety_energy._replay_telemetry(
            config, tape_path
        )
        record = safety_energy._merge_summary_record(
            record, summary, telemetry, replay, replay_error
        )
        if lifecycle is not None:
            record.update(lifecycle)
        return record
    except Exception as exc:
        record["exception"] = f"{type(exc).__name__}: {exc}"
        return record


def _as_int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _complete(record: Mapping[str, Any]) -> bool:
    """Require a successful terminal episode with every request resolved."""
    requests = _as_int_or_none(record.get("requests"))
    completed = _as_int_or_none(record.get("completed"))
    return bool(
        record.get("status") == "completed"
        and not record.get("exception")
        and bool(record.get("terminal"))
        and requests is not None
        and completed is not None
        and requests >= 0
        and completed == requests
        and _as_int_or_none(record.get("unresolved")) == 0
    )


def _counter(record: Mapping[str, Any], key: str) -> int | None:
    value = _as_int_or_none(record.get(key))
    return value


def _record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the bounded pilot evidence and preserve missing values."""
    requests = _as_int_or_none(record.get("requests"))
    completed = _as_int_or_none(record.get("completed"))
    diversions = _counter(record, "low_soc_diversions")
    safety = {
        key: _counter(record, key)
        for key in (
            "safety_invariant_errors",
            "uncertified_execution",
            "hard_separation_violations",
            "nmac",
            "collisions",
        )
    }
    replay = record.get("replay")
    replay_summary = None
    if isinstance(replay, Mapping):
        replay_summary = {
            "exact": replay.get("exact"),
            "steps_replayed": replay.get("steps_replayed"),
            "steps_expected": replay.get("steps_expected"),
        }
    return {
        "complete": _complete(record),
        "requests": requests,
        "completed": completed,
        "diversions": diversions,
        "safety": safety,
        "replay": replay_summary,
        "error": record.get("exception"),
    }


def _with_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a runner record and attach stable high-SOC summary fields."""
    value = dict(record)
    summary = _record_summary(value)
    value["complete"] = summary["complete"]
    value["diversions"] = summary["diversions"]
    value["safety_summary"] = summary["safety"]
    value["replay_summary"] = summary["replay"]
    value["pilot_summary"] = summary
    return value


def _zero_safety(summary: Mapping[str, Any]) -> bool:
    safety = summary.get("safety")
    if not isinstance(safety, Mapping):
        return False
    values = list(safety.values())
    return bool(values) and all(value == 0 for value in values)


def _exact_replay(summary: Mapping[str, Any]) -> bool:
    replay = summary.get("replay")
    return isinstance(replay, Mapping) and replay.get("exact") is True


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_record_count: int | None = None,
) -> dict[str, Any]:
    """Aggregate records with false gates for incomplete/error evidence."""
    summaries = [
        row.get("pilot_summary")
        if isinstance(row.get("pilot_summary"), Mapping)
        else _record_summary(row)
        for row in records
    ]
    count_gate = expected_record_count is None or len(records) == int(expected_record_count)
    records_complete = bool(summaries) and all(
        summary.get("complete") is True for summary in summaries
    )
    no_errors = bool(summaries) and all(not summary.get("error") for summary in summaries)
    safety_all_zero = bool(summaries) and all(_zero_safety(summary) for summary in summaries)
    exact_replay = bool(summaries) and all(_exact_replay(summary) for summary in summaries)
    diagnostic_pass = bool(
        count_gate
        and records_complete
        and no_errors
        and safety_all_zero
        and exact_replay
    )
    return {
        "record_count": len(records),
        "expected_record_count": None
        if expected_record_count is None
        else int(expected_record_count),
        "completed": sum(
            int(summary["completed"])
            for summary in summaries
            if summary.get("completed") is not None
        ),
        "requests": sum(
            int(summary["requests"])
            for summary in summaries
            if summary.get("requests") is not None
        ),
        "diversions": sum(
            int(summary["diversions"])
            for summary in summaries
            if summary.get("diversions") is not None
        ),
        "gates": {
            "expected_record_count": count_gate,
            "all_records_complete": records_complete,
            "no_record_errors": no_errors,
            "safety_all_zero": safety_all_zero,
            "exact_replay": exact_replay,
        },
        "diagnostic_pass": diagnostic_pass,
        "scope": "diagnostic-only high-SOC dense24 paired pilot; not formal qualification",
    }


def run_matrix(
    *,
    output: str | Path,
    checkpoint: str | Path | None,
    seeds: Sequence[int],
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run the bounded paired pilot and write progress plus manifest JSON."""
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"diagnostic output already exists: {output_path}")
    if max_steps is not None and int(max_steps) < 1:
        raise DiagnosticInputError("max_steps must be positive")
    cases = build_case_matrix(seeds)
    if checkpoint is None:
        raise DiagnosticInputError("--checkpoint is required for the paired pilot")
    checkpoint_path = Path(checkpoint)
    checkpoint_sha256 = effectiveness._sha256(checkpoint_path)
    if checkpoint_sha256 != ORIGINAL_K3_SHA256:
        raise DiagnosticInputError("checkpoint is not the frozen original K3 candidate")
    policy, _ = effectiveness.load_candidate_checkpoint(checkpoint_path)
    raw_protocol = load_scenario_protocol()
    output_path.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    for case in cases:
        record = run_case(
            case,
            output=output_path,
            policy=policy,
            raw_protocol=raw_protocol,
            max_steps=max_steps,
        )
        enriched = _with_summary(record)
        records.append(enriched)
        _write_json_atomic(
            output_path / "progress.json",
            {
                "stage": STAGE,
                "diagnostic_only": True,
                "PPO_ALLOWED": False,
                "completed_records": len(records),
                "expected_records": len(cases),
                "last_record": enriched,
                "summary": aggregate_records(
                    records, expected_record_count=len(cases)
                ),
            },
        )

    aggregate = aggregate_records(records, expected_record_count=len(cases))
    manifest = {
        "stage": STAGE,
        "diagnostic_only": True,
        "PPO_ALLOWED": False,
        "qualification": False,
        "formal_qualification": False,
        "P3_QUALIFIED": False,
        "checkpoint_promotable": False,
        "scenario": SCENARIO,
        "scenario_protocol": str(SCENARIO_PROTOCOL),
        "protocol_overrides": {
            "initial_soc_min": HIGH_SOC_MIN,
            "initial_soc_max": HIGH_SOC_MAX,
        },
        "constants": {
            "safe_distance_m": SAFE_DISTANCE_M,
            "protected_floor_m": PROTECTED_FLOOR_M,
            "collision_distance_m": COLLISION_DISTANCE_M,
            "energy_capacity": ENERGY_CAPACITY,
            "max_simulation_seconds": FULL_HORIZON_SECONDS,
        },
        "matrix": {
            "seeds": [int(value) for value in seeds],
            "safe_distances": [SAFE_DISTANCE_M],
            "controllers": list(DEFAULT_CONTROLLERS),
            "expected_records": len(cases),
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "records": records,
        "aggregate": aggregate,
        # PASS is explicitly a diagnostic screen result, never a qualification
        # authorization.  The authorization fields above remain false.
        "PASS": bool(aggregate["diagnostic_pass"]),
        "pass_scope": "diagnostic_only; not formal qualification",
        "completed_records": len(records),
        "failed_records": sum(row.get("status") != "completed" for row in records),
    }
    _write_json_atomic(output_path / "manifest.json", manifest)
    return manifest


run = run_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional bounded smoke horizon; omit for the 1800 s dense24 horizon",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    result = run_matrix(
        output=args.output,
        checkpoint=args.checkpoint,
        seeds=args.seeds,
        max_steps=args.max_steps,
    )
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "output": str(args.output),
                "records": result["completed_records"],
                "failed_records": result["failed_records"],
                "diagnostic_pass": result["aggregate"]["diagnostic_pass"],
                "PPO_ALLOWED": result["PPO_ALLOWED"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
