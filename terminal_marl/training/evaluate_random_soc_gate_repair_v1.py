"""Development-only paired evaluation for the random-SOC gate repair.

The runner evaluates the frozen source policy and the separately trained gate
repair on one protocol-bound scenario/seed.  It reuses the current diagnostic
configuration and the existing P3 replay/lifecycle checks, but does not call a
formal protocol runner, update a policy, or qualify a checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import is_dataclass, fields
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.training import diagnose_trained_dense30_v2 as diagnostic
from terminal_marl.training import (
    diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as pilot,
)
from terminal_marl.training import (
    run_v4_2_p3_mixed_priority_soc_training_v4 as p3,
)
from terminal_marl.training import train_random_soc_gate_repair_v1 as repair
from terminal_marl.training import student_rollout_v4_2 as student

# Name the repair loader module explicitly in the report-facing seam.
newtrainer = repair


STAGE = "RANDOM_SOC_GATE_REPAIR_DEVELOPMENT_PAIR_V1"
PROTOCOL = p3.PROTOCOL
SCENARIOS = ("flow6", "flow12", "dense24")
PROTOCOL_SEEDS = {
    "flow6": 846301,
    "flow12": 846302,
    "dense24": 846303,
}

# These small wrappers keep the evaluation seam replaceable in bounded tests,
# while still following the live source modules if one of them is patched.
def build_scenario_config(scenario: str, **kwargs: Any) -> Any:
    return diagnostic.build_scenario_config(scenario, **kwargs)


def load_trained_checkpoint(path: str | Path, config: Any):
    return diagnostic.load_trained_checkpoint(path, config)


def load_repaired_checkpoint(
    path: str | Path, config: Any, *, expected_sha256: str
):
    return newtrainer.load_repaired_checkpoint(path, config, expected_sha256=expected_sha256)


def rollout(policy: Any, config: Any, **kwargs: Any):
    return student.rollout(policy, config, **kwargs)


def _rollout_gates(*args: Any, **kwargs: Any):
    return p3._rollout_gates(*args, **kwargs)


def _lifecycle_aggregate(*args: Any, **kwargs: Any):
    return p3._lifecycle_aggregate(*args, **kwargs)


def _dev_selection_gates(*args: Any, **kwargs: Any):
    return p3._dev_selection_gates(*args, **kwargs)


def lifecycle_metrics(*args: Any, **kwargs: Any):
    return pilot.lifecycle_metrics(*args, **kwargs)


def _policy_weight_digest(policy: Any) -> str:
    return diagnostic._policy_weight_digest(policy)


def _jsonable(value: Any) -> Any:
    """Convert report values without changing the values kept in arm rows."""

    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _load_yaml_protocol() -> Mapping[str, Any]:
    """Load only the YAML mapping needed by the existing P3 gate helpers."""

    raw = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, Mapping):
        raise ValueError("random-SOC development YAML must be a mapping")
    return raw


def _validate_binding(scenario: str, seed: int) -> None:
    scenario = str(scenario)
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    expected = PROTOCOL_SEEDS[scenario]
    if int(seed) != expected:
        raise ValueError(
            f"random-SOC development seed for {scenario} is fixed at {expected}, got {seed}"
        )


def _build_config(scenario: str, protocol: Mapping[str, Any]) -> Any:
    config = build_scenario_config(scenario, reconcile_late_merge_slots=False)
    scenario_spec = protocol.get("scenarios", {}).get(scenario)
    if not isinstance(scenario_spec, Mapping):
        raise ValueError(f"YAML scenario specification missing: {scenario}")
    expected_total = int(scenario_spec["total_requested_uavs"])
    expected_horizon = float(scenario_spec["full_horizon_seconds"])
    if int(config.total_requested_uavs) != expected_total:
        raise ValueError(
            f"scenario total changed for {scenario}: "
            f"{config.total_requested_uavs!r} != {expected_total!r}"
        )
    if float(config.max_simulation_seconds) != expected_horizon:
        raise ValueError(
            f"scenario horizon changed for {scenario}: "
            f"{config.max_simulation_seconds!r} != {expected_horizon!r}"
        )
    if bool(getattr(config, "reconcile_late_merge_slots_enabled", False)):
        raise ValueError("development pair requires late-merge reconciliation disabled")
    if bool(getattr(config, "teacher_enabled", False)):
        raise ValueError("development pair requires Teacher disabled")
    if str(getattr(config, "scheduler_mode", "execution")) != "execution":
        raise ValueError("development pair requires execution scheduler mode")
    return config


def _config_snapshot(config: Any) -> dict[str, Any]:
    values = vars(config) if hasattr(config, "__dict__") else {}
    return _jsonable(dict(values))


def _outcome_fields(summary: Mapping[str, Any], requests: int) -> dict[str, int]:
    raw_outcomes = summary.get("outcomes") or {}
    if not isinstance(raw_outcomes, Mapping):
        raw_outcomes = {}
    counts = Counter(
        str(getattr(value, "value", value)) for value in raw_outcomes.values()
    )
    unresolved = int(counts.get("unresolved", 0)) + max(
        0, int(requests) - len(raw_outcomes)
    )
    return {
        "normal": int(counts.get("completed", 0)),
        "emergency": int(counts.get("emergency_landed", 0)),
        "failed_energy": int(counts.get("failed_energy", 0)),
        "unresolved": int(unresolved),
    }


def _summary_has_safety_error(summary: Mapping[str, Any]) -> bool:
    if summary.get("safety_error"):
        return True
    safety = summary.get("safety") or {}
    if not isinstance(safety, Mapping):
        return False
    try:
        return int(safety.get("SafetyInvariantError", 0)) > 0
    except (TypeError, ValueError):
        return bool(safety.get("SafetyInvariantError"))


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _arm_pass(record: Mapping[str, Any]) -> bool:
    summary = record.get("summary")
    gates = record.get("gates")
    lifecycle = record.get("lifecycle")
    return bool(
        isinstance(summary, Mapping)
        and bool(summary.get("terminal", False))
        and int(record.get("completed", 0)) == int(record.get("requests", 0))
        and int(record.get("unresolved", 0)) == 0
        and record.get("weights_unchanged") is True
        and int(record.get("label_rows", 0)) == 0
        and bool(record.get("teacher_enabled")) is False
        and bool(record.get("PPO_ALLOWED")) is False
        and isinstance(gates, Mapping)
        and bool(gates)
        and all(bool(value) for value in gates.values())
        and isinstance(lifecycle, Mapping)
        and lifecycle.get("replay_exact") is True
    )


def _arm_invariants(record: Mapping[str, Any]) -> dict[str, bool]:
    summary = record.get("summary")
    return {
        "terminal": bool(isinstance(summary, Mapping) and summary.get("terminal", False)),
        "completed_equals_requests": int(record.get("completed", 0)) == int(record.get("requests", 0)),
        "unresolved_zero": int(record.get("unresolved", 0)) == 0,
        "weights_unchanged": record.get("weights_unchanged") is True,
        "label_rows_zero": int(record.get("label_rows", 0)) == 0,
        "teacher_disabled": bool(record.get("teacher_enabled")) is False,
        "PPO_ALLOWED_false": bool(record.get("PPO_ALLOWED")) is False,
        "gates_all": bool(
            isinstance(record.get("gates"), Mapping)
            and record.get("gates")
            and all(bool(value) for value in record["gates"].values())
        ),
        "lifecycle_replay_exact": bool(
            isinstance(record.get("lifecycle"), Mapping)
            and record["lifecycle"].get("replay_exact") is True
        ),
    }


def _loader_record(metadata: Any) -> Any:
    if not isinstance(metadata, Mapping):
        return metadata

    def _bounded_json(value: Any, *, depth: int = 0) -> Any:
        """Keep small loader provenance without retaining checkpoint payloads."""

        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if depth >= 4:
            return None
        if isinstance(value, Mapping):
            items = list(value.items())[:64]
            return {
                str(key): _bounded_json(item, depth=depth + 1)
                for key, item in items
            }
        if isinstance(value, (list, tuple)):
            return [_bounded_json(item, depth=depth + 1) for item in value[:64]]
        return None

    # The repaired loader returns the checkpoint model state and report in its
    # payload; arm provenance keeps only small binding/loader facts.
    scalar_keys = {
        "checkpoint", "checkpoint_sha256", "source_checkpoint_sha256", "source_sha256",
        "stage", "architecture_id", "temperature", "weight_digest_before",
        "weight_digest_after", "weight_digest_loaded_from_payload",
        "weight_digest_matches_payload", "optimizer_restored", "optimizer_saved",
        "checkpoint_promotable", "teacher_enabled", "training_called",
        "supervised_training_called", "teacher_called", "ppo_called", "PPO_ALLOWED",
        "checkpoint_identity", "canonical_initial", "weights_match",
        "source_payload_unchanged",
    }
    result = {key: metadata[key] for key in scalar_keys if key in metadata}
    for key in ("migration", "payload_bindings"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            result[key] = _bounded_json(value)
    current = metadata.get("current_execution_metadata")
    if isinstance(current, Mapping):
        result["current_execution_metadata"] = dict(current)
    declared = metadata.get("metadata")
    if isinstance(declared, Mapping):
        small = {key: declared[key] for key in scalar_keys if key in declared}
        declared_current = declared.get("current_execution_metadata")
        if isinstance(declared_current, Mapping):
            small["current_execution_metadata"] = dict(declared_current)
        for key in ("migration", "payload_bindings"):
            value = declared.get(key)
            if isinstance(value, Mapping):
                small[key] = _bounded_json(value)
        result["metadata"] = small
    return result


def _new_arm_record(
    label: str,
    checkpoint: Path,
    config: Any,
    scenario: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "label": label,
        "scenario": scenario,
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "status": "pending",
        "development_only": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "requests": int(config.total_requested_uavs),
        "completed": 0,
        "normal": 0,
        "emergency": 0,
        "failed_energy": 0,
        "unresolved": 0,
        "summary": None,
        "tape": None,
        "action_tape": None,
        "lifecycle": None,
        "gates": None,
        "diagnostics": None,
        "runtime_shortlist_checks": None,
        "completion_fraction": None,
        "loader": None,
        "weight_digest_before": None,
        "weight_digest_after": None,
        "weights_unchanged": None,
        "teacher_enabled": bool(getattr(config, "teacher_enabled", False)),
        "label_rows": 0,
        "PPO_ALLOWED": False,
    }


def _evaluate_arm(
    *,
    label: str,
    checkpoint: Path,
    candidate_sha256: str | None,
    config: Any,
    protocol: Mapping[str, Any],
    scenario: str,
    seed: int,
    output: Path,
    checkpoint_loader: Any = None,
) -> tuple[dict[str, Any], bool, str | None]:
    record = _new_arm_record(label, checkpoint, config, scenario, seed)
    policy = None
    stop_after_arm = False
    stop_reason: str | None = None

    try:
        if checkpoint_loader is not None:
            policy, metadata = checkpoint_loader(checkpoint, config)
        elif label == "source":
            policy, metadata = load_trained_checkpoint(checkpoint, config)
        else:
            if candidate_sha256 is None:
                raise ValueError("candidate SHA256 is required")
            policy, metadata = load_repaired_checkpoint(
                checkpoint, config, expected_sha256=candidate_sha256
            )
        record["loader"] = _loader_record(metadata)
        record["weight_digest_before"] = _policy_weight_digest(policy)
        max_steps = int(round(float(config.max_simulation_seconds) / float(config.dt)))
        steps, summary = rollout(
            policy,
            config,
            seed=int(seed),
            max_steps=max_steps,
            output=output,
            deterministic=True,
            collect_labels=False,
        )
        if not isinstance(summary, Mapping):
            raise TypeError("student rollout summary must be a mapping")
        record["summary"] = summary
        record["tape"] = summary.get("action_tape")
        record["action_tape"] = summary.get("action_tape")
        record["completed"] = int(summary.get("completed", 0))
        record.update(_outcome_fields(summary, record["requests"]))
        record["label_rows"] = int(summary.get("label_rows", 0))
        record["PPO_ALLOWED"] = bool(summary.get("PPO_ALLOWED", False))
        record["teacher_enabled"] = bool(getattr(config, "teacher_enabled", False))

        if _summary_has_safety_error(summary):
            record["status"] = "failed"
            record["safety_invariant_error"] = True
            stop_after_arm = True
            stop_reason = "SafetyInvariantError returned by student rollout"
        else:
            gates, diagnostics, shortlist, completion = _rollout_gates(
                policy,
                config,
                steps,
                summary,
                protocol,
                deterministic=True,
                scenario=scenario,
            )
            record["gates"] = gates
            record["diagnostics"] = diagnostics
            record["runtime_shortlist_checks"] = shortlist
            record["completion_fraction"] = completion
            tape_path = summary.get("action_tape")
            if not isinstance(tape_path, str) or not tape_path:
                raise ValueError("student rollout returned no action tape")
            record["lifecycle"] = lifecycle_metrics(
                config,
                tape_path,
                low_soc_threshold=float(protocol["attributes"]["low_soc_analysis_threshold"]),
            )
            record["status"] = "completed"
    except SafetyInvariantError as exc:
        record["status"] = "failed"
        record["safety_invariant_error"] = True
        record["error"] = _error_payload(exc)
        stop_after_arm = True
        stop_reason = "SafetyInvariantError raised during arm evaluation"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = _error_payload(exc)
        stop_after_arm = True
        stop_reason = "unexpected arm evaluation error"
    finally:
        if policy is not None and record["weight_digest_before"] is not None:
            try:
                record["weight_digest_after"] = _policy_weight_digest(policy)
                record["weights_unchanged"] = bool(
                    record["weight_digest_before"] == record["weight_digest_after"]
                )
            except Exception as exc:
                record["digest_error"] = _error_payload(exc)
                record["weights_unchanged"] = False
                if "error" not in record:
                    record["status"] = "failed"
                    record["error"] = record["digest_error"]
                stop_after_arm = True
                stop_reason = "unexpected policy digest error"
        record["invariants"] = _arm_invariants(record)
        record["PASS"] = _arm_pass(record)
    return record, stop_after_arm, stop_reason


def _not_run_arm(
    *, label: str, checkpoint: Path, config: Any, scenario: str, seed: int, reason: str
) -> dict[str, Any]:
    record = _new_arm_record(label, checkpoint, config, scenario, seed)
    record.update({"status": "not_run", "reason": reason, "PASS": False})
    return record


def _aggregate(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        return _lifecycle_aggregate([record]), None
    except Exception as exc:
        return None, _error_payload(exc)


def run(
    *,
    source_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    candidate_sha256: str,
    output: str | Path,
    scenario: str,
    seed: int,
) -> dict[str, Any]:
    """Run one fresh, protocol-bound source/candidate development pair."""

    _validate_binding(scenario, int(seed))
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"pair output must be fresh: {output_path}")
    source_path = Path(source_checkpoint)
    candidate_path = Path(candidate_checkpoint)
    protocol = _load_yaml_protocol()
    config = _build_config(scenario, protocol)
    max_steps = int(round(float(config.max_simulation_seconds) / float(config.dt)))
    output_path.mkdir(parents=True, exist_ok=False)

    arms: list[dict[str, Any]] = []
    source_record, stop_after_source, stop_reason = _evaluate_arm(
        label="source",
        checkpoint=source_path,
        candidate_sha256=None,
        config=config,
        protocol=protocol,
        scenario=scenario,
        seed=int(seed),
        output=output_path / "source",
    )
    arms.append(source_record)
    if stop_after_source:
        candidate_record = _not_run_arm(
            label="candidate",
            checkpoint=candidate_path,
            config=config,
            scenario=scenario,
            seed=int(seed),
            reason=stop_reason or "source arm evaluation stopped subsequent arm",
        )
        candidate_record["candidate_sha256"] = str(candidate_sha256)
        arms.append(candidate_record)
    else:
        candidate_record, _, _ = _evaluate_arm(
            label="candidate",
            checkpoint=candidate_path,
            candidate_sha256=str(candidate_sha256),
            config=config,
            protocol=protocol,
            scenario=scenario,
            seed=int(seed),
            output=output_path / "candidate",
        )
        candidate_record["candidate_sha256"] = str(candidate_sha256)
        arms.append(candidate_record)

    source_metrics, source_aggregate_error = _aggregate(source_record)
    candidate_metrics, candidate_aggregate_error = _aggregate(arms[1])
    selection_gates = _dev_selection_gates(
        source_metrics if source_metrics is not None else {},
        candidate_metrics if candidate_metrics is not None else {},
    )
    arm_errors = [
        record.get("error") or {
            "arm": record.get("label"),
            "reason": record.get("reason", "arm failed"),
        }
        for record in arms
        if record.get("status") in {"failed", "not_run"}
    ]
    safety_failures = any(bool(record.get("safety_invariant_error")) for record in arms)
    result = {
        "stage": STAGE,
        "status": "FAILED" if arm_errors or safety_failures else "COMPLETED",
        "development_only": True,
        "P3_QUALIFIED": False,
        "PPO_ALLOWED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "scenario": scenario,
        "seed": int(seed),
        "candidate_sha256": str(candidate_sha256),
        "protocol_seed": int(PROTOCOL_SEEDS[scenario]),
        "horizon_steps": max_steps,
        "stopped_after_source": bool(stop_after_source),
        "config": _config_snapshot(config),
        "protocol_yaml": str(PROTOCOL),
        "records": arms,
        "arms": arms,
        "selection": {
            "development_only": True,
            "source_metrics": source_metrics,
            "candidate_metrics": candidate_metrics,
            "source_aggregate_error": source_aggregate_error,
            "candidate_aggregate_error": candidate_aggregate_error,
            "gates": selection_gates,
            "PASS": bool(all(selection_gates.values())),
        },
        "failure": {
            "safety_invariant_error": safety_failures,
            "arm_errors": arm_errors,
        },
    }
    _write_json(output_path / "pair_summary.json", result)
    return result


evaluate_pair = run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    result = run(
        source_checkpoint=args.source_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        candidate_sha256=args.candidate_sha256,
        output=args.output,
        scenario=args.scenario,
        seed=args.seed,
    )
    print(json.dumps(_jsonable(result), sort_keys=True))
    return 0 if result.get("status") == "COMPLETED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
