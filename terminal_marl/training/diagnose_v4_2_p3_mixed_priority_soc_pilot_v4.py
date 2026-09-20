"""Fresh P3 mixed priority/SOC requalification pilot after scheduler repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from terminal_marl.training import diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as v2
from terminal_marl.training import diagnose_v4_2_p3_mixed_priority_soc_pilot_v3 as v3
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.evaluate_v4_2_p2_gate_runtime_shortlist_v1 import _tape_shortlist_checks
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_PILOT_V4"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_pilot_v4.yaml")
PROTOCOL_SHA = "1597b9363b4bc3cda9da76237bd7109b36cb64bdec446beb6f48d8bdbfe80a82"
SCHEDULER_SOURCE = Path("terminal_marl/core/merge_scheduler.py")
SCHEDULER_SOURCE_SHA = "58bf3f3a29b86f00855610000a819424e5c90b0c1f1433a780bcb42f86309ce1"

CANDIDATE_INTEGRITY_KEYS = (
    "safety",
    "exact_tape_replay",
    "exact_requested_likelihood",
    "no_reverse_transfer",
    "shield_actor_fraction",
    "no_teacher",
    "no_labels",
    "no_updates",
    "runtime_shortlist_contract",
    "geometry",
    "replay",
    "student_likelihood",
    "student_only",
)
RULE_INTEGRITY_KEYS = (
    "safety",
    "exact_tape_replay",
    "no_reverse_transfer",
    "rule_shield_zero",
    "no_teacher",
    "no_updates",
    "geometry",
    "replay",
    "student_only",
)


def protocol() -> dict[str, Any]:
    if effectiveness._sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("P3 V4 pilot protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != STAGE:
        raise ValueError("P3 V4 pilot protocol id changed")
    bindings = value.get("bindings", {})
    if bindings.get("scheduler_source") != SCHEDULER_SOURCE.as_posix():
        raise ValueError("P3 V4 scheduler source binding changed")
    if bindings.get("scheduler_source_sha256") != SCHEDULER_SOURCE_SHA:
        raise ValueError("P3 V4 scheduler source SHA binding changed")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _gate_values(result: Mapping[str, Any]) -> Mapping[str, Any]:
    gates = result.get("gates", {})
    return gates if isinstance(gates, Mapping) else {}


def integrity_gates(
    result: Mapping[str, Any],
    *,
    controller: str,
    runtime_shortlist: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    """Return only non-completion integrity checks for one evaluator result."""
    gates = dict(_gate_values(result))
    if controller == "candidate_k3":
        values = {
            key: bool(gates.get(key, False)) for key in CANDIDATE_INTEGRITY_KEYS
        }
        if runtime_shortlist is None:
            values["runtime_shortlist_contract"] = False
        else:
            values["runtime_shortlist_contract"] = bool(runtime_shortlist.get("all"))
        return values
    if controller == "rule":
        return {key: bool(gates.get(key, False)) for key in RULE_INTEGRITY_KEYS}
    raise ValueError(f"unknown P3 V4 controller: {controller}")


def integrity_passes(values: Mapping[str, bool]) -> bool:
    return bool(values) and all(bool(value) for value in values.values())


def _completion_fraction(row: Mapping[str, Any], config: Any) -> float:
    result = row.get("result", {})
    requests = int(result.get("requests", config.total_requested_uavs))
    completed = int(result.get("completed", row.get("lifecycle", {}).get("completed", 0)))
    return completed / max(1, requests)


def _optional_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def run(*, output: Path, checkpoint: Path, history_root: Path) -> dict[str, Any]:
    value = protocol()
    bindings, pilot = value["bindings"], value["pilot"]
    scheduler_path = Path(str(bindings["scheduler_source"]))
    if effectiveness._sha256(scheduler_path) != SCHEDULER_SOURCE_SHA:
        raise ValueError("P3 V4 scheduler source changed")
    if effectiveness._sha256(checkpoint) != bindings["candidate_checkpoint_sha256"]:
        raise ValueError("P3 V4 pilot candidate checkpoint changed")

    output.mkdir(parents=True, exist_ok=False)
    seeds = {int(seed) for seed in pilot["seeds"]}
    history = historical_registry(history_root, output)
    overlap = sorted(seeds & set(history["seeds"]))
    if overlap:
        raise ValueError(f"P3 V4 pilot seeds already consumed: {overlap}")
    marker = checkpoint.parent / str(pilot["reservation_marker"])
    reserve_once(marker, {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "scheduler_source": bindings["scheduler_source"],
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
        "seeds": sorted(seeds),
        "output": str(output),
        "reserved_before_first_reset": True,
    })

    policy, _ = effectiveness.load_candidate_checkpoint(checkpoint)
    records: list[dict[str, Any]] = []
    attributes = value["attributes"]
    for seed_value, scenario in zip(pilot["seeds"], pilot["scenarios"]):
        seed = int(seed_value)
        config = v2.scenario_config(str(scenario), value)
        for controller, active_policy in (("candidate_k3", policy), ("rule", None)):
            folder = output / f"{scenario}_{controller}_{seed}"
            result = (
                effectiveness._student_result(active_policy, config, seed, folder, smoke=False)
                if active_policy is not None
                else effectiveness._rule_result(config, seed, folder, smoke=False)
            )
            shortlist = None
            if active_policy is not None:
                tape_path = v3.action_tape_path(result)
                shortlist = _tape_shortlist_checks(tape_path, expected_k=3)
                result["gates"]["runtime_shortlist_contract"] = bool(shortlist["all"])
                result["gates"]["all"] = all(
                    bool(item) for key, item in result["gates"].items() if key != "all"
                )
            else:
                tape_path = v3.action_tape_path(result)
            integrity = integrity_gates(
                result,
                controller=controller,
                runtime_shortlist=shortlist,
            )
            metrics = v2.lifecycle_metrics(
                config,
                tape_path,
                low_soc_threshold=float(attributes["low_soc_analysis_threshold"]),
            )
            records.append({
                "seed": seed,
                "scenario": str(scenario),
                "controller": controller,
                "result": result,
                "integrity_gates": integrity,
                "integrity_pass": integrity_passes(integrity),
                "runtime_shortlist_checks": shortlist,
                "lifecycle": metrics,
            })
            _write(output / "progress.json", {
                "completed_records": len(records),
                "expected_records": int(pilot["expected_records"]),
                "last_seed": seed,
                "last_scenario": scenario,
                "last_controller": controller,
            })

    candidate_rows = [row for row in records if row["controller"] == "candidate_k3"]
    rule_rows = [row for row in records if row["controller"] == "rule"]
    candidate_lifecycle = [item for row in candidate_rows for item in row["lifecycle"]["rows"]]
    priorities = sorted({item["base_priority"] for item in candidate_lifecycle})
    soc = [item["initial_soc"] for item in candidate_lifecycle]
    hard = value["hard_gates"]
    gates = {
        "all_records": len(records) == int(pilot["expected_records"]),
        "candidate_integrity": bool(candidate_rows)
        and all(bool(row["integrity_pass"]) for row in candidate_rows),
        "rule_integrity": bool(rule_rows)
        and all(bool(row["integrity_pass"]) for row in rule_rows),
        "rule_all_requested_complete": bool(rule_rows) and all(
            row["result"]["all_requests_complete"] for row in rule_rows
        ),
        "candidate_exact_lifecycle_replay": bool(candidate_rows) and all(
            row["lifecycle"]["replay_exact"] for row in candidate_rows
        ),
        "rule_exact_lifecycle_replay": bool(rule_rows) and all(
            row["lifecycle"]["replay_exact"] for row in rule_rows
        ),
        "priority_coverage": priorities == list(hard["priority_coverage_required"]),
        "soc_range_observed": bool(soc)
        and min(soc) <= float(hard["observed_soc_min_at_most"])
        and max(soc) >= float(hard["observed_soc_max_at_least"]),
    }
    comparisons: dict[str, Any] = {}
    for scenario in pilot["scenarios"]:
        candidate = next((row for row in candidate_rows if row["scenario"] == scenario), None)
        rule = next((row for row in rule_rows if row["scenario"] == scenario), None)
        if candidate is None or rule is None:
            continue
        candidate_metric = candidate["lifecycle"]["priority_weighted_wait_seconds"]
        rule_metric = rule["lifecycle"]["priority_weighted_wait_seconds"]
        comparisons[str(scenario)] = {
            "candidate_completion_fraction": _completion_fraction(candidate, v2.scenario_config(str(scenario), value)),
            "rule_completion_fraction": _completion_fraction(rule, v2.scenario_config(str(scenario), value)),
            "candidate_priority_weighted_wait_seconds": candidate_metric,
            "rule_priority_weighted_wait_seconds": rule_metric,
            "candidate_minus_rule_seconds": _optional_difference(candidate_metric, rule_metric),
            "candidate_by_priority": candidate["lifecycle"]["by_priority"],
            "rule_by_priority": rule["lifecycle"]["by_priority"],
            "candidate_low_soc": candidate["lifecycle"]["low_soc"],
            "rule_low_soc": rule["lifecycle"]["low_soc"],
        }

    passed = all(gates.values())
    manifest = {
        "stage": STAGE,
        "PASS": passed,
        "gates": gates,
        "records": records,
        "comparisons": comparisons,
        "diagnostics": {
            "candidate_all_requested_complete": bool(candidate_rows) and all(
                row["result"]["all_requests_complete"] for row in candidate_rows
            ),
            "candidate_completion_fraction_by_scenario": {
                name: row["candidate_completion_fraction"]
                for name, row in comparisons.items()
            },
            "rule_completion_fraction_by_scenario": {
                name: row["rule_completion_fraction"]
                for name, row in comparisons.items()
            },
        },
        "priority_coverage": priorities,
        "initial_soc_range": None if not soc else [min(soc), max(soc)],
        "training_called": False,
        "PPO_ALLOWED": False,
        "P3_MIXED_TRAINING_PROTOCOL_PREREGISTRATION_ALLOWED": bool(passed),
        "protocol_sha256": PROTOCOL_SHA,
        "scheduler_source": bindings["scheduler_source"],
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
        "checkpoint_sha256": bindings["candidate_checkpoint_sha256"],
    }
    _write(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(
        output=args.output,
        checkpoint=args.candidate_checkpoint,
        history_root=args.history_root,
    )
    print(json.dumps({
        "stage": result["stage"],
        "PASS": result["PASS"],
        "P3_MIXED_TRAINING_PROTOCOL_PREREGISTRATION_ALLOWED": result[
            "P3_MIXED_TRAINING_PROTOCOL_PREREGISTRATION_ALLOWED"
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
