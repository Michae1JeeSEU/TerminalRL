"""No-update P3 V2 pilot over sparse regression and dense24 competition."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.evaluate_v4_2_p2_gate_runtime_shortlist_v1 import _tape_shortlist_checks
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_PILOT_V2"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_pilot_v2.yaml")
PROTOCOL_SHA = "002fb59ba722b713af5e4b6147a8adc807cd688c5e7967d798f1213f1f0c43c0"


def protocol() -> dict[str, Any]:
    if effectiveness._sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("P3 V2 pilot protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if value.get("protocol") != STAGE:
        raise ValueError("P3 V2 pilot protocol id changed")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def scenario_config(name: str, value: dict[str, Any]):
    attributes = value["attributes"]
    scenario = value["scenarios"][name]
    count = int(scenario["total_requested_uavs"])
    base = replace(
        p2_config(count),
        priority_mode=str(attributes["priority_mode"]),
        initial_priority=None,
        initial_soc_mode=str(attributes["initial_soc_mode"]),
        initial_soc=None,
        initial_soc_min=float(attributes["initial_soc_min"]),
        initial_soc_max=float(attributes["initial_soc_max"]),
        mission_probabilities=tuple(float(item) for item in attributes["mission_probabilities"]),
        runtime_proposal_shortlist_k=3,
    )
    if name in {"flow6", "flow12"}:
        first = float(scenario["first_arrival_seconds"])
        interval = float(scenario["arrival_interval_seconds"])
        schedule = tuple(first + interval * index for index in range(count))
        return replace(
            base,
            reset_mode="flow_from_empty",
            warm_start_uavs=0,
            warm_start_steady_motion=False,
            fixed_arrival_schedule=schedule,
            max_simulation_seconds=schedule[-1] + 900.0,
        )
    if name != "dense24":
        raise ValueError(f"unknown P3 V2 pilot scenario: {name}")
    warm = int(scenario["warm_start_uavs"])
    continuing = int(scenario["continuing_arrivals"])
    interval = float(scenario["arrival_interval_seconds"])
    schedule = (0.0,) * warm + tuple(interval * index for index in range(1, continuing + 1))
    if len(schedule) != count:
        raise ValueError("dense24 schedule/count mismatch")
    return replace(
        base,
        reset_mode="warm_start_rings",
        warm_start_uavs=warm,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=schedule,
        max_simulation_seconds=schedule[-1] + 900.0,
        lz_assignment_mode=str(scenario["lz_assignment_mode"]),
        release_arbitration_mode=str(scenario["release_arbitration_mode"]),
        final_queue_capacity=int(scenario["final_queue_capacity"]),
    )


def summarize_lifecycle_rows(rows: list[dict[str, Any]], *, low_soc_threshold: float) -> dict[str, Any]:
    by_priority: dict[str, Any] = {}
    for priority in (1, 3, 5):
        group = [row for row in rows if row["base_priority"] == priority]
        waits = [row["wait_seconds"] for row in group]
        rtt = [row["request_to_touchdown_seconds"] for row in group]
        by_priority[str(priority)] = {
            "count": len(group),
            "mean_wait_seconds": None if not waits else float(np.mean(waits)),
            "p95_wait_seconds": None if not waits else float(np.percentile(waits, 95)),
            "mean_request_to_touchdown_seconds": None if not rtt else float(np.mean(rtt)),
        }
    low_soc = [row for row in rows if row["initial_soc"] <= low_soc_threshold]
    priority_weights = [float(row["base_priority"]) for row in rows]
    return {
        "completed": len(rows),
        "priorities_observed": sorted({row["base_priority"] for row in rows}),
        "initial_soc_min": None if not rows else min(row["initial_soc"] for row in rows),
        "initial_soc_max": None if not rows else max(row["initial_soc"] for row in rows),
        "priority_weighted_wait_seconds": (
            None if not rows else float(np.average(
                [row["wait_seconds"] for row in rows], weights=priority_weights,
            ))
        ),
        "by_priority": by_priority,
        "low_soc": {
            "threshold": float(low_soc_threshold),
            "count": len(low_soc),
            "mean_wait_seconds": None if not low_soc else float(np.mean([row["wait_seconds"] for row in low_soc])),
            "mean_landing_soc": None if not low_soc else float(np.mean([row["landing_soc"] for row in low_soc])),
            "minimum_landing_soc": None if not low_soc else min(row["landing_soc"] for row in low_soc),
        },
        "rows": rows,
    }


def lifecycle_metrics(config, action_tape: str | Path, *, low_soc_threshold: float) -> dict[str, Any]:
    env = TerminalParallelEnv(config)
    replay = SafetyTraceRecorder.replay_action_tape(env, action_tape)
    rows = []
    for uav in env.completed_uavs:
        initial_soc = float((uav.energy_remaining + uav.energy_consumed) / uav.energy_capacity)
        rows.append({
            "uid": int(uav.uid),
            "base_priority": int(uav.base_priority),
            "initial_soc": initial_soc,
            "landing_soc": float(uav.soc),
            "wait_seconds": float(uav.external_wait_time + uav.internal_wait_time),
            "request_to_touchdown_seconds": float(uav.touchdown_time - uav.request_time),
        })
    return {
        "replay_exact": bool(replay["exact"]),
        **summarize_lifecycle_rows(rows, low_soc_threshold=low_soc_threshold),
    }


def run(*, output: Path, checkpoint: Path, history_root: Path) -> dict[str, Any]:
    value = protocol()
    bindings, pilot = value["bindings"], value["pilot"]
    if effectiveness._sha256(checkpoint) != bindings["candidate_checkpoint_sha256"]:
        raise ValueError("P3 V2 pilot candidate checkpoint changed")
    output.mkdir(parents=True, exist_ok=False)
    seeds = {int(seed) for seed in pilot["seeds"]}
    history = historical_registry(history_root, output)
    overlap = sorted(seeds & set(history["seeds"]))
    if overlap:
        raise ValueError(f"P3 V2 pilot seeds already consumed: {overlap}")
    marker = checkpoint.parent / str(pilot["reservation_marker"])
    reserve_once(marker, {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "seeds": sorted(seeds),
        "output": str(output),
        "reserved_before_first_reset": True,
    })
    policy, _ = effectiveness.load_candidate_checkpoint(checkpoint)
    records: list[dict[str, Any]] = []
    attributes = value["attributes"]
    for seed_value, scenario in zip(pilot["seeds"], pilot["scenarios"]):
        seed = int(seed_value)
        config = scenario_config(str(scenario), value)
        for controller, active_policy in (("candidate_k3", policy), ("rule", None)):
            folder = output / f"{scenario}_{controller}_{seed}"
            result = (
                effectiveness._student_result(active_policy, config, seed, folder, smoke=False)
                if active_policy is not None
                else effectiveness._rule_result(config, seed, folder, smoke=False)
            )
            shortlist = None
            if active_policy is not None:
                shortlist = _tape_shortlist_checks(result["summary"]["action_tape"], expected_k=3)
                result["gates"]["runtime_shortlist_contract"] = bool(shortlist["all"])
                result["gates"]["all"] = all(
                    bool(item) for key, item in result["gates"].items() if key != "all"
                )
            metrics = lifecycle_metrics(
                config,
                result["summary"]["action_tape"],
                low_soc_threshold=float(attributes["low_soc_analysis_threshold"]),
            )
            records.append({
                "seed": seed,
                "scenario": str(scenario),
                "controller": controller,
                "result": result,
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
            if not result["gates"]["all"]:
                break
        if not records[-1]["result"]["gates"]["all"]:
            break
    candidate_rows = [row for row in records if row["controller"] == "candidate_k3"]
    rule_rows = [row for row in records if row["controller"] == "rule"]
    candidate_lifecycle = [item for row in candidate_rows for item in row["lifecycle"]["rows"]]
    priorities = sorted({item["base_priority"] for item in candidate_lifecycle})
    soc = [item["initial_soc"] for item in candidate_lifecycle]
    hard = value["hard_gates"]
    gates = {
        "all_records": len(records) == int(pilot["expected_records"]),
        "all_episode_gates": bool(records) and all(row["result"]["gates"]["all"] for row in records),
        "candidate_all_requested_complete": bool(candidate_rows) and all(
            row["result"]["all_requests_complete"] for row in candidate_rows
        ),
        "rule_all_requested_complete": bool(rule_rows) and all(
            row["result"]["all_requests_complete"] for row in rule_rows
        ),
        "candidate_exact_lifecycle_replay": bool(candidate_rows) and all(
            row["lifecycle"]["replay_exact"] for row in candidate_rows
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
            "candidate_priority_weighted_wait_seconds": candidate_metric,
            "rule_priority_weighted_wait_seconds": rule_metric,
            "candidate_minus_rule_seconds": float(candidate_metric - rule_metric),
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
        "priority_coverage": priorities,
        "initial_soc_range": None if not soc else [min(soc), max(soc)],
        "training_called": False,
        "PPO_ALLOWED": False,
        "P3_V2_TRAINING_PROTOCOL_PREREGISTRATION_ALLOWED": bool(passed),
        "protocol_sha256": PROTOCOL_SHA,
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
        "P3_V2_TRAINING_PROTOCOL_PREREGISTRATION_ALLOWED": result[
            "P3_V2_TRAINING_PROTOCOL_PREREGISTRATION_ALLOWED"
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
