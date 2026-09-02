"""Deterministic L1-S safety replay and qualification gate.

This module is deliberately evaluation-only: it never calls PPO.update and
never resumes a checkpoint.  It records the first hard-safety witness with
the policy's requested actions, then replays exactly that action tape at each
candidate safety-sampling interval.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import argparse
from collections.abc import Mapping
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..algo import SharedPPO
from ..core.checkpoints import load_checkpoint
from ..envs.parallel_env import TerminalParallelEnv
from .train_low_v2_ppo import (
    L1_VALIDATION_SCENARIO_INDICES, _episode_design, _read_frozen_capacity,
    _read_seed_partition, _rollout_l1, l1_full_teacher_config,
)


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _conflict_classification(states: list[dict], pair: tuple[int, int]) -> str:
    rows = {int(row["uid"]): row for row in states}
    left, right = (rows.get(int(uid), {}) for uid in pair)
    def kind(row: dict) -> str:
        if row.get("phase") == "final":
            return "final"
        if row.get("committed"):
            return "committed"
        if row.get("ring_role") is not None:
            return "holder"
        return "other"
    labels = {kind(left), kind(right)}
    if labels == {"holder"}: return "holder-holder"
    if labels == {"holder", "committed"}: return "holder-committed"
    if labels == {"committed"}: return "committed-committed"
    if labels == {"committed", "final"}: return "transfer-final"
    if labels == {"final"}: return "final-final"
    return "other"


def _requested_actions(policy: SharedPPO, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not observations:
        return {}
    ids = list(observations)
    matrix = np.stack([observations[uid] for uid in ids]).astype(np.float32, copy=False)
    actions, _, _ = policy.act(matrix, matrix, deterministic=True)
    return {uid: np.asarray(action, dtype=np.float32) for uid, action in zip(ids, actions)}


def _pair_cpa(states: list[dict], pair: tuple[int, int]) -> dict:
    rows = {int(row["uid"]): row for row in states}
    left, right = (rows.get(int(uid)) for uid in pair)
    if left is None or right is None:
        return {"d_cpa": None, "t_cpa": None}
    relative_position = np.asarray(left["position"], dtype=float) - np.asarray(right["position"], dtype=float)
    relative_velocity = np.asarray(left["velocity"], dtype=float) - np.asarray(right["velocity"], dtype=float)
    speed_sq = float(np.dot(relative_velocity, relative_velocity))
    t_cpa = 0.0 if speed_sq <= 1e-12 else float(np.clip(-np.dot(relative_position, relative_velocity) / speed_sq, 0.0, 30.0))
    return {"d_cpa": float(np.linalg.norm(relative_position + relative_velocity * t_cpa)), "t_cpa": t_cpa}


def _run(
    policy: SharedPPO, config, seed: int, max_steps: int, *, action_tape: list[dict[str, list[float]]] | None = None,
) -> tuple[dict, list[dict[str, list[float]]]]:
    """Run one deterministic episode and retain a first-violation ±30 s trace."""
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(seed)
    tape: list[dict[str, list[float]]] = []
    ticks: list[dict] = []
    first: dict | None = None
    post_until: float | None = None
    for index in range(min(max_steps, int(config.max_simulation_seconds / config.dt))):
        actions = ({key: np.asarray(value, dtype=np.float32) for key, value in action_tape[index].items()}
                   if action_tape is not None and index < len(action_tape) else _requested_actions(policy, observations))
        tape.append({key: value.tolist() for key, value in actions.items()})
        pre_states = [_json(env._uav_trace(uav)) for uav in env.active.values()]
        risk = env.runtime_traffic.assess(env)
        observations, _, _, _, info = env.step(actions)
        pre_by_uid = {str(row["uid"]): row for row in pre_states}
        execution_error_series = [
            {"uid": int(uid), "error_norm": float(value),
             "trajectory_kind": "committed" if pre_by_uid.get(uid, {}).get("committed") else
                                "holder" if pre_by_uid.get(uid, {}).get("ring_role") is not None else "other"}
            for uid, value in info["prediction_execution_position_error"].items()
        ]
        tick = {
            "step": index, "time": float(info["time"]), "pre_active": pre_states,
            "active": [_json(env._uav_trace(uav)) for uav in env.active.values()],
            "a_requested": tape[-1], "a_teacher_shield_applied": _json(info["action_traces"]),
            "teacher_shield_decisions": _json(info["release_intent_telemetry"]),
            "prediction_execution_position_error": _json(info["prediction_execution_position_error"]),
            "prediction_execution_error_series": execution_error_series,
            "predicted_min_separation": risk.predicted_min_swept_separation,
            "predicted_d_cpa": risk.d_cpa, "predicted_t_cpa": risk.t_cpa,
            "reservations": _json(env.scheduler.reservations), "release_state": _json(env.release_intents),
            "conflict_events": _json(info["conflict_events"]),
        }
        ticks.append(tick)
        events = [row for row in info["conflict_events"] if row["kind"] in {"first_hard_separation_trace", "nmac_entry", "collision"}]
        if first is None and events:
            event = events[0]
            pair = tuple(event.get("pair", event.get("uids", ())))
            first = {"event": event, "step": index, "time": float(info["time"]), "pair": pair,
                     "classification": _conflict_classification(pre_states, pair), "cpa": _pair_cpa(pre_states, pair)}
            post_until = float(info["time"]) + 30.0
        if post_until is not None and info["time"] >= post_until:
            break
        if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
            break
    if first is None:
        witness_ticks: list[dict] = []
    else:
        witness_ticks = [row for row in ticks if first["time"] - 30.0 <= row["time"] <= first["time"] + 30.0]
    errors = [item["error_norm"] for row in witness_ticks for item in row["prediction_execution_error_series"]]
    by_kind = {}
    for kind in ("holder", "committed", "other"):
        values = [item["error_norm"] for row in witness_ticks for item in row["prediction_execution_error_series"]
                  if item["trajectory_kind"] == kind]
        by_kind[kind] = {"samples": len(values), "max": None if not values else float(max(values)),
                         "mean": None if not values else float(np.mean(values)),
                         "series": [float(value) for value in values]}
    result = {
        "seed": seed, "safety_sampling_dt": config.safety_sampling_dt, "steps": len(ticks),
        "telemetry": _json(env.telemetry), "first_violation": first,
        "prediction_execution_error": {"samples": len(errors), "max": None if not errors else float(max(errors)),
                                         "mean": None if not errors else float(np.mean(errors)), "series": [float(value) for value in errors]},
        "prediction_execution_error_by_trajectory": by_kind,
        "witness_ticks": witness_ticks,
    }
    return result, tape


def qualify_warm_starts(policy: SharedPPO, *, medium_rate: float, seeds: tuple[int, ...], horizon_seconds: float = 60.0) -> list[dict]:
    """Qualify unseen warm starts under nominal and adversarial executable actions."""
    records: list[dict] = []
    cohorts = (8, 10, 12, 14, 16)
    scripts = ("nominal_noop", "all_speed_plus", "all_speed_minus", "alternating", "neighbour_opposite",
               "multiple_aggressive_release", "repeated_release")
    for index, cohort in enumerate(cohorts):
        seed = seeds[index % len(seeds)]
        config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                        warm_start_uavs=cohort, cohort_uavs=cohort, horizon_seconds=horizon_seconds)
        env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
        positions = [uav.position for uav in env.active.values()]
        distances = [float(np.linalg.norm(left - right)) for offset, left in enumerate(positions) for right in positions[offset + 1:]]
        initial_margin = float("inf") if not distances else min(distances)
        for script in scripts:
            env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
            teacher = shield = 0
            for step in range(int(horizon_seconds / config.dt)):
                actions = {}
                for position, uid in enumerate(sorted(observations, key=int)):
                    sign = 1.0 if (position + step) % 2 == 0 else -1.0
                    if script == "nominal_noop": action = (0.0, 0.0, 0.0)
                    elif script == "all_speed_plus": action = (0.0, 1.0, 0.0)
                    elif script == "all_speed_minus": action = (0.0, -1.0, 0.0)
                    elif script in {"alternating", "neighbour_opposite"}: action = (0.0, sign, 0.0)
                    elif script == "multiple_aggressive_release": action = (1.0, sign, 1.0)
                    else: action = (1.0, sign, -1.0 if step % 2 else 1.0)
                    actions[uid] = np.asarray(action, dtype=np.float32)
                observations, _, _, _, info = env.step(actions)
                teacher += sum(int(bool(row["teacher_used"])) for row in info["action_traces"].values())
                shield += sum(int(bool(row["shield_used"])) for row in info["action_traces"].values())
            safety = {key: int(env.telemetry.get(key, 0)) for key in ("collisions", "nmac", "hard_separation_violations")}
            nominal = script == "nominal_noop"
            records.append({"seed": seed, "warm_start_uavs": cohort, "script": script,
                            "initial_min_separation": initial_margin, "required_min_separation": 1.2 * config.safe_distance,
                            "teacher_interventions": teacher, "shield_interventions": shield, "safety": safety,
                            "passed": not any(safety.values()) and (not nominal or
                                      (initial_margin >= 1.2 * config.safe_distance and teacher == shield == 0))})
    return records


def run_l1_safety_diagnostic(*, checkpoint: str | Path, c0_report: str | Path, teacher_dataset: str | Path,
                             output_dir: str | Path, seed: int = 2003, scenario_index: int = 2, max_steps: int = 1200) -> dict:
    """Create the L1-S evidence package; this function is PPO-free."""
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    _, medium = _read_frozen_capacity(c0_report); medium_rate = sum(medium) / 2.0
    config, design = _episode_design(scenario_index, medium_rate=medium_rate, horizon_seconds=float(max_steps))
    policy = SharedPPO(87); load_checkpoint(checkpoint, policy, config); policy.model.eval()
    baseline, tape = _run(policy, config, seed, max_steps)
    dt_results = {"1.0": baseline}
    for dt in (0.5, 0.2):
        dt_results[str(dt)], _ = _run(policy, replace(config, safety_sampling_dt=dt), seed, max_steps, action_tape=tape)
    partition = _read_seed_partition(teacher_dataset)
    warm = qualify_warm_starts(policy, medium_rate=medium_rate, seeds=tuple(partition.qualification))
    payload = {"stage": "L1-S", "checkpoint": str(checkpoint), "checkpoint_status": "diagnostic_failed_training_safety_v2",
               "config": {**asdict(config), "rings": {role.value: asdict(spec) for role, spec in config.rings.items()}},
               "scenario": design, "fixed_priority": {"mode": config.priority_mode, "initial_priority": config.initial_priority},
               "deterministic_replay": dt_results, "warm_start_qualification": warm,
               "warm_start_passed": bool(warm) and all(row["passed"] for row in warm)}
    (destination / "l1_safety_diagnostic.json").write_text(json.dumps(_json(payload), indent=2, allow_nan=False), encoding="utf-8")
    return payload


def _training_failure_rows(path: str | Path) -> list[dict]:
    """Extract only episode-level historical hard-safety failures.

    The first L1 implementation persisted aggregate episode outcomes but not
    a policy snapshot/action tape for each PPO update.  This loader therefore
    makes that provenance gap explicit instead of mislabelling a replay of the
    final checkpoint as an exact reconstruction of an early PPO policy.
    """
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "episode" not in row:
            continue
        if any(int(row.get(key, 0)) > 0 for key in ("collisions", "nmac", "hard_separation_violations")):
            rows.append(row)
    return rows


def _replay_training_failure(args: tuple[str, float, int, dict]) -> dict:
    checkpoint, medium_rate, max_steps, historical = args
    config = l1_full_teacher_config(
        medium_rate=medium_rate, reset_mode=historical["reset_mode"], demand_model=historical["demand_model"],
        warm_start_uavs=int(historical["warm_start_uavs"]), cohort_uavs=int(historical["cohort_uavs"]),
        horizon_seconds=float(max_steps),
    )
    policy = SharedPPO(87)
    load_checkpoint(checkpoint, policy, config)
    policy.model.eval()
    replay, _ = _run(policy, config, int(historical["seed"]), max_steps)
    replay["historical_episode"] = {
        key: historical.get(key) for key in (
            "update", "episode", "seed", "reset_mode", "demand_model", "warm_start_uavs", "cohort_uavs",
            "arrivals", "requested", "release_intents", "completed", "collisions", "nmac", "hard_separation_violations",
        )
    }
    replay["replay_provenance"] = {
        "kind": "reconstructed_final_checkpoint_policy",
        "same_initial_state": True,
        "same_requested_action_tape": False,
        "limitation": "historical PPO update snapshots and action tapes were not persisted; this is a deterministic scenario replay, not an exact historical-policy replay",
    }
    return replay


def replay_all_training_failures(*, checkpoint: str | Path, c0_report: str | Path, training_metrics: str | Path,
                                 output_dir: str | Path, max_steps: int = 1200, workers: int = 4) -> dict:
    """Run all persisted unsafe L1 scenarios as explicit reconstructed witnesses."""
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    _, medium = _read_frozen_capacity(c0_report)
    historical = _training_failure_rows(training_metrics)
    tasks = [(str(checkpoint), 0.5 * sum(medium), max_steps, row) for row in historical]
    partial_path = destination / "l1_s2_training_witness_replays.partial.json"
    replays: list[dict] = []
    if workers <= 1:
        for task in tasks:
            replays.append(_replay_training_failure(task))
            partial_path.write_text(json.dumps(_json({"completed_replays": replays}), indent=2, allow_nan=False), encoding="utf-8")
    else:
        # Torch/OpenMP pools may already exist in the diagnostic parent.  A
        # Linux ``fork`` can inherit their locked futex state and leave every
        # replay worker permanently asleep; use a clean interpreter process.
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks) or 1),
                                 mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(_replay_training_failure, task) for task in tasks]
            for future in as_completed(futures):
                replays.append(future.result())
                partial_path.write_text(json.dumps(_json({"completed_replays": replays}), indent=2, allow_nan=False), encoding="utf-8")
    safety_passed = all(item["telemetry"].get("collisions", 0) == item["telemetry"].get("nmac", 0) ==
                        item["telemetry"].get("hard_separation_violations", 0) == 0 for item in replays)
    payload = {
        "stage": "L1-S2_training_violation_replay",
        "evidence_type": "newly_run",
        "checkpoint": str(checkpoint),
        "historical_training_metrics": str(training_metrics),
        "historical_failure_episode_count": len(historical),
        "exact_historical_policy_replay_available": False,
        "reconstructed_scenario_replay_safety_passed": safety_passed,
        "replays": replays,
    }
    (destination / "l1_s2_training_witness_replays.json").write_text(
        json.dumps(_json(payload), indent=2, allow_nan=False), encoding="utf-8",
    )
    partial_path.unlink(missing_ok=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PPO-free L1-S deterministic safety diagnostic.")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--c0-report", required=True)
    parser.add_argument("--teacher-dataset", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2003); parser.add_argument("--scenario-index", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--training-metrics", help="Optional prior L1 metrics JSONL; replays every persisted hard-safety failure.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replay-training-failures-only", action="store_true",
                        help="Resume only the historical training-failure replay closure.")
    args = parser.parse_args()
    if args.replay_training_failures_only:
        if not args.training_metrics:
            parser.error("--replay-training-failures-only requires --training-metrics")
        training = replay_all_training_failures(
            checkpoint=args.checkpoint, c0_report=args.c0_report, training_metrics=args.training_metrics,
            output_dir=args.output, max_steps=args.max_steps, workers=args.workers,
        )
        print(json.dumps({"output": args.output,
                          "training_witness_replay_passed": training["reconstructed_scenario_replay_safety_passed"]}, allow_nan=False))
        return
    result = run_l1_safety_diagnostic(checkpoint=args.checkpoint, c0_report=args.c0_report, teacher_dataset=args.teacher_dataset,
                                      output_dir=args.output, seed=args.seed, scenario_index=args.scenario_index, max_steps=args.max_steps)
    training = None if not args.training_metrics else replay_all_training_failures(
        checkpoint=args.checkpoint, c0_report=args.c0_report, training_metrics=args.training_metrics,
        output_dir=args.output, max_steps=args.max_steps, workers=args.workers,
    )
    print(json.dumps({"output": args.output, "first_violation": result["deterministic_replay"]["1.0"]["first_violation"],
                      "warm_start_passed": result["warm_start_passed"],
                      "training_witness_replay_passed": None if training is None else training["reconstructed_scenario_replay_safety_passed"]}, allow_nan=False))


if __name__ == "__main__":
    main()
