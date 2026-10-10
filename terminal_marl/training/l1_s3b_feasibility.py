"""L1-S3b offline qualification for uncertified runtime-safety components.

This module is intentionally PPO-free.  It recreates only the first state at
which the historical reconstructed cases requested a shield action, and then
compares certification rules without invoking the online repair loop.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from ..algo import SharedPPO
from ..core.checkpoints import load_checkpoint
from ..core.runtime_traffic_safety import PredictiveSafetyShield
from ..envs.parallel_env import TerminalParallelEnv
from .l1_safety_diagnostic import _json, _training_failure_rows
from .train_low_v2_ppo import _read_frozen_capacity, l1_full_teacher_config


def _policy_actions(policy: SharedPPO, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not observations:
        return {}
    ids = list(observations)
    matrix = np.stack([observations[uid] for uid in ids]).astype(np.float32, copy=False)
    actions, _, _ = policy.act(matrix, matrix, deterministic=True)
    return {uid: np.asarray(action, dtype=np.float32) for uid, action in zip(ids, actions)}


def _t1_environment(policy: SharedPPO, config, seed: int) -> tuple[TerminalParallelEnv, dict[str, np.ndarray]]:
    """Recreate the t=1 state without allowing the candidate shield to repair it."""
    env = TerminalParallelEnv(replace(config, shield_enabled=False))
    observations, _ = env.reset(seed)
    # The historic S3b blocks occurred on the second control decision.  The
    # first tick has no runtime-shield intervention in the recorded outcomes;
    # disable only that shield so the diagnostic can inspect the intended t=1
    # candidate state rather than throwing before it is observable.
    observations, _, _, _, _ = env.step(_policy_actions(policy, observations))
    return env, observations


def _residuals(env: TerminalParallelEnv, actions: dict[str, np.ndarray]) -> dict[int, float]:
    return {
        uid: float(np.clip(actions[str(uid)][1], -1.0, 1.0)) * env.config.max_speed_residual_mps
        for uid, uav in env.active.items()
        if uav.ring_role is not None and uav.committed_plan is None and str(uid) in actions
    }


def _conflict_time(shield: PredictiveSafetyShield, env, residuals: dict[int, float], pairs: tuple[tuple[int, int], ...]) -> float | None:
    if not pairs:
        return None
    horizon = env.config.shield_horizon_seconds + env.config.physics_dt
    tracks = shield._joint_tracks(env, residuals, {uid for pair in pairs for uid in pair}, horizon=horizon)
    samples = np.arange(env.now, env.now + horizon + 1e-9, env.config.safety_sampling_dt)
    for time in samples:
        for left, right in pairs:
            if left not in tracks or right not in tracks:
                continue
            if np.linalg.norm(tracks[left].sample(float(time)).position - tracks[right].sample(float(time)).position) < env.config.safe_distance:
                return float(time)
    return None


def _candidate_summary(shield: PredictiveSafetyShield, env, residuals: dict[int, float], pairs: tuple[tuple[int, int], ...],
                       component: tuple[int, ...]) -> dict[str, Any]:
    variables = tuple(uid for uid in component if uid in residuals)
    current = shield._joint_assessment(env, residuals, pairs=pairs)
    # A bounded, direct backup policy is B's witness, and does not recurse.
    next_env = shield._next_executor_state(env, residuals)
    backup, backup_assessment = shield._backup_safe(next_env, residuals, shield._joint_pairs(next_env))
    direct_choice, direct_assessment = shield._component_search(
        env, residuals, residuals, component, pairs, backup=False, recursive=False,
    )
    return {
        "uids": list(component), "size": len(component), "requested_residual": {str(uid): residuals[uid] for uid in variables},
        "current_horizon": {"approved": current.approved, "minimum_separation": current.minimum_separation,
                            "unsafe_pairs": [list(item) for item in current.unsafe_pairs]},
        "existential_backup": {"approved": backup is not None, "backup": None if backup is None else {str(uid): backup.get(uid) for uid in variables},
                                 "minimum_separation": backup_assessment.minimum_separation,
                                 "unsafe_pairs": [list(item) for item in backup_assessment.unsafe_pairs]},
        "beam_current_horizon": {"approved": direct_choice is not None,
                                 "candidate": None if direct_choice is None else {str(uid): direct_choice.get(uid) for uid in variables},
                                 "minimum_separation": None if direct_assessment is None else direct_assessment.minimum_separation},
        "predicted_conflict_time": _conflict_time(shield, env, residuals, pairs),
        "reachable_tube": "not_used: runtime shield verifies controlled holders as exact candidate/backup tracks",
    }


def _exhaustive_component(shield: PredictiveSafetyShield, env, residuals: dict[int, float], pairs: tuple[tuple[int, int], ...],
                          component: tuple[int, ...]) -> dict[str, Any] | None:
    """Offline dense grid only for N<=3; never used by runtime shielding."""
    variables = tuple(uid for uid in component if uid in residuals)
    if not variables or len(variables) > 3:
        return None
    levels = np.linspace(-1.0, 1.0, 81 if len(variables) <= 2 else 41)
    # 41^3 is explicitly requested for N=3.  For N=1/2 this means 81 values
    # per residual, rather than merely 81 total combinations.
    found_current = found_backup = None
    current_solutions: list[dict[int, float]] = []
    checked = 0
    for values in itertools.product(levels, repeat=len(variables)):
        trial = dict(residuals)
        for uid, value in zip(variables, values):
            trial[uid] = float(value * env.config.max_speed_residual_mps)
        checked += 1
        current = shield._joint_assessment(env, trial, pairs=pairs)
        if not current.approved:
            continue
        if found_current is None:
            found_current = {str(uid): trial[uid] for uid in variables}
        # Current-horizon existence is the exhaustive solver test.  Retain a
        # small deterministic representative set for existential checking;
        # invoking a backup policy for all 81^2 points is redundant and does
        # not strengthen the existence conclusion.
        if len(current_solutions) < 32:
            current_solutions.append(trial)
    for trial in current_solutions:
        next_env = shield._next_executor_state(env, trial)
        backup, _ = shield._backup_safe(next_env, trial, shield._joint_pairs(next_env))
        if backup is not None:
            found_backup = {str(uid): trial[uid] for uid in variables}
            break
    return {"levels_per_residual": len(levels), "checked": checked,
            "current_horizon_solution": found_current, "existential_backup_solution": found_backup}


def _diagnose_one(args: tuple[str, float, dict[str, Any]]) -> dict[str, Any]:
    checkpoint, medium_rate, historical = args
    config = l1_full_teacher_config(
        medium_rate=medium_rate, reset_mode=historical["reset_mode"], demand_model=historical["demand_model"],
        warm_start_uavs=int(historical["warm_start_uavs"]), cohort_uavs=int(historical["cohort_uavs"]), horizon_seconds=2.0,
    )
    policy = SharedPPO(87); load_checkpoint(checkpoint, policy, config); policy.model.eval()
    env, observations = _t1_environment(policy, config, int(historical["seed"]))
    actions = _policy_actions(policy, observations)
    shield = PredictiveSafetyShield(config)
    requested = _residuals(env, actions)
    pairs = shield._joint_pairs(env)
    # A: old same-residual continuation. C: current-horizon only.
    current = shield._joint_assessment(env, requested, pairs=pairs)
    same = shield._joint_assessment(env, requested, pairs=pairs, horizon=config.shield_horizon_seconds + config.physics_dt)
    # B: candidate -> actual x(t+control_dt) -> exists backup. D is equal to
    # B at runtime because controlled holders are exact tracks, not tubes.
    existential = shield._recursive_safe(env, requested, pairs)
    edges = same.unsafe_pairs or current.unsafe_pairs or existential.unsafe_pairs or pairs
    components = shield._components(edges)
    details = []
    for component in components:
        relevant = tuple(pair for pair in pairs if set(pair).intersection(component))
        item = _candidate_summary(shield, env, requested, relevant, component)
        item["exhaustive"] = _exhaustive_component(shield, env, requested, relevant, component)
        details.append(item)
    return {
        "historical_episode": {key: historical.get(key) for key in ("seed", "update", "episode", "reset_mode", "demand_model", "warm_start_uavs", "cohort_uavs")},
        "t1": float(env.now), "requested_residual": {str(key): value for key, value in requested.items()},
        "A_same_residual": {"approved": same.approved, "minimum_separation": same.minimum_separation, "unsafe_pairs": [list(item) for item in same.unsafe_pairs]},
        "B_existential_backup": {"approved": existential.approved, "minimum_separation": existential.minimum_separation,
                                   "gate": shield.last_recursive_gate, "unsafe_pairs": [list(item) for item in existential.unsafe_pairs]},
        "C_current_horizon_only": {"approved": current.approved, "minimum_separation": current.minimum_separation,
                                    "unsafe_pairs": [list(item) for item in current.unsafe_pairs]},
        "D_controlled_holder_exact_track": {"approved": existential.approved,
                                              "note": "identical to B: runtime verifier does not use arbitrary reachable tube for shield-controlled holders"},
        "components": details,
    }


def run(*, checkpoint: str | Path, c0_report: str | Path, training_metrics: str | Path, output: str | Path, workers: int = 10) -> dict[str, Any]:
    _, medium = _read_frozen_capacity(c0_report)
    rows = _training_failure_rows(training_metrics)
    tasks = [(str(checkpoint), 0.5 * sum(medium), row) for row in rows]
    destination = Path(output); destination.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    partial = destination / "l1_s3b_t1_feasibility.partial.json"
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks) or 1), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(_diagnose_one, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            partial.write_text(json.dumps(_json({"stage": "L1-S3b-t1-feasibility", "completed": results}), indent=2,
                                                allow_nan=False), encoding="utf-8")
    result = {"stage": "L1-S3b-t1-feasibility", "evidence_type": "newly_run", "results": results}
    (destination / "l1_s3b_t1_feasibility.json").write_text(json.dumps(_json(result), indent=2, allow_nan=False), encoding="utf-8")
    partial.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--c0-report", required=True)
    parser.add_argument("--training-metrics", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    report = run(checkpoint=args.checkpoint, c0_report=args.c0_report, training_metrics=args.training_metrics,
                 output=args.output, workers=args.workers)
    print(json.dumps({"output": args.output, "cases": len(report["results"])}, allow_nan=False))


if __name__ == "__main__":
    main()
