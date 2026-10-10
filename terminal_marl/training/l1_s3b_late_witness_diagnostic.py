"""Late-tick S3b safety-feasibility diagnosis for reconstructed witnesses.

This is intentionally PPO-free.  It stops at a fail-closed tick and evaluates
the actual recursive successor state, including a dense N<=2 residual grid.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..algo import SharedPPO
from ..core.checkpoints import load_checkpoint
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..core.safety_filter import SafetyFilter
from ..envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from .l1_safety_diagnostic import _json, _requested_actions, _training_failure_rows
from .train_low_v2_ppo import _read_frozen_capacity, l1_full_teacher_config


def _residuals(env, actions: dict[str, np.ndarray]) -> dict[int, float]:
    scale = float(env.config.max_speed_residual_mps)
    return {
        int(uid): float(np.clip(action[1], -1.0, 1.0)) * scale
        for uid, action in actions.items()
        if int(uid) in env.active and env.active[int(uid)].ring_role is not None
        and env.active[int(uid)].committed_plan is None
    }


def _dense_component_grid(shield, env, residuals: dict[int, float], pairs, focus_edges):
    """Offline 81^N diagnostic; never used by the online safety controller."""
    # The full predicted graph is often a large connected component through
    # benign holder--transfer edges.  Qualification is for the *rejected*
    # component, while each candidate still proves every pair incident to it.
    components = shield._components(focus_edges or pairs)
    report = []
    levels = np.linspace(-1.0, 1.0, 81)
    for component in components:
        variables = tuple(uid for uid in component if uid in residuals)
        if not variables or len(variables) > 2:
            continue
        current_solution = None
        existential_solution = None
        best_margin = -float("inf")
        best_candidate = None
        checked = 0
        relevant = tuple(pair for pair in pairs if set(pair).intersection(component))
        for values in itertools.product(levels, repeat=len(variables)):
            trial = dict(residuals)
            for uid, value in zip(variables, values):
                trial[uid] = float(value * env.config.max_speed_residual_mps)
            verdict = shield._joint_assessment(env, trial, pairs=relevant)
            checked += 1
            if verdict.minimum_separation > best_margin:
                best_margin, best_candidate = verdict.minimum_separation, dict(trial)
            if not verdict.approved:
                continue
            if current_solution is None:
                current_solution = {str(uid): trial[uid] for uid in variables}
            recursive = shield._recursive_safe(env, trial, relevant)
            if recursive.approved:
                existential_solution = {str(uid): trial[uid] for uid in variables}
                break
        report.append({
            "uids": list(component), "variables": list(variables), "checked": checked,
            "current_horizon_solution": current_solution,
            "existential_backup_solution": existential_solution,
            "best_minimum_separation": best_margin,
            "best_candidate": None if best_candidate is None else {str(uid): best_candidate[uid] for uid in variables},
        })
    return report


def run(*, checkpoint: str | Path, c0_report: str | Path, training_metrics: str | Path,
        seed: int, max_steps: int = 1200, audit_commit_uav: int | None = None) -> dict:
    rows = [row for row in _training_failure_rows(training_metrics) if int(row["seed"]) == int(seed)]
    if not rows:
        raise ValueError(f"no historical failed witness for seed {seed}")
    historical = rows[0]
    _, medium = _read_frozen_capacity(c0_report)
    config = l1_full_teacher_config(
        medium_rate=0.5 * sum(medium), reset_mode=historical["reset_mode"], demand_model=historical["demand_model"],
        warm_start_uavs=int(historical["warm_start_uavs"]), cohort_uavs=int(historical["cohort_uavs"]),
        horizon_seconds=float(max_steps),
    )
    policy = SharedPPO(87); load_checkpoint(checkpoint, policy, config); policy.model.eval()
    env = TerminalParallelEnv(config); observations, _ = env.reset(int(seed))
    commit_audits: list[dict] = []
    if audit_commit_uav is not None:
        original_commit = env._commit

        def audited_commit(uav, plan):
            committed = {uid: other.committed_plan for uid, other in env.active.items()
                         if other.committed_plan is not None}
            snapshot = WorldSnapshot(env.now, env.active, env.airspace)
            coarse = env.safety.filter_plan(plan, committed, snapshot)
            dense_config = replace(env.config, safety_sampling_dt=env.config.physics_dt)
            dense = SafetyFilter(dense_config).filter_plan(plan, committed, snapshot)
            commit_audits.append(_json({
                "time": env.now, "uav_id": uav.uid, "plan_id": plan.plan_id,
                "target_phase": plan.metadata.get("target_phase"), "start_time": plan.start_time,
                "end_time": plan.end_time, "committed_uav_ids": sorted(committed),
                "coarse": coarse, "physics_resolution_diagnostic": dense,
            }))
            return original_commit(uav, plan)

        env._commit = audited_commit
    for step in range(min(max_steps, int(config.max_simulation_seconds / config.dt))):
        actions = _requested_actions(policy, observations)
        try:
            observations, _, _, _, _ = env.step(actions)
        except SafetyInvariantError as error:
            shield = env.runtime_traffic
            requested = _residuals(env, actions)
            pairs = shield._joint_pairs(env)
            next_env = shield._next_executor_state(env, requested)
            next_domain = shield._successor_action_domain(next_env, requested)
            next_pairs = shield._joint_pairs(next_env)
            backup, next_assessment = shield._backup_safe(next_env, next_domain, next_pairs)
            after = None if backup is None else shield._next_executor_state(next_env, backup)
            after_domain = None if after is None else shield._successor_action_domain(after, backup)
            after_pairs = () if after is None else shield._joint_pairs(after)
            final_backup, final_assessment = ((None, None) if after is None else
                                              shield._backup_safe(after, after_domain, after_pairs))
            return _json({
                "seed": seed, "step": step, "time": env.now, "error": str(error),
                "requested_residual": requested, "pairs": pairs,
                "next": {"time": next_env.now, "domain": next_domain, "pairs": next_pairs,
                         "backup": backup, "assessment": next_assessment},
                "after_backup": None if after is None else {
                    "time": after.now, "domain": after_domain, "pairs": after_pairs,
                    "backup": final_backup, "assessment": final_assessment,
                    "dense_components": _dense_component_grid(
                        shield, after, after_domain, after_pairs,
                        () if final_assessment is None else final_assessment.unsafe_pairs,
                    ),
                },
                "invariant_diagnostic": error.diagnostic,
            })
        if audit_commit_uav is not None and any(row["uav_id"] == audit_commit_uav for row in commit_audits):
            return {"seed": seed, "step": step, "time": env.now, "commit_audits": commit_audits}
    return {"seed": seed, "completed_without_safety_invariant": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--c0-report", required=True)
    parser.add_argument("--training-metrics", required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--audit-commit-uav", type=int)
    args = parser.parse_args()
    report = run(checkpoint=args.checkpoint, c0_report=args.c0_report, training_metrics=args.training_metrics,
                 seed=args.seed, max_steps=args.max_steps, audit_commit_uav=args.audit_commit_uav)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": args.output, "step": report.get("step")}))


if __name__ == "__main__":
    main()
