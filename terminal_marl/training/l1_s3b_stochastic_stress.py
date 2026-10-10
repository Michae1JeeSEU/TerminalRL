"""PPO-free long stochastic L1 safety stress gate.

Runs the frozen Full-Teacher checkpoint with sampled (not updated) policy
actions across the standard L1 scenario mix.  It is a safety qualification,
not a training or curriculum-advance command.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from ..algo import SharedPPO
from ..core.checkpoints import load_checkpoint
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..envs.parallel_env import TerminalParallelEnv
from .l1_safety_diagnostic import _json
from .train_low_v2_ppo import _episode_design, _read_frozen_capacity


def _run_one(task: tuple[str, float, int, int, int]) -> dict:
    checkpoint, medium_rate, seed, scenario_index, max_steps = task
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    config, scenario = _episode_design(scenario_index, medium_rate=medium_rate, horizon_seconds=float(max_steps))
    policy = SharedPPO(87); load_checkpoint(checkpoint, policy, config); policy.model.eval()
    env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
    blocked = None
    release_intents = 0
    for step in range(min(max_steps, int(config.max_simulation_seconds / config.dt))):
        if observations:
            ids = list(observations)
            matrix = np.stack([observations[uid] for uid in ids]).astype(np.float32, copy=False)
            actions, _, _ = policy.act(matrix, matrix, deterministic=False)
            action_map = {uid: np.asarray(action, dtype=np.float32) for uid, action in zip(ids, actions)}
        else:
            action_map = {}
        release_intents += sum(int(action[0] > config.release_threshold) for action in action_map.values())
        try:
            observations, _, _, _, _ = env.step(action_map)
        except SafetyInvariantError as error:
            blocked = {"step": step, "time": env.now, "message": str(error), "diagnostic": _json(error.diagnostic)}
            break
        if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
            break
    telemetry = {key: int(env.telemetry.get(key, 0)) for key in (
        "collisions", "nmac", "hard_separation_violations", "joint_safety_fail_closed", "joint_unrecoverable_components",
    )}
    return {"seed": seed, "scenario_index": scenario_index, "scenario": scenario, "steps": step + 1,
            "safety": telemetry, "safety_invariant_blocked": blocked is not None, "block": blocked,
            "arrivals": int(env.arrivals.generated), "requested": len(env.ledger.requested),
            "admitted": int(env.telemetry.get("arrivals_admitted", 0)), "release_intents": release_intents,
            "completed": len(env.completed_uavs),
            "shield_interventions": int(env.telemetry.get("shield_interventions", 0))}


def run(*, checkpoint: str | Path, c0_report: str | Path, output: str | Path, episodes: int = 20,
        max_steps: int = 1200, workers: int = 10, seed_start: int = 3001) -> dict:
    _, medium = _read_frozen_capacity(c0_report)
    tasks = [(str(checkpoint), 0.5 * sum(medium), seed_start + index, index % 10, max_steps)
             for index in range(episodes)]
    records = []
    destination = Path(output); destination.mkdir(parents=True, exist_ok=True)
    partial = destination / "l1_s3b_stochastic_stress.partial.json"
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            partial.write_text(json.dumps(_json({"completed": records}), indent=2, allow_nan=False), encoding="utf-8")
    blocked = sum(int(row["safety_invariant_blocked"]) for row in records)
    totals = {key: sum(row["safety"][key] for row in records) for key in next(iter(records))["safety"]}
    traffic_totals = {key: sum(row[key] for row in records) for key in
                      ("arrivals", "requested", "admitted", "release_intents", "completed", "shield_interventions")}
    report = {"stage": "L1-S3b stochastic stress", "evidence_type": "newly_run", "ppo_updates": 0,
              "episodes": sorted(records, key=lambda row: row["scenario_index"]), "totals": totals,
              "traffic_totals": traffic_totals,
              "safety_invariant_errors": blocked,
              "passed": not blocked and not any(totals.values()),
              "config": {"teacher": "full", "priority_mode": "fixed", "initial_priority": 1,
                         "safety_sampling_dt": 1.0, "physics_dt": 0.2, "max_steps": max_steps}}
    (destination / "l1_s3b_stochastic_stress.json").write_text(json.dumps(_json(report), indent=2, allow_nan=False), encoding="utf-8")
    partial.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--c0-report", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1200); parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=3001)
    args = parser.parse_args()
    report = run(checkpoint=args.checkpoint, c0_report=args.c0_report, output=args.output, episodes=args.episodes,
                 max_steps=args.max_steps, workers=args.workers, seed_start=args.seed_start)
    print(json.dumps({"output": args.output, "passed": report["passed"], "episodes": len(report["episodes"])}))


if __name__ == "__main__":
    main()
