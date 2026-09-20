"""Paired diagnostic for the dense P2 landing-flow resource semantics.

This is deliberately a diagnostic, not a promotion protocol.  It replays the
same frozen policy and seed with historical resource semantics and with the
opt-in dense-throughput semantics, then reports landing/resource utilization
and the unchanged hard-safety counters.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from ..core.config import RingRole
from ..core.teacher_scheduler import TeacherScheduler
from ..core.uav import FlightPhase
from ..envs.parallel_env import TerminalParallelEnv
from ..visualization.ppo_checkpoint_viewer import _load_profile, _p2_dense_scene_config
from .student_rollout_v4_2 import _student_request_details
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


def config_for(semantics: str):
    config = _p2_dense_scene_config()
    if semantics == "legacy":
        return replace(
            config,
            lz_assignment_mode="lowest_id",
            release_arbitration_mode="legacy_insertion",
            final_queue_capacity=1,
        )
    if semantics == "throughput_v1":
        return config
    raise ValueError(f"unsupported dense semantics: {semantics}")


def episode(
    checkpoint: Path,
    profile: str,
    seed: int,
    semantics: str,
    max_steps: int,
    *,
    counterfactual_opportunities: bool = False,
) -> dict:
    config = config_for(semantics)
    policy = _load_profile(checkpoint, profile)
    policy.model.eval()
    torch.manual_seed(seed)
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(seed)
    if counterfactual_opportunities:
        # The bounded catalogue is attached only for a read-only opportunity
        # probe.  The environment remains Teacher-free, so student actions are
        # still the sole execution input and K=3 runtime semantics are intact.
        teacher_config = replace(
            config,
            experiment_mode="fixed_teacher_heavy",
            scheduler_mode="teacher_label_only",
            teacher_enabled=True,
            teacher_mode="teacher_label_only",
            runtime_proposal_shortlist_k=1,
        )
        env.teacher_scheduler = TeacherScheduler(teacher_config, env.scheduler)
    release_decisions: Counter[str] = Counter()
    occupancy_sum = {role: 0 for role in RingRole}
    occupancy_peak = {role: 0 for role in RingRole}
    active_peak = len(env.active)
    touchdown_times: list[float] = []
    seen_touchdowns = 0
    opportunity_counts: Counter[str] = Counter()

    for _ in range(max_steps):
        details = _student_request_details(
            policy, env, observations, deterministic=True,
            shortlist_k=config.runtime_proposal_shortlist_k,
        )
        uids, actor_mask, gates = details[0], details[2], details[4]
        requested, proposal_shortlists = details[3], details[-1]
        final_count = sum(uav.phase is FlightPhase.FINAL for uav in env.active.values())
        free_lz = any(zone.status.value == "free" for zone in env.lz_manager.zones)
        for index, uid in enumerate(uids):
            uav = env.active[int(uid)]
            if not actor_mask[index] or uav.phase is not FlightPhase.RING_INNER:
                continue
            opportunity_counts["inner_actor_steps"] += 1
            resource_available = final_count < config.final_queue_capacity and free_lz
            if not resource_available:
                continue
            opportunity_counts["inner_resource_opportunity_steps"] += 1
            if int(gates[index]) == 1:
                opportunity_counts["inner_resource_opportunity_requested_steps"] += 1
                continue
            opportunity_counts["inner_resource_opportunity_not_requested_steps"] += 1
            if counterfactual_opportunities:
                teacher, _ = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
                if teacher.release_request > 0.0:
                    opportunity_counts["inner_individually_certified_but_not_requested_steps"] += 1
        observations, _, _, _, info = env.step(requested, proposal_shortlists)
        release_decisions.update(info["release_decisions"].values())
        active_peak = max(active_peak, len(env.active))
        for role in RingRole:
            value = env.airspace.occupancy(role)
            occupancy_sum[role] += value
            occupancy_peak[role] = max(occupancy_peak[role], value)
        touchdowns = [event for event in env.events if event[0] == "touchdown"]
        if len(touchdowns) > seen_touchdowns:
            touchdown_times.extend(float(event[2]) for event in touchdowns[seen_touchdowns:])
            seen_touchdowns = len(touchdowns)
        terminal = (
            env.arrivals.exhausted() and not env.active
            and not env.admission.queue.items and not env.service_uavs
        )
        if terminal:
            break

    steps = max(1, env.step_count)
    telemetry = dict(env.telemetry)
    assignments = max(1, int(telemetry["lz_assignments"]))
    safety = {
        key: int(telemetry.get(key, 0))
        for key in ("collisions", "nmac", "hard_separation_violations", "uncertified_execution")
    }
    return {
        "semantics": semantics,
        "seed": seed,
        "seconds": float(env.now),
        "terminal": bool(
            env.arrivals.exhausted() and not env.active
            and not env.admission.queue.items and not env.service_uavs
        ),
        "completed": len(env.completed_uavs),
        "touchdowns": len(touchdown_times),
        "touchdown_times": touchdown_times,
        "active_peak": active_peak,
        "ring_occupancy_mean": {role.value: occupancy_sum[role] / steps for role in RingRole},
        "ring_occupancy_peak": {role.value: occupancy_peak[role] for role in RingRole},
        "release_decisions": dict(release_decisions),
        "max_concurrent_final": int(telemetry["max_concurrent_final"]),
        "mean_concurrent_final": float(telemetry["final_concurrency_step_sum"] / steps),
        "mean_free_lz": float(telemetry["free_lz_step_sum"] / steps),
        "lz_idle_with_inner_demand_fraction": float(
            telemetry["lz_idle_with_inner_demand_steps"] / steps
        ),
        "lz_assignments": int(telemetry["lz_assignments"]),
        "mean_lz_assignment_distance_m": float(
            telemetry["lz_assignment_distance_sum"] / assignments
        ),
        "mean_lz_assignment_regret_m": float(
            telemetry["lz_assignment_regret_sum"] / assignments
        ),
        "safety": safety,
        "runtime_shortlist_exhausted": int(telemetry["runtime_shortlist_exhausted"]),
        "opportunity_counts": dict(opportunity_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("p2_initial_k3", "p2_trained_k3"), default="p2_initial_k3")
    parser.add_argument("--seed", type=int, default=809006)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--counterfactual-opportunities", action="store_true")
    parser.add_argument(
        "--semantics",
        choices=("both", "legacy", "throughput_v1"),
        default="both",
    )
    args = parser.parse_args()
    semantics_values = (
        ("legacy", "throughput_v1")
        if args.semantics == "both"
        else (args.semantics,)
    )
    records = [
        episode(
            args.checkpoint,
            args.profile,
            args.seed,
            semantics,
            args.max_steps,
            counterfactual_opportunities=args.counterfactual_opportunities,
        )
        for semantics in semantics_values
    ]
    result = {"diagnostic": "P2_DENSE_THROUGHPUT_V1", "profile": args.profile, "records": records}
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
