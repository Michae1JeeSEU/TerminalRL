"""L1-PPO-D3: frozen three-update A/B for the bounded low-level actor.

This is a diagnostic, not a formal L1 run.  It changes only the action
distribution: the historical Normal-plus-hard-clamp control versus a tanh
change-of-variables policy.  Environment, seeds, checkpoint, PPO settings,
teacher labels, and safety stack are shared by both arms.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from ..algo import RolloutBuffer, SharedPPO
from ..core.checkpoints import load_checkpoint
from ..core.observations import LOW_OBS_V2
from ..core.runtime_traffic_safety import SafetyInvariantError
from .l1_ppo_d2_diagnostic import HEADS, _gradient_summary, _health
from .train_low_v2_ppo import (
    L1SafetyGateError, _episode_design, _grouped_metrics, _model_state,
    _read_frozen_capacity, _read_seed_partition, _rollout_task, _summary_metrics,
    authorize_low_v2_training, l1_full_teacher_config,
)


VARIANTS = {
    "A_gaussian_hard_clamp": "gaussian_hard_clamp",
    "B_tanh_squashed_gaussian": "tanh_squashed_gaussian",
}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _requested_action_provenance(policy: SharedPPO, buffer: RolloutBuffer) -> dict[str, Any]:
    """Verify stored actions/log-probs before an update mutates the policy."""
    rows = [row for agent_rows in buffer.by_agent.values() for row in agent_rows]
    if not rows:
        return {"passed": False, "rows": 0, "reason": "empty rollout"}
    observations = torch.as_tensor(np.stack([row.observation for row in rows]), dtype=torch.float32, device=policy.device)
    actions = torch.as_tensor(np.stack([row.action for row in rows]), dtype=torch.float32, device=policy.device)
    old = np.asarray([row.log_prob for row in rows], dtype=np.float64)
    with torch.no_grad():
        distribution = policy.model.actor_distribution(observations, bounded=policy.uses_bounded_distribution)
        recomputed = distribution.log_prob(actions).sum(-1).cpu().numpy().astype(np.float64)
    requested_bounded = bool(np.all(np.abs(actions.cpu().numpy()) <= 1.0 + 1e-7))
    # Applied actions are deliberately excluded from this calculation.
    return {
        "passed": bool(requested_bounded and np.allclose(old, recomputed, atol=3e-6, rtol=3e-6)),
        "rows": len(rows), "requested_action_in_normalized_range": requested_bounded,
        "max_abs_old_log_prob_error": float(np.max(np.abs(old - recomputed))),
        "ppo_behavior_action": "a_requested", "ppo_ratio_action": "a_requested",
        "a_applied_enters_ratio": False,
    }


def _variant_summary(records: list[dict[str, Any]], provenance: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [record["diagnostic"] for record in records]
    summaries = [SimpleNamespace(**summary) for record in records for summary in record["summaries"]]
    overall = _summary_metrics(summaries)
    warm = _grouped_metrics(summaries)["warm_start_finite_cohort"]
    saturation = {
        head: {
            "exact_bound_rate_by_update": [row["heads"][head]["exact_bound_pile_rate"] for row in diagnostics],
            "near_boundary_rate_by_update": [row["heads"][head]["saturation_rate"] for row in diagnostics],
            "actor_mean_by_update": [row["heads"][head]["post_squash_actor_mean"] for row in diagnostics],
            "actor_std_by_update": [row["heads"][head]["std"] for row in diagnostics],
        }
        for head in HEADS
    }
    gradient_conflict, worst = _gradient_summary(diagnostics)
    health = _health(records)
    safety = {key: int(sum(row[key] for row in summaries)) for key in ("collisions", "nmac", "hard_separation_violations")}
    return {
        "updates": records, "provenance": provenance,
        "provenance_passed": all(row["passed"] for row in provenance),
        "saturation": saturation, "teacher_burden_overall": overall["teacher_interventions_per_100_release_intents"],
        "teacher_burden_warm": warm["teacher_interventions_per_100_release_intents"],
        "correction_mean": overall["correction_magnitude_mean"], "correction_p95": overall["correction_magnitude_p95"],
        "shield_burden": overall["shield_interventions_per_100_release_intents"],
        "gradient_conflict": gradient_conflict, "worst_gradient_conflict_head": worst,
        "ppo_health": health, "completion_ratio": overall["completion_ratio"],
        "throughput_per_second": overall["throughput_per_second"], "safety": safety,
        "safety_passed": not any(safety.values()),
    }


def _mean(series: list[float]) -> float:
    return float(np.mean(series)) if series else 0.0


def _not_worse(candidate: float | None, control: float | None, *, tolerance: float = 1e-9) -> bool:
    return candidate is not None and (control is None or candidate <= control + tolerance)


def _comparison(control: dict[str, Any], bounded: dict[str, Any]) -> dict[str, Any]:
    exact_repaired = all(
        _mean(bounded["saturation"][head]["exact_bound_rate_by_update"]) < .001 and
        _mean(bounded["saturation"][head]["exact_bound_rate_by_update"]) < _mean(control["saturation"][head]["exact_bound_rate_by_update"])
        for head in ("release_request", "merge_time_residual")
    )
    health = bounded["ppo_health"]["healthy"]
    safety = bounded["safety_passed"]
    burden = all(_not_worse(bounded[key], control[key]) for key in ("teacher_burden_overall", "correction_p95", "shield_burden"))
    remaining_near_boundary = any(
        _mean(bounded["saturation"][head]["near_boundary_rate_by_update"]) >= .10
        for head in ("release_request", "merge_time_residual")
    )
    return {
        "exact_bound_artificial_pile_repaired": exact_repaired,
        "safety_passed": safety, "ppo_health_passed": health,
        "teacher_p95_shield_not_worse": burden,
        "B_passed": bool(exact_repaired and safety and health and burden and bounded["provenance_passed"]),
        "remaining_near_boundary_saturation": remaining_near_boundary,
        "allow_log_std_entropy_ablation": bool(exact_repaired and safety and health and remaining_near_boundary),
        "next_step": (
            "log_std / entropy targeted ablation is authorized; bounded sampling fixed the exact pile but near-boundary saturation remains"
            if exact_repaired and safety and health and remaining_near_boundary else
            "do not enter log_std / entropy ablation until D3 acceptance criteria are met"
        ),
    }


def run_l1_ppo_d3_bounded_distribution(*, c0_report: str | Path, teacher_dataset_path: str | Path,
                                        initial_checkpoint: str | Path, output_dir: str | Path,
                                        rollout_workers: int = 4) -> dict[str, Any]:
    """Run A/B with exactly three PPO updates per arm; never launch formal L1."""
    destination = Path(output_dir)
    audit = authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    mu_hat, medium_interval = _read_frozen_capacity(c0_report)
    medium_rate = sum(medium_interval) / 2.0
    config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                    warm_start_uavs=8, cohort_uavs=8, horizon_seconds=1200.0)
    checkpoint = Path(initial_checkpoint)
    frozen = {"updates": 3, "episodes_per_update": 4, "max_steps": 1200, "teacher_bc_weight": .05,
              "mu_hat": mu_hat, "medium_rate": medium_rate, "safety_sampling_dt": config.safety_sampling_dt,
              "physics_dt": config.physics_dt, "teacher": "FULL", "high_level": "OFF", "reward": "unchanged",
              "safety_stack": "unchanged", "action_range": "normalized [-1,1]"}
    manifest = {"stage": "L1_PPO_D3_bounded_actor_distribution", "evidence_type": "newly_run",
                "initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "variants": VARIANTS, "frozen": frozen, "teacher_dataset_audit": asdict(audit),
                "semantic_contract": {"ppo_buffer": "a_requested", "old_log_prob": "log_pi(a_requested|s)",
                                      "ppo_ratio": "a_requested only", "a_applied_enters_ratio": False,
                                      "teacher_target": "unchanged normalized +/-1 endpoints allowed"}}
    _write(destination / "l1_ppo_d3.manifest.json", manifest)
    variants: dict[str, Any] = {}
    for label, distribution_name in VARIANTS.items():
        random.seed(1701); np.random.seed(1701); torch.manual_seed(1701)
        policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05, policy_distribution=distribution_name)
        load_checkpoint(checkpoint, policy, config)
        records: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        train_seeds = itertools.cycle(partition.train)
        design_index = 0
        try:
            for update in range(3):
                state, tasks, designs = _model_state(policy), [], []
                for episode in range(4):
                    episode_config, design = _episode_design(design_index, medium_rate=medium_rate, horizon_seconds=1200.0)
                    design_index += 1
                    tasks.append((state, episode_config, next(train_seeds), update, episode, 1200, False, None, distribution_name))
                    designs.append(design)
                if rollout_workers == 1:
                    collected = [_rollout_task(task) for task in tasks]
                else:
                    with ProcessPoolExecutor(max_workers=min(rollout_workers, len(tasks))) as executor:
                        collected = list(executor.map(_rollout_task, tasks))
                buffer, summaries = RolloutBuffer(), []
                for (transitions, episode_summary), design in zip(collected, designs):
                    for transition in transitions:
                        buffer.add(transition)
                    summaries.append({**asdict(episode_summary), **design})
                if any(row["collisions"] or row["nmac"] or row["hard_separation_violations"] for row in summaries):
                    raise L1SafetyGateError(f"D3 safety gate failed in {label} before PPO update {update}")
                provenance_row = _requested_action_provenance(policy, buffer)
                if not provenance_row["passed"]:
                    raise RuntimeError(f"D3 requested-action provenance failed in {label} update {update}: {provenance_row}")
                provenance.append(provenance_row)
                diagnostic = policy.diagnostic_snapshot(buffer, teacher_bc_weight=.05)
                ppo_update = policy.update(buffer, epochs=2, minibatch_size=256)
                records.append({"update": update, "summaries": summaries, "diagnostic": diagnostic, "ppo_update": ppo_update})
                _write(destination / f"{label}.updates.json", {"manifest": manifest, "records": records})
            variants[label] = _variant_summary(records, provenance)
        except (SafetyInvariantError, L1SafetyGateError, RuntimeError) as error:
            variants[label] = {"status": "stopped", "reason": str(error), "updates": records, "provenance": provenance,
                               "safety_passed": False, "provenance_passed": False}
            break
    control = variants.get("A_gaussian_hard_clamp", {})
    bounded = variants.get("B_tanh_squashed_gaussian", {})
    comparison = _comparison(control, bounded) if "saturation" in control and "saturation" in bounded else {
        "B_passed": False, "allow_log_std_entropy_ablation": False, "next_step": "D3 did not complete both arms"}
    result = {"status": "completed" if len(variants) == 2 and "saturation" in bounded else "stopped",
              "manifest": manifest, "variants": variants, "comparison": comparison}
    _write(destination / "l1_ppo_d3_summary.json", result)
    return result


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run only the 3-update L1 PPO-D3 bounded-distribution A/B diagnostic.")
    parser.add_argument("--c0-report", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_l1_ppo_d3_bounded_distribution(c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
                                                         initial_checkpoint=args.initial_checkpoint, output_dir=args.output,
                                                         rollout_workers=args.workers), indent=2))


if __name__ == "__main__":
    _main()
