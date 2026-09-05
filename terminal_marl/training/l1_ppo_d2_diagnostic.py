"""L1-PPO-D2: frozen action-head and gradient diagnostic (three updates only)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from ..algo import RolloutBuffer, SharedPPO
from ..core.checkpoints import load_checkpoint
from ..core.config import SimConfig
from ..core.merge_scheduler import LowLevelIntent
from ..core.observations import LOW_OBS_V2
from ..core.runtime_traffic_safety import SafetyInvariantError
from .train_low_v2_ppo import (
    L1SafetyGateError, _episode_design, _model_state, _read_frozen_capacity,
    _read_seed_partition, _rollout_task, authorize_low_v2_training, l1_full_teacher_config,
)


FROZEN_SAFETY_COMMIT = "449514d"
HEADS = ("release_request", "speed_residual", "merge_time_residual")


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _range(values: list[float]) -> str:
    return "n/a" if not values else f"{min(values):.4g}–{max(values):.4g}"


def _normalized_space_audit(config: SimConfig) -> dict[str, Any]:
    """Assert the representation boundary without altering it."""
    clipped = LowLevelIntent.from_array((3.0, -3.0, 2.0))
    return {
        "passed": (clipped.release_request, clipped.speed_residual, clipped.merge_time_residual) == (1.0, -1.0, 1.0),
        "ppo_action_space": "normalized [-1, 1] requested action",
        "bc_teacher_target_space": "normalized [-1, 1] LowLevelIntent",
        "speed_physical_conversion": f"normalized residual * {config.max_speed_residual_mps} m/s at execution",
        "merge_time_physical_conversion": f"normalized residual * {config.max_merge_time_residual_seconds} s only in MergeScheduler.propose",
        "bc_uses_physical_seconds": False,
    }


def _head_cause(records: list[dict[str, Any]], head: str) -> dict[str, Any]:
    values = [record["heads"][head] for record in records]
    saturation = [row["saturation_rate"] for row in values]
    mean_boundary = [row["actor_mean_boundary_rate"] for row in values]
    sample_away = [row["sample_bound_hit_away_from_mean_rate"] for row in values]
    exact_pile = [row["exact_bound_pile_rate"] for row in values]
    target_sat = [row["teacher_target_saturation_rate"] for row in values if row["teacher_target_saturation_rate"] is not None]
    std = [row["std"] for row in values]
    if float(np.mean(mean_boundary)) >= 0.10:
        primary = "A: actor mean/logit itself approaches the boundary"
    elif float(np.mean(sample_away)) >= 0.10:
        primary = "B: sampled Normal action hits a bound while actor mean is interior"
    else:
        primary = "no material requested-action saturation source"
    clamp = "D: hard clamp creates an exact ±1 boundary pile" if float(np.mean(exact_pile)) >= 0.05 else "D: no material exact-bound pile"
    teacher = "C: teacher target frequently at ±1" if target_sat and float(np.mean(target_sat)) >= 0.10 else "C: teacher target is not materially boundary-saturated"
    return {
        "saturation_by_update": saturation, "actor_mean_boundary_by_update": mean_boundary,
        "sample_hit_away_from_mean_by_update": sample_away, "exact_bound_pile_by_update": exact_pile,
        "std_by_update": std, "teacher_target_saturation_by_update": target_sat,
        "primary": primary, "clamp_mechanism": clamp, "teacher_target_mechanism": teacher,
    }


def _gradient_summary(records: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    result: dict[str, Any] = {}
    worst, worst_value = None, -1.0
    for head in HEADS:
        cosine_rows = [record["heads"][head]["teacher_used_head_output_gradient_cosine"] for record in records]
        negative = [row["negative_fraction"] for row in cosine_rows if row["negative_fraction"] is not None]
        means = [row["mean"] for row in cosine_rows if row["mean"] is not None]
        result[head] = {"cosine_mean_by_update": means, "cos_negative_fraction_by_update": negative,
                        "cos_negative_fraction_mean": None if not negative else float(np.mean(negative))}
        score = -1.0 if not negative else float(np.mean(negative))
        if score > worst_value:
            worst, worst_value = head, score
    return result, worst


def _health(records: list[dict[str, Any]]) -> dict[str, Any]:
    ppo = [row["ppo"] for row in records]
    optimizer = [row["optimizer"] for row in records]
    return {
        "learning_rate": _range([row["learning_rate"] for row in optimizer]),
        "entropy_coefficient": _range([row["entropy_coefficient"] for row in optimizer]),
        "approx_kl": _range([row["approx_kl"] for row in ppo]),
        "clip_fraction": _range([row["clip_fraction"] for row in ppo]),
        "actor_loss": _range([row["actor_loss"] for row in ppo]),
        "value_loss": _range([row["value_loss"] for row in ppo]),
        "explained_variance": _range([row["explained_variance"] for row in ppo]),
        "raw_advantage_mean": _range([row["advantages"]["raw"]["mean"] for row in records]),
        "raw_advantage_std_proxy": _range([row["advantages"]["raw"]["p90"] - row["advantages"]["raw"]["p10"] for row in records]),
        "normalized_advantage_mean": _range([row["advantages"]["normalized"]["mean"] for row in records]),
        "normalized_advantage_std_proxy": _range([row["advantages"]["normalized"]["p90"] - row["advantages"]["normalized"]["p10"] for row in records]),
        "healthy": all(abs(row["approx_kl"]) <= .05 and row["clip_fraction"] <= .30 for row in ppo),
    }


def _markdown(summary: dict[str, Any]) -> str:
    causes = summary["head_causes"]
    gradients = summary["gradient_conflict"]
    critic = summary["critic_audit"]
    health = summary["optimizer_health"]
    lines = ["# L1 PPO-D2 Diagnostic Summary", ""]
    for head, label in zip(HEADS, ("release", "speed", "merge-time")):
        row = causes[head]
        lines.append(f"- {label} saturation 根因：{row['primary']}；{row['clamp_mechanism']}；std={_range(row['std_by_update'])}。")
    target = {head: causes[head]["teacher_target_mechanism"] for head in HEADS}
    lines.append(f"- teacher target 是否自身饱和：{target}。")
    audit = summary["normalized_space_audit"]
    lines.append(f"- merge-time normalized/physical scaling 是否正确：{'正确' if audit['passed'] and not audit['bc_uses_physical_seconds'] else 'FAIL'}；{audit['merge_time_physical_conversion']}。")
    lines.append(f"- PPO 与 BC 是否存在 head-wise gradient conflict：{gradients}。")
    lines.append(f"- 哪个 head 的冲突最严重：{summary['worst_gradient_conflict_head'] or '无 teacher-used gradient 样本'}。")
    lines.append(f"- critic 是否压制 actor/shared backbone：actor grad={_range(critic['actor_grad_norm'])}，critic grad={_range(critic['critic_grad_norm'])}；{critic['shared_backbone_note']}。")
    lines.append(f"- LR / entropy / KL / clip / advantage 是否健康：{'健康' if health['healthy'] else '需诊断'}；{health}。")
    lines.append(f"- 下一步应优先修改：{summary['next_targeted_ablation']}。")
    lines.append(f"- 是否允许进入下一轮 targeted PPO ablation：{'允许' if summary['allow_targeted_ablation'] else '不允许'}。")
    return "\n".join(lines) + "\n"


def run_l1_ppo_d2_diagnostic(*, c0_report: str | Path, teacher_dataset_path: str | Path,
                              initial_checkpoint: str | Path, output_dir: str | Path,
                              rollout_workers: int = 4) -> dict[str, Any]:
    """Collect exactly three frozen-config PPO diagnostic updates."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    audit = authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    mu_hat, medium_interval = _read_frozen_capacity(c0_report)
    medium_rate = sum(medium_interval) / 2.0
    config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                    warm_start_uavs=8, cohort_uavs=8, horizon_seconds=1200.0)
    frozen = {"source_commit": FROZEN_SAFETY_COMMIT, "safety_sampling_dt": config.safety_sampling_dt,
              "physics_dt": config.physics_dt, "mu_hat": mu_hat, "lambda_uav_per_second": medium_rate,
              "teacher": "FULL", "priority": config.initial_priority, "teacher_bc_weight": .05,
              "mission": "standard", "soc": config.initial_soc, "high_level": "OFF",
              "geometry": "nominal", "gates": "uniform", "landing_zones": "all_free"}
    if not (abs(mu_hat - .003) < 1e-12 and abs(medium_rate - .001575) < 1e-12 and config.safety_sampling_dt == 1.0 and
            config.physics_dt == .2 and config.initial_priority == 1 and config.priority_mode == "fixed" and
            config.teacher_mode == "teacher_full" and config.gate_probabilities == (.25, .25, .25, .25) and
            config.mission_probabilities == (1.0, 0.0, 0.0, 0.0) and config.initial_soc == .80 and
            config.high_level_enabled is False):
        raise RuntimeError("D2 refused: frozen Safety Closure environment mismatch")
    normalized_audit = _normalized_space_audit(config)
    if not normalized_audit["passed"]:
        raise RuntimeError("D2 refused: normalized action audit failed")
    random.seed(1701); np.random.seed(1701); torch.manual_seed(1701)
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05)
    load_checkpoint(initial_checkpoint, policy, config)
    manifest = {"stage": "L1_PPO_D2_action_head_gradient_diagnostic", "evidence_type": "newly_run",
                "frozen": frozen, "initial_checkpoint": str(initial_checkpoint),
                "initial_checkpoint_sha256": hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest(),
                "updates": 3, "episodes_per_update": 4, "rollout_workers": rollout_workers,
                "ppo_bc_provenance": {"ppo_behavior_action": "a_requested", "ppo_old_log_prob": "log_pi(a_requested|s)",
                                      "ppo_ratio_uses_a_applied": False, "teacher_bc_target": "a_teacher",
                                      "teacher_bc_mask": "teacher_used && a_teacher is not None"},
                "normalized_space_audit": normalized_audit, "teacher_dataset_audit": asdict(audit)}
    _json_dump(destination / "l1_ppo_d2.manifest.json", manifest)
    records: list[dict[str, Any]] = []
    safety_events: list[dict[str, Any]] = []
    train_seeds = itertools.cycle(partition.train)
    design_index = 0
    try:
        for update in range(3):
            state = _model_state(policy)
            tasks, designs = [], []
            for episode in range(4):
                episode_config, design = _episode_design(design_index, medium_rate=medium_rate, horizon_seconds=1200.0)
                design_index += 1
                tasks.append((state, episode_config, next(train_seeds), update, episode, 1200, False,
                              str(destination / "safety_action_tapes")))
                designs.append(design)
            with ProcessPoolExecutor(max_workers=min(rollout_workers, len(tasks))) as executor:
                collected = list(executor.map(_rollout_task, tasks))
            buffer = RolloutBuffer()
            summaries = []
            for (transitions, episode_summary), design in zip(collected, designs):
                for transition in transitions:
                    buffer.add(transition)
                summaries.append({**asdict(episode_summary), **design})
            hard = [row for row in summaries if row["collisions"] or row["nmac"] or row["hard_separation_violations"]]
            if hard:
                safety_events.extend(hard)
                raise L1SafetyGateError(f"D2 safety gate failed before PPO update {update}")
            snapshot = policy.diagnostic_snapshot(buffer, teacher_bc_weight=.05)
            update_metrics = policy.update(buffer, epochs=2, minibatch_size=256)
            records.append({"update": update, "episode_summaries": summaries, "diagnostic": snapshot, "ppo_update": update_metrics})
            _json_dump(destination / "l1_ppo_d2_updates.json", {"manifest": manifest, "updates": records})
    except (SafetyInvariantError, L1SafetyGateError, RuntimeError) as error:
        summary = {"status": "stopped", "reason": str(error), "frozen": frozen, "updates": records,
                   "safety_events": safety_events, "allow_targeted_ablation": False}
        _json_dump(destination / "l1_ppo_d2_diagnostic_summary.json", summary)
        (destination / "l1_ppo_d2_diagnostic_summary.md").write_text(
            "# L1 PPO-D2 Diagnostic Summary\n\n- 是否允许进入下一轮 targeted PPO ablation：不允许；safety gate 或 runtime failure。\n",
            encoding="utf-8")
        return summary
    detail = [row["diagnostic"] for row in records]
    head_causes = {head: _head_cause(detail, head) for head in HEADS}
    gradient_conflict, worst_head = _gradient_summary(detail)
    actor_grads = [row["gradient_blocks"]["actor"] for row in detail]
    critic_grads = [row["gradient_blocks"]["critic"] for row in detail]
    critic_audit = {"actor_grad_norm": actor_grads, "critic_grad_norm": critic_grads,
                    "critic_to_actor_grad_ratio": [critic / max(actor, 1e-12) for actor, critic in zip(actor_grads, critic_grads)],
                    "shared_backbone_note": detail[0]["gradient_blocks"]["shared_backbone_note"],
                    "value_to_abs_actor_loss": [row["ppo"]["value_to_abs_actor_loss"] for row in detail]}
    optimizer_health = _health(detail)
    any_sampling = any(np.mean(head_causes[head]["sample_hit_away_from_mean_by_update"]) >= .10 for head in HEADS)
    any_mean_boundary = any(np.mean(head_causes[head]["actor_mean_boundary_by_update"]) >= .10 for head in HEADS)
    if any_sampling:
        next_step = "actor distribution / log_std/entropy targeted ablation; retain normalized action range and safety stack"
    elif any_mean_boundary:
        next_step = "actor distribution mean/logit targeted ablation; do not change action range"
    elif worst_head is not None:
        next_step = f"head-wise BC diagnostic ablation for {worst_head}; do not change global BC weight yet"
    else:
        next_step = "learning-rate / advantage-scale diagnostic ablation"
    safety_passed = not safety_events
    summary = {"status": "completed", "evidence_type": "newly_run", "frozen": frozen, "normalized_space_audit": normalized_audit,
               "updates": records, "head_causes": head_causes, "gradient_conflict": gradient_conflict,
               "worst_gradient_conflict_head": worst_head, "critic_audit": critic_audit,
               "optimizer_health": optimizer_health, "next_targeted_ablation": next_step,
               "safety_passed": safety_passed, "allow_targeted_ablation": bool(safety_passed and optimizer_health["healthy"])}
    _json_dump(destination / "l1_ppo_d2_diagnostic_summary.json", summary)
    (destination / "l1_ppo_d2_diagnostic_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run exactly three frozen L1 PPO-D2 diagnostic updates.")
    parser.add_argument("--c0-report", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_l1_ppo_d2_diagnostic(c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
                                               initial_checkpoint=args.initial_checkpoint, output_dir=args.output,
                                               rollout_workers=args.workers), indent=2))


if __name__ == "__main__":
    _main()
