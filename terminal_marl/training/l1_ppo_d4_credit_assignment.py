"""L1-PPO-D4: teacher-rescue credit-assignment diagnostic only.

The three arms share checkpoint, seeds, rollout design, PPO hyperparameters,
reward, action semantics, teacher authority, and safety stack.  They differ
only in the requested teacher loss and whether teacher-corrected actor heads
are detached from the PPO surrogate gradient.
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
from .l1_ppo_d2_diagnostic import HEADS, _health
from .l1_ppo_d3_bounded_distribution import _requested_action_provenance
from .train_low_v2_ppo import (
    L1SafetyGateError, _episode_design, _grouped_metrics, _model_state,
    _read_frozen_capacity, _read_seed_partition, _rollout_task, _summary_metrics,
    authorize_low_v2_training, l1_full_teacher_config,
)


VARIANTS = {
    "A_current_tanh_current_bc": {"teacher_bc_loss": "action_smooth_l1", "intervention_ppo_mask": False},
    "B_tanh_squashed_bc": {"teacher_bc_loss": "squashed_nll", "intervention_ppo_mask": False},
    "C_tanh_squashed_bc_headwise_mask": {"teacher_bc_loss": "squashed_nll", "intervention_ppo_mask": True},
}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _quantiles(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": None, "p95": None}
    return {"mean": float(array.mean()), "p95": float(np.percentile(array, 95))}


def _head_credit(diagnostics: list[dict[str, Any]], head: str) -> dict[str, Any]:
    rows = [diagnostic["heads"][head]["teacher_rescue_credit_assignment"] for diagnostic in diagnostics]
    corrected = sum(row["corrected_samples"] for row in rows)
    positive = sum(row["corrected_positive_advantage_samples"] for row in rows)
    leakage = sum(row["corrected_positive_advantage_negative_cosine_samples"] for row in rows)
    active = sum(row["active_credit_leakage_samples"] for row in rows)
    masked = sum(row["ppo_masked_corrected_samples"] for row in rows)
    pre_cosines = [sample["ppo_bc_cosine_pre_mask"] for row in rows for sample in row["samples"]
                   if sample["ppo_bc_cosine_pre_mask"] is not None]
    return {
        "corrected_samples": corrected, "positive_advantage_samples": positive,
        "teacher_rescue_credit_leakage_samples": leakage,
        "teacher_rescue_credit_leakage_fraction": None if not corrected else leakage / corrected,
        "active_credit_leakage_samples": active,
        "active_credit_leakage_fraction": None if not corrected else active / corrected,
        "masked_corrected_samples": masked,
        "pre_mask_ppo_bc_cosine": {**_quantiles(pre_cosines), "negative_fraction": None if not pre_cosines else float(np.mean(np.asarray(pre_cosines) < 0.0))},
    }


def _variant_summary(records: list[dict[str, Any]], provenance: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [record["diagnostic"] for record in records]
    episode_rows = [SimpleNamespace(**row) for record in records for row in record["summaries"]]
    overall = _summary_metrics(episode_rows)
    warm = _grouped_metrics(episode_rows)["warm_start_finite_cohort"]
    heads = {}
    for head in HEADS:
        diagnostics_for_head = [row["heads"][head] for row in diagnostics]
        cosine = [row["teacher_used_head_output_gradient_cosine"] for row in diagnostics_for_head]
        negative_by_update = [row["negative_fraction"] for row in cosine]
        active_negative = [value for value in negative_by_update if value is not None]
        credit = _head_credit(diagnostics, head)
        heads[head] = {
            "credit": credit,
            "ppo_bc_negative_cosine_by_update": negative_by_update,
            "ppo_bc_negative_cosine_mean": None if not active_negative else float(np.mean(active_negative)),
            # For C, the pre-mask value reports the counterfactual conflict
            # that would have occurred; the active value above reports the
            # gradient actually admitted to the PPO actor loss.
            "pre_mask_ppo_bc_negative_cosine_mean": credit["pre_mask_ppo_bc_cosine"]["negative_fraction"],
            "near_boundary_rate_by_update": [row["saturation_rate"] for row in diagnostics_for_head],
            "actor_mean_by_update": [row["post_squash_actor_mean"] for row in diagnostics_for_head],
            "actor_std_by_update": [row["std"] for row in diagnostics_for_head],
        }
    safety = {key: int(sum(getattr(row, key) for row in episode_rows))
              for key in ("collisions", "nmac", "hard_separation_violations")}
    return {
        "records": records, "provenance": provenance, "provenance_passed": all(row["passed"] for row in provenance),
        "heads": heads, "teacher_burden_overall": overall["teacher_interventions_per_100_release_intents"],
        "teacher_burden_warm": warm["teacher_interventions_per_100_release_intents"],
        "correction_mean": overall["correction_magnitude_mean"], "correction_p95": overall["correction_magnitude_p95"],
        "shield_burden": overall["shield_interventions_per_100_release_intents"],
        "completion_ratio": overall["completion_ratio"], "throughput_per_second": overall["throughput_per_second"],
        "ppo_health": _health(records), "safety": safety, "safety_passed": not any(safety.values()),
    }


def _not_worse(candidate: float | None, reference: float | None) -> bool:
    return candidate is not None and (reference is None or candidate <= reference + 1e-9)


def _decision(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a, b, c = (variants[key] for key in VARIANTS)
    release_a = a["heads"]["release_request"]["credit"]
    release_c = c["heads"]["release_request"]["credit"]
    leakage_established = bool((release_a["teacher_rescue_credit_leakage_samples"] > 0) and
                               (release_a["teacher_rescue_credit_leakage_fraction"] or 0.0) > 0.0)
    conflict_reduced = (release_c["active_credit_leakage_samples"] < release_a["active_credit_leakage_samples"] and
                        release_c["masked_corrected_samples"] > 0)
    nonregression = all(_not_worse(c[key], a[key]) for key in
                        ("teacher_burden_overall", "teacher_burden_warm", "correction_p95", "shield_burden"))
    efficiency = bool(c["completion_ratio"] is not None and a["completion_ratio"] is not None and
                      c["completion_ratio"] >= .95 * a["completion_ratio"] and
                      c["throughput_per_second"] >= .95 * a["throughput_per_second"])
    c_passed = bool(leakage_established and conflict_reduced and nonregression and efficiency and
                    c["safety_passed"] and c["ppo_health"]["healthy"] and c["provenance_passed"])
    return {
        "teacher_rescue_credit_leakage_established": leakage_established,
        "worst_head": max(HEADS, key=lambda head: a["heads"][head]["credit"]["teacher_rescue_credit_leakage_fraction"] or -1.0),
        "squashed_bc_improves_release_leakage": b["heads"]["release_request"]["credit"]["active_credit_leakage_samples"] < release_a["active_credit_leakage_samples"],
        "headwise_mask_improves_release_leakage": conflict_reduced,
        "C_full_acceptance": c_passed,
        "allow_resume_formal_l1": c_passed,
        "next_step": ("intervention-aware head-wise PPO masking may enter the next formal L1 configuration" if c_passed else
                      "do not resume formal L1; run a release-head architecture / objective diagnostic without changing reward"),
    }


def _markdown(result: dict[str, Any]) -> str:
    decision, variants = result["decision"], result["variants"]
    lines = ["# L1 PPO-D4 Credit Assignment Summary", ""]
    lines.append(f"- teacher-rescue credit leakage：{'成立' if decision['teacher_rescue_credit_leakage_established'] else '未成立'}；最严重 head：{decision['worst_head']}。")
    lines.append(f"- squashed-consistent BC：{'改善 release leakage' if decision['squashed_bc_improves_release_leakage'] else '未改善 release leakage'}。")
    lines.append(f"- head-wise PPO masking：{'改善 release leakage' if decision['headwise_mask_improves_release_leakage'] else '未证明改善'}。")
    lines.append("\n| Arm | release leakage / corrected | teacher overall/warm | correction P95 | shield | completion | throughput | safety |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, values in variants.items():
        credit = values["heads"]["release_request"]["credit"]
        lines.append(f"| {name} | {credit['active_credit_leakage_samples']}/{credit['corrected_samples']} | "
                     f"{values['teacher_burden_overall']:.3f}/{values['teacher_burden_warm']:.3f} | {values['correction_p95']:.3f} | "
                     f"{values['shield_burden']:.3f} | {values['completion_ratio']:.4f} | {values['throughput_per_second']:.6f} | {values['safety']} |")
    lines.append(f"\n- 是否允许恢复正式 L1 Full-Teacher PPO：{'允许' if decision['allow_resume_formal_l1'] else '不允许'}。")
    lines.append(f"- 下一步：{decision['next_step']}。")
    return "\n".join(lines) + "\n"


def run_l1_ppo_d4_credit_assignment(*, c0_report: str | Path, teacher_dataset_path: str | Path,
                                      initial_checkpoint: str | Path, output_dir: str | Path,
                                      rollout_workers: int = 4) -> dict[str, Any]:
    """Run only the frozen 3x3-update D4 comparison; never enter formal L1."""
    destination = Path(output_dir)
    audit = authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    mu_hat, medium_interval = _read_frozen_capacity(c0_report)
    medium_rate = sum(medium_interval) / 2.0
    config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                    warm_start_uavs=8, cohort_uavs=8, horizon_seconds=1200.0)
    checkpoint = Path(initial_checkpoint)
    manifest = {"stage": "L1_PPO_D4_intervention_credit_assignment", "evidence_type": "newly_run",
                "initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "variants": VARIANTS, "teacher_dataset_audit": asdict(audit),
                "frozen": {"updates": 3, "episodes_per_update": 4, "max_steps": 1200, "teacher_bc_weight": .05,
                           "mu_hat": mu_hat, "medium_rate": medium_rate, "policy": "tanh_squashed_gaussian",
                           "reward": "unchanged", "observation": "V2", "action_range": "[-1,1]", "safety_stack": "unchanged"},
                "provenance": {"buffer_action": "a_requested", "old_log_prob": "bounded requested log_prob",
                               "a_applied_enters_ratio": False, "critic_return_learning": "enabled"}}
    _write(destination / "l1_ppo_d4.manifest.json", manifest)
    _write(destination / "l1_ppo_d4.startup_preflight.json", {
        "passed": True, "variants": list(VARIANTS), "updates_per_variant": 3,
        "episodes_per_update": 4, "collect_requested_log_prob_by_head": True,
        "output_contract": ["<arm>.updates.json after each update",
                            "l1_ppo_d4_credit_assignment_summary.json",
                            "l1_ppo_d4_credit_assignment_summary.md"],
    })
    variants: dict[str, Any] = {}
    for label, settings in VARIANTS.items():
        random.seed(1701); np.random.seed(1701); torch.manual_seed(1701)
        policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05, policy_distribution="tanh_squashed_gaussian", **settings)
        load_checkpoint(checkpoint, policy, config)
        records, provenance = [], []
        seeds, design_index = itertools.cycle(partition.train), 0
        try:
            for update in range(3):
                state, tasks, designs = _model_state(policy), [], []
                for episode in range(4):
                    episode_config, design = _episode_design(design_index, medium_rate=medium_rate, horizon_seconds=1200.0)
                    design_index += 1
                    tasks.append((state, episode_config, next(seeds), update, episode, 1200, False, None,
                                  "tanh_squashed_gaussian", True))
                    designs.append(design)
                if rollout_workers == 1:
                    collected = [_rollout_task(task) for task in tasks]
                else:
                    # Join every rollout worker before serialising this update.
                    # A partially shut-down executor previously made long runs
                    # look healthy while leaving no usable intermediate output.
                    with ProcessPoolExecutor(max_workers=min(rollout_workers, len(tasks))) as executor:
                        collected = list(executor.map(_rollout_task, tasks))
                buffer, summaries = RolloutBuffer(), []
                for (transitions, summary), design in zip(collected, designs):
                    for transition in transitions: buffer.add(transition)
                    summaries.append({**asdict(summary), **design})
                if any(row["collisions"] or row["nmac"] or row["hard_separation_violations"] for row in summaries):
                    raise L1SafetyGateError(f"D4 safety gate failed in {label} before update {update}")
                proof = _requested_action_provenance(policy, buffer)
                if not proof["passed"]: raise RuntimeError(f"D4 provenance failed in {label}: {proof}")
                if not all(row.log_prob_by_head is not None for agent_rows in buffer.by_agent.values() for row in agent_rows):
                    raise RuntimeError("D4 requires requested log-probability by head")
                provenance.append(proof)
                diagnostic = policy.diagnostic_snapshot(buffer, teacher_bc_weight=.05)
                update_metrics = policy.update(buffer, epochs=2, minibatch_size=256)
                records.append({"update": update, "summaries": summaries, "diagnostic": diagnostic, "ppo_update": update_metrics})
                _write(destination / f"{label}.updates.json", {"manifest": manifest, "records": records})
            variants[label] = _variant_summary(records, provenance)
        except (SafetyInvariantError, L1SafetyGateError, RuntimeError, FloatingPointError) as error:
            variants[label] = {"status": "stopped", "reason": str(error), "records": records,
                               "provenance": provenance, "safety_passed": False, "provenance_passed": False}
            break
    if len(variants) == len(VARIANTS) and all("heads" in row for row in variants.values()):
        decision = _decision(variants)
        result = {"status": "completed", "manifest": manifest, "variants": variants, "decision": decision}
    else:
        result = {"status": "stopped", "manifest": manifest, "variants": variants,
                  "decision": {"allow_resume_formal_l1": False, "next_step": "fix D4 runtime failure before another diagnostic"}}
    _write(destination / "l1_ppo_d4_credit_assignment_summary.json", result)
    (destination / "l1_ppo_d4_credit_assignment_summary.md").write_text(_markdown(result) if result["status"] == "completed" else
        "# L1 PPO-D4 Credit Assignment Summary\n\n- 诊断未完成；不允许恢复正式 L1。\n", encoding="utf-8")
    return result


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run only the frozen L1 PPO-D4 credit-assignment diagnostic.")
    parser.add_argument("--c0-report", required=True); parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_l1_ppo_d4_credit_assignment(c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
                                                      initial_checkpoint=args.initial_checkpoint, output_dir=args.output,
                                                      rollout_workers=args.workers), indent=2))


if __name__ == "__main__":
    _main()
