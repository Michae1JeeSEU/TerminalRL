"""Bounded D5a counterfactual release-objective diagnostic.

This runner changes only how the auxiliary Bernoulli release label is queried.
The executed rollout, rewards, BC weight, PPO settings, traffic, dynamics, and
safety stack remain frozen.
"""
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
from ..core.observations import LOW_OBS_V2
from ..core.runtime_traffic_safety import SafetyInvariantError
from .l1_ppo_d3_bounded_distribution import _requested_action_provenance
from .l1_release_mixed_actor_diagnostic import (
    _not_worse, _variant_summary, _write, release_decision_metrics,
    release_objective_metrics,
)
from .train_low_v2_ppo import (
    L1SafetyGateError, _episode_design, _model_state, _read_frozen_capacity,
    _read_seed_partition, _rollout_task, authorize_low_v2_training,
    l1_full_teacher_config,
)


LABEL = "C_mixed_counterfactual_release_teacher"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_completed_baseline(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError("D5a requires a completed frozen A/B summary")
    return payload


def _d5a_gate(control: dict[str, Any], candidate: dict[str, Any], *, targeted_probe_passed: bool) -> dict[str, Any]:
    release_reduced = bool(candidate["release"]["teacher_release_correction_rate"] <=
                           .80 * control["release"]["teacher_release_correction_rate"])
    burden = all(_not_worse(candidate[key], control[key]) for key in
                 ("teacher_burden_overall", "teacher_burden_warm", "correction_p95", "shield_burden"))
    performance = bool(candidate["completion_ratio"] >= .98 * control["completion_ratio"] and
                       candidate["throughput_per_second"] >= .98 * control["throughput_per_second"])
    passed = bool(targeted_probe_passed and release_reduced and burden and performance and
                  candidate["ppo_health"]["healthy"] and candidate["safety_passed"] and
                  candidate["provenance_passed"])
    return {
        "release_targeted_regression_passed": targeted_probe_passed,
        "release_correction_reduced_20pct": release_reduced,
        "teacher_p95_shield_not_worse": burden,
        "completion_throughput_healthy": performance,
        "ppo_healthy": candidate["ppo_health"]["healthy"],
        "safety_passed": candidate["safety_passed"],
        "provenance_passed": candidate["provenance_passed"],
        "window_passed": passed,
        "next_step": ("run an independent counterfactual-label confirmation window" if passed else
                      "continue D5b observation-sufficiency diagnostic; do not add PPO steps blindly"),
    }


def run_d5a_runtime_preflight(
    *, c0_report: str | Path, teacher_dataset_path: str | Path,
    initial_checkpoint: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    """Exercise the real counterfactual rollout and serialized D5a fields."""
    authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    _, medium_interval = _read_frozen_capacity(c0_report)
    medium_rate = sum(medium_interval) / 2.0
    random.seed(1701); np.random.seed(1701); torch.manual_seed(1701)
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05,
                       policy_distribution="mixed_bernoulli_tanh")
    config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings",
                                    demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8,
                                    horizon_seconds=2.0)
    load_checkpoint(Path(initial_checkpoint), policy, config)
    transitions, summary = _rollout_task((
        _model_state(policy), config, partition.train[0], 0, 0, 2, False, None,
        "mixed_bernoulli_tanh", True, True, True, True,
    ))
    buffer = RolloutBuffer()
    for transition in transitions:
        buffer.add(transition)
    provenance = _requested_action_provenance(policy, buffer)
    release = release_decision_metrics(policy, transitions)
    objective = release_objective_metrics(policy, buffer)
    safety = {name: int(getattr(summary, name)) for name in
              ("collisions", "nmac", "hard_separation_violations")}
    required = {
        "release": all(key in release for key in
                       ("label_counts", "false_release", "false_wait", "teacher_release_correction_rate")),
        "objective": all(key in objective for key in
                         ("label_balance", "advantage_by_requested_decision", "release_binary_entropy_mean", "calibration")),
        "teacher_masks": bool(transitions) and all(row.teacher_head_mask is not None for row in transitions),
        "head_log_probs": bool(transitions) and all(row.log_prob_by_head is not None for row in transitions),
    }
    result = {
        "status": "completed", "evidence_type": "newly_run", "rollout_rows": len(transitions),
        "required_fields": required, "release": release, "release_objective": objective,
        "provenance": provenance, "safety": safety,
        "passed": bool(transitions and all(required.values()) and provenance["passed"] and not any(safety.values())),
    }
    _write(Path(output_dir) / "release_d5a.runtime_preflight.json", result)
    if not result["passed"]:
        raise RuntimeError(f"D5a runtime preflight failed: {result}")
    return result


def run_release_d5a_objective(
    *, c0_report: str | Path, teacher_dataset_path: str | Path,
    initial_checkpoint: str | Path, baseline_summary: str | Path,
    targeted_probe: str | Path, output_dir: str | Path,
    rollout_workers: int = 4, source_commit: str = "unknown", seed_offset: int = 0,
) -> dict[str, Any]:
    destination = Path(output_dir)
    audit = authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    selected_seeds = tuple(partition.train[seed_offset:seed_offset + 12])
    if len(selected_seeds) != 12:
        raise RuntimeError("D5a requires twelve independent training seeds at the requested offset")
    mu_hat, medium_interval = _read_frozen_capacity(c0_report)
    medium_rate = sum(medium_interval) / 2.0
    checkpoint = Path(initial_checkpoint)
    baseline_path, probe_path = Path(baseline_summary), Path(targeted_probe)
    baseline = _load_completed_baseline(baseline_path)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    config_payload = {
        "updates": 3, "episodes_per_update": 4, "max_steps": 1200,
        "teacher_bc_weight": .05, "learning_rate": 3e-4, "entropy_coefficient": .01,
        "mu_hat": mu_hat, "medium_rate": medium_rate, "seed_offset": seed_offset,
        "seeds": list(selected_seeds),
    }
    manifest = {
        "stage": "L1_release_D5a_counterfactual_objective", "evidence_type": "newly_run",
        "source_commit": source_commit,
        "config_fingerprint": hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "baseline_summary": str(baseline_path), "baseline_summary_sha256": _sha256(baseline_path),
        "targeted_probe": str(probe_path), "targeted_probe_sha256": _sha256(probe_path),
        "teacher_dataset_audit": asdict(audit), "config": config_payload,
        "variant": {
            "policy_distribution": "mixed_bernoulli_tanh", "release_teacher_bc": True,
            "release_label_source": "isolated counterfactual all-RELEASE step through unchanged scheduler and safety stack",
        },
        "hypothesis": "policy-dependent proxy WAIT labels prevent calibrated Bernoulli release learning",
        "code_change": "replace only release auxiliary labels with counterfactual scheduler/safety decisions",
        "frozen": ["reward", "BC weight", "learning rate", "entropy coefficient", "traffic", "Observation V2",
                   "SafetyFilter/Shield", "EH216-S dynamics", "mu_hat/lambda", "action interface"],
        "provenance": {"buffer_action": "a_requested", "old_log_prob": "actual mixed requested distribution",
                       "a_applied_enters_ratio": False, "counterfactual_step_enters_return": False},
    }
    _write(destination / "release_d5a.manifest.json", manifest)
    _write(destination / "release_d5a.startup_preflight.json", {
        "passed": True,
        "expected_outputs": ["C_mixed_counterfactual_release_teacher.updates.json", "release_d5a_summary.json",
                             "release_d5a_summary.md"],
        "required_record_fields": ["release", "release_objective", "ppo_update", "summaries"],
    })

    random.seed(1701); np.random.seed(1701); torch.manual_seed(1701)
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05,
                       policy_distribution="mixed_bernoulli_tanh")
    config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings",
                                    demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8,
                                    horizon_seconds=1200.0)
    load_checkpoint(checkpoint, policy, config)
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    seeds, design_index = itertools.cycle(selected_seeds), 0
    try:
        for update in range(3):
            state, tasks, designs = _model_state(policy), [], []
            for episode in range(4):
                episode_config, design = _episode_design(design_index, medium_rate=medium_rate,
                                                         horizon_seconds=1200.0)
                design_index += 1
                tasks.append((state, episode_config, next(seeds), update, episode, 1200, False, None,
                              "mixed_bernoulli_tanh", True, True, True, True))
                designs.append(design)
            if rollout_workers == 1:
                collected = [_rollout_task(task) for task in tasks]
            else:
                with ProcessPoolExecutor(max_workers=min(rollout_workers, len(tasks))) as executor:
                    collected = list(executor.map(_rollout_task, tasks))
            buffer, summaries, transitions = RolloutBuffer(), [], []
            for (episode_transitions, summary), design in zip(collected, designs):
                transitions.extend(episode_transitions)
                for transition in episode_transitions:
                    buffer.add(transition)
                summaries.append({**asdict(summary), **design})
            if any(row["collisions"] or row["nmac"] or row["hard_separation_violations"] for row in summaries):
                raise L1SafetyGateError(f"D5a counterfactual safety gate failed in update {update}")
            proof = _requested_action_provenance(policy, buffer)
            if not proof["passed"]:
                raise RuntimeError(f"D5a counterfactual provenance failed: {proof}")
            release = release_decision_metrics(policy, transitions)
            objective = release_objective_metrics(policy, buffer)
            provenance.append(proof)
            ppo = policy.update(buffer, epochs=2, minibatch_size=256)
            records.append({"update": update, "summaries": summaries, "release": release,
                            "release_objective": objective, "ppo_update": ppo})
            _write(destination / f"{LABEL}.updates.json", {"manifest": manifest, "records": records})
        candidate = _variant_summary(records, provenance)
        control = baseline["variants"]["A_continuous_release_tanh"]
        prior_mixed = baseline["variants"]["B_bernoulli_release_tanh_residuals"]
        gate = _d5a_gate(control, candidate, targeted_probe_passed=bool(probe.get("passed")))
        result = {"status": "completed", "manifest": manifest, "control": control,
                  "prior_mixed_proxy_labels": prior_mixed, "candidate": candidate, "decision": gate}
    except (SafetyInvariantError, L1SafetyGateError, RuntimeError, FloatingPointError) as error:
        result = {"status": "stopped", "manifest": manifest, "records": records,
                  "decision": {"window_passed": False, "next_step": "repair diagnostic runtime"},
                  "reason": str(error)}
    _write(destination / "release_d5a_summary.json", result)
    decision = result["decision"]
    lines = ["# L1 PPO D5a Release Objective Diagnostic", "",
             f"- status: {result['status']}",
             f"- counterfactual-label window: {'PASS' if decision.get('window_passed') else 'FAIL'}",
             f"- targeted release regression: {'PASS' if decision.get('release_targeted_regression_passed') else 'FAIL'}"]
    if result["status"] == "completed":
        for name, values in (("A continuous control", result["control"]),
                             ("B mixed proxy labels", result["prior_mixed_proxy_labels"]),
                             ("C mixed counterfactual labels", result["candidate"])):
            lines.append(f"- {name}: release correction={values['release']['teacher_release_correction_rate']:.4f}, "
                         f"teacher overall/warm={values['teacher_burden_overall']:.3f}/{values['teacher_burden_warm']:.3f}, "
                         f"P95={values['correction_p95']:.3f}, shield={values['shield_burden']:.3f}, "
                         f"completion={values['completion_ratio']:.4f}, throughput={values['throughput_per_second']:.6f}, "
                         f"PPO healthy={values['ppo_health']['healthy']}, safety={values['safety']}")
    lines.append(f"- next: {decision['next_step']}")
    (destination / "release_d5a_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run three-update counterfactual-label D5a diagnostic.")
    parser.add_argument("--c0-report", required=True); parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True); parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--targeted-probe", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(run_d5a_runtime_preflight(
            c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
            initial_checkpoint=args.initial_checkpoint, output_dir=args.output), indent=2))
        return
    print(json.dumps(run_release_d5a_objective(
        c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
        initial_checkpoint=args.initial_checkpoint, baseline_summary=args.baseline_summary,
        targeted_probe=args.targeted_probe, output_dir=args.output, rollout_workers=args.workers,
        source_commit=args.source_commit, seed_offset=args.seed_offset), indent=2))


if __name__ == "__main__":
    _main()
