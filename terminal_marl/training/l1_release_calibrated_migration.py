"""Decision-probability-preserving Bernoulli release migration diagnostic."""
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
from .l1_release_d5a_objective import _d5a_gate
from .l1_release_mixed_actor_diagnostic import (
    _variant_summary, _write, release_decision_metrics, release_objective_metrics,
)
from .train_low_v2_ppo import (
    L1SafetyGateError, _episode_design, _model_state, _read_frozen_capacity,
    _read_seed_partition, _rollout_task, authorize_low_v2_training,
    l1_full_teacher_config,
)


LABEL = "E_probability_calibrated_bernoulli_release"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def release_checkpoint_probability_audit(policy: SharedPPO, dataset_path: str | Path) -> dict[str, Any]:
    observations = np.asarray([
        json.loads(line)["observation"] for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ], dtype=np.float32)
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    with torch.no_grad():
        hidden = policy.model.actor_body(tensor)
        scores = policy.model.actor_mean(hidden)
        scale = policy.model.log_std.exp().expand_as(scores)
        source_probability = 1.0 - torch.distributions.Normal(scores[:, 0], scale[:, 0]).cdf(
            torch.zeros_like(scores[:, 0]))
        plain_probability = torch.sigmoid(scores[:, 0])
        calibrated_probability = policy.distribution(tensor).release_probability
        plain_error = (source_probability - plain_probability).abs()
        calibrated_error = (source_probability - calibrated_probability).abs()
    result = {
        "rows": len(observations), "checkpoint_release_std": float(scale[0, 0]),
        "source_continuous_threshold_probability_mean": float(source_probability.mean()),
        "plain_bernoulli_probability_mean": float(plain_probability.mean()),
        "calibrated_bernoulli_probability_mean": float(calibrated_probability.mean()),
        "plain_mean_abs_error": float(plain_error.mean()),
        "plain_p95_abs_error": float(torch.quantile(plain_error, .95)),
        "calibrated_mean_abs_error": float(calibrated_error.mean()),
        "calibrated_p95_abs_error": float(torch.quantile(calibrated_error, .95)),
    }
    result["passed"] = bool(result["calibrated_mean_abs_error"] <= .01 and
                            result["calibrated_p95_abs_error"] <= .02)
    return result


def run_release_calibrated_migration(
    *, c0_report: str | Path, teacher_dataset_path: str | Path,
    initial_checkpoint: str | Path, baseline_summary: str | Path,
    d5a_summary: str | Path, targeted_probe: str | Path, output_dir: str | Path,
    rollout_workers: int = 4, source_commit: str = "unknown", seed_offset: int = 0,
    updates: int = 3, max_steps: int = 1200,
) -> dict[str, Any]:
    destination = Path(output_dir)
    audit = authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    selected_seeds = tuple(partition.train[seed_offset:seed_offset + 4 * updates])
    if len(selected_seeds) != 4 * updates:
        raise RuntimeError("calibrated release diagnostic lacks the requested independent seed window")
    mu_hat, medium_interval = _read_frozen_capacity(c0_report)
    medium_rate = sum(medium_interval) / 2.0
    checkpoint = Path(initial_checkpoint)
    baseline_path, d5a_path, probe_path = map(Path, (baseline_summary, d5a_summary, targeted_probe))
    baseline, d5a, probe = map(_load_json, (baseline_path, d5a_path, probe_path))
    if baseline.get("status") != "completed" or d5a.get("status") != "completed":
        raise RuntimeError("calibrated migration requires completed A/B and D5a evidence")
    random.seed(1701); np.random.seed(1701); torch.manual_seed(1701)
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05,
                       policy_distribution="calibrated_bernoulli_tanh")
    root_config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings",
                                         demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8,
                                         horizon_seconds=float(max_steps))
    checkpoint_load = load_checkpoint(checkpoint, policy, root_config)
    if checkpoint_load.get("policy_architecture_migrated"):
        raise RuntimeError("calibrated Bernoulli must use the unchanged checkpoint parameters")
    probability_audit = release_checkpoint_probability_audit(policy, teacher_dataset_path)
    config_payload = {
        "updates": updates, "episodes_per_update": 4, "max_steps": max_steps,
        "teacher_bc_weight": .05, "learning_rate": 3e-4, "entropy_coefficient": .01,
        "mu_hat": mu_hat, "medium_rate": medium_rate, "seed_offset": seed_offset,
        "seeds": list(selected_seeds),
    }
    manifest = {
        "stage": "L1_release_probability_calibrated_migration", "evidence_type": "newly_run",
        "source_commit": source_commit,
        "config_fingerprint": hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "baseline_summary_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "d5a_summary_sha256": hashlib.sha256(d5a_path.read_bytes()).hexdigest(),
        "targeted_probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "teacher_dataset_audit": asdict(audit), "config": config_payload,
        "probability_migration_audit": probability_audit,
        "variant": {
            "distribution": "Bernoulli(logit=1.702*release_score/checkpoint_release_std) + tanh residuals",
            "release_temperature_trainable": False,
            "external_action": "[-1/+1 release, tanh speed, tanh merge]",
            "release_label_source": "isolated counterfactual all-RELEASE scheduler/safety decision",
        },
        "hypothesis": "preserving the source checkpoint's threshold-decision probability removes the prior A/B migration confound",
        "frozen": ["checkpoint parameters", "optimizer state", "reward", "BC weight", "learning rate",
                   "entropy coefficient", "traffic", "Observation V2", "SafetyFilter/Shield", "EH216-S dynamics",
                   "mu_hat/lambda", "action interface"],
        "provenance": {"buffer_action": "a_requested", "old_log_prob": "actual calibrated Bernoulli+tanh joint log_prob",
                       "a_applied_enters_ratio": False, "counterfactual_step_enters_return": False},
    }
    _write(destination / "release_calibrated.manifest.json", manifest)
    _write(destination / "release_calibrated.startup_preflight.json", {
        "passed": probability_audit["passed"],
        "expected_outputs": [f"{LABEL}.updates.json", "release_calibrated_summary.json",
                             "release_calibrated_summary.md"],
        "required_record_fields": ["release", "release_objective", "ppo_update", "summaries"],
    })
    if not probability_audit["passed"]:
        raise RuntimeError(f"release checkpoint probability migration failed: {probability_audit}")

    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    seeds, design_index = itertools.cycle(selected_seeds), 0
    try:
        for update in range(updates):
            state, tasks, designs = _model_state(policy), [], []
            for episode in range(4):
                episode_config, design = _episode_design(design_index, medium_rate=medium_rate,
                                                         horizon_seconds=float(max_steps))
                design_index += 1
                tasks.append((state, episode_config, next(seeds), update, episode, max_steps, False, None,
                              "calibrated_bernoulli_tanh", True, True, True, True))
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
                raise L1SafetyGateError(f"calibrated release safety gate failed in update {update}")
            proof = _requested_action_provenance(policy, buffer)
            if not proof["passed"]:
                raise RuntimeError(f"calibrated release provenance failed: {proof}")
            release = release_decision_metrics(policy, transitions)
            objective = release_objective_metrics(policy, buffer)
            provenance.append(proof)
            ppo = policy.update(buffer, epochs=2, minibatch_size=256)
            records.append({"update": update, "summaries": summaries, "release": release,
                            "release_objective": objective, "ppo_update": ppo})
            _write(destination / f"{LABEL}.updates.json", {"manifest": manifest, "records": records})
        candidate = _variant_summary(records, provenance)
        control = baseline["variants"]["A_continuous_release_tanh"]
        gate = _d5a_gate(control, candidate, targeted_probe_passed=bool(probe.get("passed")))
        gate["checkpoint_probability_migration_passed"] = probability_audit["passed"]
        gate["window_passed"] = bool(gate["window_passed"] and probability_audit["passed"])
        if not gate["window_passed"]:
            gate["next_step"] = "run one-factor release loss-coefficient/learning-rate ablation; do not alter global settings"
        result = {"status": "completed", "manifest": manifest, "control": control,
                  "prior_plain_bernoulli": baseline["variants"]["B_bernoulli_release_tanh_residuals"],
                  "prior_counterfactual_uncalibrated": d5a["candidate"],
                  "candidate": candidate, "decision": gate}
    except (SafetyInvariantError, L1SafetyGateError, RuntimeError, FloatingPointError) as error:
        result = {"status": "stopped", "manifest": manifest, "records": records,
                  "decision": {"window_passed": False, "next_step": "repair calibrated migration runtime"},
                  "reason": str(error)}
    _write(destination / "release_calibrated_summary.json", result)
    decision = result["decision"]
    lines = ["# L1 Probability-Calibrated Bernoulli Release", "", f"- status: {result['status']}",
             f"- probability migration: {'PASS' if probability_audit['passed'] else 'FAIL'}",
             f"- window: {'PASS' if decision.get('window_passed') else 'FAIL'}"]
    if result["status"] == "completed":
        for name, values in (("A continuous", result["control"]), ("E calibrated", result["candidate"])):
            lines.append(f"- {name}: release correction={values['release']['teacher_release_correction_rate']:.4f}, "
                         f"teacher overall/warm={values['teacher_burden_overall']:.3f}/{values['teacher_burden_warm']:.3f}, "
                         f"P95={values['correction_p95']:.3f}, shield={values['shield_burden']:.3f}, "
                         f"completion={values['completion_ratio']:.4f}, throughput={values['throughput_per_second']:.6f}, "
                         f"PPO healthy={values['ppo_health']['healthy']}, safety={values['safety']}")
    lines.append(f"- next: {decision['next_step']}")
    (destination / "release_calibrated_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run probability-calibrated Bernoulli release diagnostic.")
    parser.add_argument("--c0-report", required=True); parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True); parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--d5a-summary", required=True); parser.add_argument("--targeted-probe", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-commit", default="unknown"); parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--updates", type=int, default=3); parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(run_release_calibrated_migration(
        c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
        initial_checkpoint=args.initial_checkpoint, baseline_summary=args.baseline_summary,
        d5a_summary=args.d5a_summary, targeted_probe=args.targeted_probe, output_dir=args.output,
        rollout_workers=args.workers, source_commit=args.source_commit, seed_offset=args.seed_offset,
        updates=args.updates, max_steps=args.max_steps), indent=2))


if __name__ == "__main__":
    _main()
