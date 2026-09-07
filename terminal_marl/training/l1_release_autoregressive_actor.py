"""Three-update autoregressive release-over-residual actor diagnostic."""
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


LABEL = "D_autoregressive_release_over_tanh_residuals"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_release_autoregressive_actor(
    *, c0_report: str | Path, teacher_dataset_path: str | Path,
    initial_checkpoint: str | Path, baseline_summary: str | Path,
    d5a_summary: str | Path, d5c_summary: str | Path, targeted_probe: str | Path,
    output_dir: str | Path, rollout_workers: int = 4, source_commit: str = "unknown",
    seed_offset: int = 0, updates: int = 3, max_steps: int = 1200,
) -> dict[str, Any]:
    destination = Path(output_dir)
    audit = authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    selected_seeds = tuple(partition.train[seed_offset:seed_offset + 4 * updates])
    if len(selected_seeds) != 4 * updates:
        raise RuntimeError("autoregressive diagnostic lacks the requested independent seed window")
    mu_hat, medium_interval = _read_frozen_capacity(c0_report)
    medium_rate = sum(medium_interval) / 2.0
    checkpoint = Path(initial_checkpoint)
    baseline_path, d5a_path, d5c_path, probe_path = map(
        Path, (baseline_summary, d5a_summary, d5c_summary, targeted_probe))
    baseline, d5a, d5c, probe = map(_load_json, (baseline_path, d5a_path, d5c_path, probe_path))
    if baseline.get("status") != "completed" or d5a.get("status") != "completed" or d5c.get("status") != "completed":
        raise RuntimeError("autoregressive diagnostic requires completed A/B, D5a, and D5c evidence")
    if not d5c["analysis"]["strong_release_merge_coupling"]:
        raise RuntimeError("D5c did not authorize a conditional release architecture")
    config_payload = {
        "updates": updates, "episodes_per_update": 4, "max_steps": max_steps,
        "teacher_bc_weight": .05, "learning_rate": 3e-4, "entropy_coefficient": .01,
        "mu_hat": mu_hat, "medium_rate": medium_rate, "seed_offset": seed_offset,
        "seeds": list(selected_seeds),
    }
    manifest = {
        "stage": "L1_release_autoregressive_actor", "evidence_type": "newly_run",
        "source_commit": source_commit,
        "config_fingerprint": hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "baseline_summary_sha256": _sha256(baseline_path), "d5a_summary_sha256": _sha256(d5a_path),
        "d5c_summary_sha256": _sha256(d5c_path), "targeted_probe_sha256": _sha256(probe_path),
        "teacher_dataset_audit": asdict(audit), "config": config_payload,
        "variant": {
            "distribution": "p(speed,merge|observation) * p(release|observation,speed,merge)",
            "external_action": "[-1/+1 release, tanh speed, tanh merge]",
            "release_teacher_loss": "binary BCE conditioned on teacher residual target",
            "release_label_source": "isolated counterfactual all-RELEASE scheduler/safety decision",
            "checkpoint_migration": "preserve actor/critic/optimizer state; zero-initialize only release conditioner",
        },
        "hypothesis": "conditioning Bernoulli release on residuals resolves the D5b/D5c cross-head dependency",
        "frozen": ["reward", "BC weight", "learning rate", "entropy coefficient", "traffic", "Observation V2",
                   "SafetyFilter/Shield", "EH216-S dynamics", "mu_hat/lambda", "action interface"],
        "provenance": {"buffer_action": "a_requested", "old_log_prob": "exact autoregressive joint log_prob",
                       "a_applied_enters_ratio": False, "counterfactual_step_enters_return": False},
    }
    _write(destination / "release_autoregressive.manifest.json", manifest)
    _write(destination / "release_autoregressive.startup_preflight.json", {
        "passed": True, "expected_outputs": [f"{LABEL}.updates.json", "release_autoregressive_summary.json",
                                               "release_autoregressive_summary.md"],
        "required_record_fields": ["release", "release_objective", "ppo_update", "summaries"],
    })
    random.seed(1701); np.random.seed(1701); torch.manual_seed(1701)
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05,
                       policy_distribution="autoregressive_release_tanh")
    root_config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings",
                                         demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8,
                                         horizon_seconds=float(max_steps))
    checkpoint_load = load_checkpoint(checkpoint, policy, root_config, allow_policy_migration=True)
    if not checkpoint_load.get("policy_architecture_migrated"):
        raise RuntimeError("expected zero-initialized release-conditioner checkpoint migration")
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
                              "autoregressive_release_tanh", True, True, True, True))
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
                raise L1SafetyGateError(f"autoregressive release safety gate failed in update {update}")
            proof = _requested_action_provenance(policy, buffer)
            if not proof["passed"]:
                raise RuntimeError(f"autoregressive release provenance failed: {proof}")
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
        result = {"status": "completed", "manifest": manifest, "control": control,
                  "prior_mixed_proxy_labels": baseline["variants"]["B_bernoulli_release_tanh_residuals"],
                  "prior_counterfactual_independent_heads": d5a["candidate"],
                  "candidate": candidate, "decision": gate}
    except (SafetyInvariantError, L1SafetyGateError, RuntimeError, FloatingPointError) as error:
        result = {"status": "stopped", "manifest": manifest, "records": records,
                  "decision": {"window_passed": False, "next_step": "repair autoregressive runtime"},
                  "reason": str(error)}
    _write(destination / "release_autoregressive_summary.json", result)
    decision = result["decision"]
    lines = ["# L1 Autoregressive Release Actor", "", f"- status: {result['status']}",
             f"- window: {'PASS' if decision.get('window_passed') else 'FAIL'}"]
    if result["status"] == "completed":
        for name, values in (("A continuous", result["control"]),
                             ("C independent mixed", result["prior_counterfactual_independent_heads"]),
                             ("D autoregressive", result["candidate"])):
            lines.append(f"- {name}: release correction={values['release']['teacher_release_correction_rate']:.4f}, "
                         f"teacher overall/warm={values['teacher_burden_overall']:.3f}/{values['teacher_burden_warm']:.3f}, "
                         f"P95={values['correction_p95']:.3f}, shield={values['shield_burden']:.3f}, "
                         f"completion={values['completion_ratio']:.4f}, throughput={values['throughput_per_second']:.6f}, "
                         f"PPO healthy={values['ppo_health']['healthy']}, safety={values['safety']}")
    lines.append(f"- next: {decision['next_step']}")
    (destination / "release_autoregressive_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run bounded autoregressive release actor diagnostic.")
    parser.add_argument("--c0-report", required=True); parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True); parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--d5a-summary", required=True); parser.add_argument("--d5c-summary", required=True)
    parser.add_argument("--targeted-probe", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--seed-offset", type=int, default=0); parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(run_release_autoregressive_actor(
        c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
        initial_checkpoint=args.initial_checkpoint, baseline_summary=args.baseline_summary,
        d5a_summary=args.d5a_summary, d5c_summary=args.d5c_summary, targeted_probe=args.targeted_probe,
        output_dir=args.output, rollout_workers=args.workers, source_commit=args.source_commit,
        seed_offset=args.seed_offset, updates=args.updates, max_steps=args.max_steps), indent=2))


if __name__ == "__main__":
    _main()
