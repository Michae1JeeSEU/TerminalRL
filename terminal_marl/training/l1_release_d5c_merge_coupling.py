"""D5c release/merge coupling diagnostic with no optimizer update."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from ..algo import RolloutBuffer, SharedPPO, Transition
from ..core.checkpoints import load_checkpoint
from ..core.observations import LOW_OBS_V2
from .l1_ppo_d3_bounded_distribution import _requested_action_provenance
from .l1_release_mixed_actor_diagnostic import _write
from .train_low_v2_ppo import (
    _episode_design, _model_state, _read_frozen_capacity, _read_seed_partition,
    _rollout_task, authorize_low_v2_training, l1_full_teacher_config,
)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    return ({"mean": None, "p10": None, "median": None, "p90": None} if not values.size else
            {"mean": float(values.mean()), "p10": float(np.percentile(values, 10)),
             "median": float(np.median(values)), "p90": float(np.percentile(values, 90))})


def _binned_label_rates(values: np.ndarray, labels: np.ndarray, corrected: np.ndarray) -> list[dict[str, Any]]:
    edges = np.asarray((-1.000001, -.5, 0., .5, 1.000001))
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values < high)
        rows.append({
            "low": float(max(low, -1.0)), "high": float(min(high, 1.0)), "samples": int(mask.sum()),
            "teacher_release_rate": None if not mask.any() else float(labels[mask].mean()),
            "release_correction_rate": None if not mask.any() else float(corrected[mask].mean()),
        })
    return rows


def _nearest_opposite_residual_delta(observations: np.ndarray, actions: np.ndarray,
                                     labels: np.ndarray, *, max_per_label: int = 512) -> dict[str, Any]:
    rng = np.random.default_rng(1709)
    selected = {}
    for label in (False, True):
        candidates = np.flatnonzero(labels == label)
        selected[label] = (candidates if len(candidates) <= max_per_label else
                           np.sort(rng.choice(candidates, max_per_label, replace=False)))
    if not len(selected[False]) or not len(selected[True]):
        return {"available": False}
    wait_obs, release_obs = observations[selected[False]], observations[selected[True]]
    distances = ((release_obs[:, None, :] - wait_obs[None, :, :]) ** 2).mean(axis=2)
    nearest_wait = selected[False][distances.argmin(axis=1)]
    release_rows = selected[True]
    residual_delta = np.abs(actions[release_rows, 1:] - actions[nearest_wait, 1:])
    return {
        "available": True, "pairs": len(release_rows),
        "observation_rms": _quantiles(np.sqrt(distances.min(axis=1))),
        "absolute_speed_residual_delta": _quantiles(residual_delta[:, 0]),
        "absolute_merge_residual_delta": _quantiles(residual_delta[:, 1]),
        "merge_delta_exceeds_speed_fraction": float((residual_delta[:, 1] > residual_delta[:, 0]).mean()),
    }


def analyze_release_merge_coupling(transitions: list[Transition]) -> dict[str, Any]:
    rows = [row for row in transitions if row.teacher_action is not None]
    if not rows:
        raise RuntimeError("D5c requires authoritative release targets")
    observations = np.stack([row.observation for row in rows]).astype(np.float32, copy=False)
    requested = np.stack([row.action for row in rows]).astype(np.float32, copy=False)
    teacher = np.stack([row.teacher_action for row in rows]).astype(np.float32, copy=False)
    release_corrected = (requested[:, 0] > 0.0) != (teacher[:, 0] > 0.0)
    speed_corrected = np.abs(requested[:, 1] - teacher[:, 1]) > 1e-7
    merge_corrected = np.abs(requested[:, 2] - teacher[:, 2]) > 1e-7
    labels = teacher[:, 0] > 0.0
    masks = {
        "none": ~release_corrected & ~speed_corrected & ~merge_corrected,
        "release_only": release_corrected & ~speed_corrected & ~merge_corrected,
        "speed_only": ~release_corrected & speed_corrected & ~merge_corrected,
        "merge_only": ~release_corrected & ~speed_corrected & merge_corrected,
        "release_speed": release_corrected & speed_corrected & ~merge_corrected,
        "release_merge": release_corrected & ~speed_corrected & merge_corrected,
        "speed_merge": ~release_corrected & speed_corrected & merge_corrected,
        "release_speed_merge": release_corrected & speed_corrected & merge_corrected,
    }
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    release_merge = release_corrected & merge_corrected
    release_speed = release_corrected & speed_corrected
    release_rate, merge_rate, speed_rate = release_corrected.mean(), merge_corrected.mean(), speed_corrected.mean()
    release_merge_lift = (None if release_rate == 0.0 or merge_rate == 0.0 else
                          float(release_merge.mean() / (release_rate * merge_rate)))
    release_speed_lift = (None if release_rate == 0.0 or speed_rate == 0.0 else
                          float(release_speed.mean() / (release_rate * speed_rate)))
    merge_bins = _binned_label_rates(requested[:, 2], labels, release_corrected)
    speed_bins = _binned_label_rates(requested[:, 1], labels, release_corrected)
    merge_label_rates = [row["teacher_release_rate"] for row in merge_bins if row["teacher_release_rate"] is not None]
    speed_label_rates = [row["teacher_release_rate"] for row in speed_bins if row["teacher_release_rate"] is not None]
    nearest = _nearest_opposite_residual_delta(observations, requested, labels)
    merge_rate_span = float(max(merge_label_rates) - min(merge_label_rates))
    speed_rate_span = float(max(speed_label_rates) - min(speed_label_rates))
    overlap_given_merge = None if not merge_corrected.any() else float(release_corrected[merge_corrected].mean())
    strong_merge_coupling = bool(
        (release_merge_lift is not None and release_merge_lift >= 1.10) or
        merge_rate_span >= max(.02, 1.25 * speed_rate_span) or
        (nearest.get("available") and nearest["merge_delta_exceeds_speed_fraction"] >= .60)
    )
    return {
        "samples": len(rows), "label_balance": {"WAIT": int((~labels).sum()), "RELEASE": int(labels.sum())},
        "correction_category_counts": counts,
        "correction_rates": {"release": float(release_rate), "speed": float(speed_rate), "merge": float(merge_rate),
                             "release_and_merge": float(release_merge.mean()),
                             "release_and_speed": float(release_speed.mean())},
        "coupling": {"release_merge_lift_over_independence": release_merge_lift,
                     "release_speed_lift_over_independence": release_speed_lift,
                     "release_correction_given_merge_correction": overlap_given_merge,
                     "teacher_release_rate_span_by_merge_bin": merge_rate_span,
                     "teacher_release_rate_span_by_speed_bin": speed_rate_span},
        "teacher_label_by_requested_merge_residual": merge_bins,
        "teacher_label_by_requested_speed_residual": speed_bins,
        "requested_residual_by_teacher_label": {
            "speed": {"WAIT": _quantiles(requested[~labels, 1]), "RELEASE": _quantiles(requested[labels, 1])},
            "merge": {"WAIT": _quantiles(requested[~labels, 2]), "RELEASE": _quantiles(requested[labels, 2])},
        },
        "nearest_opposite_label_residual_delta": nearest,
        "strong_release_merge_coupling": strong_merge_coupling,
        "next_step": ("test an autoregressive residual-conditioned Bernoulli release actor" if strong_merge_coupling else
                      "merge coupling not established; continue objective calibration before optimizer ablation"),
    }


def run_release_d5c_merge_coupling(
    *, c0_report: str | Path, teacher_dataset_path: str | Path,
    initial_checkpoint: str | Path, output_dir: str | Path,
    rollout_workers: int = 4, source_commit: str = "unknown", max_steps: int = 1200,
) -> dict[str, Any]:
    destination = Path(output_dir)
    audit = authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    mu_hat, medium_interval = _read_frozen_capacity(c0_report)
    medium_rate = sum(medium_interval) / 2.0
    checkpoint = Path(initial_checkpoint)
    manifest = {
        "stage": "L1_release_D5c_merge_coupling", "evidence_type": "newly_run",
        "source_commit": source_commit, "optimizer_steps": 0, "max_steps": max_steps,
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "teacher_dataset_audit": asdict(audit), "seeds": list(partition.train[:4]),
        "label_source": "isolated counterfactual all-RELEASE scheduler/safety decision",
        "frozen": ["policy weights", "reward", "BC weight", "traffic", "Observation V2", "safety stack",
                   "EH216-S dynamics", "mu_hat/lambda", "action interface"],
    }
    _write(destination / "release_d5c.manifest.json", manifest)
    _write(destination / "release_d5c.startup_preflight.json", {
        "passed": True, "expected_outputs": ["release_d5c_summary.json", "release_d5c_summary.md"],
        "required_categories": ["release_only", "merge_only", "release_merge", "release_speed_merge"],
    })
    random.seed(1701); np.random.seed(1701); torch.manual_seed(1701)
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05,
                       policy_distribution="mixed_bernoulli_tanh")
    base_config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings",
                                         demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8,
                                         horizon_seconds=float(max_steps))
    load_checkpoint(checkpoint, policy, base_config)
    state, tasks = _model_state(policy), []
    for episode, seed in enumerate(partition.train[:4]):
        config, _ = _episode_design(episode, medium_rate=medium_rate, horizon_seconds=float(max_steps))
        tasks.append((state, config, seed, 0, episode, max_steps, False, None,
                      "mixed_bernoulli_tanh", True, True, True, True))
    if rollout_workers == 1:
        collected = [_rollout_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(rollout_workers, len(tasks))) as executor:
            collected = list(executor.map(_rollout_task, tasks))
    transitions = [row for episode_rows, _ in collected for row in episode_rows]
    summaries = [summary for _, summary in collected]
    buffer = RolloutBuffer()
    for transition in transitions:
        buffer.add(transition)
    provenance = _requested_action_provenance(policy, buffer)
    safety = {name: int(sum(getattr(summary, name) for summary in summaries)) for name in
              ("collisions", "nmac", "hard_separation_violations")}
    analysis = analyze_release_merge_coupling(transitions)
    result = {"status": "completed", "manifest": manifest, "analysis": analysis,
              "provenance": provenance, "safety": safety,
              "diagnostic_integrity_passed": bool(provenance["passed"] and not any(safety.values()))}
    _write(destination / "release_d5c_summary.json", result)
    lines = ["# L1 PPO D5c Release/Merge Coupling", "", f"- samples: {analysis['samples']}",
             f"- correction categories: {analysis['correction_category_counts']}",
             f"- coupling: {analysis['coupling']}",
             f"- strong release/merge coupling: {analysis['strong_release_merge_coupling']}",
             f"- provenance: {'PASS' if provenance['passed'] else 'FAIL'}", f"- safety: {safety}",
             f"- next: {analysis['next_step']}"]
    (destination / "release_d5c_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run no-update release/merge coupling diagnostic.")
    parser.add_argument("--c0-report", required=True); parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(run_release_d5c_merge_coupling(
        c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
        initial_checkpoint=args.initial_checkpoint, output_dir=args.output,
        rollout_workers=args.workers, source_commit=args.source_commit, max_steps=args.max_steps), indent=2))


if __name__ == "__main__":
    _main()
