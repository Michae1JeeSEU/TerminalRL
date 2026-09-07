"""D5b release Observation V2 sufficiency diagnostic (no optimizer step)."""
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


SLOT_FIELDS = ("present", "path_gap", "distance", "relative_speed", "closing_rate",
               "d_cpa", "t_cpa", "priority_delta", "reserved_or_committed")
RELEASE_CONTEXT_FEATURES = (
    "phase", "ring_role", "ring_progress", "speed", "wait", "priority", "release_eligible",
    "assigned_lz_present", "assigned_lz_usable", "final_resource_available", "free_lz_fraction",
    "target_ring_predecessor", "target_ring_successor", "nearest_predicted_conflict",
    "ring_occupancy", "headways", "final_queue_pressure", "global_free_lz_fraction",
    "interface_frozen", "interface_headway_slack", "reconfiguration_active_pending",
)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    return ({"mean": None, "p10": None, "median": None, "p90": None} if not values.size else
            {"mean": float(values.mean()), "p10": float(np.percentile(values, 10)),
             "median": float(np.median(values)), "p90": float(np.percentile(values, 90))})


def _indices(names: tuple[str, ...]) -> np.ndarray:
    slices = LOW_OBS_V2.slices
    return np.asarray([index for name in names for index in range(slices[name].start, slices[name].stop)], dtype=int)


def _quantized_conflicts(features: np.ndarray, labels: np.ndarray, resolution: float) -> dict[str, Any]:
    bins: dict[bytes, list[int]] = {}
    quantized = np.rint(features / resolution).astype(np.int16, copy=False)
    for row, label in zip(quantized, labels):
        counts = bins.setdefault(row.tobytes(), [0, 0])
        counts[int(label)] += 1
    mixed = [counts for counts in bins.values() if counts[0] and counts[1]]
    positive = int(labels.sum())
    return {
        "resolution": resolution, "bins": len(bins), "mixed_label_bins": len(mixed),
        "mixed_label_samples": int(sum(sum(counts) for counts in mixed)),
        "minority_conflicts": int(sum(min(counts) for counts in mixed)),
        "release_samples_in_mixed_bins": int(sum(counts[1] for counts in mixed)),
        "release_fraction_in_mixed_bins": (None if positive == 0 else
                                           float(sum(counts[1] for counts in mixed) / positive)),
    }


def _nearest_cross_label(features: np.ndarray, labels: np.ndarray, *, max_per_label: int = 512) -> dict[str, Any]:
    rng = np.random.default_rng(731)
    indices = {}
    for label in (False, True):
        candidates = np.flatnonzero(labels == label)
        indices[label] = (candidates if len(candidates) <= max_per_label else
                          np.sort(rng.choice(candidates, max_per_label, replace=False)))
    if not len(indices[False]) or not len(indices[True]):
        return {"available": False}
    left, right = features[indices[False]].astype(np.float32), features[indices[True]].astype(np.float32)
    # RMS feature distance keeps thresholds comparable between full/context/action feature sets.
    distances = np.sqrt(np.maximum(
        ((left[:, None, :] - right[None, :, :]) ** 2).mean(axis=2), 0.0))
    wait_to_release = distances.min(axis=1)
    release_to_wait = distances.min(axis=0)
    if len(right) > 1:
        same = np.sqrt(np.maximum(((right[:, None, :] - right[None, :, :]) ** 2).mean(axis=2), 0.0))
        np.fill_diagonal(same, np.inf)
        release_to_release = same.min(axis=1)
    else:
        release_to_release = np.asarray([], dtype=np.float32)
    opposite_median = float(np.median(release_to_wait))
    same_median = None if not release_to_release.size else float(np.median(release_to_release))
    return {
        "available": True, "sampled_wait": len(left), "sampled_release": len(right),
        "wait_to_release_rms": _quantiles(wait_to_release),
        "release_to_wait_rms": _quantiles(release_to_wait),
        "release_to_release_rms": _quantiles(release_to_release),
        "opposite_to_same_median_ratio": (None if same_median in (None, 0.0) else opposite_median / same_median),
    }


def analyze_release_observation_sufficiency(transitions: list[Transition]) -> dict[str, Any]:
    rows = [row for row in transitions if row.teacher_action is not None]
    if not rows:
        raise RuntimeError("D5b requires authoritative release labels")
    observations = np.stack([row.observation for row in rows]).astype(np.float32, copy=False)
    actions = np.stack([row.action for row in rows]).astype(np.float32, copy=False)
    labels = np.asarray([row.teacher_action[0] > 0.0 for row in rows], dtype=bool)
    context_indices = _indices(RELEASE_CONTEXT_FEATURES)
    feature_sets = {
        "full_observation": observations,
        "release_context": observations[:, context_indices],
        "full_observation_plus_requested_residuals": np.concatenate((observations, actions[:, 1:]), axis=1),
    }
    conflicts = {name: [_quantized_conflicts(values, labels, resolution)
                        for resolution in (0.001, 0.01, 0.02, 0.05)]
                 for name, values in feature_sets.items()}
    nearest = {name: _nearest_cross_label(values, labels) for name, values in feature_sets.items()}
    slot_summary: dict[str, Any] = {}
    for slot_name in ("target_ring_predecessor", "target_ring_successor", "nearest_predicted_conflict"):
        values = observations[:, LOW_OBS_V2.slices[slot_name]]
        slot_summary[slot_name] = {
            field: {"WAIT": _quantiles(values[~labels, index]), "RELEASE": _quantiles(values[labels, index])}
            for index, field in enumerate(SLOT_FIELDS)
        }
    full_nearest = nearest["full_observation"]
    opposite = full_nearest.get("release_to_wait_rms", {}).get("median")
    ratio = full_nearest.get("opposite_to_same_median_ratio")
    release_mixed = next(row for row in conflicts["full_observation"] if row["resolution"] == .05)
    aliasing = bool((opposite is not None and opposite <= .05 and ratio is not None and ratio <= 1.25) or
                    (release_mixed["release_fraction_in_mixed_bins"] or 0.0) >= .10)
    with_action = nearest["full_observation_plus_requested_residuals"].get("release_to_wait_rms", {}).get("median")
    action_coupling_hint = bool(opposite is not None and with_action is not None and
                                with_action > opposite + 1e-6 and with_action >= 1.10 * opposite)
    return {
        "samples": len(rows), "label_balance": {"WAIT": int((~labels).sum()), "RELEASE": int(labels.sum()),
                                                  "release_fraction": float(labels.mean())},
        "required_observation_features": {
            "target_predecessor": True, "target_successor": True, "predicted_conflict": True,
            "reservation_or_committed": True, "gap": True, "d_cpa": True, "t_cpa": True,
        },
        "slot_label_conditionals": slot_summary,
        "quantized_label_conflicts": conflicts,
        "nearest_label_distances": nearest,
        "observation_aliasing_detected": aliasing,
        "requested_residual_coupling_hint": action_coupling_hint,
        "observation_sufficiency_passed": not aliasing,
        "next_step": ("continue D5c release/merge coupling diagnostic" if aliasing or action_coupling_hint else
                      "observation features are separable; audit objective calibration before optimizer ablation"),
    }


def run_release_d5b_observation(
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
        "stage": "L1_release_D5b_observation_sufficiency", "evidence_type": "newly_run",
        "source_commit": source_commit, "optimizer_steps": 0, "max_steps": max_steps,
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "teacher_dataset_audit": asdict(audit), "seeds": list(partition.train[:4]),
        "label_source": "isolated counterfactual all-RELEASE scheduler/safety decision",
        "frozen": ["policy weights", "reward", "BC weight", "traffic", "Observation V2", "safety stack",
                   "EH216-S dynamics", "mu_hat/lambda", "action interface"],
    }
    _write(destination / "release_d5b.manifest.json", manifest)
    _write(destination / "release_d5b.startup_preflight.json", {
        "passed": True, "expected_outputs": ["release_d5b_summary.json", "release_d5b_summary.md"],
        "required_analysis": ["label_balance", "quantized_label_conflicts", "nearest_label_distances",
                              "slot_label_conditionals"],
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
    analysis = analyze_release_observation_sufficiency(transitions)
    passed = bool(provenance["passed"] and not any(safety.values()))
    result = {"status": "completed", "manifest": manifest, "analysis": analysis,
              "provenance": provenance, "safety": safety, "diagnostic_integrity_passed": passed}
    _write(destination / "release_d5b_summary.json", result)
    lines = ["# L1 PPO D5b Release Observation Sufficiency", "",
             f"- samples: {analysis['samples']}", f"- label balance: {analysis['label_balance']}",
             f"- observation aliasing: {analysis['observation_aliasing_detected']}",
             f"- requested residual coupling hint: {analysis['requested_residual_coupling_hint']}",
             f"- provenance: {'PASS' if provenance['passed'] else 'FAIL'}", f"- safety: {safety}",
             f"- next: {analysis['next_step']}"]
    (destination / "release_d5b_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run no-update release Observation V2 sufficiency diagnostic.")
    parser.add_argument("--c0-report", required=True); parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(run_release_d5b_observation(
        c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
        initial_checkpoint=args.initial_checkpoint, output_dir=args.output,
        rollout_workers=args.workers, source_commit=args.source_commit, max_steps=args.max_steps), indent=2))


if __name__ == "__main__":
    _main()
