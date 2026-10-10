"""Run a bounded task-neutral PPO learning curve profile."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import run_p3_task_neutral_joint_batch as joint
from terminal_marl.training import run_p3_task_neutral_pilot as pilot


STAGE = pilot.STAGE
REPAIR_METHOD = "joint_team_learning_curve"
EPISODIC_REPAIR_METHOD = "episodic_joint_team_learning_curve"
EVALUATION_CELLS = tuple(dict(cell) for cell in joint.EVALUATION_CELLS[:4])
BASELINE_RECORD_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_joint_team_actor_20260928/full/evaluation/records/before"
)
UPDATE_KWARGS = {**pilot.UPDATE_KWARGS, "actor_reduction": "joint_team"}
LARGER_WARMUP_SHA256 = "a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710"


@dataclasses.dataclass(frozen=True)
class LearningCurveProfile:
    name: str
    batch_count: int
    episodes_per_batch: int
    permutation_seeds: tuple[int, ...]
    training_env_seeds: tuple[int, ...]
    training_policy_seeds: tuple[int, ...]
    evaluation_batches: tuple[int, ...]
    minibatch_size: int
    required_episodic_return: bool = False
    required_reward_scale: float | None = None
    required_warmup_sha256: str | None = None

    @property
    def training_rollouts(self) -> int:
        return self.batch_count * self.episodes_per_batch

    @property
    def evaluation_rollouts(self) -> int:
        return len(self.evaluation_batches) * len(EVALUATION_CELLS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "batch_count": self.batch_count,
            "episodes_per_batch": self.episodes_per_batch,
            "permutation_seeds": list(self.permutation_seeds),
            "training_env_seeds": list(self.training_env_seeds),
            "training_policy_seeds": list(self.training_policy_seeds),
            "evaluation_batches": list(self.evaluation_batches),
            "minibatch_size": self.minibatch_size,
            "required_episodic_return": self.required_episodic_return,
            "required_reward_scale": self.required_reward_scale,
            "required_warmup_sha256": self.required_warmup_sha256,
        }


DEFAULT_PROFILE = LearningCurveProfile(
    name="default",
    batch_count=6,
    episodes_per_batch=6,
    permutation_seeds=tuple(range(878001, 878007)),
    training_env_seeds=tuple(range(877101, 877137)),
    training_policy_seeds=tuple(range(878201, 878237)),
    evaluation_batches=(2, 4, 6),
    minibatch_size=32,
)
LARGER_BATCH_PROFILE = LearningCurveProfile(
    name="larger_batch",
    batch_count=4,
    episodes_per_batch=24,
    permutation_seeds=tuple(range(879001, 879005)),
    training_env_seeds=tuple(range(879101, 879197)),
    training_policy_seeds=tuple(range(879201, 879297)),
    evaluation_batches=(2, 4),
    minibatch_size=128,
    required_episodic_return=True,
    required_reward_scale=0.01,
    required_warmup_sha256=LARGER_WARMUP_SHA256,
)


def _profile(*, larger_batch: bool = False) -> LearningCurveProfile:
    return LARGER_BATCH_PROFILE if larger_batch else DEFAULT_PROFILE


# Compatibility aliases for callers that imported the historical default
# constants.  Runtime selection uses an explicit profile object below.
BATCH_COUNT = DEFAULT_PROFILE.batch_count
PERMUTATION_SEEDS = DEFAULT_PROFILE.permutation_seeds
TRAINING_ENV_SEEDS = DEFAULT_PROFILE.training_env_seeds
TRAINING_POLICY_SEEDS = DEFAULT_PROFILE.training_policy_seeds
EVALUATION_BATCHES = DEFAULT_PROFILE.evaluation_batches


class LearningCurveError(ValueError):
    """Raised when the fixed learning-curve contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LearningCurveError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    joint._write_json(path, value)


def _repair_method(*, episodic_return: bool) -> str:
    return EPISODIC_REPAIR_METHOD if episodic_return else REPAIR_METHOD


def _update_kwargs(*, episodic_return: bool, profile: LearningCurveProfile | None = None) -> dict[str, Any]:
    """Build per-run PPO kwargs without mutating the default contract."""

    value = dict(UPDATE_KWARGS)
    if episodic_return:
        value.update({"gamma": 1.0, "gae_lambda": 1.0})
    if profile is not None:
        value["minibatch_size"] = int(profile.minibatch_size)
    return value


def _validate_reward_scale(reward_scale: float, *, episodic_return: bool) -> float:
    value = float(reward_scale)
    _require(np.isfinite(value) and value > 0.0,
             "reward_scale must be positive and finite")
    _require(episodic_return or value == 1.0,
             "non-unit reward_scale requires --episodic-return")
    return value


def _scaled_training_steps(
    steps: Sequence[Any], reward_scale: float,
) -> list[Any]:
    """Copy rollout steps for learning while preserving raw rollout data."""

    scale = float(reward_scale)
    _require(np.isfinite(scale) and scale > 0.0,
             "reward_scale must be positive and finite")
    return [dataclasses.replace(step, reward=float(step.reward) * scale) for step in steps]


def _raw_reward_totals(
    steps: Sequence[Any], records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Read raw environment totals without changing rollout records."""

    step_total = float(sum(float(step.reward) for step in steps))
    record_totals: list[float] = []
    for record in records:
        summary = record.get("summary") or record.get("rollout_summary")
        value: Any = summary.get("reward_total") if isinstance(summary, Mapping) else None
        if value is None:
            value = record.get("reward_total")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            record_totals.append(numeric)
    if len(record_totals) == len(records) and records:
        raw_total = float(sum(record_totals))
        source = "records"
    else:
        raw_total = step_total
        source = "steps"
    _require(np.isclose(raw_total, step_total, rtol=1e-5, atol=1e-4),
             "raw reward totals disagree with collected StudentStep rewards")
    return {
        "raw_environment_reward_total": raw_total,
        "raw_step_reward_total": step_total,
        "raw_record_reward_total": (
            float(sum(record_totals)) if len(record_totals) == len(records) and records else None
        ),
        "raw_reward_total_source": source,
    }


def _reward_scale_diagnostics(
    steps: Sequence[Any], records: Sequence[Mapping[str, Any]], reward_scale: float,
) -> dict[str, Any]:
    raw = _raw_reward_totals(steps, records)
    scale = float(reward_scale)
    scaled_total = float(sum(float(step.reward) * scale for step in steps))
    expected_scaled = float(raw["raw_environment_reward_total"]) * scale
    _require(np.isclose(scaled_total, expected_scaled, rtol=1e-5, atol=1e-4),
             "scaled reward total disagrees with raw reward total")
    raw.update({
        "reward_scale": scale,
        "learning_reward_total": scaled_total,
        "expected_learning_reward_total": expected_scaled,
        "reward_scale_consistent": True,
    })
    return raw


def _episodic_return_diagnostics(
    steps: Sequence[Any], update_kwargs: Mapping[str, Any],
    *, reward_scale: float = 1.0,
) -> dict[str, Any]:
    """Check finite-horizon reward suffix returns before a PPO update."""

    gamma = float(update_kwargs["gamma"])
    gae_lambda = float(update_kwargs["gae_lambda"])
    _require(gamma == 1.0 and gae_lambda == 1.0,
             "episodic return mode requires gamma=1.0 and gae_lambda=1.0")
    _require(bool(steps), "episodic return batch is empty")
    _, returns = joint.team_gae(list(steps), gamma, gae_lambda)
    expected = joint._full_episode_mc_returns(steps)
    max_abs_error = float(np.max(np.abs(
        np.asarray(returns, dtype=np.float32) - np.asarray(expected, dtype=np.float32)
    )))
    match = bool(np.allclose(returns, expected, rtol=1e-5, atol=1e-4))
    return {
        "credit_method": "full_episode_mc",
        "credit_gamma": gamma,
        "credit_gae_lambda": gae_lambda,
        "reward_scale": float(reward_scale),
        "learning_reward_total": float(sum(float(row.reward) for row in steps)),
        "returns": joint._stats(returns),
        "full_episode_mc_returns": joint._stats(expected),
        "full_episode_mc_return_max_abs_error": max_abs_error,
        "full_episode_mc_return_match": match,
    }


def training_batches(profile: LearningCurveProfile | None = None) -> tuple[tuple[dict[str, Any], ...], ...]:
    selected = DEFAULT_PROFILE if profile is None else profile
    batches: list[tuple[dict[str, Any], ...]] = []
    for batch_index in range(selected.batch_count):
        start = batch_index * selected.episodes_per_batch
        batch = tuple(
            {
                "scenario": "dense24",
                "seed": int(selected.training_env_seeds[start + offset]),
                "policy_seed": int(selected.training_policy_seeds[start + offset]),
                "split": f"learning_curve_batch_{batch_index + 1:02d}",
            }
            for offset in range(selected.episodes_per_batch)
        )
        batches.append(batch)
    return tuple(batches)


def _training_cells(profile: LearningCurveProfile | None = None) -> tuple[dict[str, Any], ...]:
    selected = DEFAULT_PROFILE if profile is None else profile
    cells = tuple(cell for batch in training_batches(selected) for cell in batch)
    keys = {(str(cell["scenario"]), int(cell["seed"]), int(cell["policy_seed"])) for cell in cells}
    expected = selected.training_rollouts
    _require(len(cells) == expected and len(keys) == expected, "learning-curve training cells are not unique")
    _require(len({int(cell["seed"]) for cell in cells}) == expected, "learning-curve environment seeds are not unique")
    _require(len({int(cell["policy_seed"]) for cell in cells}) == expected, "learning-curve policy seeds are not unique")
    return cells


def _progress(
    output: Path, *, status: str, phase: str, batches: int = 0,
    updates: int = 0, evaluations: int = 0, check_only: bool = False,
    error: str | None = None, repair_method: str = REPAIR_METHOD,
    profile: LearningCurveProfile | None = None,
) -> None:
    selected = DEFAULT_PROFILE if profile is None else profile
    value: dict[str, Any] = {
        "stage": STAGE,
        "repair_method": str(repair_method),
        "profile": selected.name,
        "profile_spec": selected.as_dict(),
        "status": str(status),
        "phase": str(phase),
        "batches_processed": int(batches),
        "batches_expected": 0 if check_only else selected.batch_count,
        "updates_processed": int(updates),
        "updates_expected": 0 if check_only else selected.batch_count,
        "evaluation_processed": int(evaluations),
        "evaluation_expected": 0 if check_only else selected.evaluation_rollouts,
        "training_rollouts_expected": selected.training_rollouts,
        "evaluation_rollouts_expected": selected.evaluation_rollouts,
        "minibatch_size": selected.minibatch_size,
        "check_only": bool(check_only),
        "PPO_ALLOWED": False,
        "teacher_called": False,
        "heldout_consumed": False,
    }
    if error:
        value["error"] = str(error)
    _write_json(output / "progress.json", value)


def _base_report(
    output: Path, *, check_only: bool, episodic_return: bool = False,
    update_kwargs: Mapping[str, Any] | None = None,
    initial_warmup_checkpoint: str | Path | None = None,
    reward_scale: float = 1.0,
    profile: LearningCurveProfile | None = None,
) -> dict[str, Any]:
    selected = DEFAULT_PROFILE if profile is None else profile
    kwargs = dict(UPDATE_KWARGS) if update_kwargs is None else dict(update_kwargs)
    repair_method = _repair_method(episodic_return=episodic_return)
    return {
        "stage": STAGE,
        "repair_method": repair_method,
        "profile": selected.name,
        "profile_spec": selected.as_dict(),
        "status": "STARTING",
        "check_only": bool(check_only),
        "full_evaluation": False,
        "PPO_ALLOWED": False,
        "teacher_called": False,
        "heldout_consumed": False,
        "episodic_return": bool(episodic_return),
        "reward_scale": float(reward_scale),
        "reward_unit": "critic_learning_reward",
        "actor_reduction": "joint_team",
        "credit_gamma": float(kwargs["gamma"]),
        "credit_gae_lambda": float(kwargs["gae_lambda"]),
        "update_kwargs": dict(kwargs),
        "initial_warmup_checkpoint": (
            str(initial_warmup_checkpoint)
            if initial_warmup_checkpoint is not None else None
        ),
        "training_rollouts": 0,
        "training_steps": 0,
        "ppo_updates": 0,
        "evaluation_rollouts": 0,
        "training": {"batches": [], "batches_processed": 0, "updates": 0,
                      "update_calls": 0},
        "evaluation": {
            "baseline_reused_records": 0, "new_rollouts": 0, "snapshots": [],
        },
        "artifacts": {"output_dir": str(output), "checkpoints": []},
        "checks": {},
        "deviations": [],
        "unresolved": [],
    }


def _metric_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Reuse the registered all-row aggregator; the joint-batch module does
    # not expose a private summary helper.  This also retains partial-record
    # counts for the four-record reused evaluation baseline.
    return neutral_eval._aggregate_arm(records)


def _baseline_records(config: Any, *, output: Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    paths: list[str] = []
    seen: set[int] = set()
    for cell in EVALUATION_CELLS:
        env_seed = int(cell["seed"])
        policy_seed = int(cell["policy_seed"])
        _require(env_seed not in seen, f"duplicate baseline evaluation seed: {env_seed}")
        seen.add(env_seed)
        declared = BASELINE_RECORD_ROOT / f"dense24_{env_seed}.json"
        source = neutral_eval._synced_path(declared)
        _require(source.is_file(), f"baseline evaluation record missing: {declared}")
        record = neutral_eval._read_json(source)
        _require(record.get("arm") == "before", f"baseline evaluation arm mismatch: {source}")
        _require(str(record.get("scenario")) == "dense24" and int(record.get("seed")) == env_seed,
                 f"baseline evaluation identity mismatch: {source}")
        _require(int(record.get("policy_seed")) == policy_seed,
                 f"baseline evaluation policy seed mismatch: {source}")
        _require(record.get("checkpoint_sha256") == pilot.EXPECTED_CANDIDATE_SHA256,
                 f"baseline evaluation checkpoint mismatch: {source}")
        _require(record.get("status") == "completed" and record.get("lifecycle_valid") is True,
                 f"baseline evaluation lifecycle mismatch: {source}")
        _require(record.get("technical_ok") is True and not record.get("error"),
                 f"baseline evaluation technical state mismatch: {source}")
        _require(record.get("weights_unchanged") is True and record.get("model_unchanged") is True,
                 f"baseline evaluation model state mismatch: {source}")
        _require(record.get("task_priority_enabled") is False
                 and float(record.get("low_level_priority_time_weight", 1.0)) == 0.0,
                 f"baseline evaluation neutral profile mismatch: {source}")
        _require(record.get("training_or_update_called") is False
                 and record.get("teacher_called") is False,
                 f"baseline evaluation training state mismatch: {source}")
        _require(record.get("config") == neutral_eval._config_snapshot(config),
                 f"baseline evaluation config mismatch: {source}")
        _require(neutral_eval._record_fatal_reason(record, smoke=False) is None,
                 f"baseline evaluation fatal state: {source}")
        destination = (
            output / "evaluation" / "before" / f"dense24_{env_seed}.json"
            if output is not None else source
        )
        copied = dict(record)
        copied.update({
            "arm": "before", "scenario": "dense24", "seed": env_seed,
            "env_seed": env_seed, "policy_seed": policy_seed,
            "record_path": str(destination), "source_record_path": str(source),
            "source_provenance": {"kind": "reused_learning_curve_baseline", "source_path": str(source)},
        })
        records.append(copied)
        paths.append(str(source))
        if output is not None:
            _write_json(destination, copied)
    _require(len(records) == 4, "learning-curve baseline must contain four records")
    return records, paths


def _paired_metrics(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return joint._paired_metrics(before, after)


def _evaluation_pairs(
    baseline: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    left = {(int(row["seed"]), str(row["scenario"])): row for row in baseline}
    right = {(int(row["seed"]), str(row["scenario"])): row for row in after}
    rows: list[dict[str, Any]] = []
    for cell in EVALUATION_CELLS:
        key = (int(cell["seed"]), str(cell["scenario"]))
        _require(key in left and key in right, f"missing learning-curve evaluation pair: {key}")
        rows.append({
            "cell": dict(cell), "seed": key[0], "scenario": key[1],
            "metrics": _paired_metrics(left[key], right[key]),
        })
    return rows


def _episodic_preflight(
    neutral_configs: Mapping[str, Any],
    initial_warmup_checkpoint: str | Path,
    reward_scale: float,
) -> tuple[Path, Any, dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    """Bind the fixed candidate/q pair and a caller-supplied gamma=1 warmup."""

    dense = neutral_configs["dense24"]
    resolved, policy, binding = pilot._load_binding(dense)
    source_policy, source_payload = rank_eval._load_policy(resolved, dense)
    source_q_digest = pilot._model_digest(source_policy)
    warmup_path, warmup_binding = pilot._load_initial_warmup(
        initial_warmup_checkpoint,
        policy,
        dense,
        expected_gamma=1.0,
        expected_reward_scale=float(reward_scale),
    )
    warmup_path = Path(warmup_path)
    _require(warmup_path.is_file(), f"episodic warmup checkpoint missing: {warmup_path}")
    warmup_sha = str(
        warmup_binding.get("checkpoint_sha256")
        or warmup_binding.get("warmup_sha256")
        or neutral_eval._sha256(warmup_path)
    )
    _require(warmup_sha == neutral_eval._sha256(warmup_path),
             "episodic warmup checkpoint SHA binding mismatch")
    warmup = dict(warmup_binding)
    warmup.update({
        "warmup_path": str(warmup_path),
        "warmup_sha256": warmup_sha,
        "expected_gamma": 1.0,
        "expected_reward_scale": float(reward_scale),
    })
    return resolved, policy, binding, source_policy, {
        "source_payload_keys": sorted(source_payload) if isinstance(source_payload, Mapping) else [],
        "source_q_digest": source_q_digest,
        **warmup,
    }, {"source_q_digest": source_q_digest}


def _run_evaluation(
    *, output: Path, batch_index: int, checkpoint: Path, config: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    records: list[dict[str, Any]] = []
    for cell in EVALUATION_CELLS:
        policy, payload = rank_eval._load_policy(checkpoint, config)
        pilot._validate_pilot_payload(payload, config)
        record = rank_eval._runtime_record(
            policy=policy,
            checkpoint=checkpoint,
            config=config,
            scenario="dense24",
            seed=int(cell["seed"]),
            policy_seed=int(cell["policy_seed"]),
            output=output / "evaluation" / f"after_batch_{batch_index:02d}" / "rollouts" / f"dense24_{cell['seed']}",
            smoke=False,
        )
        record = pilot._record_annotations(dict(record), config)
        record.update({
            "arm": "after", "cell": dict(cell), "evaluation_batch": int(batch_index),
            "evaluation_kind": "development_existing_scenes",
        })
        path = output / "evaluation" / f"after_batch_{batch_index:02d}" / "records" / f"dense24_{cell['seed']}.json"
        record["record_path"] = str(path)
        _write_json(path, record)
        records.append(record)
        reason = neutral_eval._record_fatal_reason(record, smoke=False)
        if reason is not None:
            return records, f"evaluation batch {batch_index} dense24/{cell['seed']}: {reason}"
    return records, None


def _batch_summary(
    *, batch_index: int, cells: Sequence[Mapping[str, Any]], steps: Sequence[Any],
    records: Sequence[Mapping[str, Any]], stats: Mapping[str, Any],
    freeze: Mapping[str, Any], collection: Mapping[str, Any],
    anchor_steps: int, anchor_observations: int, anchor_weight_sum: float,
    optimizer_identity_unchanged: bool, checkpoint: Mapping[str, Any] | None,
    credit: Mapping[str, Any] | None = None,
    reward_units: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _metric_summary(records)
    actor_decisions = int(sum(int(row.actor_mask.sum()) for row in steps))
    normalizer = float(actor_decisions / len(steps)) if steps else 0.0
    value = {
        "batch_index": int(batch_index),
        "cells": [dict(cell) for cell in cells],
        "rollouts": len(records),
        "steps": len(steps),
        "requests": metrics.get("requests"),
        "normal": metrics.get("normal"),
        "emergency": metrics.get("emergency"),
        "failed": metrics.get("failed"),
        "failed_energy": sum(int(neutral_eval._record_metrics(record).get("failed_energy", 0) or 0)
                             for record in records),
        "mean_wait": metrics.get("mean_wait"),
        "p95_wait": metrics.get("p95_wait"),
        "max_wait": metrics.get("max_wait"),
        "lowSOC_wait": metrics.get("lowSOC_wait"),
        "reward_total": metrics.get("reward_total"),
        "reward_scale": float(reward_units.get("reward_scale", 1.0)) if reward_units else 1.0,
        "reward_unit": "critic_learning_reward",
        "safety_nonzero_cells": metrics.get("safety_nonzero_cells"),
        "technical_ok": metrics.get("technical_ok"),
        "actor_decisions": actor_decisions,
        "actor_count_normalizer": normalizer,
        "anchor_steps": int(anchor_steps),
        "anchor_observation_count": int(anchor_observations),
        "anchor_weight_sum": float(anchor_weight_sum),
        "collection": dict(collection),
        "update_stats": dict(stats),
        "accepted_updates": int(stats.get("accepted_updates", stats.get("updates", 0)) or 0),
        "rejected_updates": int(stats.get("rejected_updates", 0) or 0),
        "approx_kl": stats.get("approx_kl"),
        "final_rollout_kl": stats.get("final_rollout_kl"),
        "frozen_bitwise": dict(freeze),
        "optimizer_identity_unchanged": bool(optimizer_identity_unchanged),
        "record_paths": [str(record.get("record_path")) for record in records],
    }
    if checkpoint is not None:
        value["checkpoint_roundtrip"] = dict(checkpoint)
    if credit is not None:
        value.update(dict(credit))
    if reward_units is not None:
        value.update(dict(reward_units))
    return value


def run(
    *, output_dir: str | Path, check_only: bool = False,
    episodic_return: bool = False,
    initial_warmup_checkpoint: str | Path | None = None,
    reward_scale: float = 1.0,
    larger_batch: bool = False,
    profile_override: LearningCurveProfile | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"learning-curve output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    episodic_return = bool(episodic_return)
    larger_batch = bool(larger_batch)
    if profile_override is not None:
        _require(isinstance(profile_override, LearningCurveProfile),
                 "profile_override must be a LearningCurveProfile")
        _require(not larger_batch, "profile_override cannot be combined with --larger-batch")
        profile = profile_override
    else:
        profile = _profile(larger_batch=larger_batch)
    reward_scale = float(reward_scale)
    # Keep invalid values inside the report-producing failure path below.
    repair_method = _repair_method(episodic_return=episodic_return)
    update_kwargs = _update_kwargs(episodic_return=episodic_return, profile=profile)
    report = _base_report(
        output,
        check_only=check_only,
        episodic_return=episodic_return,
        update_kwargs=update_kwargs,
        initial_warmup_checkpoint=initial_warmup_checkpoint,
        reward_scale=reward_scale,
        profile=profile,
    )
    _progress(
        output, status="STARTING", phase="preflight", check_only=check_only,
        repair_method=repair_method,
        profile=profile,
    )
    cells = _training_cells(profile)
    batches = training_batches(profile)
    try:
        _require(
            not profile.required_episodic_return or episodic_return,
            "--larger-batch requires --episodic-return",
        )
        _require(
            episodic_return or initial_warmup_checkpoint is None,
            "initial_warmup_checkpoint override requires --episodic-return",
        )
        _require(
            not episodic_return or initial_warmup_checkpoint is not None,
            "--episodic-return requires --initial-warmup-checkpoint",
        )
        reward_scale = _validate_reward_scale(
            reward_scale, episodic_return=episodic_return,
        )
        if profile.required_reward_scale is not None:
            _require(
                math.isclose(reward_scale, profile.required_reward_scale, rel_tol=0.0, abs_tol=1e-12),
                "--larger-batch requires --reward-scale 0.01",
            )
        configs, neutral_configs, _ = neutral_eval._build_neutral_inputs()
        del configs
        config = neutral_configs["dense24"]
        _require(getattr(config, "task_priority_enabled") is False,
                 "task-neutral flag is not false")
        _require(float(getattr(config, "low_level_priority_time_weight")) == 0.0,
                 "task-neutral priority reward weight is not zero")
        horizon_steps = int(round(float(config.max_simulation_seconds) / float(config.dt)))
        _require(horizon_steps == 1800, f"learning-curve horizon must be 1800, got {horizon_steps}")
        if episodic_return:
            resolved, policy, binding, source_policy, warmup, source = _episodic_preflight(
                neutral_configs, initial_warmup_checkpoint, reward_scale,
            )
        else:
            resolved, policy, binding, source_policy, warmup, source = joint._preflight(
                neutral_configs,
            )
        if profile.required_warmup_sha256 is not None:
            _require(
                str(warmup.get("warmup_sha256", "")).lower() == profile.required_warmup_sha256,
                "--larger-batch requires the validated warmup_scaled checkpoint",
            )
        baseline_records, baseline_paths = _baseline_records(config)
        report["binding"] = {
            "profile": profile.as_dict(),
            "candidate_checkpoint_sha256": binding.get("checkpoint_sha256"),
            "candidate_checkpoint_sha256_expected": pilot.EXPECTED_CANDIDATE_SHA256,
            "source_q_model_digest": source.get("source_q_digest"),
            "warmup": dict(warmup),
            "neutral_config": neutral_eval._config_snapshot(config),
            "training_batches": [[dict(cell) for cell in batch] for batch in batches],
            "evaluation_cells": [dict(cell) for cell in EVALUATION_CELLS],
            "update_kwargs": dict(update_kwargs),
            "episodic_return": episodic_return,
            "reward_scale": reward_scale,
            "reward_unit": "critic_learning_reward",
            "repair_method": repair_method,
            "permutation_seeds": list(profile.permutation_seeds),
            "horizon_steps_per_rollout": horizon_steps,
            "training_rollouts_expected": profile.training_rollouts,
            "training_requests_expected": profile.training_rollouts * 24,
            "training_max_steps": len(cells) * horizon_steps,
            "evaluation_baseline_reuse_expected": 4,
            "evaluation_new_rollouts_expected": profile.evaluation_rollouts,
            "evaluation_requests_expected": profile.evaluation_rollouts * 24,
            "evaluation_new_max_steps": profile.evaluation_rollouts * horizon_steps,
            "baseline_record_paths": list(baseline_paths),
            "reuse_baseline_root": str(BASELINE_RECORD_ROOT),
            "source_q_fixed": True,
            "q_anchor_sampling": {
                "method": "pilot.actor_world_observation_weights",
                "scope": "cumulative_steps",
                "fixed_upper_limit": None,
            },
            "initial_warmup_checkpoint": str(initial_warmup_checkpoint)
            if initial_warmup_checkpoint is not None else None,
        }
        report["checks"].update({
            "profile": profile.name,
            "training_cells_unique": len(cells) == profile.training_rollouts,
            "training_batches_expected": profile.batch_count,
            "updates_expected": profile.batch_count,
            "episodic_return": episodic_return,
            "credit_gamma": float(update_kwargs["gamma"]),
            "credit_gae_lambda": float(update_kwargs["gae_lambda"]),
            "reward_scale": reward_scale,
            "repair_method": repair_method,
            "evaluation_checkpoints_expected": len(profile.evaluation_batches),
            "evaluation_new_rollouts_expected": profile.evaluation_rollouts,
            "evaluation_requests_expected": profile.evaluation_rollouts * 24,
            "baseline_reuse_expected": 4,
            "horizon_steps_per_rollout": horizon_steps,
            "training_max_steps": len(cells) * horizon_steps,
            "evaluation_new_max_steps": profile.evaluation_rollouts * horizon_steps,
            "source_q_fixed": source.get("source_q_digest") == pilot._model_digest(source_policy),
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        report.update({"status": "FAILED", "full_evaluation": False, "unresolved": [str(exc)]})
        report["checks"]["preflight_exception"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(output / "report.json", report)
        _progress(
            output, status="FAILED", phase="preflight", check_only=check_only,
            error=str(exc), repair_method=repair_method,
            profile=profile,
        )
        return report
    if check_only:
        report.update({"status": "CHECK_ONLY_OK", "full_evaluation": False})
        report["checks"].update({
            "runtime_rollouts": 0, "update_calls": 0, "evaluation_rollouts": 0,
            "training_rollouts": 0, "batches_processed": 0, "updates_processed": 0,
            "profiles_and_budgets_checked": True,
        })
        report["evaluation"]["baseline_reused_records"] = 4
        _write_json(output / "report.json", report)
        _progress(
            output, status="CHECK_ONLY_OK", phase="preflight", check_only=True,
            repair_method=repair_method,
            profile=profile,
        )
        return report

    report.update({"status": "RUNNING", "full_evaluation": False})
    _write_json(output / "report.json", report)
    try:
        dense = neutral_configs["dense24"]
        trainable_scope = pilot._configure_trainable_policy(policy)
        report["checks"]["trainable_scope"] = trainable_scope
        source_digest = source["source_q_digest"]
        optimizer_identity = id(policy.optimizer)
        cumulative_steps: list[Any] = []
        current_checkpoint = Path(warmup["warmup_path"])
        baseline_output_records, _ = _baseline_records(dense, output=output)
        report["evaluation"]["baseline_reused_records"] = len(baseline_output_records)
        evaluation_records: list[dict[str, Any]] = []
        for batch_index, batch_cells in enumerate(batches, start=1):
            batch_dir = output / "training" / f"batch_{batch_index:02d}"
            permutation_seed = int(profile.permutation_seeds[batch_index - 1])
            _progress(
                output, status="RUNNING", phase=f"batch_{batch_index:02d}_collect",
                batches=batch_index - 1, updates=batch_index - 1,
                evaluations=len(evaluation_records), repair_method=repair_method,
                profile=profile,
            )
            np.random.seed(permutation_seed)
            batch_steps, batch_records, collection, stop_reason = joint._collect_training(
                output=batch_dir, policy=policy, checkpoint=current_checkpoint, config=dense,
                update_kwargs=update_kwargs, training_cells=batch_cells,
            )
            report["training"]["batches_processed"] = batch_index - 1
            report["training_rollouts"] = sum(
                int(item.get("rollouts", 0) or 0) for item in report["training"]["batches"]
            )
            report["training_steps"] = sum(
                int(item.get("steps", 0) or 0) for item in report["training"]["batches"]
            )
            if stop_reason is not None:
                report["training"]["batches"].append({
                    "batch_index": batch_index, "cells": [dict(cell) for cell in batch_cells],
                    "rollouts": len(batch_records), "steps": len(batch_steps),
                    "record_paths": [str(row.get("record_path")) for row in batch_records],
                    "collection": dict(collection), "stop_reason": stop_reason,
                })
                raise LearningCurveError(stop_reason)
            _require(len(batch_records) == profile.episodes_per_batch and len(batch_steps) > 0,
                     f"learning-curve batch {batch_index} collection incomplete")
            reward_units = _reward_scale_diagnostics(
                batch_steps, batch_records, reward_scale,
            )
            learning_steps = (
                _scaled_training_steps(batch_steps, reward_scale)
                if episodic_return else batch_steps
            )
            cumulative_steps.extend(batch_steps)
            observations, weights = pilot.actor_world_observation_weights(cumulative_steps)
            policy.gate_reference = pilot._build_gate_reference(source_policy, observations, weights)
            before_state = copy.deepcopy(policy.model.state_dict())
            before_digest = pilot._model_digest(policy)
            np.random.seed(permutation_seed)
            _progress(
                output, status="RUNNING", phase=f"batch_{batch_index:02d}_update",
                batches=batch_index - 1, updates=batch_index - 1,
                evaluations=len(evaluation_records), repair_method=repair_method,
                profile=profile,
            )
            credit = (
                _episodic_return_diagnostics(
                    learning_steps, update_kwargs, reward_scale=reward_scale,
                )
                if episodic_return else None
            )
            if credit is not None:
                _require(
                    credit["full_episode_mc_return_match"],
                    "full episode MC returns do not match reward suffixes",
                )
            _require(
                np.isclose(
                    float(sum(float(row.reward) for row in learning_steps)),
                    float(reward_units["expected_learning_reward_total"]),
                    rtol=1e-5, atol=1e-4,
                ),
                "learning reward total disagrees with scaled raw total",
            )
            report["training"]["update_calls"] = batch_index
            report["ppo_updates"] = batch_index
            stats = policy.update(list(learning_steps), **update_kwargs)
            stats_for_report: Mapping[str, Any] = (
                dict(stats) if isinstance(stats, Mapping)
                else {"type": type(stats).__name__, "value": repr(stats)}
            )
            after_state = copy.deepcopy(policy.model.state_dict())
            freeze = pilot._frozen_bitwise_check(before_state, after_state)
            q_after = pilot._model_digest(source_policy)
            update_reason = pilot._update_failure(stats)
            if not freeze.get("frozen_parameter_bitwise_unchanged", False):
                update_reason = update_reason or "frozen parameters changed during learning-curve update"
            if not freeze.get("proposal_bitwise_unchanged", False):
                update_reason = update_reason or "proposal parameters changed during learning-curve update"
            if not freeze.get("gate_temperature_bitwise_unchanged", False):
                update_reason = update_reason or "gate temperature changed during learning-curve update"
            if q_after != source_digest:
                update_reason = update_reason or "source q policy changed during learning-curve update"
            optimizer_ok = id(policy.optimizer) == optimizer_identity
            if not optimizer_ok:
                update_reason = update_reason or "optimizer identity changed across learning-curve updates"
            checkpoint_info: dict[str, Any] | None = None
            if update_reason is None:
                checkpoint_path = batch_dir / "policy.pt"
                stage_metadata = {
                    "stage": STAGE,
                    "curriculum_id": "task_neutral_development",
                    "task_priority_enabled": False,
                    "low_level_priority_time_weight": 0.0,
                    "PPO_ALLOWED": False,
                    "development_pilot_authorized": True,
                    "repair_method": repair_method,
                    "profile": profile.name,
                    "training_batches": profile.batch_count,
                    "episodes_per_batch": profile.episodes_per_batch,
                    "training_rollouts": profile.training_rollouts,
                    "evaluation_batches": list(profile.evaluation_batches),
                    "episodic_return": episodic_return,
                    "credit_method": credit.get("credit_method") if credit else "gae_bootstrap",
                    "credit_gamma": float(update_kwargs["gamma"]),
                    "credit_gae_lambda": float(update_kwargs["gae_lambda"]),
                    "reward_scale": reward_scale,
                    "reward_unit": "critic_learning_reward",
                    "batch_index": batch_index,
                    "actor_reduction": "joint_team",
                    "training_budget": {
                        "updates": profile.batch_count,
                        "batch_index": batch_index,
                        "epochs": update_kwargs["epochs"],
                        "minibatch_size": update_kwargs["minibatch_size"],
                        "learning_rate": pilot.LEARNING_RATE,
                        "permutation_seed": permutation_seed,
                        "ppo_update_kwargs": dict(update_kwargs),
                    },
                }
                checkpoint_info = pilot._roundtrip_pilot(checkpoint_path, policy, dense, stage_metadata)
                checkpoint_info["path"] = str(checkpoint_path)
                current_checkpoint = checkpoint_path
                report["artifacts"]["checkpoints"].append(str(checkpoint_path))
            actor_decisions = int(sum(int(row.actor_mask.sum()) for row in batch_steps))
            batch_summary = _batch_summary(
                batch_index=batch_index, cells=batch_cells, steps=batch_steps,
                records=batch_records, stats=stats_for_report, freeze=freeze, collection=collection,
                anchor_steps=len(cumulative_steps), anchor_observations=len(observations),
                anchor_weight_sum=float(weights.sum()),
                optimizer_identity_unchanged=optimizer_ok, checkpoint=checkpoint_info,
                credit=credit, reward_units=reward_units,
            )
            batch_summary["permutation_seed"] = permutation_seed
            batch_summary["before_model_digest"] = before_digest
            batch_summary["source_q_model_digest"] = source_digest
            batch_summary["source_q_model_digest_after"] = q_after
            batch_summary["update_stop_reason"] = update_reason
            batch_summary["batch_steps_for_actor_normalizer"] = len(batch_steps)
            batch_summary["batch_actor_decisions"] = actor_decisions
            _write_json(batch_dir / "summary.json", batch_summary)
            report["training"]["batches"].append(batch_summary)
            report["training"]["batches_processed"] = batch_index
            report["training"]["updates"] = batch_index
            report["training_rollouts"] = sum(
                int(item.get("rollouts", 0) or 0) for item in report["training"]["batches"]
            )
            report["training_steps"] = sum(
                int(item.get("steps", 0) or 0) for item in report["training"]["batches"]
            )
            if update_reason is not None:
                raise LearningCurveError(update_reason)
            if batch_index in profile.evaluation_batches:
                _progress(
                    output, status="RUNNING", phase=f"evaluation_after_batch_{batch_index:02d}",
                    batches=batch_index, updates=batch_index,
                    evaluations=len(evaluation_records), repair_method=repair_method,
                    profile=profile,
                )
                after_records, eval_reason = _run_evaluation(
                    output=output, batch_index=batch_index, checkpoint=current_checkpoint, config=dense,
                )
                evaluation_records.extend(after_records)
                paired_differences = (
                    _evaluation_pairs(baseline_output_records, after_records)
                    if eval_reason is None and len(after_records) == len(EVALUATION_CELLS)
                    else []
                )
                snapshot = {
                    "batch_index": batch_index,
                    "checkpoint": str(current_checkpoint),
                    "after_records": after_records,
                    "after_summary": _metric_summary(after_records),
                    "before_reused_records": 4,
                    "before_summary": _metric_summary(baseline_output_records),
                    "paired_differences": paired_differences,
                    "new_rollouts": len(after_records),
                    "evaluation_error": eval_reason,
                }
                report["evaluation"]["snapshots"].append(snapshot)
                report["evaluation"]["new_rollouts"] = len(evaluation_records)
                report["evaluation_rollouts"] = len(evaluation_records)
                _write_json(output / "report.json", report)
                if eval_reason is not None:
                    raise LearningCurveError(eval_reason)
            _progress(
                output, status="RUNNING", phase=f"batch_{batch_index:02d}_complete",
                batches=batch_index, updates=batch_index,
                evaluations=len(evaluation_records), repair_method=repair_method,
                profile=profile,
            )
            _write_json(output / "report.json", report)
        report["training"]["steps"] = int(sum(int(item["steps"]) for item in report["training"]["batches"]))
        report["training"]["rollouts"] = int(sum(int(item["rollouts"]) for item in report["training"]["batches"]))
        report["training_rollouts"] = report["training"]["rollouts"]
        report["training_steps"] = report["training"]["steps"]
        report["ppo_updates"] = report["training"]["updates"]
        report["evaluation_rollouts"] = len(evaluation_records)
        report["evaluation"]["records_total"] = 4 + len(evaluation_records)
        report["evaluation"]["baseline_record_paths"] = [str(row.get("record_path")) for row in baseline_output_records]
        report["checks"].update({
            "batches_completed": report["training"]["batches_processed"] == profile.batch_count,
            "updates_completed": report["training"]["updates"] == profile.batch_count,
            "evaluation_snapshots_completed": len(report["evaluation"]["snapshots"]) == len(profile.evaluation_batches),
            "new_evaluations_completed": len(evaluation_records) == profile.evaluation_rollouts,
            "baseline_reused_once": len(baseline_output_records) == 4,
            "optimizer_identity_unchanged": all(
                item.get("optimizer_identity_unchanged") is True
                for item in report["training"]["batches"]
            ),
            "source_q_fixed": all(
                item.get("source_q_model_digest_after") == source_digest
                for item in report["training"]["batches"]
            ),
            "teacher_called": False,
            "heldout_consumed": False,
        })
        if profile.name == DEFAULT_PROFILE.name:
            report["checks"].update({
                "six_batches_completed": report["training"]["batches_processed"] == 6,
                "six_updates_completed": report["training"]["updates"] == 6,
                "three_evaluation_snapshots": len(report["evaluation"]["snapshots"]) == 3,
                "twelve_new_evaluations": len(evaluation_records) == 12,
            })
        report.update({"status": "COMPLETED", "full_evaluation": True, "technical_ok": True})
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        report.update({"status": "FAILED", "full_evaluation": False, "unresolved": [str(exc)]})
        report["checks"]["runtime_exception"] = {"type": type(exc).__name__, "message": str(exc)}
        _progress(
            output, status="FAILED", phase="runtime",
            batches=report["training"].get("batches_processed", 0),
            updates=report["training"].get("updates", 0),
            evaluations=report["evaluation"].get("new_rollouts", 0), error=str(exc),
            repair_method=repair_method,
            profile=profile,
        )
    _write_json(output / "report.json", report)
    if report.get("status") == "COMPLETED":
        _progress(
            output, status="COMPLETED", phase="complete", batches=profile.batch_count,
            updates=profile.batch_count, evaluations=profile.evaluation_rollouts,
            repair_method=repair_method, profile=profile,
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--episodic-return", action="store_true")
    parser.add_argument("--larger-batch", action="store_true")
    parser.add_argument("--initial-warmup-checkpoint", type=Path)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(
            output_dir=args.output_dir,
            check_only=bool(args.check_only),
            episodic_return=bool(args.episodic_return),
            larger_batch=bool(args.larger_batch),
            initial_warmup_checkpoint=args.initial_warmup_checkpoint,
            reward_scale=float(args.reward_scale),
        )
    except (LearningCurveError, FileNotFoundError, OSError, ValueError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({
        "stage": STAGE, "status": report.get("status"),
        "batches_processed": report.get("training", {}).get("batches_processed", 0),
        "updates": report.get("training", {}).get("updates", 0),
        "evaluation_rollouts": report.get("evaluation", {}).get("new_rollouts", 0),
        "PPO_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return 0 if report.get("status") in {"CHECK_ONLY_OK", "COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
