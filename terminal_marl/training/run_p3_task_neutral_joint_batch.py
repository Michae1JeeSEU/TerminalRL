"""Run the bounded multi-scene task-neutral development batch.

The runner collects six native stochastic rollouts with one unchanged actor,
performs one joint PPO update, and then evaluates the fixed source and the
resulting checkpoint on six new physical cells.  It is deliberately separate
from the diagnostic tape replay path: training steps keep the native
``stochastic_behavior=True`` marker and their native log probabilities.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.algo.student_ppo_v4_2 import StudentStepV42, team_gae
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import run_p3_task_neutral_pilot as pilot


STAGE = pilot.STAGE
REPAIR_METHOD = "multi_scene_joint_batch"
PERMUTATION_SEED = 875001
EXPECTED_CANDIDATE_SHA256 = pilot.EXPECTED_CANDIDATE_SHA256
CANDIDATE_CHECKPOINT = pilot.CANDIDATE_CHECKPOINT
WARMUP_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_critic_warmup_20260928/warmup/warmup.pt"
)
WARMUP_CHECKPOINT_SHA256 = "b136a57a8a977b60d5ae71fe3925c8cf133433c0c5fc4e6bef9872295353844d"

TRAINING_CELLS = (
    {"scenario": "dense24", "seed": 873103, "policy_seed": 873103, "split": "development_training"},
    {"scenario": "dense24", "seed": 873106, "policy_seed": 873106, "split": "development_training"},
    {"scenario": "dense24", "seed": 874601, "policy_seed": 874701, "split": "development_training"},
    {"scenario": "dense24", "seed": 874602, "policy_seed": 874702, "split": "development_training"},
    {"scenario": "dense24", "seed": 874603, "policy_seed": 874703, "split": "development_training"},
    {"scenario": "dense24", "seed": 874604, "policy_seed": 874704, "split": "development_training"},
)
EVALUATION_CELLS = (
    {"scenario": "dense24", "seed": 875101, "policy_seed": 875201, "split": "new_physical"},
    {"scenario": "dense24", "seed": 875102, "policy_seed": 875202, "split": "new_physical"},
    {"scenario": "dense24", "seed": 875103, "policy_seed": 875203, "split": "new_physical"},
    {"scenario": "dense24", "seed": 875104, "policy_seed": 875204, "split": "new_physical"},
    {"scenario": "flow6", "seed": 875105, "policy_seed": 875205, "split": "new_physical"},
    {"scenario": "flow12", "seed": 875106, "policy_seed": 875206, "split": "new_physical"},
)
EXPECTED_EVALUATION_REQUESTS = sum(
    int(neutral_eval.EXPECTED_REQUESTS[str(cell["scenario"])]) for cell in EVALUATION_CELLS
)


class JointBatchError(ValueError):
    """Raised when the fixed joint-batch contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JointBatchError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError, RuntimeError):
            pass
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_state_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_state_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _credit_update_kwargs(*, full_episode_credit: bool = False) -> dict[str, Any]:
    """Return per-run PPO kwargs without mutating the pilot defaults."""

    value = dict(pilot.UPDATE_KWARGS)
    if full_episode_credit:
        value.update({"gamma": 1.0, "gae_lambda": 1.0})
    return value


def _full_episode_mc_returns(steps: Sequence[StudentStepV42]) -> np.ndarray:
    """Compute undiscounted, non-bootstrap returns independently per episode."""

    returns = np.zeros(len(steps), dtype=np.float32)
    running = 0.0
    for index in range(len(steps) - 1, -1, -1):
        row = steps[index]
        if row.episode_end:
            running = float(row.reward)
        else:
            running = float(row.reward) + running
        returns[index] = running
    return returns


def _compact_gae(
    steps: Sequence[StudentStepV42],
    update_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = dict(pilot.UPDATE_KWARGS) if update_kwargs is None else dict(update_kwargs)
    diagnostic = pilot._gae_diagnostics(
        steps,
        gamma=float(kwargs["gamma"]),
        gae_lambda=float(kwargs["gae_lambda"]),
    )
    return {
        "gamma": diagnostic["gamma"],
        "gae_lambda": diagnostic["gae_lambda"],
        "actor_steps": diagnostic["actor_steps"],
        "advantages": _stats(diagnostic["advantages"]),
        "returns": _stats(diagnostic["returns"]),
        "normalized_advantages": _stats(diagnostic["normalized_advantages"]),
    }


def _episode_boundary_check(steps: Sequence[StudentStepV42]) -> dict[str, Any]:
    _require(bool(steps), "native training rollout returned no steps")
    ends = [index for index, row in enumerate(steps) if bool(row.episode_end)]
    _require(ends == [len(steps) - 1], "training episode boundary is not a single final step")
    final = steps[-1]
    _require(bool(final.terminated), "training episode did not terminate")
    _require(float(final.next_value) == 0.0, "terminal training step bootstraps a value")
    _require(all(not row.teacher_used for row in steps), "Teacher behavior was recorded")
    _require(all(bool(row.stochastic_behavior) for row in steps),
             "native training rollout contains deterministic behavior")
    return {
        "steps": int(len(steps)),
        "episode_end_indices": ends,
        "terminated_final": True,
        "next_value_final": float(final.next_value),
        "teacher_used": False,
        "stochastic_behavior": True,
    }


def _training_stop_reason(record: Mapping[str, Any], steps: Sequence[StudentStepV42]) -> str | None:
    if record.get("error"):
        error = record.get("error")
        return str(error.get("message", "training rollout error")) if isinstance(error, Mapping) else str(error)
    metrics = neutral_eval._record_metrics(record)
    expected = int(neutral_eval.EXPECTED_REQUESTS[str(record.get("scenario"))])
    if record.get("weights_unchanged") is not True:
        return "training rollout changed model weights"
    if bool(metrics.get("safety_nonzero")):
        return "training rollout reported a safety violation"
    if metrics.get("technical_ok") is not True or metrics.get("lifecycle_valid") is not True:
        return "training lifecycle fields are incomplete"
    if int(metrics.get("requests", 0) or 0) != expected:
        return f"training request count is not {expected}"
    try:
        _episode_boundary_check(steps)
    except BaseException as exc:
        return str(exc)
    # A terminal failed-energy outcome is retained as a training sample.  It
    # is represented in the record and does not enter this stop predicate.
    return None


def _training_cell_record(
    record: Mapping[str, Any], steps: Sequence[StudentStepV42], cell: Mapping[str, Any],
    update_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = dict(pilot.UPDATE_KWARGS) if update_kwargs is None else dict(update_kwargs)
    value = dict(record)
    critic_mc = pilot._critic_discounted_mc_error(
        steps, gamma=float(kwargs["gamma"]),
    )
    value.update({
        "cell": dict(cell),
        "credit_method": (
            "full_episode_mc"
            if float(kwargs["gamma"]) == 1.0 and float(kwargs["gae_lambda"]) == 1.0
            else "gae_bootstrap"
        ),
        "credit_gamma": float(kwargs["gamma"]),
        "credit_gae_lambda": float(kwargs["gae_lambda"]),
        "sampling": {
            "deterministic": False,
            "native_stochastic_behavior": bool(steps) and all(row.stochastic_behavior for row in steps),
            "teacher": False,
        },
        "episode_boundary": _episode_boundary_check(steps),
        "update_attempted": False,
        "training_or_update_called": False,
        "pre_batch": {
            "gae": _compact_gae(steps, kwargs),
            "critic_mc": critic_mc,
            "critic_discounted_mc_error": critic_mc,
        },
    })
    return value


def _paired_metrics(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_metrics = neutral_eval._record_metrics(before)
    after_metrics = neutral_eval._record_metrics(after)

    def delta(path: Sequence[str]) -> float | None:
        left: Any = before_metrics
        right: Any = after_metrics
        for key in path:
            left = left.get(key) if isinstance(left, Mapping) else None
            right = right.get(key) if isinstance(right, Mapping) else None
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if np.isfinite(float(left)) and np.isfinite(float(right)):
                return float(right) - float(left)
        return None

    numeric = {
        "requests": delta(("requests",)),
        "normal": delta(("normal",)),
        "emergency": delta(("emergency",)),
        "failed": delta(("failed",)),
        "failed_energy": delta(("failed_energy",)),
        "mean_wait": delta(("mean_wait",)),
        "p95_wait": delta(("p95_wait",)),
        "max_wait": delta(("max_wait",)),
        "lowSOC_wait": delta(("lowSOC_wait",)),
        "reward_total": delta(("reward", "total")),
    }
    return {
        "before": before_metrics,
        "after": after_metrics,
        "after_minus_before": numeric,
    }


def _pair_evaluation_records(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    def index(records: Sequence[Mapping[str, Any]], label: str) -> dict[tuple[int, str], Mapping[str, Any]]:
        result: dict[tuple[int, str], Mapping[str, Any]] = {}
        for record in records:
            key = (int(record["seed"]), str(record["scenario"]))
            _require(key not in result, f"duplicate {label} evaluation cell: {key}")
            result[key] = record
        return result

    before_by_key = index(before, "before")
    after_by_key = index(after, "after")
    rows: list[dict[str, Any]] = []
    for cell in EVALUATION_CELLS:
        key = (int(cell["seed"]), str(cell["scenario"]))
        _require(key in before_by_key and key in after_by_key, f"missing paired evaluation cell: {key}")
        rows.append({
            "cell": dict(cell),
            "seed": key[0],
            "scenario": key[1],
            "metrics": _paired_metrics(before_by_key[key], after_by_key[key]),
        })
    return rows


def _progress(output: Path, *, status: str, phase: str, training_processed: int = 0,
              evaluation_processed: int = 0, update_calls: int = 0,
              check_only: bool = False, error: str | None = None) -> None:
    value: dict[str, Any] = {
        "stage": STAGE,
        "status": str(status),
        "phase": str(phase),
        "training_cells_processed": int(training_processed),
        "training_cells_expected": 0 if check_only else len(TRAINING_CELLS),
        "evaluation_rollouts_processed": int(evaluation_processed),
        "evaluation_rollouts_expected": 0 if check_only else len(EVALUATION_CELLS) * 2,
        "training_update_calls": int(update_calls),
        "check_only": bool(check_only),
        "PPO_ALLOWED": False,
        "teacher_called": False,
    }
    if error:
        value["error"] = str(error)
    _write_json(output / "progress.json", value)


def _base_report(
    output: Path, *, check_only: bool, full_episode_credit: bool = False,
    update_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = dict(pilot.UPDATE_KWARGS) if update_kwargs is None else dict(update_kwargs)
    return {
        "stage": STAGE,
        "status": "STARTING",
        "check_only": bool(check_only),
        "repair_method": REPAIR_METHOD,
        "credit_method": "full_episode_mc" if full_episode_credit else "gae_bootstrap",
        "full_episode_credit": bool(full_episode_credit),
        "credit_update_kwargs": dict(kwargs),
        "training_rollouts": 0,
        "ppo_updates": 0,
        "evaluation_rollouts": 0,
        "PPO_ALLOWED": False,
        "teacher_called": False,
        "heldout_consumed": False,
        "actions": {"teacher": False, "heldout": False, "optimizer_tuning": False},
        "artifacts": {"output_dir": str(output)},
        "checks": {},
        "deviations": [],
        "unresolved": [],
    }


def _preflight(neutral_configs: Mapping[str, Any]) -> tuple[Path, Any, dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    dense = neutral_configs["dense24"]
    resolved, policy, binding = pilot._load_binding(dense)
    _require(_sha256(resolved) == EXPECTED_CANDIDATE_SHA256, "candidate checkpoint SHA mismatch")
    source_policy, source_payload = rank_eval._load_policy(resolved, dense)
    source_q_digest = pilot._model_digest(source_policy)
    warmup_path, warmup_binding = pilot._load_initial_warmup(WARMUP_CHECKPOINT, policy, dense)
    warmup_sha = _sha256(warmup_path)
    _require(warmup_sha == WARMUP_CHECKPOINT_SHA256, "warmup checkpoint SHA mismatch")
    return resolved, policy, binding, source_policy, {
        "source_payload_keys": sorted(source_payload) if isinstance(source_payload, Mapping) else [],
        "source_q_digest": source_q_digest,
        "warmup_path": str(warmup_path),
        "warmup_sha256": warmup_sha,
        "warmup_binding": warmup_binding,
    }, {"source_q_digest": source_q_digest}


def _binding_report(
    binding: Mapping[str, Any], warmup: Mapping[str, Any],
    update_kwargs: Mapping[str, Any] | None = None,
    *, full_episode_credit: bool = False,
) -> dict[str, Any]:
    kwargs = dict(pilot.UPDATE_KWARGS) if update_kwargs is None else dict(update_kwargs)
    value = dict(binding)
    value.update({
        "candidate_checkpoint_sha256_expected": EXPECTED_CANDIDATE_SHA256,
        "candidate_checkpoint_sha256_actual": binding.get("checkpoint_sha256"),
        "warmup": dict(warmup),
        "training_cells": [dict(cell) for cell in TRAINING_CELLS],
        "evaluation_cells": [dict(cell) for cell in EVALUATION_CELLS],
        "evaluation_requests_per_arm": EXPECTED_EVALUATION_REQUESTS,
        "update_budget": dict(kwargs),
        "credit_method": "full_episode_mc" if full_episode_credit else "gae_bootstrap",
        "full_episode_credit": bool(full_episode_credit),
        "credit_gamma": float(kwargs["gamma"]),
        "credit_gae_lambda": float(kwargs["gae_lambda"]),
        "permutation_seed": PERMUTATION_SEED,
        "policy_profile": "dense24 neutral task-priority config",
    })
    return value


def _collect_training(
    *, output: Path, policy: Any, checkpoint: Path, config: Any,
    update_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[StudentStepV42], list[dict[str, Any]], dict[str, Any], str | None]:
    kwargs = dict(pilot.UPDATE_KWARGS) if update_kwargs is None else dict(update_kwargs)
    all_steps: list[StudentStepV42] = []
    records: list[dict[str, Any]] = []
    source_digest = pilot._model_digest(policy)
    cell_digests: list[dict[str, Any]] = []
    for index, cell in enumerate(TRAINING_CELLS, start=1):
        _progress(output, status="RUNNING", phase=f"training_cell_{index}", training_processed=index - 1)
        episode_dir = output / "training" / f"episode_{index:02d}"
        steps, raw_record = pilot._training_record(
            policy=policy,
            checkpoint=checkpoint,
            config=config,
            scenario=str(cell["scenario"]),
            seed=int(cell["seed"]),
            policy_seed=int(cell["policy_seed"]),
            output=episode_dir / "rollout",
        )
        record = pilot._record_annotations(dict(raw_record), config)
        reason = _training_stop_reason(record, steps)
        if reason is None:
            if update_kwargs is None:
                record = _training_cell_record(record, steps, cell)
            else:
                record = _training_cell_record(record, steps, cell, kwargs)
            all_steps.extend(steps)
        else:
            record.update({"cell": dict(cell), "update_attempted": False,
                           "training_or_update_called": False, "stop_reason": reason})
        record["record_path"] = str(episode_dir / "record.json")
        _write_json(episode_dir / "record.json", record)
        records.append(record)
        cell_digest = pilot._model_digest(policy)
        cell_digests.append({"cell": dict(cell), "model_digest_after": cell_digest,
                             "unchanged": cell_digest == source_digest})
        if cell_digest != source_digest:
            return (
                all_steps,
                records,
                {"model_digest_before_collection": source_digest,
                 "model_digest_after_collection": cell_digest,
                 "per_cell": cell_digests},
                "policy model changed during native training collection",
            )
        if reason is not None:
            return all_steps, records, {"model_digest_after_collection": source_digest}, reason
    _require(len(records) == len(TRAINING_CELLS), "joint training collection is incomplete")
    return all_steps, records, {
        "model_digest_before_collection": source_digest,
        "model_digest_after_collection": source_digest,
        "per_cell": cell_digests,
        "bitwise_unchanged": True,
    }, None


def _reuse_before_records(
    root: Path, *, output: Path | None, neutral_configs: Mapping[str, Any],
    evaluation_kind: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate and copy the six fixed before records without runtime calls."""

    records: list[dict[str, Any]] = []
    source_paths: list[str] = []
    seen: set[tuple[int, str, int]] = set()
    for cell in EVALUATION_CELLS:
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        policy_seed = int(cell["policy_seed"])
        key = (seed, scenario, policy_seed)
        _require(key not in seen, f"duplicate reused before identity: {key}")
        seen.add(key)
        source = neutral_eval._synced_path(root / f"{scenario}_{seed}.json")
        _require(source.is_file(), f"reused before record missing: {source}")
        record = neutral_eval._read_json(source)
        _require(record.get("arm") == "before", f"reused before arm mismatch: {source}")
        _require(str(record.get("scenario")) == scenario,
                 f"reused before scenario mismatch: {source}")
        _require(int(record.get("seed")) == seed,
                 f"reused before environment seed mismatch: {source}")
        _require(int(record.get("policy_seed")) == policy_seed,
                 f"reused before policy seed mismatch: {source}")
        cell_metadata = record.get("cell")
        if isinstance(cell_metadata, Mapping):
            _require(
                str(cell_metadata.get("scenario")) == scenario
                and int(cell_metadata.get("seed")) == seed
                and int(cell_metadata.get("policy_seed")) == policy_seed,
                f"reused before cell identity mismatch: {source}",
            )
        expected_requests = int(neutral_eval.EXPECTED_REQUESTS[scenario])
        _require(int(record.get("requests", 0) or 0) == expected_requests,
                 f"reused before request count mismatch: {source}")
        checkpoint_sha = (
            record.get("checkpoint_sha256")
            or record.get("checkpoint_sha256_expected")
            or record.get("candidate_checkpoint_sha256")
        )
        _require(checkpoint_sha == EXPECTED_CANDIDATE_SHA256,
                 f"reused before checkpoint SHA mismatch: {source}")
        _require(record.get("status") == "completed" and record.get("lifecycle_valid") is True,
                 f"reused before status/lifecycle incomplete: {source}")
        _require(record.get("technical_ok") is True and not record.get("error"),
                 f"reused before technical fields incomplete: {source}")
        _require(record.get("weights_unchanged") is True
                 and record.get("model_unchanged", True) is True,
                 f"reused before model state is not frozen: {source}")
        _require(record.get("training_or_update_called") is False
                 and record.get("teacher_called") is False,
                 f"reused before update/Teacher state mismatch: {source}")
        _require(record.get("task_priority_enabled") is False
                 and float(record.get("low_level_priority_time_weight", 1.0)) == 0.0,
                 f"reused before task-neutral flags mismatch: {source}")
        config = neutral_configs[scenario]
        _require(record.get("config") == neutral_eval._config_snapshot(config),
                 f"reused before config mismatch: {source}")
        metrics = neutral_eval._record_metrics(record)
        _require(metrics.get("technical_ok") is True and not bool(metrics.get("safety_nonzero")),
                 f"reused before safety/technical metrics mismatch: {source}")
        _require(neutral_eval._record_fatal_reason(record, smoke=False) is None,
                 f"reused before fatal record state: {source}")
        summary = record.get("summary")
        if isinstance(summary, Mapping):
            _require(int(summary.get("seed")) == seed,
                     f"reused before summary seed mismatch: {source}")
            _require(int(summary.get("policy_seed")) == policy_seed,
                     f"reused before summary policy seed mismatch: {source}")
        destination = (
            output / "evaluation" / "records" / "before" / f"{scenario}_{seed}.json"
            if output is not None else source
        )
        copied = dict(record)
        copied.update({
            "arm": "before",
            "cell": dict(cell),
            "evaluation_kind": evaluation_kind,
            "record_path": str(destination),
            "source_record_path": str(source),
            "source_provenance": {"kind": "reused_before_record", "source_path": str(source)},
        })
        if output is not None:
            neutral_eval._write_json(destination, copied)
        records.append(copied)
        source_paths.append(str(source))
    _require(len(records) == len(EVALUATION_CELLS) and len(seen) == len(EVALUATION_CELLS),
             "reuse root must contain exactly six before records")
    return records, source_paths


def _evaluate_arm(
    *, output: Path, arm: str, checkpoint: Path, neutral_configs: Mapping[str, Any],
    progress_offset: int = 0, evaluation_kind: str = "new_physical_scenes",
) -> tuple[list[dict[str, Any]], str | None]:
    records: list[dict[str, Any]] = []
    for index, cell in enumerate(EVALUATION_CELLS, start=1):
        _progress(output, status="RUNNING", phase=f"evaluation_{arm}",
                  training_processed=len(TRAINING_CELLS),
                  evaluation_processed=progress_offset + len(records), update_calls=1)
        config = neutral_configs[str(cell["scenario"])]
        policy, payload = rank_eval._load_policy(checkpoint, config)
        if arm == "after":
            pilot._validate_pilot_payload(payload, config)
        record = rank_eval._runtime_record(
            policy=policy,
            checkpoint=checkpoint,
            config=config,
            scenario=str(cell["scenario"]),
            seed=int(cell["seed"]),
            policy_seed=int(cell["policy_seed"]),
            output=output / "evaluation" / "rollouts" / arm / f"{cell['scenario']}_{cell['seed']}",
            smoke=False,
        )
        record = pilot._record_annotations(dict(record), config)
        record.update({"arm": arm, "cell": dict(cell), "evaluation_kind": evaluation_kind})
        path = output / "evaluation" / "records" / arm / f"{cell['scenario']}_{cell['seed']}.json"
        record["record_path"] = str(path)
        _write_json(path, record)
        records.append(record)
        _progress(output, status="RUNNING", phase=f"evaluation_{arm}",
                  training_processed=len(TRAINING_CELLS),
                  evaluation_processed=progress_offset + len(records), update_calls=1)
        reason = neutral_eval._record_fatal_reason(record, smoke=False)
        if reason is not None:
            return records, f"evaluation/{arm}/{cell['scenario']}/{cell['seed']}: {reason}"
    return records, None


def run(
    *, output_dir: str | Path, check_only: bool = False,
    full_episode_credit: bool = False,
    reuse_before_root: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    full_episode_credit = bool(full_episode_credit)
    reuse_root = Path(reuse_before_root) if reuse_before_root is not None else None
    update_kwargs = _credit_update_kwargs(full_episode_credit=full_episode_credit)
    evaluation_kind = (
        "development_existing_scenes" if full_episode_credit else "new_physical_scenes"
    )
    reuse_preflight_paths: list[str] = []
    _require(not output.exists(), f"joint-batch output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _progress(output, status="STARTING", phase="preflight", check_only=check_only)
    report = _base_report(
        output, check_only=check_only, full_episode_credit=full_episode_credit,
        update_kwargs=update_kwargs,
    )
    try:
        _require(full_episode_credit or reuse_root is None,
                 "--reuse-before-root requires --full-episode-credit")
        configs, neutral_configs, _runtime_protocol = neutral_eval._build_neutral_inputs()
        del _runtime_protocol
        _require(all(
            getattr(config, "task_priority_enabled", None) is False
            and float(getattr(config, "low_level_priority_time_weight", 1.0)) == 0.0
            for config in neutral_configs.values()
        ), "neutral task-priority config binding is incomplete")
        resolved, policy, binding, source_policy, warmup, source = _preflight(neutral_configs)
        binding = _binding_report(
            binding, warmup, update_kwargs,
            full_episode_credit=full_episode_credit,
        )
        binding["source_q_model_digest"] = source["source_q_digest"]
        binding["config_snapshots"] = {
            scenario: {
                "original": neutral_eval._config_snapshot(configs[scenario]),
                "neutral": neutral_eval._config_snapshot(neutral_configs[scenario]),
                "changed_fields": neutral_eval._config_delta(configs[scenario], neutral_configs[scenario]),
            }
            for scenario in sorted(neutral_configs)
        }
        training_horizon = int(round(float(neutral_configs["dense24"].max_simulation_seconds)
                                     / float(neutral_configs["dense24"].dt)))
        evaluation_horizons = {
            f"{cell['scenario']}_{cell['seed']}": int(round(
                float(neutral_configs[str(cell["scenario"])].max_simulation_seconds)
                / float(neutral_configs[str(cell["scenario"])].dt)
            ))
            for cell in EVALUATION_CELLS
        }
        binding["simulation_budgets"] = {
            "training_horizon_steps_per_cell": training_horizon,
            "training_total_horizon_steps": training_horizon * len(TRAINING_CELLS),
            "evaluation_horizon_steps_by_cell": evaluation_horizons,
            "evaluation_total_horizon_steps": int(sum(evaluation_horizons.values()) * 2),
        }
        paired_evaluation_horizon_steps = int(sum(evaluation_horizons.values()) * 2)
        reused_before_expected = len(EVALUATION_CELLS) if reuse_root is not None else 0
        new_eval_expected = (
            len(EVALUATION_CELLS)
            if reuse_root is not None else len(EVALUATION_CELLS) * 2
        )
        binding["simulation_budgets"].update({
            "paired_evaluation_total_horizon_steps": paired_evaluation_horizon_steps,
            "reused_before_rollouts_expected": reused_before_expected,
            "new_eval_rollouts_expected": new_eval_expected,
            "new_evaluation_horizon_steps": int(
                sum(evaluation_horizons.values())
                * (1 if reuse_root is not None else 2)
            ),
        })
        binding.update({
            "reused_before_rollouts_expected": reused_before_expected,
            "new_eval_rollouts_expected": new_eval_expected,
            "new_evaluation_horizon_steps": binding["simulation_budgets"]["new_evaluation_horizon_steps"],
        })
        report["binding"] = binding
        if full_episode_credit:
            report["scope"] = {
                "training": "development_existing_scenes",
                "evaluation": "development_existing_scenes",
                "fresh_evaluation": False,
            }
        report["checks"]["reuse_before_requested"] = reuse_root is not None
        report["checks"].update({
            "candidate_checkpoint_sha256": binding["candidate_checkpoint_sha256_actual"],
            "warmup_checkpoint_sha256": warmup["warmup_sha256"],
            "config_neutralized": True,
            "training_cells_expected": len(TRAINING_CELLS),
            "evaluation_cells_expected": len(EVALUATION_CELLS),
            "evaluation_rollouts_expected": len(EVALUATION_CELLS) * 2,
            "evaluation_requests_per_arm": EXPECTED_EVALUATION_REQUESTS,
            "check_only_runtime_calls": 0 if check_only else None,
            "reused_before_rollouts_expected": reused_before_expected,
            "new_eval_rollouts_expected": new_eval_expected,
        })
        if reuse_root is not None:
            _require(reuse_root.exists(), f"reuse-before root missing: {reuse_root}")
            _, reuse_preflight_paths = _reuse_before_records(
                reuse_root, output=None, neutral_configs=neutral_configs,
                evaluation_kind=evaluation_kind,
            )
            report["checks"]["reuse_before_records_validated"] = len(reuse_preflight_paths) == len(EVALUATION_CELLS)
            report["checks"]["reuse_before_source_paths"] = list(reuse_preflight_paths)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        report.update({"status": "FAILED", "unresolved": [str(exc)]})
        report["checks"]["preflight_exception"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(output / "report.json", report)
        _progress(output, status="FAILED", phase="preflight", check_only=check_only, error=str(exc))
        return report

    if check_only:
        report.update({"status": "CHECK_ONLY_OK", "full_evaluation": False,
                       "training_rollouts": 0, "ppo_updates": 0, "evaluation_rollouts": 0})
        report["checks"].update({"runtime_rollout_calls": 0, "update_calls": 0,
                                  "env_steps": 0, "profiles_and_budgets_checked": True,
                                  "credit_kwargs_explicit": True,
                                  "reused_before_rollouts": 0,
                                  "new_eval_rollouts": 0})
        _write_json(output / "report.json", report)
        _progress(output, status="CHECK_ONLY_OK", phase="preflight", check_only=True)
        return report

    try:
        dense = neutral_configs["dense24"]
        source_digest = source["source_q_digest"]
        scope = pilot._configure_trainable_policy(policy)
        report["checks"]["trainable_scope"] = scope
        all_steps, training_records, collection, stop_reason = _collect_training(
            output=output, policy=policy, checkpoint=Path(warmup["warmup_path"]), config=dense,
            update_kwargs=update_kwargs if full_episode_credit else None,
        )
        report["training"] = {
            "records": training_records,
            "cells_processed": len(training_records),
            "steps": len(all_steps),
            "collection": collection,
            "sampling": {"deterministic": False, "teacher": False,
                          "native_stochastic_behavior": all(bool(row.stochastic_behavior) for row in all_steps)},
        }
        report["training_rollouts"] = len(training_records)
        if stop_reason is not None:
            raise JointBatchError(stop_reason)
        _require(len(all_steps) > 0, "joint training batch is empty")
        before_update_state = copy.deepcopy(policy.model.state_dict())
        before_digest = pilot._model_digest(policy)
        observations, weights = pilot.actor_world_observation_weights(all_steps)
        policy.gate_reference = pilot._build_gate_reference(source_policy, observations, weights)
        reference_digest = pilot._gate_reference_digest(policy.gate_reference)
        advantages, returns = team_gae(
            all_steps, float(update_kwargs["gamma"]), float(update_kwargs["gae_lambda"]),
        )
        pooled = {
            "steps": len(all_steps),
            "actor_steps": int(sum(bool(row.actor_mask.any()) for row in all_steps)),
            "reward": _stats([float(row.reward) for row in all_steps]),
            "advantages": _stats(advantages),
            "returns": _stats(returns),
            "observation_count": int(len(observations)),
            "weight_sum": float(weights.sum()),
            "world_count": int(sum(bool(row.actor_mask.any()) for row in all_steps)),
        }
        report["training"]["pooled"] = pooled
        if full_episode_credit:
            expected_mc = _full_episode_mc_returns(all_steps)
            max_abs_error = float(np.max(np.abs(np.asarray(returns, dtype=np.float32) - expected_mc)))
            pooled.update({
                "full_episode_mc_returns": _stats(expected_mc),
                "full_episode_mc_return_max_abs_error": max_abs_error,
                "full_episode_mc_return_match": bool(
                    np.allclose(returns, expected_mc, rtol=1e-5, atol=1e-4)
                ),
            })
            _require(pooled["full_episode_mc_return_match"],
                     "full episode MC returns do not match team GAE returns")
        p_source = pilot._actor_gate_release_mean(source_policy, all_steps)
        p_before = pilot._actor_gate_release_mean(policy, all_steps)
        report["ppo_updates"] = 1
        report["training"].update({"update_calls": 1, "pooled": pooled,
                                    "before_model_digest": before_digest,
                                    "actor_gate_p_release_mean_source_q": p_source,
                                    "actor_gate_p_release_mean_before": p_before,
                                    "permutation_seed": PERMUTATION_SEED})
        _progress(output, status="RUNNING", phase="joint_update",
                  training_processed=len(TRAINING_CELLS), update_calls=1)
        np.random.seed(PERMUTATION_SEED)
        try:
            stats = policy.update(list(all_steps), **update_kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            report["training"]["update_exception"] = {
                "type": type(exc).__name__, "message": str(exc),
            }
            raise
        after_update_state = copy.deepcopy(policy.model.state_dict())
        freeze = pilot._frozen_bitwise_check(before_update_state, after_update_state)
        q_after = pilot._model_digest(source_policy)
        stats_value = dict(stats) if isinstance(stats, Mapping) else {
            "type": type(stats).__name__, "value": repr(stats),
        }
        update_reason = pilot._update_failure(stats)
        if isinstance(stats, Mapping) and stats.get("error"):
            update_reason = update_reason or "policy.update returned an error field"
        if not freeze.get("frozen_parameter_bitwise_unchanged", False):
            update_reason = update_reason or "frozen parameters changed during joint update"
        if not freeze.get("proposal_bitwise_unchanged", False):
            update_reason = update_reason or "proposal parameters changed during joint update"
        if not freeze.get("gate_temperature_bitwise_unchanged", False):
            update_reason = update_reason or "gate temperature changed during joint update"
        if q_after != source_digest:
            update_reason = update_reason or "source q policy changed during joint update"
        report["training"].update({
            "update_stats": stats_value,
            "after_model_digest": pilot._model_digest(policy),
            "source_q_model_digest_before": source_digest,
            "source_q_model_digest_after": q_after,
            "gate_reference_digest": reference_digest,
            "actor_gate_p_release_mean_source_q": p_source,
            "actor_gate_p_release_mean_before": p_before,
            "actor_gate_p_release_mean_after": pilot._actor_gate_release_mean(policy, all_steps),
            "frozen_bitwise": freeze,
            "permutation_seed": PERMUTATION_SEED,
            "update_stop_reason": update_reason,
            "credit_method": "full_episode_mc" if full_episode_credit else "gae_bootstrap",
            "credit_gamma": float(update_kwargs["gamma"]),
            "credit_gae_lambda": float(update_kwargs["gae_lambda"]),
        })
        if update_reason is not None:
            raise JointBatchError(update_reason)
        stage_metadata = {
            "stage": STAGE,
            "curriculum_id": "task_neutral_development",
            "task_priority_enabled": False,
            "low_level_priority_time_weight": 0.0,
            "PPO_ALLOWED": False,
            "development_pilot_authorized": True,
            "repair_method": REPAIR_METHOD,
            "credit_method": "full_episode_mc" if full_episode_credit else "gae_bootstrap",
            "credit_gamma": float(update_kwargs["gamma"]),
            "credit_gae_lambda": float(update_kwargs["gae_lambda"]),
            "training_budget": {
                "updates": 1,
                "epochs": pilot.UPDATE_KWARGS["epochs"],
                "minibatch_size": pilot.UPDATE_KWARGS["minibatch_size"],
                "learning_rate": pilot.LEARNING_RATE,
                "permutation_seed": PERMUTATION_SEED,
                "ppo_update_kwargs": dict(update_kwargs),
                "training_cells": [dict(cell) for cell in TRAINING_CELLS],
            },
        }
        joint_path = output / "joint_batch.pt"
        roundtrip = pilot._roundtrip_pilot(joint_path, policy, dense, stage_metadata)
        report["artifacts"]["joint_checkpoint"] = str(joint_path)
        report["checks"]["checkpoint_roundtrip"] = roundtrip
        report["ppo_updates"] = 1
        _progress(output, status="RUNNING", phase="evaluation", training_processed=len(TRAINING_CELLS), update_calls=1)

        reused_before_paths: list[str] = []
        if reuse_root is not None:
            before_records, reused_before_paths = _reuse_before_records(
                reuse_root, output=output, neutral_configs=neutral_configs,
                evaluation_kind=evaluation_kind,
            )
            stop_reason = None
            _progress(output, status="RUNNING", phase="evaluation_before_reused",
                      training_processed=len(TRAINING_CELLS),
                      evaluation_processed=len(before_records), update_calls=1)
        else:
            before_records, stop_reason = _evaluate_arm(
                output=output, arm="before", checkpoint=resolved, neutral_configs=neutral_configs,
                progress_offset=0, evaluation_kind=evaluation_kind,
            )
        if stop_reason is None:
            after_records, stop_reason = _evaluate_arm(
                output=output, arm="after", checkpoint=joint_path, neutral_configs=neutral_configs,
                progress_offset=len(before_records), evaluation_kind=evaluation_kind,
            )
        else:
            after_records = []
        report["evaluation_rollouts"] = len(before_records) + len(after_records)
        report["evaluation"] = {
            "kind": evaluation_kind,
            "fresh_evaluation": not full_episode_credit,
            "before_checkpoint_sha256": EXPECTED_CANDIDATE_SHA256,
            "after_checkpoint_sha256": roundtrip["checkpoint_sha256"],
            "before": neutral_eval._aggregate_arm(before_records),
            "after": neutral_eval._aggregate_arm(after_records),
            "before_records": before_records,
            "after_records": after_records,
            "paired_cells": _pair_evaluation_records(before_records, after_records) if stop_reason is None else [],
            "cells_expected": len(EVALUATION_CELLS),
            "requests_per_arm_expected": EXPECTED_EVALUATION_REQUESTS,
            "reused_before_rollouts": len(before_records) if reuse_root is not None else 0,
            "new_eval_rollouts": len(after_records) + (len(before_records) if reuse_root is None else 0),
            "reused_before_source_paths": reused_before_paths,
        }
        if stop_reason is not None:
            raise JointBatchError(stop_reason)
        report.update({"status": "COMPLETED", "full_evaluation": True, "technical_ok": True})
        report["checks"].update({
            "six_training_rollouts": len(training_records) == 6,
            "one_joint_update": report["ppo_updates"] == 1,
            "twelve_evaluation_rollouts": report["evaluation_rollouts"] == 12,
            "source_q_fixed": q_after == source_digest,
            "teacher_called": False,
            "heldout_consumed": False,
            "credit_kwargs_explicit": True,
            "reused_before_rollouts": len(before_records) if reuse_root is not None else 0,
            "new_eval_rollouts": len(after_records) + (len(before_records) if reuse_root is None else 0),
        })
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        report.update({"status": "FAILED", "unresolved": [str(exc)]})
        report["checks"]["runtime_exception"] = {"type": type(exc).__name__, "message": str(exc)}
        _progress(output, status="FAILED", phase="runtime", training_processed=len(report.get("training", {}).get("records", [])),
                  evaluation_processed=report.get("evaluation_rollouts", 0), update_calls=report.get("ppo_updates", 0),
                  error=str(exc))
    _write_json(output / "report.json", report)
    if report.get("status") == "COMPLETED":
        _progress(output, status="COMPLETED", phase="complete", training_processed=len(TRAINING_CELLS),
                  evaluation_processed=len(EVALUATION_CELLS) * 2, update_calls=1)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--full-episode-credit", action="store_true")
    parser.add_argument("--reuse-before-root", type=Path)
    args = parser.parse_args(argv)
    report = run(
        output_dir=args.output_dir,
        check_only=bool(args.check_only),
        full_episode_credit=bool(args.full_episode_credit),
        reuse_before_root=args.reuse_before_root,
    )
    print(json.dumps(_jsonable(report), sort_keys=True))
    return 0 if report.get("status") in {"CHECK_ONLY_OK", "COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
