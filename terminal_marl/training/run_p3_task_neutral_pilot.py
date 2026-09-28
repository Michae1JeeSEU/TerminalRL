"""Run the bounded task-neutral development pilot.

The pilot is intentionally a small execution wrapper around the registered
V4.2 rollout, PPO, lifecycle, and checkpoint helpers.  It is development
scoped: it does not change formal admission state, consume held-out data, or
invoke a Teacher.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.algo.gate_reference_v4_2 import GateReference
from terminal_marl.algo.student_ppo_v4_2 import StudentStepV42, team_gae
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import run_p3_task_neutral_critic_warmup as critic_warmup
from terminal_marl.training import student_rollout_v4_2
from terminal_marl.training import terminal_lifecycle_metrics


STAGE = "P3_TASK_NEUTRAL_DEVELOPMENT_PILOT_V1"
PPO_ALLOWED = False
DEVELOPMENT_PILOT_AUTHORIZED = True
LEARNING_RATE = 3e-5
UPDATE_KWARGS = {
    "training_authorized": True,
    "epochs": 2,
    "minibatch_size": 32,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.1,
    "value_coef": 0.5,
    "entropy_coef": 0.001,
    "target_kl": 0.01,
}
TRAINING_CELLS = (
    {"seed": 873101, "scenario": "flow6", "split": "train"},
    {"seed": 873102, "scenario": "flow12", "split": "train"},
    {"seed": 873103, "scenario": "dense24", "split": "train"},
)
FROZEN_PREFIXES = ("actor_encoder.", "gate_head.", "proposal_head.")
TRAINABLE_PREFIXES = ("gate_adapter.", "critic_encoder.", "critic_head.")
EXPECTED_CANDIDATE_SHA256 = neutral_eval.CANDIDATE_CHECKPOINT_SHA256
CANDIDATE_CHECKPOINT = neutral_eval.CANDIDATE_CHECKPOINT
BASELINE_RECORD_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_baseline_20260928/full/records/candidate"
)
PHASE_CONFIG_KEYS = (
    "schema_version",
    "airspace_schema_version",
    "vehicle_dynamics_schema_version",
    "trajectory_schema_version",
    "vehicle_profile_id",
    "action_schema_version",
    "high_action_schema_version",
    "observation_schema_version",
    "dt",
    "physics_dt",
    "safety_sampling_dt",
    "high_level_enabled",
    "shield_enabled",
    "scheduler_mode",
    "teacher_enabled",
    "teacher_release_label_enabled",
    "task_priority_enabled",
    "low_level_priority_time_weight",
    "low_level_soc_urgency_time_weight",
)


class TaskNeutralPilotError(ValueError):
    """Raised for a pilot binding or execution contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaskNeutralPilotError(message)


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


def _model_digest(policy: Any) -> str:
    return str(rank_eval._model_digest(policy))


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


def _config_snapshot(config: Any) -> dict[str, Any]:
    return neutral_eval._config_snapshot(config)


def _digest_tensors(items: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(items):
        digest.update(str(name).encode("utf-8"))
        value = items[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy().tobytes()
        elif isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value).tobytes()
        else:
            value = repr(value).encode("utf-8")
        digest.update(value)
    return digest.hexdigest()


def _gate_reference_digest(reference: GateReference | None) -> str | None:
    if reference is None:
        return None
    return _digest_tensors({
        "observations": reference.observations,
        "reference_logits": reference.reference_logits,
        "reference_probs": reference.reference_probs,
        "weights": reference.weights,
    })


def _actor_world_chunks(steps: Sequence[StudentStepV42]) -> list[np.ndarray]:
    return [
        np.asarray(step.world, dtype=np.float32)[np.asarray(step.actor_mask, dtype=bool)]
        for step in steps
        if bool(np.asarray(step.actor_mask, dtype=bool).any())
    ]


def actor_world_observation_weights(
    steps: Sequence[StudentStepV42],
) -> tuple[np.ndarray, np.ndarray]:
    """Return actor observations with equal weight per world and per actor.

    A world contributes one unit of mass, split equally over its actor rows;
    the world masses are then split equally over all worlds in the supplied
    rollout collection.
    """

    worlds = _actor_world_chunks(steps)
    if not worlds:
        raise TaskNeutralPilotError("training rollouts contain no actor observations")
    world_count = len(worlds)
    observations = np.concatenate(worlds, axis=0).astype(np.float32, copy=False)
    weights = np.concatenate([
        np.full(len(world), 1.0 / (world_count * len(world)), dtype=np.float32)
        for world in worlds
    ])
    _require(observations.ndim == 2 and observations.shape[1] == 88,
             "actor observations must be 88D")
    _require(np.isfinite(observations).all() and np.isfinite(weights).all(),
             "actor observations and weights must be finite")
    _require(np.all(weights > 0.0) and np.isclose(float(weights.sum()), 1.0),
             "actor observation weights must be positive and normalized")
    return observations, weights


def _configure_trainable_policy(policy: Any, *, learning_rate: float = LEARNING_RATE) -> dict[str, Any]:
    model = getattr(policy, "model", None)
    _require(model is not None, "candidate policy has no model")
    names: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES)
        if parameter.requires_grad:
            names.append(name)
    _require(names, "pilot trainable scope is empty")
    _require(all(any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES) for name in names),
             "pilot trainable scope contains an unexpected parameter")
    _require(any(name.startswith("gate_adapter.") for name in names),
             "gate_adapter is missing from pilot trainable scope")
    _require(any(name.startswith("critic_encoder.") for name in names),
             "critic_encoder is missing from pilot trainable scope")
    _require(any(name.startswith("critic_head.") for name in names),
             "critic_head is missing from pilot trainable scope")
    frozen_parameters = sorted(name for name, parameter in model.named_parameters() if not parameter.requires_grad)
    _require(all(name.startswith(FROZEN_PREFIXES) for name in frozen_parameters),
             "pilot frozen scope contains an unexpected parameter")
    _require(hasattr(model, "gate_temperature"), "calibrated gate_temperature buffer is missing")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    policy.optimizer = torch.optim.Adam(trainable, lr=float(learning_rate))
    policy.model.train()
    return {
        "trainable_parameter_names": sorted(names),
        "frozen_parameter_names": frozen_parameters,
        "learning_rate": float(learning_rate),
        "gate_temperature_frozen": not bool(getattr(model.gate_temperature, "requires_grad", False)),
    }


def _frozen_bitwise_check(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    _require(set(before) == set(after), "model state keys changed during pilot update")
    frozen = [name for name in before if not any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES)]
    changed = [name for name in frozen if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu())]
    proposal = [name for name in before if name.startswith("proposal_head.")]
    proposal_changed = [name for name in proposal if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu())]
    trainable = [name for name in before if any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES)]
    trainable_changed = [name for name in trainable if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu())]
    return {
        "frozen_parameter_bitwise_unchanged": not changed,
        "frozen_changed_names": sorted(changed),
        "proposal_bitwise_unchanged": not proposal_changed,
        "proposal_changed_names": sorted(proposal_changed),
        "trainable_changed": bool(trainable_changed),
        "trainable_changed_names": sorted(trainable_changed),
        "gate_temperature_bitwise_unchanged": "gate_temperature" not in changed,
    }


def _actor_gate_release_mean(policy: Any, steps: Sequence[StudentStepV42]) -> float | None:
    chunks = [
        np.asarray(step.world, dtype=np.float32)[np.asarray(step.actor_mask, dtype=bool)]
        for step in steps
        if bool(np.asarray(step.actor_mask, dtype=bool).any())
    ]
    if not chunks:
        return None
    observations = np.concatenate(chunks, axis=0)
    with torch.no_grad():
        distributions, _ = policy.model.distributions(
            torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
        )
        return float(distributions.probs[:, 1].mean().detach().cpu())


def _critic_discounted_mc_error(steps: Sequence[StudentStepV42], *, gamma: float) -> dict[str, Any]:
    target = np.zeros(len(steps), dtype=np.float64)
    running = 0.0
    for index in range(len(steps) - 1, -1, -1):
        row = steps[index]
        running = float(row.reward) + float(gamma) * (0.0 if row.terminated else running)
        target[index] = running
    values = np.asarray([float(row.value) for row in steps], dtype=np.float64)
    error = target - values
    finite = np.isfinite(target) & np.isfinite(values) & np.isfinite(error)
    if not finite.any():
        return {"count": 0, "mean_signed": None, "mean_abs": None, "max_abs": None}
    return {
        "count": int(finite.sum()),
        "mean_signed": float(error[finite].mean()),
        "mean_abs": float(np.abs(error[finite]).mean()),
        "max_abs": float(np.abs(error[finite]).max()),
    }


def _gae_diagnostics(steps: Sequence[StudentStepV42], *, gamma: float, gae_lambda: float) -> dict[str, Any]:
    advantages, returns = team_gae(list(steps), gamma, gae_lambda)
    actor_mask = np.asarray([bool(row.actor_mask.any()) for row in steps], dtype=bool)
    normalized = np.asarray(advantages, dtype=np.float32).copy()
    if actor_mask.any():
        actor_values = normalized[actor_mask]
        normalized = (normalized - actor_values.mean()) / (actor_values.std() + 1e-8)
    _require(np.isfinite(advantages).all() and np.isfinite(returns).all() and np.isfinite(normalized).all(),
             "rollout GAE is non-finite")
    return {
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "advantages": advantages.tolist(),
        "returns": returns.tolist(),
        "normalized_advantages": normalized.tolist(),
        "actor_steps": int(actor_mask.sum()),
    }


def _build_gate_reference(
    source_policy: Any,
    observations: np.ndarray,
    weights: np.ndarray,
) -> GateReference:
    return GateReference.from_policy(
        source_policy,
        observations,
        weights,
        coefficient=0.1,
        max_kl=0.05,
        soft_kl_target=0.02,
        soft_kl_gain=400.0,
    )


def _training_record(
    *, policy: Any, checkpoint: Path, config: Any, scenario: str, seed: int, output: Path,
    policy_seed: int | None = None,
) -> tuple[list[StudentStepV42], dict[str, Any]]:
    """Run exactly one training rollout and retain its full lifecycle record."""

    before = _model_digest(policy)
    summary: Mapping[str, Any] | None = None
    action_tape: Any = None
    lifecycle: Mapping[str, Any] | None = None
    requests = int(neutral_eval.EXPECTED_REQUESTS[scenario])
    effective_policy_seed = int(seed if policy_seed is None else policy_seed)
    try:
        rollout_kwargs = {
            "seed": int(seed),
            "max_steps": int(round(float(config.max_simulation_seconds) / float(config.dt))),
            "output": output,
            "deterministic": False,
            "collect_labels": False,
        }
        if policy_seed is not None:
            rollout_kwargs["policy_seed"] = effective_policy_seed
        steps, raw_summary = student_rollout_v4_2.rollout(policy, config, **rollout_kwargs)
        _require(isinstance(raw_summary, Mapping), "training rollout summary must be a mapping")
        summary = dict(raw_summary)
        action_tape = summary.get("action_tape")
        _require(action_tape is not None, "training rollout action_tape is missing")
        lifecycle = terminal_lifecycle_metrics.terminal_lifecycle_metrics(
            config, action_tape, neutral_eval.LOW_SOC_THRESHOLD,
        )
        after = _model_digest(policy)
        record: dict[str, Any] = {
            "label": "task_neutral_pilot_training",
            "scenario": str(scenario),
            "seed": int(seed),
            "policy_seed": effective_policy_seed,
            "status": "completed" if before == after else "failed",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
            "requests": requests,
            "summary": dict(summary),
            "rollout_summary": dict(summary),
            "action_tape": action_tape,
            "lifecycle": dict(lifecycle),
            "outcomes": dict(lifecycle.get("outcomes") or {}),
            "weights_unchanged": bool(before == after),
            "weights_digest_before": before,
            "weights_digest_after": after,
            "weights": {"before": before, "after": after, "unchanged": bool(before == after)},
            "teacher_enabled": False,
            "teacher_called": False,
            "training_or_update_called": False,
            "PPO_ALLOWED": False,
        }
        normalized = rank_eval._normalize_runtime_record(
            record, scenario=scenario, seed=seed, requests=requests,
        )
        return list(steps), dict(normalized)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        after = _model_digest(policy)
        failure = rank_eval._failure_record(
            scenario=scenario, seed=seed, checkpoint=checkpoint, requests=requests,
            error=exc, summary=summary, weights_unchanged=(before == after),
            weights_before=before, weights_after=after, action_tape=action_tape,
            lifecycle=lifecycle, policy_seed=policy_seed,
        )
        return [], dict(rank_eval._normalize_runtime_record(
            failure, scenario=scenario, seed=seed, requests=requests,
        ))


def _record_annotations(record: dict[str, Any], config: Any) -> dict[str, Any]:
    metrics = neutral_eval._record_metrics(record)
    record.update({
        "config": _config_snapshot(config),
        "task_priority_enabled": False,
        "low_level_priority_time_weight": 0.0,
        "model_unchanged": record.get("weights_unchanged"),
        "safety": dict(metrics.get("safety") or {}),
        "reward": dict(metrics.get("reward") or {}),
        "wait": {
            key: metrics.get(key)
            for key in ("mean_wait", "p95_wait", "max_wait", "lowSOC_wait", "priority_wait")
        },
        "technical_ok": metrics.get("technical_ok"),
    })
    return record


def _training_terminal_failure(record: Mapping[str, Any], steps: Sequence[StudentStepV42]) -> str | None:
    summary = record.get("summary") if isinstance(record.get("summary"), Mapping) else {}
    if record.get("error"):
        return str(record["error"].get("message", "training rollout error")) if isinstance(record["error"], Mapping) else str(record["error"])
    if not steps or summary.get("terminal") is not True or steps[-1].terminated is not True:
        return "training rollout did not reach a terminal episode"
    if record.get("weights_unchanged") is not True:
        return "training rollout changed model weights"
    if bool(record.get("safety_nonzero")):
        return "training rollout reported a safety violation"
    metrics = neutral_eval._record_metrics(record)
    if int(metrics.get("failed_energy", 0) or 0) > 0:
        return "training rollout reported failed_energy"
    if metrics.get("lifecycle_valid") is not True:
        return "training lifecycle ledger is incomplete"
    return None


def _update_failure(stats: Any) -> str | None:
    if not isinstance(stats, Mapping):
        return "policy.update did not return a mapping"
    if not _state_finite(stats):
        return "policy.update returned non-finite statistics"
    if int(stats.get("updates", 0) or 0) <= 0:
        return "policy.update accepted zero updates"
    if int(stats.get("accepted_updates", stats.get("updates", 0)) or 0) <= 0:
        return "policy.update accepted zero updates"
    return None


def _training_progress(
    output: Path, *, status: str, phase: str, training_processed: int = 0,
    evaluation_processed: int = 0, update_calls: int = 0,
    error: str | None = None, check_only: bool = False,
) -> None:
    value: dict[str, Any] = {
        "stage": STAGE,
        "status": str(status),
        "phase": str(phase),
        "episodes_processed": int(training_processed),
        "episodes_expected": 0 if check_only else len(TRAINING_CELLS),
        "evaluation_processed": int(evaluation_processed),
        "evaluation_cells_expected": 0 if check_only else 6,
        "check_only": bool(check_only),
        "full_evaluation": bool(not check_only),
        "PPO_ALLOWED": False,
        "development_pilot_authorized": bool(DEVELOPMENT_PILOT_AUTHORIZED),
        "training_update_calls": int(update_calls),
        "teacher_called": False,
        "heldout_consumed": False,
    }
    if error:
        value["error"] = str(error)
    _write_json(output / "progress.json", value)


def _base_report(*, status: str, output: Path, check_only: bool, stop_reason: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "stage": STAGE,
        "status": str(status),
        "check_only": bool(check_only),
        "full_evaluation": bool(not check_only and status == "COMPLETED"),
        "PPO_ALLOWED": False,
        "development_pilot_authorized": bool(DEVELOPMENT_PILOT_AUTHORIZED),
        "formal_admission": {"PPO_ALLOWED": False, "P3_QUALIFIED": False, "P4_ALLOWED": False},
        "teacher_called": False,
        "heldout_consumed": False,
        "training_update_calls": 0,
        "actions": {"old_formal_runner": False, "teacher": False, "heldout": False, "optimizer_tuning": False},
        "artifacts": {"output_dir": str(output)},
        "checks": {},
        "deviations": [],
        "unresolved": [],
    }
    if stop_reason:
        value["stop_reason"] = str(stop_reason)
    return value


def _write_terminal_report(output: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    _write_json(output / "report.json", report)
    return dict(report)


def _final_eval_fatal_reason(record: Mapping[str, Any]) -> str | None:
    if record.get("error"):
        error = record.get("error")
        return str(error.get("message", "evaluation runtime error")) if isinstance(error, Mapping) else str(error)
    if record.get("weights_unchanged") is not True:
        return "evaluation changed model weights"
    if bool(record.get("safety_nonzero")):
        return "evaluation reported a safety violation"
    if neutral_eval._record_metrics(record).get("technical_ok") is not True:
        return "evaluation technical fields are incomplete"
    return None


def _roundtrip_pilot(
    pilot_path: Path,
    policy: Any,
    config: Any,
    stage_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    before = _model_digest(policy)
    save_checkpoint(
        pilot_path,
        policy,
        config,
        stage=STAGE,
        curriculum_id="task_neutral_development",
        task_priority_enabled=False,
        low_level_priority_time_weight=0.0,
        PPO_ALLOWED=False,
        development_pilot_authorized=True,
        stage_metadata=dict(stage_metadata),
        neutral_config_snapshot=_config_snapshot(config),
    )
    return_payload: dict[str, Any] = {}
    # Loading through the registered loader validates the calibrated model and
    # leaves the extra stage metadata available for the explicit payload check.
    roundtrip_policy, payload = rank_eval._load_policy(pilot_path, config)
    payload_check = _validate_pilot_payload(payload, config)
    return_payload["model_digest"] = _model_digest(roundtrip_policy)
    return_payload["model_exact"] = bool(return_payload["model_digest"] == before)
    return_payload["payload_stage"] = payload.get("stage") if isinstance(payload, Mapping) else None
    return_payload["payload_curriculum_id"] = payload.get("curriculum_id") if isinstance(payload, Mapping) else None
    return_payload["payload_task_priority_enabled"] = payload.get("task_priority_enabled") if isinstance(payload, Mapping) else None
    return_payload["payload_priority_weight"] = payload.get("low_level_priority_time_weight") if isinstance(payload, Mapping) else None
    return_payload["payload_config_check"] = payload_check
    _require(return_payload["model_exact"] is True, "pilot checkpoint model roundtrip changed weights")
    _require(return_payload["payload_task_priority_enabled"] is False,
             "pilot checkpoint neutral task flag is missing")
    _require(float(return_payload["payload_priority_weight"]) == 0.0,
             "pilot checkpoint neutral priority weight is missing")
    return_payload["checkpoint_sha256"] = _sha256(pilot_path)
    return_payload["exact"] = True
    return return_payload


def _load_binding(config: Any) -> tuple[Path, Any, dict[str, Any]]:
    resolved, locator_report = neutral_eval._checkpoint_from_report(
        CANDIDATE_CHECKPOINT, EXPECTED_CANDIDATE_SHA256,
    )
    actual_sha = _sha256(resolved)
    _require(actual_sha == EXPECTED_CANDIDATE_SHA256, "candidate checkpoint SHA mismatch")
    policy, payload = rank_eval._load_policy(resolved, config)
    binding = {
        "declared_checkpoint": str(CANDIDATE_CHECKPOINT),
        "resolved_checkpoint": str(resolved),
        "checkpoint_sha256": actual_sha,
        "expected_checkpoint_sha256": EXPECTED_CANDIDATE_SHA256,
        "locator_report": locator_report,
        "model_digest": _model_digest(policy),
        "payload_keys": sorted(payload) if isinstance(payload, Mapping) else [],
        "baseline_scope": "fixed_task_neutral_baseline_20260928/full/records/candidate",
    }
    return resolved, policy, binding


def _baseline_path(scenario: str, seed: int) -> Path:
    declared = BASELINE_RECORD_ROOT / f"{scenario}_{seed}.json"
    resolved = neutral_eval._synced_path(declared)
    _require(resolved.is_file(), f"task-neutral baseline record missing: {declared}")
    return resolved


def _baseline_checkpoint_sha(record: Mapping[str, Any]) -> str | None:
    for key in ("checkpoint_sha256", "checkpoint_sha256_expected", "candidate_checkpoint_sha256"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _validate_pilot_payload(payload: Mapping[str, Any], config: Any) -> dict[str, Any]:
    _require(payload.get("stage") == STAGE, "pilot checkpoint stage metadata mismatch")
    _require(payload.get("curriculum_id") == "task_neutral_development",
             "pilot checkpoint curriculum metadata mismatch")
    _require(payload.get("task_priority_enabled") is False,
             "pilot checkpoint task_priority_enabled metadata mismatch")
    _require(float(payload.get("low_level_priority_time_weight")) == 0.0,
             "pilot checkpoint priority weight metadata mismatch")
    snapshot = payload.get("neutral_config_snapshot")
    _require(isinstance(snapshot, Mapping), "pilot checkpoint neutral config snapshot is missing")
    current = _config_snapshot(config)
    phase_mismatch = {
        key: {"checkpoint": snapshot.get(key), "evaluation": current.get(key)}
        for key in PHASE_CONFIG_KEYS
        if snapshot.get(key) != current.get(key)
    }
    _require(not phase_mismatch, f"pilot checkpoint phase config mismatch: {sorted(phase_mismatch)}")
    return {
        "stage": payload.get("stage"),
        "curriculum_id": payload.get("curriculum_id"),
        "task_priority_enabled": payload.get("task_priority_enabled"),
        "low_level_priority_time_weight": payload.get("low_level_priority_time_weight"),
        "phase_config_checked": list(PHASE_CONFIG_KEYS),
        "phase_config_match": True,
    }


def _validate_initial_warmup_payload(
    payload: Mapping[str, Any], policy: Any, config: Any,
    *, expected_parent_sha256: str = EXPECTED_CANDIDATE_SHA256,
) -> dict[str, Any]:
    """Validate a critic-only warmup before applying its model state."""
    _require(payload.get("stage") == critic_warmup.STAGE,
             "warmup checkpoint stage metadata mismatch")
    _require(payload.get("curriculum_id") == critic_warmup.CURRICULUM_ID,
             "warmup checkpoint curriculum metadata mismatch")
    _require(payload.get("task_priority_enabled") is False,
             "warmup checkpoint task_priority_enabled metadata mismatch")
    _require(float(payload.get("low_level_priority_time_weight")) == 0.0,
             "warmup checkpoint priority weight metadata mismatch")
    _require(payload.get("parent_checkpoint_sha256") == expected_parent_sha256,
             "warmup parent checkpoint SHA mismatch")
    snapshots = payload.get("neutral_config_snapshots")
    _require(isinstance(snapshots, Mapping), "warmup neutral config snapshots are missing")
    snapshot = snapshots.get("dense24")
    _require(isinstance(snapshot, Mapping), "warmup dense24 config snapshot is missing")
    _require(dict(snapshot) == _config_snapshot(config),
             "warmup dense24 neutral config snapshot mismatch")
    state = payload.get("model_state")
    _require(isinstance(state, Mapping), "warmup model state is missing")
    current = policy.model.state_dict()
    _require(set(state) == set(current), "warmup model state keys mismatch")
    _require(all(isinstance(state[name], torch.Tensor) and bool(torch.isfinite(state[name]).all().item())
                 for name in state),
             "warmup model state contains non-finite values")
    noncritic_names = [name for name in current if not name.startswith(critic_warmup.CRITIC_PREFIXES)]
    _require(all(torch.equal(current[name].detach().cpu(), state[name].detach().cpu())
                 for name in noncritic_names),
             "warmup changed non-critic state relative to candidate")
    invariance = payload.get("actor_invariance")
    _require(isinstance(invariance, Mapping), "warmup actor invariance metadata is missing")
    _require(invariance.get("non_critic_state_bitwise_unchanged") is True,
             "warmup actor invariance metadata is false")
    return {
        "stage": payload.get("stage"),
        "curriculum_id": payload.get("curriculum_id"),
        "parent_checkpoint_sha256": payload.get("parent_checkpoint_sha256"),
        "neutral_config_match": True,
        "non_critic_state_bitwise_unchanged": True,
        "critic_parameter_names": list(invariance.get("critic_parameter_names", [])),
    }


def _load_initial_warmup(
    path: str | Path, policy: Any, config: Any,
) -> tuple[Path, dict[str, Any]]:
    resolved = neutral_eval._synced_path(Path(path))
    _require(resolved.is_file(), f"warmup checkpoint missing: {path}")
    payload = torch.load(resolved, map_location=policy.device, weights_only=False)
    _require(isinstance(payload, Mapping), "warmup checkpoint payload must be a mapping")
    check = _validate_initial_warmup_payload(payload, policy, config)
    policy.model.load_state_dict(payload["model_state"], strict=True)
    policy.model.eval()
    check["checkpoint"] = str(resolved)
    check["checkpoint_sha256"] = _sha256(resolved)
    check["model_state_loaded"] = True
    check["model_digest_after_load"] = _model_digest(policy)
    return resolved, check


def _load_baseline_records(neutral_configs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    paths: list[str] = []
    for cell in neutral_eval._evaluation_cells():
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        path = _baseline_path(scenario, seed)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskNeutralPilotError(f"invalid task-neutral baseline record: {path}") from exc
        _require(isinstance(record, Mapping), f"baseline record must be a mapping: {path}")
        record = dict(record)
        key = (seed, scenario)
        _require(key not in seen, f"duplicate baseline cell: {key}")
        seen.add(key)
        _require(str(record.get("scenario")) == scenario and int(record.get("seed")) == seed,
                 f"baseline cell mismatch: {path}")
        _require(_baseline_checkpoint_sha(record) == EXPECTED_CANDIDATE_SHA256,
                 f"baseline checkpoint SHA mismatch: {path}")
        _require(record.get("task_priority_enabled") is False,
                 f"baseline task-neutral flag missing: {path}")
        _require(float(record.get("low_level_priority_time_weight", 1.0)) == 0.0,
                 f"baseline priority weight mismatch: {path}")
        _require(record.get("weights_unchanged") is True and record.get("technical_ok") is True,
                 f"baseline technical state is incomplete: {path}")
        _require(record.get("lifecycle_valid") is True and not record.get("error"),
                 f"baseline lifecycle state is incomplete: {path}")
        config_snapshot = record.get("config")
        _require(isinstance(config_snapshot, Mapping), f"baseline config snapshot missing: {path}")
        _require(dict(config_snapshot) == _config_snapshot(neutral_configs[scenario]),
                 f"baseline config mismatch for {scenario}/{seed}")
        records.append(record)
        paths.append(str(path))
    expected = {(int(item["seed"]), str(item["scenario"])) for item in neutral_eval._evaluation_cells()}
    _require(seen == expected, "baseline record set does not contain exactly six cells")
    return records, {
        "root": str(BASELINE_RECORD_ROOT),
        "resolved_records": paths,
        "records": len(records),
        "checkpoint_sha256": EXPECTED_CANDIDATE_SHA256,
        "old_records_consumed": False,
        "summary": neutral_eval.aggregate_arm(records),
    }


def _write_training_record(output: Path, index: int, record: Mapping[str, Any]) -> None:
    episode = output / "training" / f"episode_{index:02d}"
    _write_json(episode / "record.json", record)


def _write_eval_record(output: Path, record: Mapping[str, Any]) -> None:
    _write_json(output / "evaluation" / "records" / f"{record['scenario']}_{record['seed']}.json", record)


def _pilot_comparison(
    baseline_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = {(int(item["seed"]), str(item["scenario"])): item for item in baseline_records}
    evaluation = {(int(item["seed"]), str(item["scenario"])): item for item in evaluation_records}
    rows: list[dict[str, Any]] = []
    for cell in neutral_eval._evaluation_cells():
        key = (int(cell["seed"]), str(cell["scenario"]))
        rows.append({
            "seed": key[0],
            "scenario": key[1],
            "split": str(cell.get("split", "development")),
            "baseline": neutral_eval._record_metrics(baseline[key]) if key in baseline else None,
            "evaluation": neutral_eval._record_metrics(evaluation[key]) if key in evaluation else None,
        })
    return {
        "baseline_scope": "fixed_preexisting_six_cell_candidate_baseline",
        "baseline_cells": len(baseline),
        "evaluation_cells": len(evaluation),
        "baseline_summary": neutral_eval.aggregate_arm(list(baseline.values())),
        "evaluation_summary": neutral_eval.aggregate_arm(list(evaluation.values())),
        "cells": rows,
    }


def _final_evaluation(
    *, output: Path, pilot_path: Path, neutral_configs: Mapping[str, Any], checkpoint_sha: str,
    progress_start: int,
) -> tuple[list[dict[str, Any]], str | None]:
    records: list[dict[str, Any]] = []
    for offset, cell in enumerate(neutral_eval._evaluation_cells()):
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        config = neutral_configs[scenario]
        policy, payload = rank_eval._load_policy(pilot_path, config)
        payload_check = _validate_pilot_payload(payload, config)
        record_output = output / "evaluation" / "rollouts" / f"{scenario}_{seed}"
        record = rank_eval._runtime_record(
            policy=policy,
            checkpoint=pilot_path,
            config=config,
            scenario=scenario,
            seed=seed,
            output=record_output,
            smoke=False,
        )
        record = _record_annotations(dict(record), config)
        record.update({"arm": "pilot", "split": str(cell.get("split", "development")),
                       "pilot_checkpoint_sha256": checkpoint_sha,
                       "pilot_checkpoint_config_check": payload_check})
        records.append(record)
        _write_eval_record(output, record)
        reason = _final_eval_fatal_reason(record)
        _training_progress(output, status="FAILED" if reason else "RUNNING",
                           training_processed=progress_start,
                           evaluation_processed=offset + 1,
                           update_calls=progress_start,
                           phase="evaluation", error=reason)
        if reason:
            return records, f"evaluation/{scenario}/{seed}: {reason}"
    return records, None


def run(*, output_dir: str | Path, check_only: bool = False,
        initial_checkpoint: str | Path | None = None) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"pilot output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _training_progress(output, status="STARTING", training_processed=0,
                       evaluation_processed=0, update_calls=0,
                       phase="preflight", check_only=check_only)
    report = _base_report(status="STARTING", output=output, check_only=check_only)
    configs: dict[str, Any] = {}
    neutral_configs: dict[str, Any] = {}
    training_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    stop_reason: str | None = None
    policy: Any = None
    source_policy: Any = None
    resolved_checkpoint: Path | None = None
    current_checkpoint: Path | None = None
    initial_binding: dict[str, Any] | None = None
    binding: dict[str, Any] = {}
    baseline_records: list[dict[str, Any]] = []
    baseline_binding: dict[str, Any] = {}
    training_manifest: list[dict[str, Any]] = []
    gate_reference_digests: list[str | None] = []
    try:
        configs, neutral_configs, runtime_protocol = neutral_eval._build_neutral_inputs()
        del runtime_protocol
        _require(all(getattr(config, "task_priority_enabled", None) is False and
                     float(getattr(config, "low_level_priority_time_weight", 1.0)) == 0.0
                     for config in neutral_configs.values()),
                 "neutral configs do not carry the required task flags")
        resolved_checkpoint, policy, binding = _load_binding(neutral_configs["dense24"])
        current_checkpoint = resolved_checkpoint
        if initial_checkpoint is not None:
            parent_model_digest = binding.get("model_digest")
            current_checkpoint, initial_binding = _load_initial_warmup(
                initial_checkpoint, policy, neutral_configs["dense24"],
            )
            binding["initial_checkpoint"] = dict(initial_binding)
            binding["parent_model_digest"] = parent_model_digest
            binding["model_digest_after_initial_checkpoint"] = initial_binding.get("model_digest_after_load")
        baseline_records, baseline_binding = _load_baseline_records(neutral_configs)
        binding["config_snapshots"] = {
            scenario: {"original": _config_snapshot(configs[scenario]),
                       "neutral": _config_snapshot(neutral_configs[scenario]),
                       "changed_fields": neutral_eval._config_delta(configs[scenario], neutral_configs[scenario])}
            for scenario in sorted(neutral_configs)
        }
        binding["source_checkpoint"] = "fixed candidate initial policy used as q reference"
        binding["PPO_ALLOWED"] = False
        binding["development_pilot_authorized"] = True
        binding["update_budget"] = dict(UPDATE_KWARGS)
        binding["training_cells"] = [dict(cell) for cell in TRAINING_CELLS]
        binding["evaluation_cells"] = [dict(cell) for cell in neutral_eval._evaluation_cells()]
        report["binding"] = binding
        report["baseline_binding"] = baseline_binding
        report["checks"]["config_neutralized"] = True
        report["checks"]["candidate_weight_sha256"] = binding["checkpoint_sha256"]
        report["checks"]["baseline_binding"] = binding["baseline_scope"]
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        stop_reason = str(exc)
        report["status"] = "FAILED"
        report["checks"]["preflight_exception"] = {"type": type(exc).__name__, "message": str(exc)}
        report["artifacts"]["report"] = str(output / "report.json")
        report["unresolved"] = [stop_reason]
        _write_terminal_report(output, report)
        _training_progress(output, status="FAILED", training_processed=0,
                           evaluation_processed=0, update_calls=0,
                           phase="preflight", error=stop_reason, check_only=check_only)
        return report

    if check_only:
        report["status"] = "CHECK_ONLY_OK"
        report["full_evaluation"] = False
        report["technical_ok"] = None
        report["artifacts"].update({"report": str(output / "report.json"), "progress": str(output / "progress.json")})
        _write_terminal_report(output, report)
        _training_progress(output, status="CHECK_ONLY_OK", training_processed=0,
                           evaluation_processed=0, update_calls=0,
                           phase="preflight", check_only=True)
        return report

    try:
        # The q policy is a separate load and is never attached to the pilot
        # optimizer or mutated by an update.
        source_policy, _source_payload = rank_eval._load_policy(resolved_checkpoint, neutral_configs["dense24"])
        source_q_digest = _model_digest(source_policy)
        scope = _configure_trainable_policy(policy)
        report["checks"]["trainable_scope"] = scope
        report["checks"]["formal_policy_flags"] = {"PPO_ALLOWED": False, "teacher_called": False}
        actor_worlds: list[np.ndarray] = []

        for episode_index, cell in enumerate(TRAINING_CELLS, start=1):
            scenario = str(cell["scenario"])
            seed = int(cell["seed"])
            _training_progress(output, status="RUNNING", training_processed=episode_index - 1,
                               evaluation_processed=0, update_calls=episode_index - 1,
                               phase=f"training_episode_{episode_index}")
            steps, record = _training_record(
                policy=policy,
                checkpoint=current_checkpoint,
                config=neutral_configs[scenario],
                scenario=scenario,
                seed=seed,
                output=output / "training" / f"episode_{episode_index:02d}" / "rollout",
            )
            record = _record_annotations(record, neutral_configs[scenario])
            reason = _training_terminal_failure(record, steps)
            if reason:
                record.update({"update_attempted": False, "update_stats": None,
                               "stop_reason": reason, "training_or_update_called": False})
                record["record_path"] = str(output / "training" / f"episode_{episode_index:02d}" / "record.json")
                _write_training_record(output, episode_index, record)
                training_records.append(record)
                training_manifest.append({
                    "episode": episode_index,
                    "path": str(output / "training" / f"episode_{episode_index:02d}" / "record.json"),
                    "scenario": scenario,
                    "seed": seed,
                    "update_stats": None,
                    "update_diagnostics": None,
                })
                stop_reason = f"training/{scenario}/{seed}: {reason}"
                break

            episode_observations, episode_weights = actor_world_observation_weights(steps)
            del episode_observations, episode_weights
            actor_worlds.extend(_actor_world_chunks(steps))
            all_observations = np.concatenate(actor_worlds, axis=0)
            # Every observed actor world receives equal mass across all three
            # training episodes; actors within one world split that mass.
            world_weights = np.concatenate([
                np.full(len(world), 1.0 / (len(actor_worlds) * len(world)), dtype=np.float32)
                for world in actor_worlds
            ])
            world_weights = world_weights / float(world_weights.sum())
            policy.gate_reference = _build_gate_reference(source_policy, all_observations, world_weights)
            gate_reference_digest = _gate_reference_digest(policy.gate_reference)
            gate_reference_digests.append(gate_reference_digest)
            before_update_state = copy.deepcopy(policy.model.state_dict())
            reference_before = gate_reference_digest
            p_release_before = _actor_gate_release_mean(policy, steps)
            diagnostics = {
                "gae": _gae_diagnostics(steps, gamma=UPDATE_KWARGS["gamma"], gae_lambda=UPDATE_KWARGS["gae_lambda"]),
                "critic_discounted_mc_error": _critic_discounted_mc_error(steps, gamma=UPDATE_KWARGS["gamma"]),
            }
            try:
                np.random.seed(seed)
                stats = policy.update(list(steps), **UPDATE_KWARGS)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                stats = {"error": {"type": type(exc).__name__, "message": str(exc)}, "updates": 0}
            update_reason = _update_failure(stats)
            after_update_state = copy.deepcopy(policy.model.state_dict())
            freeze = _frozen_bitwise_check(before_update_state, after_update_state)
            if not freeze["frozen_parameter_bitwise_unchanged"] or not freeze["proposal_bitwise_unchanged"]:
                update_reason = update_reason or "frozen or proposal parameters changed"
            if _model_digest(source_policy) != source_q_digest:
                update_reason = update_reason or "source q policy changed"
            p_release_after = _actor_gate_release_mean(policy, steps)
            diagnostics.update({
                "actor_gate_p_release_mean_before": p_release_before,
                "actor_gate_p_release_mean_after": p_release_after,
                "source_q_model_digest_before": source_q_digest,
                "source_q_model_digest_after": _model_digest(source_policy),
                "gate_reference_digest": gate_reference_digest,
                "gate_reference_digest_before_update": reference_before,
                "frozen_bitwise": freeze,
            })
            record.update({
                "update_attempted": True,
                "update_stats": stats,
                "update_diagnostics": diagnostics,
                "training_or_update_called": True,
                "gate_reference": {
                    "observation_count": int(len(all_observations)),
                    "weight_sum": float(world_weights.sum()),
                    "world_count": int(len(actor_worlds)),
                    "coefficient": 0.1,
                    "max_kl": 0.05,
                    "soft_kl_target": 0.02,
                    "soft_kl_gain": 400.0,
                    "digest": gate_reference_digest,
                    "source_q_digest": source_q_digest,
                },
                "stop_reason": update_reason,
                "update_rng_seed": seed,
            })
            if update_reason is None:
                episode_checkpoint = output / "training" / f"episode_{episode_index:02d}" / "pilot_after_update.pt"
                try:
                    save_checkpoint(
                        episode_checkpoint,
                        policy,
                        neutral_configs[scenario],
                        stage=STAGE,
                        curriculum_id="task_neutral_development",
                        task_priority_enabled=False,
                        low_level_priority_time_weight=0.0,
                        PPO_ALLOWED=False,
                        development_pilot_authorized=True,
                        episode=int(episode_index),
                        update_rng_seed=int(seed),
                        neutral_config_snapshot=_config_snapshot(neutral_configs[scenario]),
                    )
                    record["episode_checkpoint"] = str(episode_checkpoint)
                    record["episode_checkpoint_sha256"] = _sha256(episode_checkpoint)
                    current_checkpoint = episode_checkpoint
                except BaseException as exc:
                    update_reason = f"episode checkpoint save failed: {exc}"
                    record["stop_reason"] = update_reason
            record["record_path"] = str(output / "training" / f"episode_{episode_index:02d}" / "record.json")
            _write_training_record(output, episode_index, record)
            training_records.append(record)
            training_manifest.append({
                "episode": episode_index,
                "path": str(output / "training" / f"episode_{episode_index:02d}" / "record.json"),
                "scenario": scenario,
                "seed": seed,
                "update_stats": record.get("update_stats"),
                "update_diagnostics": record.get("update_diagnostics"),
            })
            if update_reason:
                stop_reason = f"training/{scenario}/{seed}: {update_reason}"
                break
            report["training_update_calls"] = episode_index

        if stop_reason is None:
            stage_metadata = {
                "stage": STAGE,
                "curriculum_id": "task_neutral_development",
                "task_priority_enabled": False,
                "low_level_priority_time_weight": 0.0,
                "PPO_ALLOWED": False,
                "development_pilot_authorized": True,
                "updates": len(training_records),
                "update_budget": dict(UPDATE_KWARGS),
                "training_cells": [dict(cell) for cell in TRAINING_CELLS],
            }
            _require(len(training_records) == 3 and report["training_update_calls"] == 3,
                     "pilot update budget did not complete exactly three updates")
            pilot_path = output / "pilot.pt"
            roundtrip = _roundtrip_pilot(pilot_path, policy, neutral_configs["dense24"], stage_metadata)
            report["artifacts"]["pilot_checkpoint"] = str(pilot_path)
            report["checks"]["checkpoint_roundtrip"] = roundtrip
            evaluation_records, stop_reason = _final_evaluation(
                output=output,
                pilot_path=pilot_path,
                neutral_configs=neutral_configs,
                checkpoint_sha=roundtrip["checkpoint_sha256"],
                progress_start=len(training_records),
            )
            report["evaluation"] = {
                "records": evaluation_records,
                "summary": neutral_eval.aggregate_arm(evaluation_records),
                "cells_expected": 6,
                "cells_processed": len(evaluation_records),
                "runtime_helper": "rank_eval._runtime_record",
            }
            report["comparison"] = _pilot_comparison(baseline_records, evaluation_records)
            report["training"] = {
                "records": training_manifest,
                "episodes_processed": len(training_records),
                "updates_attempted": sum(1 for item in training_records if item.get("update_attempted")),
                "updates_accepted": sum(
                    int((item.get("update_stats") or {}).get("accepted_updates", 0) or 0)
                    for item in training_records
                    if isinstance(item.get("update_stats"), Mapping)
                ),
                "reference_digests": gate_reference_digests,
            }
            report["checks"]["source_q_fixed"] = all(
                item.get("update_diagnostics", {}).get("source_q_model_digest_after") == source_q_digest
                for item in training_records
            )
            report["checks"]["evaluation_six_cells_once"] = len(evaluation_records) == 6
            report["technical_ok"] = None if stop_reason else all(
                neutral_eval._record_metrics(record).get("technical_ok") is True
                for record in evaluation_records
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        stop_reason = str(exc)
        report["checks"]["exception"] = {"type": type(exc).__name__, "message": str(exc)}

    report.setdefault("training", {
        "records": training_manifest,
        "episodes_processed": len(training_records),
        "updates_attempted": sum(1 for item in training_records if item.get("update_attempted")),
        "updates_accepted": sum(
            int((item.get("update_stats") or {}).get("accepted_updates", 0) or 0)
            for item in training_records
            if isinstance(item.get("update_stats"), Mapping)
        ),
        "reference_digests": gate_reference_digests,
    })
    if stop_reason is not None:
        report["status"] = "FAILED"
        report["full_evaluation"] = False
        report["unresolved"] = [stop_reason]
    else:
        report["status"] = "COMPLETED"
        report["full_evaluation"] = True
    report["training_update_calls"] = len([item for item in training_records if item.get("update_attempted")])
    report["artifacts"].update({
        "progress": str(output / "progress.json"),
        "report": str(output / "report.json"),
        "training_records": str(output / "training"),
        "evaluation_records": str(output / "evaluation"),
    })
    _write_terminal_report(output, report)
    update_calls = sum(1 for item in training_records if item.get("update_attempted"))
    _training_progress(output, status=report["status"], training_processed=len(training_records),
                       evaluation_processed=len(evaluation_records), update_calls=update_calls,
                       phase="complete" if report["status"] == "COMPLETED" else "failed", error=stop_reason)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(
            output_dir=args.output_dir,
            check_only=args.check_only,
            initial_checkpoint=args.initial_checkpoint,
        )
    except (TaskNeutralPilotError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({
        "stage": STAGE,
        "status": result.get("status"),
        "training_update_calls": result.get("training_update_calls", 0),
        "technical_ok": result.get("technical_ok"),
        "PPO_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"CHECK_ONLY_OK", "COMPLETED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
