"""Measure per-episode gate-adapter gradients on fixed task-neutral tapes.

This diagnostic consumes only the six already-recorded episodic action tapes.
It replays their requested actions to recover the production transition rows,
but never samples an action, calls PPO, or steps an optimizer.  The analysis
uses the frozen actor from the candidate checkpoint and the critic-only
warmup checkpoint supplied by the episodic-learning protocol.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.algo import student_ppo_v4_2 as ppo
from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_joint_logprob
from terminal_marl.training import diagnose_p3_ppo_credit_signal as credit
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import run_p3_task_neutral_pilot as pilot


STAGE = "P3_TASK_NEUTRAL_EPISODE_GRADIENTS_V1"
ROOT = Path("/home/user/training_outputs/terminal/p3_task_neutral_episodic_learning_20260929")
TRAINING_ROOT = ROOT / "full" / "training" / "batch_01" / "training"
WARMUP_CHECKPOINT = ROOT / "warmup_scaled" / "warmup.pt"
CANDIDATE_CHECKPOINT_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
WARMUP_CHECKPOINT_SHA256 = "a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710"
EXPECTED_STEPS = (725, 864, 699, 729, 724, 756)
EXPECTED_ENV_SEEDS = (877101, 877102, 877103, 877104, 877105, 877106)
EXPECTED_POLICY_SEEDS = (878201, 878202, 878203, 878204, 878205, 878206)
EXPECTED_REQUESTS = 24
MAX_REPLAY_STEPS = 1000
MAX_TOTAL_REPLAY_STEPS = 10800
EXPECTED_TOTAL_STEPS = sum(EXPECTED_STEPS)
GAMMA = 1.0
GAE_LAMBDA = 1.0
REWARD_SCALE = 0.01
OBSERVATION_DIM = 88


class EpisodeGradientError(ValueError):
    """Raised when the fixed diagnostic input/output contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EpisodeGradientError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeGradientError(f"invalid JSON input: {path}") from exc
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and np.isfinite(array).all(), "stats require a finite vector")
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _mean_abs_error(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)).mean())


def _mse(predicted: np.ndarray, target: np.ndarray) -> float:
    delta = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.square(delta).mean())


def _episode_path(index: int) -> Path:
    return TRAINING_ROOT / f"episode_{index:02d}" / "record.json"


def _preflight() -> dict[str, Any]:
    """Read and validate only the fixed records/tapes; no environment is made."""
    cells: list[dict[str, Any]] = []
    total_steps = 0
    for index, (expected_steps, env_seed, policy_seed) in enumerate(
        zip(EXPECTED_STEPS, EXPECTED_ENV_SEEDS, EXPECTED_POLICY_SEEDS, strict=True), start=1
    ):
        declared_record = _episode_path(index)
        record_path = neutral_eval._synced_path(declared_record)
        record = _read_json(record_path)
        _require(int(record.get("seed", -1)) == env_seed, f"episode {index} env seed mismatch")
        _require(int(record.get("policy_seed", -1)) == policy_seed, f"episode {index} policy seed mismatch")
        _require(str(record.get("scenario")) == "dense24", f"episode {index} scenario mismatch")
        _require(int(record.get("requests", -1)) == EXPECTED_REQUESTS, f"episode {index} request count mismatch")
        _require(record.get("technical_ok") is True, f"episode {index} technical flag is incomplete")
        _require(record.get("weights_unchanged") is True, f"episode {index} weight flag is incomplete")
        _require(record.get("lifecycle_valid") is True, f"episode {index} lifecycle flag is incomplete")
        _require(str(record.get("checkpoint_sha256", "")).lower() == WARMUP_CHECKPOINT_SHA256,
                 f"episode {index} warmup checkpoint SHA mismatch")
        declared_tape = Path(str(record.get("action_tape", "")))
        _require(str(record.get("action_tape", "")), f"episode {index} action tape is missing")
        tape_path = neutral_eval._synced_path(declared_tape)
        tape_payload, tape_rows = credit._load_tape(tape_path)
        _require(len(tape_rows) == expected_steps, f"episode {index} tape step count mismatch")
        _require(len(tape_rows) <= MAX_REPLAY_STEPS, f"episode {index} tape exceeds replay limit")
        record_steps = int(record.get("rollout_summary", {}).get("steps", expected_steps))
        _require(record_steps == expected_steps, f"episode {index} record step count mismatch")
        cells.append({
            "index": index,
            "record_path": str(record_path),
            "tape_path": str(tape_path),
            "declared_tape_path": str(declared_tape),
            "record": record,
            "tape_payload": tape_payload,
            "tape_rows": tape_rows,
            "steps": expected_steps,
            "requests": EXPECTED_REQUESTS,
            "env_seed": env_seed,
            "policy_seed": policy_seed,
        })
        total_steps += expected_steps
    _require(len(cells) == 6, "exactly six episode inputs are required")
    _require(total_steps == EXPECTED_TOTAL_STEPS, "fixed total step count mismatch")
    return {
        "root": str(ROOT),
        "training_root": str(TRAINING_ROOT),
        "cells": cells,
        "episodes": len(cells),
        "total_steps": total_steps,
        "total_requests": len(cells) * EXPECTED_REQUESTS,
        "max_replay_steps": MAX_TOTAL_REPLAY_STEPS,
    }


def _scaled_step(step: ppo.StudentStepV42, reward_scale: float) -> ppo.StudentStepV42:
    return dataclasses.replace(step, reward=float(step.reward) * float(reward_scale))


def _credit_for_episode(
    steps: Sequence[ppo.StudentStepV42], *, gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA, reward_scale: float = REWARD_SCALE,
) -> dict[str, Any]:
    """Return production GAE plus complete-episode MC targets for one tape."""
    raw_rewards = np.asarray([float(step.reward) for step in steps], dtype=np.float64)
    scaled_steps = [_scaled_step(step, reward_scale) for step in steps]
    scaled_rewards = np.asarray([float(step.reward) for step in scaled_steps], dtype=np.float64)
    raw_mc = credit.mc_returns(raw_rewards, gamma)
    scaled_mc = credit.mc_returns(scaled_rewards, gamma)
    advantage, returns = ppo.team_gae(scaled_steps, gamma=gamma, gae_lambda=gae_lambda)
    actor_mask = np.asarray([bool(step.actor_mask.any()) for step in scaled_steps], dtype=bool)
    normalized = credit.actor_step_normalize(advantage, actor_mask)
    values = np.asarray([float(step.value) for step in scaled_steps], dtype=np.float64)
    return {
        "steps": scaled_steps,
        "raw_rewards": raw_rewards,
        "scaled_rewards": scaled_rewards,
        "mc_raw": raw_mc,
        "mc_scaled": scaled_mc,
        "advantages": np.asarray(advantage, dtype=np.float32),
        "returns": np.asarray(returns, dtype=np.float32),
        "actor_mask": actor_mask,
        "normalized": np.asarray(normalized["values"], dtype=np.float32),
        "normalization": {key: value for key, value in normalized.items() if key != "values"},
        "critic_error": {
            "mse": _mse(values, scaled_mc),
            "mae": _mean_abs_error(values, scaled_mc),
            "value_stats": _stats(values),
            "target_stats": _stats(scaled_mc),
        },
    }


def compute_credit_batch(
    episodes: Sequence[Sequence[ppo.StudentStepV42]], *, gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA, reward_scale: float = REWARD_SCALE,
) -> dict[str, Any]:
    """Compute one shared actor normalization across complete episode tapes."""
    _require(len(episodes) > 0, "credit batch must contain an episode")
    credits = [_credit_for_episode(rows, gamma=gamma, gae_lambda=gae_lambda, reward_scale=reward_scale)
               for rows in episodes]
    all_steps = [step for item in credits for step in item["steps"]]
    advantages, returns = ppo.team_gae(all_steps, gamma=gamma, gae_lambda=gae_lambda)
    actor_mask = np.asarray([bool(step.actor_mask.any()) for step in all_steps], dtype=bool)
    normalized = credit.actor_step_normalize(advantages, actor_mask)
    offset = 0
    for episode_index, item in enumerate(credits, start=1):
        count = len(item["steps"])
        item["advantages_shared"] = np.asarray(advantages[offset:offset + count], dtype=np.float32)
        item["returns_shared"] = np.asarray(returns[offset:offset + count], dtype=np.float32)
        item["normalized_shared"] = np.asarray(normalized["values"][offset:offset + count], dtype=np.float32)
        item["normalization_shared"] = {key: value for key, value in normalized.items() if key != "values"}
        item["mc_suffix_match"] = bool(np.allclose(
            item["returns_shared"], item["mc_scaled"], rtol=1e-5, atol=1e-4,
        ))
        _require(item["mc_suffix_match"], f"episode {episode_index} MC suffix mismatch")
        offset += count
    _require(offset == len(all_steps), "credit episode offsets are misaligned")
    return {
        "episodes": credits,
        "steps": all_steps,
        "advantages": np.asarray(advantages, dtype=np.float32),
        "returns": np.asarray(returns, dtype=np.float32),
        "normalized": np.asarray(normalized["values"], dtype=np.float32),
        "actor_mask": actor_mask,
        "normalization": {key: value for key, value in normalized.items() if key != "values"},
        "actor_decisions": int(sum(int(step.actor_mask.sum()) for step in all_steps)),
    }


def _gate_parameters(policy: Any) -> list[torch.nn.Parameter]:
    parameters = [parameter for name, parameter in policy.model.named_parameters()
                  if str(name).startswith("gate_adapter.")]
    _require(parameters, "gate_adapter parameters are missing")
    return parameters


def _actor_rows(steps: Sequence[ppo.StudentStepV42], normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = [step for step in steps if bool(step.actor_mask.any())]
    if not active:
        return (np.zeros((0, OBSERVATION_DIM), dtype=np.float32),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
    observations = np.concatenate([step.world[step.actor_mask] for step in active]).astype(np.float32, copy=False)
    gates = np.concatenate([step.requested_gate[step.actor_mask] for step in active]).astype(np.int64, copy=False)
    proposals = np.concatenate([step.requested_proposal[step.actor_mask] for step in active]).astype(np.int64, copy=False)
    weights = np.concatenate([
        np.full(int(step.actor_mask.sum()), float(normalized[index]), dtype=np.float32)
        for index, step in enumerate(steps) if bool(step.actor_mask.any())
    ])
    _require(len(observations) == len(weights), "actor observations and advantages are misaligned")
    return observations, gates, proposals, weights


def _loss_gradient(
    policy: Any, steps: Sequence[ppo.StudentStepV42], normalized: np.ndarray,
    denominator: int, parameters: Sequence[torch.nn.Parameter],
) -> tuple[np.ndarray, np.ndarray]:
    observations, gates, proposals, weights = _actor_rows(steps, normalized)
    zero = np.zeros(sum(int(parameter.numel()) for parameter in parameters), dtype=np.float64)
    if len(observations) == 0:
        return zero, np.zeros(0, dtype=np.float64)
    _require(denominator > 0, "actor denominator must be positive")
    tensor_obs = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    tensor_gates = torch.as_tensor(gates, dtype=torch.long, device=policy.device)
    tensor_proposals = torch.as_tensor(proposals, dtype=torch.long, device=policy.device)
    tensor_weights = torch.as_tensor(weights, dtype=torch.float32, device=policy.device)
    gate_dist, proposal_dist = policy.model.distributions(tensor_obs)
    _, _, joint = hierarchical_joint_logprob(gate_dist, proposal_dist, tensor_gates, tensor_proposals)
    loss = -(tensor_weights * joint).sum() / float(denominator)
    gradients = torch.autograd.grad(loss, tuple(parameters), allow_unused=True)
    flat = torch.cat([
        (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
        for gradient, parameter in zip(gradients, parameters, strict=True)
    ])
    probabilities = gate_dist.probs[:, 1].detach().cpu().numpy().astype(np.float64)
    return flat.detach().cpu().numpy().astype(np.float64), probabilities


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def gradient_statistics(vectors: Sequence[Sequence[float]] | np.ndarray,
                        aggregate: Sequence[float] | np.ndarray | None = None) -> dict[str, Any]:
    """Summarize per-episode vectors without retaining their raw tensor values."""
    values = np.asarray(vectors, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] > 0 and np.isfinite(values).all(),
             "gradient vectors must be a finite nonempty matrix")
    total = values.sum(axis=0)
    norms = np.linalg.norm(values, axis=1)
    matrix = [[_cosine(values[i], values[j]) for j in range(values.shape[0])]
              for i in range(values.shape[0])]
    leave_one_out = [_cosine(total - values[i], total) for i in range(values.shape[0])]
    result: dict[str, Any] = {
        "episode_count": int(values.shape[0]),
        "vector_dimension": int(values.shape[1]),
        "episode_gradient_norms": [float(value) for value in norms],
        "cosine_matrix": matrix,
        "cosine_episode_to_total": [_cosine(values[i], total) for i in range(values.shape[0])],
        "cosine_leave_one_out_to_total": leave_one_out,
        "total_gradient_norm": float(np.linalg.norm(total)),
        "sum_episode_gradient_norms": float(norms.sum()),
        "coherence": float(np.linalg.norm(total) / norms.sum()) if norms.sum() else 0.0,
        "mean_vector_norm": float(np.linalg.norm(values.mean(axis=0))),
        "vector_standard_error": float(
            np.sqrt(np.square(values - values.mean(axis=0)).sum() / (values.shape[0] * (values.shape[0] - 1)))
        ) if values.shape[0] > 1 else 0.0,
        "sum_vector": total,
    }
    if aggregate is not None:
        actual = np.asarray(aggregate, dtype=np.float64)
        _require(actual.shape == total.shape and np.isfinite(actual).all(), "aggregate gradient shape mismatch")
        delta = actual - total
        result["aggregate_check"] = {
            "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "l2_error": float(np.linalg.norm(delta)),
            "allclose": bool(np.allclose(actual, total, rtol=1e-5, atol=1e-6)),
        }
    return result


def _gradient_analysis(policy: Any, credit_batch: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _gate_parameters(policy)
    flags = {parameter: bool(parameter.requires_grad) for parameter in policy.model.parameters()}
    before_digest = pilot._model_digest(policy)
    try:
        for parameter in policy.model.parameters():
            parameter.requires_grad_(False)
        for parameter in parameters:
            parameter.requires_grad_(True)
        denominator = int(credit_batch["actor_decisions"])
        vectors: list[np.ndarray] = []
        release_probabilities: list[np.ndarray] = []
        for item in credit_batch["episodes"]:
            vector, probabilities = _loss_gradient(
                policy, item["steps"], item["normalized_shared"], denominator, parameters,
            )
            vectors.append(vector)
            release_probabilities.append(probabilities)
        all_steps = list(credit_batch["steps"])
        aggregate, aggregate_probabilities = _loss_gradient(
            policy, all_steps, credit_batch["normalized"], denominator, parameters,
        )
    finally:
        for parameter, requires_grad in flags.items():
            parameter.requires_grad_(requires_grad)
    after_digest = pilot._model_digest(policy)
    _require(before_digest == after_digest, "gradient diagnostic changed model state")
    matrix = gradient_statistics(vectors, aggregate)
    # The aggregate probabilities duplicate the six episode rows and are not
    # needed for quantiles; use the per-episode values only when available.
    probabilities = np.concatenate([item for item in release_probabilities if len(item)]) if any(
        len(item) for item in release_probabilities) else np.zeros(0, dtype=np.float64)
    matrix.update({
        "model_digest_before": before_digest,
        "model_digest_after": after_digest,
        "model_unchanged": True,
        "gate_adapter_only": True,
        "release_probability": {
            "count": int(probabilities.size),
            "quantiles": ({
                "q0": float(np.quantile(probabilities, 0.0)),
                "q25": float(np.quantile(probabilities, 0.25)),
                "q50": float(np.quantile(probabilities, 0.5)),
                "q75": float(np.quantile(probabilities, 0.75)),
                "q100": float(np.quantile(probabilities, 1.0)),
            } if probabilities.size else None),
            "fraction_below_1e-4": float(np.mean(probabilities < 1e-4)) if probabilities.size else None,
            "fraction_above_1_minus_1e-4": float(np.mean(probabilities > 1.0 - 1e-4)) if probabilities.size else None,
        },
    })
    matrix.pop("sum_vector", None)
    return matrix


def _episode_report(item: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    steps = item["steps"]
    actor_mask = np.asarray(item["actor_mask"], dtype=bool)
    actor_steps = int(actor_mask.sum())
    normalized = np.asarray(item["normalized_shared"], dtype=np.float64)
    advantages = np.asarray(item["advantages_shared"], dtype=np.float64)
    return {
        "index": int(cell["index"]),
        "record_path": str(cell["record_path"]),
        "action_tape": str(cell["tape_path"]),
        "env_seed": int(cell["env_seed"]),
        "policy_seed": int(cell["policy_seed"]),
        "steps": len(steps),
        "requests": int(cell["requests"]),
        "actor_steps": actor_steps,
        "actor_decisions": int(sum(int(step.actor_mask.sum()) for step in steps)),
        "replay_exact": True,
        "terminal": bool(steps[-1].terminated) if steps else False,
        "episode_end": bool(steps[-1].episode_end) if steps else False,
        "mc_suffix_match": bool(item["mc_suffix_match"]),
        "mc_returns_raw": _stats(item["mc_raw"]),
        "mc_returns_reward_scaled": _stats(item["mc_scaled"]),
        "mc_returns_actor_normalized": _stats(
            credit.actor_step_normalize(item["mc_scaled"], actor_mask)["values"]
        ),
        "critic_error": dict(item["critic_error"]),
        "advantage_raw": _stats(advantages),
        "advantage_actor_normalized": _stats(normalized[actor_mask]),
        "shared_advantage_normalization": dict(item["normalization_shared"]),
        "release_rows": int(sum(int(np.sum(step.requested_gate[step.actor_mask] == 1)) for step in steps)),
    }


def _base_report(output_dir: Path, *, status: str, check_only: bool) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "status": status,
        "check_only": bool(check_only),
        "output_dir": str(output_dir),
        "PPO_ALLOWED": False,
        "policy_sampling": False,
        "optimizer_used": False,
        "teacher_used": False,
        "training_called": False,
        "scope": {
            "episodes": 6,
            "expected_total_steps": EXPECTED_TOTAL_STEPS,
            "max_replay_steps": MAX_TOTAL_REPLAY_STEPS,
            "requests_per_episode": EXPECTED_REQUESTS,
            "total_requests": 6 * EXPECTED_REQUESTS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "reward_scale": REWARD_SCALE,
            "complete_episode_mc": True,
            "heldout_used": False,
            "evaluation_used": False,
        },
        "artifacts": {"output_dir": str(output_dir)},
    }


def _failed_report(output_dir: Path, error: BaseException, *, processed: int, check_only: bool) -> dict[str, Any]:
    report = _base_report(output_dir, status="FAILED", check_only=check_only)
    report.update({
        "processed_episodes": int(processed),
        "error": {"type": type(error).__name__, "message": str(error)},
        "deviations": [],
        "unresolved": [],
    })
    _write_json(output_dir / "report.json", report)
    _write_json(output_dir / "progress.json", report)
    _write_json(output_dir / "summary.json", {
        "stage": STAGE,
        "status": "FAILED",
        "processed_episodes": int(processed),
        "error": report["error"],
    })
    return report


def run(output_dir: Path, *, check_only: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    _require(not output_dir.exists(), f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "progress.json", _base_report(output_dir, status="STARTING", check_only=check_only))
    processed_episodes = 0
    parent_rng = credit.terminal_credit._rng_snapshot()
    try:
        preflight = _preflight()
        report = _base_report(output_dir, status="CHECK_ONLY_OK" if check_only else "RUNNING", check_only=check_only)
        report["inputs"] = {
            "root": str(ROOT),
            "training_root": str(TRAINING_ROOT),
            "cells": [
                {key: cell[key] for key in ("index", "record_path", "tape_path", "env_seed", "policy_seed", "steps", "requests")}
                for cell in preflight["cells"]
            ],
            "total_steps": preflight["total_steps"],
            "total_requests": preflight["total_requests"],
        }
        report["checks"] = {
            "preflight_records": 6,
            "preflight_total_steps": preflight["total_steps"],
            "preflight_total_requests": preflight["total_requests"],
            "runtime_replay_calls": 0 if check_only else None,
            "runtime_env_reset": False if check_only else None,
            "runtime_optimizer_calls": 0,
        }
        _write_json(output_dir / "progress.json", report)
        if check_only:
            report["artifacts"].update({"report": str(output_dir / "report.json"), "progress": str(output_dir / "progress.json")})
            _write_json(output_dir / "report.json", report)
            _write_json(output_dir / "summary.json", {"stage": STAGE, "status": report["status"], "checks": report["checks"], "inputs": report["inputs"]})
            return report

        configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
        del configs, _runtime
        config = neutral_configs["dense24"]
        _, policy, binding = pilot._load_binding(config)
        candidate_digest = pilot._model_digest(policy)
        warmup_resolved, warmup_check = pilot._load_initial_warmup(
            WARMUP_CHECKPOINT, policy, config, expected_gamma=GAMMA, expected_reward_scale=REWARD_SCALE,
        )
        _require(_sha256(warmup_resolved).lower() == WARMUP_CHECKPOINT_SHA256, "warmup checkpoint SHA mismatch")
        warmup_digest = pilot._model_digest(policy)
        policy.model.eval()
        episodes: list[list[ppo.StudentStepV42]] = []
        replay_rows: list[dict[str, Any]] = []
        for cell in preflight["cells"]:
            _write_json(output_dir / "progress.json", {
                **report, "status": "RUNNING", "phase": "REPLAYING",
                "processed_episodes": processed_episodes, "processed_steps": int(sum(item["steps"] for item in replay_rows)),
            })
            replay_result = credit._capture_replay(
                policy, config, Path(cell["tape_path"]), cell["tape_rows"],
            )
            captures = replay_result["captures"]
            _require(bool(replay_result["replay"].get("exact")), f"episode {cell['index']} replay was not exact")
            _require(int(replay_result["replay"].get("steps_replayed", -1)) == int(cell["steps"]),
                     f"episode {cell['index']} replay count mismatch")
            _require(bool(replay_result["lifecycle"].get("terminal")), f"episode {cell['index']} did not terminate")
            _require(bool(replay_result["lifecycle"].get("all_requests_observed")),
                     f"episode {cell['index']} request lifecycle incomplete")
            _require(len(replay_result["lifecycle"].get("request_rows", [])) == EXPECTED_REQUESTS,
                     f"episode {cell['index']} request count mismatch after replay")
            steps = [credit._step_from_capture(row) for row in captures]
            _require(len(steps) == int(cell["steps"]), f"episode {cell['index']} capture count mismatch")
            _require(steps[-1].terminated and steps[-1].episode_end and steps[-1].next_value == 0.0,
                     f"episode {cell['index']} terminal boundary is incomplete")
            for step in steps:
                step.validate(OBSERVATION_DIM)
            episodes.append(steps)
            replay_rows.append({
                "index": int(cell["index"]), "steps": len(steps), "requests": EXPECTED_REQUESTS,
                "replay_exact": bool(replay_result["replay"].get("exact")),
                "lifecycle_terminal": bool(replay_result["lifecycle"].get("terminal")),
                "lifecycle_alignment": bool(replay_result["lifecycle"].get("ledger_alignment")),
            })
            processed_episodes += 1
            _write_json(output_dir / "progress.json", {
                **report, "status": "RUNNING", "phase": "REPLAYING",
                "processed_episodes": processed_episodes,
                "processed_steps": int(sum(item["steps"] for item in replay_rows)),
                "replay": replay_rows,
            })

        credit_batch = compute_credit_batch(episodes, gamma=GAMMA, gae_lambda=GAE_LAMBDA, reward_scale=REWARD_SCALE)
        gradient = _gradient_analysis(policy, credit_batch)
        _require(bool(gradient.get("aggregate_check", {}).get("allclose")),
                 "aggregate gradient does not match the sum of episode gradients")
        digest_after = pilot._model_digest(policy)
        _require(digest_after == warmup_digest, "model state changed during gradient analysis")
        episode_reports = [_episode_report(item, cell) for item, cell in zip(credit_batch["episodes"], preflight["cells"], strict=True)]
        report.update({
            "status": "COMPLETED",
            "phase": "ANALYZED",
            "processed_episodes": processed_episodes,
            "processed_steps": int(len(credit_batch["steps"])),
            "binding": {
                **binding,
                "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
                "candidate_model_digest": candidate_digest,
                "warmup_checkpoint": str(warmup_resolved),
                "warmup_checkpoint_sha256": WARMUP_CHECKPOINT_SHA256,
                "warmup_model_digest": warmup_digest,
                "warmup_validation": warmup_check,
            },
            "replay": {"episodes": replay_rows, "total_steps": len(credit_batch["steps"]),
                       "total_requests": 6 * EXPECTED_REQUESTS, "exact": all(row["replay_exact"] for row in replay_rows)},
            "credit": {
                "gamma": GAMMA, "gae_lambda": GAE_LAMBDA, "reward_scale": REWARD_SCALE,
                "actor_decisions": credit_batch["actor_decisions"],
                "shared_advantage_normalization": credit_batch["normalization"],
                "episodes": episode_reports,
            },
            "gradient": gradient,
            "gradient_objective": {
                "type": "initial_ratio_1_actor_surrogate",
                "denominator": "total_actor_decisions_across_six_episodes",
                "includes": ["gate_adapter_logprob", "shared_normalized_advantage"],
                "excludes": ["entropy", "critic_value_loss", "Adam", "PPO_epochs"],
            },
            "checks": {
                **report["checks"],
                "runtime_replay_calls": 6,
                "runtime_env_reset": True,
                "runtime_optimizer_calls": 0,
                "model_unchanged_after_replay_and_gradient": digest_after == warmup_digest,
                "model_digest_after": digest_after,
                "aggregate_gradient_matches_sum": gradient.get("aggregate_check", {}).get("allclose"),
                "mc_suffix_match_all": all(bool(item["mc_suffix_match"]) for item in credit_batch["episodes"]),
            },
            "artifacts": {
                "output_dir": str(output_dir), "report": str(output_dir / "report.json"),
                "progress": str(output_dir / "progress.json"), "summary": str(output_dir / "summary.json"),
            },
            "deviations": [],
            "unresolved": [],
        })
        _write_json(output_dir / "report.json", report)
        _write_json(output_dir / "summary.json", {
            "stage": STAGE, "status": report["status"], "processed_episodes": processed_episodes,
            "processed_steps": len(credit_batch["steps"]), "actor_decisions": credit_batch["actor_decisions"],
            "episode_reports": episode_reports, "gradient": gradient,
            "model_unchanged": report["checks"]["model_unchanged_after_replay_and_gradient"],
        })
        _write_json(output_dir / "progress.json", report)
        return report
    except BaseException as exc:
        return _failed_report(output_dir, exc, processed=processed_episodes, check_only=check_only)
    finally:
        credit.terminal_credit._restore_rng(parent_rng)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.output_dir, check_only=args.check_only)
    except BaseException as exc:
        print(json.dumps({"stage": STAGE, "status": "FAILED", "error": str(exc)}, sort_keys=True), flush=True)
        return 1
    print(json.dumps({"stage": STAGE, "status": result.get("status"), "output_dir": str(args.output_dir)}, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"COMPLETED", "CHECK_ONLY_OK"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
