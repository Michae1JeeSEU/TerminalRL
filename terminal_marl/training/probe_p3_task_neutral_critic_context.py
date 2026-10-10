"""Probe a privileged centralized-critic context on fixed task-neutral tapes.

This diagnostic compares the existing mean/max centralized critic with a
ten-feature context-expanded critic.  The actor is loaded from the fixed 3fe7
checkpoint, initialized with the fixed a82 critic warmup, and never updated.
The first larger-batch training cohort is replayed from its existing action
tapes; no policy action is sampled and no PPO update is called.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from terminal_marl.core.observations import pad_agent_sets
from terminal_marl.training import diagnose_p3_ppo_credit_signal as credit
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import run_p3_task_neutral_pilot as pilot


STAGE = "P3_TASK_NEUTRAL_CRITIC_CONTEXT_PROBE_V1"
SOURCE_CHECKPOINT = neutral_eval.CANDIDATE_CHECKPOINT
SOURCE_CHECKPOINT_SHA256 = neutral_eval.CANDIDATE_CHECKPOINT_SHA256
WARMUP_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_episodic_learning_20260929/warmup_scaled/warmup.pt"
)
WARMUP_CHECKPOINT_SHA256 = "a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710"
INPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_larger_batch_20260929/full/training/batch_01/training"
)
OBSERVATION_DIM = 88
EXPECTED_REQUESTS = 24
EXPECTED_EPISODES = 24
FIT_EPISODES = 18
VALIDATION_EPISODES = 6
EXPECTED_TOTAL_STEPS = 18560
MAX_REPLAY_STEPS = 1800
GAMMA = 1.0
REWARD_SCALE = 0.01
EPOCHS = 20
MINIBATCH_SIZE = 64
LEARNING_RATE = 3e-4
PERMUTATION_SEED = 879501
GRAD_CLIP = 1.0
CONTEXT_DIM = 10
CONTEXT_FEATURE_NAMES = (
    "sim_time_norm",
    "active_uav_count_norm",
    "admission_queue_count_norm",
    "service_uav_count_norm",
    "remaining_arrivals_norm",
    "queue_soc_mean",
    "queue_soc_min",
    "queue_external_wait_norm",
    "lz_service_remaining_sum_norm",
    "lz_service_remaining_max_norm",
)
CRITIC_PREFIXES = ("critic_encoder.", "critic_head.")


class CriticContextProbeError(ValueError):
    """Raised when fixed probe inputs or diagnostic invariants are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticContextProbeError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


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
        raise CriticContextProbeError(f"invalid JSON input: {path}") from exc
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _synced_path(path: Path) -> Path:
    return neutral_eval._synced_path(Path(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(str(name).encode("utf-8"))
        value = state[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy().tobytes()
        elif isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value).tobytes()
        else:
            value = repr(value).encode("utf-8")
        digest.update(value)
    return digest.hexdigest()


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(name): value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in state.items()
    }


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CriticContextProbeError(f"{name} must be finite") from exc
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def critic_context(env: Any) -> dict[str, float]:
    """Read the fixed ten pre-step privileged critic-only context features."""
    config = getattr(env, "config", None)
    _require(config is not None, "context env config is missing")
    now = _finite(getattr(env, "now", None), "env.now")
    horizon = _finite(getattr(config, "max_simulation_seconds", None), "config.max_simulation_seconds")
    _require(horizon > 0.0, "config.max_simulation_seconds must be positive")
    active = getattr(env, "active", None)
    _require(isinstance(active, Mapping), "env.active is missing")
    admission = getattr(env, "admission", None)
    queue_manager = getattr(admission, "queue", None)
    queue = getattr(queue_manager, "items", None)
    _require(queue is not None, "admission.queue.items is missing")
    service = getattr(env, "service_uavs", None)
    _require(service is not None, "env.service_uavs is missing")
    arrivals = getattr(env, "arrivals", None)
    generated = _finite(getattr(arrivals, "generated", None), "arrivals.generated") if arrivals is not None else None
    total = _finite(getattr(config, "total_requested_uavs", None), "config.total_requested_uavs")
    _require(generated is not None, "env.arrivals.generated is missing")
    zones_manager = getattr(env, "lz_manager", None)
    zones = getattr(zones_manager, "zones", None)
    _require(zones is not None, "env.lz_manager.zones is missing")

    queue_socs: list[float] = []
    queue_wait = 0.0
    for index, item in enumerate(queue):
        queue_socs.append(_finite(getattr(item, "soc", None), f"queue[{index}].soc"))
        queue_wait += _finite(getattr(item, "external_wait_time", None), f"queue[{index}].external_wait_time")
    service_values = [
        _finite(getattr(zone, "service_remaining", None), f"zones[{index}].service_remaining")
        for index, zone in enumerate(zones)
    ]
    if service_values:
        lz_sum = float(sum(service_values))
        lz_max = float(max(service_values))
        zone_count = float(len(service_values))
    else:
        lz_sum = lz_max = zone_count = 0.0
    queue_mean = float(np.mean(queue_socs)) if queue_socs else 0.0
    queue_min = float(np.min(queue_socs)) if queue_socs else 0.0
    values = {
        "sim_time_norm": now / horizon,
        "active_uav_count_norm": float(len(active)) / 24.0,
        "admission_queue_count_norm": float(len(queue)) / 24.0,
        "service_uav_count_norm": float(len(service)) / 24.0,
        "remaining_arrivals_norm": max(total - generated, 0.0) / 24.0,
        "queue_soc_mean": queue_mean,
        "queue_soc_min": queue_min,
        "queue_external_wait_norm": queue_wait / (24.0 * 300.0),
        "lz_service_remaining_sum_norm": 0.0 if zone_count == 0.0 else lz_sum / (zone_count * 120.0),
        "lz_service_remaining_max_norm": lz_max / 120.0,
    }
    _require(tuple(values) == CONTEXT_FEATURE_NAMES, "critic context feature order changed")
    for name, value in values.items():
        _finite(value, f"context.{name}")
    return {name: float(values[name]) for name in CONTEXT_FEATURE_NAMES}


def _context_array(context: Mapping[str, Any]) -> np.ndarray:
    _require(set(context) == set(CONTEXT_FEATURE_NAMES), "critic context keys changed")
    values = np.asarray([_finite(context[name], f"context.{name}") for name in CONTEXT_FEATURE_NAMES], dtype=np.float32)
    _require(values.shape == (CONTEXT_DIM,) and bool(np.isfinite(values).all()), "critic context is not finite 10D")
    return values


def _replace_first_linear(head: nn.Module, extra_inputs: int) -> nn.Module:
    result = copy.deepcopy(head)
    if isinstance(result, nn.Linear):
        old = result
        new = nn.Linear(old.in_features + extra_inputs, old.out_features, bias=old.bias is not None)
        with torch.no_grad():
            new.weight[:, :old.in_features].copy_(old.weight)
            new.weight[:, old.in_features:].zero_()
            if old.bias is not None:
                new.bias.copy_(old.bias)
        return new
    for index, module in enumerate(result.children()):
        if isinstance(module, nn.Linear):
            new = nn.Linear(module.in_features + extra_inputs, module.out_features, bias=module.bias is not None)
            with torch.no_grad():
                new.weight[:, :module.in_features].copy_(module.weight)
                new.weight[:, module.in_features:].zero_()
                if module.bias is not None:
                    new.bias.copy_(module.bias)
            if isinstance(result, nn.Sequential):
                result[index] = new
                return result
            raise CriticContextProbeError("critic head first Linear is not replaceable")
    raise CriticContextProbeError("critic head first Linear is missing")


class ContextExpandedCritic(nn.Module):
    """Mean/max critic with ten additional critic-only inputs."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.critic_encoder = copy.deepcopy(base_model.critic_encoder)
        self.critic_head = _replace_first_linear(base_model.critic_head, CONTEXT_DIM)

    @staticmethod
    def _pool(agent_sets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = agent_sets
        valid = mask.unsqueeze(-1)
        mean = (encoded * valid).sum(1) / valid.sum(1).clamp_min(1)
        maximum = encoded.masked_fill(~valid, float("-inf")).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return mean, maximum

    def critic(self, agent_sets: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        encoded = self.critic_encoder(agent_sets)
        mean, maximum = self._pool(encoded, mask)
        if context.ndim != 2 or context.shape[0] != agent_sets.shape[0] or context.shape[1] != CONTEXT_DIM:
            raise ValueError("critic context must be Bx10")
        return self.critic_head(torch.cat((mean, maximum, context), -1)).squeeze(-1)


def _padded_worlds(worlds: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(world, dtype=np.float32) for world in worlds]
    _require(all(array.ndim == 2 and array.shape[1] == OBSERVATION_DIM for array in arrays), "worlds must be 88D matrices")
    padded, valid = pad_agent_sets(arrays)
    if padded.shape[1] == 0:
        padded = np.zeros((len(arrays), 1, OBSERVATION_DIM), dtype=np.float32)
        valid = np.zeros((len(arrays), 1), dtype=bool)
    return padded, valid


def _legacy_predictions(model: nn.Module, worlds: Sequence[np.ndarray]) -> np.ndarray:
    padded, valid = _padded_worlds(worlds)
    device = next(model.parameters()).device
    with torch.no_grad():
        values = model.critic(
            torch.as_tensor(padded, dtype=torch.float32, device=device),
            torch.as_tensor(valid, dtype=torch.bool, device=device),
        )
    return values.detach().cpu().numpy().astype(np.float64)


def _context_predictions(model: ContextExpandedCritic, worlds: Sequence[np.ndarray], contexts: Sequence[np.ndarray]) -> np.ndarray:
    padded, valid = _padded_worlds(worlds)
    context = np.asarray(contexts, dtype=np.float32)
    _require(context.shape == (len(worlds), CONTEXT_DIM), "context batch shape mismatch")
    device = next(model.parameters()).device
    with torch.no_grad():
        values = model.critic(
            torch.as_tensor(padded, dtype=torch.float32, device=device),
            torch.as_tensor(valid, dtype=torch.bool, device=device),
            torch.as_tensor(context, dtype=torch.float32, device=device),
        )
    return values.detach().cpu().numpy().astype(np.float64)


def _legacy_predictions_with_grad(model: nn.Module, worlds: Sequence[np.ndarray]) -> torch.Tensor:
    padded, valid = _padded_worlds(worlds)
    device = next(model.parameters()).device
    return model.critic(
        torch.as_tensor(padded, dtype=torch.float32, device=device),
        torch.as_tensor(valid, dtype=torch.bool, device=device),
    )


def _context_predictions_with_grad(model: ContextExpandedCritic, worlds: Sequence[np.ndarray], contexts: Sequence[np.ndarray]) -> torch.Tensor:
    padded, valid = _padded_worlds(worlds)
    context = np.asarray(contexts, dtype=np.float32)
    _require(context.shape == (len(worlds), CONTEXT_DIM), "context batch shape mismatch")
    device = next(model.parameters()).device
    return model.critic(
        torch.as_tensor(padded, dtype=torch.float32, device=device),
        torch.as_tensor(valid, dtype=torch.bool, device=device),
        torch.as_tensor(context, dtype=torch.float32, device=device),
    )


def _metrics(prediction: np.ndarray, targets: np.ndarray, reward_scale: float = REWARD_SCALE) -> dict[str, Any]:
    _require(math.isfinite(float(reward_scale)) and float(reward_scale) > 0.0,
             "reward_scale must be finite and positive")
    prediction = np.asarray(prediction, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    _require(prediction.shape == targets.shape, "critic metric vectors are misaligned")
    if not len(targets):
        return {"count": 0, "mse": None, "mae": None, "normalized_mse": None, "normalized_mae": None,
                "raw_mse": None, "raw_mae": None, "explained_variance": None}
    error = prediction - targets
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    variance = float(np.var(targets))
    return {
        "count": int(len(targets)),
        "mse": mse,
        "mae": mae,
        "normalized_mse": mse,
        "normalized_mae": mae,
        "raw_mse": float(mse / (reward_scale ** 2)),
        "raw_mae": float(mae / reward_scale),
        "explained_variance": None if variance <= 1e-15 else float(1.0 - np.var(error) / variance),
    }


def _trajectory_metrics(
    prediction: np.ndarray,
    targets: np.ndarray,
    actor_world_mask: np.ndarray,
) -> dict[str, Any]:
    actor_world_mask = np.asarray(actor_world_mask, dtype=bool)
    return {
        "all_worlds": _metrics(prediction, targets),
        "actor_worlds": _metrics(prediction[actor_world_mask], targets[actor_world_mask]),
    }


def _critic_parameters(model: nn.Module) -> list[nn.Parameter]:
    values = [parameter for name, parameter in model.named_parameters()
              if str(name).startswith(CRITIC_PREFIXES)]
    _require(values, "critic parameter scope is empty")
    return values


def _fit_models(
    legacy: nn.Module,
    expanded: ContextExpandedCritic,
    episodes: Sequence[Mapping[str, Any]],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    rows = [
        (world, context, target)
        for episode in episodes
        for world, context, target in zip(episode["worlds"], episode["contexts"], episode["targets"], strict=True)
    ]
    _require(rows, "critic fit requires training rows")
    worlds = [row[0] for row in rows]
    contexts = np.asarray([row[1] for row in rows], dtype=np.float32)
    targets = np.asarray([row[2] for row in rows], dtype=np.float64)
    before_legacy = _legacy_predictions(legacy, worlds)
    before_expanded = _context_predictions(expanded, worlds, contexts)
    initial_prediction_max_abs_diff = float(
        np.max(np.abs(before_legacy - before_expanded))
    ) if before_legacy.size else 0.0
    _require(np.allclose(before_legacy, before_expanded, rtol=1e-6, atol=1e-6),
             "expanded critic initial prediction does not match legacy critic")
    legacy_params = _critic_parameters(legacy)
    expanded_params = _critic_parameters(expanded)
    legacy_param_ids = {id(parameter) for parameter in legacy_params}
    expanded_param_ids = {id(parameter) for parameter in expanded_params}
    for parameter in legacy.parameters():
        parameter.requires_grad = id(parameter) in legacy_param_ids
    for parameter in expanded.parameters():
        parameter.requires_grad = id(parameter) in expanded_param_ids
    legacy_optimizer = torch.optim.Adam(legacy_params, lr=LEARNING_RATE)
    expanded_optimizer = torch.optim.Adam(expanded_params, lr=LEARNING_RATE)
    rng = np.random.RandomState(PERMUTATION_SEED)
    for epoch in range(EPOCHS):
        permutation = rng.permutation(len(rows))
        for start in range(0, len(rows), MINIBATCH_SIZE):
            indices = permutation[start:start + MINIBATCH_SIZE]
            batch_worlds = [worlds[int(index)] for index in indices]
            batch_contexts = contexts[indices]
            batch_targets = torch.as_tensor(
                targets[indices], dtype=torch.float32, device=next(legacy.parameters()).device,
            )
            legacy_prediction = _legacy_predictions_with_grad(legacy, batch_worlds)
            expanded_prediction = _context_predictions_with_grad(expanded, batch_worlds, batch_contexts)
            legacy_loss = (legacy_prediction - batch_targets).square().mean()
            expanded_targets = batch_targets.to(device=next(expanded.parameters()).device)
            expanded_loss = (expanded_prediction - expanded_targets).square().mean()
            _require(bool(torch.isfinite(legacy_loss).item()) and bool(torch.isfinite(expanded_loss).item()),
                     "critic context fit produced non-finite MSE")
            legacy_optimizer.zero_grad(set_to_none=True)
            legacy_loss.backward()
            torch.nn.utils.clip_grad_norm_(legacy_params, GRAD_CLIP, error_if_nonfinite=True)
            legacy_optimizer.step()
            expanded_optimizer.zero_grad(set_to_none=True)
            expanded_loss.backward()
            torch.nn.utils.clip_grad_norm_(expanded_params, GRAD_CLIP, error_if_nonfinite=True)
            expanded_optimizer.step()
        if progress_callback is not None:
            progress_callback(epoch + 1, EPOCHS)
    legacy.eval()
    expanded.eval()
    after_legacy = _legacy_predictions(legacy, worlds)
    after_expanded = _context_predictions(expanded, worlds, contexts)
    return {
        "before": {"legacy": _metrics(before_legacy, targets), "context_expanded": _metrics(before_expanded, targets)},
        "after": {"legacy": _metrics(after_legacy, targets), "context_expanded": _metrics(after_expanded, targets)},
        "initial_prediction_max_abs_diff": initial_prediction_max_abs_diff,
        "epochs": EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "permutation_seed": PERMUTATION_SEED,
        "grad_clip": GRAD_CLIP,
        "optimizer_models": "separate_critic_optimizers_same_permutations",
        "shared_permutations": True,
    }


def _episode_path(index: int) -> Path:
    return _synced_path(INPUT_ROOT / f"episode_{index:02d}" / "record.json")


def _preflight() -> dict[str, Any]:
    checkpoint, locator = neutral_eval._checkpoint_from_report(SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256)
    warmup = _synced_path(WARMUP_CHECKPOINT)
    _require(warmup.is_file(), f"warmup checkpoint missing: {WARMUP_CHECKPOINT}")
    _require(_sha256(warmup) == WARMUP_CHECKPOINT_SHA256, "warmup checkpoint SHA mismatch")
    _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    config = neutral_configs["dense24"]
    cells: list[dict[str, Any]] = []
    total_steps = 0
    for index in range(1, EXPECTED_EPISODES + 1):
        path = _episode_path(index)
        record = _read_json(path)
        expected_seed = 879100 + index
        expected_policy_seed = 879200 + index
        _require(str(record.get("scenario")) == "dense24", f"episode {index} scenario mismatch")
        _require(int(record.get("seed", -1)) == expected_seed, f"episode {index} env seed mismatch")
        _require(int(record.get("policy_seed", -1)) == expected_policy_seed, f"episode {index} policy seed mismatch")
        _require(int(record.get("requests", -1)) == EXPECTED_REQUESTS, f"episode {index} request count mismatch")
        _require(record.get("task_priority_enabled") is False, f"episode {index} task-neutral flag mismatch")
        _require(record.get("technical_ok") is True and record.get("error") in (None, ""),
                 f"episode {index} technical record mismatch")
        _require(record.get("checkpoint_sha256") == WARMUP_CHECKPOINT_SHA256,
                 f"episode {index} warmup checkpoint SHA mismatch")
        config_snapshot = record.get("config")
        _require(isinstance(config_snapshot, Mapping) and dict(config_snapshot) == neutral_eval._config_snapshot(config),
                 f"episode {index} config snapshot mismatch")
        tape_value = record.get("action_tape")
        _require(isinstance(tape_value, str) and tape_value, f"episode {index} action tape missing")
        tape_path = _synced_path(Path(tape_value))
        _payload, tape_rows = credit._load_tape(tape_path, max_replay_steps=MAX_REPLAY_STEPS)
        total_steps += len(tape_rows)
        cells.append({
            "index": index, "split": "fit" if index <= FIT_EPISODES else "validation",
            "seed": expected_seed, "policy_seed": expected_policy_seed,
            "record_path": path, "record": record, "tape_path": tape_path,
            "tape_rows": tape_rows, "config": config,
        })
    _require(len(cells) == EXPECTED_EPISODES, "fixed first cohort must contain 24 episodes")
    _require(total_steps == EXPECTED_TOTAL_STEPS,
             f"fixed first cohort steps changed: {total_steps} != {EXPECTED_TOTAL_STEPS}")
    return {
        "checkpoint": checkpoint, "checkpoint_locator": locator,
        "checkpoint_sha256": _sha256(checkpoint), "warmup": warmup,
        "warmup_sha256": _sha256(warmup), "config": config, "cells": cells,
        "total_steps": total_steps,
    }


def _safety_check(env: Any) -> dict[str, Any]:
    snapshot = credit.terminal_credit._safety_snapshot(env)
    required = {}
    for name in credit.terminal_credit.REQUIRED_SAFETY_COUNTERS:
        value = snapshot.get(name)
        _require(value is not None and math.isfinite(float(value)) and float(value) == 0.0,
                 f"required safety counter invalid: {name}")
        required[name] = float(value)
    optional = snapshot.get("uncertified_execution")
    optional_present = optional is not None
    if optional_present:
        _require(math.isfinite(float(optional)) and float(optional) == 0.0,
                 "optional uncertified_execution counter is nonzero")
    return {
        "required_counters_zero": True,
        "required": required,
        "uncertified_execution_available": optional_present,
        "uncertified_execution_zero": True if optional_present else None,
    }


def _trajectory_contract(cell: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = replay.get("lifecycle") if isinstance(replay.get("lifecycle"), Mapping) else {}
    _require(replay.get("replay", {}).get("exact") is True, f"{cell['index']} action tape replay was not exact")
    _require(int(replay.get("replay", {}).get("steps_replayed", -1)) == len(cell["tape_rows"]),
             f"{cell['index']} replay step count mismatch")
    _require(lifecycle.get("terminal") is True, f"{cell['index']} replay was not terminal")
    _require(lifecycle.get("all_requests_observed") is True, f"{cell['index']} requests were incomplete")
    _require(lifecycle.get("ledger_alignment") is True, f"{cell['index']} ledger alignment failed")
    _require(len(lifecycle.get("request_rows", [])) == EXPECTED_REQUESTS,
             f"{cell['index']} request rows mismatch")
    return {
        "index": int(cell["index"]), "split": str(cell["split"]), "seed": int(cell["seed"]),
        "policy_seed": int(cell["policy_seed"]), "steps": len(cell["tape_rows"]),
        "requests": EXPECTED_REQUESTS, "replay_exact": True, "terminal": True,
        "ledger_alignment": True, "safety": _safety_check(replay["env"]),
    }


def _capture_trajectories(inputs: Mapping[str, Any], policy: Any,
                          *, progress_callback: Callable[[int, int], None] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectories: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    before_digest = rank_eval._model_digest(policy)
    for processed, cell in enumerate(inputs["cells"], start=1):
        replay = credit._capture_replay(
            policy, cell["config"], Path(cell["tape_path"]), cell["tape_rows"],
            state_observer=critic_context, max_replay_steps=MAX_REPLAY_STEPS,
        )
        captures = list(replay["captures"])
        _require(captures and len(captures) == len(cell["tape_rows"]), f"episode {cell['index']} capture length mismatch")
        worlds = [np.asarray(row["world"], dtype=np.float32) for row in captures]
        contexts = [_context_array(row.get("state_context", {})) for row in captures]
        rewards = np.asarray([float(row["reward"]) for row in captures], dtype=np.float64)
        ends = np.asarray([bool(row.get("episode_end") or row.get("terminated")) for row in captures], dtype=bool)
        _require(bool(ends[-1]) and int(ends.sum()) == 1, f"episode {cell['index']} terminal boundary mismatch")
        targets = credit.mc_returns(rewards, GAMMA) * REWARD_SCALE
        _require(np.isfinite(targets).all(), f"episode {cell['index']} MC target is non-finite")
        actor_world_mask = np.asarray([bool(np.asarray(row["actor_mask"], dtype=bool).any()) for row in captures], dtype=bool)
        reports.append(_trajectory_contract(cell, replay))
        trajectories.append({
            "index": int(cell["index"]), "split": str(cell["split"]), "seed": int(cell["seed"]),
            "policy_seed": int(cell["policy_seed"]), "worlds": worlds,
            "contexts": contexts, "rewards": rewards, "targets": targets,
            "actor_world_mask": actor_world_mask,
        })
        if progress_callback is not None:
            progress_callback(processed, len(inputs["cells"]))
    after_digest = rank_eval._model_digest(policy)
    _require(before_digest == after_digest, "source actor/model changed during fixed replay")
    return trajectories, reports


def _prediction_metrics_for_episodes(
    legacy: nn.Module, expanded: ContextExpandedCritic,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cache = _prediction_cache(legacy, expanded, episodes)
    return _episode_metrics_from_cache(cache, episodes)


def _prediction_cache(
    legacy: nn.Module, expanded: ContextExpandedCritic,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Keep raw predictions internal so split metrics retain exact weighting."""
    result: dict[str, Any] = {}
    for episode in episodes:
        worlds = episode["worlds"]
        contexts = episode["contexts"]
        legacy_prediction = _legacy_predictions(legacy, worlds)
        expanded_prediction = _context_predictions(expanded, worlds, contexts)
        result[f"episode_{int(episode['index']):02d}"] = {
            "index": int(episode["index"]), "split": str(episode["split"]),
            "seed": int(episode["seed"]), "policy_seed": int(episode["policy_seed"]),
            "legacy_prediction": legacy_prediction,
            "context_expanded_prediction": expanded_prediction,
        }
    return result


def _episode_metrics_from_cache(
    cache: Mapping[str, Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for episode in episodes:
        key = f"episode_{int(episode['index']):02d}"
        cached = cache[key]
        targets = np.asarray(episode["targets"], dtype=np.float64)
        actor_mask = np.asarray(episode["actor_world_mask"], dtype=bool)
        result[key] = {
            "index": int(episode["index"]), "split": str(episode["split"]),
            "seed": int(episode["seed"]), "policy_seed": int(episode["policy_seed"]),
            "steps": int(len(targets)), "actor_world_steps": int(actor_mask.sum()),
            "legacy": _trajectory_metrics(np.asarray(cached["legacy_prediction"]), targets, actor_mask),
            "context_expanded": _trajectory_metrics(
                np.asarray(cached["context_expanded_prediction"]), targets, actor_mask,
            ),
        }
    return result


def _aggregate_prediction_cache(
    cache: Mapping[str, Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    targets = np.concatenate(
        [np.asarray(episode["targets"], dtype=np.float64) for episode in episodes]
    ) if episodes else np.zeros(0, dtype=np.float64)
    actors = np.concatenate(
        [np.asarray(episode["actor_world_mask"], dtype=bool) for episode in episodes]
    ) if episodes else np.zeros(0, dtype=bool)
    legacy_prediction = np.concatenate(
        [np.asarray(cache[f"episode_{int(episode['index']):02d}"]["legacy_prediction"], dtype=np.float64)
         for episode in episodes]
    ) if episodes else np.zeros(0, dtype=np.float64)
    expanded_prediction = np.concatenate(
        [np.asarray(cache[f"episode_{int(episode['index']):02d}"]["context_expanded_prediction"], dtype=np.float64)
         for episode in episodes]
    ) if episodes else np.zeros(0, dtype=np.float64)
    return {
        "steps": int(len(targets)),
        "legacy": _trajectory_metrics(legacy_prediction, targets, actors),
        "context_expanded": _trajectory_metrics(expanded_prediction, targets, actors),
    }


def _aggregate_metrics(
    legacy: nn.Module, expanded: ContextExpandedCritic,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    worlds = [world for episode in episodes for world in episode["worlds"]]
    contexts = [context for episode in episodes for context in episode["contexts"]]
    targets = np.concatenate([np.asarray(episode["targets"], dtype=np.float64) for episode in episodes]) if episodes else np.zeros(0)
    actors = np.concatenate([np.asarray(episode["actor_world_mask"], dtype=bool) for episode in episodes]) if episodes else np.zeros(0, dtype=bool)
    legacy_prediction = _legacy_predictions(legacy, worlds)
    expanded_prediction = _context_predictions(expanded, worlds, contexts)
    return {
        "steps": int(len(targets)), "legacy": _trajectory_metrics(legacy_prediction, targets, actors),
        "context_expanded": _trajectory_metrics(expanded_prediction, targets, actors),
    }


def _save_critic(path: Path, model: nn.Module, *, kind: str) -> None:
    critic_state = {
        str(name): value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if str(name).startswith(CRITIC_PREFIXES)
    }
    _require(critic_state, "diagnostic critic state is empty")
    torch.save({
        "stage": STAGE, "diagnostic_only": True, "critic_kind": kind,
        "observation_dim": OBSERVATION_DIM,
        "context_feature_names": list(CONTEXT_FEATURE_NAMES),
        "state_dict": critic_state,
        "gamma": GAMMA, "reward_scale": REWARD_SCALE,
    }, path)


def run(output_dir: str | Path, *, check_only: bool = False) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"output directory must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    progress_path = output / "progress.json"
    completed_episodes = 0
    fit_started = False
    _write_json(progress_path, {"stage": STAGE, "status": "STARTING", "check_only": bool(check_only),
                                "processed_episodes": 0, "expected_episodes": EXPECTED_EPISODES})
    try:
        inputs = _preflight()
        config = inputs["config"]
        policy, _payload = rank_eval._load_policy(Path(inputs["checkpoint"]), config)
        _warmup_path, warmup_check = pilot._load_initial_warmup(
            inputs["warmup"], policy, config, expected_gamma=GAMMA,
            expected_reward_scale=REWARD_SCALE,
        )
        actor_before = {
            name: value.detach().cpu().clone()
            for name, value in policy.model.state_dict().items()
            if not str(name).startswith(CRITIC_PREFIXES)
        }
        if check_only:
            report = {
                "stage": STAGE, "status": "CHECK_ONLY_OK", "check_only": True,
                "binding": {
                    "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                    "warmup_checkpoint_sha256": WARMUP_CHECKPOINT_SHA256,
                    "warmup": warmup_check, "input_root": str(INPUT_ROOT),
                    "episodes": EXPECTED_EPISODES, "fit_episodes": FIT_EPISODES,
                    "validation_episodes": VALIDATION_EPISODES,
                    "total_steps": int(inputs["total_steps"]), "requests": EXPECTED_EPISODES * EXPECTED_REQUESTS,
                    "max_replay_steps_per_episode": MAX_REPLAY_STEPS,
                    "gamma": GAMMA, "reward_scale": REWARD_SCALE,
                    "context_feature_names": list(CONTEXT_FEATURE_NAMES),
                },
                "scope": {"env_reset": False, "env_resets": 0, "replay_calls": 0,
                          "fit_started": False, "optimizer_used": False, "ppo_updates": False,
                          "policy_sampling": False, "teacher_used": False, "actor_updated": False},
                "artifacts": {"progress": str(progress_path), "report": str(output / "report.json"),
                              "summary": str(output / "summary.json")},
            }
            _write_json(output / "report.json", report)
            _write_json(output / "summary.json", report)
            _write_json(progress_path, {"stage": STAGE, "status": "CHECK_ONLY_OK", "check_only": True,
                                        "processed_episodes": 0, "expected_episodes": EXPECTED_EPISODES})
            return report

        expanded = ContextExpandedCritic(policy.model).to(policy.device)
        empty_world = [np.zeros((0, OBSERVATION_DIM), dtype=np.float32)]
        empty_context = [np.zeros(CONTEXT_DIM, dtype=np.float32)]
        empty_legacy = _legacy_predictions(policy.model, empty_world)
        empty_expanded = _context_predictions(expanded, empty_world, empty_context)
        empty_initial_max_abs_diff = float(np.max(np.abs(empty_legacy - empty_expanded)))
        _require(np.allclose(empty_legacy, empty_expanded, rtol=1e-6, atol=1e-6),
                 "expanded critic empty-world initialization mismatch")

        def replay_progress(processed: int, expected: int) -> None:
            nonlocal completed_episodes
            completed_episodes = int(processed)
            _write_json(progress_path, {"stage": STAGE, "status": "REPLAYING", "check_only": False,
                                        "processed_episodes": int(processed), "expected_episodes": int(expected),
                                        "fit_started": False})

        trajectories, trajectory_reports = _capture_trajectories(inputs, policy, progress_callback=replay_progress)
        _require(len(trajectories) == EXPECTED_EPISODES, "replay episode count mismatch")
        train = [item for item in trajectories if item["split"] == "fit"]
        validation = [item for item in trajectories if item["split"] == "validation"]
        _require(len(train) == FIT_EPISODES and len(validation) == VALIDATION_EPISODES, "episode split changed")
        torch.save({
            "stage": STAGE, "gamma": GAMMA, "reward_scale": REWARD_SCALE,
            "episodes": [
                {key: value for key, value in episode.items()}
                for episode in trajectories
            ],
        }, output / "replay_cache.pt")
        before_cache = _prediction_cache(policy.model, expanded, trajectories)
        before_episode = _episode_metrics_from_cache(before_cache, trajectories)
        fit_started = True
        _write_json(progress_path, {"stage": STAGE, "status": "FITTING", "check_only": False,
                                    "processed_episodes": EXPECTED_EPISODES, "expected_episodes": EXPECTED_EPISODES,
                                    "fit_started": True, "fit_epoch": 0, "fit_epochs": EPOCHS})

        def fit_progress(epoch: int, epochs: int) -> None:
            _write_json(progress_path, {"stage": STAGE, "status": "FITTING", "check_only": False,
                                        "processed_episodes": EXPECTED_EPISODES, "expected_episodes": EXPECTED_EPISODES,
                                        "fit_started": True, "fit_epoch": int(epoch), "fit_epochs": int(epochs)})

        fit = _fit_models(policy.model, expanded, train, progress_callback=fit_progress)
        after_cache = _prediction_cache(policy.model, expanded, trajectories)
        after_episode = _episode_metrics_from_cache(after_cache, trajectories)
        metrics = {
            "fit": {
                "before": _aggregate_prediction_cache(before_cache, train),
                "after": _aggregate_prediction_cache(after_cache, train),
            },
            "validation": {
                "before": _aggregate_prediction_cache(before_cache, validation),
                "after": _aggregate_prediction_cache(after_cache, validation),
            },
        }
        actor_after = {
            name: value.detach().cpu().clone()
            for name, value in policy.model.state_dict().items()
            if not str(name).startswith(CRITIC_PREFIXES)
        }
        actor_unchanged = all(torch.equal(actor_before[name], actor_after[name]) for name in actor_before)
        _require(actor_unchanged, "source actor model changed during critic context fit")
        _save_critic(output / "legacy_critic.pt", policy.model, kind="legacy_mean_max")
        _save_critic(output / "context_critic.pt", expanded, kind="context_expanded_mean_max")
        report = {
            "stage": STAGE, "status": "COMPLETED", "check_only": False,
            "binding": {
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "warmup_checkpoint_sha256": WARMUP_CHECKPOINT_SHA256,
                "warmup": warmup_check, "input_root": str(INPUT_ROOT),
                "episodes": EXPECTED_EPISODES, "fit_episodes": FIT_EPISODES,
                "validation_episodes": VALIDATION_EPISODES,
                "total_steps": int(inputs["total_steps"]), "requests": EXPECTED_EPISODES * EXPECTED_REQUESTS,
                "fit_steps": int(sum(len(item["targets"]) for item in train)),
                "validation_steps": int(sum(len(item["targets"]) for item in validation)),
                "gamma": GAMMA, "reward_scale": REWARD_SCALE,
                "context_feature_names": list(CONTEXT_FEATURE_NAMES),
            },
            "fit": fit,
            "metrics": metrics,
            "episode_metrics": {"before": before_episode, "after": after_episode},
            "checks": {
                "replay_exact": all(item["replay_exact"] for item in trajectory_reports),
                "terminal": all(item["terminal"] for item in trajectory_reports),
                "ledger_alignment": all(item["ledger_alignment"] for item in trajectory_reports),
                "safety_required_zero": all(item["safety"]["required_counters_zero"] for item in trajectory_reports),
                "source_actor_model_bitwise_unchanged": actor_unchanged,
                "validation_excluded_from_optimizer": True,
                "expanded_initial_prediction_matches_legacy": True,
                "expanded_initial_prediction_max_abs_diff": float(empty_initial_max_abs_diff),
                "same_source_policy_cross_episode_prediction": True,
                "generalization_claim": False,
            },
            "scope": {"env_resets": EXPECTED_EPISODES, "replay_calls": EXPECTED_EPISODES,
                      "fit_episodes": FIT_EPISODES, "validation_episodes": VALIDATION_EPISODES,
                      "optimizer_used": True, "ppo_updates": False, "policy_sampling": False,
                      "teacher_used": False, "actor_updated": False},
            "trajectory_reports": trajectory_reports,
            "artifacts": {
                "progress": str(progress_path), "report": str(output / "report.json"),
                "summary": str(output / "summary.json"), "replay_cache": str(output / "replay_cache.pt"),
                "legacy_critic": str(output / "legacy_critic.pt"), "context_critic": str(output / "context_critic.pt"),
            },
        }
        _write_json(output / "report.json", report)
        _write_json(output / "summary.json", {
            "stage": STAGE, "status": "COMPLETED", "metrics": metrics,
            "checks": report["checks"], "fit": fit,
        })
        _write_json(progress_path, {"stage": STAGE, "status": "COMPLETED", "check_only": False,
                                    "processed_episodes": EXPECTED_EPISODES, "fit_started": True,
                                    "fit_epochs": EPOCHS, "replay_calls": EXPECTED_EPISODES})
        return report
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        report = {
            "stage": STAGE, "status": "FAILED", "check_only": bool(check_only),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "scope": {"env_resets": int(completed_episodes), "replay_calls": int(completed_episodes),
                      "fit_started": bool(fit_started), "optimizer_used": bool(fit_started),
                      "ppo_updates": False, "policy_sampling": False,
                      "teacher_used": False},
            "artifacts": {"progress": str(progress_path), "report": str(output / "report.json"),
                          "summary": str(output / "summary.json")},
        }
        _write_json(output / "report.json", report)
        _write_json(output / "summary.json", report)
        _write_json(progress_path, {"stage": STAGE, "status": "FAILED", "check_only": bool(check_only),
                                    "error": str(exc), "processed_episodes": int(completed_episodes),
                                    "expected_episodes": EXPECTED_EPISODES, "fit_started": bool(fit_started)})
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.output_dir, check_only=bool(args.check_only))
    print(json.dumps(_jsonable({"stage": STAGE, "status": result.get("status"),
                                "output_dir": str(args.output_dir)}), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"COMPLETED", "CHECK_ONLY_OK"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
