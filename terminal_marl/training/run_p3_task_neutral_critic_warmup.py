"""Fit only the centralized critic on the frozen task-neutral tapes.

The warmup consumes the six already-recorded task-neutral action tapes.  It
does not sample an action, alter a tape, or run a PPO update.  The resulting
checkpoint is an optional pilot starting point; the candidate actor, gate,
proposal head, and temperature remain bitwise unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.core.observations import pad_agent_sets
from terminal_marl.training import diagnose_p3_ppo_credit_signal as credit
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval


STAGE = "P3_TASK_NEUTRAL_CRITIC_WARMUP_V1"
CURRICULUM_ID = "task_neutral_critic_warmup"
PARENT_CHECKPOINT = neutral_eval.CANDIDATE_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = neutral_eval.CANDIDATE_CHECKPOINT_SHA256
BASELINE_RECORD_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_baseline_20260928/full/records/candidate"
)
GAMMA = 0.99
EPOCHS = 20
MINIBATCH_SIZE = 64
LEARNING_RATE = 3e-4
PERMUTATION_SEED = 874501
GRAD_CLIP = 1.0
CRITIC_PREFIXES = ("critic_encoder.", "critic_head.")
OBSERVATION_DIM = 88


class CriticWarmupError(ValueError):
    """Raised when the fixed warmup input or output contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticWarmupError(message)


def _validate_gamma(gamma: float) -> float:
    try:
        value = float(gamma)
    except (TypeError, ValueError) as exc:
        raise CriticWarmupError("gamma must be finite and in [0, 1]") from exc
    _require(math.isfinite(value) and 0.0 <= value <= 1.0,
             "gamma must be finite and in [0, 1]")
    return value


def _gamma_target_label(gamma: float) -> str:
    return f"production_mc_returns_gamma_{_validate_gamma(gamma):g}_complete_terminal_zero_bootstrap"


def _gamma_argument(value: str) -> float:
    try:
        gamma = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("gamma must be finite and in [0, 1]") from exc
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise argparse.ArgumentTypeError("gamma must be finite and in [0, 1]")
    return gamma


def _validate_reward_scale(reward_scale: float) -> float:
    try:
        value = float(reward_scale)
    except (TypeError, ValueError) as exc:
        raise CriticWarmupError("reward_scale must be finite and positive") from exc
    _require(math.isfinite(value) and value > 0.0,
             "reward_scale must be finite and positive")
    return value


def _reward_scale_argument(value: str) -> float:
    try:
        reward_scale = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("reward_scale must be finite and positive") from exc
    if not math.isfinite(reward_scale) or reward_scale <= 0.0:
        raise argparse.ArgumentTypeError("reward_scale must be finite and positive")
    return reward_scale


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
        raise CriticWarmupError(f"invalid JSON input: {path}") from exc
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
        str(name): value.detach().cpu().clone() if isinstance(value, torch.Tensor)
        else value
        for name, value in state.items()
    }


def _is_critic_name(name: str) -> bool:
    return str(name).startswith(CRITIC_PREFIXES)


def _noncritic_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {str(name): value for name, value in state.items() if not _is_critic_name(str(name))}


def _states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        torch.equal(left[name], right[name])
        if isinstance(left[name], torch.Tensor) and isinstance(right[name], torch.Tensor)
        else left[name] == right[name]
        for name in left
    )


def mc_targets(rewards: Sequence[float] | np.ndarray,
               episode_end: Sequence[bool] | np.ndarray | None = None,
               *, gamma: float = GAMMA, reward_scale: float = 1.0) -> np.ndarray:
    """Compute complete-episode targets using production MC returns.

    ``episode_end`` permits several trajectories to be concatenated while
    keeping each terminal return's bootstrap at zero.  Each contiguous
    episode is delegated to ``credit.mc_returns`` rather than reimplementing
    its discount arithmetic.
    """
    gamma = _validate_gamma(gamma)
    reward_scale = _validate_reward_scale(reward_scale)
    values = np.asarray(rewards, dtype=np.float64)
    _require(values.ndim == 1 and np.isfinite(values).all(), "rewards must be a finite vector")
    if episode_end is None:
        ends = np.zeros(values.size, dtype=bool)
        if values.size:
            ends[-1] = True
    else:
        ends = np.asarray(episode_end, dtype=bool)
        _require(ends.shape == values.shape, "episode_end must align with rewards")
    result = np.zeros(values.size, dtype=np.float64)
    start = 0
    for index, is_end in enumerate(ends):
        if not bool(is_end):
            continue
        result[start:index + 1] = (
            credit.mc_returns(values[start:index + 1], float(gamma)) * reward_scale
        )
        start = index + 1
    _require(start == values.size, "MC targets require a terminal boundary for every trajectory")
    return result


def _critic_predictions(policy: Any, worlds: Sequence[np.ndarray]) -> np.ndarray:
    if not worlds:
        return np.zeros(0, dtype=np.float64)
    arrays = [np.asarray(world, dtype=np.float32) for world in worlds]
    _require(all(array.ndim == 2 and array.shape[1] == OBSERVATION_DIM for array in arrays),
             "critic worlds must be 88D matrices")
    padded, valid = pad_agent_sets(arrays)
    if padded.shape[1] == 0:
        padded = np.zeros((len(arrays), 1, OBSERVATION_DIM), dtype=np.float32)
        valid = np.zeros((len(arrays), 1), dtype=bool)
    with torch.no_grad():
        values = policy.model.critic(
            torch.as_tensor(padded, dtype=torch.float32, device=policy.device),
            torch.as_tensor(valid, dtype=torch.bool, device=policy.device),
        )
    return values.detach().cpu().numpy().astype(np.float64)


def _critic_predictions_with_grad(policy: Any, worlds: Sequence[np.ndarray]) -> torch.Tensor:
    arrays = [np.asarray(world, dtype=np.float32) for world in worlds]
    _require(arrays and all(array.ndim == 2 and array.shape[1] == OBSERVATION_DIM for array in arrays),
             "critic worlds must be nonempty 88D matrices")
    padded, valid = pad_agent_sets(arrays)
    if padded.shape[1] == 0:
        padded = np.zeros((len(arrays), 1, OBSERVATION_DIM), dtype=np.float32)
        valid = np.zeros((len(arrays), 1), dtype=bool)
    return policy.model.critic(
        torch.as_tensor(padded, dtype=torch.float32, device=policy.device),
        torch.as_tensor(valid, dtype=torch.bool, device=policy.device),
    )


def _critic_output_layer(policy: Any) -> Any:
    head = getattr(getattr(policy, "model", None), "critic_head", None)
    _require(head is not None, "critic output head is missing")
    if isinstance(head, torch.nn.Linear):
        return head
    linears = [module for module in head.modules() if isinstance(module, torch.nn.Linear)]
    _require(linears, "critic output layer is missing")
    return linears[-1]


def _scale_critic_output(policy: Any, reward_scale: float) -> None:
    reward_scale = _validate_reward_scale(reward_scale)
    layer = _critic_output_layer(policy)
    _require(getattr(layer, "bias", None) is not None,
             "critic output layer bias is missing")
    with torch.no_grad():
        layer.weight.mul_(reward_scale)
        layer.bias.mul_(reward_scale)


def _metrics(policy: Any, worlds: Sequence[np.ndarray], targets: Sequence[float], *,
             reward_scale: float = 1.0) -> dict[str, Any]:
    reward_scale = _validate_reward_scale(reward_scale)
    target_array = np.asarray(targets, dtype=np.float64)
    prediction = _critic_predictions(policy, worlds)
    _require(prediction.shape == target_array.shape, "critic predictions and targets are misaligned")
    error = prediction - target_array
    mse = float(np.mean(error ** 2)) if len(error) else None
    mae = float(np.mean(np.abs(error))) if len(error) else None
    target_variance = float(np.var(target_array)) if len(target_array) else 0.0
    explained = None if target_variance <= 1e-15 else float(1.0 - np.var(error) / target_variance)
    return {
        "count": int(len(target_array)),
        "mse": mse,
        "mae": mae,
        "normalized_mse": mse,
        "normalized_mae": mae,
        "raw_mse": None if mse is None else float(mse / (reward_scale ** 2)),
        "raw_mae": None if mae is None else float(mae / reward_scale),
        "explained_variance": explained,
    }


def _trajectory_record(*, scenario: str, seed: int, requests: int, tape_path: Path,
                       replay: Mapping[str, Any], steps: Sequence[Any],
                       checkpoint_sha256: str, model_digest_before: str,
                       model_digest_after: str) -> dict[str, Any]:
    lifecycle = replay.get("lifecycle") if isinstance(replay.get("lifecycle"), Mapping) else {}
    env = replay.get("env")
    safety = credit.terminal_credit._safety_snapshot(env) if env is not None else {}
    _require(replay.get("replay", {}).get("exact") is True, f"{scenario}/{seed} replay was not exact")
    _require(bool(lifecycle.get("terminal")), f"{scenario}/{seed} replay was not terminal")
    _require(bool(lifecycle.get("all_requests_observed")), f"{scenario}/{seed} requests were incomplete")
    _require(bool(lifecycle.get("ledger_alignment")), f"{scenario}/{seed} ledger alignment failed")
    _require(len(lifecycle.get("request_rows", [])) == int(requests),
             f"{scenario}/{seed} request count mismatch")
    required_safety = {
        name: safety.get(name)
        for name in credit.terminal_credit.REQUIRED_SAFETY_COUNTERS
    }
    _require(
        all(
            value is not None
            and math.isfinite(float(value))
            and float(value) == 0.0
            for value in required_safety.values()
        ),
        f"{scenario}/{seed} required safety counter is missing, non-finite, or nonzero",
    )
    safety["required_counters_zero"] = True
    optional_uncertified = safety.get("uncertified_execution")
    safety["uncertified_execution_available"] = optional_uncertified is not None
    if optional_uncertified is None:
        safety["uncertified_execution_zero"] = None
    else:
        _require(
            math.isfinite(float(optional_uncertified)) and float(optional_uncertified) == 0.0,
            f"{scenario}/{seed} uncertified execution counter is non-finite or nonzero",
        )
        safety["uncertified_execution_zero"] = True
    return {
        "scenario": str(scenario),
        "seed": int(seed),
        "requests": int(requests),
        "action_tape": str(tape_path),
        "steps": int(len(steps)),
        "replay": dict(replay.get("replay") or {}),
        "lifecycle": dict(lifecycle),
        "safety": dict(safety),
        "checkpoint_sha256": str(checkpoint_sha256),
        "model_digest_before": str(model_digest_before),
        "model_digest_after": str(model_digest_after),
        "model_unchanged": bool(model_digest_before == model_digest_after),
    }


def _record_path(scenario: str, seed: int) -> Path:
    return _synced_path(BASELINE_RECORD_ROOT / f"{scenario}_{seed}.json")


def _preflight() -> dict[str, Any]:
    resolved_checkpoint, locator = neutral_eval._checkpoint_from_report(
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    del configs, _runtime
    cells: list[dict[str, Any]] = []
    for cell in neutral_eval._evaluation_cells():
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        config = neutral_configs[scenario]
        path = _record_path(scenario, seed)
        record = _read_json(path)
        _require(str(record.get("scenario")) == scenario and int(record.get("seed", -1)) == seed,
                 f"baseline cell mismatch: {path}")
        requests = int(getattr(config, "total_requested_uavs", -1))
        _require(requests > 0, f"{scenario}/{seed} config request count is invalid")
        _require(int(record.get("requests", -1)) == requests,
                 f"{scenario}/{seed} baseline request count mismatch")
        _require(record.get("task_priority_enabled") is False,
                 f"{scenario}/{seed} baseline is not task-neutral")
        _require(float(record.get("low_level_priority_time_weight", 1.0)) == 0.0,
                 f"{scenario}/{seed} baseline priority weight mismatch")
        declared_sha = next((record.get(key) for key in (
            "checkpoint_sha256", "checkpoint_sha256_expected", "candidate_checkpoint_sha256"
        ) if isinstance(record.get(key), str)), None)
        _require(declared_sha == PARENT_CHECKPOINT_SHA256,
                 f"{scenario}/{seed} baseline checkpoint SHA mismatch")
        config_snapshot = record.get("config")
        _require(isinstance(config_snapshot, Mapping),
                 f"{scenario}/{seed} baseline config snapshot missing")
        _require(dict(config_snapshot) == neutral_eval._config_snapshot(config),
                 f"{scenario}/{seed} baseline config snapshot mismatch")
        tape_value = record.get("action_tape")
        _require(isinstance(tape_value, str) and tape_value, f"{scenario}/{seed} tape path missing")
        tape_path = _synced_path(Path(tape_value))
        _payload, tape_rows = credit._load_tape(tape_path)
        if len(tape_rows) > credit.MAX_REPLAY_STEPS:
            raise CriticWarmupError(
                f"{scenario}/{seed} tape has {len(tape_rows)} steps, exceeds capture limit {credit.MAX_REPLAY_STEPS}"
            )
        cells.append({
            "scenario": scenario,
            "seed": seed,
            "split": str(cell.get("split", "development")),
            "requests": requests,
            "record_path": path,
            "record": record,
            "tape_path": tape_path,
            "tape_rows": tape_rows,
            "config": config,
        })
    _require(len(cells) == 6, "warmup requires exactly six registered cells")
    return {
        "checkpoint": resolved_checkpoint,
        "checkpoint_locator": locator,
        "checkpoint_sha256": _sha256(resolved_checkpoint),
        "cells": cells,
        "neutral_configs": neutral_configs,
    }


def _collect_trajectories(
    inputs: Mapping[str, Any], *, gamma: float = GAMMA, reward_scale: float = 1.0,
    progress_callback: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gamma = _validate_gamma(gamma)
    reward_scale = _validate_reward_scale(reward_scale)
    trajectories: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    checkpoint = Path(inputs["checkpoint"])
    checkpoint_sha = str(inputs["checkpoint_sha256"])
    for cell_index, cell in enumerate(inputs["cells"]):
        if progress_callback is not None:
            progress_callback("replay", cell_index, len(inputs["cells"]))
        config = cell["config"]
        policy, _payload = rank_eval._load_policy(checkpoint, config)
        before_digest = rank_eval._model_digest(policy)
        replay = credit._capture_replay(
            policy,
            config,
            Path(cell["tape_path"]),
            cell["tape_rows"],
        )
        captures = list(replay["captures"])
        steps = [credit._step_from_capture(row) for row in captures]
        rewards = np.asarray([float(step.reward) for step in steps], dtype=np.float64)
        ends = np.asarray([bool(step.episode_end or step.terminated) for step in steps], dtype=bool)
        _require(len(steps) > 0 and bool(ends[-1]), f"{cell['scenario']}/{cell['seed']} has no terminal step")
        targets = mc_targets(rewards, ends, gamma=gamma, reward_scale=reward_scale)
        after_digest = rank_eval._model_digest(policy)
        _require(before_digest == after_digest, f"{cell['scenario']}/{cell['seed']} model changed during replay")
        reports.append(_trajectory_record(
            scenario=cell["scenario"], seed=cell["seed"], requests=cell["requests"],
            tape_path=Path(cell["tape_path"]), replay=replay, steps=steps,
            checkpoint_sha256=checkpoint_sha, model_digest_before=before_digest,
            model_digest_after=after_digest,
        ))
        trajectories.append({
            "scenario": cell["scenario"],
            "seed": cell["seed"],
            "steps": steps,
            "worlds": [np.asarray(step.world, dtype=np.float32) for step in steps],
            "targets": targets,
        })
        if progress_callback is not None:
            progress_callback("replay", cell_index + 1, len(inputs["cells"]))
    return trajectories, reports


def fit_critic(policy: Any, trajectories: Sequence[Mapping[str, Any]], *, epochs: int = EPOCHS,
               minibatch_size: int = MINIBATCH_SIZE, learning_rate: float = LEARNING_RATE,
               permutation_seed: int = PERMUTATION_SEED, grad_clip: float = GRAD_CLIP,
               gamma: float = GAMMA, reward_scale: float = 1.0,
               progress_callback: Any = None) -> dict[str, Any]:
    """Fit only critic parameters and return before/after error evidence."""
    gamma = _validate_gamma(gamma)
    reward_scale = _validate_reward_scale(reward_scale)
    _require(trajectories, "critic warmup requires trajectories")
    worlds = [world for trajectory in trajectories for world in trajectory["worlds"]]
    targets = np.concatenate([
        np.asarray(trajectory["targets"], dtype=np.float64) for trajectory in trajectories
    ])
    _require(len(worlds) == len(targets), "warmup worlds and targets are misaligned")
    before_state = _clone_state(policy.model.state_dict())
    _scale_critic_output(policy, reward_scale)
    for name, parameter in policy.model.named_parameters():
        parameter.requires_grad = _is_critic_name(name)
    critic_parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
    _require(critic_parameters, "critic warmup scope is empty")
    optimizer = torch.optim.Adam(critic_parameters, lr=float(learning_rate))
    before_metrics = _metrics(policy, worlds, targets, reward_scale=reward_scale)
    before_cells = {
        f"{trajectory['scenario']}_{int(trajectory['seed'])}": _metrics(
            policy, trajectory["worlds"], trajectory["targets"], reward_scale=reward_scale,
        )
        for trajectory in trajectories
    }
    rng = np.random.RandomState(int(permutation_seed))
    for epoch in range(int(epochs)):
        permutation = rng.permutation(len(worlds))
        for start in range(0, len(worlds), int(minibatch_size)):
            indices = permutation[start:start + int(minibatch_size)]
            batch_worlds = [worlds[int(index)] for index in indices]
            batch_targets = torch.as_tensor(targets[indices], dtype=torch.float32, device=policy.device)
            predictions = _critic_predictions_with_grad(policy, batch_worlds)
            loss = (predictions - batch_targets).square().mean()
            _require(bool(torch.isfinite(loss).item()), "critic warmup produced non-finite MSE")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_parameters, float(grad_clip), error_if_nonfinite=True)
            optimizer.step()
        if progress_callback is not None:
            progress_callback(int(epoch + 1), int(epochs))
    policy.model.eval()
    after_state = _clone_state(policy.model.state_dict())
    after_metrics = _metrics(policy, worlds, targets, reward_scale=reward_scale)
    after_cells = {
        f"{trajectory['scenario']}_{int(trajectory['seed'])}": _metrics(
            policy, trajectory["worlds"], trajectory["targets"], reward_scale=reward_scale,
        )
        for trajectory in trajectories
    }
    noncritic_before = _noncritic_state(before_state)
    noncritic_after = _noncritic_state(after_state)
    actor_invariance = {
        "non_critic_state_bitwise_unchanged": _states_equal(noncritic_before, noncritic_after),
        "non_critic_digest_before": _state_digest(noncritic_before),
        "non_critic_digest_after": _state_digest(noncritic_after),
        "critic_parameter_names": sorted(name for name in before_state if _is_critic_name(name)),
    }
    _require(actor_invariance["non_critic_state_bitwise_unchanged"],
             "critic warmup changed non-critic model state")
    return {
        "before": before_metrics,
        "after": after_metrics,
        "before_per_cell": before_cells,
        "after_per_cell": after_cells,
        "actor_invariance": actor_invariance,
        "fit_parameters": {
            "gamma": gamma,
            "reward_scale": reward_scale,
            "epochs": int(epochs),
            "minibatch_size": int(minibatch_size),
            "learning_rate": float(learning_rate),
            "permutation_seed": int(permutation_seed),
            "grad_clip": float(grad_clip),
            "loss": "MSE",
            "target": _gamma_target_label(gamma),
            "early_select": False,
            "reward_changed": False,
            "gae_lambda_changed": False,
        },
    }


def _warmup_payload(policy: Any, inputs: Mapping[str, Any], fit: Mapping[str, Any], reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    snapshots = {
        str(scenario): neutral_eval._config_snapshot(config)
        for scenario, config in inputs["neutral_configs"].items()
    }
    state = _clone_state(policy.model.state_dict())
    fit_parameters = dict(fit["fit_parameters"])
    fit_parameters.setdefault("reward_scale", 1.0)
    return {
        "format": "p3_task_neutral_critic_warmup_v1",
        "stage": STAGE,
        "curriculum_id": CURRICULUM_ID,
        "gamma": float(fit_parameters["gamma"]),
        "reward_scale": float(fit_parameters["reward_scale"]),
        "task_priority_enabled": False,
        "low_level_priority_time_weight": 0.0,
        "neutral_config_snapshots": snapshots,
        "parent_checkpoint": str(inputs["checkpoint"]),
        "parent_checkpoint_sha256": str(inputs["checkpoint_sha256"]),
        "model_state": state,
        "actor_invariance": dict(fit["actor_invariance"]),
        "fit_parameters": fit_parameters,
        "fit_metrics": {
            "before": dict(fit["before"]),
            "after": dict(fit["after"]),
            "before_per_cell": dict(fit["before_per_cell"]),
            "after_per_cell": dict(fit["after_per_cell"]),
        },
        "trajectory_reports": [dict(item) for item in reports],
        "generalization_claim": False,
        "same_data_fit_not_generalization": True,
    }


def run(output_dir: str | Path, *, check_only: bool = False,
        gamma: float = GAMMA, reward_scale: float = 1.0) -> dict[str, Any]:
    gamma = _validate_gamma(gamma)
    reward_scale = _validate_reward_scale(reward_scale)
    output = Path(output_dir)
    _require(not output.exists(), f"output directory must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    progress_path = output / "progress.json"
    _write_json(progress_path, {"stage": STAGE, "status": "STARTING", "check_only": bool(check_only),
                                "gamma": gamma, "reward_scale": reward_scale,
                                "processed_cells": 0})
    completed_cells = 0
    fit_started = False
    try:
        inputs = _preflight()
        preflight = {
            "parent_checkpoint": str(inputs["checkpoint"]),
            "parent_checkpoint_sha256": str(inputs["checkpoint_sha256"]),
            "cells": [
                {"scenario": item["scenario"], "seed": item["seed"], "requests": item["requests"],
                 "steps": len(item["tape_rows"]), "action_tape": str(item["tape_path"])}
                for item in inputs["cells"]
            ],
        }
        if check_only:
            report = {
                "stage": STAGE,
                "status": "CHECK_ONLY_OK",
                "check_only": True,
                "gamma": gamma,
                "reward_scale": reward_scale,
                "preflight": preflight,
                "scope": {"replay_calls": 0, "env_reset": False, "optimizer_used": False,
                          "gamma": gamma, "reward_scale": reward_scale},
                "artifacts": {"progress": str(progress_path), "report": str(output / "report.json")},
            }
            _write_json(output / "report.json", report)
            _write_json(progress_path, {"stage": STAGE, "status": "CHECK_ONLY_OK", "check_only": True,
                                        "gamma": gamma, "reward_scale": reward_scale,
                                        "processed_cells": 0})
            return report

        def replay_progress(phase: str, processed: int, expected: int) -> None:
            nonlocal completed_cells
            completed_cells = int(processed)
            _write_json(progress_path, {
                "stage": STAGE, "status": "REPLAYING", "check_only": False,
                "gamma": gamma, "reward_scale": reward_scale,
                "phase": phase, "processed_cells": int(processed), "expected_cells": int(expected),
                "fit_started": False,
            })

        trajectories, trajectory_reports = _collect_trajectories(
            inputs, gamma=gamma, reward_scale=reward_scale,
            progress_callback=replay_progress,
        )
        _write_json(progress_path, {"stage": STAGE, "status": "FITTING", "check_only": False,
                                    "gamma": gamma, "reward_scale": reward_scale,
                                    "processed_cells": 6, "fit_started": True,
                                    "fit_epoch": 0, "fit_epochs": EPOCHS})
        policy, _payload = rank_eval._load_policy(Path(inputs["checkpoint"]), inputs["neutral_configs"]["dense24"])
        # Replay values are data only; reload the fixed candidate before the fit.
        fit_started = True

        def fit_progress(epoch: int, epochs: int) -> None:
            _write_json(progress_path, {
                "stage": STAGE, "status": "FITTING", "check_only": False,
                "gamma": gamma, "reward_scale": reward_scale,
                "processed_cells": 6, "fit_started": True,
                "fit_epoch": int(epoch), "fit_epochs": int(epochs),
            })

        fit = fit_critic(policy, trajectories, gamma=gamma, reward_scale=reward_scale,
                         progress_callback=fit_progress)
        payload = _warmup_payload(policy, inputs, fit, trajectory_reports)
        warmup_path = output / "warmup.pt"
        _require(not warmup_path.exists(), f"refusing to overwrite warmup checkpoint: {warmup_path}")
        torch.save(payload, warmup_path)
        report = {
            "stage": STAGE,
            "status": "COMPLETED",
            "check_only": False,
            "gamma": gamma,
            "reward_scale": reward_scale,
            "preflight": preflight,
            "trajectory_reports": trajectory_reports,
            "fit": fit,
            "scope": {
                "replay_calls": 6,
                "policy_sampling": False,
                "optimizer_used": True,
                "ppo_updates": False,
                "teacher_used": False,
                "reward_changed": False,
                "safety_changed": False,
                "same_data_fit_not_generalization": True,
                "gamma": gamma, "reward_scale": reward_scale,
            },
            "artifacts": {
                "progress": str(progress_path),
                "report": str(output / "report.json"),
                "warmup_checkpoint": str(warmup_path),
            },
        }
        _write_json(output / "report.json", report)
        _write_json(progress_path, {"stage": STAGE, "status": "COMPLETED", "check_only": False, "processed_cells": 6})
        return report
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        report = {
            "stage": STAGE,
            "status": "FAILED",
            "check_only": bool(check_only),
            "gamma": gamma,
            "reward_scale": reward_scale,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "scope": {"replay_calls": int(completed_cells), "optimizer_used": bool(fit_started),
                      "ppo_updates": False, "gamma": gamma, "reward_scale": reward_scale},
            "artifacts": {"progress": str(progress_path), "report": str(output / "report.json")},
        }
        _write_json(output / "report.json", report)
        _write_json(progress_path, {
            "stage": STAGE, "status": "FAILED", "check_only": bool(check_only),
            "gamma": gamma, "reward_scale": reward_scale,
            "processed_cells": int(completed_cells), "fit_started": bool(fit_started),
            "error": str(exc),
        })
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--gamma", type=_gamma_argument, default=GAMMA)
    parser.add_argument("--reward-scale", type=_reward_scale_argument, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.output_dir, check_only=bool(args.check_only), gamma=float(args.gamma),
                 reward_scale=float(args.reward_scale))
    print(json.dumps(_jsonable({"stage": STAGE, "status": result.get("status"), "output_dir": str(args.output_dir)}), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"COMPLETED", "CHECK_ONLY_OK"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
