"""Strict checkpoint routing for the console visualization workbench.

The workbench owns presentation through ``observer_factory`` while the
checkpoint viewer remains the single source of truth for profile loading,
schema validation, deterministic requests, and the frozen shortlist path.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import ppo_checkpoint_viewer as _viewer


_P3_PROFILE = "p3_task_neutral"
_ALLOWED_PROFILES = frozenset(("checkpoint12", "p2_initial_k3", "p2_trained_k3", _P3_PROFILE))
_ALLOWED_SCENARIOS = {
    "checkpoint12": frozenset(("warm3", "flow3", "flow6")),
    "p2_initial_k3": frozenset(("flow6", "flow8", "flow12", "p2_dense24")),
    "p2_trained_k3": frozenset(("flow6", "flow8", "flow12", "p2_dense24")),
    _P3_PROFILE: frozenset(("flow6", "flow12", "dense24")),
}
_NEUTRAL_KEYS = frozenset(
    {
        "task_priority_enabled",
        "low_level_priority_time_weight",
        "neutral_config_snapshot",
        "neutral_config_snapshots",
    }
)
_ALLOWED_KEYS = frozenset(
    {
        "mode",
        "profile",
        "scenario",
        "seed",
        "low_checkpoint",
        "high_checkpoint",
        "start_paused",
        "max_steps",
        "max_simulation_seconds",
        "playback_speed",
        "frame_stride",
        "trajectories",
        "labels",
        "show_trajectories",
        "show_labels",
        "trajectory_max_points",
        "deterministic",
        *_NEUTRAL_KEYS,
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_bool(value: Any, field: str) -> bool:
    _require(isinstance(value, bool), f"{field} must be a boolean")
    return value


def _strict_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an integer")
    result = int(value)
    if minimum is not None:
        _require(result >= minimum, f"{field} must be at least {minimum}")
    return result


def _strict_float(value: Any, field: str, *, minimum: float | None = None) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{field} must be finite")
    if minimum is not None:
        _require(result >= minimum, f"{field} must be at least {minimum}")
    return result


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_neutral_snapshot(snapshot: Any, expected_config: Any | None = None) -> None:
    """Validate only task-neutral semantics carried by an optional snapshot.

    Scenario population, arrival timing, and other physical fields remain
    owned by the formal scenario config.  This check only binds schema and
    task-neutral flags when a checkpoint or request explicitly supplies them.
    """

    _require(isinstance(snapshot, Mapping), "neutral_config_snapshot must be a mapping")
    if expected_config is not None:
        schema_fields = {
            "observation_dimension": 88,
            "observation_schema_version": getattr(expected_config, "observation_schema_version", None),
            "action_schema_version": getattr(expected_config, "action_schema_version", None),
            "high_level_enabled": getattr(expected_config, "high_level_enabled", None),
            "scheduler_mode": getattr(expected_config, "scheduler_mode", None),
            "teacher_enabled": getattr(expected_config, "teacher_enabled", None),
        }
    else:
        schema_fields = {}
    for field, expected in schema_fields.items():
        if field in snapshot and expected is not None:
            _require(snapshot[field] == expected, f"neutral config {field} is incompatible")
    if "task_priority_enabled" in snapshot:
        _require(snapshot["task_priority_enabled"] is False,
                 "task-neutral checkpoint requires task_priority_enabled=False")
    if "low_level_priority_time_weight" in snapshot:
        value = snapshot["low_level_priority_time_weight"]
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == 0.0,
            "task-neutral checkpoint requires low_level_priority_time_weight=0",
        )


def _validate_neutral_semantics(
    payload: Mapping[str, Any],
    *,
    expected_config: Any,
    requested: Mapping[str, Any] | None = None,
) -> None:
    """Check optional neutral flags without requiring legacy checkpoints to carry them."""

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        containers: list[Mapping[str, Any]] = [metadata]
    else:
        containers = []
    containers.append(payload)
    report = payload.get("report")
    if isinstance(report, Mapping):
        containers.append(report)
        report_metadata = report.get("metadata")
        if isinstance(report_metadata, Mapping):
            containers.append(report_metadata)

    for container in containers:
        if "task_priority_enabled" in container:
            _require(container["task_priority_enabled"] is False,
                     "task-neutral checkpoint requires task_priority_enabled=False")
        if "low_level_priority_time_weight" in container:
            value = container["low_level_priority_time_weight"]
            _require(
                isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == 0.0,
                "task-neutral checkpoint requires low_level_priority_time_weight=0",
            )
        for field in ("neutral_config_snapshot", "neutral_config_snapshots"):
            if field not in container:
                continue
            snapshot = container[field]
            if field == "neutral_config_snapshots" and isinstance(snapshot, Mapping):
                scenario = str((requested or {}).get("scenario", ""))
                snapshot = snapshot.get(scenario, snapshot)
            _validate_neutral_snapshot(snapshot, expected_config)

    for field in ("task_priority_enabled", "low_level_priority_time_weight"):
        if requested is not None and field in requested:
            value = requested[field]
            if field == "task_priority_enabled":
                _require(value is False, "task-neutral request requires task_priority_enabled=False")
            else:
                _require(
                    isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == 0.0,
                    "task-neutral request requires low_level_priority_time_weight=0",
                )
    if requested is not None:
        if "neutral_config_snapshot" in requested:
            _validate_neutral_snapshot(requested["neutral_config_snapshot"], expected_config)
        if "neutral_config_snapshots" in requested:
            snapshots = requested["neutral_config_snapshots"]
            _require(isinstance(snapshots, Mapping), "neutral_config_snapshots must be a mapping")
            scenario = str(requested.get("scenario", ""))
            _validate_neutral_snapshot(snapshots.get(scenario, snapshots), expected_config)


def _policy_digest(policy: Any) -> str | None:
    """Return a model digest when the loaded policy exposes a state dict."""

    try:
        state = policy.model.state_dict()
    except (AttributeError, TypeError):
        return None
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        try:
            raw = value.detach().cpu().contiguous().numpy().tobytes()
        except (AttributeError, TypeError, ValueError):
            return None
        digest.update(str(name).encode("utf-8"))
        digest.update(raw)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Keep diagnostic metadata serializable when unavailable metrics are inf."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            return value
    return value


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    unknown = sorted(set(config) - _ALLOWED_KEYS)
    _require(not unknown, f"unsupported checkpoint config fields: {unknown}")

    mode = config.get("mode")
    _require(mode == "checkpoint", "checkpoint workbench requires mode='checkpoint'")
    profile = config.get("profile")
    _require(isinstance(profile, str) and profile in _ALLOWED_PROFILES,
             f"unsupported checkpoint profile: {profile!r}")
    scenario = config.get("scenario")
    _require(isinstance(scenario, str) and scenario in _ALLOWED_SCENARIOS[profile],
             f"unsupported scenario {scenario!r} for {profile}")

    seed = _strict_int(config.get("seed"), "seed")
    raw_checkpoint = config.get("low_checkpoint")
    _require(isinstance(raw_checkpoint, (str, Path)) and str(raw_checkpoint), "low_checkpoint is required")
    checkpoint = Path(raw_checkpoint).expanduser().resolve()
    _require(checkpoint.is_file(), f"checkpoint does not exist: {checkpoint}")

    high_checkpoint = config.get("high_checkpoint")
    _require(
        high_checkpoint is None or isinstance(high_checkpoint, (str, Path)),
        "high_checkpoint must be a path or null",
    )
    start_paused = _strict_bool(config.get("start_paused", False), "start_paused")

    max_steps = config.get("max_steps")
    if max_steps is not None:
        max_steps = _strict_int(max_steps, "max_steps", minimum=1)
    max_seconds = config.get("max_simulation_seconds")
    if max_seconds is not None:
        max_seconds = _strict_float(max_seconds, "max_simulation_seconds", minimum=0.0)
        _require(max_seconds > 0.0, "max_simulation_seconds must be positive")

    playback_speed = _strict_float(config.get("playback_speed", 10.0), "playback_speed", minimum=0.0)
    frame_stride = _strict_int(config.get("frame_stride", 2), "frame_stride", minimum=1)
    trajectory_max_points = _strict_int(
        config.get("trajectory_max_points", 12),
        "trajectory_max_points",
        minimum=2,
    )
    trajectories = config.get("show_trajectories", config.get("trajectories", True))
    labels = config.get("show_labels", config.get("labels", True))
    trajectories = _strict_bool(trajectories, "show_trajectories")
    labels = _strict_bool(labels, "show_labels")
    deterministic = _strict_bool(
        config.get("deterministic", False if profile == _P3_PROFILE else True),
        "deterministic",
    )
    if profile != _P3_PROFILE:
        _require(deterministic, "checkpoint profiles require deterministic=True")
        _require(not any(field in config for field in _NEUTRAL_KEYS),
                 "task-neutral fields are only valid for p3_task_neutral")

    return {
        "mode": "checkpoint",
        "profile": str(profile),
        "scenario": str(scenario),
        "seed": seed,
        "checkpoint": checkpoint,
        "high_checkpoint": high_checkpoint,
        "start_paused": start_paused,
        "max_steps": max_steps,
        "max_simulation_seconds": max_seconds,
        "playback_speed": playback_speed,
        "frame_stride": frame_stride,
        "show_trajectories": trajectories,
        "show_labels": labels,
        "trajectory_max_points": trajectory_max_points,
        "deterministic": deterministic,
        **{field: config[field] for field in _NEUTRAL_KEYS if field in config},
    }


def _run_p3_checkpoint(
    values: Mapping[str, Any],
    observer_factory: Callable[[Any, Any], Any] | None,
) -> dict[str, Any]:
    """Run the registered task-neutral V4.2 policy against one formal scene."""

    import torch

    from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
    from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval

    _, neutral_configs, _runtime_protocol = neutral_eval._build_neutral_inputs()
    scenario = str(values["scenario"])
    _require(scenario in neutral_configs, f"task-neutral scenario config is missing: {scenario}")
    sim_config = neutral_configs[scenario]
    _validate_neutral_semantics({}, expected_config=sim_config, requested=values)

    checkpoint: Path = values["checkpoint"]
    torch.set_num_threads(1)
    torch.manual_seed(int(values["seed"]))
    policy, payload = rank_eval._load_policy(checkpoint, sim_config)
    _require(isinstance(payload, Mapping), "task-neutral checkpoint payload must be a mapping")
    _validate_neutral_semantics(payload, expected_config=sim_config, requested=values)
    _require(int(getattr(policy, "observation_dim", -1)) == 88,
             "task-neutral checkpoint observation_dim must be 88")
    policy.model.eval()
    shortlist_k = int(getattr(sim_config, "runtime_proposal_shortlist_k", 0))
    _require(shortlist_k == 3, "task-neutral formal runtime requires shortlist_k=3")
    optimizer = getattr(policy, "optimizer", None)
    optimizer_state = getattr(optimizer, "state", None)
    optimizer_state_empty = isinstance(optimizer_state, Mapping) and len(optimizer_state) == 0
    _require(optimizer_state_empty, "task-neutral checkpoint optimizer state must be empty")

    try:
        horizon = int(round(float(sim_config.max_simulation_seconds) / float(sim_config.dt)))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("task-neutral scenario has no valid simulation horizon") from exc
    _require(horizon >= 1, "task-neutral scenario horizon must be positive")
    limits = [horizon]
    if values.get("max_steps") is not None:
        limits.append(int(values["max_steps"]))
    if values.get("max_simulation_seconds") is not None:
        limits.append(max(0, int(float(values["max_simulation_seconds"]) / float(sim_config.dt))))
    step_limit = min(limits)

    # ConsoleStepObserver reads this before it is constructed.  It prevents
    # the final post-step frame from waiting for another UI permit.
    env = _viewer.TerminalParallelEnv(sim_config)
    env._console_step_limit = int(step_limit)
    observations, _ = env.reset(int(values["seed"]))
    observer = None
    render_config = _viewer.VisualizationConfig(
        enabled=observer_factory is not None,
        frame_stride=int(values["frame_stride"]),
        playback_speed=float(values["playback_speed"]),
        show_trajectories=bool(values["show_trajectories"]),
        show_labels=bool(values["show_labels"]),
        trajectory_max_points=int(values["trajectory_max_points"]),
    )
    digest_before = _policy_digest(policy)
    torch.manual_seed(int(values["seed"]))
    try:
        if observer_factory is not None:
            observer = observer_factory(env, render_config)
            if observer is None:
                raise TypeError("observer_factory must return an observer")
            env.set_step_observer(observer)
            wait_ready = getattr(observer, "wait_ready", None)
            if callable(wait_ready) and not wait_ready():
                raise RuntimeError(f"viewer failed: {getattr(observer, 'error', None)!r}")

        tick = time.perf_counter()

        def stopped() -> bool:
            return observer is not None and bool(getattr(observer, "stop_requested", False))

        for _ in range(step_limit):
            if stopped():
                break
            details = _viewer._student_request_details(
                policy,
                env,
                observations,
                deterministic=bool(values["deterministic"]),
                shortlist_k=shortlist_k,
            )
            requested, proposal_shortlists = details[3], details[-1]
            observations, *_ = env.step(requested, proposal_shortlists)
            if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
                break
            if observer is not None and all(
                hasattr(observer, attribute) for attribute in ("paused", "stop_requested", "playback_speed")
            ):
                tick = _viewer._pace_live_rollout(observer, sim_config.dt, tick)
            else:
                tick = time.perf_counter()
        digest_after = _policy_digest(policy)
        unchanged = digest_before == digest_after if digest_before is not None and digest_after is not None else None
        return {
            "profile": _P3_PROFILE,
            "scenario": scenario,
            "seed": int(values["seed"]),
            "seconds": float(env.now),
            "completed": len(env.completed_uavs),
            "telemetry": _json_safe(dict(env.telemetry)),
            "teacher_enabled": bool(sim_config.teacher_enabled),
            "observation_dim": int(policy.observation_dim),
            "runtime_proposal_shortlist_k": shortlist_k,
            "deterministic": bool(values["deterministic"]),
            "policy_seed": int(values["seed"]),
            "diagnostic_only": True,
            "task_neutral": True,
            "task_priority_enabled": False,
            "low_level_priority_time_weight": 0.0,
            "PPO_ALLOWED": False,
            "teacher_called": False,
            "training_or_update_called": False,
            "weights_unchanged": unchanged,
            "optimizer_loaded": False,
            "optimizer_state_empty": optimizer_state_empty,
            "diagnostic_max_simulation_seconds": values["max_simulation_seconds"],
            "diagnostic_step_limit": int(step_limit),
            "configured_horizon_steps": int(horizon),
        }
    finally:
        if observer is not None:
            try:
                finish = getattr(observer, "finish", None)
                if callable(finish):
                    try:
                        finish(env, wait_for_close=False)
                    except TypeError:
                        finish(env)
            finally:
                close = getattr(observer, "close", None)
                if callable(close):
                    close()


def run_checkpoint(
    config: dict[str, Any],
    observer_factory: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    """Run one frozen checkpoint profile through the visualization adapter.

    ``ppo_checkpoint_viewer.run`` performs the profile-specific SHA and schema
    checks.  This adapter validates the workbench payload, supplies the
    observer hook, and returns explicit provenance metadata.
    """
    values = _validate_config(config)
    if observer_factory is not None and not callable(observer_factory):
        raise TypeError("observer_factory must be callable")

    checkpoint: Path = values["checkpoint"]
    if values["profile"] == _P3_PROFILE:
        result = _run_p3_checkpoint(values, observer_factory)
    else:
        result = _viewer.run(
            checkpoint,
            profile=values["profile"],
            scenario=values["scenario"],
            seed=values["seed"],
            gui=False,
            playback_speed=values["playback_speed"],
            max_steps=values["max_steps"],
            max_simulation_seconds=values["max_simulation_seconds"],
            frame_stride=values["frame_stride"],
            show_trajectories=values["show_trajectories"],
            show_labels=values["show_labels"],
            trajectory_max_points=values["trajectory_max_points"],
            observer_factory=observer_factory,
        )
    if not isinstance(result, Mapping):
        raise TypeError("checkpoint viewer returned a non-mapping result")

    observation_dim = result.get("observation_dim")
    shortlist_k = result.get("runtime_proposal_shortlist_k")
    _require(isinstance(observation_dim, int) and not isinstance(observation_dim, bool),
             "checkpoint viewer did not report observation_dim")
    _require(observation_dim == 88, f"unsupported checkpoint observation_dim: {observation_dim}")
    _require(isinstance(shortlist_k, int) and not isinstance(shortlist_k, bool) and shortlist_k >= 1,
             "checkpoint viewer did not report a valid shortlist_k")

    metadata = dict(result)
    metadata.update(
        {
            "mode": "checkpoint",
            "profile": values["profile"],
            "scenario": values["scenario"],
            "seed": values["seed"],
            "low_checkpoint": str(checkpoint),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _checkpoint_sha256(checkpoint),
            "deterministic": bool(values["deterministic"]),
            "shortlist_k": int(shortlist_k),
            "observation_dim": int(observation_dim),
            "diagnostic_max_simulation_seconds": values["max_simulation_seconds"],
        }
    )
    return metadata


__all__ = ["run_checkpoint"]
