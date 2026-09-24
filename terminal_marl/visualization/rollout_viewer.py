"""Deterministic rollout command for the optional terminal 3D viewer."""
from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np

from ..core.config import SimConfig
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, FrozenLowLevelPolicy, RuleLowLevelAdapter
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import DynamicRulePolicy
from ..training.high_level_a6 import load_high_level
from .config import VisualizationConfig
from .pyvista_viewer import AsyncTerminalViewerObserver, BufferedProcessTerminalViewerObserver
from .snapshot import build_world_render_snapshot


def _fixed_rule_warm_flow_config(max_simulation_seconds: float) -> SimConfig:
    """Deterministic visual stress scene: populated rings plus four-gate flow.

    The initial 12 aircraft are allocated by ``TerminalParallelEnv`` in the
    canonical capacity-proportional 6/4/2 outer/middle/inner layout.  The
    remaining schedule is a reproducible 8-second stream; gate assignment is
    still sampled uniformly over all four physical gates from the run seed.
    This is a visualization scenario, not a training curriculum condition.
    """
    initial_count = 12
    interval_seconds = 8.0
    continuing = tuple(interval_seconds * index for index in range(1, int(max_simulation_seconds // interval_seconds) + 1))
    schedule = (0.0,) * initial_count + continuing
    return SimConfig(
        experiment_mode="fixed_rule", high_level_enabled=False,
        reset_mode="warm_start_rings", warm_start_uavs=initial_count,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=schedule, total_requested_uavs=len(schedule),
        gate_probabilities=(0.25, 0.25, 0.25, 0.25),
        max_simulation_seconds=max_simulation_seconds,
    )


def _config_for_mode(mode: str, max_simulation_seconds: float, scenario: str = "default") -> SimConfig:
    if scenario == "fixed_rule_warm_flow":
        if mode != "fixed_rule":
            raise ValueError("fixed_rule_warm_flow is available only with fixed_rule")
        return _fixed_rule_warm_flow_config(max_simulation_seconds)
    if scenario != "default":
        raise ValueError(f"unsupported visualization scenario: {scenario}")
    dynamic = mode in {"dynamic_rule", "learned_high_rule_low", "learned_high_frozen_low"}
    return SimConfig(experiment_mode=mode, high_level_enabled=dynamic, max_simulation_seconds=max_simulation_seconds)


def _attach_viewer(
    env: TerminalParallelEnv,
    config: VisualizationConfig,
) -> tuple[None, BufferedProcessTerminalViewerObserver]:
    observer = BufferedProcessTerminalViewerObserver(
        build_world_render_snapshot(env, trajectory_max_points=config.trajectory_max_points), config
    )
    env.set_step_observer(observer)
    return None, observer


def _pace_live_rollout(
    observer: AsyncTerminalViewerObserver | BufferedProcessTerminalViewerObserver | None,
    simulated_dt: float,
    tick_started: float,
) -> float:
    """Honor an explicit live-playback request without waiting for rendering.

    A zero speed remains the performance/default mode.  A positive value is a
    user-facing simulation-time rate, not renderer back-pressure; pauses are
    likewise explicit UI control rather than a stale-frame dependency.
    """
    if observer is None:
        return time.perf_counter()
    while observer.paused and not observer.stop_requested:
        time.sleep(0.03)
    if observer.stop_requested:
        return time.perf_counter()
    if observer.playback_speed > 0.0:
        remaining = simulated_dt / observer.playback_speed - (time.perf_counter() - tick_started)
        if remaining > 0.0:
            time.sleep(remaining)
    return time.perf_counter()


def run_rollout(
    *, mode: str, seed: int, gui: bool = False, frame_stride: int = 1,
    playback_speed: float = 0.0, show_trajectories: bool = False, show_labels: bool = False,
    low_checkpoint: str | None = None, high_checkpoint: str | None = None,
    max_simulation_seconds: float = 3600.0, max_steps: int | None = None, scenario: str = "default",
) -> TerminalParallelEnv:
    """Run one deterministic policy rollout, optionally with an interactive viewer."""
    if mode == "fixed_low_rl" and low_checkpoint is None:
        raise ValueError("--low-checkpoint is required for fixed_low_rl")
    if mode == "learned_high_rule_low" and high_checkpoint is None:
        raise ValueError("--high-checkpoint is required for learned_high_rule_low")
    if mode == "learned_high_frozen_low" and (low_checkpoint is None or high_checkpoint is None):
        raise ValueError("--low-checkpoint and --high-checkpoint are required for learned_high_frozen_low")
    config = _config_for_mode(mode, max_simulation_seconds, scenario)
    render_config = VisualizationConfig(
        enabled=gui, frame_stride=frame_stride, playback_speed=playback_speed,
        show_trajectories=show_trajectories, show_labels=show_labels,
    )
    observer: AsyncTerminalViewerObserver | BufferedProcessTerminalViewerObserver | None = None
    completed_normally = False
    rollout_env: TerminalParallelEnv | None = None

    def stopped() -> bool:
        return observer is not None and observer.stop_requested

    try:
        if mode in {"fixed_rule", "fixed_low_rl"}:
            env = TerminalParallelEnv(config)
            rollout_env = env
            observations, _ = env.reset(seed)
            if gui:
                _, observer = _attach_viewer(env, render_config)
                if not observer.wait_ready():
                    raise RuntimeError(f"viewer failed to start: {observer.error!r}")
            if mode == "fixed_rule":
                tick_started = time.perf_counter()
                for _ in range(max_steps or int(config.max_simulation_seconds / config.dt)):
                    observations, _, _, _, _ = env.step()
                    if stopped():
                        break
                    if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
                        completed_normally = True
                        break
                    tick_started = _pace_live_rollout(observer, config.dt, tick_started)
                    if stopped():
                        break
                if not stopped():
                    completed_normally = True
            else:
                low = FrozenLowLevelPolicy(low_checkpoint, config)
                tick_started = time.perf_counter()
                for _ in range(max_steps or int(config.max_simulation_seconds / config.dt)):
                    observations, _, _, _, _ = env.step(low.act(observations))
                    if stopped():
                        break
                    if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
                        completed_normally = True
                        break
                    tick_started = _pace_live_rollout(observer, config.dt, tick_started)
                    if stopped():
                        break
                if not stopped():
                    completed_normally = True
            return env

        if mode == "dynamic_rule":
            low_level = RuleLowLevelAdapter()
            dynamic_policy = DynamicRulePolicy()
            high_policy: Callable[[np.ndarray, DynamicHighLevelWrapper], np.ndarray] = (
                lambda observation, wrapper: dynamic_policy.act(observation, wrapper._context())
            )
        elif mode == "learned_high_rule_low":
            low_level = RuleLowLevelAdapter()
            learned = load_high_level(high_checkpoint, config)
            high_policy = lambda observation, wrapper: learned.act(observation, deterministic=True)[0]
        elif mode == "learned_high_frozen_low":
            low_level = FrozenLowLevelPolicy(low_checkpoint, config)
            learned = load_high_level(high_checkpoint, config, expected_low_hash=low_level.checkpoint_hash)
            high_policy = lambda observation, wrapper: learned.act(observation, deterministic=True)[0]
        else:
            raise ValueError(f"unsupported rollout mode: {mode}")

        wrapper = DynamicHighLevelWrapper(config, low_level)
        observation, _ = wrapper.reset(seed)
        rollout_env = wrapper.base
        if gui:
            _, observer = _attach_viewer(wrapper.base, render_config)
            if not observer.wait_ready():
                raise RuntimeError(f"viewer failed to start: {observer.error!r}")
        macro_limit = max_steps or int(np.ceil(config.max_simulation_seconds / config.high_level_decision_interval_seconds))
        tick_started = time.perf_counter()
        for _ in range(macro_limit):
            observation, _, terminated, truncated, _ = wrapper.step(high_policy(observation, wrapper))
            if stopped():
                break
            if terminated or truncated:
                completed_normally = True
                break
            tick_started = _pace_live_rollout(observer, config.high_level_decision_interval_seconds, tick_started)
            if stopped():
                break
        if not stopped():
            completed_normally = True
        return wrapper.base
    finally:
        if observer is not None:
            try:
                if gui and completed_normally and rollout_env is not None and not observer.stop_requested:
                    observer.finish(rollout_env)
            finally:
                observer.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic terminal-MARL rollout with an optional PyVista viewer.")
    parser.add_argument("--mode", required=True, choices=("fixed_rule", "fixed_low_rl", "dynamic_rule", "learned_high_rule_low", "learned_high_frozen_low"))
    parser.add_argument("--scenario", choices=("default", "fixed_rule_warm_flow"), default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--playback-speed", type=float, default=0.0, help="0 is unthrottled; positive values are simulated seconds per wall-clock second.")
    parser.add_argument("--trajectories", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--low-checkpoint")
    parser.add_argument("--high-checkpoint")
    parser.add_argument("--max-simulation-seconds", type=float, default=3600.0)
    parser.add_argument("--max-steps", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    env = run_rollout(
        mode=args.mode, seed=args.seed, gui=args.gui, frame_stride=args.frame_stride,
        playback_speed=args.playback_speed, show_trajectories=args.trajectories,
        show_labels=args.labels, low_checkpoint=args.low_checkpoint, high_checkpoint=args.high_checkpoint,
        max_simulation_seconds=args.max_simulation_seconds, max_steps=args.max_steps,
        scenario=args.scenario,
    )
    print({"time": env.now, "completed": len(env.completed_uavs), "outcomes": dict(env.ledger.outcomes), "telemetry": dict(env.telemetry)})


if __name__ == "__main__":
    main()
