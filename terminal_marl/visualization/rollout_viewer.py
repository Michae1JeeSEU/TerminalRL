"""Deterministic rollout command for the optional terminal 3D viewer."""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from ..core.config import SimConfig
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, FrozenLowLevelPolicy, RuleLowLevelAdapter
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import DynamicRulePolicy
from ..training.high_level_a6 import load_high_level
from .config import VisualizationConfig
from .pyvista_viewer import Terminal3DViewer, ViewerStepObserver
from .snapshot import build_world_render_snapshot


def _config_for_mode(mode: str, max_simulation_seconds: float) -> SimConfig:
    dynamic = mode in {"dynamic_rule", "learned_high_rule_low", "learned_high_frozen_low"}
    return SimConfig(experiment_mode=mode, high_level_enabled=dynamic, max_simulation_seconds=max_simulation_seconds)


def _attach_viewer(env: TerminalParallelEnv, config: VisualizationConfig) -> tuple[Terminal3DViewer, ViewerStepObserver]:
    viewer = Terminal3DViewer(build_world_render_snapshot(env), config)
    observer = ViewerStepObserver(viewer, config)
    env.set_step_observer(observer)
    return viewer, observer


def run_rollout(
    *, mode: str, seed: int, gui: bool = False, frame_stride: int = 1,
    playback_speed: float = 0.0, show_trajectories: bool = True, show_labels: bool = False,
    low_checkpoint: str | None = None, high_checkpoint: str | None = None,
    max_simulation_seconds: float = 3600.0, max_steps: int | None = None,
) -> TerminalParallelEnv:
    """Run one deterministic policy rollout, optionally with an interactive viewer."""
    if mode == "fixed_low_rl" and low_checkpoint is None:
        raise ValueError("--low-checkpoint is required for fixed_low_rl")
    if mode == "learned_high_rule_low" and high_checkpoint is None:
        raise ValueError("--high-checkpoint is required for learned_high_rule_low")
    if mode == "learned_high_frozen_low" and (low_checkpoint is None or high_checkpoint is None):
        raise ValueError("--low-checkpoint and --high-checkpoint are required for learned_high_frozen_low")
    config = _config_for_mode(mode, max_simulation_seconds)
    render_config = VisualizationConfig(
        enabled=gui, frame_stride=frame_stride, playback_speed=playback_speed,
        show_trajectories=show_trajectories, show_labels=show_labels,
    )
    viewer: Terminal3DViewer | None = None
    observer: ViewerStepObserver | None = None

    def stopped() -> bool:
        return observer is not None and observer.stop_requested

    try:
        if mode in {"fixed_rule", "fixed_low_rl"}:
            env = TerminalParallelEnv(config)
            observations, _ = env.reset(seed)
            if gui:
                viewer, observer = _attach_viewer(env, render_config)
            if mode == "fixed_rule":
                for _ in range(max_steps or int(config.max_simulation_seconds / config.dt)):
                    observations, _, _, _, _ = env.step()
                    if stopped() or (env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs):
                        break
            else:
                low = FrozenLowLevelPolicy(low_checkpoint, config)
                for _ in range(max_steps or int(config.max_simulation_seconds / config.dt)):
                    observations, _, _, _, _ = env.step(low.act(observations))
                    if stopped() or (env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs):
                        break
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
        if gui:
            viewer, observer = _attach_viewer(wrapper.base, render_config)
        macro_limit = max_steps or int(np.ceil(config.max_simulation_seconds / config.high_level_decision_interval_seconds))
        for _ in range(macro_limit):
            observation, _, terminated, truncated, _ = wrapper.step(high_policy(observation, wrapper))
            if stopped() or terminated or truncated:
                break
        return wrapper.base
    finally:
        if viewer is not None:
            viewer.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic terminal-MARL rollout with an optional PyVista viewer.")
    parser.add_argument("--mode", required=True, choices=("fixed_rule", "fixed_low_rl", "dynamic_rule", "learned_high_rule_low", "learned_high_frozen_low"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--playback-speed", type=float, default=0.0, help="0 is unthrottled; positive values are simulated seconds per wall-clock second.")
    parser.add_argument("--no-trajectories", action="store_true")
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
        playback_speed=args.playback_speed, show_trajectories=not args.no_trajectories,
        show_labels=args.labels, low_checkpoint=args.low_checkpoint, high_checkpoint=args.high_checkpoint,
        max_simulation_seconds=args.max_simulation_seconds, max_steps=args.max_steps,
    )
    print({"time": env.now, "completed": len(env.completed_uavs), "outcomes": dict(env.ledger.outcomes), "telemetry": dict(env.telemetry)})


if __name__ == "__main__":
    main()
