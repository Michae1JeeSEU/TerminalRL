"""Gym-style high-level HRL wrapper over ``TerminalParallelEnv``."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from ..config import ExperimentMode
from ..core.specs import BoxSpec
from ..rewards.high_level_reward import HighLevelReward, high_level_snapshot
from .parallel_env import TerminalParallelEnv

try:
    from gymnasium.spaces import Box
except ImportError:  # pragma: no cover - exercised only in minimal deployments.
    Box = None


LowLevelPolicy = Callable[
    [TerminalParallelEnv, Mapping[str, np.ndarray]],
    Mapping[str, np.ndarray],
]


class HighLevelHRLWrapper:
    """Expose one high-level action per block of low-level simulation steps."""

    def __init__(
        self,
        env: TerminalParallelEnv,
        *,
        low_level_policy: str | LowLevelPolicy = "rule-only",
        reward_model: HighLevelReward | None = None,
        seed: int | None = None,
    ) -> None:
        if env.experiment_mode not in {
            ExperimentMode.DYNAMIC_HRL,
            ExperimentMode.FULL_HRL_NO_SAFETY,
        }:
            raise ValueError("HighLevelHRLWrapper requires a dynamic HRL mode")
        if isinstance(low_level_policy, str) and low_level_policy not in {
            "rule-only",
            "intention-residual",
            "random-policy",
        }:
            raise ValueError("unsupported low_level_policy")
        self.env = env
        self.low_level_policy = low_level_policy
        self.reward_model = reward_model or HighLevelReward()
        self.rng = np.random.default_rng(seed)
        observation_shape = (len(self.env.get_high_level_observation()),)
        if Box is not None:
            self.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=observation_shape,
                dtype=np.float32,
            )
            self.action_space = Box(
                low=-1.0,
                high=1.0,
                shape=(8,),
                dtype=np.float32,
            )
        else:
            self.observation_space = BoxSpec(
                shape=observation_shape,
                low=np.full(observation_shape, -np.inf, dtype=np.float32),
                high=np.full(observation_shape, np.inf, dtype=np.float32),
            )
            self.action_space = BoxSpec(
                shape=(8,),
                low=-np.ones(8, dtype=np.float32),
                high=np.ones(8, dtype=np.float32),
            )
        self._low_observations: Mapping[str, np.ndarray] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._low_observations, _ = self.env.reset(seed=seed)
        self.reward_model.reset()
        return self.env.get_high_level_observation(), {
            "low_level_step": self.env.step_count,
            "experiment_mode": self.env.experiment_mode.value,
        }

    def _actions(self) -> Mapping[str, np.ndarray]:
        if callable(self.low_level_policy):
            return self.low_level_policy(self.env, self._low_observations)
        size = self.env.action_spec(self.env.possible_agents[0]).shape[0]
        if self.low_level_policy == "random-policy":
            return {
                agent: self.rng.uniform(-1.0, 1.0, size=size).astype(np.float32)
                for agent in self.env.agents
            }
        return {
            agent: np.zeros(size, dtype=np.float32)
            for agent in self.env.agents
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        before = high_level_snapshot(self.env)
        old_airspace = self.env.airspace.state_vector()
        action_info = self.env.apply_high_level_action(action)
        low_steps = 0
        for _ in range(self.env.config.high_level_interval_steps):
            if not self.env.agents:
                break
            (
                self._low_observations,
                _rewards,
                terminations,
                truncations,
                _infos,
            ) = self.env.step(self._actions())
            low_steps += 1
            if all(
                terminations.get(agent, False) or truncations.get(agent, False)
                for agent in self.env.possible_agents
            ):
                break
        after = high_level_snapshot(self.env)
        reward, diagnostics = self.reward_model.compute(
            before,
            after,
            old_airspace,
            self.env.airspace.state_vector(),
        )
        terminated = bool(self.env.aircraft) and all(
            aircraft.finished for aircraft in self.env.aircraft
        )
        truncated = self.env.step_count >= self.env.config.max_steps and not terminated
        return (
            self.env.get_high_level_observation(),
            reward,
            terminated,
            truncated,
            {
                "high_level_action": action_info,
                "reward_diagnostics": diagnostics,
                "low_level_steps": low_steps,
                "metrics": self.env.metrics(),
            },
        )
