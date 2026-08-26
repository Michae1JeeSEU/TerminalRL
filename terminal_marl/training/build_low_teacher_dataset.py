"""Generate auditable, training-only A5-v2 teacher-correction datasets.

This module does not optimize a policy.  It merely records the environment
mapping from a requested intention to a teacher-corrected execution, together
with the predictive-safety context needed to audit a later BC warm start.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from ..core.config import SimConfig
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import RuleLowLevelPolicy


class LowPolicy(Protocol):
    def act(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True)
class TeacherDatasetSummary:
    rows: int
    corrected_rows: int
    no_correction_rows: int
    shield_rows: int
    output_path: str
    observation_schema_version: str
    scheduler_mode: str
    seed: int


def _teacher_config(config: SimConfig) -> SimConfig:
    if config.experiment_mode in {"fixed_low_student", "dynamic_hrl_student"}:
        raise ValueError("formal student/deployment configuration cannot generate teacher data")
    return replace(config, experiment_mode="fixed_teacher_heavy", high_level_enabled=False,
                   scheduler_mode="teacher_full", teacher_mode="teacher_full", teacher_enabled=True)


def _finite_or_none(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def build_low_teacher_dataset(
    config: SimConfig | None = None, *, output_path: str | Path = "runs/low_v2_teacher_dataset.jsonl",
    episodes: int = 1, max_steps: int | None = None, seed: int = 0, policy: LowPolicy | None = None,
) -> TeacherDatasetSummary:
    """Collect balanced-ready teacher labels without running PPO or BC.

    The generated JSONL keeps `requested_low_action` separate from the label.
    Consumers must balance corrected/no-correction rows explicitly; this
    function reports the distribution rather than silently oversampling one
    class.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    runtime_config = _teacher_config(config or SimConfig())
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    actor = policy or RuleLowLevelPolicy()
    rows: list[dict] = []
    step_limit = max_steps or int(runtime_config.max_simulation_seconds / runtime_config.dt)
    for episode in range(episodes):
        env = TerminalParallelEnv(runtime_config)
        observations, _ = env.reset(seed + episode)
        for step in range(step_limit):
            actions = actor.act(observations)
            risk_before = env.runtime_traffic.assess(env)
            next_observations, rewards, terminated, truncated, info = env.step(actions)
            for agent_id, observation in observations.items():
                trace = info["action_traces"].get(agent_id)
                if trace is None:
                    continue
                rows.append({
                    "episode": episode, "step": step, "agent_id": agent_id,
                    "observation": np.asarray(observation, dtype=float).tolist(),
                    "requested_action": trace["requested_low_action"],
                    "teacher_action": trace["teacher_corrected_action"],
                    "execution_action": trace["execution_applied_action"],
                    "shield_action": trace["shield_applied_action"],
                    "teacher_used": trace["teacher_used"], "teacher_kind": trace["teacher_kind"],
                    "correction_magnitude": trace["correction_magnitude"],
                    "predicted_min_sep_before": _finite_or_none(risk_before.predicted_min_swept_separation),
                    "d_cpa_before": _finite_or_none(risk_before.d_cpa), "t_cpa_before": _finite_or_none(risk_before.t_cpa),
                    "predicted_min_sep_after": _finite_or_none(env.safety.last_predicted_track_audit["minimum_separation"]),
                    "release_decision": info["release_decisions"].get(agent_id),
                    "reward": float(rewards.get(agent_id, 0.0)),
                    "terminated": bool(terminated.get(agent_id, False)),
                    "truncated": bool(truncated.get(agent_id, False)),
                })
            observations = next_observations
            if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
                break
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    corrected = sum(bool(row["teacher_used"]) for row in rows)
    shield = sum(row["shield_action"] is not None for row in rows)
    return TeacherDatasetSummary(len(rows), corrected, len(rows) - corrected, shield, str(destination),
                                 runtime_config.observation_schema_version, runtime_config.scheduler_mode, seed)


if __name__ == "__main__":
    print(asdict(build_low_teacher_dataset()))
