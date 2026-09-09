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
from ..core.low_level_action import ACTION_SCHEMA_V3
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


@dataclass(frozen=True)
class TeacherDatasetCohort:
    name: str
    config: SimConfig
    episodes: int
    max_steps: int | None = None


@dataclass(frozen=True)
class TeacherDatasetMatrixSummary:
    rows: int
    corrected_rows: int
    correction_fraction: float
    output_path: str
    cohorts: tuple[str, ...]


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
                    "action_schema_version": ACTION_SCHEMA_V3,
                    "episode": episode, "step": step, "agent_id": agent_id,
                    "seed": seed + episode, "active_agent_count": len(observations),
                    "observation": np.asarray(observation, dtype=float).tolist(),
                    "requested_action": trace["requested_low_action"],
                    "requested_action_v3": trace["requested_action_v3"],
                    "teacher_action": trace["teacher_corrected_action"],
                    "teacher_action_v3": trace["teacher_action_v3"],
                    "execution_action": trace["execution_applied_action"],
                    "shield_action": trace["shield_applied_action"],
                    "teacher_used": trace["teacher_used"], "teacher_kind": trace["teacher_kind"],
                    "teacher_delay_seconds": trace["teacher_delay_seconds"],
                    "teacher_witness_plan_id": trace["teacher_witness_plan_id"],
                    "teacher_verified": trace["teacher_verified"],
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
    summary = TeacherDatasetSummary(len(rows), corrected, len(rows) - corrected, shield, str(destination),
                                    runtime_config.observation_schema_version, runtime_config.scheduler_mode, seed)
    destination.with_suffix(destination.suffix + ".summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary


def build_low_teacher_dataset_matrix(
    cohorts: tuple[TeacherDatasetCohort, ...], *, output_path: str | Path, seed: int = 0,
    policy: LowPolicy | None = None, min_corrected_fraction: float = 0.05,
) -> TeacherDatasetMatrixSummary:
    """Collect named fixed-airspace cohorts and fail if labels collapse.

    Cohorts are intentionally explicit: later training manifests can state the
    exact N/demand coverage rather than inferring it from an opaque JSONL.
    """
    if not cohorts:
        raise ValueError("at least one teacher dataset cohort is required")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    offset = 0
    for cohort in cohorts:
        if not cohort.name or cohort.episodes <= 0:
            raise ValueError("each cohort needs a name and positive episode count")
        part = destination.with_name(f"{destination.stem}.{cohort.name}.jsonl")
        build_low_teacher_dataset(cohort.config, output_path=part, episodes=cohort.episodes,
                                  max_steps=cohort.max_steps, seed=seed + offset, policy=policy)
        parts.append(part)
        offset += cohort.episodes
    destination.write_text("".join(part.read_text(encoding="utf-8") for part in parts), encoding="utf-8")
    from .audit_low_teacher_dataset import audit_teacher_dataset
    audit = audit_teacher_dataset(destination, min_corrected_fraction=min_corrected_fraction)
    if not audit.passed:
        raise RuntimeError("teacher dataset matrix audit failed: " + "; ".join(audit.errors))
    manifest = {
        "output_path": str(destination), "seed": seed, "rows": audit.rows,
        "corrected_rows": audit.corrected_rows, "correction_fraction": audit.correction_fraction,
        "cohorts": [{"name": cohort.name, "episodes": cohort.episodes, "max_steps": cohort.max_steps,
                     "warm_start_uavs": cohort.config.warm_start_uavs, "total_requested_uavs": cohort.config.total_requested_uavs,
                     "arrival_model": cohort.config.arrival_model} for cohort in cohorts],
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return TeacherDatasetMatrixSummary(audit.rows, audit.corrected_rows, audit.correction_fraction,
                                       str(destination), tuple(cohort.name for cohort in cohorts))


if __name__ == "__main__":
    print(asdict(build_low_teacher_dataset()))
