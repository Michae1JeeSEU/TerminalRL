"""Small, reproducible V3 action-catalogue safety integration qualification."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from ..core.config import SimConfig
from ..core.low_level_action import ACTION_SCHEMA_V3, LowLevelActionV3
from ..envs.parallel_env import TerminalParallelEnv


@dataclass(frozen=True)
class V3SafetyQualification:
    passed: bool
    steps: int
    requested_actions: int
    teacher_actions: int
    shield_actions: int
    collisions: int
    nmac: int
    hard_separation_violations: int
    physics_dt: float
    safety_sampling_dt: float
    action_schema_version: str


def qualify_v3_safety(config: SimConfig | None = None, *, seed: int = 0, max_steps: int = 20) -> V3SafetyQualification:
    """Exercise execution, teacher candidates, and shield with only V3 values."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    root = config or SimConfig(reset_mode="warm_start_rings", warm_start_uavs=8,
                               fixed_arrival_schedule=(0.0,) * 8, total_requested_uavs=8,
                               max_simulation_seconds=float(max_steps), teacher_enabled=True,
                               scheduler_mode="teacher_full", teacher_mode="teacher_full")
    if root.action_schema_version != ACTION_SCHEMA_V3:
        raise ValueError("V3 safety qualification requires the V3 action schema")
    if root.physics_dt != .2 or root.safety_sampling_dt != 1.0:
        raise ValueError("V3 safety qualification freezes production sampling intervals")
    env = TerminalParallelEnv(root)
    observations, _ = env.reset(seed)
    requested = teacher = shield = 0
    for step in range(max_steps):
        actions = {
            uid: LowLevelActionV3((step + index) % 2, (step + index) % 5, (step + 2 * index) % 5).normalized
            for index, uid in enumerate(observations)
        }
        observations, _, _, _, info = env.step(actions)
        for trace in info.get("action_traces", {}).values():
            requested += 1
            if "legacy_continuous_action" in trace["requested_action_v3"]:
                raise AssertionError("V3 requested action escaped the catalogue")
            for name in ("teacher_action_v3", "execution_action_v3", "shield_action_v3"):
                value = trace.get(name)
                if value is not None and "legacy_continuous_action" in value:
                    raise AssertionError(f"{name} escaped the V3 catalogue")
            teacher += int(bool(trace.get("teacher_used")))
            shield += int(bool(trace.get("shield_used")))
        if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
            break
    result = V3SafetyQualification(
        passed=all(int(env.telemetry.get(name, 0)) == 0 for name in ("collisions", "nmac", "hard_separation_violations")),
        steps=env.step_count, requested_actions=requested, teacher_actions=teacher, shield_actions=shield,
        collisions=int(env.telemetry.get("collisions", 0)), nmac=int(env.telemetry.get("nmac", 0)),
        hard_separation_violations=int(env.telemetry.get("hard_separation_violations", 0)),
        physics_dt=root.physics_dt, safety_sampling_dt=root.safety_sampling_dt,
        action_schema_version=root.action_schema_version,
    )
    return result


def write_v3_safety_qualification(path: str | Path, **kwargs) -> V3SafetyQualification:
    result = qualify_v3_safety(**kwargs)
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result
