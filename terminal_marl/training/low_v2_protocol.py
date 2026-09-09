"""Versioned, fail-closed protocol definitions for future A5-v2 training.

No optimizer is invoked here.  These objects make the intended curriculum and
its teacher authority explicit before any concrete low-level training starts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.config import SimConfig
from ..core.observations import LOW_OBS_V2


@dataclass(frozen=True)
class TeacherPhase:
    name: str
    scheduler_mode: str
    teacher_mode: str
    teacher_executes: bool


LOW_V2_TEACHER_CURRICULUM = (
    TeacherPhase("teacher_full", "teacher_full", "teacher_full", True),
    TeacherPhase("teacher_immediate", "teacher_immediate", "teacher_immediate", True),
    TeacherPhase("teacher_label_only", "teacher_label_only", "teacher_label_only", False),
    TeacherPhase("teacher_off", "execution", "teacher_label_only", False),
)


def phase_config(base: SimConfig, phase: TeacherPhase) -> SimConfig:
    """Return the explicit config for a curriculum phase without running it."""
    if phase.name == "teacher_off":
        return replace(base, experiment_mode="fixed_low_student", high_level_enabled=False,
                       scheduler_mode="execution", teacher_enabled=False, teacher_mode="teacher_label_only")
    return replace(base, experiment_mode="fixed_teacher_heavy", high_level_enabled=False,
                   scheduler_mode=phase.scheduler_mode, teacher_mode=phase.teacher_mode, teacher_enabled=True)


def validate_low_v2_config(config: SimConfig, *, for_training: bool) -> None:
    if config.observation_schema_version != "phaseA_low_obs_v2_topology_87d":
        raise ValueError("A5-v2 requires the canonical 87D topology observation")
    if config.action_schema_version not in {"low_intent_v3_factorized_discrete", "low_action_schema_v4_joint_discrete"}:
        raise ValueError("low-level action schema must be an explicitly versioned discrete catalogue")
    if not config.shield_enabled:
        raise ValueError("A5-v2 requires PredictiveSafetyShield enabled")
    if for_training and config.scheduler_mode == "execution":
        raise ValueError("training launch must select an explicit teacher phase or use student qualification")
    if not for_training and (config.teacher_enabled or config.scheduler_mode != "execution"):
        raise ValueError("student qualification requires execution scheduler with teacher OFF")
    if LOW_OBS_V2.dimension != 87:
        raise RuntimeError("unexpected V2 observation schema drift")
