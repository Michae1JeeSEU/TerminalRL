from dataclasses import replace

import pytest

from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.run_v4_2_hierarchical_s0_qualification import _config
from terminal_marl.training.v4_1_joint_teacher import (
    STATE_ONLY_JOINT_CATALOGUE_V4_1,
    teacher_joint_expert_v4_1,
)


def _config_for_teacher(enabled: bool):
    config = _config()
    if enabled:
        return config
    return replace(
        config,
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        teacher_mode="teacher_label_only",
        scheduler_mode="execution",
        teacher_release_label_enabled=False,
    )


def _fixture(enabled: bool):
    env = TerminalParallelEnv(_config_for_teacher(enabled))
    observations, _ = env.reset(3)
    uav = env.active[int(next(iter(observations)))]
    return env, uav


def test_state_only_catalogue_matches_teacher_scheduler_order(monkeypatch):
    teacher_env, teacher_uav = _fixture(True)
    state_env, state_uav = _fixture(False)
    assert teacher_env.teacher_scheduler is not None
    assert state_env.teacher_scheduler is None

    expected = tuple(row.index for row in teacher_env.teacher_scheduler.joint_catalogue())
    actual = tuple(row.index for row in STATE_ONLY_JOINT_CATALOGUE_V4_1)
    assert actual == expected

    for env in (teacher_env, state_env):
        monkeypatch.setattr(
            env, "_candidate_plan",
            lambda *_args: (object(), object(), None),
        )
    old_label, old_audit = teacher_joint_expert_v4_1(
        teacher_env, teacher_uav, audit_catalogue=True,
    )
    state_label, state_audit = teacher_joint_expert_v4_1(
        state_env, state_uav, audit_catalogue=True, state_only=True,
    )
    assert state_label.index == old_label.index
    assert [row["joint_index"] for row in state_audit] == [
        row["joint_index"] for row in old_audit
    ]


def test_state_only_rejects_teacher_enabled_and_default_still_requires_scheduler():
    teacher_env, teacher_uav = _fixture(True)
    with pytest.raises(ValueError, match="state-only Teacher"):
        teacher_joint_expert_v4_1(teacher_env, teacher_uav, state_only=True)

    state_env, state_uav = _fixture(False)
    with pytest.raises(RuntimeError, match="TeacherScheduler"):
        teacher_joint_expert_v4_1(state_env, state_uav)


def test_state_only_requires_execution_scheduler():
    config = replace(_config(), scheduler_mode="teacher_label_only")
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(3)
    uav = env.active[int(next(iter(observations)))]
    with pytest.raises(ValueError, match="state-only Teacher"):
        teacher_joint_expert_v4_1(env, uav, state_only=True)
