"""Training-free closure tests for the A5-v2 low-level architecture."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from terminal_marl.core.config import SimConfig
from terminal_marl.core.config import RingRole
from terminal_marl.core.execution_scheduler import ExecutionScheduler
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeScheduler
from terminal_marl.core.observations import LOW_OBS_V2, build_low_observation_v2
from terminal_marl.core.teacher_scheduler import TeacherScheduler
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _student_env(count: int = 3) -> TerminalParallelEnv:
    config = SimConfig(
        experiment_mode="fixed_low_student", reset_mode="warm_start_rings", warm_start_uavs=count,
        fixed_arrival_schedule=tuple(0.0 for _ in range(count)), total_requested_uavs=count,
        max_simulation_seconds=30.0,
    )
    env = TerminalParallelEnv(config)
    env.reset(7)
    return env


def test_v2_schema_is_named_87d_and_no_neighbor_is_zero_masked():
    env = _student_env(1)
    vector = next(iter(env._observations().values()))
    assert LOW_OBS_V2.dimension == 87
    assert vector.shape == (87,)
    for name in ("same_ring_leader", "same_ring_follower", "target_ring_predecessor", "target_ring_successor", "nearest_predicted_conflict"):
        assert np.array_equal(vector[LOW_OBS_V2.slices[name]], np.zeros(9, np.float32))


def test_v2_topology_observation_is_invariant_to_active_mapping_order():
    env = _student_env(6)
    uid, uav = next(iter(env.active.items()))
    original = build_low_observation_v2(uav, env)
    env.active = OrderedDict(reversed(list(env.active.items())))
    reordered = build_low_observation_v2(env.active[uid], env)
    assert np.array_equal(original, reordered)


def test_execution_scheduler_has_one_requested_candidate_and_no_teacher_search():
    env = _student_env(1)
    scheduler = ExecutionScheduler(env.config, MergeScheduler(env.config))
    assert not hasattr(scheduler, "repair_candidates")
    uav = next(iter(env.active.values()))
    proposal = scheduler.propose_requested(uav, LowLevelIntent(1.0, 0.0, 0.0), WorldSnapshot(env.now, env.active, env.airspace), RingRole.MIDDLE)
    assert scheduler.check_requested(proposal).approved


def test_teacher_is_rejected_for_formal_student_preset_and_available_for_training_preset():
    with pytest.raises(ValueError, match="teacher_enabled"):
        SimConfig(experiment_mode="fixed_low_student", teacher_enabled=True)
    teacher_config = SimConfig(experiment_mode="fixed_teacher_heavy", scheduler_mode="teacher_full", teacher_enabled=True)
    teacher = TeacherScheduler(teacher_config, MergeScheduler(teacher_config))
    assert teacher.candidates(LowLevelIntent(1.0, 0.0, 0.0))[0].kind == "requested"


def test_step_emits_requested_execution_shield_provenance():
    env = _student_env(2)
    observations, _, _, _, info = env.step({uid: np.asarray((0.0, 0.0, 0.0), np.float32) for uid in env.agents})
    assert observations
    for trace in info["action_traces"].values():
        assert trace["requested_low_action"] == trace["execution_applied_action"]
        assert trace["teacher_corrected_action"] is None
        assert trace["teacher_used"] is False
