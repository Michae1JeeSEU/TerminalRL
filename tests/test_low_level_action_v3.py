"""V3 action semantics and PPO provenance regression tests."""
from __future__ import annotations

import numpy as np
import torch

from terminal_marl.algo.shared_ppo import RolloutBuffer, SharedPPO, Transition
from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_action import (
    ACTION_SCHEMA_V3, HEAD_CARDINALITIES, LowLevelActionV3, RESIDUAL_VALUES,
    indices_to_normalized, normalized_to_indices, quantize_normalized,
)
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeScheduler
from terminal_marl.core.teacher_scheduler import TeacherScheduler
from terminal_marl.training.low_action_v3_safety_qualification import qualify_v3_safety


def test_v3_catalog_is_factorized_and_normalized():
    assert HEAD_CARDINALITIES == (2, 5, 5)
    assert LowLevelActionV3(0, 0, 0).normalized.tolist() == [0.0, -1.0, -1.0]
    assert LowLevelActionV3(1, 4, 4).normalized.tolist() == [1.0, 1.0, 1.0]
    assert np.array_equal(normalized_to_indices(indices_to_normalized([[0, 2, 4], [1, 0, 3]])), [[0, 2, 4], [1, 0, 3]])
    assert SimConfig().action_schema_version == ACTION_SCHEMA_V3


def test_teacher_quantization_is_unique_and_does_not_change_endpoint_semantics():
    assert np.array_equal(quantize_normalized((1.0, 0.74, -0.74)), (1.0, 0.5, -0.5))
    assert LowLevelActionV3.quantized_teacher((-1.0, 1.0, -1.0)).normalized.tolist() == [0.0, 1.0, -1.0]
    config = SimConfig(teacher_enabled=True)
    teacher = TeacherScheduler(config, MergeScheduler(config))
    candidates = teacher.candidates(LowLevelIntent(1.0, .41, -.41))
    assert candidates
    for row in candidates:
        assert row.intent.release_request in (0.0, 1.0)
        assert row.intent.speed_residual in RESIDUAL_VALUES
        assert row.intent.merge_time_residual in RESIDUAL_VALUES


def test_categorical_sampling_joint_logprob_and_deterministic_evaluation_are_consistent():
    torch.manual_seed(12)
    policy = SharedPPO(7)
    observations = np.zeros((4, 7), np.float32)
    batch = policy.act_details(observations, observations)
    assert batch.categories.shape == batch.normalized.shape == (4, 3)
    assert np.array_equal(normalized_to_indices(batch.normalized), batch.categories)
    with torch.no_grad():
        distribution = policy.model.actor_distribution(torch.as_tensor(observations))
        expected = distribution.log_prob_per_head(torch.as_tensor(batch.categories)).sum(-1).numpy()
    assert np.allclose(batch.joint_log_prob, expected)
    first = policy.act_details(observations, observations, deterministic=True)
    second = policy.act_details(observations, observations, deterministic=True)
    assert np.array_equal(first.categories, second.categories)
    assert np.array_equal(first.normalized, second.normalized)


def test_ppo_buffer_uses_requested_categories_and_categorical_bc_is_finite():
    torch.manual_seed(13)
    policy = SharedPPO(5, teacher_bc_weight=.1)
    observation = np.zeros((1, 5), np.float32)
    batch = policy.act_details(observation, observation)
    transition = Transition("agent", observation[0], observation, batch.categories[0], float(batch.joint_log_prob[0]),
                            batch.value, 1.0, True, teacher_action=np.asarray((1, 4, 0)),
                            applied_action=np.asarray((1.0, 1.0, -1.0)), teacher_used=True,
                            requested_normalized_action=batch.normalized[0], old_head_log_prob=batch.head_log_prob[0])
    buffer = RolloutBuffer(); buffer.add(transition)
    assert np.array_equal(buffer.by_agent["agent"][0].action, batch.categories[0])
    assert np.array_equal(buffer.by_agent["agent"][0].requested_normalized_action, batch.normalized[0])
    metrics = policy.update(buffer, epochs=1, minibatch_size=1)
    assert np.isfinite(metrics["teacher_bc_loss"])
    assert set(("entropy_release", "entropy_speed", "entropy_merge_time", "approx_kl", "clip_fraction")) <= set(metrics)


def test_v3_discrete_actions_pass_small_production_sampling_safety_integration():
    result = qualify_v3_safety(seed=17, max_steps=4)
    assert result.passed
    assert result.requested_actions > 0
    assert result.physics_dt == .2 and result.safety_sampling_dt == 1.0
