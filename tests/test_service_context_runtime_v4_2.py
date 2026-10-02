from __future__ import annotations

import numpy as np
import pytest
import torch

from terminal_marl.algo.joint_ppo_v4_2 import JointPPOV42
from terminal_marl.algo.service_context_runtime_v4_2 import (
    SOURCE_ACTION_DISTRIBUTION,
    SOURCE_JOINT_ACTION_COUNT,
    initialize_service_context_policy,
)
from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2
from terminal_marl.core.observations import LOW_OBS_V2_1, LOW_OBS_V2_4
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import student_rollout_v4_2 as student


SOURCE_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
TARGET_SCHEMA = "phaseA_low_obs_v2_4_service_context_90d"


def _source_metadata() -> dict[str, object]:
    return {
        "observation_schema_version": SOURCE_SCHEMA,
        "observation_dimension": 88,
        "observation_feature_hash": LOW_OBS_V2_1.feature_hash,
        "action_schema_version": ACTION_SCHEMA_V4_2,
        "low_level_joint_action_count": SOURCE_JOINT_ACTION_COUNT,
        "low_level_policy_distribution": SOURCE_ACTION_DISTRIBUTION,
    }


def _source_policy() -> JointPPOV42:
    torch.manual_seed(20261002)
    policy = JointPPOV42(88)
    policy.model.eval()
    return policy


def test_runtime_initialization_preserves_outputs_state_rng_and_metadata() -> None:
    source = _source_policy()
    before_state = {
        key: value.detach().clone()
        for key, value in source.model.state_dict().items()
    }
    torch.manual_seed(20261003)
    rng_before = torch.random.get_rng_state().clone()

    target, metadata = initialize_service_context_policy(
        source,
        source_checkpoint_metadata=_source_metadata(),
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert target.observation_dim == 90
    assert target.device == torch.device("cpu")
    assert not hasattr(target, "optimizer")
    assert metadata["source_observation_schema"] == SOURCE_SCHEMA
    assert metadata["source_observation_dim"] == 88
    assert metadata["source_observation_feature_hash"] == LOW_OBS_V2_1.feature_hash
    assert metadata["target_observation_schema"] == TARGET_SCHEMA
    assert metadata["target_observation_dim"] == 90
    assert metadata["target_observation_feature_hash"] == LOW_OBS_V2_4.feature_hash
    assert metadata["migration_initialization"] == "zero_initialized_appended_service_columns"
    assert metadata["checkpoint_weights_unchanged"] is True
    assert metadata["optimizer_created"] is False

    for key, value in before_state.items():
        assert torch.equal(value, source.model.state_dict()[key]), key

    observations = torch.randn(4, 88)
    service_context = torch.randn(4, 2)
    world = torch.randn(2, 3, 88)
    world_service_context = torch.randn(2, 3, 2)
    mask = torch.ones((2, 3), dtype=torch.bool)
    with torch.no_grad():
        source_gate, source_proposal = source.model.distributions(observations)
        source_value = source.model.critic(world, mask)
        target_gate, target_proposal = target.model.distributions(
            torch.cat((observations, service_context), dim=-1)
        )
        target_value = target.model.critic(
            torch.cat((world, world_service_context), dim=-1), mask
        )
    assert torch.allclose(source_gate.logits, target_gate.logits, atol=1e-6, rtol=0.0)
    assert torch.allclose(source_proposal.logits, target_proposal.logits, atol=1e-6, rtol=0.0)
    assert torch.allclose(source_value, target_value, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observation_schema_version", "phaseA_low_obs_v2_3_source_bound_88d"),
        ("observation_dimension", 90),
        ("observation_feature_hash", "wrong-feature-hash"),
        ("action_schema_version", "low_action_schema_v4_joint_discrete"),
        ("low_level_joint_action_count", 30),
        ("low_level_policy_distribution", "joint_categorical"),
    ),
)
def test_runtime_initialization_rejects_wrong_checkpoint_metadata(field, value) -> None:
    metadata = _source_metadata()
    metadata[field] = value
    with pytest.raises(ValueError, match="metadata"):
        initialize_service_context_policy(
            _source_policy(),
            source_checkpoint_metadata=metadata,
        )


def test_runtime_initialization_rejects_missing_metadata_and_unsupported_policy() -> None:
    metadata = _source_metadata()
    metadata.pop("observation_feature_hash")
    with pytest.raises(ValueError, match="missing required fields"):
        initialize_service_context_policy(
            _source_policy(),
            source_checkpoint_metadata=metadata,
        )

    with pytest.raises(TypeError, match="torch.nn.Module"):
        initialize_service_context_policy(
            object(),
            source_checkpoint_metadata=_source_metadata(),
        )


def test_runtime_initialization_rejects_non_88d_source_policy() -> None:
    source = JointPPOV42(90)
    with pytest.raises(ValueError, match="88D"):
        initialize_service_context_policy(
            source,
            source_checkpoint_metadata=_source_metadata(),
        )


def test_runtime_wrapper_is_compatible_with_v24_action_and_critic_inference() -> None:
    source = _source_policy()
    target, _ = initialize_service_context_policy(
        source,
        source_checkpoint_metadata=_source_metadata(),
    )
    config = SimConfig(
        action_schema_version=ACTION_SCHEMA_V4_2,
        observation_schema_version=TARGET_SCHEMA,
        experiment_mode="fixed_low_student",
        teacher_enabled=False,
        scheduler_mode="execution",
        reset_mode="warm_start_rings",
        warm_start_uavs=1,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
        max_simulation_seconds=2.0,
    )
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(20261004)

    details = student._student_request_details(
        target,
        env,
        observations,
        deterministic=True,
        shortlist_k=1,
    )
    uids, world, actor_mask, _, gates, proposals = details[:6]
    assert target.observation_dim == LOW_OBS_V2_4.dimension == 90
    assert world.shape == (len(uids), 90)
    assert actor_mask.shape == (len(uids),)
    assert gates.shape == (len(uids),)
    assert proposals.shape == (len(uids),)
    value = student.world_value(target, world)
    assert np.isfinite(value)
