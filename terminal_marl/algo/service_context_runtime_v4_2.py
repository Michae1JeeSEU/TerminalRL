"""Read-only 88D to 90D service-context inference initialization for V4.2.

This module keeps the checkpoint loading boundary explicit.  A caller supplies
an already-loaded 88D policy and its checkpoint metadata; the helper validates
the semantic V2.1/V4.2 binding, initializes the existing 90D model migration,
and returns only the attributes consumed by the student rollout inference path.
No optimizer is created or copied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2
from ..core.observations import LOW_OBS_V2_1, LOW_OBS_V2_4
from .service_context_migration_v4_2 import (
    SOURCE_OBSERVATION_DIM,
    SOURCE_OBSERVATION_SCHEMA,
    TARGET_OBSERVATION_DIM,
    migrate_service_context_model,
)


SOURCE_ACTION_DISTRIBUTION = "hierarchical_gate_2_conditional_proposal_25"
SOURCE_JOINT_ACTION_COUNT = 26


@dataclass(frozen=True)
class ServiceContextInferencePolicyV42:
    """Minimal policy surface required by ``student_rollout_v4_2``.

    The wrapper deliberately has no optimizer or update method.  The rollout
    accesses ``model``, ``observation_dim`` and ``device`` for action and
    centralized-critic inference.
    """

    model: nn.Module
    observation_dim: int
    device: torch.device


def _validate_source_checkpoint_metadata(
    metadata: Mapping[str, Any],
) -> None:
    if not isinstance(metadata, Mapping):
        raise TypeError("source_checkpoint_metadata must be a mapping")

    expected = {
        "observation_schema_version": SOURCE_OBSERVATION_SCHEMA,
        "observation_dimension": SOURCE_OBSERVATION_DIM,
        "observation_feature_hash": LOW_OBS_V2_1.feature_hash,
        "action_schema_version": ACTION_SCHEMA_V4_2,
        "low_level_joint_action_count": SOURCE_JOINT_ACTION_COUNT,
        "low_level_policy_distribution": SOURCE_ACTION_DISTRIBUTION,
    }
    missing = sorted(key for key in expected if key not in metadata)
    if missing:
        raise ValueError(
            "source checkpoint metadata is missing required fields: "
            f"{missing}"
        )
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "source checkpoint metadata is incompatible with V4.2 V2.1/88D: "
            f"{mismatches}"
        )


def _source_model(source_policy: Any) -> nn.Module:
    model = getattr(source_policy, "model", None)
    if not isinstance(model, nn.Module):
        raise TypeError("source_policy must expose a torch.nn.Module model")
    observation_dim = getattr(source_policy, "observation_dim", None)
    if isinstance(observation_dim, bool) or not isinstance(observation_dim, int):
        raise TypeError("source_policy.observation_dim must be an integer")
    if observation_dim != SOURCE_OBSERVATION_DIM:
        raise ValueError(
            f"source_policy must use {SOURCE_OBSERVATION_DIM}D observations, "
            f"got {observation_dim}"
        )
    return model


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _state_unchanged(
    before: Mapping[str, torch.Tensor],
    model: nn.Module,
) -> bool:
    after = model.state_dict()
    return (
        list(before) == list(after)
        and all(torch.equal(before[key], after[key]) for key in before)
    )


def initialize_service_context_policy(
    source_policy: Any,
    *,
    source_checkpoint_metadata: Mapping[str, Any],
) -> tuple[ServiceContextInferencePolicyV42, dict[str, Any]]:
    """Initialize a read-only 90D inference wrapper from a validated 88D policy.

    ``source_policy`` must already contain model weights loaded from a
    checkpoint.  The metadata is checked before the model migration so missing
    or semantically different 88D schemas cannot be accepted by position.
    """

    _validate_source_checkpoint_metadata(source_checkpoint_metadata)
    source_model = _source_model(source_policy)
    source_state = _snapshot_state(source_model)

    target_model, migration_metadata = migrate_service_context_model(
        source_model,
        source_observation_schema=source_checkpoint_metadata[
            "observation_schema_version"
        ],
    )
    source_weights_unchanged = _state_unchanged(source_state, source_model)
    if not source_weights_unchanged:
        raise RuntimeError("source policy weights changed during service-context initialization")

    target_parameters = tuple(target_model.parameters())
    if not target_parameters:
        raise RuntimeError("migrated service-context model has no parameters")
    target_devices = {parameter.device for parameter in target_parameters}
    if len(target_devices) != 1:
        raise RuntimeError(
            "migrated service-context model parameters span multiple devices"
        )
    target_device = next(iter(target_devices))
    wrapper = ServiceContextInferencePolicyV42(
        model=target_model,
        observation_dim=TARGET_OBSERVATION_DIM,
        device=target_device,
    )

    metadata = dict(migration_metadata)
    metadata.update({
        "runtime_wrapper": "service_context_inference_v4_2",
        "source_observation_feature_hash": LOW_OBS_V2_1.feature_hash,
        "target_observation_feature_hash": LOW_OBS_V2_4.feature_hash,
        "source_action_schema_version": ACTION_SCHEMA_V4_2,
        "source_action_distribution": SOURCE_ACTION_DISTRIBUTION,
        "source_joint_action_count": SOURCE_JOINT_ACTION_COUNT,
        "migration_initialization": "zero_initialized_appended_service_columns",
        "source_policy_weights_unchanged": source_weights_unchanged,
        "checkpoint_weights_unchanged": source_weights_unchanged,
        "optimizer_created": False,
        "inference_only": True,
    })
    return wrapper, metadata


__all__ = [
    "SOURCE_ACTION_DISTRIBUTION",
    "SOURCE_JOINT_ACTION_COUNT",
    "ServiceContextInferencePolicyV42",
    "initialize_service_context_policy",
]
