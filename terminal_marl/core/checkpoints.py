"""Reject checkpoints that do not declare the Phase-A v1 schemas."""
from __future__ import annotations
from .config import SimConfig
from .observations import observation_schema_for_version
from .low_level_joint_action import N_JOINT_ACTIONS
import torch
class IncompatibleCheckpointError(ValueError): pass
def _joint_schema(config: SimConfig) -> tuple[int, str]:
    if config.action_schema_version == "low_action_schema_v4_1_joint_discrete_26":
        return 26, "joint_categorical_v4_1"
    if config.action_schema_version == "low_action_schema_v4_2_hierarchical":
        return 26, "hierarchical_gate_2_conditional_proposal_25"
    return N_JOINT_ACTIONS, "joint_categorical"
def validate_checkpoint_metadata(metadata:dict, config:SimConfig|None=None)->None:
    c=config or SimConfig()
    joint_count, distribution = _joint_schema(c)
    schema=observation_schema_for_version(c.observation_schema_version)
    expected={"schema_version":c.schema_version,"airspace_schema_version":c.airspace_schema_version,
              "vehicle_dynamics_schema_version":c.vehicle_dynamics_schema_version,
              "trajectory_schema_version":c.trajectory_schema_version,
              "vehicle_profile_id":c.vehicle_profile_id,
              "vehicle_profile_version":c.vehicle_profile.profile_version,
              "action_schema_version":c.action_schema_version,"high_action_schema_version":c.high_action_schema_version,
              "observation_schema_version":c.observation_schema_version,"observation_dimension":schema.dimension,
              "observation_feature_hash":schema.feature_hash,
              "low_level_joint_action_count": joint_count,
              "low_level_policy_distribution": distribution}
    mismatch={k:(metadata.get(k),v) for k,v in expected.items() if metadata.get(k)!=v}
    if mismatch:raise IncompatibleCheckpointError(f"incompatible checkpoint schema: {mismatch}")

def checkpoint_metadata(config: SimConfig | None = None, *, policy_kind: str = "low_level", low_level_checkpoint_hash: str | None = None) -> dict:
    c = config or SimConfig()
    joint_count, distribution = _joint_schema(c)
    schema = observation_schema_for_version(c.observation_schema_version)
    return {"schema_version": c.schema_version, "airspace_schema_version": c.airspace_schema_version,
            "vehicle_dynamics_schema_version": c.vehicle_dynamics_schema_version,
            "trajectory_schema_version": c.trajectory_schema_version,
            "vehicle_profile_id": c.vehicle_profile_id,
            "vehicle_profile_version": c.vehicle_profile.profile_version,
            "action_schema_version": c.action_schema_version, "high_action_schema_version": c.high_action_schema_version,
            "observation_schema_version": c.observation_schema_version, "observation_dimension": schema.dimension,
            "observation_feature_hash": schema.feature_hash, "low_level_action_dim": 3,
            "low_level_joint_action_count": joint_count,
            "low_level_policy_distribution": distribution,
            "high_level_action_dim": 10, "high_level_enabled": c.high_level_enabled,
            "policy_kind": policy_kind, "low_level_checkpoint_hash": low_level_checkpoint_hash,
            "scheduler_mode": c.scheduler_mode, "teacher_enabled": c.teacher_enabled}

def save_checkpoint(path, policy, config: SimConfig, **extra) -> None:
    torch.save({"metadata": checkpoint_metadata(config), "model": policy.model.state_dict(), "optimizer": policy.optimizer.state_dict(), **extra}, path)

def load_checkpoint(path, policy, config: SimConfig | None = None, *,
                    allow_policy_migration: bool = False, load_optimizer: bool = True):
    payload = torch.load(path, map_location=policy.device, weights_only=False)
    validate_checkpoint_metadata(payload.get("metadata", {}), config)
    if hasattr(policy, "load_model_state"):
        migrated = policy.load_model_state(payload["model"], allow_architecture_migration=allow_policy_migration)
    else:
        policy.model.load_state_dict(payload["model"])
        migrated = False
    if "optimizer" in payload and load_optimizer:
        optimizer_state = payload["optimizer"]
        if migrated:
            current = policy.optimizer.state_dict()
            if len(optimizer_state["param_groups"]) != 1 or len(current["param_groups"]) != 1:
                raise RuntimeError("conditional actor optimizer migration requires one parameter group")
            optimizer_state = {"state": dict(optimizer_state["state"]),
                               "param_groups": [dict(optimizer_state["param_groups"][0])]}
            optimizer_state["param_groups"][0]["params"] = current["param_groups"][0]["params"]
        policy.optimizer.load_state_dict(optimizer_state)
    payload["policy_architecture_migrated"] = migrated
    return payload
