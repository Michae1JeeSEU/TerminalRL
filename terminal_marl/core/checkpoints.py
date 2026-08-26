"""Reject checkpoints that do not declare the Phase-A v1 schemas."""
from __future__ import annotations
from .config import SimConfig
from .observations import observation_schema_for_version
import torch
class IncompatibleCheckpointError(ValueError): pass
def validate_checkpoint_metadata(metadata:dict, config:SimConfig|None=None)->None:
    c=config or SimConfig()
    schema=observation_schema_for_version(c.observation_schema_version)
    expected={"schema_version":c.schema_version,"airspace_schema_version":c.airspace_schema_version,"action_schema_version":c.action_schema_version,"high_action_schema_version":c.high_action_schema_version,"observation_schema_version":c.observation_schema_version,"observation_dimension":schema.dimension,"observation_feature_hash":schema.feature_hash}
    mismatch={k:(metadata.get(k),v) for k,v in expected.items() if metadata.get(k)!=v}
    if mismatch:raise IncompatibleCheckpointError(f"incompatible checkpoint schema: {mismatch}")

def checkpoint_metadata(config: SimConfig | None = None, *, policy_kind: str = "low_level", low_level_checkpoint_hash: str | None = None) -> dict:
    c = config or SimConfig()
    schema = observation_schema_for_version(c.observation_schema_version)
    return {"schema_version": c.schema_version, "airspace_schema_version": c.airspace_schema_version,
            "action_schema_version": c.action_schema_version, "high_action_schema_version": c.high_action_schema_version,
            "observation_schema_version": c.observation_schema_version, "observation_dimension": schema.dimension,
            "observation_feature_hash": schema.feature_hash, "low_level_action_dim": 3,
            "high_level_action_dim": 10, "high_level_enabled": c.high_level_enabled,
            "policy_kind": policy_kind, "low_level_checkpoint_hash": low_level_checkpoint_hash,
            "scheduler_mode": c.scheduler_mode, "teacher_enabled": c.teacher_enabled}

def save_checkpoint(path, policy, config: SimConfig, **extra) -> None:
    torch.save({"metadata": checkpoint_metadata(config), "model": policy.model.state_dict(), "optimizer": policy.optimizer.state_dict(), **extra}, path)

def load_checkpoint(path, policy, config: SimConfig | None = None):
    payload = torch.load(path, map_location=policy.device, weights_only=False)
    validate_checkpoint_metadata(payload.get("metadata", {}), config)
    policy.model.load_state_dict(payload["model"])
    if "optimizer" in payload: policy.optimizer.load_state_dict(payload["optimizer"])
    return payload
