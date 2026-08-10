"""Reject checkpoints that do not declare the Phase-A v1 schemas."""
from __future__ import annotations
from .config import SimConfig
import torch
class IncompatibleCheckpointError(ValueError): pass
def validate_checkpoint_metadata(metadata:dict, config:SimConfig|None=None)->None:
    c=config or SimConfig()
    expected={"schema_version":c.schema_version,"airspace_schema_version":c.airspace_schema_version,"action_schema_version":c.action_schema_version,"high_action_schema_version":c.high_action_schema_version,"observation_schema_version":c.observation_schema_version}
    mismatch={k:(metadata.get(k),v) for k,v in expected.items() if metadata.get(k)!=v}
    if mismatch:raise IncompatibleCheckpointError(f"incompatible checkpoint schema: {mismatch}")

def checkpoint_metadata(config: SimConfig | None = None) -> dict:
    c = config or SimConfig()
    return {"schema_version": c.schema_version, "airspace_schema_version": c.airspace_schema_version,
            "action_schema_version": c.action_schema_version, "high_action_schema_version": c.high_action_schema_version,
            "observation_schema_version": c.observation_schema_version, "low_level_action_dim": 3,
            "high_level_enabled": False}

def save_checkpoint(path, policy, config: SimConfig, **extra) -> None:
    torch.save({"metadata": checkpoint_metadata(config), "model": policy.model.state_dict(), "optimizer": policy.optimizer.state_dict(), **extra}, path)

def load_checkpoint(path, policy, config: SimConfig | None = None):
    payload = torch.load(path, map_location=policy.device, weights_only=False)
    validate_checkpoint_metadata(payload.get("metadata", {}), config)
    policy.model.load_state_dict(payload["model"])
    if "optimizer" in payload: policy.optimizer.load_state_dict(payload["optimizer"])
    return payload
