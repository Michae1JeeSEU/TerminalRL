"""Reject checkpoints that do not declare the Phase-A v1 schemas."""
from __future__ import annotations
from .config import SimConfig
class IncompatibleCheckpointError(ValueError): pass
def validate_checkpoint_metadata(metadata:dict, config:SimConfig|None=None)->None:
    c=config or SimConfig()
    expected={"schema_version":c.schema_version,"airspace_schema_version":c.airspace_schema_version,"action_schema_version":c.action_schema_version,"high_action_schema_version":c.high_action_schema_version,"observation_schema_version":c.observation_schema_version}
    mismatch={k:(metadata.get(k),v) for k,v in expected.items() if metadata.get(k)!=v}
    if mismatch:raise IncompatibleCheckpointError(f"incompatible checkpoint schema: {mismatch}")
