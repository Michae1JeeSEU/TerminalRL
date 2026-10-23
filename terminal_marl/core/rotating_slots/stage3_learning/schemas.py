"""Portable records for FT-RS-S3-INITIAL-1.0."""
from dataclasses import asdict, dataclass

DESCRIPTOR_SCHEMA_ID = "work_descriptor_v1_0"
FEATURE_SCHEMA_ID = "candidate_context_sets_v1_0"
FEATURE_COST_POLICY_ID = "pre_materialization_only_v1_0"


@dataclass(frozen=True)
class WorkDescriptor:
    descriptor_id: str
    snapshot_id: str
    source_node_id: str
    aircraft_id: str
    source_ring: str
    source_slot: int
    target_kind: str
    target_id: int
    start_offset_s: float
    duration_s: float
    parent_prefix_id: str | None
    construction_stage: str
    proposal_origin: str
    inherited_order: int

    def as_dict(self):
        return dict(record_type="WorkDescriptor", **asdict(self))
