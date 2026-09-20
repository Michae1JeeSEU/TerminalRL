"""Action provenance across policy, teacher, execution, and shield layers."""
from __future__ import annotations

from dataclasses import dataclass

from .merge_scheduler import LowLevelIntent
from .low_level_action import LowLevelActionV3
from .low_level_joint_action import joint_index_from_normalized
from .low_level_joint_action_v4_1 import ACTION_SCHEMA_V4_1, joint_index_from_normalized_v4_1
from .low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, hierarchical_from_normalized_v4_2


@dataclass
class LowLevelActionTrace:
    requested: LowLevelIntent
    action_schema_version: str | None = None
    teacher_corrected: LowLevelIntent | None = None
    execution_applied: LowLevelIntent | None = None
    shield_applied: LowLevelIntent | None = None
    teacher_used: bool = False
    shield_used: bool = False
    teacher_kind: str | None = None
    teacher_delay_seconds: float | None = None
    teacher_witness_plan_id: str | None = None
    teacher_verified: bool = False
    shield_reason: str | None = None
    correction_magnitude: float = 0.0
    runtime_shortlist_used: bool = False
    runtime_shortlist_rank: int | None = None
    runtime_shortlist_candidates_checked: int = 0

    def __post_init__(self) -> None:
        if self.execution_applied is None:
            self.execution_applied = self.requested

    def as_dict(self) -> dict:
        encode = lambda action: None if action is None else [float(action.release_request), float(action.speed_residual), float(action.merge_time_residual)]
        def v3(action):
            if action is None:
                return None
            try:
                return LowLevelActionV3.from_normalized(encode(action)).as_dict()
            except ValueError:
                # Archived continuous callers can still be inspected, but
                # cannot be treated as an exact V3 policy/teacher label.
                return {"legacy_continuous_action": encode(action)}
        def v4(action):
            if action is None:
                return None
            try:
                if self.action_schema_version == ACTION_SCHEMA_V4_2:
                    return hierarchical_from_normalized_v4_2(encode(action)).as_dict()
                encoder = joint_index_from_normalized_v4_1 if self.action_schema_version == ACTION_SCHEMA_V4_1 else joint_index_from_normalized
                index = int(encoder(encode(action)))
                return {"joint_action_index": index, "decoded": encode(action)}
            except ValueError:
                return {"non_catalog_action": encode(action)}
        requested_joint = v4(self.requested)
        requested_hierarchical = requested_joint if self.action_schema_version == ACTION_SCHEMA_V4_2 else None
        teacher_hierarchical = v4(self.teacher_corrected) if self.action_schema_version == ACTION_SCHEMA_V4_2 else None
        return {
            "requested_low_action": encode(self.requested),
            "teacher_corrected_action": encode(self.teacher_corrected),
            "execution_applied_action": encode(self.execution_applied),
            "shield_applied_action": encode(self.shield_applied),
            "teacher_used": self.teacher_used, "shield_used": self.shield_used,
            "teacher_kind": self.teacher_kind, "teacher_delay_seconds": self.teacher_delay_seconds,
            "teacher_witness_plan_id": self.teacher_witness_plan_id, "teacher_verified": self.teacher_verified,
            "shield_reason": self.shield_reason,
            "correction_magnitude": self.correction_magnitude,
            "runtime_shortlist_used": self.runtime_shortlist_used,
            "runtime_shortlist_rank": self.runtime_shortlist_rank,
            "runtime_shortlist_candidates_checked": self.runtime_shortlist_candidates_checked,
            "action_schema_version": self.action_schema_version,
            "requested_action_v3": v3(self.requested),
            "teacher_action_v3": v3(self.teacher_corrected),
            "execution_action_v3": v3(self.execution_applied),
            "shield_action_v3": v3(self.shield_applied),
            "joint_action_index": None if requested_joint is None else requested_joint.get("joint_action_index", requested_joint.get("joint_index_v4_1")),
            "decoded_joint_action": requested_joint,
            "requested_hierarchical_action": requested_hierarchical,
            "teacher_hierarchical_action": teacher_hierarchical,
            "teacher_joint_action": v4(self.teacher_corrected),
            "execution_applied_joint_action": v4(self.execution_applied),
            "shield_applied_joint_action": v4(self.shield_applied),
        }
