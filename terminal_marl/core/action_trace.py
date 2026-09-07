"""Action provenance across policy, teacher, execution, and shield layers."""
from __future__ import annotations

from dataclasses import dataclass

from .merge_scheduler import LowLevelIntent
from .low_level_action import LowLevelActionV3


@dataclass
class LowLevelActionTrace:
    requested: LowLevelIntent
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

    def __post_init__(self) -> None:
        if self.execution_applied is None:
            self.execution_applied = self.requested

    def as_dict(self) -> dict:
        encode = lambda action: None if action is None else [action.release_request, action.speed_residual, action.merge_time_residual]
        def v3(action):
            if action is None:
                return None
            try:
                return LowLevelActionV3.from_normalized(encode(action)).as_dict()
            except ValueError:
                # Archived continuous callers can still be inspected, but
                # cannot be treated as an exact V3 policy/teacher label.
                return {"legacy_continuous_action": encode(action)}
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
            "requested_action_v3": v3(self.requested),
            "teacher_action_v3": v3(self.teacher_corrected),
            "execution_action_v3": v3(self.execution_applied),
            "shield_action_v3": v3(self.shield_applied),
        }
