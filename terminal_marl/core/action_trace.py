"""Action provenance across policy, teacher, execution, and shield layers."""
from __future__ import annotations

from dataclasses import dataclass

from .merge_scheduler import LowLevelIntent


@dataclass
class LowLevelActionTrace:
    requested: LowLevelIntent
    teacher_corrected: LowLevelIntent | None = None
    execution_applied: LowLevelIntent | None = None
    shield_applied: LowLevelIntent | None = None
    teacher_used: bool = False
    shield_used: bool = False
    teacher_kind: str | None = None
    shield_reason: str | None = None
    correction_magnitude: float = 0.0

    def __post_init__(self) -> None:
        if self.execution_applied is None:
            self.execution_applied = self.requested

    def as_dict(self) -> dict:
        encode = lambda action: None if action is None else [action.release_request, action.speed_residual, action.merge_time_residual]
        return {
            "requested_low_action": encode(self.requested),
            "teacher_corrected_action": encode(self.teacher_corrected),
            "execution_applied_action": encode(self.execution_applied),
            "shield_applied_action": encode(self.shield_applied),
            "teacher_used": self.teacher_used, "shield_used": self.shield_used,
            "teacher_kind": self.teacher_kind, "shield_reason": self.shield_reason,
            "correction_magnitude": self.correction_magnitude,
        }
