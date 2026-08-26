"""Training-only bounded repair search and correction witnesses."""
from __future__ import annotations

from dataclasses import dataclass

from .merge_scheduler import LowLevelIntent, RepairCandidate, MergeScheduler


@dataclass(frozen=True)
class TeacherCorrection:
    requested: LowLevelIntent
    corrected: LowLevelIntent
    candidate_kind: str
    delay_seconds: float
    correction_magnitude: float
    verified: bool = False
    witness_plan_id: str | None = None


class TeacherScheduler:
    """Owns every active repair search; never construct in formal modes."""
    def __init__(self, config, scheduler: MergeScheduler) -> None:
        if not config.teacher_enabled:
            raise ValueError("TeacherScheduler requires teacher_enabled=True")
        self.config = config
        self._scheduler = scheduler

    def candidates(self, requested: LowLevelIntent) -> tuple[RepairCandidate, ...]:
        rows = self._scheduler.repair_candidates(requested)
        if self.config.teacher_mode == "teacher_immediate":
            rows = tuple(row for row in rows if row.delay_seconds == 0.0)
        if self.config.teacher_mode == "teacher_label_only":
            return rows
        return rows

    def correction(self, requested: LowLevelIntent, candidate: RepairCandidate, *, verified: bool = False,
                   witness_plan_id: str | None = None) -> TeacherCorrection:
        magnitude = abs(candidate.intent.speed_residual-requested.speed_residual) + abs(candidate.intent.merge_time_residual-requested.merge_time_residual) + candidate.delay_seconds/max(1.0, max(self.config.prospective_repair_delays_seconds, default=1.0))
        return TeacherCorrection(requested, candidate.intent, candidate.kind, candidate.delay_seconds, magnitude, verified, witness_plan_id)
