"""Training-only bounded repair search and correction witnesses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .merge_scheduler import LowLevelIntent, RepairCandidate, MergeScheduler
from .low_level_action import LowLevelActionV3


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
        # The scheduler may internally reason over a continuous optimum in a
        # future implementation. Teacher labels may not expose that private
        # representation: quantize once, here, onto the policy's V3 catalog.
        quantized: list[RepairCandidate] = []
        for row in rows:
            action = LowLevelActionV3.quantized_teacher((row.intent.release_request,
                                                          row.intent.speed_residual,
                                                          row.intent.merge_time_residual))
            candidate = RepairCandidate(row.kind, LowLevelIntent(*action.normalized), row.delay_seconds)
            if candidate not in quantized:
                quantized.append(candidate)
        rows = tuple(quantized)
        if self.config.teacher_mode == "teacher_immediate":
            rows = tuple(row for row in rows if row.delay_seconds == 0.0)
        if self.config.teacher_mode == "teacher_label_only":
            return rows
        return rows

    def correction(self, requested: LowLevelIntent, candidate: RepairCandidate, *, verified: bool = False,
                   witness_plan_id: str | None = None) -> TeacherCorrection:
        magnitude = abs(candidate.intent.speed_residual-requested.speed_residual) + abs(candidate.intent.merge_time_residual-requested.merge_time_residual) + candidate.delay_seconds/max(1.0, max(self.config.prospective_repair_delays_seconds, default=1.0))
        return TeacherCorrection(requested, candidate.intent, candidate.kind, candidate.delay_seconds, magnitude, verified, witness_plan_id)

    def choose_verified(self, requested: LowLevelIntent, verify: Callable[[RepairCandidate], object]) -> TeacherCorrection | None:
        """Return the first verifier-approved bounded teacher witness.

        ``verify`` is supplied by the environment because it owns the
        canonical trajectory generator and SafetyFilter snapshot.  This keeps
        candidate search in the teacher while preserving SafetyFilter as the
        sole safety authority.
        """
        for candidate in self.candidates(requested):
            verdict = verify(candidate)
            approved = bool(getattr(verdict, "approved", verdict))
            if approved:
                return self.correction(requested, candidate, verified=True,
                                       witness_plan_id=getattr(verdict, "checked_plan_id", None))
        return None
