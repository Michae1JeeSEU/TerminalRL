"""Deployment scheduler: interpret one requested intent, never repair it."""
from __future__ import annotations

from dataclasses import dataclass

from .merge_scheduler import LowLevelIntent, MergeProposal, MergeReservation, MergeScheduler


@dataclass(frozen=True)
class SchedulerDecision:
    approved: bool
    proposal: MergeProposal
    reason: str = ""


class ExecutionScheduler:
    """Thin deployment facade over common reservation/gap primitives.

    Its deliberately small public API makes accidental candidate enumeration
    visible in code review: there is no ``repair_candidates`` method here.
    """
    def __init__(self, config, scheduler: MergeScheduler | None = None) -> None:
        self.config = config
        self._scheduler = scheduler or MergeScheduler(config)

    @property
    def reservations(self):
        return self._scheduler.reservations

    @property
    def last_release(self):
        return self._scheduler.last_release

    @property
    def last_executed_release(self):
        return self._scheduler.last_executed_release

    def propose_requested(self, uav, intent: LowLevelIntent, snapshot, target) -> MergeProposal:
        return self._scheduler.propose(uav, intent, snapshot, target)

    def check_requested(self, proposal: MergeProposal) -> SchedulerDecision:
        if not self._scheduler.can_reserve(proposal):
            return SchedulerDecision(False, proposal, "headway_conflict")
        return SchedulerDecision(True, proposal)

    def reserve_requested(self, uav, proposal: MergeProposal, *, now: float | None = None) -> MergeReservation:
        return self._scheduler.reserve(uav, proposal, now=now)

    def cancel(self, reservation_id: str, reason: str = "") -> None:
        self._scheduler.cancel(reservation_id, reason)

    def consume(self, reservation_id: str) -> None:
        self._scheduler.consume(reservation_id)

    def expire(self, now: float, active_uav_ids) -> list[str]:
        return self._scheduler.expire(now, active_uav_ids)
