"""Single-source lifecycle transition validation and outcome ledger."""
from __future__ import annotations
from dataclasses import dataclass, field
from .uav import FlightPhase, TerminalOutcome, UAVState

ALLOWED_TRANSITIONS = {
    FlightPhase.EXTERNAL_QUEUE: {FlightPhase.INBOUND, FlightPhase.DIVERTED},
    FlightPhase.INBOUND: {FlightPhase.RING_OUTER, FlightPhase.DIVERTED, FlightPhase.FAILED},
    FlightPhase.RING_OUTER: {FlightPhase.TRANSFER_O2M, FlightPhase.DIVERTED, FlightPhase.FAILED},
    FlightPhase.TRANSFER_O2M: {FlightPhase.RING_MIDDLE, FlightPhase.FAILED},
    FlightPhase.RING_MIDDLE: {FlightPhase.TRANSFER_M2I, FlightPhase.DIVERTED, FlightPhase.FAILED},
    FlightPhase.TRANSFER_M2I: {FlightPhase.RING_INNER, FlightPhase.FAILED},
    FlightPhase.RING_INNER: {FlightPhase.FINAL, FlightPhase.DIVERTED, FlightPhase.FAILED},
    FlightPhase.FINAL: {FlightPhase.LANDING, FlightPhase.FAILED},
    FlightPhase.LANDING: {FlightPhase.SERVICE, FlightPhase.FAILED},
    FlightPhase.SERVICE: {FlightPhase.COMPLETED},
}

def transition(uav: UAVState, target: FlightPhase, now: float) -> None:
    if target not in ALLOWED_TRANSITIONS.get(uav.phase, set()):
        raise ValueError(f"illegal lifecycle transition {uav.phase.value}->{target.value}")
    uav.phase = target
    uav.phase_entry_time = now
    if target is FlightPhase.RING_OUTER:
        uav.outer_entry_time = now
    if target is FlightPhase.FINAL:
        uav.final_entry_time = now
    if target is FlightPhase.LANDING:
        uav.touchdown_time = now

@dataclass
class OutcomeLedger:
    requested: set[int] = field(default_factory=set)
    outcomes: dict[int, TerminalOutcome] = field(default_factory=dict)

    def request(self, arrival_id: int) -> None: self.requested.add(arrival_id)
    def resolve(self, arrival_id: int, outcome: TerminalOutcome) -> None:
        if arrival_id not in self.requested: raise ValueError("unknown arrival outcome")
        if arrival_id in self.outcomes: raise ValueError("duplicate terminal outcome")
        self.outcomes[arrival_id] = outcome
    @property
    def unresolved(self) -> set[int]: return self.requested - set(self.outcomes)
    def closes(self) -> bool: return not self.unresolved
