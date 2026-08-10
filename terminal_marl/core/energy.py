"""Normalized, phase-dependent SOC accounting for Phase A."""
from __future__ import annotations
from dataclasses import dataclass
from .config import SimConfig
from .uav import FlightPhase

@dataclass(frozen=True)
class EnergyModel:
    config: SimConfig
    def phase_power(self, phase: FlightPhase | str) -> float:
        values = {
            FlightPhase.EXTERNAL_QUEUE: self.config.power_external_wait,
            FlightPhase.INBOUND: self.config.power_inbound,
            FlightPhase.RING_OUTER: self.config.power_ring_hold,
            FlightPhase.RING_MIDDLE: self.config.power_ring_hold,
            FlightPhase.RING_INNER: self.config.power_ring_hold,
            FlightPhase.TRANSFER_O2M: self.config.power_ring_transfer,
            FlightPhase.TRANSFER_M2I: self.config.power_ring_transfer,
            FlightPhase.FINAL: self.config.power_final,
            FlightPhase.LANDING: self.config.power_landing,
        }
        return self.config.power_reference * values.get(phase, 0.0)
    def consume(self, state, phase: FlightPhase, dt: float) -> None:
        used = max(0.0, self.phase_power(phase) * dt)
        state.energy_remaining = max(0.0, state.energy_remaining - used)
        state.energy_consumed += used
    def estimate_to_complete(self, state, *_args) -> float:
        # Conservative engineering estimate; it intentionally is not an endurance countdown.
        return self.config.power_reference * 120.0
