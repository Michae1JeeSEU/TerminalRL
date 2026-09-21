"""Normalized, phase-dependent SOC accounting for Phase A."""
from __future__ import annotations
from dataclasses import dataclass
from .config import SimConfig
from .uav import FlightPhase

@dataclass(frozen=True)
class EnergyModel:
    config: SimConfig

    @property
    def min_landing_soc(self) -> float:
        """Vehicle-profile SOC floor below which touchdown is not successful."""
        return self.config.min_landing_soc

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

    def projected_soc(self, state, phase: FlightPhase | str, dt: float) -> float:
        """SOC after the same phase accounting used by :meth:`consume`."""
        used = max(0.0, self.phase_power(phase) * max(0.0, float(dt)))
        return max(0.0, (state.energy_remaining - used) / state.energy_capacity)

    def has_landing_reserve(self, state) -> bool:
        """Whether the current state still meets the profile landing floor."""
        return state.soc + 1e-12 >= self.min_landing_soc

    def projected_landing_reserve(self, state, phase: FlightPhase | str, dt: float) -> bool:
        """Whether a touchdown at the end of this tick remains serviceable."""
        return self.projected_soc(state, phase, dt) + 1e-12 >= self.min_landing_soc

    def estimate_to_complete(self, state, *_args) -> float:
        # Conservative engineering estimate; it intentionally is not an endurance countdown.
        return self.config.power_reference * 120.0
