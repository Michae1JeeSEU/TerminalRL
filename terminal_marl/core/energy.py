"""Normalized, phase-dependent SOC accounting for Phase A."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
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
        if self.config.energy_model_mode == "phase_motion_v2":
            # Research surrogate, normalized to cruise. Ratios are anchored
            # to the EHang 184 scheduling study (36 kW cruise, 42 kW hover,
            # 28 kW *marginal delay* in shallow descent); they are not
            # measured EH216-S powers. Vertical descent uses hover-like
            # power, never a zero-energy assumption.
            ratios = {
                FlightPhase.EXTERNAL_QUEUE: 42.0 / 36.0,
                FlightPhase.INBOUND: 1.0,
                FlightPhase.RING_OUTER: 1.0,
                FlightPhase.RING_MIDDLE: 1.0,
                FlightPhase.RING_INNER: 1.0,
                FlightPhase.TRANSFER_O2M: 28.0 / 36.0,
                FlightPhase.TRANSFER_M2I: 28.0 / 36.0,
                FlightPhase.FINAL: 28.0 / 36.0,
                FlightPhase.LANDING: 42.0 / 36.0,
                FlightPhase.EMERGENCY_LANDING: 42.0 / 36.0,
            }
            return self.config.power_reference * ratios.get(phase, 0.0)
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
            FlightPhase.EMERGENCY_LANDING: self.config.power_landing,
        }
        return self.config.power_reference * values.get(phase, 0.0)
    def consume(self, state, phase: FlightPhase, dt: float) -> None:
        used = max(0.0, self.phase_power(phase) * dt)
        self.consume_units(state, used)

    @staticmethod
    def consume_units(state, used: float) -> None:
        used = max(0.0, float(used))
        state.energy_remaining = max(0.0, state.energy_remaining - used)
        state.energy_consumed += used

    def motion_power(self, velocity, *, emergency: bool = False) -> float:
        """Normalized instantaneous power for the opt-in motion surrogate."""
        horizontal = float(np.linalg.norm(np.asarray(velocity, dtype=float)[:2]))
        vertical = float(np.asarray(velocity, dtype=float)[2])
        if self.config.energy_model_mode == "legacy_phase":
            return self.config.power_reference
        if vertical > 0.2:
            ratio = 1.30  # uncalibrated conservative climb sensitivity
        elif vertical < -0.2 and horizontal > 2.0:
            ratio = 28.0 / 36.0
        elif horizontal <= 2.0:
            ratio = 42.0 / 36.0
        else:
            ratio = 1.0
        return self.config.power_reference * ratio

    def estimate_plan_energy(self, plan, start: float | None = None, end: float | None = None) -> float:
        """Integrate executable trajectory power over an interval."""
        left = plan.start_time if start is None else max(plan.start_time, float(start))
        right = plan.end_time if end is None else min(plan.end_time, float(end))
        if right <= left:
            return 0.0
        boundaries = [left, *(p.t for p in plan.points if left < p.t < right), right]
        return sum((b - a) * self.motion_power(plan.sample(.5 * (a + b)).velocity,
                    emergency=plan.mode == "emergency_landing") for a, b in zip(boundaries, boundaries[1:]))

    def emergency_required_energy(self, state, plan) -> float:
        return (self.min_landing_soc * state.energy_capacity
                + self.estimate_plan_energy(plan)
                + self.config.emergency_energy_margin_seconds * self.config.power_reference)

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
        if self.config.energy_model_mode == "phase_motion_v2":
            # Frozen from the current 4-gate x 4-mission congestion-free solo
            # matrix (maximum 240.9 normalized units). This is an admission
            # reachability bound, not an EH216-S measured endurance curve.
            return self.config.phase_motion_nominal_solo_energy
        # Historical compatibility estimate; it intentionally is not an endurance countdown.
        return self.config.power_reference * 120.0

    def admission_required_energy(self, state) -> float:
        required = self.estimate_to_complete(state)
        if self.config.energy_model_mode == "phase_motion_v2":
            required += self.min_landing_soc * state.request.energy_capacity
        return required
