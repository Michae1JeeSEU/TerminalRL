"""Route-energy and pad-workload estimates for the opt-in static V3 profile.

These are scheduling estimates, never safety certificates or pad reservations.
Every actual route still passes the canonical commit and runtime safety checks.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from .config import RingRole
from .energy_reachability import _advance_ring_holder, _copy_state_at_ring, _future_ring_holder
from .merge_scheduler import MergeReservation, MergeScheduler
from .trajectory_time_parameterization import TrajectoryInfeasibleError
from .uav import MISSION_SERVICE_SECONDS, UAVState


class ExecutableLandingBudget:
    def __init__(self, config, generator, energy, airspace, lz_manager):
        self.config, self.generator, self.energy = config, generator, energy
        self.airspace, self.lz_manager = airspace, lz_manager
        self._solo_cache = {}

    def holder_budget(self, uav, now):
        """Minimum available route budget including reported pad-release wait."""
        if uav.ring_role is None or uav.committed_plan is not None:
            return math.inf
        best = math.inf
        for zone in self.lz_manager.zones:
            if uav.mission_type not in zone.compatible_missions:
                continue
            ready = max(float(now), self.lz_manager.earliest_release_time(zone.lz_id, now=now))
            if not math.isfinite(ready):
                continue
            holder = _copy_state_at_ring(uav, uav.ring_role, now)
            holder = _advance_ring_holder(holder, uav.ring_role, ready, self.airspace, self.generator)
            try:
                route = self.generator.emergency_landing(holder, zone, self.airspace, ready)
            except TrajectoryInfeasibleError:
                continue
            required = (self.energy.emergency_required_energy(uav, route)
                        + (ready-now)*self.energy.phase_power(uav.phase))
            best = min(best, required)
        return best

    def _solo_route(self, queued, gate, zone):
        """Actual V3 inbound -> middle -> inner -> final energy and duration.

        Static geometry permits caching by gate and pad. No traffic is removed
        from the actual environment: this detached estimate makes no safety
        claim and commits/reserves nothing. The cached numbers exclude reserve.
        """
        key = (gate.gate_id, zone.lz_id)
        if key in self._solo_cache:
            return self._solo_cache[key]
        uav = UAVState.from_request(-1, queued.request, queued, gate.spawn_point, 0.0)
        try:
            plan = self.generator.inbound(uav, gate, self.airspace, 0.0)
            used = self.energy.estimate_plan_energy(plan)
            uav = _future_ring_holder(uav, plan, RingRole.OUTER)
            now = plan.end_time
            scheduler = MergeScheduler(self.config)
            for target in (RingRole.MIDDLE, RingRole.INNER):
                snapshot = SimpleNamespace(now=now, active={}, airspace=self.airspace)
                duration = scheduler.transfer_duration_estimate(uav, target, snapshot)
                # The real generator closes the endpoint; the surrogate only
                # seeds the search. No estimated slot is entered in a ledger.
                reservation = MergeReservation('energy-estimate', -1, uav.phase.value,
                    target.value, now+6.0+duration, uav.ring_progress+math.pi/2, None, None)
                plan = self.generator.ring_transfer(uav, uav.ring_role, target,
                    reservation, self.airspace, now, 0.0)
                used += self.energy.estimate_plan_energy(plan)
                uav = _future_ring_holder(uav, plan, target)
                now = plan.end_time
            reservation = MergeReservation('energy-estimate', -1, uav.phase.value,
                'final', now, None, None, None)
            plan = self.generator.final_approach(uav, zone, reservation, now,
                speed_residual_mps=0.0, airspace=self.airspace)
            used += self.energy.estimate_plan_energy(plan)
            result = (float(used), float(plan.end_time))
        except TrajectoryInfeasibleError:
            result = (math.inf, math.inf)
        self._solo_cache[key] = result
        return result

    def admission_budget(self, queued, gate, now, active, pending):
        """Executable solo cost plus a compatible-pad workload estimate.

        Busy/reserved pad ETAs are authoritative. Unassigned aircraft are
        allocated once to a virtual calendar in SOC/request order; accepted
        requests in this admission batch are included. This estimate does not
        guarantee a future collision-free route and is not mutated into a real
        reservation. Admission capacity and actual inbound safety remain separate.
        """
        ready = {z.lz_id: max(now, self.lz_manager.earliest_release_time(z.lz_id, now=now))
                 for z in self.lz_manager.zones}
        demand = [(u.soc, u.request_time, u.arrival_id, u.mission_type)
                  for u in active.values() if u.assigned_lz is None and not u.emergency_landing]
        demand += [(q.soc, q.request.request_time, q.request.arrival_id, q.request.mission_type)
                   for q, _gate in pending]
        for _soc, _time, _id, mission in sorted(demand):
            compatible = [z for z in self.lz_manager.zones if mission in z.compatible_missions]
            if compatible:
                zone = min(compatible, key=lambda z: (ready[z.lz_id], z.lz_id))
                ready[zone.lz_id] += MISSION_SERVICE_SECONDS[mission] + self.config.dt
        requirements = []
        for zone in self.lz_manager.zones:
            if queued.request.mission_type not in zone.compatible_missions:
                continue
            used, travel = self._solo_route(queued, gate, zone)
            if not math.isfinite(used) or not math.isfinite(ready[zone.lz_id]):
                continue
            wait = max(0.0, ready[zone.lz_id]-now-travel)
            requirements.append(used + wait*self.energy.phase_power('ring_inner')
                + self.energy.min_landing_soc*queued.request.energy_capacity
                + self.config.emergency_energy_margin_seconds*self.config.power_reference)
        return min(requirements, default=math.inf)
