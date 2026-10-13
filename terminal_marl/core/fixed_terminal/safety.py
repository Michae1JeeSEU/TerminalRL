"""Central scheduler-independent feasibility and continuous spacing checks.

Uncommitted aircraft are forecast to keep holding; committed full-route plans
are fixed. Search statements are conditional on this explicit snapshot model.
An exhausted computational budget or uncertified continuous bound is UNKNOWN,
never a proof authorizing emergency. Temporal windows are half-open.
"""
from dataclasses import dataclass, replace
from math import floor, isfinite
from .domain import Segment, distance
from .routes import build_plan, hold_segment, plan_energy


@dataclass(frozen=True)
class Check:
    status: str  # feasible, infeasible, unknown
    reason: str
    searched: int = 0


@dataclass(frozen=True)
class SearchResult:
    status: str  # feasible, infeasible, unknown
    plan: object | None
    searched: int
    reason: str


def overlap(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)-1e-9


def swept_spacing(a, b, start, end, minimum, tolerance=.005):
    """Certify continuous distance; exact linear/linear and common-rate arc/arc.

    For mixed paths midpoint distance minus relative speed*half-interval is a
    rigorous lower bound. Unresolved intervals conservatively return unknown.
    """
    if end < start:
        return "feasible"
    if a.mode == b.mode == "hold" and abs(a.omega-b.omega) < 1e-12:
        return "feasible" if distance(a.position(start), b.position(start)) >= minimum-1e-9 else "infeasible"
    if a.mode != "hold" and b.mode != "hold":
        p, q = a.position(start), b.position(start)
        u, v = a.position(end), b.position(end)
        delta = [x-y for x, y in zip(p, q)]
        velocity = [(x-y)-(z-w) for x, y, z, w in zip(u, p, v, q)]
        norm = sum(x*x for x in velocity)
        fraction = min(1., max(0., -sum(x*y for x, y in zip(delta, velocity))/norm)) if norm else 0.
        closest = sum((x+fraction*y)**2 for x, y in zip(delta, velocity))**.5
        return "feasible" if closest >= minimum-1e-9 else "infeasible"
    pending = [(start, end)]
    speed = a.speed_bound+b.speed_bound
    uncertain = False
    while pending:
        lo, hi = pending.pop()
        mid = (lo+hi)/2
        d = distance(a.position(mid), b.position(mid))
        if d < minimum-1e-9:
            return "infeasible"
        if d-speed*(hi-lo)/2 >= minimum:
            continue
        if hi-lo <= tolerance:
            uncertain = True
        else:
            pending.extend(((lo, mid), (mid, hi)))
    return "unknown" if uncertain else "feasible"


def forecast(aircraft, now, end, config):
    if aircraft.plan:
        segments = list(aircraft.plan.segments)
        last = segments[-1]
        segments.append(Segment(last.end, last.end+config.pad_service, "stationary", last.p1, last.p1))
        return tuple(s for s in segments if s.end >= now and s.start <= end)
    if aircraft.state == "holding":
        return (hold_segment(aircraft, now, max(end, now+1e-8), config),)
    return ()


class Feasibility:
    def __init__(self, config):
        self.config = config

    def check_batch(self, plans, fleet, now):
        """Atomic normal-plan feasibility against the prospective entire fleet.

        Selected holders simultaneously receive their candidate routes. All
        unselected holders and previous commitments remain in the check. A
        failing/uncertified member rejects the whole batch without mutation.
        """
        originals = {a.aircraft_id: a for a in fleet}
        ids = [p.aircraft_id for p in plans]
        if len(set(ids)) != len(ids) or any(p.emergency for p in plans):
            return Check("infeasible", "batch_duplicate_or_emergency")
        if any(key not in originals or originals[key].state != "holding" for key in ids):
            return Check("infeasible", "batch_requires_observed_holders")
        selected = {p.aircraft_id: p for p in plans}
        future = [replace(a, state="scheduled", plan=selected[a.aircraft_id])
                  if a.aircraft_id in selected else a for a in fleet]
        for plan in plans:
            result = self.check(plan, originals[plan.aircraft_id], future, now, check_emergency=False)
            if result.status != "feasible":
                return result
        return Check("feasible", "joint_normal_batch")

    def check(self, plan, aircraft, fleet, now, *, check_emergency=True, search_limit=256):
        """Validate canonical route identity, energy, resources, capacities, spacing."""
        c = self.config
        try:
            canonical = build_plan(aircraft, now, c, plan.wait_cycles, plan.emergency)
        except (ValueError, TypeError, IndexError):
            return Check("infeasible", "invalid_intent")
        if plan != canonical:
            return Check("infeasible", "noncanonical_route_or_reservations")
        if not isfinite(aircraft.soc) or not 0 <= aircraft.soc <= 1:
            return Check("infeasible", "invalid_soc")
        if plan_energy(plan, c)+c.energy.reserve > aircraft.soc+1e-10:
            return Check("infeasible", "insufficient_energy_reserve")
        for segment in plan.segments:
            if segment.speed_bound > c.max_speed+1e-8:
                return Check("infeasible", "speed_limit")
            if abs(segment.p1[2]-segment.p0[2])/(segment.end-segment.start) > c.max_descent+1e-8:
                return Check("infeasible", "descent_limit")
        others = [a for a in fleet if a.aircraft_id != aircraft.aircraft_id]
        for other in others:
            if other.plan:
                for a in plan.reservations:
                    if a.resource.startswith("ring:"):
                        continue
                    for b in other.plan.reservations:
                        if a.resource == b.resource and overlap(a.start, a.end, b.start, b.end):
                            return Check("infeasible", "resource_conflict:"+a.resource)
        capacity = self._capacity(plan, others, now)
        if capacity:
            return Check("infeasible", capacity)
        uncertain = False
        # Candidate remains physically on pad through service completion.
        last = plan.segments[-1]
        candidate = plan.segments+(Segment(last.end, last.end+c.pad_service, "stationary", last.p1, last.p1),)
        for other in others:
            for b in forecast(other, now, plan.end+c.pad_service, c):
                for a in candidate:
                    lo, hi = max(a.start, b.start, now), min(a.end, b.end)
                    if hi < lo:
                        continue
                    status = swept_spacing(a, b, lo, hi, c.protected_distance, c.continuous_tolerance)
                    if status == "infeasible":
                        return Check("infeasible", "continuous_spacing:"+other.aircraft_id)
                    uncertain |= status == "unknown"
        if uncertain:
            return Check("unknown", "continuous_spacing_uncertified")
        if plan.emergency and check_emergency:
            normal = self.search_normal(aircraft, fleet, now, search_limit)
            if normal.status == "feasible":
                return Check("infeasible", "normal_plan_exists", normal.searched)
            if normal.status == "unknown":
                return Check("unknown", "normal_feasibility_unknown", normal.searched)
            return Check("feasible", "feasible", normal.searched)
        return Check("feasible", "feasible")

    def _capacity(self, plan, others, now):
        for ring in range(len(self.config.rings)):
            name = f"ring:{ring}"
            windows = [(plan.aircraft_id, r.start, r.end) for r in plan.reservations if r.resource == name]
            for a in others:
                if a.plan:
                    windows.extend((a.aircraft_id, r.start, r.end) for r in a.plan.reservations if r.resource == name)
                elif a.state == "holding" and a.ring == ring:
                    windows.append((a.aircraft_id, now, plan.end+self.config.merge_window))
            times = sorted({max(now, t) for _, lo, hi in windows for t in (lo, hi) if hi > now})
            for lo, hi in zip(times, times[1:]):
                t = (lo+hi)/2
                owners = {owner for owner, start, end in windows if start <= t < end}
                if len(owners) > self.config.operational_capacity(ring):
                    return "operational_capacity:"+str(ring)
        return None

    def search_normal(self, aircraft, fleet, now, limit=256):
        """Complete SOC-bounded domain unless explicit computation limit reached.

        Sum of extra laps is bounded by available SOC after the zero-wait route.
        Every vector with that sum is visited, earliest total service first.
        Uncertified candidates prevent an infeasible conclusion.
        """
        base = build_plan(aircraft, now, self.config)
        budget = aircraft.soc-self.config.energy.reserve-plan_energy(base, self.config)
        if budget < -1e-10:
            return SearchResult("infeasible", None, 0, "all_normal_routes_exceed_energy")
        max_laps = max(0, floor((budget+1e-10)/(self.config.lap_time*self.config.energy.hold)))
        searched, uncertain = 0, False
        for total in range(max_laps+1):
            for cycles in compositions(total, aircraft.ring+1):
                if limit is not None and searched >= limit:
                    return SearchResult("unknown", None, searched, "search_limit")
                plan = build_plan(aircraft, now, self.config, cycles)
                result = self.check(plan, aircraft, fleet, now, check_emergency=False)
                searched += 1
                if result.status == "feasible":
                    return SearchResult("feasible", plan, searched, "normal_plan_found")
                uncertain |= result.status == "unknown"
        return SearchResult("unknown" if uncertain else "infeasible", None, searched,
                            "uncertified_candidates" if uncertain else "normal_domain_exhausted")

    def search_emergency(self, aircraft, fleet, now, limit=256):
        normal = self.search_normal(aircraft, fleet, now, limit)
        if normal.status != "infeasible":
            return SearchResult("unknown" if normal.status == "unknown" else "infeasible", None,
                                normal.searched, "normal_feasibility_unknown" if normal.status == "unknown" else "normal_plan_exists")
        base = build_plan(aircraft, now, self.config, emergency=True)
        budget = aircraft.soc-self.config.energy.reserve-plan_energy(base, self.config)
        if budget < -1e-10:
            return SearchResult("infeasible", None, normal.searched, "emergency_energy_reserve")
        max_laps = max(0, floor((budget+1e-10)/(self.config.lap_time*self.config.energy.hold)))
        searched, uncertain = normal.searched, False
        for cycle in range(max_laps+1):
            if limit is not None and searched >= limit:
                return SearchResult("unknown", None, searched, "search_limit")
            plan = build_plan(aircraft, now, self.config, (cycle,), True)
            result = self.check(plan, aircraft, fleet, now, check_emergency=False)
            searched += 1
            if result.status == "feasible":
                return SearchResult("feasible", plan, searched, "normal_domain_infeasible;emergency_feasible")
            uncertain |= result.status == "unknown"
        return SearchResult("unknown" if uncertain else "infeasible", None, searched,
                            "uncertified_emergency_candidates" if uncertain else "emergency_domain_exhausted")


def compositions(total, length):
    if length == 1:
        yield (total,)
    else:
        for first in range(total+1):
            for rest in compositions(total-first, length-1):
                yield (first,)+rest
