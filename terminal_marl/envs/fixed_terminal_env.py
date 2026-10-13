"""Lightweight fixed-terminal stepping environment, independent of Gym/RL.

External demand waits in an abstract boundary queue: no outside trajectories or
outside collision claims. Queue time consumes the holding SOC rate. Admission
creates the request at the outer-ring node only after a future-holding safety
check; arrival time includes all queue delay. Energy exhaustion is an explicit
absorbing failure (crash/recovery dynamics are outside this model). Exhaustion
times are exact for unscheduled holders; state absorbs at the tick boundary,
and the runtime motion monitor conservatively checks their entire final tick.
"""
from dataclasses import asdict
from math import pi, isfinite
from time import perf_counter
from terminal_marl.core.fixed_terminal.domain import Aircraft, Intent, Plan, Reservation, Segment
from terminal_marl.core.fixed_terminal.routes import build_plan, plan_energy, required_energy
from terminal_marl.core.fixed_terminal.safety import Feasibility, forecast, swept_spacing, Check
from terminal_marl.core.fixed_terminal.scenarios import generate_scenario
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG


class FixedTerminalEnv:
    def __init__(self, config=DEFAULT_CONFIG, scenario=None):
        self.config = config
        self.scenario = scenario or generate_scenario(config)
        self.feasibility = Feasibility(config)
        self.reset()

    def reset(self):
        c = self.config
        self.time = 0.
        self.aircraft = {}
        self.events = []
        self.decision_seconds = 0.
        self.decision_calls = 0
        self.search_candidates = 0
        self.infeasible_intents = 0
        self.unknown_intents = 0
        self.separation_violations = 0
        self.spacing_uncertified = 0
        self.admission_rejection_counts = {}
        self.search_status_counts = {"feasible": 0, "infeasible": 0, "unknown": 0}
        for request in self.scenario.requests:
            if request.aircraft_id in self.aircraft or not 0 <= request.soc <= 1 or not 0 <= request.arrival_time <= c.horizon:
                raise ValueError("unique IDs, SOC in [0,1], arrivals inside observation horizon required")
            if not isfinite(request.theta):
                raise ValueError("finite request theta required")
            if request.ring is not None:
                if isinstance(request.ring, bool) or not isinstance(request.ring, int) or not 0 <= request.ring < len(c.rings):
                    raise ValueError("initial ring index must be an integer inside configured rings")
                if request.arrival_time != 0:
                    raise ValueError("ring-present initial requests must arrive at time zero")
            state = "holding" if request.ring is not None else ("external" if request.arrival_time == 0 else "pending")
            a = Aircraft(request.aircraft_id, request.arrival_time, request.soc, request.soc,
                         request.ring, request.theta, state=state, minimum_soc=request.soc,
                         admitted_time=0. if request.ring is not None else None)
            # Unimpeded reference is the same fixed normal route at original
            # request position; external requests reference immediate node admission.
            reference = Aircraft(a.aircraft_id, a.arrival_time, a.soc, a.soc,
                                 a.ring if a.ring is not None else len(c.rings)-1,
                                 a.theta if a.ring is not None else 0.)
            a.unimpeded_duration = build_plan(reference, a.arrival_time, c).end-a.arrival_time
            self.aircraft[a.aircraft_id] = a
        initial = [a for a in self.aircraft.values() if a.state == "holding"]
        from terminal_marl.core.fixed_terminal.domain import distance
        for ring in range(len(c.rings)):
            if sum(a.ring == ring for a in initial) > c.operational_capacity(ring):
                raise ValueError("initial operational capacity exceeded")
        for i, a in enumerate(initial):
            for b in initial[i+1:]:
                if distance(a.position(0, c), b.position(0, c)) < c.protected_distance-1e-9:
                    raise ValueError("initial protected spacing violated")
        return self.snapshot()

    @property
    def done(self):
        return self.time >= self.config.horizon-1e-9

    def snapshot(self):
        return {"time": self.time, "states": {key: a.state for key, a in self.aircraft.items()}}

    def propose(self, intent):
        return build_plan(self.aircraft[intent.aircraft_id], self.time, self.config,
                          intent.wait_cycles, intent.emergency)

    def submit(self, intent, search_limit=256):
        start = perf_counter()
        self.decision_calls += 1
        try:
            a = self.aircraft[intent.aircraft_id]
            plan = self.propose(intent)
            result = self.feasibility.check(plan, a, list(self.aircraft.values()), self.time,
                                            search_limit=search_limit)
        except (KeyError, ValueError, TypeError, IndexError):
            result = Check("infeasible", "invalid_intent")
        self.search_candidates += result.searched
        if result.status == "feasible":
            self._commit(a, plan, "explicit_intent")
        else:
            self.infeasible_intents += result.status == "infeasible"
            self.unknown_intents += result.status == "unknown"
            self.events.append({"time": self.time, "event": "intent_rejected",
                                "aircraft_id": intent.aircraft_id, **asdict(result)})
        self.decision_seconds += perf_counter()-start
        return result

    def submit_batch(self, plans):
        """Validate all normal plans before committing any; no partial changes."""
        plans = tuple(plans)
        result = self.feasibility.check_batch(plans, list(self.aircraft.values()), self.time)
        if result.status == "feasible":
            for plan in plans:
                self._commit(self.aircraft[plan.aircraft_id], plan, "joint_normal_batch")
        return result

    def _commit(self, aircraft, plan, reason):
        aircraft.plan, aircraft.state = plan, "scheduled"
        if plan.emergency:
            aircraft.reason = "no_normal_plan_in_snapshot_domain"
            aircraft.emergency_energy = plan_energy(plan, self.config)
            # Union of dedicated resource windows, separately for each resource.
            groups = {}
            for r in plan.reservations:
                if r.resource.startswith("emergency"):
                    groups.setdefault(r.resource, []).append((r.start, r.end))
            for windows in groups.values():
                end = float("-inf")
                for lo, hi in sorted(windows):
                    aircraft.emergency_resource_seconds += max(0., hi-max(lo, end))
                    end = max(end, hi)
        self.events.append({"time": self.time, "event": "plan_committed", "aircraft_id": aircraft.aircraft_id,
                            "emergency": plan.emergency, "wait_cycles": plan.wait_cycles,
                            "planned_end": plan.end, "required_soc_including_reserve": plan_energy(plan, self.config)+self.config.energy.reserve,
                            "reason": reason, "reservations": [asdict(r) for r in plan.reservations]})

    def exercise_feasibility(self, search_limit=256):
        """Deterministic ID-order infrastructure exerciser; not a strong baseline.

        Whole-route window search is used only to exercise the feasibility API.
        UNKNOWN leaves the aircraft holding. No threshold emergency fallback.
        """
        start = perf_counter()
        fleet = list(self.aircraft.values())
        for a in fleet:
            if a.state != "holding":
                continue
            self.decision_calls += 1
            result = self.feasibility.search_normal(a, fleet, self.time, search_limit)
            count = result.searched
            if result.status == "infeasible":
                # This repeats the certificate deliberately through the public
                # emergency validation path rather than bypassing eligibility.
                emergency = self.feasibility.search_emergency(a, fleet, self.time, search_limit)
                count += emergency.searched
                result = emergency
            self.search_candidates += count
            self.search_status_counts[result.status] += 1
            self.events.append({"time": self.time, "event": "search", "aircraft_id": a.aircraft_id,
                                "status": result.status, "reason": result.reason, "candidates": count,
                                "domain": "SOC-bounded ring-window vectors; committed routes fixed; other aircraft keep holding"})
            if result.plan is not None:
                self._commit(a, result.plan, result.reason)
        self.decision_seconds += perf_counter()-start

    def _admit(self):
        c = self.config

        def reject(aircraft, reason):
            self.admission_rejection_counts[reason] = self.admission_rejection_counts.get(reason, 0)+1
            self.events.append({"time": self.time, "event": "admission_rejected",
                                "aircraft_id": aircraft.aircraft_id, "reason": reason})

        for a in self.aircraft.values():
            if a.state != "external":
                continue
            ring_index = len(c.rings)-1
            ring = c.rings[ring_index]
            candidate = Aircraft(a.aircraft_id, a.arrival_time, a.initial_soc, a.soc, ring_index, 0.)
            minimum_normal = required_energy(candidate, self.time, c)
            minimum_emergency = required_energy(candidate, self.time, c, emergency=True)
            if a.soc+1e-10 < min(minimum_normal, minimum_emergency):
                reject(a, "minimum_route_energy_reserve")
                continue
            # A new holder cannot invalidate any existing full-route commitment.
            end = max([self.time+c.lap_time]+[b.plan.end+c.pad_service for b in self.aircraft.values() if b.plan])
            point = (ring.radius, 0., ring.height)
            segment = Segment(self.time, end, "hold", point, point, ring_index, 0., c.angular_speed)
            placeholder = Plan(a.aircraft_id, False, (), (segment,),
                               (Reservation(f"ring:{ring_index}", self.time, end),))
            others = [b for b in self.aircraft.values() if b.aircraft_id != a.aircraft_id]
            if self.feasibility._capacity(placeholder, others, self.time):
                reject(a, "operational_capacity")
                continue
            safe = True
            for b in others:
                for other in forecast(b, self.time, end, c):
                    lo, hi = max(self.time, other.start), min(end, other.end)
                    if hi >= lo and swept_spacing(segment, other, lo, hi, c.protected_distance, c.continuous_tolerance) != "feasible":
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                a.ring, a.theta, a.state, a.admitted_time = ring_index, 0., "holding", self.time
                self.events.append({"time": self.time, "event": "admitted", "aircraft_id": a.aircraft_id,
                                    "external_wait_seconds": self.time-a.arrival_time})
            else:
                reject(a, "committed_future_spacing_or_uncertified")

    def step(self, intents=()):
        if self.done:
            raise RuntimeError("episode already reached observation cutoff")
        self._admit()
        for intent in intents:
            self.submit(intent)
        c, now = self.config, self.time
        end = min(c.horizon, now+c.dt)
        # Runtime swept monitor independent of the proposal checker.
        bodies = [(a.aircraft_id, forecast(a, now, end, c)) for a in self.aircraft.values()
                  if a.state not in ("pending", "external", "failed_energy")]
        for i, (_, first) in enumerate(bodies):
            for _, second in bodies[i+1:]:
                statuses = []
                for x in first:
                    for y in second:
                        lo, hi = max(now, x.start, y.start), min(end, x.end, y.end)
                        if hi >= lo:
                            statuses.append(swept_spacing(x, y, lo, hi, c.separation, c.continuous_tolerance))
                self.separation_violations += "infeasible" in statuses
                self.spacing_uncertified += "unknown" in statuses
        for a in self.aircraft.values():
            if a.state in ("normal_landed", "emergency_landed", "failed_energy"):
                continue
            if a.state == "pending":
                if a.arrival_time > end:
                    continue
                a.state = "external"
            active_start = max(now, a.arrival_time)
            old_soc = a.soc
            cost = plan_energy(a.plan, c, active_start, end) if a.plan else (end-active_start)*c.energy.hold
            cost = min(cost, a.soc)
            a.soc = max(0., a.soc-cost)
            a.consumed += cost
            a.minimum_soc = min(a.minimum_soc, a.soc)
            if a.soc <= 1e-12:
                a.state, a.reason = "failed_energy", "SOC_exhausted_before_landing"
                a.completion_time = min(end, active_start+old_soc/c.energy.hold) if a.plan is None else end
                self.events.append({"time": a.completion_time, "event": "failed_energy", "aircraft_id": a.aircraft_id})
            elif a.plan and a.plan.end <= end+1e-9:
                a.state = "emergency_landed" if a.plan.emergency else "normal_landed"
                a.completion_time = a.plan.end
                self.events.append({"time": a.plan.end, "event": a.state, "aircraft_id": a.aircraft_id, "soc": a.soc})
            elif a.state == "holding":
                a.theta = (a.theta-c.angular_speed*(end-now)) % (2*pi)
        self.time = end
        return self.snapshot()

    def report(self):
        from terminal_marl.core.fixed_terminal.metrics import report
        return report(self)
