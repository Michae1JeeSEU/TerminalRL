"""Non-learning scheduling baselines with declared bounded search domains.

Schedulers receive observed_clone(), never pending future demand or SOC. Rules
commit greedily; joint search selects compatible full-route window sets; rolling
search simulates observable traffic and replans in cloned future states.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import comb, floor
from time import perf_counter
from .domain import Intent
from .routes import build_plan, plan_energy
from .safety import compositions
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG, METHODS


@dataclass
class Statistics:
    generated: int = 0
    examined: int = 0
    joint_assignments: int = 0
    truncated: bool = False
    rejection_reasons: dict = field(default_factory=dict)
    priority_windows: dict = field(default_factory=dict)
    lookahead_states: int = 0

    def reject(self, result):
        if result.status != "feasible":
            key = result.status+":"+result.reason
            self.rejection_reasons[key] = self.rejection_reasons.get(key, 0)+1

    def merge(self, other):
        for key in ("generated", "examined", "joint_assignments", "lookahead_states"):
            setattr(self, key, getattr(self, key)+getattr(other, key))
        self.truncated |= other.truncated
        self.rejection_reasons = dict(Counter(self.rejection_reasons)+Counter(other.rejection_reasons))
        self.priority_windows.update(other.priority_windows)


@dataclass
class Decision:
    method: str
    plans: tuple
    statistics: Statistics
    seconds: float = 0.
    objective: tuple = ()


def observed_clone(env):
    """Remove all unarrived identities, draw times and SOC from the scheduler."""
    clone = deepcopy(env)
    clone.aircraft = {key: a for key, a in clone.aircraft.items()
                      if a.arrival_time <= clone.time and a.state != "pending"}
    clone.scenario = replace(clone.scenario, requests=tuple(r for r in clone.scenario.requests if r.aircraft_id in clone.aircraft))
    clone.events = []
    return clone


def holders(env):
    return [a for a in env.aircraft.values() if a.state == "holding"]


def normal_domain(aircraft, now, config):
    base = build_plan(aircraft, now, config)
    budget = aircraft.soc-config.energy.reserve-plan_energy(base, config)
    if budget < -1e-10:
        return 0, iter(())
    maximum = max(0, floor((budget+1e-10)/(config.lap_time*config.energy.hold)))
    length = aircraft.ring+1
    count = comb(maximum+length, length)

    def plans():
        for total in range(maximum+1):
            for cycles in compositions(total, length):
                yield build_plan(aircraft, now, config, cycles)
    return count, plans()


def analyse_windows(env, aircraft, limit=256, stats=None):
    """Min feasible energy and latest feasible completion, with bound status.

    Enumeration is nondecreasing in extra-lap SOC cost, so its first certified
    feasible plan is an exact energy minimum unless an earlier UNKNOWN exists.
    The deadline scans total laps in descending order. Equal-sum vectors share
    completion time; a certified feasible vector is latest if no higher tier
    was unknown. Both passes share one deduplicated check budget.
    """
    stats = stats if stats is not None else Statistics()
    total, domain = normal_domain(aircraft, env.time, env.config)
    feasible, unknown = [], False
    fleet = list(env.aircraft.values())
    cache = {}

    def check(plan):
        if plan.wait_cycles not in cache:
            if len(cache) >= limit:
                return None
            stats.generated += 1
            stats.examined += 1
            result = env.feasibility.check(plan, aircraft, fleet, env.time, check_emergency=False)
            stats.reject(result)
            cache[plan.wait_cycles] = result
        return cache[plan.wait_cycles]

    earlier_unknown = min_exact = False
    for plan in domain:
        result = check(plan)
        if result is None:
            break
        if result.status == "feasible":
            min_exact = not earlier_unknown
            feasible.append(plan)
            break
        else:
            unknown |= result.status == "unknown"
            earlier_unknown |= result.status == "unknown"
    base = build_plan(aircraft, env.time, env.config)
    budget = aircraft.soc-env.config.energy.reserve-plan_energy(base, env.config)
    maximum = floor((budget+1e-10)/(env.config.lap_time*env.config.energy.hold))
    deadline_exact = False
    higher_unknown = False
    stop = False
    for laps in range(maximum, -1, -1):
        tier_unknown = False
        for cycles in compositions(laps, aircraft.ring+1):
            plan = build_plan(aircraft, env.time, env.config, cycles)
            result = check(plan)
            if result is None:
                stop = True
                break
            if result.status == "feasible":
                feasible.append(plan)
                deadline_exact = not higher_unknown
                stop = True
                break
            tier_unknown |= result.status == "unknown"
            unknown |= result.status == "unknown"
        if stop:
            break
        higher_unknown |= tier_unknown
    complete = len(cache) == total and not unknown
    stats.truncated |= not (min_exact and deadline_exact) and len(cache) < total
    status = "feasible" if feasible else "infeasible" if complete else "unknown"
    minimum = min((plan_energy(p, env.config)+env.config.energy.reserve for p in feasible), default=None)
    latest = max((p.end for p in feasible), default=None)
    # Energy-only upper bound is not claimed resource-feasible.
    upper = base.end+max(0, maximum)*env.config.lap_time
    result = {"status": status, "minimum_feasible_required_soc": minimum,
              "minimum_status": "exact" if minimum is not None and min_exact else "upper_bound" if minimum is not None else status,
              "energy_slack": aircraft.soc-minimum if minimum is not None else None,
              "latest_feasible_completion": latest,
              "deadline_status": "exact" if deadline_exact else "lower_bound" if latest is not None else "infeasible" if complete else "unknown",
              "deadline_slack": latest-env.time if latest is not None else None,
              "completion_energy_only_upper_bound": upper, "domain_size": total, "examined": len(cache)}
    stats.priority_windows[aircraft.aircraft_id] = result
    return result


def rule_order(env, method, settings=DEFAULT_SCHEDULER_CONFIG, stats=None):
    stats = stats if stats is not None else Statistics()
    aircraft = holders(env)
    if method in ("energy_slack", "deadline_slack"):
        values = {a.aircraft_id: analyse_windows(env, a, settings.priority_scan_limit, stats) for a in aircraft}

        def key(a):
            row = values[a.aircraft_id]
            metric = row["energy_slack" if method == "energy_slack" else "deadline_slack"]
            # Infeasible/unknown have an explicit category, never a fabricated
            # exact slack. Certified/bounded feasible requests come first.
            return (metric is None, metric if metric is not None else 0., a.arrival_time, a.aircraft_id)
        return sorted(aircraft, key=key)
    return sorted(aircraft, key=(lambda a: (a.soc, a.arrival_time, a.aircraft_id)) if method == "soc"
                  else (lambda a: (a.arrival_time, a.aircraft_id)))


def objective(env, plans, interval_end=None):
    end = env.config.horizon if interval_end is None else interval_end
    completed = [p for p in plans if p.end <= end+1e-9]
    delay = sum(p.end-env.aircraft[p.aircraft_id].arrival_time-env.aircraft[p.aircraft_id].unimpeded_duration for p in completed)
    return len(completed), -delay, -sum(plan_energy(p, env.config) for p in plans)


def make_catalogue(env, settings=DEFAULT_SCHEDULER_CONFIG, stats=None, *, reference=False):
    stats = stats if stats is not None else Statistics()
    ordered = sorted(holders(env), key=lambda a: (a.arrival_time, a.aircraft_id))
    selected = ordered if reference else ordered[:settings.joint_aircraft_limit]
    stats.truncated |= len(selected) < len(ordered)
    # Candidate generation intentionally removes all uncommitted holders. They
    # are restored at final batch validation, allowing simultaneous departures.
    fixed = [a for a in env.aircraft.values() if a.state != "holding"]
    catalogue = {}
    for a in selected:
        total, domain = normal_domain(a, env.time, env.config)
        kept, scanned = [], 0
        cap = settings.oracle_candidates_per_aircraft if reference else settings.candidates_per_aircraft
        scan_limit = cap if reference else settings.catalogue_scan_limit
        for p in domain:
            if scanned >= scan_limit or len(kept) >= cap:
                break
            scanned += 1
            stats.generated += 1
            stats.examined += 1
            check = env.feasibility.check(p, a, fixed+[a], env.time, check_emergency=False)
            stats.reject(check)
            if check.status == "feasible":
                kept.append(p)
        stats.truncated |= scanned < total
        catalogue[a.aircraft_id] = tuple(kept)
    return catalogue


def joint_choices(env, catalogue, settings=DEFAULT_SCHEDULER_CONFIG, stats=None):
    stats = stats if stats is not None else Statistics()
    beam = [()]
    fixed = [a for a in env.aircraft.values() if a.state != "holding"]
    for key, candidates in catalogue.items():
        expanded = []
        for batch in beam:
            for option in (None,)+tuple(candidates):
                selection = batch+((option,) if option else ())
                relaxed = fixed+[env.aircraft[p.aircraft_id] for p in selection]
                stats.joint_assignments += 1
                result = env.feasibility.check_batch(selection, relaxed, env.time)
                stats.reject(result)
                if result.status == "feasible":
                    expanded.append(selection)
        expanded.sort(key=lambda plans: (objective(env, plans), tuple((p.aircraft_id, p.wait_cycles) for p in plans)), reverse=True)
        stats.truncated |= len(expanded) > settings.beam_width
        beam = expanded[:settings.beam_width]
        if () not in beam:
            beam.append(())  # HOLD is always evaluated as a root alternative.
    valid = []
    for batch in beam:
        stats.joint_assignments += 1
        check = env.feasibility.check_batch(batch, list(env.aircraft.values()), env.time)
        stats.reject(check)
        if check.status == "feasible":
            valid.append(batch)
    valid.sort(key=lambda plans: (objective(env, plans), tuple((p.aircraft_id, p.wait_cycles) for p in plans)), reverse=True)
    return valid or [()]


def _rule_decision(env, method, settings, stats, catalogue=None):
    work = observed_clone(env)
    selected = []
    for a in rule_order(work, method, settings, stats):
        if catalogue is None:
            result = work.feasibility.search_normal(a, list(work.aircraft.values()), work.time, settings.normal_search_limit)
            stats.generated += result.searched
            stats.examined += result.searched
            stats.truncated |= result.status == "unknown"
            stats.reject(result)
            plan = result.plan
        else:
            plan = None
            for candidate in catalogue.get(a.aircraft_id, ()):
                stats.examined += 1
                result = work.feasibility.check_batch((candidate,), list(work.aircraft.values()), work.time)
                stats.reject(result)
                if result.status == "feasible":
                    plan = candidate
                    break
        if plan is not None:
            work._commit(a, plan, "rule_internal_snapshot")
            selected.append(plan)
    return tuple(selected)


def common_emergency_fallback(env, settings, stats):
    """Same stage1 certificate and budget for every method, in ID order."""
    observed_fleet = [a for a in env.aircraft.values() if a.arrival_time <= env.time and a.state != "pending"]
    for a in sorted(holders(env), key=lambda a: a.aircraft_id):
        result = env.feasibility.search_emergency(a, observed_fleet, env.time, settings.normal_search_limit)
        stats.generated += result.searched
        stats.examined += result.searched
        stats.truncated |= result.status == "unknown"
        stats.reject(result)
        if result.plan is not None:
            env._commit(a, result.plan, result.reason)


def rolling_choice(env, choices, settings, stats):
    roots = list(choices[:max(1, settings.rolling_root_alternatives-1)])
    if () not in roots:
        roots.append(())
    scored = []
    for root in roots:
        future = observed_clone(env)
        assert future.submit_batch(root).status == "feasible"
        local = Statistics()
        common_emergency_fallback(future, settings, local)
        stop = min(env.config.horizon, env.time+settings.rolling_seconds)
        next_decision = env.time+settings.decision_interval
        while future.time < stop-1e-9:
            future.step()
            stats.lookahead_states += 1
            if future.time >= next_decision-1e-9 and future.time < stop-1e-9:
                candidates = make_catalogue(future, settings, local)
                batch = joint_choices(future, candidates, settings, local)[0]
                assert future.submit_batch(batch).status == "feasible"
                common_emergency_fallback(future, settings, local)
                next_decision += settings.decision_interval
        # Preregistered terminal surrogate: future commitments are counted at
        # the observation horizon, although dynamics are rolled out only20s.
        normal = [a.plan for a in future.aircraft.values() if a.plan and not a.plan.emergency]
        failures = sum(a.state == "failed_energy" for a in future.aircraft.values())
        emergencies = sum(a.plan is not None and a.plan.emergency for a in future.aircraft.values())
        exposure = sum(max(0., min(future.time, a.completion_time if a.completion_time is not None else future.time)-a.arrival_time)
                       for a in future.aircraft.values())
        normal_value = objective(future, normal)
        committed_energy = sum(plan_energy(a.plan, future.config) for a in future.aircraft.values() if a.plan)
        value = (-failures, normal_value[0], -emergencies, -exposure, normal_value[1], -committed_energy)
        scored.append((value, root))
        stats.merge(local)
    return max(scored, key=lambda item: item[0])[1]


def choose_normal_batch(method, observed_env, settings=DEFAULT_SCHEDULER_CONFIG, catalogue=None):
    if method not in METHODS:
        raise ValueError("unknown baseline")
    if any(a.arrival_time > observed_env.time or a.state == "pending" for a in observed_env.aircraft.values()):
        raise ValueError("scheduler input must contain only observed requests")
    start, stats = perf_counter(), Statistics()
    if method in METHODS[:4]:
        batch = _rule_decision(observed_env, method, settings, stats, catalogue)
    else:
        if catalogue is None:
            catalogue = make_catalogue(observed_env, settings, stats)
        choices = joint_choices(observed_env, catalogue, settings, stats)
        batch = choices[0] if method == "joint_window" else rolling_choice(observed_env, choices, settings, stats)
    return Decision(method, batch, stats, perf_counter()-start, objective(observed_env, batch))


def apply_scheduler(env, method, settings=DEFAULT_SCHEDULER_CONFIG):
    start = perf_counter()
    decision = choose_normal_batch(method, observed_clone(env), settings)
    check = env.submit_batch(decision.plans)
    decision.statistics.reject(check)
    if check.status != "feasible":
        raise RuntimeError("scheduler selected a batch rejected by independent feasibility: "+check.reason)
    common_emergency_fallback(env, settings, decision.statistics)
    decision.seconds = perf_counter()-start
    env.decision_calls += 1
    env.decision_seconds += decision.seconds
    env.search_candidates += decision.statistics.examined
    return decision
