"""93 observable state/batch features for the frozen three-ring configuration.

Arrival age includes external waiting. Scheduled ring is the current segment's
ring (None off-ring); other states use their recorded ring. Resource statistics
count reservation entries, including overlapping entries, not union occupancy.
No sorting value, identifier, scenario label, root provenance or outcome label
is supplied to the model. Historical observed terminal states are allowed.
"""
from math import cos, sin, fsum, sqrt, isfinite
from .routes import plan_energy

STATES = ("holding", "external", "scheduled", "normal_landed", "emergency_landed", "failed_energy")
RESOURCES = ("transfer", "approach", "pad")
STAT4 = ("min", "mean", "std", "max")
STAT3 = ("min", "mean", "max")


def resource_type(name):
    if name.startswith("transfer:") or name == "emergency_corridor":
        return "transfer"
    if name.endswith("_approach"):
        return "approach"
    if name.endswith("_pad"):
        return "pad"
    return None


def _stats(values, std=True):
    values = sorted(float(v) for v in values)
    if not values:
        return (0.,) * (4 if std else 3)
    mean = fsum(values)/len(values)
    return ((values[0], mean, sqrt(fsum((v-mean)**2 for v in values)/len(values)), values[-1])
            if std else (values[0], mean, values[-1]))


def _extract(env, plans):
    if len(env.config.rings) != 3:
        raise ValueError("stage4 features require the frozen three-ring layout")
    now = env.time
    aircraft = [a for a in env.aircraft.values() if a.arrival_time <= now and a.state != "pending"]
    by_id = {a.aircraft_id: a for a in aircraft}
    plans = tuple(plans)
    if len({p.aircraft_id for p in plans}) != len(plans) or any(p.aircraft_id not in by_id for p in plans):
        raise ValueError("candidate must contain unique observed aircraft")
    out = {"time": now, "remaining_horizon": max(0., env.config.horizon-now)}

    def add_stats(prefix, values, std=True):
        out.update({f"{prefix}_{k}": v for k, v in zip(STAT4 if std else STAT3, _stats(values, std))})

    def current_ring(a):
        if a.state == "scheduled" and a.plan:
            return next((s.ring for s in a.plan.segments if s.start <= now < s.end), None)
        return a.ring

    for state in STATES:
        out[f"state_{state}_count"] = sum(a.state == state for a in aircraft)
    for state in STATES:
        for ring in range(3):
            out[f"state_{state}_ring{ring}_count"] = sum(a.state == state and current_ring(a) == ring for a in aircraft)
    for state in ("holding", "external"):
        selected = [a for a in aircraft if a.state == state]
        add_stats(f"{state}_soc", (a.soc for a in selected))
        add_stats(f"{state}_arrival_age", (max(0., now-a.arrival_time) for a in selected))
    for ring in range(3):
        theta = [a.theta for a in aircraft if a.state == "holding" and a.ring == ring]
        for name, fn in (("sin", sin), ("cos", cos)):
            out[f"holding_ring{ring}_theta_{name}_mean"] = fsum(sorted(fn(x) for x in theta))/len(theta) if theta else 0.
    add_stats("scheduled_completion_remaining", (max(0., a.plan.end-now) for a in aircraft if a.state == "scheduled" and a.plan))
    reservations = [r for a in aircraft if a.plan for r in a.plan.reservations if r.end > now]
    for kind in RESOURCES:
        selected = [r for r in reservations if resource_type(r.resource) == kind]
        out[f"reserved_{kind}_entry_count"] = len(selected)
        out[f"reserved_{kind}_remaining_duration_sum"] = fsum(sorted(r.end-max(now, r.start) for r in selected))
        out[f"reserved_{kind}_earliest_end_remaining"] = min((r.end-now for r in selected), default=0.)
    out["candidate_plan_count"] = len(plans)
    for ring in range(3):
        out[f"candidate_start_ring{ring}_count"] = sum(by_id[p.aircraft_id].ring == ring for p in plans)
    add_stats("candidate_selected_soc", (by_id[p.aircraft_id].soc for p in plans))
    out["candidate_wait_cycles_sum"] = sum(sum(p.wait_cycles) for p in plans)
    for ring in range(3):
        out[f"candidate_ring{ring}_wait_cycles_sum"] = sum(p.wait_cycles[ring] for p in plans if len(p.wait_cycles) > ring)
    add_stats("candidate_first_nonhold_remaining", (next(s.start for s in p.segments if s.mode != "hold")-now for p in plans), False)
    add_stats("candidate_completion_remaining", (p.end-now for p in plans), False)
    add_stats("candidate_energy", (plan_energy(p, env.config) for p in plans), False)
    add_stats("candidate_postroute_reserve_slack", (by_id[p.aircraft_id].soc-plan_energy(p, env.config)-env.config.energy.reserve for p in plans), False)
    for kind in RESOURCES:
        out[f"candidate_{kind}_reservation_duration_sum"] = fsum(sorted(r.end-r.start for p in plans for r in p.reservations if resource_type(r.resource) == kind))
    selected_ids = {p.aircraft_id for p in plans}
    remaining = [a for a in aircraft if a.state == "holding" and a.aircraft_id not in selected_ids]
    out["unselected_holding_count"] = len(remaining)
    add_stats("unselected_holding_soc", (a.soc for a in remaining))
    return out


def _names():
    # Derive names through the same path, with no simulation and no dependencies.
    from types import SimpleNamespace
    env = SimpleNamespace(time=0., aircraft={}, config=SimpleNamespace(rings=(0, 1, 2), horizon=240.))
    return tuple(_extract(env, ()).keys())


FEATURE_NAMES = _names()
assert len(FEATURE_NAMES) == 93


def extract_features(env, plans):
    row = _extract(env, plans)
    assert tuple(row) == FEATURE_NAMES
    values = [float(row[name]) for name in FEATURE_NAMES]
    if not all(isfinite(v) for v in values):
        raise ValueError("features must be finite")
    return values
