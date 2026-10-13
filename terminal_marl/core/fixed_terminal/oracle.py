"""Offline exact finite-catalogue snapshot reference, not a dynamic optimum.

MILP constraints cover one choice per request and pair incompatibilities with
unassigned holders removed. Independent full-fleet checks enforce unselected
holders and multiway capacity. A failing full assignment receives a no-good cut
that excludes exactly that assignment, including its zeros. Thus a superset
which frees a blocking holder is never incorrectly excluded.
"""
from dataclasses import dataclass, asdict
from itertools import product
from math import isfinite
from time import perf_counter
from .baselines import objective


@dataclass
class OracleResult:
    status: str
    plans: tuple
    objective: tuple
    seconds: float
    candidates: int
    iterations: int
    cuts: int
    solver_status: int | None
    solver_gap: float | None
    certified_tiers: int
    unknown_assignments: int
    reason: str
    solver_dual_bound: float | None = None
    solver_bound_tier: str | None = None
    solver_bound_scope: str = "current_finite_MILP_relaxation_tier_not_dynamic_bound"

    def serializable(self):
        result = asdict(self)
        result["plans"] = [{"aircraft_id": p.aircraft_id, "wait_cycles": p.wait_cycles, "completion": p.end} for p in self.plans]
        return result


def brute_force_catalogue(env, catalogue):
    """Tiny-test reference with the exact same HOLD/options and joint checker."""
    best, best_value = (), objective(env, ())
    unknown, checked = 0, 0
    for selection in product(*[(None,)+tuple(plans) for plans in catalogue.values()]):
        batch = tuple(p for p in selection if p is not None)
        checked += 1
        check = env.feasibility.check_batch(batch, list(env.aircraft.values()), env.time)
        unknown += check.status == "unknown"
        if check.status == "feasible" and objective(env, batch) > best_value:
            best, best_value = batch, objective(env, batch)
    return {"plans": best, "objective": best_value, "checked": checked,
            "status": "UNKNOWN" if unknown else "OPTIMAL", "unknown_assignments": unknown}


def finite_catalogue_milp(env, catalogue, seconds=30., iteration_limit=1000):
    """Three exact objective tiers; a single wall budget spans cuts and tiers."""
    # Optional dependency stays outside all stage1 and ordinary baseline imports.
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from .routes import plan_energy
    start = perf_counter()
    plans = tuple(p for group in catalogue.values() for p in group)
    size = len(plans)
    if size == 0:
        return OracleResult("OPTIMAL", (), objective(env, ()), perf_counter()-start, 0, 0, 0, 0, 0., 3, 0, "empty_catalogue")
    if len({(p.aircraft_id, p.wait_cycles) for p in plans}) != size or any(p.emergency for p in plans):
        raise ValueError("oracle requires unique normal candidates")
    rows, lower, upper = [], [], []

    def constraint(row, lo, hi):
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    for key in catalogue:
        constraint(np.array([float(p.aircraft_id == key) for p in plans]), -np.inf, 1.)
    fixed = [a for a in env.aircraft.values() if a.state != "holding"]
    # These exclusions are hereditary: neither missing holders nor added
    # candidates can repair a conflict between the chosen pair/committed paths.
    pair_unknown = 0
    for i, a in enumerate(plans):
        for j in range(i+1, size):
            b = plans[j]
            if a.aircraft_id == b.aircraft_id:
                continue
            if perf_counter()-start >= seconds:
                return OracleResult("FEASIBLE_LIMIT", (), objective(env, ()), perf_counter()-start, size, 0, 0, None, None, 0, pair_unknown, "pair_precheck_time_limit")
            relaxed = fixed+[env.aircraft[a.aircraft_id], env.aircraft[b.aircraft_id]]
            result = env.feasibility.check_batch((a, b), relaxed, env.time)
            if result.status == "infeasible":
                row = np.zeros(size)
                row[i] = row[j] = 1.
                constraint(row, -np.inf, 1.)
            # Uncertified pairs cannot justify a proof-based exclusion.
            pair_unknown += result.status == "unknown"
    count = np.array([float(p.end <= env.config.horizon+1e-9) for p in plans])
    delay = np.array([(p.end-env.aircraft[p.aircraft_id].arrival_time-env.aircraft[p.aircraft_id].unimpeded_duration)
                      if count[i] else 0. for i, p in enumerate(plans)])
    energy = np.array([plan_energy(p, env.config) for p in plans])
    objectives = (-count, delay, energy)
    best, best_value = (), objective(env, ())
    iterations = cuts = certified = unknown = 0
    solver_status = None
    solver_gap = None
    solver_dual_bound = None
    solver_bound_tier = None
    reason = "iteration_limit"
    for tier, coefficients in enumerate(objectives):
        tier_solved = False
        while iterations < iteration_limit and perf_counter()-start < seconds:
            remaining = seconds-(perf_counter()-start)
            result = milp(coefficients, integrality=np.ones(size), bounds=Bounds(np.zeros(size), np.ones(size)),
                          constraints=LinearConstraint(np.array(rows), np.array(lower), np.array(upper)),
                          options={"time_limit": remaining, "mip_rel_gap": 0.})
            iterations += 1
            solver_status = int(result.status)
            gap = getattr(result, "mip_gap", None)
            bound = getattr(result, "mip_dual_bound", None)
            solver_gap = float(gap) if gap is not None and isfinite(gap) else None
            solver_dual_bound = float(bound) if bound is not None and isfinite(bound) else None
            solver_bound_tier = ("negative_normal_count", "delay_seconds", "route_energy_soc")[tier]
            if result.x is None:
                reason = "solver_no_incumbent_status_"+str(solver_status)
                break
            bits = np.rint(result.x).astype(int)
            if np.max(np.abs(result.x-bits)) > 1e-5:
                reason = "solver_nonintegral_incumbent"
                break
            chosen = tuple(p for p, bit in zip(plans, bits) if bit)
            check = env.feasibility.check_batch(chosen, list(env.aircraft.values()), env.time)
            if check.status != "feasible":
                unknown += check.status == "unknown"
                # sum(selected x)-sum(unselected x) <= |selected|-1
                constraint(np.where(bits == 1, 1., -1.), -np.inf, float(bits.sum()-1))
                cuts += 1
                continue
            value = objective(env, chosen)
            if value >= best_value:
                best, best_value = chosen, value
            if solver_status != 0:
                reason = "solver_limit_status_"+str(solver_status)
                break
            target = float(coefficients@bits)
            tolerance = 0. if tier == 0 else 1e-8
            constraint(coefficients, target-tolerance, target+tolerance)
            certified += 1
            tier_solved = True
            reason = "all_lexicographic_tiers_certified"
            break
        if not tier_solved:
            if perf_counter()-start >= seconds:
                reason = "time_limit"
            break
    status = "OPTIMAL" if certified == 3 and unknown == 0 else "FEASIBLE_LIMIT"
    if unknown:
        reason = "uncertified_assignments_excluded;"+reason
    return OracleResult(status, best, best_value, perf_counter()-start, size, iterations, cuts,
                        solver_status, solver_gap, certified, unknown, reason,
                        solver_dual_bound, solver_bound_tier)


def objective_gap(incumbent, oracle_result):
    """No numerical optimality claim unless all finite-domain tiers certified."""
    if oracle_result.status != "OPTIMAL":
        return {"valid": False, "normal_count_gap": None, "delay_gap_seconds": None, "energy_gap_soc": None}
    optimum = oracle_result.objective
    same_count = incumbent[0] == optimum[0]
    same_delay = same_count and abs(incumbent[1]-optimum[1]) <= 1e-7
    return {"valid": True, "normal_count_gap": optimum[0]-incumbent[0],
            "delay_gap_seconds": optimum[1]-incumbent[1] if same_count else None,
            "energy_gap_soc": optimum[2]-incumbent[2] if same_delay else None}
