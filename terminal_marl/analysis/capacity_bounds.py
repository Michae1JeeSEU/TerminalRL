"""Optimistic necessary bounds, independent of any scheduling policy.

The finite-cohort LP is an admission-count relaxation, not a steady-state
capacity estimator. Its upper bound comes from repaired dual variables rather
than treating a numerical maximization objective as an upper certificate.
"""
from __future__ import annotations

import math
import numpy as np

from terminal_marl.core.rotating_slots.config import load_service_profile


def _pads(pads):
    if type(pads) is not int or pads < 1:
        raise ValueError("pads must be a positive integer")


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional sequence")
    return array


def loss_adjusted_pad_upper(durations_s, probabilities, pads=6, epsilon=0.,
                            protected_s=13.5):
    """Necessary upper offered rate allowing at most epsilon nonnormal jobs.

    q_c is selected normal mass per offered job, not a conditional class mix.
    Shortest compulsory pad work is retained first; no empirical fit is used.
    """
    _pads(pads)
    durations = _vector(durations_s, "durations_s")
    p = _vector(probabilities, "probabilities")
    if (not durations.size or durations.shape != p.shape or np.any(durations < 0)
            or np.any(p < 0) or not math.isclose(float(p.sum()), 1., rel_tol=0., abs_tol=1e-12)
            or not math.isfinite(epsilon) or not 0 <= epsilon < 1
            or not math.isfinite(protected_s) or protected_s < 0):
        raise ValueError("nonnegative matching service/mix, unit probability, and 0 <= epsilon < 1 required")
    p = p / p.sum()
    b = durations + protected_s
    q = np.zeros_like(p)
    remaining = 1. - epsilon
    for index in np.argsort(b, kind="stable"):
        q[index] = min(float(p[index]), remaining)
        remaining = max(0., remaining - q[index])
    work = float(q @ b)
    if work <= 0:
        raise ValueError("selected normal jobs must require positive pad work")
    return {
        "epsilon": float(epsilon), "pads": pads,
        "class_work_s": b.tolist(), "selected_mass": q.tolist(),
        "discarded_mass": (p - q).tolist(), "selected_total_mass": float(q.sum()),
        "minimum_work_per_offered_job_s": work,
        "offered_rate_upper_per_h": 3600. * pads / work,
        "bound_kind": "necessary_long_run_offered_rate_upper",
        "assumptions": [
            "Fixed offered class mix; normal fraction at least 1-epsilon.",
            "Each normal job requires protected descent plus class service on one pad.",
            "Long-run boundary work divided by horizon vanishes; all other constraints are relaxed.",
            "Selective nonnormal outcomes may discard the longest jobs first; attainability is not asserted.",
        ],
    }


def kinematic_touchdown_lower_bound(config=None):
    """Necessary release-to-touchdown time; not an attainable-time guarantee.

    Geometry is read from a frozen profile or the capacity-config factory's
    supported 250/300/350 m inner-ring variants. Template search minima are
    deliberately absent from this physical relaxation.
    """
    c = config or load_service_profile("service_half_v1")
    rings = {ring.id: ring for ring in c.airspace.rings}
    if (set(rings) != {"outer", "middle", "inner"}
            or (rings["outer"].radius_m, rings["middle"].radius_m) != (600., 450.)
            or rings["inner"].radius_m not in (250., 300., 350.)
            or tuple(rings[name].height_agl_m for name in ("outer", "middle", "inner")) != (110., 70., 30.)
            or c.entry.radius_m != 900. or c.pads.layout_radius_m != 100.):
        raise ValueError("geometry must match a frozen profile or supported capacity-config geometry")
    vh = float(c.vehicle.max_horizontal_speed_mps)
    vz = float(c.vehicle.max_descent_speed_mps)
    accel = float(c.vehicle.max_horizontal_acceleration_norm_mps2)
    capture = float(c.pads.final_capture_height_agl_m)
    protected = float(c.pads.vertical_descent_profile.total_duration_s)
    omega = abs(float(c.airspace.common_angular_speed_radps))
    if min(vh, vz, accel) <= 0 or not all(map(math.isfinite, (vh, vz, accel, capture, protected, omega))):
        raise ValueError("finite physical constants and positive speed/acceleration limits required")
    entry = max(0., float(c.entry.radius_m) - rings["outer"].radius_m) / vh
    transfers = {}
    for source, target in (("outer", "middle"), ("middle", "inner")):
        a, b = rings[source], rings[target]
        transfers[f"{source}_to_{target}_s"] = max(abs(a.radius_m - b.radius_m) / vh,
                                                   abs(a.height_agl_m - b.height_agl_m) / vz)
    inner = rings["inner"]
    final_components = {
        "radial_s": max(0., inner.radius_m - c.pads.layout_radius_m) / vh,
        "vertical_s": max(0., inner.height_agl_m - capture) / vz,
        "deceleration_s": omega * inner.radius_m / accel,
    }
    final = max(final_components.values())
    total = entry + sum(transfers.values()) + final + protected
    return {"bound_kind": "necessary_touchdown_time_lower",
            "touchdown_lower_s": total, "protected_start_lower_s": total - protected,
            "entry_s": entry, **transfers, "final_predescent_s": final,
            "final_components_s": final_components, "protected_descent_s": protected,
            "capture_height_m": capture, "inner_radius_m": float(inner.radius_m),
            "assumptions": ["Sequential entry, outer-middle, middle-inner, final and descent stages.",
                            "Only necessary displacement, speed-change and fixed descent constraints are retained.",
                            "Phase, trajectory shape, jerk, conflicts and waiting may require additional time."]}


def mandatory_pad_overlap(earliest_start_s, latest_completion_s, work_s, u, v):
    """Minimum work inside [u,v] for a feasible job window [r,d]."""
    r = _vector(earliest_start_s, "earliest_start_s")
    d = _vector(latest_completion_s, "latest_completion_s")
    b = _vector(work_s, "work_s")
    if (r.shape != d.shape or r.shape != b.shape or np.any(b < 0)
            or np.any(b > d - r) or not math.isfinite(u) or not math.isfinite(v) or u >= v):
        raise ValueError("feasible matching windows/work and finite u < v required")
    return np.maximum(0., b - np.maximum(0., u - r) - np.maximum(0., d - v))


def pad_window_lp_upper(earliest_start_s, latest_completion_s, work_s, *, pads=6,
                        intervals=None, grid_step_s=60., eligible=None):
    """Numerical dual upper bound on optional job count from interval pad work.

    Explicit interval subsets remain valid but can weaken the relaxation.
    Invalid per-job windows are excluded before constructing the LP. scipy is
    imported lazily. The returned certificate uses floating-point arithmetic,
    with an explicit feasibility repair and guard, not exact rational proof.
    """
    _pads(pads)
    r = _vector(earliest_start_s, "earliest_start_s")
    d = _vector(latest_completion_s, "latest_completion_s")
    b = _vector(work_s, "work_s")
    if r.shape != d.shape or r.shape != b.shape or np.any(b < 0):
        raise ValueError("matching windows and nonnegative work required")
    n = len(r)
    allowed = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if allowed.shape != r.shape:
        raise ValueError("eligible must match job shape")
    allowed &= b <= d - r
    selected = np.flatnonzero(allowed)
    r, d, b = r[allowed], d[allowed], b[allowed]
    count = len(r)
    if intervals is None:
        if not math.isfinite(grid_step_s) or grid_step_s <= 0:
            raise ValueError("grid_step_s must be positive and finite")
        if count:
            lo, hi = float(r.min()), float(d.max())
            points = np.unique(np.r_[lo, np.arange(math.ceil(lo / grid_step_s) * grid_step_s,
                                                   hi, grid_step_s), hi])
            windows = [(float(u), float(v)) for i, u in enumerate(points) for v in points[i + 1:]]
        else:
            windows = []
    else:
        windows = []
        for pair in intervals:
            if len(pair) != 2:
                raise ValueError("intervals must contain pairs")
            u, v = map(float, pair)
            if not math.isfinite(u) or not math.isfinite(v) or u >= v:
                raise ValueError("interval endpoints must be finite u < v")
            windows.append((u, v))
    guard = 1.e-8 * max(1, count)
    alpha = np.zeros(len(windows))
    z = np.ones(count)
    objective = None
    status, message = 0, "No eligible jobs or no interval constraints; trivial box certificate."
    if count and windows:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix, vstack
        matrix = vstack([csr_matrix(mandatory_pad_overlap(r, d, b, u, v).reshape(1, -1))
                         for u, v in windows], format="csr")
        rhs = np.asarray([pads * (v - u) for u, v in windows])
        solution = linprog(-np.ones(count), A_ub=matrix, b_ub=rhs,
                           bounds=(0., 1.), method="highs")
        status, message = int(solution.status), str(solution.message)
        if solution.success:
            objective = -float(solution.fun)
            alpha = np.maximum(0., -np.asarray(solution.ineqlin.marginals))
            z = np.maximum(0., -np.asarray(solution.upper.marginals))
            coverage = np.asarray(matrix.T @ alpha).reshape(-1)
            z = np.maximum(z, 1. - coverage) + 1.e-10
        coverage = np.asarray(matrix.T @ alpha).reshape(-1)
        dual = float(rhs @ alpha + z.sum())
        residual = max(0., float(np.max(1. - coverage - z)))
    else:
        dual, residual = float(count), 0.
        if count:
            objective = float(count)
    guarded = min(float(count), dual + guard)
    return {"bound_kind": "finite_cohort_pad_window_relaxation_upper",
            "N": n, "eligible_jobs": count, "excluded_jobs": n - count,
            "eligible_indices": selected.tolist(), "interval_count": len(windows),
            "relaxed_count_upper": guarded,
            "integer_count_upper": min(count, math.floor(guarded)),
            "lp_objective": objective, "solver_status": status, "solver_message": message,
            "dual_certificate_bound": dual, "dual_feasibility_residual": residual,
            "numerical_guard": guard,
            "dual_alpha": alpha.tolist(), "dual_z": z.tolist(),
            "assumptions": ["Only supplied interval pad-work constraints and optional job box bounds are retained.",
                            "Other cohorts, flight conflicts, geometry coupling and energy are omitted.",
                            "Floating-point repaired dual certificate, not exact rational proof or attainable capacity."]}


def finite_cohort_pad_window_upper(releases_s, service_s, *, sla_s, pads=6,
                                   touchdown_lower_s=73.5, protected_s=13.5,
                                   intervals=None, grid_step_s=60.):
    """Upper normal count meeting touchdown SLA measured from external release.

    Pad completion may occur S seconds after the touchdown deadline. External
    waiting does not consume SOC in the current engine, so no external-release
    SOC deadline is imposed by this deliberately optimistic relaxation.
    """
    releases = _vector(releases_s, "releases_s")
    service = _vector(service_s, "service_s")
    if (releases.shape != service.shape or np.any(service < 0)
            or not all(map(math.isfinite, (sla_s, touchdown_lower_s, protected_s)))
            or min(sla_s, protected_s) < 0 or touchdown_lower_s < protected_s):
        raise ValueError("matching nonnegative service, nonnegative SLA/descent and valid touchdown lower bound required")
    earliest = releases + touchdown_lower_s - protected_s
    deadline = releases + sla_s + service
    result = pad_window_lp_upper(earliest, deadline, service + protected_s, pads=pads,
                                intervals=intervals, grid_step_s=grid_step_s,
                                eligible=np.full(len(releases), sla_s >= touchdown_lower_s))
    result.update(sla_s=float(sla_s), touchdown_lower_s=float(touchdown_lower_s),
                  protected_s=float(protected_s))
    return result
