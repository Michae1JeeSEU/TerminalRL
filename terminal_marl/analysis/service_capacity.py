"""Mechanistic capacity primitives and a deliberately reduced inner-entry queue.

The FCFS reduced_model is neither an upper nor a lower bound on the optimized
terminal. It omits horizontal final/transfer paths, separation and upstream rings.
Rejected jobs are abstract nonnormal outcomes, never simulated energy failures.
No fitted efficiency coefficient or change to frozen engine configuration is used.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import heapq
import math
from typing import Iterable

import numpy as np
from scipy.special import ndtr, ndtri

from terminal_marl.core.rotating_slots.config import load_service_profile
from terminal_marl.core.rotating_slots.energy import evaluate_power, stable_power
from terminal_marl.core.rotating_slots.rescue_rules import normal_energy_lower_bound

MODEL_KIND = "reduced_model_inner_entry_fcfs"
DEFAULT_RATES = (60., 90., 120., 150., 180., 210., 240., 270., 300.)
DEFAULT_SEEDS = (11, 23, 47, 71, 101)
ASSUMPTIONS = [
    "Poisson arrivals enter directly at the inner ring; independent class and SOC marks.",
    "FCFS earliest-available pad: protected descent 13.5 s, then class ground service.",
    "K waiting slots are occupied only until protected descent starts; immediate service uses none.",
    "Reject on full waiting buffer before checking optimistic energy waiting allowance.",
    "Energy allowance is necessary and optimistic, not a feasible trajectory certificate.",
    "Horizontal final/transfer, separation, upstream rings and engine computation costs are omitted.",
    "Rejections are abstract nonnormal outcomes; no emergency or physical energy failure is simulated.",
    "Full calendar drain resolves the arrival cohort; end-of-window backlog is reported separately.",
    "empirical_normal_rate_per_h is normal outcomes of the arrival cohort divided by cohort duration; it includes later drained service, not departures within the window.",
    "Radius variants are reduced-model counterfactuals, not engine-approved geometry.",
    "FCFS reduced_model results are neither mathematical upper nor lower bounds on the full terminal.",
]


def ring_max_packing(radius_m: float, separation_m: float) -> int:
    """Maximum equally spaced point count on a circle under pairwise chord spacing.

    A singleton is permitted even when the diameter is below separation. This is
    within-ring point packing only, not a simultaneous multi-ring safety proof.
    """
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius must be positive and finite")
    if not math.isfinite(separation_m) or separation_m <= 0:
        raise ValueError("separation must be positive and finite")
    if 2 * radius_m < separation_m:
        return 1
    return math.floor(math.pi / math.asin(separation_m / (2 * radius_m)))


def _mix(durations_s, probabilities):
    durations = tuple(map(float, durations_s))
    probabilities = tuple(map(float, probabilities))
    if not durations or len(durations) != len(probabilities):
        raise ValueError("nonempty matching durations and probabilities required")
    if any(not math.isfinite(x) or x < 0 for x in durations + probabilities):
        raise ValueError("durations and probabilities must be finite and nonnegative")
    if not math.isclose(sum(probabilities), 1., abs_tol=1e-12):
        raise ValueError("probabilities must sum to one")
    return durations, probabilities


def pad_workload_moments(durations_s, probabilities, pads=6, protected_s=13.5):
    """All-served fixed-class-mix saturation upper rates, not service guarantees.

    Every admitted class must retain the supplied mix for this bound; selective
    rejection can alter the mix. Ground-only upper ignores protected descent.
    """
    durations, probabilities = _mix(durations_s, probabilities)
    if not isinstance(pads, int) or pads <= 0 or not math.isfinite(protected_s) or protected_s < 0:
        raise ValueError("positive integer pads and nonnegative protected time required")
    mean = sum(p * s for p, s in zip(probabilities, durations))
    second = sum(p * s * s for p, s in zip(probabilities, durations))
    occupied = mean + protected_s
    if mean <= 0 or occupied <= 0:
        raise ValueError("positive mean service required")
    return dict(ground_mean_s=mean, ground_second_moment_s2=second,
                ground_variance_s2=max(0., second - mean * mean),
                occupied_mean_s=occupied,
                occupied_second_moment_s2=second + 2 * protected_s * mean + protected_s**2,
                occupied_variance_s2=max(0., second - mean * mean),
                all_served_fixed_mix_upper_per_h=3600 * pads / occupied,
                ground_only_loose_upper_per_h=3600 * pads / mean)


def resource_workload_upper(capacities, per_class_workload, probabilities):
    """min_j m_j / sum_c p_c a_jc, all-served fixed mix only (per second).

    per_class_workload is a resource-by-class matrix in resource-seconds/job.
    This workload upper rate supplies no guaranteed service curve.
    """
    p = np.asarray(probabilities, dtype=float)
    a = np.asarray(per_class_workload, dtype=float)
    m = np.asarray(capacities, dtype=float)
    if (p.ndim != 1 or not p.size or a.shape != (m.size, p.size) or m.ndim != 1
            or not m.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(p))
            or not np.all(np.isfinite(m)) or np.any(a < 0) or np.any(p < 0)
            or np.any(m <= 0) or not math.isclose(float(p.sum()), 1., abs_tol=1e-12)):
        raise ValueError("invalid resource workload inputs")
    work = a @ p
    return min((float(x / y) for x, y in zip(m, work) if y > 0), default=math.inf)


def rate_latency_bounds(burst_jobs, arrival_rate_per_s, guaranteed_rate_per_s, latency_s):
    """Bounds ONLY assuming valid guaranteed beta=R(t-T)+ and alpha=b+lambda*t.

    lambda <= R implies backlog <= b+lambda*T and delay <= T+b/R.
    Saturation workload upper rates cannot establish this guaranteed R.
    """
    b, lam, rate, latency = map(float, (burst_jobs, arrival_rate_per_s, guaranteed_rate_per_s, latency_s))
    if not all(map(math.isfinite, (b, lam, rate, latency))) or min(b, lam, latency) < 0 or rate <= 0 or lam > rate:
        raise ValueError("finite nonnegative inputs, positive guaranteed R and lambda <= R required")
    return {"backlog_jobs": b + lam * latency, "delay_s": latency + b / rate}


def optimistic_waiting_allowance_s(soc, *, holding_power_w=None, config=None):
    """Optimistic necessary energy allowance; never a trajectory certificate.

    Exact energy accounting requires sum_r P_r W_r + E_move <= Q(SOC-reserve).
    Here reserve=.25 and the inner-ring normal completion lower bound replaces
    E_move/Q. Omitting transfers and actual motion makes the allowance optimistic.
    Negative allowances are retained: even zero wait fails this necessary test.
    """
    c = config or load_service_profile("service_half_v1")
    power = stable_power("inner", c).battery_w if holding_power_w is None else holding_power_w
    if not math.isfinite(float(soc)) or not math.isfinite(power) or power <= 0:
        raise ValueError("finite SOC and positive finite holding power required")
    capacity = c.energy.reference_capacity_Ah * c.energy.equivalent_voltage_V * 3600.
    return (float(soc) - .25 - normal_energy_lower_bound(c, "inner")) * capacity / power


@dataclass(frozen=True)
class Scenario:
    name: str
    pads: int = 6
    durations_s: tuple = (30., 60., 120.)
    probabilities: tuple = (.2, .6, .2)
    class_names: tuple = ("short", "standard", "long")
    soc_mean: float = .75
    soc_std: float = .08
    soc_low: float = .60
    soc_high: float = .90
    inner_radius_m: float = 300.


def scenarios():
    """Pre-specified one-factor controls at identical absolute arrival rates."""
    half = load_service_profile("service_half_v1")
    original = load_service_profile("baseline")
    names = tuple(half.pads.service_classes)
    durations = tuple(float(half.pads.service_classes[n].duration_s) for n in names)
    probs = tuple(float(half.pads.service_classes[n].probability) for n in names)
    baseline = dict(durations_s=durations, probabilities=probs, class_names=names)
    return {s.name: s for s in (
        Scenario("baseline6half", **baseline),
        Scenario("original6", durations_s=tuple(float(original.pads.service_classes[n].duration_s) for n in names), probabilities=probs, class_names=names),
        Scenario("pads3half", pads=3, **baseline),
        Scenario("low_soc6half", soc_mean=.60, soc_low=.45, soc_high=.75, **baseline),
        Scenario("same_mean_deterministic", durations_s=(66.,), probabilities=(1.,), class_names=("deterministic",)),
        Scenario("same_mean_high_variance", durations_s=(30., 120.), probabilities=(.6, .4), class_names=("short", "long")),
        Scenario("inner_radius250", inner_radius_m=250., **baseline),
        Scenario("inner_radius350", inner_radius_m=350., **baseline),
    )}


def scenario_parameters(scenario):
    c = load_service_profile("service_half_v1")
    omega = c.airspace.common_angular_speed_radps
    radius = scenario.inner_radius_m
    if radius == 300.:
        power = stable_power("inner", c)
    else:
        power = evaluate_power((0., radius * omega, 0.), (-radius * omega**2, 0., 0.), c)
    if not power.valid:
        raise ValueError("invalid counterfactual holding power: " + power.reason)
    return {"waiting_buffer": ring_max_packing(radius, c.airspace.slot_design_min_chord_m),
            "holding_power_w": power.battery_w,
            "protected_s": float(c.pads.vertical_descent_profile.total_duration_s),
            "radius_counterfactual": radius != 300.}


def constants_table():
    c = load_service_profile("service_half_v1")
    return {"config_hash": c.config_hash, "vehicle_profile": c.vehicle.profile_id,
            "capacity_j": c.energy.reference_capacity_Ah * c.energy.equivalent_voltage_V * 3600.,
            "normal_reserve_soc": .25, "inner_normal_energy_lower_bound_soc": normal_energy_lower_bound(c, "inner"),
            "protected_descent_s": c.pads.vertical_descent_profile.total_duration_s,
            "pad_layout_radius_m": c.pads.layout_radius_m,
            "rings": [{"id": r.id, "radius_m": r.radius_m, "configured_slots": r.slot_count,
                       "design_d100_packing": ring_max_packing(r.radius_m, 100.),
                       "hypothetical_hard_d80_packing": ring_max_packing(r.radius_m, 80.),
                       "holding_power_w": stable_power(r.id, c).battery_w} for r in c.airspace.rings],
            "scenarios": {name: {**asdict(s), **scenario_parameters(s),
                          "pad_moments": pad_workload_moments(s.durations_s, s.probabilities, s.pads)}
                          for name, s in scenarios().items()}}


def generate_marks(seed, rate_per_h, horizon_s, scenario):
    """Separate streams; same seed reuses arrival/class/SOC uniforms across scenarios."""
    if not math.isfinite(rate_per_h) or rate_per_h <= 0 or not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("positive finite rate and horizon required")
    arrival_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
    class_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    soc_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2]))
    arrivals = []
    now = 0.
    while True:
        now += -math.log1p(-float(arrival_rng.random())) * 3600. / rate_per_h
        if now >= horizon_s:
            break
        arrivals.append(now)
    n = len(arrivals)
    classes = np.searchsorted(np.cumsum(scenario.probabilities), class_rng.random(n), side="right")
    if scenario.soc_std <= 0 or not scenario.soc_low < scenario.soc_high:
        raise ValueError("positive SOC standard deviation and ordered truncation bounds required")
    lo = ndtr((scenario.soc_low - scenario.soc_mean) / scenario.soc_std)
    hi = ndtr((scenario.soc_high - scenario.soc_mean) / scenario.soc_std)
    socs = scenario.soc_mean + scenario.soc_std * ndtri(lo + soc_rng.random(n) * (hi - lo))
    return np.asarray(arrivals), classes, socs


def fcfs_queue(arrivals_s, class_indices, waiting_allowances_s, *, durations_s, pads,
               waiting_buffer, protected_s=13.5):
    """Resolve every supplied arrival with an exact event calendar for this queue.

    Scheduled future starts release waiting slots at their start timestamp, before
    arrivals at that timestamp. Rejected jobs reserve no pad or queue capacity.
    Returned event records support deterministic checks; campaign rows are compact.
    """
    arrivals = list(map(float, arrivals_s))
    classes = list(class_indices)
    allowances = list(map(float, waiting_allowances_s))
    durations = tuple(map(float, durations_s))
    if len(arrivals) != len(classes) or len(arrivals) != len(allowances):
        raise ValueError("matching arrival marks required")
    if (not isinstance(pads, int) or pads < 1 or not isinstance(waiting_buffer, int) or waiting_buffer < 0
            or not math.isfinite(protected_s) or protected_s < 0 or not durations
            or any(not math.isfinite(s) or s < 0 for s in durations)):
        raise ValueError("invalid queue resources or durations")
    if any(not math.isfinite(t) or t < 0 for t in arrivals) or any(a > b for a, b in zip(arrivals, arrivals[1:])):
        raise ValueError("arrivals must be finite, nonnegative and ordered")
    if any(not math.isfinite(b) for b in allowances):
        raise ValueError("finite waiting allowances required")
    release_heap = [(0., pad) for pad in range(pads)]
    waiting_starts = []
    events = []
    for arrival, cls, allowance in zip(arrivals, classes, allowances):
        if int(cls) != cls or not 0 <= cls < len(durations):
            raise ValueError("invalid class index")
        cls = int(cls)
        while waiting_starts and waiting_starts[0] <= arrival:
            heapq.heappop(waiting_starts)
        earliest, pad = release_heap[0]
        start = max(arrival, earliest)
        wait = start - arrival
        event = {"arrival_s": arrival, "class_index": cls, "allowance_s": allowance,
                 "outcome": "normal", "wait_s": wait, "start_s": None,
                 "touchdown_s": None, "release_s": None, "pad": None}
        if wait > 0 and len(waiting_starts) >= waiting_buffer:
            event["outcome"] = "reject_buffer"
        elif wait > allowance:
            event["outcome"] = "reject_deadline"
        else:
            release = start + protected_s + durations[cls]
            heapq.heapreplace(release_heap, (release, pad))
            if wait > 0:
                heapq.heappush(waiting_starts, start)
            event.update(start_s=start, touchdown_s=start + protected_s, release_s=release, pad=pad)
        events.append(event)
    return events


def _quantiles(values):
    return {f"p{q}": float(np.percentile(values, q)) if values else None for q in (50, 90, 95, 99, 100)}


def run_task(task):
    """One independent (scenario, seed, absolute rate) task; no external writes."""
    name, seed, rate, warmup, duration = task
    if warmup < 0 or duration <= 0 or not all(map(math.isfinite, (warmup, duration))):
        raise ValueError("nonnegative warmup and positive duration required")
    scenario = scenarios()[name]
    params = scenario_parameters(scenario)
    end = warmup + duration
    arrivals, classes, socs = generate_marks(seed, rate, end, scenario)
    config = load_service_profile("service_half_v1")
    allowances = [optimistic_waiting_allowance_s(s, holding_power_w=params["holding_power_w"], config=config) for s in socs]
    events = fcfs_queue(arrivals, classes, allowances, durations_s=scenario.durations_s,
                        pads=scenario.pads, waiting_buffer=params["waiting_buffer"], protected_s=params["protected_s"])
    cohort = [e for e in events if e["arrival_s"] >= warmup]
    accepted = [e for e in events if e["outcome"] == "normal"]
    normal = [e for e in cohort if e["outcome"] == "normal"]
    counts = {key: sum(e["outcome"] == key for e in cohort) for key in ("normal", "reject_buffer", "reject_deadline")}
    classes_result = {}
    for i, cls in enumerate(scenario.class_names):
        subset = [e for e in cohort if e["class_index"] == i]
        count = {key: sum(e["outcome"] == key for e in subset) for key in counts}
        classes_result[cls] = {"arrivals": len(subset), **count,
                               "normal_fraction": count["normal"] / len(subset) if subset else None,
                               "normal_wait_quantiles_s": _quantiles([e["wait_s"] for e in subset if e["outcome"] == "normal"])}
    return {"model_kind": MODEL_KIND, "scenario": name, "seed": int(seed), "offered_rate_per_h": float(rate),
            "warmup_s": float(warmup), "cohort_duration_s": float(duration), "cohort_end_s": end,
            "arrivals": len(cohort), **counts,
            "empirical_offered_rate_per_h": len(cohort) * 3600. / duration,
            "empirical_normal_rate_per_h": len(normal) * 3600. / duration,
            "normal_fraction": len(normal) / len(cohort) if cohort else None,
            "normal_wait_quantiles_s": _quantiles([e["wait_s"] for e in normal]),
            "class_outcomes": classes_result,
            "end_waiting_backlog": sum(e["start_s"] > end for e in accepted),
            "end_in_service": sum(e["start_s"] <= end < e["release_s"] for e in accepted),
            "calendar_drain_end_s": max([end] + [e["release_s"] for e in accepted]),
            "drained_waiting_backlog": 0, "drained_in_service": 0,
            "class_probabilities": dict(zip(scenario.class_names, scenario.probabilities)),
            "class_durations_s": dict(zip(scenario.class_names, scenario.durations_s)),
            **params, "pad_moments": pad_workload_moments(scenario.durations_s, scenario.probabilities, scenario.pads)}


def run_tasks(tasks: Iterable, workers=1, progress_callback=None):
    """Collect all outcomes, including visible failed-task evidence, before writing."""
    tasks = [tuple(task) for task in tasks]
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    rows, failed = [], []
    def progress():
        if progress_callback is not None:
            progress_callback({"expected": len(tasks), "completed": len(rows), "failed": len(failed)})
    if workers == 1:
        for task in tasks:
            try:
                rows.append(run_task(task))
            except Exception as error:
                failed.append({"task": list(task), "error": f"{type(error).__name__}: {error}"})
            progress()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    rows.append(future.result())
                except Exception as error:
                    failed.append({"task": list(task), "error": f"{type(error).__name__}: {error}"})
                progress()
    rows.sort(key=lambda r: (r["scenario"], r["seed"], r["offered_rate_per_h"]))
    failed.sort(key=lambda f: tuple(f["task"]))
    keys = [(r["scenario"], r["seed"], r["offered_rate_per_h"], r["warmup_s"], r["cohort_duration_s"]) for r in rows]
    expected_keys = set(tasks)
    counts = {"expected": len(tasks), "completed": len(rows), "failed": len(failed),
              "duplicate": len(keys) - len(set(keys)),
              "missing": len(expected_keys - set(keys)), "duplicate_requested": len(tasks) - len(expected_keys)}
    return rows, failed, counts
