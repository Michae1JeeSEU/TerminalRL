"""Paired demand experiments in the complete rotating-slot physical world.

This is a separate capacity study, not a continuation of a frozen Stage2 campaign.
Measured scheduler throughput is descriptive and never an optimal capacity bound.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
from statistics import NormalDist
import sys
import time
import traceback

from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest

SCHEMA = "full_service_capacity_v1"
ROOT = Path(__file__).resolve().parents[2]
METHODS = ("energy_slack", "rh_cp_sat")
CLASSES = ("short", "standard", "long")
POLICY = dict(planning_profile="downstream_service_v8", budget_profile="quality_v5",
              budget_scale=1., online_policy=None, prefix_reuse=False)
_ASSIGNED_CPU = None


@dataclass(frozen=True)
class Profile:
    name: str
    inner_radius_m: float = 300.
    pad_count: int = 6
    service_seconds: tuple = (30., 60., 120.)
    service_probabilities: tuple = (.2, .6, .2)
    soc_mean: float = .75
    soc_std: float = .08
    soc_low: float = .60
    soc_high: float = .90


def profiles():
    return {p.name: p for p in (
        Profile("half6"), Profile("original6", service_seconds=(60., 120., 240.)),
        Profile("pads3half", pad_count=3), Profile("radius250", inner_radius_m=250.),
        Profile("radius350", inner_radius_m=350.),
        Profile("low_soc", soc_mean=.60, soc_low=.45, soc_high=.75),
        Profile("deterministic66", service_seconds=(66., 66., 66.)),
        Profile("high_variance66", service_probabilities=(.6, 0., .4)),
    )}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False).encode("utf-8")


def digest(value):
    return sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append(path, rows):
    if rows:
        with Path(path).open("ab") as stream:
            for row in rows:
                stream.write(canonical(row) + b"\n")
            stream.flush()


def validate_spec(value):
    """Validate the small explicit experiment schema; no implicit rate sweep."""
    allowed = {"profiles", "methods", "seeds", "rates_per_hour", "warmup_s",
               "measurement_s", "drain_s", "sla_delay_s", "progress_interval_s", "demand_kind"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError("unknown spec fields: " + ", ".join(sorted(unknown)))
    spec = dict(profiles=list(profiles()), methods=list(METHODS), seeds=[11],
                rates_per_hour=[120.], warmup_s=0., measurement_s=3600., drain_s=600.,
                sla_delay_s=600., progress_interval_s=60., demand_kind="poisson")
    spec.update(value)
    for key, choices in (("profiles", profiles()), ("methods", METHODS)):
        items = spec[key]
        if not isinstance(items, list) or not items or len(set(items)) != len(items) or any(x not in choices for x in items):
            raise ValueError("invalid or duplicate " + key)
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    rates = spec["rates_per_hour"]
    if not isinstance(rates, list) or not rates or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in rates) or len(set(rates)) != len(rates):
        raise ValueError("rates_per_hour must be unique positive finite numbers")
    for key in ("warmup_s", "measurement_s", "drain_s", "sla_delay_s", "progress_interval_s"):
        x = spec[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or (key in ("measurement_s", "progress_interval_s") and x == 0):
            raise ValueError("invalid " + key)
        spec[key] = float(x)
    if spec["demand_kind"] not in ("poisson", "single"):
        raise ValueError("demand_kind must be poisson or single")
    if spec["demand_kind"] == "single" and spec["warmup_s"] != 0:
        raise ValueError("single request experiments require warmup_s=0")
    return spec


def _rng(seed, stream):
    return random.Random(int.from_bytes(sha256(f"{SCHEMA}:{seed}:{stream}".encode()).digest(), "big"))


def soc_from_uniform(uniform, profile):
    normal = NormalDist(profile.soc_mean, profile.soc_std)
    low, high = normal.cdf(profile.soc_low), normal.cdf(profile.soc_high)
    return normal.inv_cdf(low + uniform * (high - low))


def task_from_uniform(uniform, profile):
    cumulative = 0.
    for name, probability in zip(CLASSES, profile.service_probabilities):
        cumulative += probability
        if uniform < cumulative:
            return name
    return CLASSES[-1]


def generate_manifest(seed, rate_per_hour, horizon_s, profile, demand_kind="poisson"):
    """Hash-separated streams pair releases, gates and latent marks across profiles."""
    if isinstance(profile, str):
        profile = profiles()[profile]
    if not math.isfinite(rate_per_hour) or rate_per_hour <= 0 or not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("positive finite rate and horizon required")
    if demand_kind not in ("poisson", "single"):
        raise ValueError("unknown demand kind")
    arrival, gate, soc, task = (_rng(seed, stream) for stream in ("arrival", "gate", "SOC", "task"))
    releases, now = [], 0.
    if demand_kind == "single":
        releases = [0.]
    else:
        while True:
            now += arrival.expovariate(rate_per_hour / 3600.)
            if now >= horizon_s:
                break
            releases.append(now)
    rows, marks = [], []
    for index, release in enumerate(releases):
        u_soc, u_task = soc.random(), task.random()
        gate_id = gate.randrange(4)
        initial_soc = (.85 if profile.name != "low_soc" else soc_from_uniform(.5, profile)) if demand_kind == "single" else soc_from_uniform(u_soc, profile)
        service_class = "standard" if demand_kind == "single" else task_from_uniform(u_task, profile)
        request = ArrivalRequest(f"req_{index:06d}", release, gate_id, initial_soc, service_class)
        rows.append(request)
        marks.append(dict(id=request.id, release_time=release, gate_id=gate_id, soc_uniform=u_soc, task_uniform=u_task))
    manifest = ArrivalManifest(tuple(rows), seed, float(rate_per_hour), float(horizon_s))
    return dict(schema=SCHEMA, demand_kind=demand_kind, profile=profile.name,
                manifest=manifest.to_dict(), latent_marks=marks, latent_marks_hash=digest(marks))


def make_config(profile):
    from terminal_marl.core.rotating_slots.capacity_config import make_capacity_config
    return make_capacity_config(inner_radius_m=profile.inner_radius_m, pad_count=profile.pad_count,
                                service_seconds=profile.service_seconds,
                                service_probabilities=profile.service_probabilities)


def design_metadata(profile, config=None):
    from terminal_marl.core.rotating_slots.energy import stable_power
    config = config or make_config(profile)
    mean = sum(p * s for p, s in zip(profile.service_probabilities, profile.service_seconds))
    variance = sum(p * (s - mean)**2 for p, s in zip(profile.service_probabilities, profile.service_seconds))
    descent = float(config.pads.vertical_descent_profile.total_duration_s)
    return dict(profile=asdict(profile), service_mean_s=mean, service_variance_s2=variance,
                protected_descent_s=descent, pad_workload_necessary_upper_per_h=3600. * profile.pad_count / (descent + mean),
                bound_definition="Necessary pad workload bound for all-served fixed class mix; no service or system capacity guarantee.",
                rings=[dict(id=r.id, radius_m=r.radius_m, height_agl_m=r.height_agl_m,
                            design_K=r.slot_count, angular_speed_radps=config.airspace.common_angular_speed_radps,
                            speed_mps=r.radius_m * config.airspace.common_angular_speed_radps,
                            stable_holding_power_w=stable_power(r.id, config).battery_w)
                       for r in config.airspace.rings],
                battery_capacity_j=config.energy.reference_capacity_Ah * config.energy.equivalent_voltage_V * 3600.)


def _quantiles(values):
    if not values:
        return dict(p50=None, p90=None, p95=None, p99=None, maximum=None)
    ordered = sorted(values)
    def percentile(p):
        x = (len(ordered) - 1) * p
        lo = math.floor(x)
        hi = math.ceil(x)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)
    return dict(p50=percentile(.5), p90=percentile(.9), p95=percentile(.95),
                p99=percentile(.99), maximum=ordered[-1])


def cohort_metrics(requests, aircraft, events, *, start_s, end_s, sla_delay_s, failure_ids=()):
    """Pure accounting helper: every offered request remains in its denominator."""
    requests = [r.to_dict() if hasattr(r, "to_dict") else r for r in requests]
    aircraft = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in aircraft.items()}
    touchdowns = {e["aircraft_id"]: e for e in events if e["type"] == "touchdown"}
    failures = set(failure_ids)
    counts = dict(offered=0, normal_touchdown=0, normal_within_sla=0,
                  normal_reserve_ge_025=0, normal_within_sla_and_reserve_ge_025=0,
                  emergency=0, failure=0, rerouted=0, upstream_pending=0, airborne_pending=0)
    delays, per_class, reserve_unknown = [], {}, 0
    for request in requests:
        if not start_s <= request["release_time"] < end_s:
            continue
        name, key = request["service_class"], request["id"]
        row = per_class.setdefault(name, {field: 0 for field in counts})
        counts["offered"] += 1
        row["offered"] += 1
        a, event = aircraft.get(key), touchdowns.get(key)
        increments = []
        if a is not None and a.get("touchdown_time") is not None:
            emergency = (event["details"]["emergency"] if event else a.get("emergency", {}).get("charged", False))
            if emergency:
                increments.append("emergency")
            else:
                increments.append("normal_touchdown")
                delay = a["touchdown_time"] - request["release_time"]
                delays.append(delay)
                within = delay <= sla_delay_s
                if within:
                    increments.append("normal_within_sla")
                touchdown_soc = event["details"].get("soc") if event else None
                if touchdown_soc is None:
                    reserve_unknown += 1
                elif touchdown_soc >= .25:
                    increments.append("normal_reserve_ge_025")
                    if within:
                        increments.append("normal_within_sla_and_reserve_ge_025")
        elif key in failures or (a and a.get("status") in ("ENERGY_FAILURE", "EXECUTED_HARD_SEPARATION_VIOLATION", "CENSORED_DUE_TO_FAILURE")):
            increments.append("failure")
        elif a and a.get("status") == "REROUTED":
            increments.append("rerouted")
        elif a is None:
            increments.append("upstream_pending")
        else:
            increments.append("airborne_pending")
        for increment in increments:
            counts[increment] += 1
            row[increment] += 1
    outcomes = ("normal_touchdown", "emergency", "failure", "rerouted", "upstream_pending", "airborne_pending")
    return dict(**counts, per_class=per_class, normal_delay_quantiles_s=_quantiles(delays),
                normal_touchdown_soc_unknown=reserve_unknown,
                normal_ontime_fraction_all_offered=counts["normal_within_sla"] / counts["offered"] if counts["offered"] else None,
                normal_ontime_reserve_fraction_all_offered=counts["normal_within_sla_and_reserve_ge_025"] / counts["offered"] if counts["offered"] else None,
                population_conservation=counts["offered"] == sum(counts[k] for k in outcomes))


def source_identity():
    files = list((ROOT / "terminal_marl/core/rotating_slots").rglob("*.py"))
    files += list((ROOT / "terminal_marl/core/rotating_slots").rglob("*.yaml"))
    files += list((ROOT / "configs").rglob("*.yaml"))
    files += [ROOT / "new10.16/fixed_terminal_rotating_slots_stage1_config_v1_0.yaml",
              ROOT / "new10.16/fixed_terminal_rotating_slots_stage2_service_half_v1.yaml"]
    files += [Path(__file__), ROOT / "scripts/analysis/run_full_service_capacity.py"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(set(files)) if p.exists()}
    return dict(files=hashes, sha256=digest(hashes))


def runtime_identity():
    versions = {}
    for name in ("numpy", "scipy", "PyYAML", "ortools"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, interpreter=sys.executable, platform=platform.platform(),
                packages=versions, numerical_threads=1, cp_workers=1)


def solver_preflight():
    """Exercise the concrete runtime API. Missing solvers never change methods."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    x = model.NewBoolVar("capacity_runtime_probe")
    model.Add(x == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL or solver.Value(x) != 1:
        raise RuntimeError("CP-SAT concrete runtime preflight failed: " + str(status))
    return dict(status=solver.StatusName(status), solver_version=importlib.metadata.version("ortools"))


def prepare_jobs(spec, output):
    """Freeze every paired manifest and identity before any worker is launched."""
    spec, output = validate_spec(spec), Path(output)
    source, runtime = source_identity(), runtime_identity()
    output.mkdir(parents=True, exist_ok=True)
    campaign = dict(schema=SCHEMA, spec=spec, policy=POLICY, source=source, runtime=runtime)
    campaign_path = output / "campaign.json"
    if campaign_path.exists() and json.loads(campaign_path.read_bytes()) != campaign:
        raise ValueError("campaign identity mismatch; choose a new output directory")
    atomic_json(campaign_path, campaign)
    jobs = []
    for name in spec["profiles"]:
        profile = profiles()[name]
        config = make_config(profile)
        for rate in spec["rates_per_hour"]:
            for seed in spec["seeds"]:
                manifest = generate_manifest(seed, rate, spec["warmup_s"] + spec["measurement_s"], profile, spec["demand_kind"])
                manifest_path = output / "manifests" / f"{name}_rate{rate:g}_seed{seed}.json"
                if manifest_path.exists() and json.loads(manifest_path.read_bytes()) != manifest:
                    raise ValueError("manifest identity mismatch: " + str(manifest_path))
                atomic_json(manifest_path, manifest)
                for method in spec["methods"]:
                    params = dict(profile=name, rate_per_hour=rate, seed=seed, method=method,
                                  warmup_s=spec["warmup_s"], measurement_s=spec["measurement_s"],
                                  drain_s=spec["drain_s"], sla_delay_s=spec["sla_delay_s"],
                                  progress_interval_s=spec["progress_interval_s"], demand_kind=spec["demand_kind"])
                    identity = dict(schema=SCHEMA, job=params, config_hash=config.config_hash,
                                    manifest_hash=digest(manifest), source=source, runtime=runtime, policy=POLICY)
                    job_dir = output / "jobs" / f"{name}_rate{rate:g}_seed{seed}_{method}"
                    identity_path = job_dir / "identity.json"
                    if identity_path.exists() and json.loads(identity_path.read_bytes()) != identity:
                        raise ValueError("job identity mismatch: " + str(job_dir))
                    for filename, expected in (("config.json", config.to_dict()), ("manifest.json", manifest)):
                        path = job_dir / filename
                        if path.exists() and json.loads(path.read_bytes()) != expected:
                            raise ValueError("job " + filename + " identity mismatch: " + str(job_dir))
                    atomic_json(identity_path, identity)
                    atomic_json(job_dir / "config.json", config.to_dict())
                    atomic_json(job_dir / "manifest.json", manifest)
                    jobs.append(dict(directory=str(job_dir.resolve()), identity=identity))
    return jobs


def _compact_decision(record):
    # Preserve solver histories (statuses, bounds/gaps), budgets, catalog counts,
    # validation/timing and the actually submitted actions, without giant catalogs.
    excluded = {"aircraft_choices", "downstream", "resource_aligned_starts", "virtual_queue_order", "virtual_pad_assignment"}
    compact = {key: value for key, value in record.items() if key not in excluded}
    if "solver" in compact:
        compact["solver"] = [dict(row, absolute_objective_bound_gap=(
            abs(row["objective"] - row["bound"]) if row.get("objective") is not None and row.get("bound") is not None else None))
            for row in compact["solver"]]
    return compact


def _set_worker_runtime(cpu=None):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    if cpu is not None:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {cpu})
        else:
            try:
                import psutil
                psutil.Process().cpu_affinity([cpu])
            except ImportError as exc:
                raise RuntimeError("CPU affinity on this platform requires psutil") from exc


def run_job(job, cpu=None):
    """Execute one immutable job; exceptions are explicit failed artifacts."""
    cpu = _ASSIGNED_CPU if cpu is None else cpu
    directory, identity = Path(job["directory"]), job["identity"]
    result_path = directory / "results.json"
    if json.loads((directory / "identity.json").read_bytes()) != identity:
        raise ValueError("stored job identity mismatch")
    if source_identity() != identity["source"] or runtime_identity() != identity["runtime"]:
        raise ValueError("source/runtime changed after job preparation")
    if digest(json.loads((directory / "manifest.json").read_bytes())) != identity["manifest_hash"]:
        raise ValueError("stored manifest identity mismatch")
    from terminal_marl.core.rotating_slots.config import digest as config_digest
    if config_digest(json.loads((directory / "config.json").read_bytes())) != identity["config_hash"]:
        raise ValueError("stored config identity mismatch")
    if result_path.exists():
        previous = json.loads(result_path.read_bytes())
        if previous.get("identity_hash") != digest(identity):
            raise ValueError("result identity mismatch")
        if previous.get("status") == "COMPLETED":
            return dict(directory=str(directory), status="COMPLETED", skipped=True)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    world, scheduler = None, None
    params = identity["job"]
    callback_count, latest_callback_wall = 0, None
    boundary_rows = []
    for filename in ("events.jsonl", "decisions.jsonl", "dispatch_timings.jsonl", "timeseries.jsonl"):
        (directory / filename).write_bytes(b"")
    def queue_counts():
        if world is None:
            return dict(upstream=0, airborne=0)
        return dict(upstream=sum(key not in world.aircraft for key in world.requests),
                    airborne=sum(a.status in AIRBORNE for a in world.aircraft.values()))
    def progress(status):
        atomic_json(directory / "progress.json", dict(status=status, sim_time_s=world.time if world else 0.,
                    wall_s=time.perf_counter() - wall_start, cpu_s=time.process_time() - cpu_start,
                    callback_count=callback_count, latest_callback_wall_s=latest_callback_wall,
                    queues=queue_counts(), pid=os.getpid(), boundaries=boundary_rows))
    try:
        _set_worker_runtime(cpu)
        if params["method"] == "rh_cp_sat":
            solver_preflight()
        from terminal_marl.core.rotating_slots.stage2_campaign import shared_fifo_admission
        from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
        from terminal_marl.core.rotating_slots.world import AIRBORNE, RotatingSlotWorld, STAGE2_POLICY_ID
        profile = profiles()[params["profile"]]
        config = make_config(profile)
        frozen = json.loads((directory / "manifest.json").read_bytes())
        if digest(frozen) != identity["manifest_hash"] or config.config_hash != identity["config_hash"]:
            raise ValueError("config or manifest changed after preparation")
        data = frozen["manifest"]
        manifest = ArrivalManifest(tuple(ArrivalRequest(**r) for r in data["requests"]), data["seed"], data["rate_per_hour"], data["arrival_window"])
        world = RotatingSlotWorld(config=config, manifest=manifest, state_sink=lambda record: None,
                                  experiment_policy_id=STAGE2_POLICY_ID)
        world.rescue_rule_profile = "rescue_need_v7"
        world.admission_callback = shared_fifo_admission
        scheduler = NormalSchedulerCallback(method=params["method"], **POLICY)
        def callback(snapshot, port):
            nonlocal callback_count, latest_callback_wall
            callback_count += 1
            callback_started = time.perf_counter()
            progress("DISPATCHING")
            try:
                result = scheduler(snapshot, port)
            finally:
                latest_callback_wall = time.perf_counter() - callback_started
            _append(directory / "decisions.jsonl", [_compact_decision(row) for row in scheduler.records])
            scheduler.records.clear()
            progress("RUNNING")
            return result
        warmup = params["warmup_s"]
        observation_end = warmup + params["measurement_s"]
        end = observation_end + params["drain_s"]
        boundaries = {warmup: "measurement_start", observation_end: "measurement_end", end: "final"}
        # Duplicate endpoints (zero warmup/drain) retain every boundary label.
        boundary_labels = [(warmup, "measurement_start"), (observation_end, "measurement_end"), (end, "final")]
        targets = set(boundaries)
        targets.update(float(x) for x in range(60, math.ceil(end), 60))
        interval = params["progress_interval_s"]
        targets.update(min(end, index * interval) for index in range(1, math.ceil(end / interval) + 1))
        pad_at_start = pad_at_end = None
        event_cursor, timing_cursor = 0, 0
        progress("RUNNING")
        for target in sorted(targets):
            world.advance(target, decision_callback=callback)
            _append(directory / "events.jsonl", world.events[event_cursor:])
            event_cursor = len(world.events)
            _append(directory / "dispatch_timings.jsonl", world.dispatch_timings[timing_cursor:])
            timing_cursor = len(world.dispatch_timings)
            for boundary, label in boundary_labels:
                if target == boundary:
                    boundary_rows.append(dict(label=label, sim_time_s=world.time, **queue_counts(),
                                              pad_busy_seconds=dict(world.pad_busy_seconds)))
            if target == warmup:
                pad_at_start = dict(world.pad_busy_seconds)
            if target == observation_end:
                pad_at_end = dict(world.pad_busy_seconds)
            _append(directory / "timeseries.jsonl", [dict(sim_time_s=world.time, **queue_counts(),
                offered=world._manifest_index, aircraft_count=len(world.aircraft),
                touchdown_count=sum(a.touchdown_time is not None for a in world.aircraft.values()),
                failure_count=len(world.failure_archive), pad_busy_seconds=dict(world.pad_busy_seconds),
                pad_busy_count=len({r.resource for r in world.pads.reservations if r.start <= world.time < r.end}),
                wall_s=time.perf_counter() - wall_start)])
            progress("RUNNING")
            if world.failed or world.terminated_invalid or abs(world.time - target) > 1e-7:
                raise RuntimeError(f"physical world stopped before valid completion: time={world.time}, target={target}, failed={world.failed}, terminated_invalid={world.terminated_invalid}")
        metrics = world.metrics()
        if not metrics["conservation"]:
            raise RuntimeError("physical world population conservation failed")
        measurement = cohort_metrics(manifest.requests, world.aircraft, world.events, start_s=warmup,
                                     end_s=observation_end, sla_delay_s=params["sla_delay_s"], failure_ids=world.failure_archive)
        warmup_metrics = cohort_metrics(manifest.requests, world.aircraft, world.events, start_s=0.,
                                       end_s=warmup, sla_delay_s=params["sla_delay_s"], failure_ids=world.failure_archive)
        if not measurement["population_conservation"] or not warmup_metrics["population_conservation"]:
            raise RuntimeError("cohort population conservation failed")
        meta = design_metadata(profile, config)
        capacity = meta["battery_capacity_j"]
        initial = {r.id: r.initial_soc for r in manifest.requests}
        energy = [dict(id=a.id, energy_j=a.energy_j, soc_initial=initial[a.id], soc_final=a.soc,
                       capacity_times_soc_drop_j=capacity * (initial[a.id] - a.soc),
                       residual_j=a.energy_j - capacity * (initial[a.id] - a.soc))
                  for a in world.aircraft.values() if a.admitted]
        window_events = [e for e in world.events if e["type"] == "touchdown" and warmup <= e["time"] < observation_end]
        busy = {str(pad): pad_at_end[pad] - pad_at_start[pad] for pad in pad_at_start}
        start_queue = next(row for row in boundary_rows if row["label"] == "measurement_start")
        end_queue = next(row for row in boundary_rows if row["label"] == "measurement_end")
        result = dict(schema=SCHEMA, status="COMPLETED", identity_hash=digest(identity), params=params,
                      design=meta, measurement_cohort=measurement, warmup_cohort=warmup_metrics,
                      observation_window=dict(start_s=warmup, end_s=observation_end,
                          touchdowns=len(window_events), normal_touchdowns=sum(not e["details"]["emergency"] for e in window_events),
                          touchdown_rate_per_h=3600. * len(window_events) / params["measurement_s"],
                          normal_touchdown_rate_per_h=3600. * sum(not e["details"]["emergency"] for e in window_events) / params["measurement_s"],
                          pad_busy_seconds=busy, pad_busy_utilization=sum(busy.values()) / (profile.pad_count * params["measurement_s"]),
                          pad_utilization_definition="Difference of existing physical pad_busy_seconds counters over observation window (protected descent plus ground service calendar occupancy)."),
                      cohort_normal_throughput_per_h=3600. * measurement["normal_touchdown"] / params["measurement_s"],
                      boundaries=boundary_rows,
                      empirical_queue_drift=dict(upstream_per_h=3600. * (end_queue["upstream"] - start_queue["upstream"]) / params["measurement_s"],
                          airborne_per_h=3600. * (end_queue["airborne"] - start_queue["airborne"]) / params["measurement_s"],
                          interpretation="Descriptive finite-window drift; no stationarity or stability proof."),
                      safety=dict(minimum_separation=world.minimum_separation, terminated_invalid=world.terminated_invalid,
                          conservation=metrics["conservation"], failure_count=len(world.failure_archive), failures=world.failure_archive),
                      energy_identity=energy, world_metrics=metrics, callback_count=callback_count,
                      latest_callback_wall_s=latest_callback_wall,
                      wall_s=time.perf_counter() - wall_start, cpu_s=time.process_time() - cpu_start,
                      pid=os.getpid(), assigned_cpu=cpu)
        atomic_json(directory / "aircraft.json", {k: a.to_dict() for k, a in world.aircraft.items()})
        atomic_json(directory / "trajectories.json", world.trajectories)
        result.update(wall_s=time.perf_counter() - wall_start, cpu_s=time.process_time() - cpu_start)
        atomic_json(result_path, result)
        progress("COMPLETED")
        return dict(directory=str(directory), status="COMPLETED", skipped=False)
    except Exception as exc:
        status = "INVALID" if world is not None and (world.terminated_invalid or world.failed) else "FAILED"
        result = dict(schema=SCHEMA, status=status, identity_hash=digest(identity), params=params,
                      exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc(),
                      sim_time_s=world.time if world else 0., callback_count=callback_count,
                      wall_s=time.perf_counter() - wall_start, cpu_s=time.process_time() - cpu_start)
        if world is not None:
            # Incomplete physical state is evidence, never a completed scientific result.
            atomic_json(directory / "aircraft.json", {k: a.to_dict() for k, a in world.aircraft.items()})
            atomic_json(directory / "trajectories.json", world.trajectories)
            (directory / "events.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in world.events))
            (directory / "dispatch_timings.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in world.dispatch_timings))
        atomic_json(result_path, result)
        progress(status)
        return dict(directory=str(directory), status=status, exception=str(exc))


def available_cpus():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    try:
        import psutil
        return psutil.Process().cpu_affinity()
    except (ImportError, AttributeError):
        return list(range(os.cpu_count() or 1))


def _worker_initializer(cpu_queue):
    global _ASSIGNED_CPU
    _ASSIGNED_CPU = cpu_queue.get()
    _set_worker_runtime(_ASSIGNED_CPU)


def run_campaign(spec, output, *, workers, cpus=None):
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    available = available_cpus()
    if cpus is not None:
        cpus = list(cpus)
        if not cpus or len(set(cpus)) != len(cpus) or any(type(cpu) is not int or cpu not in available for cpu in cpus):
            raise ValueError("cpus must be unique available CPU identifiers")
        if not hasattr(os, "sched_setaffinity"):
            try:
                import psutil  # noqa: F401 -- validate explicit affinity before worker launch
            except ImportError as exc:
                raise RuntimeError("Explicit CPU affinity on this platform requires psutil") from exc
    if workers > len(cpus if cpus is not None else available):
        raise ValueError("workers cannot exceed assigned CPUs")
    _set_worker_runtime()
    jobs = prepare_jobs(spec, output)
    rows = []
    context = multiprocessing.get_context("spawn")
    cpu_queue = context.Queue()
    for cpu in (cpus[:workers] if cpus is not None else [None] * workers):
        cpu_queue.put(cpu)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_worker_initializer, initargs=(cpu_queue,)) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(dict(directory=job["directory"], status="FAILED", exception=str(exc), traceback=traceback.format_exc()))
            atomic_json(Path(output) / "summary.json", dict(schema=SCHEMA, expected_jobs=len(jobs), finished_jobs=len(rows),
                        jobs=sorted(rows, key=lambda row: row["directory"]), all_completed=len(rows) == len(jobs) and all(r["status"] == "COMPLETED" for r in rows)))
    cpu_queue.close()
    cpu_queue.join_thread()
    return rows
