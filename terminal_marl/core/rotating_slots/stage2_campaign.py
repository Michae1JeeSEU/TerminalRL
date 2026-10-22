"""Bounded FT-RS-S2-1.0 campaign infrastructure.

The runner alone owns the unrevealed manifest. Checkpoints use the existing
whitelisted, hexadecimal-float StateCodec, under a separate Stage2 identity.
No callable or pickle is loaded from an archive.
"""
from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
from statistics import NormalDist
import subprocess
import sys
import time
import traceback

from .config import load_config, load_service_profile, service_duration
from .demand import ArrivalManifest, ArrivalRequest
from .performance.checkpoint import (StateCodec, WORLD_FIELDS, CACHE_FIELDS,
    OPTIONAL_RUNTIME_FIELDS, RUNTIME_FIELDS, CheckpointError, _capture_sink, _stream_bytes)
from .replay import BufferedStateJSONLSink, ReplayBundle, _plain
from .world import AIRBORNE, RotatingSlotWorld, SchedulerPort, STAGE2_POLICY_ID

METHODS = ("energy_slack", "rh_cp_sat")
SCHEMA = "ft_rs_stage2_campaign_checkpoint_v1"
STAGE2_FIELDS = frozenset("""experiment_policy_id has_energy_losses terminated_invalid
failure_archive failure_resources_released dispatch_timings _pending_dispatch _refinement_time
_monitor_timing_pending""".split())
STAGE2_FIELDS = STAGE2_FIELDS | frozenset(("rescue_rule_profile",))
DERIVED_RESCUE_FIELDS = frozenset(("_rescue_search_cache", "_rescue_monitor_v6"))
SCENARIO_PROFILES = ("legacy_busy_v5", "stress36_rate180_v1", "diagnostic18_rate120_v1")
RESCUE_RULE_PROFILES = ("legacy_v5", "rescue_opportunity_v6")
RUNTIME_EXTRA = frozenset(("admission_callback",))
ROOT = Path(__file__).resolve().parents[3]
CONTRACT_DIR = ROOT / "new10.16/rotating_slots_stage2_baseline_v1_0"


def canonical(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_bytes())


class CampaignBudget:
    """Persisted real campaign epoch, including earlier measured test costs.

    Restarts retain the original epoch and limits. The default is ten hours;
    explicit shorter validation limits may be supplied only at first creation.
    A persisted elapsed lower bound also prevents a clock rollback granting time.
    """
    def __init__(self, path, preparation_wall_seconds=0., clock=time.time, *,
                 total_seconds=None, archive_reserve_seconds=None):
        self.path, self.clock = Path(path), clock
        if self.path.exists():
            self.data = load_json(self.path)
            if preparation_wall_seconds:
                raise ValueError("preparation credit may only be supplied at first creation")
            if total_seconds is not None or archive_reserve_seconds is not None:
                raise ValueError("campaign budget limits may only be supplied at first creation")
        else:
            total=36000. if total_seconds is None else float(total_seconds)
            reserve=1800. if archive_reserve_seconds is None else float(archive_reserve_seconds)
            if not (math.isfinite(total) and math.isfinite(reserve) and 0 < reserve < total):
                raise ValueError("campaign total and archive reserve must be finite and 0 < reserve < total")
            preparation_limit=min(3600.,total-reserve)
            if not 0 <= preparation_wall_seconds <= preparation_limit:
                raise ValueError("preparation credit exceeds preparation/compute allocation")
            self.data = dict(schema="stage2_wall_budget_v1", started_epoch=clock()-preparation_wall_seconds,
                credited_tests_wall_s=preparation_wall_seconds, elapsed_floor_s=preparation_wall_seconds,
                total_s=total, compute_stop_s=total-reserve, preparation_limit_s=preparation_limit)
            self.persist()
        self._start_mono = time.monotonic()
        self._initial_elapsed = max(self.data["elapsed_floor_s"], self.clock()-self.data["started_epoch"])

    @property
    def elapsed(self):
        mono = (time.monotonic()-self._start_mono+self._initial_elapsed
                if hasattr(self, "_start_mono") else self.data["elapsed_floor_s"])
        return max(self.data["elapsed_floor_s"], mono, self.clock()-self.data["started_epoch"])

    @property
    def compute_remaining(self): return max(0., self.data["compute_stop_s"]-self.elapsed)

    def persist(self):
        self.data["elapsed_floor_s"] = self.elapsed
        atomic_json(self.path, self.data)


def frozen_manifest(scenario_profile="legacy_busy_v5"):
    if scenario_profile not in SCENARIO_PROFILES:
        raise ValueError("unknown scenario profile: "+scenario_profile)
    rate = 120. if scenario_profile == "diagnostic18_rate120_v1" else 180.
    source = ArrivalManifest.generate(seed=71, rate_per_hour=rate, arrival_window=3600.)
    # Existing generator has a closed final test; Stage2 explicitly right-opens it.
    return ArrivalManifest(tuple(r for r in source.requests if r.release_time < 3600.), 71, rate, 3600.)


def read_manifest(path):
    value = load_json(path)
    return ArrivalManifest(tuple(ArrivalRequest(**r) for r in value["requests"]),
                           value["seed"], value["rate_per_hour"], value["arrival_window"])


def busy_fixture(manifest=None, *, state_sink=None, config=None,
                 scenario_profile="legacy_busy_v5", rescue_rule_profile="legacy_v5"):
    if scenario_profile not in SCENARIO_PROFILES or rescue_rule_profile not in RESCUE_RULE_PROFILES:
        raise ValueError("unknown scenario/rescue rule profile")
    world = RotatingSlotWorld(config=config, manifest=manifest, state_sink=state_sink,
                              experiment_policy_id=STAGE2_POLICY_ID)
    world.rescue_rule_profile = rescue_rule_profile
    soc_rng, service_rng = random.Random(2071), random.Random(2072)
    normal = NormalDist(.75, .08)
    lower, upper = normal.cdf(.60), normal.cdf(.90)
    counts = (("outer",6), ("middle",6), ("inner",6)) if scenario_profile == "diagnostic18_rate120_v1" else (("outer",14), ("middle",12), ("inner",10))
    for ring, count in counts:
        slots = next(r.slot_count for r in world.config.airspace.rings if r.id == ring)
        for k in range(count):
            soc = normal.inv_cdf(lower + soc_rng.random()*(upper-lower))
            u = service_rng.random()
            service = "short" if u < .2 else "standard" if u < .8 else "long"
            world.seed_hold(f"initial_{ring}_{k:02d}",ring, k*slots//count, soc, service)
    for pad, remaining in ((0,30.), (2,60.), (4,90.)):
        remaining *= service_duration(world.config, "standard") / 120.
        world.seed_service(f"initial_ground_pad{pad}",pad,remaining,"standard",.60)
    return world


def shared_fifo_admission(snapshot, port):
    """Identical method-independent lazy admission; blocked gate heads stay FIFO."""
    blocked = set()
    for request in sorted(port.requests.values(), key=lambda r:(r.release_time,r.id)):
        if request.id in port.aircraft or request.gate_id in blocked:
            continue
        result = port.admit(request.id)
        if not result.accepted and request.id not in port.aircraft:
            blocked.add(request.gate_id)


def initial_references(world, max_wall_seconds=180.):
    from .stage2_planning import ReferenceProvider, PlanningBudget
    snapshot = world.build_planning_snapshot()
    provider = ReferenceProvider()
    deadline = time.monotonic()+max_wall_seconds
    rows = []
    for aircraft in snapshot.aircraft.values():
        if aircraft.status not in AIRBORNE: continue
        left = deadline-time.monotonic()
        started = time.perf_counter()
        if left <= 0:
            rows.append(dict(id=aircraft.id,status="UNKNOWN",reason="PREPARATION_BUDGET",soc=aircraft.soc,
                             peak_remaining_s=120.-aircraft.peak_seconds,wall_s=0.)); continue
        reference = provider.get(snapshot, aircraft, budget=PlanningBudget(soft_seconds=min(5.,left)))
        rows.append(dict(id=aircraft.id,status=reference.status,source=reference.source,soc=aircraft.soc,
            soc_required=reference.soc_required,
            soc_margin=None if reference.soc_required is None else aircraft.soc-.25-reference.soc_required,
            peak_required_s=reference.peak_required,
            peak_margin_s=None if reference.peak_required is None else 120.-aircraft.peak_seconds-reference.peak_required,
            energy_j=reference.energy_j,duration_s=reference.duration,identity=reference.identity,
            trajectories=[p.to_dict() for p in reference.plans],wall_s=time.perf_counter()-started))
    return rows


def environment_lock():
    required = ("numpy","scipy","pytest","PyYAML","matplotlib","ortools")
    versions, missing = {}, []
    for name in required:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: missing.append(name)
    if missing:
        raise RuntimeError("Stage2 environment validation missing: " + ", ".join(missing))
    if versions["ortools"] != "9.15.6755":
        raise RuntimeError("frozen campaign requires ortools==9.15.6755")
    # Import the concrete solver API before freezing dependency versions.
    from ortools.sat.python import cp_model
    model = cp_model.CpModel(); model.AddBoolOr([model.NewBoolVar("environment_probe")])
    solver = cp_model.CpSolver(); solver.parameters.num_search_workers=1
    if solver.Solve(model) != cp_model.OPTIMAL: raise RuntimeError("OR-Tools environment probe failed")
    return dict(python=platform.python_version(), executable=sys.executable, platform=platform.platform(),
        required=versions, installed={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
        numerical_threads=1,cp_workers=1,solver_seed=71)


def source_provenance(service_profile="baseline", *, scenario_profile="legacy_busy_v5", rescue_rule_profile="legacy_v5"):
    """Local-only freeze operation; campaign execution never runs Git."""
    from .stage2_planning import CANDIDATE_POLICY_ID, QUALITY_BUDGET_PROFILE, PlanningBudget
    files = sorted((ROOT/"terminal_marl/core/rotating_slots").glob("*.py"))
    files += [ROOT/"terminal_marl/core/rotating_slots/performance/checkpoint.py"]
    files += [ROOT/"scripts/server/run_rotating_slots_stage2.py",ROOT/"terminal_marl/visualization/rotating_slots_replay.py"]
    source = {str(p.relative_to(ROOT)).replace("\\","/"):sha256(p.read_bytes()).hexdigest()
              for p in files if p.is_file()}
    def git(*args):
        return subprocess.check_output(["git",*args],cwd=ROOT,stderr=subprocess.DEVNULL).decode().strip()
    if scenario_profile not in SCENARIO_PROFILES or rescue_rule_profile not in RESCUE_RULE_PROFILES:
        raise ValueError("unknown scenario/rescue rule profile")
    config=load_service_profile(service_profile)
    original=load_service_profile("baseline")
    return dict(revision=git("rev-parse","HEAD"),workspace_status_short=git("status","--short"),
        tracked_diff=git("diff","--binary","HEAD","--",
        "terminal_marl/core/rotating_slots","terminal_marl/visualization/rotating_slots_replay.py",
        "scripts/server/run_rotating_slots_stage2.py"),source_files=source,source_hash=digest(source),
        contract_sha256=sha256((CONTRACT_DIR/"rotating_slots_stage2_baseline_contract_v1_0.md").read_bytes()).hexdigest(),
        overlay_sha256=sha256((CONTRACT_DIR/"stage2_baseline_config_v1_0.yaml").read_bytes()).hexdigest(),
        base_config_hash=config.config_hash,base_config_bytes_sha256=config.raw_hash,
        service_profile=config.service_profile,scenario_profile=scenario_profile,rescue_rule_profile=rescue_rule_profile,
        effective_config_path=Path(config.path).relative_to(ROOT).as_posix(),
        original_config_hash=original.config_hash,original_config_bytes_sha256=original.raw_hash,
        numeric_backend_hash=digest({k:v for k,v in source.items() if k.endswith(("energy.py","safety.py","trajectory.py"))}),
        candidate_policy_id=CANDIDATE_POLICY_ID,planning_budget_profile=QUALITY_BUDGET_PROFILE,
        planning_budget=PlanningBudget.for_profile(QUALITY_BUDGET_PROFILE).settings(),
        emergency_policy_id="rescue_opportunity_v6" if rescue_rule_profile=="rescue_opportunity_v6" else "energy_urgency_yield_v5",
        objective_id="reliability_throughput_progress_tail_v2")


def freeze_provenance(path, service_profile="baseline", **profiles):
    value=source_provenance(service_profile, **profiles)
    atomic_json(path,value)
    return value


def verify_source_provenance(provenance):
    """Verify supplied local fingerprint by bytes only, including on the server."""
    for relative,expected in provenance["source_files"].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError("source differs from local frozen provenance: "+relative)
    if digest(provenance["source_files"]) != provenance["source_hash"]:
        raise RuntimeError("source provenance aggregate mismatch")
    config=load_service_profile(provenance.get("service_profile","baseline"))
    original=load_service_profile("baseline")
    if (provenance["base_config_hash"] != config.config_hash
        or provenance["base_config_bytes_sha256"] != config.raw_hash
        or provenance.get("effective_config_path",Path(config.path).relative_to(ROOT).as_posix()) != Path(config.path).relative_to(ROOT).as_posix()
        or provenance.get("original_config_hash",original.config_hash) != original.config_hash
        or provenance.get("original_config_bytes_sha256",original.raw_hash) != original.raw_hash
        or provenance["contract_sha256"] != sha256((CONTRACT_DIR/"rotating_slots_stage2_baseline_contract_v1_0.md").read_bytes()).hexdigest()
        or provenance["overlay_sha256"] != sha256((CONTRACT_DIR/"stage2_baseline_config_v1_0.yaml").read_bytes()).hexdigest()):
        raise RuntimeError("configuration/contract differs from frozen provenance")


class Stage2Codec(StateCodec):
    """Adds only the named Stage2 prediction dataclasses to the existing codec."""
    def __init__(self):
        super().__init__()
        from . import stage2_planning
        for name in ("NormalReference","PlanningNode","PredictionColumn"):
            self.classes["stage2_planning."+name] = getattr(stage2_planning,name)


def checkpoint_identity(provenance, manifest_hash):
    identity=dict(policy_id=STAGE2_POLICY_ID,source_hash=provenance["source_hash"],
        base_config_hash=provenance["base_config_hash"],overlay_sha256=provenance["overlay_sha256"],
        manifest_sha256=manifest_hash)
    if "service_profile" in provenance: identity["service_profile"]=provenance["service_profile"]
    for name in ("scenario_profile", "rescue_rule_profile"):
        if name in provenance: identity[name]=provenance[name]
    return identity


def save_checkpoint(path, world, scheduler, identity, *, boundary_summary=None):
    expected = WORLD_FIELDS|CACHE_FIELDS|OPTIONAL_RUNTIME_FIELDS|STAGE2_FIELDS|RUNTIME_FIELDS|RUNTIME_EXTRA
    if set(vars(world))-DERIVED_RESCUE_FIELDS != expected:
        raise CheckpointError("Stage2 world field inventory mismatch: "+str((set(vars(world))-DERIVED_RESCUE_FIELDS)^expected))
    if world.rescue_rule_profile != identity.get("rescue_rule_profile","legacy_v5"):
        raise CheckpointError("Stage2 checkpoint rescue rule profile mismatch")
    if (world.config.config_hash != identity["base_config_hash"]
        or world.config.service_profile != identity.get("service_profile",world.config.service_profile)):
        raise CheckpointError("Stage2 checkpoint configuration identity mismatch")
    codec = Stage2Codec()
    state = {k:v for k,v in vars(world).items() if k not in RUNTIME_FIELDS|RUNTIME_EXTRA|DERIVED_RESCUE_FIELDS}
    payload = dict(schema=SCHEMA,identity=identity,time=world.time,
        world=codec.encode(state),scheduler=codec.encode(scheduler.state_dict()),
        sink=None if world.state_sink is None else _capture_sink(world.state_sink,codec),
        boundary_summary=boundary_summary)
    envelope = dict(payload=payload,sha256=digest(payload))
    atomic_json(path,envelope)
    return envelope


def restore_checkpoint(path, scheduler, identity, *, stream_copy=None):
    envelope = load_json(path)
    payload = envelope["payload"]
    if envelope["sha256"] != digest(payload) or payload["schema"] != SCHEMA or payload["identity"] != identity:
        raise CheckpointError("Stage2 checkpoint identity/content mismatch")
    codec = Stage2Codec()
    state = codec.decode(payload["world"])
    # Archives created before v6 had the implicit legacy rescue rule.
    state.setdefault("rescue_rule_profile","legacy_v5")
    expected = WORLD_FIELDS|CACHE_FIELDS|OPTIONAL_RUNTIME_FIELDS|STAGE2_FIELDS
    if set(state) != expected or state["experiment_policy_id"] != STAGE2_POLICY_ID:
        raise CheckpointError("Stage2 checkpoint world inventory/policy mismatch")
    if state["rescue_rule_profile"] != identity.get("rescue_rule_profile","legacy_v5"):
        raise CheckpointError("Stage2 checkpoint rescue rule profile mismatch")
    if (state["config"].config_hash != identity["base_config_hash"] or state["time"] != payload["time"]
        or state["config"].service_profile != identity.get("service_profile",state["config"].service_profile)):
        raise CheckpointError("Stage2 checkpoint configuration/time mismatch")
    if digest(state["manifest"].to_dict()) != identity["manifest_sha256"]:
        raise CheckpointError("Stage2 checkpoint manifest mismatch")
    world = object.__new__(RotatingSlotWorld); world.__dict__.update(state)
    world.state_sink = None
    if payload["sink"] is not None:
        metadata = payload["sink"]
        pending, timestamp = _stream_bytes(metadata,codec,world.records)
        sink = object.__new__(BufferedStateJSONLSink)
        sink.path=Path(metadata["path"])
        if stream_copy is not None:
            target=Path(stream_copy)
            with sink.path.open("rb") as src, target.open("wb") as dst:
                left=metadata["offset"]
                while left:
                    block=src.read(min(left,1024*1024))
                    if not block: raise CheckpointError("truncated stream during export copy")
                    dst.write(block);left-=len(block)
                if pending: dst.write(pending)
            sink.path=target
        sink._stream=sink.path.open("r+b")
        sink._pending=pending; sink._time=timestamp; sink._offset=metadata["offset"]
        sink._committed_count=metadata["committed_count"]
        sink._stream.seek(sink._offset)
        if pending: sink._stream.write(pending)
        sink._stream.truncate(); sink._stream.flush()
        world.state_sink=sink
    world._scheduler_port=SchedulerPort(world)
    world.admission_callback=shared_fifo_admission
    world.decision_callback=None
    scheduler.load_state_dict(codec.decode(payload["scheduler"]))
    return world


def episode_metrics(world):
    """Window counts from actual touchdown timestamps, with initial ground excluded."""
    metrics = dict(world.metrics())
    return metrics | {"K_episode":list(episode_rank(metrics)),"human_review":"HUMAN_PENDING"}


def episode_rank(metrics):
    return (metrics.get("F_energy_total",metrics.get("energy_failures",0)),
        -metrics.get("L_success",metrics.get("physically_landed",0)),
        metrics.get("cost_emergency",0)+metrics.get("cost_reroute",0),metrics.get("energy_j",0),
        metrics.get("airborne_time",0)+metrics.get("upstream_wait_time",0))


def quantiles(rows, field):
    values=sorted(float(r[field]) for r in rows if r.get(field) is not None)
    if not values: return {"count":0,"p50":None,"p95":None,"p99":None,"max":None,"mean":None,"over_5s":0}
    def percentile(q):
        x=(len(values)-1)*q; lo=math.floor(x); hi=math.ceil(x)
        return values[lo]+(values[hi]-values[lo])*(x-lo)
    return dict(count=len(values),p50=percentile(.5),p95=percentile(.95),p99=percentile(.99),
        max=max(values),mean=sum(values)/len(values),over_5s=sum(v>5 for v in values),
        over_5s_fraction=sum(v>5 for v in values)/len(values),raw_s=values,
        small_sample_notice="descriptive quantiles; unstable for small N")


def pad_opportunity_samples(world):
    """Direct pre-planning observations; future reservations are not ground service."""
    offered=any(a.status.startswith("HOLD_") for a in world.aircraft.values())
    rows=[]
    for pad in range(world.config.pads.count):
        ground=any(a.pad_id==pad and a.status=="LANDED_SERVICE" for a in world.aircraft.values())
        approach=any(p.pad_id==pad for p in world.active.values())
        future=any(r.resource==pad and r.start>world.time+1.e-8 for r in world.pads.reservations)
        rows.append(dict(time=world.time,pad_id=pad,grounded_service=ground,ground_idle=not ground,
            approach_reserved=approach,idle_without_active_approach=not (ground or approach),
            future_calendar=future,emergency_protected=False,offered=offered,
            reason="GROUNDED_SERVICE" if ground else "ACTIVE_APPROACH_RESERVED" if approach else
                "FUTURE_CALENDAR" if future else "UNKNOWN"))
    return rows


def annotate_pad_opportunities(pads, row, protected=()):
    """Attach this call's planning evidence to its pre-planning pad observations."""
    evidence=row.get("pad_opportunities",{})
    choices=row.get("aircraft_choices",())
    known=bool(choices) and all("candidate_pad_ids" in c for c in choices)
    candidate_pads={p for c in choices for p in c.get("candidate_pad_ids",())}
    selected_pads={p for c in choices for p in c.get("selected_pad_ids",())}
    for pad in pads:
        pid=pad["pad_id"]
        observed=evidence.get(str(pid),{}) if isinstance(evidence,dict) else {}
        pad["planning_evidence"]=observed
        pad["emergency_protected"]=pid in protected
        if pad["reason"]!="UNKNOWN" or not pad["offered"]: continue
        if pid in protected: pad["reason"]="EMERGENCY_PROTECTED"
        elif observed.get("submitted_count",0): pad["reason"]="PLANNING_SUBMITTED"
        elif observed.get("candidate_count",0): pad["reason"]="PLANNING_NOT_SUBMITTED"
        elif observed.get("explicit_conflict_count",0): pad["reason"]="EXPLICIT_TRAJECTORY_CONFLICT_OBSERVED"
        elif not known: pad["reason"]="UNKNOWN"
        elif pid not in candidate_pads: pad["reason"]="CANDIDATE_NOT_FOUND"
        elif pid not in selected_pads or not row.get("submitted",0): pad["reason"]="PLANNING_NOT_SUBMITTED"
        else: pad["reason"]="PLANNING_SUBMITTED"
    row["pad_opportunities"]=pads
    return pads


def rescue_observations(world, normal_rows, dispatch):
    """Descriptive v6 table with explicit denominators and censored rescue waits."""
    aircraft=[]; events={}
    for event in world.events:
        events.setdefault(event.get("aircraft_id"),[]).append(event)
    # Identical to the existing _stage2_metrics initial-ground cohort definition.
    initial_ground={a.id for a in world.aircraft.values()
                    if a.fixture and a.touchdown_time==0. and a.airborne_time==0.}
    for a in world.aircraft.values():
        if a.id in initial_ground: continue
        emergency=a.emergency; own=events.get(a.id,())
        first=lambda kind: next((e["time"] for e in own if e["type"]==kind),None)
        warning=first("rescue_warning"); triggered=first("emergency_triggered")
        started=emergency.start_time
        wait=None if triggered is None else max(0.,(world.time if started is None else started)-triggered)
        aircraft.append(dict(aircraft_id=a.id,status=a.status,warning_time=warning,
            declaration_time=emergency.declaration_time,rescue_trigger_time=triggered,rescue_start_time=started,
            touchdown_time=a.touchdown_time,emergency_charged=emergency.charged,
            low_reserve_touchdown=a.touchdown_time is not None and a.soc<.25,
            rescue_wait_s=wait,rescue_wait_status=None if wait is None else "CENSORED" if started is None else "COMPLETED",
            soc=a.soc,soc_reserve_margin=a.soc-.25,peak_remaining_s=120.-a.peak_seconds,
            normal_margin=getattr(emergency,"normal_margin",None),normal_status=getattr(emergency,"normal_status","UNKNOWN")))
    choices=[choice for row in normal_rows for choice in row.get("aircraft_choices",())]
    coverage=[]
    for aid in sorted({c["aircraft_id"] for c in choices}):
        own=[c for c in choices if c["aircraft_id"]==aid]
        coverage.append(dict(aircraft_id=aid,planning_observations=len(own),
            with_move_candidate=sum(c.get("move_candidates",0)>0 for c in own),
            with_immediate_move_candidate=sum(c.get("immediate_move_candidates",0)>0 for c in own),
            selected_move=sum(c.get("selected_kind")=="MOVE" for c in own),
            unknown_reference=sum(c.get("source_reference_status")!="KNOWN" for c in own)))
    timings=[]
    for row in normal_rows:
        values=row.get("timings",{})
        wall=lambda name: values[name][0] if name in values else None
        total=lambda names: sum(values[name][0] for name in names if name in values) if any(name in values for name in names) else None
        timings.append(dict(candidate_exclusive_wall_s=total(("candidate_generation_and_single_vehicle",
                "safe_progress_generation","reference_seed_generation","geometry_template_screen")),
            safety_validation_wall_s=total(("trajectory_validation","prefix_validation")),
            selected_validation_overhead_wall_s=wall("selected_validation"),solver_wall_s=row.get("solver_wall_s")))
    pads=[pad for row in normal_rows for pad in row.get("pad_opportunities",())]
    completed=[r for r in aircraft if r["rescue_wait_status"]=="COMPLETED"]
    censored=[r for r in aircraft if r["rescue_wait_status"]=="CENSORED"]
    return dict(observation_status="OBSERVED" if world.rescue_rule_profile=="rescue_opportunity_v6" else "LEGACY_FIELDS_MAY_BE_UNKNOWN",
        denominators=dict(dispatch_calls=len(dispatch),normal_calls=len(normal_rows),aircraft_planning_observations=len(choices),
            pad_dispatch_samples=len(pads),ground_idle_offered_pad_samples=sum(p["ground_idle"] and p["offered"] for p in pads),
            idle_without_active_approach_offered_pad_samples=sum(p["idle_without_active_approach"] and p["offered"] for p in pads),
            approach_reserved_pad_samples=sum(p["approach_reserved"] for p in pads)),
        initial_ground_excluded=len(initial_ground),
        warning_aircraft=sum(r["warning_time"] is not None for r in aircraft),
        declared_aircraft=sum(r["declaration_time"] is not None for r in aircraft),
        rescue_started_aircraft=sum(r["rescue_start_time"] is not None for r in aircraft),
        low_reserve_touchdowns=sum(r["low_reserve_touchdown"] for r in aircraft),
        rescue_wait_completed=quantiles(completed,"rescue_wait_s"),rescue_wait_censored=quantiles(censored,"rescue_wait_s"),
        candidate_exclusive_timing=quantiles(timings,"candidate_exclusive_wall_s"),
        safety_validation_timing=quantiles(timings,"safety_validation_wall_s"),
        selected_validation_overhead_timing=quantiles(timings,"selected_validation_overhead_wall_s"),
        solver_timing=quantiles(timings,"solver_wall_s"),
        emergency_protected_pad_samples=sum(p.get("emergency_protected",False) for p in pads),
        future_calendar_emergency_protected_pad_samples=sum(p.get("emergency_protected",False) and p["future_calendar"] for p in pads),
        pad_reason_counts={reason:sum(p["reason"]==reason for p in pads) for reason in sorted({p["reason"] for p in pads})},
        aircraft=aircraft,candidate_coverage=coverage,
        interpretation="CANDIDATE_NOT_FOUND and UNKNOWN provide no physical infeasibility certificate")


def common_boundary(root):
    sets=[]
    for method in METHODS:
        sets.append({int(p.stem.split("_")[-1]) for p in (Path(root)/method).glob("summary_*.json")})
    return max(set.intersection(*sets),default=0)


def rescue_comparison_tables(root):
    """Read-only common-boundary tables, including explicit legacy unknowns."""
    root=Path(root);boundary=common_boundary(root);campaign=load_json(root/"campaign.json")
    rows=[]
    for method in METHODS:
        if not boundary: continue
        summary=load_json(root/method/f"summary_{boundary:04d}.json")
        metrics=summary["metrics"];observed=summary.get("rescue_observations",{})
        row=dict(scenario_profile=campaign.get("scenario_profile","legacy_busy_v5"),method=method,time_s=boundary,
            normal=metrics["L_normal"],emergency=metrics["L_emergency"],total=metrics["L_success"],
            F=metrics["F_energy_total"],airborne_pending=metrics["N_airborne_pending"],upstream_pending=metrics["N_upstream_pending"])
        for field in ("warning_aircraft","declared_aircraft","rescue_started_aircraft","low_reserve_touchdowns",
                      "denominators","rescue_wait_completed","rescue_wait_censored","candidate_exclusive_timing",
                      "safety_validation_timing","selected_validation_overhead_timing","solver_timing",
                      "emergency_protected_pad_samples","future_calendar_emergency_protected_pad_samples",
                      "pad_reason_counts","candidate_coverage","aircraft"):
            row[field]=observed.get(field,"UNKNOWN")
        for field in ("normal_timing","dispatch_timing"):
            row[field]=summary.get(field,"UNKNOWN")
        rows.append(row)
    return dict(schema="stage2_rescue_v6_observational_tables_v1",common_time_s=boundary,
        rescue_rule_profile=campaign.get("rescue_rule_profile","legacy_v5"),rows=rows,
        interpretation="Single fixed seed, bounded common window; missing v6 evidence is UNKNOWN. No candidate is not proof of physical infeasibility.")


def report_campaign(root, *, plots=True):
    root=Path(root); boundary=common_boundary(root)
    methods={m:load_json(root/m/f"summary_{boundary:04d}.json") for m in METHODS} if boundary else {}
    result=dict(comparison_time_s=boundary,target_observation_s=3600,human_review="HUMAN_PENDING",
        interpretation="one seed, synthetic initial backlog; partial windows are not one-hour capacity",
        methods=methods,ranking=sorted(methods,key=lambda m:tuple(methods[m]["metrics"]["K_episode"])))
    if (root/"campaign.json").exists(): result["rescue_tables"]=rescue_comparison_tables(root)
    atomic_json(root/"comparison.json",result)
    if methods:
        fields=("method","T_common_s","failures","touchdowns","normal","emergency","reroutes",
            "airborne_pending","upstream_pending","terminal_energy_J","airborne_plus_upstream_s",
            "normal_p50_s","normal_p95_s","normal_max_s","dispatch_p95_s","fallback_fraction",
            "cp_status_counts","episode_compute_wall_s")
        with (root/"comparison.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for method,summary in methods.items():
                metrics=summary["metrics"];normal=summary["normal_timing"]
                writer.writerow(dict(method=method,T_common_s=boundary,failures=metrics["F_energy_total"],
                    touchdowns=metrics["L_success"],normal=metrics["L_normal"],emergency=metrics["L_emergency"],
                    reroutes=metrics["N_rerouted"],airborne_pending=metrics["N_airborne_pending"],
                    upstream_pending=metrics["N_upstream_pending"],terminal_energy_J=metrics["energy_j"],
                    airborne_plus_upstream_s=metrics["K_episode"][-1],normal_p50_s=normal["p50"],
                    normal_p95_s=normal["p95"],normal_max_s=normal["max"],dispatch_p95_s=summary["dispatch_timing"]["p95"],
                    fallback_fraction=summary["fallback_calls"]/max(1,normal["count"]),
                    cp_status_counts=json.dumps(summary["cp_status_counts"],sort_keys=True),
                    episode_compute_wall_s=summary["episode_compute_wall_s"]))
    if plots and boundary:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Five requested physical-result curves, all truncated to the same window.
        fig,axes=plt.subplots(5,1,figsize=(8,13),sharex=True)
        for method in METHODS:
            summaries=[load_json(p) for p in sorted((root/method).glob("summary_*.json"))
                       if int(p.stem.split("_")[-1]) <= boundary]
            times=[s["time"] for s in summaries]
            for field,style in (("L_success","-"),("F_energy_total","--")):
                axes[0].plot(times,[s["metrics"][field] for s in summaries],style,label=method+" "+field)
            for field,style in (("N_airborne_pending","-"),("N_upstream_pending","--")):
                axes[1].plot(times,[s["metrics"][field] for s in summaries],style,label=method+" "+field)
            axes[2].plot(times,[s["pad_occupied_count"] for s in summaries],label=method)
            axes[3].plot(times,[s["metrics"]["energy_j"] for s in summaries],label=method)
            decisions=root/method/"dispatch_timings.jsonl"
            if decisions.exists():
                rows=[json.loads(line) for line in decisions.read_text().splitlines() if line]
                rows=[r for r in rows if r["time"]<=boundary]
                axes[4].plot([r["time"] for r in rows],[r["normal_wall_s"] for r in rows],label=method)
        for axis,label in zip(axes,("Touchdowns / losses","Airborne / upstream","Pads occupied","Terminal energy (J)","Normal planning wall (s)")):
            axis.set_ylabel(label);axis.grid(alpha=.25);axis.legend(fontsize=6)
        axes[-1].set_xlabel("Shared completed simulation time (s)")
        fig.tight_layout();fig.savefig(root/"comparison_curves.png",dpi=160);plt.close(fig)
    return result


def physical_cores():
    """Linux topology within allowed affinity; unknown topology uses serial fallback."""
    if not hasattr(os,"sched_getaffinity"): return []
    groups={}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology=Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try: key=((topology/"physical_package_id").read_text().strip(),(topology/"core_id").read_text().strip())
        except OSError: return []
        groups.setdefault(key,cpu)
    return list(groups.values())


def prepare_campaign(root, preparation_wall_seconds=0., *, provenance_path=None, service_profile="baseline",
                     campaign_wall_budget_seconds=None, archive_reserve_seconds=None,
                     scenario_profile="legacy_busy_v5", rescue_rule_profile="legacy_v5"):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/"campaign.json").exists(): raise ValueError("campaign already prepared; use resume")
    budget=CampaignBudget(root/"budget.json",preparation_wall_seconds,total_seconds=campaign_wall_budget_seconds,
                          archive_reserve_seconds=archive_reserve_seconds)
    lock=environment_lock()
    if (root/"environment.lock.json").exists() and load_json(root/"environment.lock.json") != lock:
        raise RuntimeError("resolved environment changed after first successful validation")
    atomic_json(root/"environment.lock.json",lock)
    manifest=frozen_manifest() if scenario_profile=="legacy_busy_v5" else frozen_manifest(scenario_profile)
    atomic_json(root/"manifest.json",manifest.to_dict())
    provenance_file=Path(provenance_path) if provenance_path is not None else root/"provenance.json"
    if not provenance_file.is_file():
        raise ValueError("prepare requires locally frozen provenance JSON; use local freeze-provenance first")
    provenance=load_json(provenance_file);verify_source_provenance(provenance)
    config=load_service_profile(service_profile)
    if (provenance.get("service_profile","baseline") != service_profile
        or provenance["base_config_hash"] != config.config_hash):
        raise ValueError("requested service profile differs from frozen provenance")
    for name,value,default in (("scenario_profile",scenario_profile,"legacy_busy_v5"),
                               ("rescue_rule_profile",rescue_rule_profile,"legacy_v5")):
        if provenance.get(name,default)!=value:
            raise ValueError("requested "+name+" differs from frozen provenance")
    atomic_json(root/"provenance.json",provenance)
    (root/"config.yaml").write_bytes(Path(config.path).read_bytes())
    config=load_config(root/"config.yaml")
    profiles={}
    if scenario_profile!="legacy_busy_v5": profiles["scenario_profile"]=scenario_profile
    if rescue_rule_profile!="legacy_v5": profiles["rescue_rule_profile"]=rescue_rule_profile
    world=busy_fixture(manifest,config=config,**profiles)
    atomic_json(root/"fixture.json",dict(aircraft=[a.to_dict() for a in world.aircraft.values()],
        slots=world.slots.to_dict(),pads=world.pads.to_dict(),synthetic_fixture=True,
        geometry_validation="seed_hold/seed_service continuous certification",metrics=episode_metrics(world)))
    references=initial_references(world,max(0.,min(180.,budget.data["preparation_limit_s"]-budget.elapsed)))
    atomic_json(root/"initial_references.json",references)
    budget.persist()
    if budget.elapsed > budget.data["preparation_limit_s"]: raise RuntimeError("preparation exceeded campaign allocation")
    campaign=dict(schema="stage2_first_campaign_v1",identity=checkpoint_identity(provenance,digest(manifest.to_dict())),
        service_profile=service_profile,scenario_profile=scenario_profile,rescue_rule_profile=rescue_rule_profile,
        config_path="config.yaml",config_bytes_sha256=config.raw_hash,
        preparation_wall_s=budget.elapsed,environment_lock_sha256=digest(lock),
        fixture_sha256=sha256((root/"fixture.json").read_bytes()).hexdigest(),
        status="PREPARED",human_review="HUMAN_PENDING",target_s=3600.)
    atomic_json(root/"campaign.json",campaign)
    return campaign


def campaign_config(root, campaign):
    """Use materialized experiment bytes, with legacy baseline archive support."""
    path=Path(root)/campaign.get("config_path","config.yaml")
    if "config_path" in campaign:
        if not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("campaign configuration path escapes campaign")
        config=load_config(path)
    else:
        config=load_service_profile(campaign.get("service_profile","baseline"))
    if (config.config_hash != campaign["identity"]["base_config_hash"]
        or config.raw_hash != campaign.get("config_bytes_sha256",config.raw_hash)
        or config.service_profile != campaign.get("service_profile","baseline")
        or config.service_profile != campaign["identity"].get("service_profile",config.service_profile)):
        raise ValueError("campaign configuration identity/content mismatch")
    for name,default,allowed in (("scenario_profile","legacy_busy_v5",SCENARIO_PROFILES),
                                 ("rescue_rule_profile","legacy_v5",RESCUE_RULE_PROFILES)):
        value=campaign.get(name,default)
        if value not in allowed or value!=campaign["identity"].get(name,default):
            raise ValueError("campaign "+name+" identity mismatch")
    return config


def run_episode_block(root, method, target, cpu=None, connection=None):
    """One bounded continuation; no initial reset after the first 60 seconds."""
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]="1"
    if cpu is not None:
        os.sched_setaffinity(0,{cpu})
    from .stage2_planning import NormalSchedulerCallback
    root=Path(root);directory=root/method;directory.mkdir(exist_ok=True)
    campaign=load_json(root/"campaign.json");identity=campaign["identity"]
    verify_source_provenance(load_json(root/"provenance.json"))
    config=campaign_config(root,campaign)
    budget=CampaignBudget(root/"budget.json")
    if budget.compute_remaining <= 0: return
    startup=time.perf_counter()
    scheduler=NormalSchedulerCallback(method)
    latest=directory/"checkpoint.json"
    numbered=sorted(directory.glob("checkpoint_[0-9]*.json"))
    if numbered or latest.exists(): world=restore_checkpoint(numbered[-1] if numbered else latest,scheduler,identity)
    else:
        sink=BufferedStateJSONLSink(directory/"states.jsonl")
        fixture_kwargs=dict(state_sink=sink)
        if "service_profile" in campaign: fixture_kwargs["config"]=config
        for name,default in (("scenario_profile","legacy_busy_v5"),("rescue_rule_profile","legacy_v5")):
            if campaign.get(name,default)!=default: fixture_kwargs[name]=campaign[name]
        world=busy_fixture(read_manifest(root/"manifest.json"),**fixture_kwargs)
        if "fixture_sha256" in campaign:
            frozen=load_json(root/"fixture.json")
            if sha256((root/"fixture.json").read_bytes()).hexdigest()!=campaign["fixture_sha256"]:
                raise RuntimeError("materialized fixture content changed")
            actual=dict(aircraft=[a.to_dict() for a in world.aircraft.values()],
                        slots=world.slots.to_dict(),pads=world.pads.to_dict())
            if any(canonical(actual[key])!=canonical(frozen[key]) for key in actual):
                raise RuntimeError("method initial state differs from materialized fixture")
        world.admission_callback=shared_fifo_admission
        save_checkpoint(latest,world,scheduler,identity)
    startup_wall=time.perf_counter()-startup
    def normal(snapshot,port):
        pads=pad_opportunity_samples(world)
        protected={p.pad_id for p in (*world.active.values(),*world.soft.values()) if p.kind=="FINAL_EMERGENCY"}
        result=scheduler(port)
        if scheduler.records:
            row=scheduler.records[-1]
            annotate_pad_opportunities(pads,row,protected)
        return result
    try:
        while True:
            block_target=target if connection is None else connection.recv()
            if block_target is None: break
            if budget.compute_remaining<=0:
                if connection: connection.send(("budget",world.time))
                break
            if block_target <= world.time:
                marker=directory/f"summary_{int(block_target):04d}.json"
                if not marker.exists():
                    saved=load_json(directory/f"checkpoint_{int(block_target):04d}.json")["payload"].get("boundary_summary")
                    if saved is None: raise CheckpointError("completed checkpoint has no recoverable boundary summary")
                    saved["checkpoint_wall_s"]=None
                    saved["checkpoint_timing_status"]="UNKNOWN_AFTER_INTERRUPTED_PUBLICATION"
                    for filename,rows in (("decisions.jsonl",scheduler.records),("dispatch_timings.jsonl",world.dispatch_timings)):
                        tmp=directory/(filename+".partial");tmp.write_bytes(b"".join(canonical(r)+b"\n" for r in rows));os.replace(tmp,directory/filename)
                    atomic_json(marker,saved)
                if connection: connection.send(("completed",world.time))
                if connection is None: break
                continue
            start=time.perf_counter();cpu_start=time.process_time()
            dispatch_before=sum(r["dispatch_wall_s"] for r in world.dispatch_timings)
            world.advance(float(block_target),normal,settle_only_at_end=True)
            advance_wall=time.perf_counter()-start
            if world.failed or getattr(world,"terminated_invalid",False):
                atomic_json(directory/"invalid.json",dict(time=world.time,metrics=episode_metrics(world)))
                if connection: connection.send(("invalid",world.time))
                break
            if abs(world.time-block_target)>1e-7: raise RuntimeError("incomplete block without invalid status")
            # Complete checkpoint and logs before publishing the summary marker.
            normal_rows=scheduler.records
            dispatch=world.dispatch_timings
            nonempty_times={r["time"] for r in normal_rows if r.get("aircraft",0)>0}
            summary=dict(time=world.time,method=method,metrics=episode_metrics(world),
                block_wall_s=time.perf_counter()-start,block_cpu_s=time.process_time()-cpu_start,
                process_startup_restore_wall_s=startup_wall,checkpoint_wall_s=None,
                advance_wall_s=advance_wall,
                executor_non_dispatch_wall_s=max(0.,advance_wall-(sum(r["dispatch_wall_s"] for r in dispatch)-dispatch_before)),
                executor_internal_wall_counters=dict(world.cpu),
                episode_compute_wall_s=sum(load_json(p)["block_wall_s"] for p in directory.glob("summary_*.json"))
                    +time.perf_counter()-start,
                dispatch_wall_sum_s=sum(r["dispatch_wall_s"] for r in dispatch),
                normal_timing=quantiles(dispatch,"normal_wall_s"),
                inner_algorithm_timing=quantiles(normal_rows,"wall_s"),dispatch_timing=quantiles(dispatch,"dispatch_wall_s"),
                nonempty_normal_timing=quantiles([r for r in dispatch if r["time"] in nonempty_times],"normal_wall_s"),
                noop_calls=sum(r.get("submitted",0)==0 for r in normal_rows),
                fallback_calls=sum(bool(r.get("fallback",False)) for r in normal_rows),
                cp_status_counts={status:sum(s.get("status")==status for r in normal_rows for s in r.get("solver",()))
                    for status in sorted({s.get("status") for r in normal_rows for s in r.get("solver",())})},
                rescue_observations=rescue_observations(world,normal_rows,dispatch),
                pad_occupied_count=sum(any(a.pad_id==pad and a.status=="LANDED_SERVICE" for a in world.aircraft.values()) for pad in range(6)),
                dispatch_wall_per_sim_second=sum(r["dispatch_wall_s"] for r in dispatch)/world.time,
                event_frequency_hz=len(world.events)/world.time,
                post_failure_model="abstract_isolation_no_cascade",cpu_affinity=cpu)
            checkpoint_started=time.perf_counter()
            save_checkpoint(directory/f"checkpoint_{int(block_target):04d}.json",world,scheduler,identity,boundary_summary=summary)
            save_checkpoint(latest,world,scheduler,identity,boundary_summary=summary)
            summary["checkpoint_wall_s"]=time.perf_counter()-checkpoint_started
            summary["block_wall_s"]=time.perf_counter()-start
            for filename,rows in (("decisions.jsonl",normal_rows),("dispatch_timings.jsonl",dispatch)):
                tmp=directory/(filename+".partial");tmp.write_bytes(b"".join(canonical(r)+b"\n" for r in rows));os.replace(tmp,directory/filename)
            atomic_json(directory/f"summary_{int(block_target):04d}.json",summary)
            if connection is None: break
            connection.send(("completed",world.time))
    finally:
        if world.state_sink is not None: world.state_sink.close()


def _run_episode_guard(root,method,target,cpu,connection=None):
    try: run_episode_block(root,method,target,cpu,connection)
    except Exception:
        atomic_json(Path(root)/method/"error.json",dict(target=target,error=traceback.format_exc()))
        raise


def _archive_episode(root,method,boundary):
    started=time.perf_counter()
    path=export_episode(root,method,boundary)
    export_wall=time.perf_counter()-started
    from .audit import audit_replay
    audit_started=time.perf_counter()
    report=audit_replay(path,Path(root)/method/f"audit_{boundary:04d}.json")
    atomic_json(Path(root)/method/"archive_status.json",dict(export_wall_s=export_wall,
        audit_wall_s=time.perf_counter()-audit_started,audit_status=report["status"],replay=path))


def resource_observation():
    observation=dict(logical_cpu_count=os.cpu_count(),physical_core_bindings=physical_cores(),
        processor=platform.processor(),platform=platform.platform(),load_average=None,memory=None,cpuinfo=None)
    if hasattr(os,"getloadavg"): observation["load_average"]=os.getloadavg()
    for name,path in (("memory","/proc/meminfo"),("cpuinfo","/proc/cpuinfo")):
        if Path(path).exists(): observation[name]=Path(path).read_text()
    return observation


def export_episode(root, method, boundary):
    from .stage2_planning import NormalSchedulerCallback
    root=Path(root);directory=root/method
    config=campaign_config(root,load_json(root/"campaign.json"))
    archive=directory/f"replay_{boundary:04d}"
    if archive.exists(): return str(archive)
    scheduler=NormalSchedulerCallback(method)
    world=restore_checkpoint(directory/f"checkpoint_{boundary:04d}.json",scheduler,
        load_json(root/"campaign.json")["identity"],stream_copy=directory/f"export_states_{boundary:04d}.jsonl")
    try:
        record=world.to_dict()
        record["experiment_policy_id"]=STAGE2_POLICY_ID
        record["decisions"]=scheduler.records
        record["dispatch_timings"]=world.dispatch_timings
        record["failure_archive"]=world.failure_archive
        record["tentative_predictions"]=[dict(aircraft_id=aid,
            as_of=scheduler.records[-1]["time"] if scheduler.records else 0.,plans=[p.to_dict() for p in plans])
            for aid,plans in scheduler.provider.previous.items()]
        record["stage2_provenance"]=load_json(root/"provenance.json")
        ReplayBundle.write(archive,record=record,code_commit=record["stage2_provenance"]["revision"],
            config_path=config.path,command="run_rotating_slots_stage2.py",seed=71,eager=False)
    finally:
        if world.state_sink is not None: world.state_sink.close()
    return str(archive)


def run_campaign(root, *, target=3600., export=True, cpus=None):
    """Supervisor starts two single-core children or alternates shared blocks.

    Only complete 60-second checkpoints are compared. At the compute deadline,
    children terminate and the last completed checkpoints remain authoritative.
    """
    root=Path(root);budget=CampaignBudget(root/"budget.json")
    if target <= 0 or target > 3600 or target % 60: raise ValueError("target must be a 60s boundary <=3600")
    campaign=load_json(root/"campaign.json")
    if any((root/m/"invalid.json").exists() or (root/m/"error.json").exists() for m in METHODS):
        raise RuntimeError("existing invalid/error evidence requires diagnosis before campaign resume")
    observed_lock=environment_lock()
    if observed_lock != load_json(root/"environment.lock.json"):
        raise RuntimeError("environment differs from first validated exact lock")
    atomic_json(root/"resource_observation.json",resource_observation())
    available=physical_cores()
    requested=None if cpus is None else list(cpus)
    if requested is not None and (len(requested)!=2 or len(set(requested))!=2 or any(c not in available for c in requested)):
        raise ValueError("cpus must name two available distinct physical-core representatives")
    if "requested_cpus" in campaign and requested is not None and requested!=campaign["requested_cpus"]:
        raise ValueError("campaign CPU selection is frozen after first run")
    cores=campaign.get("requested_cpus",requested) or available
    if any(c not in available for c in cores): raise ValueError("frozen campaign CPU selection unavailable")
    if requested is not None: campaign["requested_cpus"]=requested
    parallel=len(cores)>=2
    campaign.update(status="RUNNING",execution_mode="two_processes_distinct_physical_cores" if parallel else "alternating_same_60s_blocks",
        physical_cores=cores[:2],thread_environment={name:"1" for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")})
    atomic_json(root/"campaign.json",campaign)
    # Environment is inherited before interpreter/library imports in each child.
    for key,value in campaign["thread_environment"].items(): os.environ[key]=value
    ctx=mp.get_context("spawn")
    invalid=False
    jobs={}
    try:
        # Both worlds and their numerical/solver caches remain resident. The
        # serial fallback sends work to only one child at a time.
        for index,method in enumerate(METHODS):
            parent,child=ctx.Pipe()
            cpu=cores[index] if parallel else cores[0] if cores else None
            job=ctx.Process(target=_run_episode_guard,args=(str(root),method,None,cpu,child))
            job.start();child.close();jobs[method]=(job,parent)
        for boundary in range(60,int(target)+1,60):
            pending=[m for m in METHODS if not (root/m/f"summary_{boundary:04d}.json").exists()]
            groups=[pending] if parallel else [[m] for m in pending]
            for group in groups:
                if budget.compute_remaining<=0: break
                for method in group: jobs[method][1].send(boundary)
                waiting=set(group)
                while waiting and budget.compute_remaining>0:
                    for method in tuple(waiting):
                        job,pipe=jobs[method]
                        if pipe.poll(.25):
                            try: status,_=pipe.recv()
                            except EOFError: status="invalid"
                            waiting.remove(method)
                            if status!="completed": invalid=status=="invalid"
                        elif not job.is_alive():
                            waiting.remove(method);invalid=True
                    if invalid: break
                if invalid: break
            budget.persist()
            report_campaign(root,plots=False)
            if invalid or budget.compute_remaining<=0: break
    finally:
        for method,(job,pipe) in jobs.items():
            if job.is_alive():
                if budget.compute_remaining<=0 or invalid: job.terminate()
                else:
                    try: pipe.send(None)
                    except (BrokenPipeError,EOFError,OSError): pass
            job.join(timeout=2.)
            if job.is_alive(): job.terminate();job.join()
            pipe.close()
    common=common_boundary(root)
    campaign.update(status="INVALID_STOP" if invalid else "COMPLETED_3600" if common==3600 else "BUDGET_LIMITED" if budget.compute_remaining<=0 else "PARTIAL_REQUESTED_WINDOW",
        common_completed_time_s=common,campaign_wall_s=budget.elapsed,
        real_time_observation="MEASURED_NOT_CERTIFIED" if common else "NOT_MEASURED")
    atomic_json(root/"campaign.json",campaign)
    if export and common:
        for method in METHODS:
            if budget.elapsed>=budget.data["total_s"]: break
            job=ctx.Process(target=_archive_episode,args=(str(root),method,common));job.start()
            while job.is_alive() and budget.elapsed<budget.data["total_s"]: job.join(timeout=.5)
            if job.is_alive():
                job.terminate();job.join()
                atomic_json(root/method/"archive_status.json",dict(status="NOT_RUN_OR_INCOMPLETE_BUDGET",elapsed_s=budget.elapsed))
    # Plotting is bounded by the same archive allocation in the supervisor.
    result=report_campaign(root,plots=False)
    if budget.elapsed<budget.data["total_s"]:
        plot_job=ctx.Process(target=report_campaign,args=(str(root),));plot_job.start()
        while plot_job.is_alive() and budget.elapsed<budget.data["total_s"]: plot_job.join(timeout=.5)
        if plot_job.is_alive(): plot_job.terminate();plot_job.join()
    budget.persist()
    campaign.update(campaign_wall_s=budget.elapsed,archive_status={m:load_json(root/m/"archive_status.json")
        if (root/m/"archive_status.json").exists() else {"status":"NOT_RUN"} for m in METHODS})
    atomic_json(root/"campaign.json",campaign)
    return result
