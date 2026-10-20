"""One cold, same-state practical 60 s run; offline checks use frozen source.

Invoke each command in a fresh interpreter. No profiler, rendering or workers.
The saved checkpoint identity is intentionally accepted for a numerical/runtime
comparison; this is not a declaration of equivalent checkpoint migration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import wraps
import importlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

CORE = "terminal_marl.core.rotating_slots"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_core(source_root):
    """Fail rather than accidentally reuse a different already imported core."""
    if any(name == "terminal_marl" or name.startswith("terminal_marl.") for name in sys.modules):
        raise RuntimeError("source selection requires a fresh interpreter")
    root = Path(source_root).resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    modules = {name: importlib.import_module(CORE + "." + name) for name in
               ("config", "energy", "trajectory", "safety", "world", "replay", "performance.checkpoint")}
    for module in modules.values():
        if root not in Path(module.__file__).resolve().parents:
            raise RuntimeError("core module escaped selected source root: " + module.__file__)
    return modules


def cold_caches(modules):
    seen = set()
    for module in modules.values():
        for value in vars(module).values():
            if id(value) not in seen and callable(getattr(value, "cache_clear", None)):
                value.cache_clear()
                seen.add(id(value))
    for name in ("root_cache_clear", "reset_practical_numeric_stats"):
        fn = getattr(modules["energy"], name, None)
        if fn is not None:
            fn()


def load_payload(codec, checkpoint):
    # Read the stored identity only to select this intentional comparison;
    # the normal codec still verifies the envelope hash and entire payload.
    document = json.loads(Path(checkpoint).read_bytes())
    identity = codec.CheckpointIdentity(**document["payload"]["identity"])
    return codec.load_checkpoint(checkpoint, expected_identity=identity), identity


class CallCounts:
    """Small explicit wrappers; imported aliases are replaced by identity."""
    def __init__(self):
        self.calls = Counter()
        self.precompute_seconds = 0.
        self.precompute_depth = 0
        self.replacements = []

    def install(self, modules):
        targets = {
            "energy": ("evaluate_power", "integrate_energy", "power_bounds", "_power_array", "_peak_measure", "_segment_table", "_peak_table", "_segment_initial", "_plan_numerical_policy"),
            "trajectory": ("generate_final", "generate_transfer", "generate_inbound"),
            "safety": ("validate_plan", "validate_separation"),
        }
        for module_name, names in targets.items():
            for name in names:
                module = modules[module_name]
                if not hasattr(module, name):
                    continue
                old = getattr(module, name)
                miss = ((lambda args, kwargs, fn=old: fn.cache_info().misses)
                        if name in ("_segment_table", "_peak_table", "_segment_initial", "_plan_numerical_policy") else None)
                wrapped = self.wrap(old, module_name + "." + name, miss)
                for imported_name, imported in list(sys.modules.items()):
                    if imported_name.startswith(CORE) and imported is not None:
                        for alias, value in list(vars(imported).items()):
                            if value is old:
                                self.replacements.append((imported, alias, old))
                                setattr(imported, alias, wrapped)
        cls = modules["world"].RotatingSlotWorld
        for name in ("_reference", "_reference_pad", "_record", "_check_execution", "_canonical_reference_duration"):
            if not hasattr(cls, name):
                continue
            old = getattr(cls, name)
            miss = None
            if name == "_canonical_reference_duration":
                miss = lambda args, kwargs: (args[1], args[2], args[0].config.profile_hash) not in args[0]._reference_templates
            self.replacements.append((cls, name, old))
            setattr(cls, name, self.wrap(old, "world." + name, miss))

    def wrap(self, fn, name, miss=None):
        @wraps(fn)
        def observed(*args, **kwargs):
            self.calls[name] += 1
            before = miss(args, kwargs) if miss else None
            # Miss timing includes the actual cached call, never __wrapped__.
            began = time.perf_counter() if miss else None
            outer = self.precompute_depth == 0
            if miss:
                self.precompute_depth += 1
            try:
                return fn(*args, **kwargs)
            finally:
                if miss:
                    self.precompute_depth -= 1
                    happened = before if isinstance(before, bool) else fn.cache_info().misses > before
                    if happened:
                        self.calls[name + ".misses"] += 1
                        if outer:
                            self.precompute_seconds += time.perf_counter() - began
        return observed

    def remove(self):
        for owner, name, old in reversed(self.replacements):
            setattr(owner, name, old)


def numeric_delta(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()
            if type(value) in (int, float) and type(before.get(key, 0)) in (int, float)}


def run(args):
    for name in THREAD_VARS:
        os.environ[name] = "1"
    modules = load_core(args.source_root)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    codec = modules["performance.checkpoint"]
    payload, identity = load_payload(codec, args.checkpoint)
    restored = codec.restore_checkpoint(payload, expected_identity=identity, cache_mode="cold",
                                        new_attempt_dir=output / "attempt")
    if restored.cursor.phase != "loop_start":
        raise ValueError("comparison requires an original loop_start checkpoint")
    world = restored.world
    if hasattr(world, "configure_practical_runtime"):
        world.configure_practical_runtime(enabled=True, reference_cache_path=output / "reference_cache.json")
    cold_caches(modules)
    start = world.time
    target = start + 60.
    initial = world.metrics()
    report = dict(label=args.label, source_root=str(Path(args.source_root).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_identity=payload["identity"],
        comparison_scope="intentional numerical/runtime comparison; no equivalence migration",
        cursor=codec.StateCodec().decode(payload["cursor"]), start_time=start, target_time=target,
        workers=0, cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        numerical_threads={key: os.environ[key] for key in THREAD_VARS}, profiler=False,
        initial_metrics=initial, initial_status_counts=dict(Counter(a.status for a in world.aircraft.values())),
        driver_calls_initial=list(restored.driver.calls) if restored.driver else [], status="running")
    write_json(output / "initial.json", report)
    counts = CallCounts()
    counts.install(modules)
    cls = type(world)
    execution = cls._check_execution
    next_progress = start + 5.
    began = time.perf_counter()
    process_began = time.process_time()
    def progress(instance, end):
        nonlocal next_progress
        result = execution(instance, end)
        if instance.time >= next_progress:
            write_json(output / "progress.json", dict(time=instance.time, target_time=target,
                wall_seconds=time.perf_counter()-began, calls=dict(counts.calls)))
            next_progress = instance.time + 5.
        return result
    cls._check_execution = progress
    try:
        world.advance(target, restored.driver)
        world.flush_records()
        wall_seconds = time.perf_counter() - began
        process_seconds = time.process_time() - process_began
    except BaseException:
        report.update(status="failed", end_time=world.time,
            wall_seconds=time.perf_counter()-began,process_seconds=time.process_time()-process_began,
            calls=dict(counts.calls),traceback=traceback.format_exc())
        write_json(output / "result.json", report)
        raise
    finally:
        cls._check_execution = execution
        counts.remove()
    final = world.metrics()
    report.update(status="completed" if abs(world.time-target) <= 1e-8 and not world.failed else "stopped",
        end_time=world.time, simulated_seconds=world.time-start, wall_seconds=wall_seconds,
        process_seconds=process_seconds, precompute_wall_seconds=counts.precompute_seconds,
        precompute_scope="union of outer initial-Gauss, plan-policy, energy-table, peak-table and canonical-reference cache-miss calls; phase-LUT separately in practical_runtime, do not sum",
        calls=dict(counts.calls), final_metrics=final, delta_metrics=numeric_delta(initial, final),
        final_status_counts=dict(Counter(a.status for a in world.aircraft.values())),
        driver_calls_final=list(restored.driver.calls) if restored.driver else [])
    if hasattr(world, "practical_runtime_stats"):
        report["practical_runtime"] = world.practical_runtime_stats()
    if hasattr(modules["energy"], "practical_numeric_stats"):
        report["practical_numeric"] = modules["energy"].practical_numeric_stats()
    if hasattr(modules["energy"], "root_cache_stats"):
        report["root_cache"] = modules["energy"].root_cache_stats()
    replay_path = output / "replay"
    modules["replay"].ReplayBundle.write(replay_path, world=world, code_commit=args.code_commit,
        config_path=modules["config"].DEFAULT_CONFIG, command=" ".join(sys.argv), eager=False)
    report["replay_path"] = str(replay_path)
    write_json(output / "result.json", report)
    if world.state_sink is not None:
        world.state_sink.close()
    print(json.dumps(dict(status=report["status"], result=str(output / "result.json"), wall_seconds=wall_seconds)))
    return 0 if report["status"] == "completed" else 1


def window_intervals(frames, start, end):
    """Use old-frame plans until the next authoritative change, ground included."""
    physical = frozenset(("INBOUND", "HOLD_OUTER", "HOLD_MIDDLE", "HOLD_INNER", "TRANSFER_O_M",
        "TRANSFER_M_I", "FINAL_NORMAL", "FINAL_EMERGENCY", "VERTICAL_DESCENT", "LANDED_SERVICE"))
    previous = None
    for frame in frames:
        t = float(frame["time"])
        if previous is not None:
            lo, hi = max(start, float(previous["time"])), min(end, t)
            if hi > lo:
                rows = {a["id"]: a for a in previous["aircraft"]}
                current = {a["id"]: a for a in frame["aircraft"]}
                yield lo, hi, rows, current, physical
        previous = frame
        if t >= end:
            break


def _select_energy_samples(spans, limit):
    """Deterministic kind coverage followed by evenly spaced remaining spans."""
    ordered=sorted(spans,key=lambda s:(s['kind'],s['aircraft_id'],s['trajectory_hash'],s['start'],s['end']))
    chosen=[]; kinds=set()
    for i,span in enumerate(ordered):
        if len(chosen)>=limit: break
        if span['kind'] not in kinds:
            chosen.append(i); kinds.add(span['kind'])
    remaining=[i for i in range(len(ordered)) if i not in chosen]
    count=min(limit-len(chosen),len(remaining))
    if count==1: chosen.append(remaining[len(remaining)//2])
    elif count>1:
        chosen.extend(remaining[round(i*(len(remaining)-1)/(count-1))] for i in range(count))
    return [ordered[i] for i in sorted(chosen)]


def audit(args):
    audit_began=time.perf_counter(); audit_process_began=time.process_time()
    energy_limit=getattr(args,'energy_samples',0)
    minimum_tolerance=getattr(args,'minimum_tolerance_m',.01)
    if energy_limit<0: raise ValueError('energy sample count must be nonnegative')
    if not math.isfinite(minimum_tolerance) or minimum_tolerance<=0: raise ValueError('minimum tolerance must be positive finite')
    modules = load_core(args.source_root)
    root = Path(args.replay).resolve()
    try:
        modules["replay"].ReplayBundle.validate(root)
    except Exception as exc:
        if not energy_limit: raise
        report=dict(status='FAIL',source_root=str(Path(args.source_root).resolve()),replay_path=str(root),
                    window=[args.start,args.end],checks={},failures=[dict(kind='replay_validation',reason=str(exc))],
                    energy_mode='sampled_merged_execution',energy_sample_limit=energy_limit,energy_sample_count=0,
                    energy_samples=[],minimum_tolerance_m=minimum_tolerance,
                    audit_wall_seconds=time.perf_counter()-audit_began,
                    audit_process_seconds=time.process_time()-audit_process_began)
        write_json(args.output,report)
        print(json.dumps(dict(status='FAIL',output=str(Path(args.output).resolve()),checks={})))
        return 1
    config = modules["config"].load_config(root / "config.yaml")
    plans = {key: modules["trajectory"].TrajectoryPlan.from_dict(value, config=config)
             for key, value in json.loads((root / "trajectories.json").read_bytes()).items()}
    safety, energy = modules["safety"], modules["energy"]
    failures, checks, runs, live, used = [], Counter(), [], {}, set()
    coverage_start = coverage_end = None
    max_energy_error = max_soc_error = max_peak_error = 0.
    energy_spans=[]; energy_latest={}; sample_results=[]
    exact_start = exact_end = False
    def issue(kind, **details):
        failures.append(dict(kind=kind, **details))
    with (root / "states.jsonl").open(encoding="utf-8") as stream:
        def frames():
            nonlocal exact_start, exact_end
            for line in stream:
                if not line.strip():
                    continue
                frame = json.loads(line)
                exact_start |= abs(float(frame["time"])-args.start) <= 1e-8
                exact_end |= abs(float(frame["time"])-args.end) <= 1e-8
                yield frame
        for lo, hi, rows, current, physical in window_intervals(frames(), args.start, args.end):
            coverage_start = lo if coverage_start is None else coverage_start
            coverage_end = hi
            checks["frame_intervals"] += 1
            next_live = {}
            for a in rows.values():
                if a.get("status") not in physical:
                    continue
                key = a["trajectory_hash"]
                if energy_limit and key not in plans:
                    issue('energy_sample_fields',aircraft_id=a.get('id'),hash=key,reason='trajectory is missing')
                    continue
                if key not in used:
                    result = safety.validate_plan(plans[key], config=config)
                    checks["plan_checks"] += 1
                    used.add(key)
                    if not result.accepted:
                        issue("plan", hash=key, status=result.status, reason=result.reason, details=result.details)
                if a["status"] == "LANDED_SERVICE":
                    continue
                # Replay records exact event boundaries; no partial-window delta
                # is compared with a complete-frame ledger delta.
                if energy_limit:
                    try:
                        b=current[a['id']]
                        fields=('energy_j','soc','peak_seconds')
                        initial_ledger={name:float(a[name]) for name in fields}
                        final_ledger={name:float(b[name]) for name in fields}
                        if not all(math.isfinite(v) for row in (initial_ledger,final_ledger) for v in row.values()):
                            raise ValueError('nonfinite ledger field')
                        identity=(a['id'],key); previous=energy_latest.get(identity)
                        if previous is not None and abs(previous['end']-lo)<=1e-8:
                            if any(abs(previous['ledger_end'][name]-initial_ledger[name])>1e-9 for name in fields):
                                raise ValueError('adjacent ledger boundary mismatch')
                            previous['end']=hi; previous['ledger_end']=final_ledger
                        else:
                            span=dict(aircraft_id=a['id'],trajectory_hash=key,kind=plans[key].kind,
                                      start=lo,end=hi,ledger_start=initial_ledger,ledger_end=final_ledger)
                            energy_spans.append(span); energy_latest[identity]=span
                    except (KeyError,TypeError,ValueError) as exc:
                        issue('energy_sample_fields',aircraft_id=a.get('id'),hash=key,interval=[lo,hi],reason=str(exc))
                    continue
                b = current[a["id"]]
                q = energy.integrate_energy(plans[key], lo, hi, max_dt=.2, config=config)
                checks["energy_intervals"] += 1
                de = abs(b["energy_j"]-a["energy_j"]-q.joules)
                ds = abs(a["soc"]-b["soc"]-q.soc_used)
                dp = abs(b["peak_seconds"]-a["peak_seconds"]-q.peak_seconds)
                max_energy_error, max_soc_error, max_peak_error = max(max_energy_error,de), max(max_soc_error,ds), max(max_peak_error,dp)
                if not q.valid or de > max(.05,1e-7*abs(q.joules)) or ds > 1e-7 or dp > 1e-5:
                    issue("energy", aircraft_id=a["id"], interval=[lo,hi], valid=q.valid,
                          energy_error_j=de, soc_error=ds, peak_error_s=dp)
            bodies = sorted((a for a in rows.values() if a.get("status") in physical and
                             (not energy_limit or a.get('trajectory_hash') in plans)), key=lambda a:a["id"])
            for a, b in combinations(bodies, 2):
                pair = (a["id"],b["id"])
                hashes = (a["trajectory_hash"],b["trajectory_hash"])
                old = live.pop(pair, None)
                if old is not None and old[0:2] == hashes and abs(old[3]-lo) <= 1e-8:
                    next_live[pair] = (*hashes,old[2],hi)
                else:
                    if old is not None:
                        runs.append(old)
                    next_live[pair] = (*hashes,lo,hi)
            runs.extend(live.values())
            live = next_live
    runs.extend(live.values())
    if coverage_start is None or abs(coverage_start-args.start)>1e-8 or abs(coverage_end-args.end)>1e-8:
        issue("window_coverage", observed=[coverage_start,coverage_end], expected=[args.start,args.end])
    if not exact_start or not exact_end:
        issue("window_boundary_frame_missing", exact_start=exact_start, exact_end=exact_end)
    if energy_limit:
        complete_energy={}
        capacity=config.energy.equivalent_voltage_V*config.energy.reference_capacity_Ah*3600
        if not energy_spans:
            issue('energy_sample_uncertain',reason='no executed airborne ledger intervals are available')
        for span in _select_energy_samples(energy_spans,energy_limit):
            key=span['trajectory_hash']; plan=plans[key]
            entry={k:span[k] for k in ('kind','aircraft_id','trajectory_hash','start','end','ledger_start','ledger_end')}
            a,b=span['ledger_start'],span['ledger_end']
            observed=dict(energy_j=b['energy_j']-a['energy_j'],soc_used=a['soc']-b['soc'],
                          peak_seconds=b['peak_seconds']-a['peak_seconds'])
            entry['observed']=observed
            try:
                if key not in complete_energy:
                    complete_energy[key]=energy.integrate_energy(plan,plan.start_time,plan.end_time,max_dt=.2,config=config)
                    checks['energy_complete_plan_integrations']+=1
                full=complete_energy[key]
                reference=energy.integrate_energy(plan,span['start'],span['end'],max_dt=.2,config=config)
                checks['energy_sample_integrations']+=1
                if not full.valid or not reference.valid:
                    entry.update(status='FAIL',reason='frozen energy kernel rejected reference')
                    issue('energy_sample_reference_invalid',sample=entry)
                    sample_results.append(entry)
                    continue
                expected=dict(energy_j=float(reference.joules),soc_used=float(reference.soc_used),peak_seconds=float(reference.peak_seconds))
                if not all(math.isfinite(v) for v in (*expected.values(),full.joules)):
                    raise ValueError('nonfinite reference energy')
                budget_j=max(1.,1e-6*abs(full.joules))
                budgets=dict(energy_j=budget_j,soc_used=budget_j/capacity,peak_seconds=.01)
                signed={name:observed[name]-expected[name] for name in expected}
                errors={name:abs(value) for name,value in signed.items()}
                passed=all(errors[name]<=budgets[name] for name in expected)
                entry.update(status='PASS' if passed else 'FAIL',reference=expected,errors=errors,
                             signed_errors=signed,budgets=budgets,complete_plan_energy_j=full.joules)
                max_energy_error=max(max_energy_error,errors['energy_j'])
                max_soc_error=max(max_soc_error,errors['soc_used'])
                max_peak_error=max(max_peak_error,errors['peak_seconds'])
                if not passed: issue('energy_sample_budget',sample=entry)
            except Exception as exc:
                entry.update(status='FAIL',reason=str(exc))
                issue('energy_sample_uncertain',sample=entry)
            sample_results.append(entry)
        checks['energy_sample_count']=len(sample_results)
        checks['energy_candidate_intervals']=len(energy_spans)
    bounds = []
    for h1,h2,lo,hi in runs:
        result = safety.validate_separation(plans[h1],plans[h2],lo,hi,config=config)
        checks["separation_runs"] += 1
        if not result.accepted:
            issue("separation", hashes=[h1,h2], interval=[lo,hi], status=result.status, details=result.details)
        bounds.append((float(result.details.get("lower_bound_m",0.)),h1,h2,lo,hi))
    best_upper, minimum_lower, witness = math.inf, math.inf, None
    for lower,h1,h2,lo,hi in sorted(bounds):
        if lower >= best_upper:
            checks["minimum_pruned"] += 1
            minimum_lower = min(minimum_lower,lower)
            continue
        value = safety.certified_minimum_distance(plans[h1],plans[h2],lo,hi,config=config,tolerance_m=minimum_tolerance)
        checks["minimum_refinements"] += 1
        minimum_lower = min(minimum_lower,value["lower_bound_m"])
        if value["upper_bound_m"] < best_upper:
            best_upper, witness = value["upper_bound_m"], dict(value,hashes=[h1,h2],interval=[lo,hi])
        if value["status"] != "CERTIFIED":
            issue("minimum_unknown", hashes=[h1,h2], interval=[lo,hi], result=value)
    report = dict(status="PASS" if not failures else "FAIL", source_root=str(Path(args.source_root).resolve()),
        checker_scope="selected frozen source; original continuous plan/separation and energy kernels; executed window only",
        replay_path=str(root), window=[args.start,args.end], checks=dict(checks), failures=failures,
        minimum_distance=None if not bounds else dict(lower_bound_m=minimum_lower,upper_bound_m=best_upper,witness=witness),
        max_energy_error_j=max_energy_error,max_soc_error=max_soc_error,max_peak_error_s=max_peak_error,
        energy_mode='sampled_merged_execution' if energy_limit else 'strict_per_frame',
        energy_sample_limit=energy_limit,energy_sample_count=len(sample_results),energy_samples=sample_results,
        minimum_tolerance_m=minimum_tolerance,audit_wall_seconds=time.perf_counter()-audit_began,
        audit_process_seconds=time.process_time()-audit_process_began)
    write_json(args.output, report)
    print(json.dumps(dict(status=report["status"], output=str(Path(args.output).resolve()), checks=dict(checks))))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("--source-root", required=True)
    one.add_argument("--checkpoint", required=True)
    one.add_argument("--output", required=True)
    one.add_argument("--label", choices=("before", "after"), required=True)
    one.add_argument("--code-commit", required=True)
    check = commands.add_parser("audit")
    check.add_argument("--source-root", required=True, help="frozen original source root")
    check.add_argument("--replay", required=True)
    check.add_argument("--start", type=float, required=True)
    check.add_argument("--end", type=float, required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--energy-samples",type=int,default=0,help="merged execution samples; 0 retains strict per-frame energy checking")
    check.add_argument("--minimum-tolerance-m",type=float,default=.01,help="minimum-distance refinement tolerance only; hard separation remains 80m")
    args = parser.parse_args()
    return run(args) if args.command == "run" else audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
