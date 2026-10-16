"""Measured diagnostics for the frozen stage1 protocol; no training or optimizer.

Workers own independent worlds and streamed archives. cProfile uses the process
CPU clock; reported component times are exclusive measured function times.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import cProfile
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

from .qualification import THREAD_VARIABLES, provenance, task_coverage, write_json, limit_threads


def _rss():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == "Darwin" else value*1024), "resource.ru_maxrss_process_lifetime"
    except ImportError:
        try:
            import psutil
            m = psutil.Process().memory_info()
            return int(getattr(m, "peak_wset", m.rss)), "psutil.peak_wset_process_lifetime" if hasattr(m, "peak_wset") else "psutil.current_rss_not_peak"
        except ImportError: return None, "UNAVAILABLE"


def _component_profile(profile):
    groups = {k: 0. for k in ("generation", "motion_power", "separation", "resources", "state_update", "serialization", "emergency", "other")}
    functions = []
    for entry in profile.getstats():
        code = entry.code
        if isinstance(code, str): filename, name = "builtin", code
        else: filename, name = code.co_filename.replace("\\", "/"), code.co_name
        if filename.endswith("world.py") and name in ("_update_emergency", "_reference", "handle_emergencies", "_try_final", "_prove_pad_deadline"):
            category = "emergency"
        elif filename.endswith("resources.py"): category = "resources"
        elif filename.endswith("safety.py"): category = "separation"
        elif filename.endswith("energy.py"): category = "motion_power"
        elif filename.endswith("trajectory.py"):
            category = "generation" if name.startswith("generate_") or name in ("_candidate", "_build", "hermite", "final_height", "descent_segments") else "motion_power"
        elif filename.endswith("replay.py") or "/json/" in filename or "json" in filename and filename == "builtin": category = "serialization"
        elif filename.endswith("world.py"): category = "state_update"
        else: category = "other"
        groups[category] += entry.inlinetime
        if "rotating_slots" in filename:
            functions.append({"file": filename, "function": name, "calls": entry.callcount,
                              "exclusive_cpu_seconds": entry.inlinetime, "inclusive_cpu_seconds": entry.totaltime,
                              "category": category})
    return {"exclusive_cpu_seconds": groups, "classification": "mutually_exclusive_by_function; nested callees belongto their owncategory",
            "functions": sorted(functions, key=lambda r: -r["exclusive_cpu_seconds"]),
            "viewer": {"enabled": False, "cpu_seconds": 0., "reason": "viewer neverimported orstarted inmeasuredsimulation"}}


def _dependencies():
    result = {}
    for name in ("numpy", "scipy", "PyYAML", "psutil"):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = "NOT_INSTALLED"
    return result


def _new_world(root, **kwargs):
    from .replay import BufferedStateJSONLSink
    from .world import RotatingSlotWorld
    sink = BufferedStateJSONLSink(root/"states.jsonl")
    return RotatingSlotWorld(state_sink=sink, **kwargs), sink


def _diagnostic_job(job):
    limit_threads()
    from . import (generate_final, generate_inbound, generate_transfer, integrate_energy, slot_state, validate_separation,
                   certified_minimum_distance, make_stable_plan, TrajectoryPlan)
    from .demand import ArrivalManifest, ArrivalRequest
    from .driver import DeterministicDriver
    from .replay import ReplayBundle
    root = Path(job["output"])/job["id"]
    root.mkdir(parents=True, exist_ok=False)
    info = provenance(job["command"])
    profile = cProfile.Profile(timer=time.process_time)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    sink = None; status = "COMPLETED"; reason = ""; bundle_path = None
    result = {"id": job["id"], "job": job, "provenance": info, "dependencies": _dependencies()}
    write_json(root/"progress.json", {"status": "STARTING", "time": 0, "job": job})
    try:
        profile.enable()
        if job["kind"] in ("traffic", "scheduler"):
            manifest = ArrivalManifest.generate(job["seed"], job["rate"], job["window"])
            w, sink = _new_world(root, manifest=manifest, scheduler_period=job.get("period", 5.))
            driver = DeterministicDriver(); last = [-math.inf]
            def callback(snapshot, world):
                if snapshot.time-last[0] >= 30 or snapshot.failed:
                    write_json(root/"progress.json", {"status": "RUNNING", "sim_time": snapshot.time,
                        "target_sim_time": job["window"]+job["drain"], "wall_seconds": time.perf_counter()-wall_start,
                        "metrics": world.metrics(), "pid": os.getpid(), "observed_at": datetime.now(timezone.utc).isoformat()})
                    last[0] = snapshot.time
                driver(snapshot, world)
            w.advance(job["window"], callback)
            window_metrics = w.metrics() if abs(w.time-job["window"]) <= .001 and not w.failed else None
            if not w.failed: w.advance(job["window"]+job["drain"], callback)
            clear = not w.failed and all(a.status in ("CLEARED", "REROUTED") for a in w.aircraft.values())
            clear_time = max([job["window"], *(a.service_end for a in w.aircraft.values() if a.service_end is not None)]) if clear else None
            result.update(manifest=manifest.to_dict(), metrics=w.metrics(), decision_times=w.decision_times,
                          world_cpu_seconds=w.cpu, simulated_seconds=w.time,
                          arrival_window_metrics=window_metrics, cleared_by_drain_end=clear,
                          clear_completed_time=clear_time, failure_truncated_metrics=w.metrics() if w.failed else None)
        elif job["kind"] == "dt":
            w, sink = _new_world(root, physics_dt=job["dt"])
            w.seed_hold("a", "inner", 0, .8, service_class="short")
            ground = w.seed_service("other_ground", 3, remaining=300)
            other = TrajectoryPlan.from_dict(w.trajectories[ground.trajectory_hash])
            p = generate_final("a", "inner", 0, 0, 0).plan
            if p is None: raise RuntimeError("committedscript fixture NOT_FOUND")
            accepted = w.submit((p,))
            if not accepted.accepted: raise RuntimeError(accepted.reason)
            w.advance(p.end_time+60)
            a = w.aircraft["a"]
            #Actual continuous pairminimum is independent ofsamplegrid.
            minimum = certified_minimum_distance(p, other, 0, p.end_time)
            result.update(metrics=w.metrics(), events=w.events, energy_j=a.energy_j, soc=a.soc,
                final_p=a.p, touchdown_time=a.touchdown_time, minimum=minimum,
                committed_script={"plans": [p.to_dict()], "initial_soc": .8, "service_class": "short"}, simulated_seconds=w.time)
        elif job["kind"] == "benchmark":
            inflight = job["mode"] == "inflight_attempt"
            manifest = ArrivalManifest((ArrivalRequest("inflight", 0, 0, .9, "short"),)) if inflight else ArrivalManifest()
            w, sink = _new_world(root, manifest=manifest)
            #Keep knownsafe inboundsector vacant atpartialoccupancy. This
            #only chooses initialoccupiedslots, never changesfrozengeometry.
            layout = [("inner", j) for j in range(18)]+[("middle", j) for j in range(28)]+[("outer", (j+10) % 37) for j in range(37)]
            #Stable occupancy is exactlyN, noold60aircraftlimit.
            for ring, slot in layout[:job["count"]]:
                w.seed_hold(f"{ring}_{slot}", ring, slot, .9)
            attempts = []
            selected_inbound = None
            if inflight and job["count"] == 83:
                final = generate_final("inner_0", "inner", 0, 5, 0, durations=[35.])
                middle = generate_transfer("middle_1", "middle", 1, 0, .001, durations=[40.])
                outer = generate_transfer("outer_1", "outer", 1, 1, .002, durations=[40.])
                if any(c.plan is None for c in (final, middle, outer)):
                    raise RuntimeError("knownA05 releasechain generation NOT_FOUND")
                joint = w.submit((final.plan, middle.plan, outer.plan))
                attempts.append({"operation": "knownA05_jointreleasechain", "accepted": joint.accepted, "reason": joint.reason,
                    "plans": [c.plan.to_dict() for c in (final, middle, outer)]})
                if not joint.accepted: raise RuntimeError("knownA05 releasechain rejected:"+joint.reason)
                inbound = generate_inbound("inflight", 0, 1, 0, durations=[55.])
                if inbound.plan is None: raise RuntimeError("knownA05 inbound55s NOT_FOUND")
                selected_inbound = inbound.plan
            elif inflight:
                for slot in range(37):
                    if w.slots.active(("outer", slot), w.time):
                        attempts.append({"slot": slot, "skip_reason": "occupiedwithout committedrelease"}); continue
                    c = generate_inbound("inflight", 0, slot, w.time)
                    if c.plan is None:
                        attempts.append({"slot": slot, "generation_status": c.status}); continue
                    safe = all(validate_separation(c.plan, w._traffic(a), w.time,
                        c.plan.end_time+2*math.pi/.035).accepted for a in w.aircraft.values())
                    if not safe:
                        attempts.append({"slot": slot, "generation_status": c.status, "skip_reason": "continuous trafficconflict"}); continue
                    selected_inbound = c.plan
                    break
            def callback(snapshot, wrld):
                if inflight and selected_inbound is not None and "inflight" not in wrld.aircraft:
                    #admitreroutesonce afterfailure; do notretryarejectedrequest.
                    r = wrld.admit("inflight", selected_inbound)
                    attempts.append({"slot": selected_inbound.target_slot, "generation_status": "FEASIBLE", "accepted": r.accepted, "reason": r.reason})
            w.advance(job["duration"], callback)
            result.update(metrics=w.metrics(), world_cpu_seconds=w.cpu, simulated_seconds=w.time,
                actual_airborne_count=w.metrics()["physical_airborne_count"], inflight_attempts=attempts,
                inflight_admitted=bool(inflight and w.aircraft.get("inflight") and w.aircraft["inflight"].admitted))
            if inflight and not result["inflight_admitted"]:
                status, reason = "BLOCKED", "no certified additionalinflight fixture for thisstablecount; attemptsretained andnotcounted as acceptedinflightbenchmark"
        else: raise ValueError("unknownjobkind")
        bundle = ReplayBundle.write(root/"replay", world=w, code_commit=info["code_commit"],
            command=job["command"], seed=job.get("seed", 11), assertions=[{"conservation": w.metrics()["conservation"]}])
        bundle_path = str(bundle.path)
        if not w.metrics()["conservation"]: raise AssertionError("requestconservation failed")
        result["events_path"] = str(bundle.path/"events.jsonl")
        result["manifest_path"] = str(bundle.path/"manifest.json")
        result["state_count"] = w.to_dict().get("states_count", len(w.records))
    except Exception as exc:
        import traceback
        status, reason = "FAILED", f"{type(exc).__name__}: {exc}"
        (root/"error.log").write_text(traceback.format_exc(), encoding="utf8")
    finally:
        profile.disable()
        if sink is not None: sink.close()
    result.update(status=status, reason=reason, replay_path=bundle_path,
        wall_seconds=time.perf_counter()-wall_start, cpu_seconds=time.process_time()-cpu_start,
        peak_rss_bytes=_rss()[0], peak_rss_source=_rss()[1], measured_components=_component_profile(profile),
        researcher_visual_acceptance="PENDING")
    result["measurement_instrumentation"] = {"profiler": "cProfile", "clock": "process_time",
        "profiler_enabled_during_run": True, "wall_and_cpu_include_profiler_overhead": True}
    result["sim_seconds_per_wall_second"] = result.get("simulated_seconds", 0)/result["wall_seconds"]
    write_json(root/"result.json", result)
    write_json(root/"progress.json", {"status": status, "sim_time": result.get("simulated_seconds"), "reason": reason,
        "wall_seconds": result["wall_seconds"], "pid": os.getpid(), "result_path": str(root/"result.json")})
    return result


def execute_jobs(jobs, output, workers):
    if workers < 1: raise ValueError("workers must bepositive")
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    rows = []
    def progress():
        write_json(root/"progress.json", {"completed": len(rows), "total": len(jobs),
            "updated_at": datetime.now(timezone.utc).isoformat(), "coverage": task_coverage([j["id"] for j in jobs], rows),
            "jobs": [{"id": r["id"], "status": r["status"], "wall_seconds": r.get("wall_seconds")} for r in rows]})
    progress()
    if workers == 1:
        for job in jobs: rows.append(_diagnostic_job(job)); progress()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(_diagnostic_job, job): job["id"] for job in jobs}
            for future in as_completed(pending):
                try: rows.append(future.result())
                except Exception as exc: rows.append({"id": pending[future], "status": "FAILED", "reason": str(exc)})
                progress()
    return rows, task_coverage([j["id"] for j in jobs], rows)


def benchmark_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("job", {}).get("kind") == "benchmark" and row["status"] == "COMPLETED":
            groups[(row["job"]["count"], row["job"]["mode"])].append(row)
    result = []
    for (count, mode), values in sorted(groups.items()):
        walls = sorted(x["wall_seconds"] for x in values)
        #Nearest-rank P95, explicitly defined for thefiverepeats.
        result.append({"initial_stable_count": count, "mode": mode, "repeats": len(values),
            "median_wall_seconds": statistics.median(walls), "p95_wall_seconds": walls[math.ceil(.95*len(walls))-1],
            "p95_definition": "nearest_rank", "median_cpu_seconds": statistics.median(x["cpu_seconds"] for x in values),
            "maximum_peak_rss_bytes": max((x["peak_rss_bytes"] or 0) for x in values),
            "median_simsec_per_wallsec": statistics.median(x["sim_seconds_per_wall_second"] for x in values),
            "inflight_admitted_repeats": sum(x.get("inflight_admitted", False) for x in values),
            "absolute_performance_pass_threshold": None})
    return result


def run_benchmarks(output, counts=(10, 30, 60, 83), repeats=5, duration=.2, workers=1):
    root = Path(output).resolve()
    jobs = [{"id": f"benchmark_n{n}_{mode}_r{repeat}", "kind": "benchmark", "count": n,
        "mode": mode, "repeat": repeat, "duration": duration, "output": str(root),
        "command": "run_stage1_tests G09 actualbenchmark"}
        for n in counts for mode in ("stable", "inflight_attempt") for repeat in range(repeats)]
    rows, coverage = execute_jobs(jobs, root, workers)
    result = {"jobs": rows, "coverage": coverage, "summary": benchmark_summary(rows)}
    write_json(root/"benchmark_results.json", result)
    return result


def run_diagnostics(output, *, workers=1, pilot=None, pilot_window=2., pilot_drain=0.,
                    pilot_counts=(10,), pilot_repeats=1, pilot_duration=.2, command=None):
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    if (root/"diagnostics.json").exists(): raise FileExistsError("useafreshoutputdirectory")
    if command is None:
        command = f"python -m terminal_marl.scripts.run_stage1_diagnostics --output {root} --workers {workers}"
        if pilot: command += f" --pilot {pilot} --pilot-window {pilot_window} --pilot-drain {pilot_drain} --pilot-counts {','.join(map(str, pilot_counts))} --pilot-repeats {pilot_repeats} --pilot-duration {pilot_duration}"
    jobs = []
    if pilot is None or pilot in ("traffic", "all"):
        for rate in ((60, 120, 180) if pilot is None else (60,)):
            for seed in ((11, 23, 47, 71, 101) if pilot is None else (11, 23)):
                jobs.append({"id": f"traffic_rate{rate}_seed{seed}", "kind": "traffic", "rate": rate, "seed": seed,
                    "window": 3600. if pilot is None else pilot_window, "drain": 1800. if pilot is None else pilot_drain})
    if pilot is None or pilot in ("benchmark", "all"):
        for n in ((10, 30, 60, 83) if pilot is None else pilot_counts):
            for mode in ("stable", "inflight_attempt"):
                for repeat in range(5 if pilot is None else pilot_repeats):
                    jobs.append({"id": f"benchmark_n{n}_{mode}_r{repeat}", "kind": "benchmark", "count": n,
                        "mode": mode, "repeat": repeat, "duration": 1. if pilot is None else pilot_duration})
    if pilot is None or pilot in ("sweeps", "all"):
        jobs += [{"id": f"dt_{dt}", "kind": "dt", "dt": dt} for dt in (.2, .1, .05)]
        jobs += [{"id": f"scheduler_{period}", "kind": "scheduler", "period": period, "seed": 11, "rate": 120,
            "window": 3600. if pilot is None else pilot_window, "drain": 1800. if pilot is None else pilot_drain} for period in (1., 5., 10.)]
    for job in jobs: job.update(output=str(root), command=command, pilot=bool(pilot))
    write_json(root/"jobs.json", jobs)
    rows, coverage = execute_jobs(jobs, root, workers)
    dtrows = [r for r in rows if r.get("job", {}).get("kind") == "dt" and r["status"] == "COMPLETED"]
    dtcomparison = {"status": "NOT_RUN"}
    if len(dtrows) == 3:
        energies = [r["energy_j"] for r in dtrows]
        minimums = [r["minimum"] for r in dtrows]
        dtcomparison = {"status": "MEASURED", "energy_relative_spread": max(energies)/min(energies)-1,
            "energy_within_0p005": max(energies)/min(energies)-1 <= .005,
            "events_within_contract_tolerance": all(__import__("terminal_marl.core.rotating_slots.fixtures", fromlist=["event_equivalent"]).event_equivalent(r["events"], dtrows[0]["events"]) for r in dtrows),
            "identical_committed_script": all(r["committed_script"] == dtrows[0]["committed_script"] for r in dtrows),
            "minimum_width_across_dt_m": max(x["upper_bound_m"] for x in minimums)-min(x["lower_bound_m"] for x in minimums)}
    result = {"schema": "FT-RS-S1-1.0-diagnostics", "protocol": "PILOT_NOT_FORMAL_MATRIX" if pilot else "FROZEN_FORMAL_MATRIX",
        "jobs": rows, "coverage": coverage, "benchmark_summary": benchmark_summary(rows), "dt_comparison": dtcomparison,
        "scheduler_sensitivity": [{"period": r["job"]["period"], "metrics": r.get("metrics"), "cpu_seconds": r.get("cpu_seconds"),
            "decision_count": len(r.get("decision_times", [])), "wall_seconds": r.get("wall_seconds")} for r in rows if r.get("job", {}).get("kind") == "scheduler"],
        "researcher_visual_acceptance": "PENDING", "stage1_status": "NOT_ACCEPTED", "command": command}
    write_json(root/"diagnostics.json", result)
    return result
