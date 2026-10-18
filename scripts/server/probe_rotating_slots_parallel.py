"""Bounded exact-output probes. These outputs are never formal/G09 evidence."""
from __future__ import annotations

import argparse
import cProfile
import json
import math
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.server import recover_rotating_slots_diagnostics as recovery
from scripts.server.parallel_rotating_slots_runtime import (
    ParallelReferenceWorld, ReferencePool, execution_identity)
from terminal_marl.core.rotating_slots import diagnostics
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.demand import ArrivalManifest
from terminal_marl.core.rotating_slots.driver import DeterministicDriver
from terminal_marl.core.rotating_slots.qualification import write_json
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld


DIAGNOSTIC_ONLY_FIELDS = ("cpu_seconds", "cache_hits", "reference_counts",
                          "last_reference_fallback", "cache")


def physical_record(world):
    record = world.to_dict()
    for key in DIAGNOSTIC_ONLY_FIELDS:
        record.pop(key, None)
    return record


def differences(a, b, path="$", limit=100):
    """Exact values and types; keep bounded concrete mismatch witnesses."""
    found = []
    def visit(x, y, p):
        if len(found) >= limit:
            return
        if type(x) is not type(y):
            found.append({"path": p, "serial": x, "parallel": y, "reason": "type"})
        elif isinstance(x, dict):
            for key in sorted(x.keys() | y.keys()):
                if key not in x or key not in y:
                    found.append({"path": p+"."+str(key), "reason": "missing key"})
                else:
                    visit(x[key], y[key], p+"."+str(key))
        elif isinstance(x, (list, tuple)):
            if len(x) != len(y):
                found.append({"path": p, "serial_length": len(x), "parallel_length": len(y)})
            for i, (xx, yy) in enumerate(zip(x, y)):
                visit(xx, yy, p+f"[{i}]")
        elif x != y:
            found.append({"path": p, "serial": x, "parallel": y})
    visit(a, b, path)
    return found


def reference_payload(result):
    soc, plan = result
    return {"soc_used": soc, "plan": plan.to_dict() if plan is not None else None}


def representative_queries():
    # Pad0 relative phase at every bin center, plus exact current-time queries.
    omega = load_config().airspace.common_angular_speed_radps
    bins = RotatingSlotWorld.REFERENCE_PHASE_BINS
    queries = [(ring, 0, (phase+.5)*2*math.pi/bins/omega, 0)
               for ring in ("outer", "middle", "inner") for phase in range(bins)]
    queries += [(ring, slot, 1235., 2)
                for ring in ("outer", "middle", "inner") for slot in (0, 1, 5)]
    return queries


def mixed_3ring_fixture():
    """Nine synthetic fixture aircraft, never a restored seed71 checkpoint."""
    return [{"aircraft_id": f"probe_{ring}_{slot}", "ring": ring, "slot": slot,
             "soc": .26 if (ring, slot) == ("inner", 0) else .8,
             "service_class": "standard"}
            for ring, slots in (("outer", (0, 12, 24)), ("middle", (0, 9, 18)),
                                ("inner", (0, 6, 12))) for slot in slots]


def measure(call):
    profile = cProfile.Profile(timer=time.process_time)
    wall, cpu = time.perf_counter(), time.process_time()
    profile.enable()
    try:
        value = call()
    finally:
        profile.disable()
    return value, {"wall_seconds": time.perf_counter()-wall,
        "parent_cpu_seconds": time.process_time()-cpu,
        "parent_components": diagnostics._component_profile(profile),
        "scope": "parent cProfile across call; child per-task CPU/profile is separate"}


def reference_probe(queries, workers):
    world = RotatingSlotWorld()
    serial, serial_times = [], []
    passes = [("cold", queries), ("exact_cache", queries),
              ("advance_time_warm_templates", [(r, s, t+.2, v) for r, s, t, v in queries])]
    for label, pass_queries in passes:
        values, timing = measure(lambda: [reference_payload(world._reference(*q)) for q in pass_queries])
        serial.append(values)
        serial_times.append(dict(timing, cache_pass=label, queries=pass_queries))
    variants = []
    for count in workers:
        with ReferencePool(count) as pool:
            measured_passes = []
            cache = {}
            for i, (label, pass_queries) in enumerate(passes):
                # ParallelReferenceWorld uses the parent exact cache on warm
                # passes. Mirror that lookup rather than forcing pad rework.
                def query_batch():
                    missing = [q for q in pass_queries if q not in cache]
                    cache.update(pool.references(missing) if missing else {})
                    return [reference_payload(cache[q]) for q in pass_queries]
                child_before = pool.summary()["child_task_cpu_seconds"]
                values, timing = measure(query_batch)
                timing["child_task_cpu_seconds"] = pool.summary()["child_task_cpu_seconds"]-child_before
                timing["known_parent_plus_child_task_cpu_seconds"] = (
                    timing["parent_cpu_seconds"]+timing["child_task_cpu_seconds"])
                measured_passes.append({"cache_pass": label, "queries": pass_queries, "timing": timing,
                    "differences": differences(serial[i], values), "values": values})
        variants.append({"inner_workers": count, "passes": measured_passes, "pool": pool.summary()})
    return {"queries": queries, "serial_passes": serial_times,
            "serial_values": serial, "variants": variants}


def _make_world(pool=None, fixture="traffic"):
    manifest = ArrivalManifest.generate(71, 180, 3600.) if fixture == "traffic" else None
    kwargs = {"manifest": manifest}
    world = (RotatingSlotWorld(**kwargs) if pool is None else
             ParallelReferenceWorld(reference_pool=pool, **kwargs))
    if fixture == "seeded_inner":
        world.seed_hold("probe_a", "inner", 0, .8, service_class="short")
    elif fixture == "mixed_3ring":
        for a in mixed_3ring_fixture():
            world.seed_hold(**a)
    return world, False if fixture == "mixed_3ring" else DeterministicDriver()


def run_world(until, pool=None, fixture="traffic"):
    world, callback = _make_world(pool, fixture)
    world.advance(until, callback)
    return physical_record(world)


def _world_stages(until, pool=None, fixture="traffic"):
    world = callback = None
    def initial():
        nonlocal world, callback
        world, callback = _make_world(pool, fixture)
        world.advance(until, callback)
        return physical_record(world)
    def continuation():
        world.advance(2*until, callback)
        return physical_record(world)
    stages = []
    for name, target, call in (("initial", until, initial),
                                ("continuation", 2*until, continuation)):
        before = pool.summary()["child_task_cpu_seconds"] if pool else 0.
        record, timing = measure(call)
        child = pool.summary()["child_task_cpu_seconds"]-before if pool else 0.
        timing["child_task_cpu_seconds"] = child
        timing["known_parent_plus_child_task_cpu_seconds"] = timing["parent_cpu_seconds"]+child
        stages.append({"stage": name, "target_sim_time": target,
                       "timing": timing, "record": record})
    return stages


def _stage_totals(stages):
    keys = ("wall_seconds", "parent_cpu_seconds", "child_task_cpu_seconds",
            "known_parent_plus_child_task_cpu_seconds")
    return {k: sum(s["timing"][k] for s in stages) for k in keys}


def world_probe(until, workers, fixture):
    serial = _world_stages(until, fixture=fixture)
    variants = []
    for count in workers:
        with ReferencePool(count) as pool:
            stages = _world_stages(until, pool, fixture)
        mismatch = []
        for expected, actual in zip(serial, stages):
            actual["differences"] = differences(expected["record"], actual["record"])
            mismatch.extend(actual["differences"])
        variants.append({"inner_workers": count, "timing": _stage_totals(stages),
            "stages": stages, "differences": mismatch,
            "record": stages[-1]["record"], "pool": pool.summary()})
    return {"until": until, "continuation_until": 2*until, "fixture": fixture,
            "fixture_aircraft": mixed_3ring_fixture() if fixture == "mixed_3ring" else None,
            "fixture_scope": "mixed_3ring is a synthetic nine-aircraft callback=False probe; "
                "automatic emergency handling retained; not seed71's actual late stage or checkpoint",
            "timing_scope": "initial includes world fixture creation and lazy spawn; continuation "
                "uses the same world/pool and advances by another until interval; no whole-run ETA",
            "excluded_diagnostic_fields": DIAGNOSTIC_ONLY_FIELDS,
            "serial_timing": _stage_totals(serial), "serial_stages": serial,
            "serial_record": serial[-1]["record"], "variants": variants}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("reference", "world", "both"), default="reference")
    parser.add_argument("--workers", default="1,2")
    parser.add_argument("--until", type=float, default=.2)
    parser.add_argument("--fixture", choices=("traffic", "seeded_inner", "mixed_3ring"), default="traffic")
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0, help="0 means remaining 81 queries")
    parser.add_argument("--recovery-commit")
    args = parser.parse_args(argv)
    counts = [int(x) for x in args.workers.split(",")]
    if not counts or any(n < 1 for n in counts) or args.until < 0:
        parser.error("positive worker counts and nonnegative until required")
    if args.batch_start < 0 or args.batch_size < 0:
        parser.error("batch indices must be nonnegative")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()),
                                      *(sys.argv[1:] if argv is None else argv)])
    report = {"schema": "f67-parallel-probe-1.0", "protocol": "PROBE_NOT_FORMAL_OR_G09",
        "status": "RUNNING", "command": command,
        "provenance": recovery._current_provenance(command, recovery.FROZEN_COMMIT),
        "execution": execution_identity(args.recovery_commit, counts),
        "comparison": "exact physical metrics/states/events/rejections/plan payload; "
            "only named CPU/cache diagnostics excluded; no speed or stage acceptance gate"}
    write_json(root/"probe.json", report)
    try:
        if args.mode in ("reference", "both"):
            all_queries = representative_queries()
            end = args.batch_start+args.batch_size if args.batch_size else None
            queries = all_queries[args.batch_start:end]
            if not queries:
                raise ValueError("empty reference batch")
            report["reference"] = reference_probe(queries, counts)
            report["reference_coverage"] = {"total_representative_queries": len(all_queries),
                "batch_start": args.batch_start, "batch_count": len(queries),
                "all_3_rings_24_phase_bins_and_1235_slots": queries == all_queries}
        if args.mode in ("world", "both"):
            report["world"] = world_probe(args.until, counts, args.fixture)
        mismatches = []
        for variant in report.get("reference", {}).get("variants", []):
            for row in variant["passes"]:
                mismatches.extend(row["differences"])
        for variant in report.get("world", {}).get("variants", []):
            mismatches.extend(variant["differences"])
        report.update(status="MATCH" if not mismatches else "MISMATCH", differences=mismatches)
        write_json(root/"probe.json", report)
        print(json.dumps({"status": report["status"], "output": str(root),
                          "differences": len(mismatches)}))
        return 0 if not mismatches else 1
    except Exception as exc:
        report.update(status="FAILED", failure={"type": type(exc).__name__, "message": str(exc)})
        write_json(root/"probe.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
