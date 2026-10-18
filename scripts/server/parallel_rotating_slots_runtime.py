"""Separate execution layer for f67 reference queries; never changes the core.

Only six independent pad candidates are evaluated in spawned processes. The
parent owns time, aircraft, calendars, submission, and every physical event.
"""
from __future__ import annotations

import cProfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _name in THREADS:
    os.environ[_name] = "1"

from terminal_marl.core.rotating_slots import diagnostics
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.qualification import write_json
from terminal_marl.core.rotating_slots.trajectory import generate_final, slot_state
from terminal_marl.core.rotating_slots.world import AIRBORNE, RotatingSlotWorld

_WORKER_WORLD = None
_WORKER_NATIVE_THREADS = None


def _worker_init(config_hash):
    global _WORKER_WORLD, _WORKER_NATIVE_THREADS
    diagnostics.limit_threads()
    config = load_config()
    if config.config_hash != config_hash:
        raise ValueError("worker configuration differs from parent")
    _WORKER_WORLD = RotatingSlotWorld(config=config)
    try:
        from threadpoolctl import threadpool_info
        _WORKER_NATIVE_THREADS = threadpool_info()
    except ImportError:
        _WORKER_NATIVE_THREADS = "UNAVAILABLE"


def _pad_candidate(world, query, pad):
    """Exact pad body of f67 World._reference, with worker-local caches."""
    ring, slot, t, version = query
    phase = (pad * math.pi / 3 - math.atan2(
        *slot_state(ring, slot, t, world.config).p[1::-1])) % (2 * math.pi)
    phase_bin = min(world.REFERENCE_PHASE_BINS - 1,
                    int(phase / (2 * math.pi) * world.REFERENCE_PHASE_BINS))
    duration = world._canonical_reference_duration(ring, phase_bin)
    result = None
    if duration is not None:
        world.reference_counts["template_instantiation"] += 1
        result = generate_final("reference", ring, slot, pad, t, emergency=True,
            source_version=version, config=world.config, durations=[duration])
    if result is None or result.plan is None or result.status != "FEASIBLE":
        world.reference_counts["fallback_search"] += 1
        result = generate_final("reference", ring, slot, pad, t, emergency=True,
                                source_version=version, config=world.config)
    if result.plan is not None and result.status == "FEASIBLE":
        q = world._energy(result.plan, t, result.plan.end_time)
        if q.valid:
            return q.soc_used, result.plan
    return None


def _pad_task(task):
    query, pad = task
    world = _WORKER_WORLD
    if world is None:
        raise RuntimeError("reference worker was not initialized")
    before = dict(world.reference_counts)
    profile = cProfile.Profile(timer=time.process_time)
    wall, cpu = time.perf_counter(), time.process_time()
    profile.enable()
    try:
        candidate = _pad_candidate(world, query, pad)
    finally:
        profile.disable()
    elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
    return {"query": query, "pad": pad, "candidate": candidate,
        "pid": os.getpid(), "cpu_seconds": elapsed_cpu, "wall_seconds": elapsed_wall,
        "peak_rss_bytes": diagnostics._rss()[0], "peak_rss_source": diagnostics._rss()[1],
        "components": diagnostics._component_profile(profile),
        "reference_counts": {k: world.reference_counts[k] - before[k] for k in before},
        "threads": {k: os.environ.get(k) for k in THREADS},
        "native_threadpools": _WORKER_NATIVE_THREADS}


class ReferencePool:
    """Persistent spawn pool. Task exceptions propagate; no serial fallback."""
    def __init__(self, workers, config=None, stats_path=None):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("inner workers must be a positive integer")
        self.workers = workers
        self.config = config or load_config()
        self.stats_path = Path(stats_path) if stats_path else None
        self.executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=mp.get_context("spawn"), initializer=_worker_init,
            initargs=(self.config.config_hash,))
        self.closed = False
        self.tasks = self.batches = 0
        self.failures = []
        self.worker_stats = {}
        self.category_cpu = {}
        self.functions = {}
        self.last_write = -math.inf

    def _record(self, row):
        self.tasks += 1
        worker = self.worker_stats.setdefault(row["pid"], {
            "pid": row["pid"], "tasks": 0, "cpu_seconds": 0., "task_wall_seconds": 0.,
            "peak_rss_bytes": None, "peak_rss_source": row["peak_rss_source"],
            "threads": row["threads"], "native_threadpools": row["native_threadpools"],
            "reference_counts": {}})
        worker["tasks"] += 1
        worker["cpu_seconds"] += row["cpu_seconds"]
        worker["task_wall_seconds"] += row["wall_seconds"]
        if row["peak_rss_bytes"] is not None:
            worker["peak_rss_bytes"] = max(worker["peak_rss_bytes"] or 0, row["peak_rss_bytes"])
        for k, v in row["reference_counts"].items():
            worker["reference_counts"][k] = worker["reference_counts"].get(k, 0) + v
        for k, v in row["components"]["exclusive_cpu_seconds"].items():
            self.category_cpu[k] = self.category_cpu.get(k, 0.) + v
        # The frozen profiler omits code line numbers. Same-name entries are
        # deliberately grouped, rather than presented as unique call sites.
        for f in row["components"]["functions"]:
            key = (f["file"], f["function"], f["category"])
            target = self.functions.setdefault(key, dict(file=f["file"],
                function=f["function"], category=f["category"], calls=0,
                exclusive_cpu_seconds=0., inclusive_cpu_seconds=0.))
            for name in ("calls", "exclusive_cpu_seconds", "inclusive_cpu_seconds"):
                target[name] += f[name]

    def references(self, queries):
        if self.closed:
            raise RuntimeError("reference pool is closed")
        queries = list(dict.fromkeys(tuple(q) for q in queries))
        tasks = [(q, pad) for q in queries for pad in range(6)]
        futures = [self.executor.submit(_pad_task, task) for task in tasks]
        choices = {q: [] for q in queries}
        self.batches += 1
        try:
            # Completion order cannot change pad ordering or min's tie break.
            for expected, future in zip(tasks, futures):
                row = future.result()
                if (row["query"], row["pad"]) != expected:
                    raise RuntimeError("reference task identity mismatch")
                self._record(row)
                if row["candidate"] is not None:
                    choices[row["query"]].append(row["candidate"])
        except BaseException as exc:
            self.failures.append({"type": type(exc).__name__, "message": str(exc)})
            for future in futures:
                future.cancel()
            self.close()
            raise
        self.flush_stats()
        return {q: min(choices[q], key=lambda x: x[0]) if choices[q] else (None, None)
                for q in queries}

    def summary(self):
        return {"inner_workers_requested": self.workers, "start_method": "spawn",
            "closed_and_joined": self.closed, "batches": self.batches, "pad_tasks": self.tasks,
            "worker_pids": sorted(self.worker_stats),
            "workers": [self.worker_stats[k] for k in sorted(self.worker_stats)],
            "child_task_cpu_seconds": sum(w["cpu_seconds"] for w in self.worker_stats.values()),
            "child_measured_components": {"exclusive_cpu_seconds": dict(self.category_cpu),
                "functions": sorted(self.functions.values(), key=lambda x: -x["exclusive_cpu_seconds"]),
                "classification": "same-name file/function grouped; inclusive values overlap",
                "scope": "cProfile enabled inside each completed pad task only"},
            "failures": list(self.failures),
            "measurement_scope": "child task CPU excludes spawn/initializer, result marshaling, "
                "idle time and profile-summary extraction; RSS is per-process lifetime peak, "
                "not simultaneous aggregate; failed task partial CPU is unavailable"}

    def flush_stats(self, force=False):
        if self.stats_path and (force or time.perf_counter() - self.last_write >= 30):
            write_json(self.stats_path, self.summary())
            self.last_write = time.perf_counter()

    def close(self):
        if not self.closed:
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            finally:
                self.closed = True
                self.flush_stats(force=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ParallelReferenceWorld(RotatingSlotWorld):
    def __init__(self, *args, reference_pool, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.cache_enabled:
            raise ValueError("parallel prefetch requires the frozen default exact cache")
        if self.config.config_hash != reference_pool.config.config_hash:
            raise ValueError("parent/pool config mismatch")
        self.reference_pool = reference_pool
        self.prefetched_queries = 0

    def _update_emergency(self):
        queries = []
        for key, a in sorted(tuple(self.aircraft.items())):
            if (a.status not in AIRBORNE or a.emergency.declared or
                    a.status.startswith("FINAL") or a.status == "VERTICAL_DESCENT"):
                continue
            p = self.active.get(key)
            if p is not None:
                if p.pad_id is not None:
                    continue
                ring, slot, t = p.target_ring, p.target_slot, p.end_time
            else:
                ring, slot, t = a.ring, a.slot, self.time
            query = (ring, slot, float(t), a.version)
            if (*query, self.config.profile_hash) not in self._q_cache:
                queries.append(query)
        for q, result in self.reference_pool.references(queries).items() if queries else ():
            self._store_cache(self._q_cache, (*q, self.config.profile_hash), result, self.Q_CACHE_CAPACITY)
            self.prefetched_queries += 1
        # Frozen serial mutation order, energy/SOC logic, events and resources.
        return super()._update_emergency()


def execution_identity(recovery_commit, inner_workers):
    base = Path(__file__).resolve().parent
    names = ("parallel_rotating_slots_runtime.py", "recover_rotating_slots_parallel.py",
             "recover_rotating_slots_diagnostics.py")
    return {"schema": "f67-parallel-reference-execution-1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "supplied_execution_commit": recovery_commit, "commit_verification": "external; no server Git",
        "implementation_files_sha256": {n: sha256((base/n).read_bytes()).hexdigest() for n in names},
        "inner_workers": inner_workers, "independent_jobs": 1,
        "core_commit": "f67b5c8462433ba701191c493a37e4a8e472334e",
        "scope": "separate execution implementation plus frozen core; reference pad candidates only",
        "parent_serial": ["time", "aircraft", "events", "resource submit", "deterministic decisions"],
        "accounting_changes": "reference template/cache counters and CPU are distributed; "
            "parent exact cache_hits include prefetch; not comparable to serial cache diagnostics"}


@contextmanager
def diagnostic_world_factory(pool):
    """Scoped injection in a single parent thread; never edits the frozen file."""
    from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink
    original = diagnostics._new_world
    def factory(root, **kwargs):
        sink = BufferedStateJSONLSink(root/"states.jsonl")
        try:
            return ParallelReferenceWorld(state_sink=sink, reference_pool=pool, **kwargs), sink
        except BaseException:
            sink.close()
            raise
    diagnostics._new_world = factory
    try:
        yield
    finally:
        diagnostics._new_world = original
