"""Six-pad parallel execution adapter; the parent retains world ownership.

No core imports at module import time: spawn children select and verify the
frozen import root before importing the R0 numerical implementation.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
import importlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
POLICY_ID = "upstream_capacity_only_pad_arrival_v1_20261018"
_WORLD = _IDENTITY = _READY_BARRIER = _STARTUP = None
_PATCH_LOCK = threading.Lock()
_OPTIONAL_ACCELERATORS = (("energy", "root", "root_cache"),
                          ("energy", "peak", "peak_cache"),
                          ("trajectory", "bernstein", "_BERNSTEIN_METADATA"))


def _metrics_class():
    # Also works when this file is imported flat for isolated spawn startup.
    if __package__:
        return importlib.import_module(__package__+".process_metrics").OwnedProcessMetrics
    return importlib.import_module("process_metrics").OwnedProcessMetrics


def _assert_source(root, manifest):
    root = Path(root).resolve()
    if not manifest or "world.py" not in manifest or "trajectory.py" not in manifest:
        raise ValueError("core byte manifest must include world.py and trajectory.py")
    required = {p.name for p in root.glob("*.py")}
    if not required.issubset(manifest):
        raise ValueError("core byte manifest omits core Python source files")
    for name, expected in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or path.suffix != ".py":
            raise ValueError("invalid core manifest path")
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"core source hash mismatch: {name}")
    for name, module in tuple(sys.modules.items()):
        if name.startswith("terminal_marl.core.rotating_slots.") and ".performance" not in name:
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(root):
                raise ValueError(f"core import root mismatch: {name}: {location}")


def _config_identity(config):
    return {"config_hash": config.config_hash, "physical_config_bytes_sha256": config.raw_hash,
            "profile_hash": config.profile_hash}


def _implementation_identity():
    directory = Path(__file__).resolve().parent
    return {name: sha256((directory/name).read_bytes()).hexdigest()
            for name in ("parallel.py", "process_metrics.py", "deadline.py")}


def _deadline_module(identity):
    """Use the real package helper, including its relative checkpoint import."""
    directory = Path(__file__).resolve().parent
    helper_root = str(directory.parent)
    if helper_root not in sys.path:
        sys.path.insert(0, helper_root)
    module = importlib.import_module("performance.deadline")
    path = Path(module.__file__).resolve()
    if path != directory / "deadline.py" or sha256(path.read_bytes()).hexdigest() != identity["implementation_files_sha256"]["deadline.py"]:
        raise ValueError("worker deadline helper source identity mismatch")
    return module


def _worker_cache_snapshot():
    """Observe only the named pure numerical accelerators in this worker."""
    result = {}
    for module_name, names in (("trajectory", ("poly_bounds",)),
                               ("energy", ("_integrable_plan", "_segment_table")),
                               ("safety", ("_NUMERIC_CERTIFICATES", "_PLAN_CERTIFICATES"))):
        module = importlib.import_module("terminal_marl.core.rotating_slots." + module_name)
        for name in names:
            cache = getattr(module, name)
            key = module_name + "." + name
            if hasattr(cache, "cache_info") and hasattr(cache, "cache_clear"):
                result[key] = dict(cache.cache_info()._asdict(), present=True)
            elif type(cache) is dict:
                result[key] = {"present": True, "entries": len(cache)}
            else:
                raise RuntimeError("unexpected worker acceleration cache: " + key)
    for module_name, prefix, cache_name in _OPTIONAL_ACCELERATORS:
        module = importlib.import_module("terminal_marl.core.rotating_slots." + module_name)
        clear = getattr(module, prefix + "_cache_clear", None)
        observe = getattr(module, prefix + "_cache_stats", None)
        key = module_name + "." + cache_name
        if clear is None and observe is None:
            result[key] = {"present": False, "entries": 0}
        else:
            if not callable(clear) or not callable(observe):
                raise RuntimeError("optional worker cache requires clear/stats pair: " + key)
            value = observe()
            if not isinstance(value, dict) or type(value.get("entries")) is not int or value["entries"] < 0:
                raise RuntimeError("invalid optional worker cache observation: " + key)
            result[key] = dict(value, present=True)
    return result


def _clear_worker_accelerators():
    before = _worker_cache_snapshot()
    for module_name, names in (("trajectory", ("poly_bounds",)),
                               ("energy", ("_integrable_plan", "_segment_table")),
                               ("safety", ("_NUMERIC_CERTIFICATES", "_PLAN_CERTIFICATES"))):
        module = importlib.import_module("terminal_marl.core.rotating_slots." + module_name)
        for name in names:
            cache = getattr(module, name)
            if hasattr(cache, "cache_clear"):
                cache.cache_clear()
            else:
                cache.clear()
    for module_name, prefix, cache_name in _OPTIONAL_ACCELERATORS:
        module = importlib.import_module("terminal_marl.core.rotating_slots." + module_name)
        clear = getattr(module, prefix + "_cache_clear", None)
        if clear is not None:
            clear()
    after = _worker_cache_snapshot()
    if any(value.get("currsize", value.get("entries")) != 0 for value in after.values()):
        raise RuntimeError("worker numerical accelerator reset was not empty")
    for key in before:
        for flag in ("enabled", "policy", "max_entries", "max_bytes"):
            if flag in before[key] and after[key].get(flag) != before[key][flag]:
                raise RuntimeError("worker cache reset changed policy: " + key + "." + flag)
    return {"before": before, "after": after,
            "scope": "named pure numerical caches only; config/Gauss/canonical policy retained"}


def _worker_init(root, manifest, config_path, identity, cache_enabled, expected_parent_pid, ready_barrier):
    global _WORLD, _IDENTITY, _READY_BARRIER, _STARTUP
    sys.dont_write_bytecode = True
    for name in THREADS:
        os.environ[name] = "1"
    if _implementation_identity() != identity["implementation_files_sha256"]:
        raise ValueError("worker parallel helper source identity mismatch")
    inherited_core_modules = sum(name.startswith("terminal_marl.core.rotating_slots.")
        and ".performance" not in name for name in sys.modules)
    if sys.platform.startswith("linux"):
        deadline = _deadline_module(identity)
        deadline.bind_owned_worker_to_parent(expected_parent_pid)
        parent_death = {"status": "BOUND_TO_EXPECTED_PARENT", "supported": True,
                        "helper_sha256": identity["implementation_files_sha256"]["deadline.py"]}
    else:
        parent_death = {"status": "UNSUPPORTED_PLATFORM", "supported": False}
    # Binding precedes the worker's core-import statements. Fork may already
    # contain inherited core modules, which are explicitly reported below.
    sys.path.insert(0, str(Path(root).resolve().parents[2]))
    _assert_source(root, manifest)
    from terminal_marl.core.rotating_slots.config import load_config
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, POLICY_ID as actual_policy
    _assert_source(root, manifest)
    if actual_policy != POLICY_ID:
        raise ValueError("worker actual core policy differs from R0")
    config = load_config(config_path)
    if _config_identity(config) != identity["config"]:
        raise ValueError("worker configuration/profile identity mismatch")
    try:
        from threadpoolctl import threadpool_limits, threadpool_info
        # Keep controller alive for the full worker lifetime.
        globals()["_NATIVE_LIMITER"] = threadpool_limits(limits=1)
        globals()["_NATIVE_THREADPOOLS"] = threadpool_info()
        if any(row["num_threads"] != 1 for row in _NATIVE_THREADPOOLS):
            raise RuntimeError("worker native thread limit was not applied")
    except ImportError:
        if sys.platform.startswith("linux"):
            raise RuntimeError("Linux fork native thread verification requires threadpoolctl")
        globals()["_NATIVE_THREADPOOLS"] = "unavailable; environment set before numerical imports"
    cache_reset = _clear_worker_accelerators()
    _WORLD = RotatingSlotWorld(config=config, cache_enabled=cache_enabled)
    _IDENTITY = identity
    _READY_BARRIER = ready_barrier
    _STARTUP = {"expected_parent_pid": expected_parent_pid, "actual_parent_pid": os.getppid(),
        "dont_write_bytecode": sys.dont_write_bytecode,
        "isolated": bool(sys.flags.isolated),
        "parent_death": parent_death, "inherited_core_modules_before_binding": inherited_core_modules,
        "binding_before_worker_core_import_statements": True,
        "cache_reset": cache_reset, "cache_after_world_construction": _worker_cache_snapshot(),
        "canonical_phase_bins": _WORLD.REFERENCE_PHASE_BINS}


def _pad_task(task):
    if _WORLD is None or task["identity"] != _IDENTITY:
        raise ValueError("worker task semantic/source/profile identity mismatch")
    world = _WORLD
    query, pad = task["query"], task["pad"]
    ring, slot, t, version = query
    from terminal_marl.core.rotating_slots.trajectory import generate_final, slot_state
    before = dict(world.reference_counts)
    before_hits = world.cache_hits
    clock = time.perf_counter()
    profiler = None
    if task["mode"] == "profile":
        import cProfile
        profiler = cProfile.Profile(timer=time.process_time)
        profiler.enable()
    try:
        phase = (pad*math.pi/3-math.atan2(*slot_state(ring, slot, t, world.config).p[1::-1])) % (2*math.pi)
        phase_bin = min(world.REFERENCE_PHASE_BINS-1,
                        int(phase/(2*math.pi)*world.REFERENCE_PHASE_BINS))
        template_key = (ring, phase_bin, world.config.profile_hash)
        if task["known_template"][0]:
            world._reference_templates[template_key] = task["known_template"][1]
        duration = world._canonical_reference_duration(ring, phase_bin)
        result = None
        if duration is not None:
            world.reference_counts["template_instantiation"] += 1
            result = generate_final("reference", ring, slot, pad, t, emergency=True,
                source_version=version, config=world.config, durations=[duration])
        fallback = result is None or result.plan is None or result.status != "FEASIBLE"
        if fallback:
            world.reference_counts["fallback_search"] += 1
            result = generate_final("reference", ring, slot, pad, t, emergency=True,
                                    source_version=version, config=world.config)
        candidate = None
        if result.plan is not None and result.status == "FEASIBLE":
            q = world._energy(result.plan, t, result.plan.end_time)
            if q.valid:
                candidate = (q.soc_used, result.plan)
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(task["profile_path"])
    return {"query": query, "pad": pad, "identity": _IDENTITY,
        "candidate": candidate, "pid": os.getpid(), "phase_bin": phase_bin,
        "canonical_duration": duration, "fallback": fallback,
        "reference_counts": {k: world.reference_counts[k]-before[k] for k in before},
        "cache_hits": world.cache_hits-before_hits,
        "task_wall_seconds": time.perf_counter()-clock,
        "native_thread_limits": {name: os.environ.get(name) for name in THREADS}}


def _ready_task(identity):
    if _WORLD is None or identity != _IDENTITY or _READY_BARRIER is None or _STARTUP is None:
        raise ValueError("worker startup identity mismatch")
    # W tasks block until W distinct workers have entered; a fast worker cannot
    # report another startup before all parties arrive. Timeout propagates.
    _READY_BARRIER.wait(timeout=30)
    return dict(_STARTUP, pid=os.getpid(), native_threadpools=_NATIVE_THREADPOOLS)


def _spawn_module():
    """Give spawn an importable helper without requiring it inside frozen R0.

    A flat helper directory in sys.path is inherited by spawned children. No
    module-global core caches or sys.modules core replacement is performed.
    """
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    module = importlib.import_module("parallel")
    if Path(module.__file__).resolve() != Path(__file__).resolve():
        raise RuntimeError("flat parallel helper module name collision")
    return module


class ReferencePool:
    """Persistent pool, with explicit resource and provenance budgets.

    Every call submits exactly six pads for one exact cache miss. No prefetch,
    busy-pad filtering, exception-to-None conversion, or serial fallback.
    """
    def __init__(self, workers, *, config, expected_core_root, expected_core_manifest,
                 semantic_identity, process_budget, core_budget,
                 exclusive_process_accounting, mode="benchmark", profile_directory=None,
                 max_profile_tasks=6, rss_sample_interval=.1, rss_series_path=None, cache_enabled=True,
                 kernel="R0"):
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 6:
            raise ValueError("workers must be an integer in [1, 6]")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (process_budget, core_budget)):
            raise ValueError("explicit integer process/core budgets required")
        if process_budget < workers+1 or core_budget < workers:
            raise ValueError("worker allocation exceeds explicit process/core budget")
        if mode not in ("benchmark", "profile"):
            raise ValueError("mode must be benchmark or profile")
        if kernel not in ("R0", "OPT"):
            raise ValueError("kernel label must be R0 or OPT")
        if mode == "profile" and (profile_directory is None or not 1 <= max_profile_tasks <= 1000):
            raise ValueError("profile requires an output directory and bounded task count")
        required = {"policy_id", "effective_rules_hash", "candidate_policy_hash", "numerical_policy_hash"}
        if not required.issubset(semantic_identity) or semantic_identity["policy_id"] != POLICY_ID:
            raise ValueError("R0 semantic identity is missing or policy_id differs")
        if any(not semantic_identity[key] for key in required):
            raise ValueError("empty semantic hash")
        self.root = str(Path(expected_core_root).resolve())
        self.manifest = dict(expected_core_manifest)
        _assert_source(self.root, self.manifest)
        self.identity = {"semantic": dict(semantic_identity), "config": _config_identity(config),
                         "core_manifest": self.manifest, "core_root": self.root,
                         "implementation_files_sha256": _implementation_identity()}
        self.config, self.cache_enabled, self.workers = config, cache_enabled, workers
        self.kernel = kernel
        self.mode, self.max_profile_tasks = mode, max_profile_tasks
        self.profile_directory = Path(profile_directory).resolve() if profile_directory else None
        if self.profile_directory:
            self.profile_directory.mkdir(parents=True, exist_ok=True)
        self.process_budget, self.core_budget = process_budget, core_budget
        self.closed, self.joined, self.aborted = False, False, False
        self.tasks = self.batches = 0
        self.worker_counts = {"initial_search": 0, "fallback_search": 0, "template_instantiation": 0}
        self.worker_cache_hits = 0
        self.failures, self.worker_pids = [], set()
        self.last_rows = ()
        # Linux spawn starts an additional persistent resource_tracker process.
        # Fork keeps the strict parent+workers budget; it is only permitted
        # before any Python sampler/heartbeat/executor threads exist.
        self.start_method = "fork" if sys.platform.startswith("linux") else "spawn"
        if self.start_method == "fork" and threading.active_count() != 1:
            raise RuntimeError("Linux fork pool must start before sampler/heartbeat threads")
        if mp.active_children():
            raise RuntimeError("exclusive pool accounting requires no pre-existing multiprocessing children")
        self.metrics = _metrics_class()(exclusive=exclusive_process_accounting,
                                        sample_interval=rss_sample_interval, defer_sampler=True,
                                        rss_series_path=rss_series_path)
        self.resource_summary = None
        self.worker_startups = {}
        self._parent_native_limiter = None
        self._environment = {name: os.environ.get(name) for name in THREADS}
        for name in THREADS:
            os.environ[name] = "1"
        try:
            try:
                from threadpoolctl import threadpool_limits
                self._parent_native_limiter = threadpool_limits(limits=1)
            except ImportError:
                if self.start_method == "fork":
                    raise RuntimeError("Linux fork native thread verification requires threadpoolctl")
            module = _spawn_module()
            self._task = module._pad_task
            self.parent_pid = os.getpid()
            context = mp.get_context(self.start_method)
            ready_barrier = context.Barrier(workers)
            self.executor = ProcessPoolExecutor(max_workers=workers,
                mp_context=context, initializer=module._worker_init,
                initargs=(self.root, self.manifest, config.path, self.identity, cache_enabled,
                          self.parent_pid, ready_barrier))
            ready = []
            for _ in range(workers):
                ready.append(self.executor.submit(module._ready_task, self.identity))
                self._register_workers()
            self.metrics.begin_sampling()
            for future in ready:
                row = future.result(timeout=35)
                self.worker_startups[row["pid"]] = row
            self._register_workers()
            live = {p.pid for p in self.executor._processes.values() if p.is_alive()}
            if (len(self.worker_startups) != workers or
                    set(self.worker_startups) != self.worker_pids or self.worker_pids != live):
                raise RuntimeError("startup reports do not cover exactly the registered live workers")
        except BaseException:
            if hasattr(self, "executor"):
                self.abort()
            else:
                self.metrics.finish(joined=True)
            self._restore_environment()
            raise

    def _restore_environment(self):
        if self._parent_native_limiter is not None:
            self._parent_native_limiter.restore_original_limits()
            self._parent_native_limiter = None
        for name, value in self._environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _register_workers(self):
        # CPython executor process objects are retained until shutdown; capture
        # OS handles now so Windows CPU remains readable after process exit.
        processes = tuple(self.executor._processes.values())
        self.worker_pids.update(p.pid for p in processes)
        self.metrics.register(p.pid for p in processes)

    def pad_candidates(self, query, templates):
        if self.closed:
            raise RuntimeError("reference pool is closed")
        query = tuple(query)
        if len(query) != 4:
            raise ValueError("reference query requires ring, slot, exact time, source version")
        if self.mode == "profile" and self.tasks+6 > self.max_profile_tasks:
            raise RuntimeError("bounded profile task budget exceeded")
        from terminal_marl.core.rotating_slots.trajectory import slot_state
        futures = []
        self.batches += 1
        try:
            for pad in range(6):
                phase = (pad*math.pi/3-math.atan2(*slot_state(*query[:3], self.config).p[1::-1])) % (2*math.pi)
                phase_bin = min(23, int(phase/(2*math.pi)*24))
                key = (query[0], phase_bin, self.config.profile_hash)
                task = {"query": query, "pad": pad, "identity": self.identity, "mode": self.mode,
                        "known_template": (key in templates, templates.get(key))}
                if self.mode == "profile":
                    path = self.profile_directory/f"batch_{self.batches:06d}_pad_{pad}.prof"
                    if path.exists():
                        raise FileExistsError(path)
                    task["profile_path"] = str(path)
                futures.append(self.executor.submit(self._task, task))
                self._register_workers()
            rows = []
            # Future order, never completion order, preserves min's tie break.
            for pad, future in enumerate(futures):
                row = future.result()
                if row["query"] != query or row["pad"] != pad or row["identity"] != self.identity:
                    raise RuntimeError("reference result query/pad/source/profile identity mismatch")
                rows.append(row)
                self.tasks += 1
                for key, value in row["reference_counts"].items():
                    self.worker_counts[key] += value
                self.worker_cache_hits += row.get("cache_hits", 0)
            self.last_rows = tuple(rows)
            return self.last_rows
        except BaseException as exc:
            self.failures.append({"type": type(exc).__name__, "message": str(exc), "query": query})
            for future in futures:
                future.cancel()
            self.abort()
            raise

    def summary(self):
        return {"schema": "r0-pad-parallel-1.0", "execution_backend": self.kernel + "_PARALLEL",
            "prefix_execution": self.kernel + "_PARALLEL", "kernel": self.kernel, "profiler_mode": self.mode,
            "profiler_timer": "process_time" if self.mode == "profile" else None,
            "workers": self.workers, "process_budget": self.process_budget, "core_budget": self.core_budget,
            "native_thread_limits": {name: "1" for name in THREADS}, "start_method": self.start_method,
            "identity": self.identity, "closed_and_joined": self.closed and self.joined,
            "hard_aborted": self.aborted, "pad_tasks_completed": self.tasks, "query_batches": self.batches,
            "actual_worker_reference_counts": dict(self.worker_counts),
            "actual_worker_cache_hits": self.worker_cache_hits,
            "worker_pids": sorted(self.worker_pids), "failures": list(self.failures),
            "observed_worker_startups": list(self.worker_startups.values()),
            "expected_parent_pid": self.parent_pid,
            "worker_startup_coverage_complete": len(self.worker_startups) == self.workers
                and set(self.worker_startups) == self.worker_pids,
            "resources": self.resource_summary,
            "accounting_note": "worker template initialization and cache reuse are actual counts; "
                "they need not equal serial work; no per-function table in benchmark"}

    def close(self):
        if self.closed:
            return
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.joined = True
        finally:
            self.closed = True
            self._restore_environment()
            self.resource_summary = self.metrics.finish(joined=self.joined)

    def abort(self):
        """Hard abort only this executor's processes, then join all of them."""
        if self.closed:
            return
        self.aborted = True
        self._register_workers()
        processes = tuple(self.executor._processes.values())
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        if typ is not None:
            self.abort()
        else:
            self.close()


@contextmanager
def install_parallel_reference(world, pool):
    """Scoped class patch: no extra fields in checkpoint-authoritative world.

    The original emergency loop, cache-hit branch, exact cache store, query
    order, pad order and last-fallback diagnostic retain their R0 semantics.
    """
    if _config_identity(world.config) != pool.identity["config"] or world.cache_enabled != pool.cache_enabled:
        raise ValueError("parent/pool configuration or cache mode mismatch")
    _assert_source(pool.root, pool.manifest)
    actual_module = importlib.import_module("terminal_marl.core.rotating_slots.world")
    if actual_module.POLICY_ID != POLICY_ID or world.REFERENCE_PHASE_BINS != 24:
        raise ValueError("parent actual core policy/candidate definition differs from R0")
    if not _PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("another parallel world binding is active")
    cls = type(world)
    original = cls._reference
    def reference(self, ring, slot, t, version=0):
        if self is not world:
            return original(self, ring, slot, t, version)
        key = (ring, slot, float(t), version, self.config.profile_hash)
        if self.cache_enabled and key in self._q_cache:
            self.cache_hits += 1
            return self._q_cache[key]
        rows = pool.pad_candidates((ring, slot, t, version), self._reference_templates)
        choices = []
        for row in rows:
            self._reference_templates[(ring, row["phase_bin"], self.config.profile_hash)] = row["canonical_duration"]
            for name, count in row["reference_counts"].items():
                self.reference_counts[name] += count
            if row["fallback"]:
                self._last_reference_fallback = {"ring": ring, "slot": slot, "query_time": t,
                    "pad_id": row["pad"], "phase_bin": row["phase_bin"],
                    "canonical_duration": row["canonical_duration"],
                    "reason": "canonical template missing or invalid at exact current phase"}
            if row["candidate"] is not None:
                choices.append(row["candidate"])
        result = min(choices, key=lambda value: value[0]) if choices else (None, None)
        if self.cache_enabled:
            self._store_cache(self._q_cache, key, result, self.Q_CACHE_CAPACITY)
        return result
    cls._reference = reference
    try:
        yield world
    finally:
        cls._reference = original
        _PATCH_LOCK.release()
