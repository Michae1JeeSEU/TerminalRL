"""Explicit bounded profiling and passive wall/process-CPU measurement.

World.cpu uses perf_counter in R0 and must never be summed as CPU. Process-tree
CPU is deliberately unavailable here until a supervisor supplies child totals.
"""
from __future__ import annotations

from contextlib import contextmanager
import cProfile
import faulthandler
from pathlib import Path
import threading
import time

from .checkpoint import atomic_write_json


class Measurements:
    def __init__(self, mode="benchmark", *, wall_clock=time.perf_counter, cpu_clock=time.process_time):
        if mode not in ("benchmark", "profile"):
            raise ValueError("mode must be benchmark or profile")
        self.mode = mode
        self.wall_clock, self.cpu_clock = wall_clock, cpu_clock
        self.started_wall, self.started_cpu = wall_clock(), cpu_clock()
        self.spans = []
        self.counters = {}
        self.profile_files = []

    def count(self, name, amount=1):
        self.counters[name] = self.counters.get(name, 0) + amount

    @contextmanager
    def span(self, name):
        wall, cpu = self.wall_clock(), self.cpu_clock()
        try:
            yield
        finally:
            self.spans.append({"name": name, "wall_s": self.wall_clock() - wall,
                               "parent_process_cpu_s": self.cpu_clock() - cpu,
                               "nested_spans_are_not_additive": True})

    @contextmanager
    def profile_batch(self, output_path):
        """Only explicitly delimited task batches/windows are profiled."""
        if self.mode != "profile":
            raise ValueError("benchmark cannot enable cProfile")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        profiler = cProfile.Profile(timer=time.process_time)
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            profiler.dump_stats(str(path))
            self.profile_files.append(str(path.resolve()))

    def snapshot(self):
        return {"mode": self.mode, "end_to_end_wall_s": self.wall_clock() - self.started_wall,
                "parent_process_cpu_s": self.cpu_clock() - self.started_cpu,
                "process_tree_cpu_s": None, "process_tree_cpu_coverage": "parent_only_no_child_accounting",
                "spans": list(self.spans), "counters": dict(self.counters),
                "profile_files": list(self.profile_files), "profile_timer": "time.process_time"}


class PassiveHeartbeat:
    """Independent observer thread; never reads or advances live world state.

    The execution thread publishes a small immutable snapshot after safe cuts
    and when entering a slow query. A heartbeat can then report stalled work.
    """
    def __init__(self, path, *, interval_s=30., slow_s=300., stack_s=1800., clock=time.perf_counter):
        if not 0 < interval_s <= slow_s <= stack_s:
            raise ValueError("invalid heartbeat intervals")
        self.path = Path(path)
        self.interval_s, self.slow_s, self.stack_s = interval_s, slow_s, stack_s
        self.clock = clock
        self.started = self.last_progress = clock()
        self.last_complete_sim_time = None
        self.context = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._stack_dumped = False
        self.observer_error = None

    def publish(self, *, complete_sim_time=None, context=None):
        with self._lock:
            if complete_sim_time is not None and complete_sim_time != self.last_complete_sim_time:
                self.last_complete_sim_time = complete_sim_time
                self.last_progress = self.clock()
                self._stack_dumped = False
            if context is not None:
                self.context = dict(context)

    def emit(self):
        with self._lock:
            now = self.clock()
            idle = now - self.last_progress
            state = {"elapsed_wall_s": now - self.started,
                     "last_complete_sim_time": self.last_complete_sim_time,
                     "no_completed_step_wall_s": idle, "active_context": dict(self.context),
                     "status": "SLOW_OPERATION" if idle >= self.slow_s else "RUNNING"}
            if idle >= self.stack_s and not self._stack_dumped:
                stack_path = self.path.with_suffix(".stacks.txt")
                stack_path.parent.mkdir(parents=True, exist_ok=True)
                with stack_path.open("a") as output:
                    faulthandler.dump_traceback(file=output, all_threads=True)
                state["parent_stack_path"] = str(stack_path.resolve())
                state["child_stack_coverage"] = "not_available_requires_worker_supervisor"
                self._stack_dumped = True
        atomic_write_json(self.path, state)
        return state

    def start(self):
        if self._thread is not None:
            raise RuntimeError("heartbeat already started")
        self.emit()
        def work():
            while not self._stop.wait(self.interval_s):
                try:
                    self.emit()
                except Exception as exc:
                    self.observer_error = exc
                    self._stop.set()
        self._thread = threading.Thread(target=work, name="performance-heartbeat", daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1., self.interval_s))
        self.check_health()

    def check_health(self):
        """Execution runner calls this at cuts; errors remain instrumentation errors."""
        if self.observer_error is not None:
            raise RuntimeError("passive heartbeat observer failed") from self.observer_error

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
