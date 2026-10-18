"""OS process accounting for an exclusively owned spawn-pool lifetime.

RSS is sampled contemporaneously, never a sum of process lifetime maxima.
The sampling maximum is a lower bound on the true instantaneous maximum.
"""
from __future__ import annotations

import ctypes
from collections import deque
import json
import os
from pathlib import Path
import sys
import threading
import time

_EXCLUSIVE = threading.Lock()


def _windows_memory(handle):
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    if not ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo(
            ctypes.c_void_p(handle), ctypes.byref(value), value.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo")
    return value.WorkingSetSize, value.PeakWorkingSetSize


def _rss(pid, *, peak=False):
    if sys.platform.startswith("linux"):
        if peak:
            line = next(x for x in Path(f"/proc/{pid}/status").read_text().splitlines()
                        if x.startswith("VmHWM:"))
            return int(line.split()[1])*1024
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), "OpenProcess RSS")
        try:
            return _windows_memory(handle)[int(peak)]
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    raise OSError("RSS accounting unsupported on this platform")


def _open_cpu_handle(pid):
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x1000 | 0x0400 | 0x0010, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "OpenProcess CPU")
    return handle


def _handle_cpu(handle):
    from ctypes import wintypes
    values = [wintypes.FILETIME() for _ in range(4)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel.GetProcessTimes(ctypes.c_void_p(handle), *(ctypes.byref(x) for x in values)):
        raise OSError(ctypes.get_last_error(), "GetProcessTimes")
    return sum((x.dwHighDateTime << 32) + x.dwLowDateTime for x in values[2:]) / 1e7


def _joined_cpu():
    import resource
    value = resource.getrusage(resource.RUSAGE_CHILDREN)
    return value.ru_utime + value.ru_stime


class OwnedProcessMetrics:
    """Start before spawn; register live workers; finish only after all joins.

    The caller must prevent unrelated child creation/reaping during this
    lifetime. Linux RUSAGE_CHILDREN is process-wide; the lock prevents two
    adapters in this process, not arbitrary third-party subprocess calls.
    """
    def __init__(self, *, exclusive, sample_interval=.1, defer_sampler=False, rss_series_path=None):
        if not exclusive:
            raise ValueError("exclusive owned-process accounting must be explicit")
        if not .01 <= sample_interval <= 5:
            raise ValueError("RSS sample interval must be in [.01, 5] seconds")
        if not _EXCLUSIVE.acquire(blocking=False):
            raise RuntimeError("another owned-process accounting context is active")
        self.interval = sample_interval
        self.start_wall, self.start_cpu = time.perf_counter(), time.process_time()
        self.pids, self.handles, self.errors = set(), {}, []
        self.samples = deque(maxlen=512)
        self.sample_count = self.sampled_peak = 0
        self._last_sample = -float("inf")
        self._sample_lock = threading.Lock()
        self.rss_series_path = str(Path(rss_series_path).resolve()) if rss_series_path else None
        self._series = None
        self._lock, self._stop = threading.Lock(), threading.Event()
        self.finished = False
        self.result = None
        self.children_start = None
        if sys.platform.startswith("linux"):
            self.children_start = _joined_cpu()
        elif os.name != "nt":
            self.errors.append("full child CPU unavailable on this platform")
        if self.rss_series_path:
            try:
                self._series = Path(self.rss_series_path).open("x", encoding="utf-8")
            except BaseException:
                _EXCLUSIVE.release()
                raise
        self._thread = threading.Thread(target=self._sample_loop, daemon=True,
                                        name="r0-pool-rss")
        self._sample()
        self._sampling_started = False
        if not defer_sampler:
            self.begin_sampling()

    def begin_sampling(self):
        if not self._sampling_started:
            self._thread.start()
            self._sampling_started = True

    def register(self, pids):
        changed = False
        with self._lock:
            for pid in pids:
                if pid in self.pids:
                    continue
                self.pids.add(pid)
                changed = True
                if os.name == "nt":
                    try:
                        self.handles[pid] = _open_cpu_handle(pid)
                    except OSError as exc:
                        self.errors.append(f"CPU PID {pid}: {exc}")
        if changed:
            self._sample()

    def _sample(self, *, force=True):
        with self._sample_lock:
            if not force and time.perf_counter()-self._last_sample < self.interval:
                return
            self._sample_locked()

    def _sample_locked(self):
        sample_start = time.perf_counter()
        with self._lock:
            pids = [os.getpid(), *sorted(self.pids)]
        values, unavailable = {}, []
        for pid in pids:
            try:
                values[str(pid)] = _rss(pid)
            except (OSError, ValueError, IndexError):
                unavailable.append(pid)
        row = {"elapsed_wall_seconds": time.perf_counter()-self.start_wall,
                             "sweep_wall_seconds": time.perf_counter()-sample_start,
                             "rss_bytes_by_pid": values, "unavailable_pids": unavailable,
                             "simultaneous_rss_bytes": sum(values.values())}
        self._last_sample = time.perf_counter()
        self.sample_count += 1
        self.sampled_peak = max(self.sampled_peak, row["simultaneous_rss_bytes"])
        self.samples.append(row)
        if self._series:
            self._series.write(json.dumps(row, separators=(",", ":"))+"\n")

    def _sample_loop(self):
        while not self._stop.wait(self.interval):
            self._sample(force=False)

    def finish(self, *, joined):
        if self.finished:
            return self.result
        try:
            self._stop.set()
            if self._sampling_started:
                self._thread.join()
            self._sample()
            child_cpu = None
            if not joined:
                self.errors.append("owned workers were not all joined")
            elif sys.platform.startswith("linux"):
                child_cpu = _joined_cpu()-self.children_start
            elif os.name == "nt":
                try:
                    if len(self.handles) != len(self.pids):
                        raise OSError("missing held process handle")
                    child_cpu = sum(_handle_cpu(h) for h in self.handles.values())
                except OSError as exc:
                    self.errors.append(str(exc))
            parent_cpu = time.process_time()-self.start_cpu
            rss_upper, rss_upper_source, rss_upper_error = None, None, None
            try:
                if not joined:
                    raise OSError("not all children joined")
                if sys.platform.startswith("linux"):
                    import resource
                    parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                    # ru_maxrss for CHILDREN is their maximum, not their sum.
                    # Multiplying by the owned worker count is conservative,
                    # including child teardown after the final live RSS sweep.
                    child_max = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024
                    rss_upper = parent_peak+len(self.pids)*child_max
                    rss_upper_source = "parent lifetime HWM + owned worker count * maximum reaped child HWM; " \
                        "past unrelated reaped children may make this looser"
                elif os.name == "nt":
                    peaks = [_windows_memory(h)[1] for h in self.handles.values()]
                    if len(peaks) != len(self.pids) or any(x <= 0 for x in peaks):
                        raise OSError("post-exit lifetime HWM unavailable")
                    rss_upper = _rss(os.getpid(), peak=True)+sum(peaks)
                    rss_upper_source = "sum of parent and exited owned worker OS lifetime HWM; NOT simultaneous RSS"
            except (OSError, ValueError) as exc:
                rss_upper_error = str(exc)
            self.result = {"wall_seconds": time.perf_counter()-self.start_wall,
                "parent_cpu_seconds": parent_cpu, "joined_children_cpu_seconds": child_cpu,
                "process_tree_cpu_seconds": parent_cpu+child_cpu if child_cpu is not None else None,
                "cpu_complete": joined and child_cpu is not None and not self.errors,
                "cpu_source": "RUSAGE_CHILDREN delta after join" if sys.platform.startswith("linux")
                    else "held OS handles/GetProcessTimes" if os.name == "nt" else "unavailable",
                "coverage": "parent + owned worker lifetimes: spawn, initializer, IPC, serialization, join; "
                    "requires no unrelated child creation/reaping; no resource_tracker is spawned by Linux fork",
                "rss_samples_recent": list(self.samples), "rss_sample_count": self.sample_count,
                "rss_recent_sample_capacity": self.samples.maxlen,
                "rss_full_series_path": self.rss_series_path,
                "rss_sample_interval_seconds": self.interval,
                "sampled_peak_simultaneous_rss_bytes": self.sampled_peak,
                "sum_lifetime_peak_upper_bound_bytes": rss_upper,
                "rss_upper_bound_source": rss_upper_source, "rss_upper_bound_error": rss_upper_error,
                "rss_limit_gate": "upper bound <= limit proves compliance; sampled lower bound > limit proves "
                    "failure; otherwise unresolved (lower bound below limit alone never proves compliance)",
                "rss_error": "lower bound; sequential per-PID reads within each synchronous sweep; "
                    "spikes between sweeps and inaccessible/exited PID RSS may be missed; no finite upper error bound",
                "worker_pids": sorted(self.pids), "errors": list(self.errors)}
            return self.result
        finally:
            if self._series:
                self._series.flush()
                self._series.close()
            for handle in self.handles.values():
                ctypes.WinDLL("kernel32").CloseHandle(ctypes.c_void_p(handle))
            self.finished = True
            _EXCLUSIVE.release()
