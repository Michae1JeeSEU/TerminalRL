"""Unintegrated Linux deadline primitives; no simulation or runner side effects."""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import threading
import time
import uuid

from .checkpoint import atomic_write_json


def _require_linux():
    if sys.platform != "linux" or not all(
        hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer", "getitimer")
    ):
        raise RuntimeError("deadline primitives require Linux signal timer support")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class LinuxWallDeadline:
    """Terminate this main process via kernel SIGALRM after a wall budget.

    Enter only before owned task work, without concurrent signal/timer mutation.
    No Python signal callback runs on expiry. An external exit collector must
    interpret exact SIGALRM termination together with this guard's ARMED record.
    """

    def __init__(self, budget_seconds, evidence_directory):
        if isinstance(budget_seconds, bool):
            raise ValueError("budget_seconds must be finite and positive")
        try:
            budget = float(budget_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("budget_seconds must be finite and positive") from exc
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("budget_seconds must be finite and positive")
        self.budget_seconds = budget
        self.evidence_directory = Path(evidence_directory)
        self.evidence_path = None
        self._record = None
        self._entered = False
        self._active = False
        self._pid = None
        self._previous_handler = None

    def __enter__(self):
        _require_linux()
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("deadline must be entered on the main thread")
        if multiprocessing.parent_process() is not None:
            raise RuntimeError("deadline must be entered in the main process")
        if self._entered:
            raise RuntimeError("deadline instances cannot be reused")
        if signal.getsignal(signal.SIGALRM) != signal.SIG_DFL:
            raise RuntimeError("refusing to replace an existing SIGALRM handler")
        if any(signal.getitimer(signal.ITIMER_REAL)):
            raise RuntimeError("refusing to replace an existing ITIMER_REAL timer")
        if signal.SIGALRM in signal.pthread_sigmask(signal.SIG_BLOCK, set()):
            raise RuntimeError("SIGALRM must be unblocked before deadline entry")

        self._entered = True
        self._pid = os.getpid()
        self._previous_handler = signal.getsignal(signal.SIGALRM)
        guard_id = uuid.uuid4().hex
        self.evidence_path = self.evidence_directory / ("deadline_" + guard_id + ".json")
        if self.evidence_path.exists():
            raise FileExistsError(self.evidence_path)
        started = time.monotonic()
        self._record = {
            "schema": "linux_wall_deadline_v1", "guard_id": guard_id,
            "state": "ARMED", "pid": self._pid, "start_utc": _utc_now(),
            "start_monotonic_seconds": started,
            "deadline_monotonic_seconds": started + self.budget_seconds,
            "budget_seconds": self.budget_seconds,
            "expiry_signal": int(signal.SIGALRM),
        }
        # Persist evidence before arming. Its cost is part of this wall budget.
        atomic_write_json(self.evidence_path, self._record)
        try:
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            remaining = self._record["deadline_monotonic_seconds"] - time.monotonic()
            signal.setitimer(signal.ITIMER_REAL, max(remaining, 1e-6))
            self._active = True
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous_handler)
            self._record.update(state="ARM_FAILED", end_utc=_utc_now())
            atomic_write_json(self.evidence_path, self._record)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if os.getpid() != self._pid or threading.current_thread() is not threading.main_thread():
            raise RuntimeError("only the owning main process/thread may exit the deadline")
        if self._active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous_handler)
            self._active = False
            self._record.update(
                state="DISARMED", end_utc=_utc_now(),
                end_monotonic_seconds=time.monotonic(),
                exit_kind="normal" if exc_type is None else "exception",
            )
            atomic_write_json(self.evidence_path, self._record)
        return False


def bind_owned_worker_to_parent(expected_parent_pid):
    """Bind ONLY the calling Linux child to an explicitly known parent.

    Call at the beginning of every owned worker initializer, before task work.
    An unexpected/replaced parent kills the calling child, closing the race
    around prctl registration. No other PID or process group is signalled.
    Normal parent paths must still join their children. This is not integrated
    into any current runner and does not recursively bind grandchildren.
    """
    _require_linux()
    if (isinstance(expected_parent_pid, bool)
            or not isinstance(expected_parent_pid, int) or expected_parent_pid <= 1):
        raise ValueError("expected_parent_pid must be a known parent PID greater than one")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("worker binding must run on the child's main thread")

    def require_expected_parent():
        if os.getppid() != expected_parent_pid:
            os.kill(os.getpid(), signal.SIGKILL)
            # SIGKILL cannot be caught; this guards against an unexpected mock.
            raise RuntimeError("unexpected parent; calling child must terminate")

    require_expected_parent()
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                      ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    require_expected_parent()
