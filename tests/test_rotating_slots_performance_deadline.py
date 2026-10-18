"""Sacrificial, owned subprocess checks for unintegrated deadline primitives."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots.performance import deadline


ROOT = Path(__file__).resolve().parents[1]
LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="Linux kernel integration only")


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("-inf"), float("nan"),
                                     True, None, "invalid"])
def test_budget_validation(budget, tmp_path):
    with pytest.raises(ValueError, match="finite and positive"):
        deadline.LinuxWallDeadline(budget, tmp_path)
    assert not list(tmp_path.iterdir())


def test_unsupported_platform_does_not_write_or_arm(monkeypatch, tmp_path):
    monkeypatch.setattr(deadline.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="require Linux"):
        with deadline.LinuxWallDeadline(1, tmp_path):
            pytest.fail("unsupported platform entered")
    with pytest.raises(RuntimeError, match="require Linux"):
        deadline.bind_owned_worker_to_parent(123)
    assert not list(tmp_path.iterdir())


def _fake_linux(monkeypatch, timer=(0.0, 0.0), handler=0, blocked=False):
    calls = []
    fake = SimpleNamespace(
        SIGALRM=14, SIG_DFL=0, SIG_BLOCK=0, ITIMER_REAL=0,
        getsignal=lambda sig: handler,
        getitimer=lambda which: timer,
        pthread_sigmask=lambda how, mask: {14} if blocked else set(),
        signal=lambda *args: calls.append(("handler", args)),
        setitimer=lambda *args: calls.append(("timer", args)),
    )
    monkeypatch.setattr(deadline, "signal", fake)
    monkeypatch.setattr(deadline.sys, "platform", "linux")
    monkeypatch.setattr(deadline.multiprocessing, "parent_process", lambda: None)
    return calls


@pytest.mark.parametrize("timer", [(1.0, 0.0), (0.0, 2.0)])
def test_refuses_existing_timer_without_mutation(monkeypatch, tmp_path, timer):
    calls = _fake_linux(monkeypatch, timer=timer)
    with pytest.raises(RuntimeError, match="existing ITIMER_REAL"):
        deadline.LinuxWallDeadline(2, tmp_path).__enter__()
    assert calls == []
    assert not list(tmp_path.iterdir())


def test_refuses_existing_handler_and_blocked_signal(monkeypatch, tmp_path):
    calls = _fake_linux(monkeypatch, handler=object())
    with pytest.raises(RuntimeError, match="existing SIGALRM handler"):
        deadline.LinuxWallDeadline(2, tmp_path).__enter__()
    assert calls == []
    calls = _fake_linux(monkeypatch, blocked=True)
    with pytest.raises(RuntimeError, match="must be unblocked"):
        deadline.LinuxWallDeadline(2, tmp_path).__enter__()
    assert calls == []
    assert not list(tmp_path.iterdir())


def test_refuses_non_main_thread_and_process(monkeypatch, tmp_path):
    calls = _fake_linux(monkeypatch)
    observed = []

    def enter():
        try:
            deadline.LinuxWallDeadline(2, tmp_path).__enter__()
        except RuntimeError as exc:
            observed.append(str(exc))

    thread = threading.Thread(target=enter)
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert observed == ["deadline must be entered on the main thread"]
    monkeypatch.setattr(deadline.multiprocessing, "parent_process", lambda: object())
    with pytest.raises(RuntimeError, match="main process"):
        deadline.LinuxWallDeadline(2, tmp_path).__enter__()
    assert calls == []


def test_evidence_precedes_arming_and_exception_disarms(monkeypatch, tmp_path):
    calls = _fake_linux(monkeypatch)
    write = deadline.atomic_write_json

    def record_write(path, record):
        calls.append(("write", record["state"]))
        write(path, record)

    monkeypatch.setattr(deadline, "atomic_write_json", record_write)
    guard = deadline.LinuxWallDeadline(2, tmp_path)
    with pytest.raises(LookupError):
        with guard:
            record = json.loads(guard.evidence_path.read_text())
            assert record["pid"] == os.getpid()
            assert record["state"] == "ARMED"
            assert record["deadline_monotonic_seconds"] == record["start_monotonic_seconds"] + 2
            raise LookupError("body error")
    assert calls[0] == ("write", "ARMED")
    assert calls[-3:] == [("timer", (0, 0)), ("handler", (14, 0)), ("write", "DISARMED")]
    record = json.loads(guard.evidence_path.read_text())
    assert record["exit_kind"] == "exception"
    with pytest.raises(RuntimeError, match="cannot be reused"):
        guard.__enter__()


def test_persistence_failure_never_arms(monkeypatch, tmp_path):
    calls = _fake_linux(monkeypatch)

    def fail(*args):
        raise OSError("write unavailable")

    monkeypatch.setattr(deadline, "atomic_write_json", fail)
    with pytest.raises(OSError, match="write unavailable"):
        deadline.LinuxWallDeadline(2, tmp_path).__enter__()
    assert calls == []


def _python_command(body):
    prelude = "import sys; sys.dont_write_bytecode = True; sys.path.insert(0, " + repr(str(ROOT)) + ");\n"
    return [sys.executable, "-I", "-B", "-c", prelude + body]


def _stop_owned(process):
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=5)


def _wait_file(path, process, timeout=5):
    limit = time.monotonic() + timeout
    while time.monotonic() < limit:
        if path.exists():
            return
        assert process.poll() is None, "owned process exited before expected artifact"
        time.sleep(0.02)
    pytest.fail("owned process did not produce expected artifact within timeout")


@LINUX_ONLY
def test_normal_subprocess_cancels_timer_and_records_cleanup(tmp_path):
    code = f"""
import json, signal, time
from terminal_marl.core.rotating_slots.performance.deadline import LinuxWallDeadline
with LinuxWallDeadline(0.3, {str(tmp_path)!r}) as guard:
    pass
assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
time.sleep(0.5)
print(guard.evidence_path)
"""
    result = subprocess.run(_python_command(code), capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    paths = list(tmp_path.glob("deadline_*.json"))
    assert len(paths) == 1
    assert json.loads(paths[0].read_text())["state"] == "DISARMED"


@LINUX_ONLY
def test_unresponsive_compute_exits_by_sigalrm(tmp_path):
    code = f"""
import ctypes
from terminal_marl.core.rotating_slots.performance.deadline import LinuxWallDeadline
with LinuxWallDeadline(0.3, {str(tmp_path)!r}):
    # PyDLL keeps the GIL while this native computation runs.
    libc = ctypes.PyDLL(None)
    libc.strspn.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    libc.strspn.restype = ctypes.c_size_t
    payload = b'a' * 10000000
    while True:
        libc.strspn(payload, b'a')
"""
    started = time.monotonic()
    process = subprocess.Popen(_python_command(code), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == -signal.SIGALRM, (stdout, stderr)
        assert time.monotonic() - started < 8
        record = json.loads(next(tmp_path.glob("deadline_*.json")).read_text())
        assert record["pid"] == process.pid
        assert record["state"] == "ARMED"
        assert "PERFORMANCE_TIMEOUT" not in record.values()
    finally:
        _stop_owned(process)


@LINUX_ONLY
def test_unexpected_parent_fails_closed_in_calling_child(tmp_path):
    marker = tmp_path / "must_not_exist"
    code = f"""
import os
from pathlib import Path
from terminal_marl.core.rotating_slots.performance.deadline import bind_owned_worker_to_parent
bind_owned_worker_to_parent(os.getppid() + 1)
Path({str(marker)!r}).write_text('task executed')
"""
    result = subprocess.run(_python_command(code), capture_output=True, text=True, timeout=8)
    assert result.returncode == -signal.SIGKILL, result.stderr
    assert not marker.exists()


@LINUX_ONLY
def test_bound_worker_stops_after_parent_deadline_unrelated_sentinel_survives(tmp_path):
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        pytest.skip("identity-safe owned orphan cleanup requires Linux pidfd support")
    ready = tmp_path / "worker_ready.json"
    beat = tmp_path / "worker_beat"
    worker_code = f"""
import json, os, time
from pathlib import Path
from terminal_marl.core.rotating_slots.performance.deadline import bind_owned_worker_to_parent
bind_owned_worker_to_parent(int(sys.argv[1]))
Path({str(ready)!r}).write_text(json.dumps({{'pid': os.getpid()}}))
while True:
    Path({str(beat)!r}).write_text(str(time.monotonic()))
    time.sleep(0.02)
"""
    worker_command = _python_command(worker_code)
    parent_code = f"""
import os, subprocess
from terminal_marl.core.rotating_slots.performance.deadline import LinuxWallDeadline
with LinuxWallDeadline(2, {str(tmp_path / 'parent')!r}):
    worker = subprocess.Popen({worker_command!r} + [str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while True:
        pass
"""
    sentinel = subprocess.Popen(_python_command("import time; time.sleep(30)"),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    parent = None
    worker_fd = None
    try:
        parent = subprocess.Popen(_python_command(parent_code), stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        _wait_file(ready, parent)
        worker_pid = json.loads(ready.read_text())["pid"]
        worker_fd = os.pidfd_open(worker_pid)
        _wait_file(beat, parent)
        stdout, stderr = parent.communicate(timeout=8)
        assert parent.returncode == -signal.SIGALRM, (stdout, stderr)
        limit = time.monotonic() + 3
        state = None
        while time.monotonic() < limit:
            try:
                # A dead orphan may remain a zombie until the OS reaps it.
                state = Path(f"/proc/{worker_pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            except FileNotFoundError:
                state = None
            if state in (None, "Z", "X"):
                break
            time.sleep(0.02)
        assert state in (None, "Z", "X"), state
        last_beat = beat.read_bytes()
        time.sleep(0.15)
        assert beat.read_bytes() == last_beat
        assert sentinel.poll() is None
    finally:
        if worker_fd is not None:
            try:
                signal.pidfd_send_signal(worker_fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.close(worker_fd)
        if parent is not None:
            _stop_owned(parent)
        _stop_owned(sentinel)
