from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import threading
import urllib.error
import urllib.request

import pytest

from terminal_marl.visualization.console_server import (
    ConsoleInputError,
    ConsoleSession,
    ConsoleStepObserver,
    MAX_FRAMES,
    build_server,
    load_replay,
)


def _request(server, method: str, path: str, *, body=None, token: str | None = None):
    host, port = server.server_address[:2]
    headers = {"Host": f"127.0.0.1:{port}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["X-Terminal-Token"] = token
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers=headers, method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _synthetic_error_worker(_config, _frames, statuses, stop_event, _run_event, _step, _speed):
    statuses.put({"kind": "error", "error": "synthetic worker error"})
    stop_event.wait(0.2)


@pytest.fixture()
def running_server():
    session = ConsoleSession(token="test-token")
    server = build_server(port=0, session=session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, session
    finally:
        session.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_session_token_and_invalid_start_do_not_stop_existing_state(running_server) -> None:
    server, session = running_server
    status, payload = _request(server, "GET", "/api/session")
    assert status == 200
    assert payload["token"] == "test-token"

    status, payload = _request(
        server, "POST", "/api/start", body={"mode": "fixed_rule", "max_steps": 0}, token="test-token",
    )
    assert status == 400
    assert payload["error"] == "invalid_request"
    assert session.state()["status"] == "idle"

    status, payload = _request(server, "POST", "/api/control", body={"action": "stop"}, token="wrong")
    assert status == 403
    assert payload["error"] == "invalid_token"

    status, payload = _request(server, "GET", "/../AGENTS.md")
    assert status == 404
    assert "AGENTS" not in json.dumps(payload)


def test_frames_generation_cursor_and_batch_limit() -> None:
    session = ConsoleSession(token="test-token")
    try:
        with session._lock:
            session._generation = 3
            session._frames = deque(
                [{"time": float(i), "step_count": i} for i in range(250)], maxlen=MAX_FRAMES,
            )
        first = session.frames_payload(after=0, generation=3)
        assert len(first["frames"]) == 200
        assert first["next_cursor"] == 200
        restart = session.frames_payload(after=200, generation=2)
        assert restart["frames"][0]["step_count"] == 0
        assert restart["generation"] == 3
    finally:
        session.close()


def test_replay_json_and_legacy_html_are_data_only(tmp_path: Path) -> None:
    frames = [{"time": 0.0, "step_count": 0}, {"time": 1.0, "step_count": 1}]
    json_path = tmp_path / "replay.json"
    json_path.write_text(json.dumps({"title": "json", "frames": frames}), encoding="utf-8")
    title, loaded, _ = load_replay(json_path)
    assert title == "json"
    assert loaded == frames

    html_path = tmp_path / "replay.html"
    html_path.write_text(
        "<script>window.__TERMINAL_REPLAY__ = "
        + json.dumps({"title": "html", "frames": frames})
        + ";</script>",
        encoding="utf-8",
    )
    title, loaded, _ = load_replay(html_path)
    assert title == "html"
    assert loaded == frames


def test_replay_rejects_nonfinite_time_bad_order_and_bad_point(tmp_path: Path) -> None:
    bad_values = [
        {"frames": [{"time": float("nan"), "step_count": 0}]},
        {"frames": [{"time": 1.0, "step_count": 1}, {"time": 0.0, "step_count": 2}]},
        {"frames": [{"time": 0.0, "step_count": 0, "uavs": [{"uid": 1, "position": [0.0, 0.0, float("inf")]}]}]},
    ]
    for index, payload in enumerate(bad_values):
        path = tmp_path / f"bad-{index}.json"
        text = json.dumps(payload, allow_nan=True)
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConsoleInputError):
            load_replay(path)


def test_replay_path_api_token_reference_and_atomic_failure(running_server, tmp_path: Path) -> None:
    server, session = running_server
    path = tmp_path / "recorded.json"
    path.write_text(json.dumps({"title": "recorded", "frames": [{"time": 0., "step_count": 0}]}), encoding="utf-8")
    assert _request(server, "POST", "/api/replay", body={"path": str(path)}, token="wrong")[0] == 403
    status, state = _request(server, "POST", "/api/replay", body={"path": str(path)}, token="test-token")
    assert status == 200 and state["status"] == "replay" and state["generation"] == 1
    original = session.export_payload()
    status, reference = _request(server, "POST", "/api/replay", body={"path": str(path), "reference": True}, token="test-token")
    assert status == 200 and reference["title"] == "recorded" and len(reference["frames"]) == 1
    assert session.export_payload() == original
    status, _ = _request(server, "POST", "/api/replay", body={"path": str(tmp_path / "missing")}, token="test-token")
    assert status == 400 and session.export_payload() == original
    status, _ = _request(server, "POST", "/api/replay", body={"path": str(path), "reference": "yes"}, token="test-token")
    assert status == 400 and session.export_payload() == original


def test_console_observer_does_not_require_second_permit_at_max_step() -> None:
    frame_queue = Queue()
    status_queue = Queue()
    stop_event = threading.Event()
    run_event = threading.Event()
    step_semaphore = threading.Semaphore(0)
    speed = SimpleNamespace(value=0.0)
    env = SimpleNamespace(
        now=0.0,
        step_count=0,
        active={},
        service_uavs={},
        arrivals=SimpleNamespace(exhausted=lambda: False),
        admission=SimpleNamespace(queue=SimpleNamespace(items=[])),
        config=SimpleNamespace(dt=1.0, max_simulation_seconds=10.0, high_level_enabled=False),
    )
    # The observer's initial snapshot needs a real snapshot-producing object;
    # use a minimal fake by replacing its emitter after construction.
    observer = object.__new__(ConsoleStepObserver)
    observer._env = env
    observer._run_config = {"max_steps": 1}
    observer.config = SimpleNamespace(trajectory_max_points=30)
    observer._frame_queue = frame_queue
    observer._status_queue = status_queue
    observer._stop_event = stop_event
    observer._run_event = run_event
    observer._step_semaphore = step_semaphore
    observer._speed_value = speed
    observer._closed = False
    observer._error = None
    observer._last_step = -1
    observer._base_step_limit = 1
    observer._emit = lambda _env: None

    env.step_count = 1
    observer._wait_for_permission = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected wait"))
    observer.__call__(env)


def test_console_observer_initial_pause_and_single_permit_per_tick(monkeypatch) -> None:
    frame_queue = Queue()
    status_queue = Queue()
    stop_event = threading.Event()
    run_event = threading.Event()
    step_semaphore = threading.Semaphore(0)
    speed = SimpleNamespace(value=0.0)
    env = SimpleNamespace(
        now=0.0,
        step_count=0,
        active={},
        service_uavs={},
        arrivals=SimpleNamespace(exhausted=lambda: False),
        admission=SimpleNamespace(queue=SimpleNamespace(items=[])),
        config=SimpleNamespace(dt=1.0, max_simulation_seconds=10.0, high_level_enabled=False),
    )
    emitted = []

    def fake_emit(self, current_env, *, final=False):
        emitted.append((current_env.step_count, final))
        self._last_step = current_env.step_count

    monkeypatch.setattr(ConsoleStepObserver, "_emit", fake_emit)
    observer = ConsoleStepObserver(
        env,
        SimpleNamespace(),
        {"max_steps": 2},
        frame_queue,
        status_queue,
        stop_event,
        run_event,
        step_semaphore,
        speed,
    )
    assert emitted == [(0, False)]

    ready = []
    ready_thread = threading.Thread(target=lambda: ready.append(observer.wait_ready()), daemon=True)
    ready_thread.start()
    ready_thread.join(timeout=0.05)
    assert ready == []
    step_semaphore.release()
    ready_thread.join(timeout=1.0)
    assert ready == [True]

    env.step_count = 1
    tick_one = threading.Thread(target=observer, args=(env,), daemon=True)
    tick_one.start()
    tick_one.join(timeout=0.05)
    assert tick_one.is_alive()
    step_semaphore.release()
    tick_one.join(timeout=1.0)
    assert not tick_one.is_alive()

    env.step_count = 2
    observer(env)
    assert emitted == [(0, False), (1, False), (2, False)]
    assert not step_semaphore.acquire(timeout=0.05)


def test_repeated_start_stop_with_injected_worker_cleans_terminal_state() -> None:
    session = ConsoleSession(token="test-token", worker_target=_synthetic_error_worker)
    try:
        session.start({"mode": "fixed_rule", "scenario": "default", "start_paused": True})
        deadline = threading.Event()
        for _ in range(50):
            if session.state()["status"] == "error":
                break
            deadline.wait(0.02)
        assert session.state()["status"] == "error"
        session.start({"mode": "fixed_rule", "scenario": "default", "start_paused": True})
        session.control({"action": "stop"})
        assert session.state()["status"] == "stopped"
        assert session.state()["process_alive"] is False
    finally:
        session.close()
