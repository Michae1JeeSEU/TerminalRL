from __future__ import annotations

import pickle
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np

import terminal_marl.visualization.pyvista_viewer as viewer_module
from terminal_marl.core.config import SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.visualization.config import VisualizationConfig
from terminal_marl.visualization.pyvista_viewer import (
    BufferedProcessTerminalViewerObserver,
    _buffered_process_render_main,
    _process_transition_seconds,
    _snapshot_for_process,
)
from terminal_marl.visualization.schema import RingRenderState, WorldRenderSnapshot
from terminal_marl.visualization.snapshot import build_world_render_snapshot


def _config() -> SimConfig:
    return SimConfig(
        arrival_model="finite_cohort",
        fixed_arrival_schedule=(0.0, 0.0, 0.0),
        total_requested_uavs=3,
        reset_mode="warm_start_rings",
        warm_start_uavs=3,
        max_simulation_seconds=20.0,
    )


def _state(env: TerminalParallelEnv) -> tuple:
    return (
        env.now,
        tuple(
            (
                uid,
                uav.phase.value,
                tuple(float(value) for value in uav.position),
                float(uav.energy_remaining),
            )
            for uid, uav in sorted(env.active.items())
        ),
        tuple(sorted(env.ledger.outcomes.items())),
        dict(env.telemetry),
    )


def test_process_snapshot_uses_pickleable_plain_telemetry() -> None:
    env = TerminalParallelEnv(_config())
    env.reset(7)
    snapshot = build_world_render_snapshot(env, trajectory_max_points=20)
    copied = _snapshot_for_process(snapshot)

    assert isinstance(copied.telemetry, dict)
    assert pickle.loads(pickle.dumps(copied)) == copied
    assert snapshot.telemetry == copied.telemetry


def test_process_transition_speed_mapping_is_bounded_and_consistent() -> None:
    previous = type("Snapshot", (), {"time": 4.0})()
    current = type("Snapshot", (), {"time": 5.0})()

    assert _process_transition_seconds(previous, current, 0.0) == 0.30
    assert _process_transition_seconds(previous, current, 2.0) == 0.5
    assert _process_transition_seconds(previous, current, 100.0) >= 1e-3


def test_spawn_process_observer_preserves_simulator_state(monkeypatch) -> None:
    monkeypatch.setenv("PYVISTA_OFF_SCREEN", "true")
    reference = TerminalParallelEnv(_config())
    observed = TerminalParallelEnv(_config())
    reference.reset(17)
    observed.reset(17)
    initial = build_world_render_snapshot(observed, trajectory_max_points=20)
    observer = BufferedProcessTerminalViewerObserver(
        initial,
        VisualizationConfig(frame_stride=1, trajectory_max_points=20),
    )
    try:
        assert observer.wait_ready(timeout=20.0), observer.error
        for _ in range(3):
            action = {
                str(uid): np.zeros(3, dtype=np.float32)
                for uid in reference.active
            }
            reference.step(action)
            observed.step(action)
            observer(observed)
            assert observer.error is None
    finally:
        observer.close()

    assert observer.stop_requested
    assert observer.error is None
    assert _state(observed) == _state(reference)


def test_spawn_process_exception_reports_and_cleans_up(monkeypatch) -> None:
    monkeypatch.setenv("PYVISTA_OFF_SCREEN", "true")
    invalid = WorldRenderSnapshot(
        time=0.0,
        step_count=0,
        rings=(RingRenderState("invalid", 1.0, 1.0, 1.0, 1.0, 1, 0),),
        gates=(),
        landing_zones=(),
        uavs=(),
        external_queue_size=0,
        service_count=0,
        completed_count=0,
        transition_active=False,
        transition_progress=0.0,
        telemetry={},
        recent_conflict_uids=(),
    )
    observer = BufferedProcessTerminalViewerObserver(invalid, VisualizationConfig())
    try:
        assert not observer.wait_ready(timeout=20.0)
        assert observer.error is not None
    finally:
        observer.close()
    assert observer.stop_requested
    assert not observer._process.is_alive()


def test_process_start_failure_close_skips_unstarted_process_join(monkeypatch) -> None:
    created = []

    class FailedProcess:
        def __init__(self, **_kwargs):
            self.join_calls = 0
            self.terminate_calls = 0
            created.append(self)

        def start(self):
            raise RuntimeError("synthetic process start failure")

        def join(self, timeout=None):
            self.join_calls += 1

        def is_alive(self):
            raise AssertionError("is_alive must not be called for an unstarted process")

        def terminate(self):
            self.terminate_calls += 1

    class FakeContext:
        Queue = staticmethod(lambda maxsize=0: queue.Queue(maxsize=maxsize))
        Event = staticmethod(threading.Event)
        Value = staticmethod(lambda _kind, value: SimpleNamespace(value=value))
        Process = staticmethod(lambda **kwargs: FailedProcess(**kwargs))

    monkeypatch.setattr(viewer_module.mp, "get_context", lambda _name: FakeContext())
    observer = BufferedProcessTerminalViewerObserver(
        WorldRenderSnapshot(
            time=0.0,
            step_count=0,
            rings=(),
            gates=(),
            landing_zones=(),
            uavs=(),
            external_queue_size=0,
            service_count=0,
            completed_count=0,
            transition_active=False,
            transition_progress=0.0,
            telemetry={},
            recent_conflict_uids=(),
        ),
        VisualizationConfig(),
    )
    observer.close()

    assert created[0].join_calls == 0
    assert created[0].terminate_calls == 0
    assert observer.stop_requested
    assert observer.error is not None


def test_unexpected_process_exit_sets_stop_and_error(monkeypatch) -> None:
    class ExitedProcess:
        exitcode = 17

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

        def terminate(self):
            return None

    class FakeContext:
        Queue = staticmethod(lambda maxsize=0: queue.Queue(maxsize=maxsize))
        Event = staticmethod(threading.Event)
        Value = staticmethod(lambda _kind, value: SimpleNamespace(value=value))
        Process = staticmethod(lambda **kwargs: ExitedProcess())

    monkeypatch.setattr(viewer_module.mp, "get_context", lambda _name: FakeContext())
    observer = BufferedProcessTerminalViewerObserver(
        WorldRenderSnapshot(
            time=0.0,
            step_count=0,
            rings=(),
            gates=(),
            landing_zones=(),
            uavs=(),
            external_queue_size=0,
            service_count=0,
            completed_count=0,
            transition_active=False,
            transition_progress=0.0,
            telemetry={},
            recent_conflict_uids=(),
        ),
        VisualizationConfig(),
    )
    try:
        assert observer.stop_requested
        assert observer.error is not None
        assert "exitcode=17" in str(observer.error)
    finally:
        observer.close()


def test_short_rollout_starts_after_bounded_prefill_timeout(monkeypatch) -> None:
    class FakeViewer:
        instances = []

        def __init__(self, _initial, _config):
            self.paused = False
            self.playback_speed = 0.0
            self.closed = False
            self.source_updates = []
            self.interpolated_updates = 0
            self.instances.append(self)

        def update(self, snapshot, *, uav_positions=None, render=True):
            self.source_updates.append(snapshot.step_count)
            return True

        def update_dynamic(self, snapshot, positions, *, render=True):
            self.source_updates.append(snapshot.step_count)
            return True

        def update_interpolated_positions(self, _snapshot, _positions):
            self.interpolated_updates += 1
            return True

        def wait_until_resumed(self, _stop_event=None):
            return True

        def close(self):
            self.closed = True

    monkeypatch.setattr(viewer_module, "Terminal3DViewer", FakeViewer)
    env = TerminalParallelEnv(_config())
    env.reset(13)
    initial = build_world_render_snapshot(env, trajectory_max_points=20)
    env.step({str(uid): np.zeros(3, dtype=np.float32) for uid in env.active})
    first = build_world_render_snapshot(env, trajectory_max_points=20)
    env.step({str(uid): np.zeros(3, dtype=np.float32) for uid in env.active})
    second = build_world_render_snapshot(env, trajectory_max_points=20)

    frames = queue.Queue(maxsize=4)
    ready = queue.Queue(maxsize=1)
    status = queue.Queue(maxsize=2)
    stop = threading.Event()
    paused = SimpleNamespace(value=False)
    playback_speed = SimpleNamespace(value=0.0)
    stop_requested = SimpleNamespace(value=False)
    worker = threading.Thread(
        target=_buffered_process_render_main,
        args=(frames, stop, ready, status, paused, playback_speed, stop_requested, initial, VisualizationConfig()),
        daemon=True,
    )
    worker.start()
    assert ready.get(timeout=1.0)["ready"] is True
    frames.put((first, time.perf_counter()))
    frames.put((second, time.perf_counter()))
    time.sleep(1.0)
    stop.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert FakeViewer.instances[0].source_updates
    assert FakeViewer.instances[0].interpolated_updates
