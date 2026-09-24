from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

from terminal_marl.core.config import SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.visualization import pyvista_viewer as viewer_module
from terminal_marl.visualization.config import VisualizationConfig
from terminal_marl.visualization.pyvista_viewer import (
    BufferedProcessTerminalViewerObserver,
    Terminal3DViewer,
    _buffered_process_render_main,
    _interpolated_uav_positions,
)
from terminal_marl.visualization.snapshot import build_world_render_snapshot


class _FakePlotter:
    def __init__(self) -> None:
        self.text: list[tuple[str, object]] = []
        self.meshes: list[tuple[object, dict]] = []
        self.labels: list[tuple[list[str], dict]] = []

    def add_text(self, text, **kwargs):
        self.text.append((str(text), kwargs))

    def add_mesh(self, mesh, **kwargs):
        self.meshes.append((mesh, kwargs))
        return SimpleNamespace()

    def add_point_labels(self, points, labels, **kwargs):
        if isinstance(labels, str):
            labels = getattr(points, labels)
        self.labels.append((list(labels), kwargs))
        return SimpleNamespace()

    def remove_actor(self, *_args, **_kwargs):
        return None


class _FakeColor:
    _values = {
        "red": (255, 0, 0),
        "darkorange": (255, 140, 0),
        "#1f77b4": (31, 119, 180),
    }

    def __init__(self, name: str) -> None:
        self.int_rgb = self._values.get(name, (1, 2, 3))


class _FakeCloud:
    def __setitem__(self, key, value):
        setattr(self, key, value)


class _FakePolyData:
    def __init__(self, points):
        self.points = points

    def __setitem__(self, key, value):
        setattr(self, key, value)


def _event_viewer() -> Terminal3DViewer:
    viewer = object.__new__(Terminal3DViewer)
    viewer._plotter = _FakePlotter()
    viewer._event_page = 0
    viewer._event_keys = set()
    viewer._terminal_event_keys = set()
    viewer._terminal_events = []
    viewer._event_markers = {}
    viewer._pv = SimpleNamespace(Color=_FakeColor, PolyData=_FakePolyData)
    viewer._line_mesh = lambda points, close=False: points
    viewer._selected_uid = None
    return viewer


def _event(index: int, outcome: str = "failed_energy") -> SimpleNamespace:
    return SimpleNamespace(
        uid=index,
        arrival_id=index,
        outcome=outcome,
        reason="energy_depleted",
        time=float(index),
        position=(float(index), 0.0, 0.0),
        soc=0.0,
        energy_remaining=0.0,
        phase="final",
    )


def test_terminal_events_are_deduplicated_printed_and_paginated(capsys) -> None:
    viewer = _event_viewer()
    snapshot = SimpleNamespace(terminal_events=tuple(_event(index) for index in range(8)))

    viewer._record_terminal_events(snapshot)
    viewer._record_terminal_events(snapshot)

    assert len(viewer._terminal_events) == 8
    assert len(viewer._event_markers) == 8
    assert capsys.readouterr().out.count("[terminal-viewer-event]") == 8
    viewer._next_event_page()
    assert viewer._event_page == 1
    assert "page 2/2" in viewer._plotter.text[-1][0]


def test_alert_colour_and_warning_label_are_independent_of_normal_labels() -> None:
    assert viewer_module._uav_color(3, "emergency") == "red"
    assert viewer_module._uav_color(3, "low_soc") == "darkorange"
    assert viewer_module._uav_color(3, "normal") == viewer_module._identity_color(3)

    viewer = object.__new__(Terminal3DViewer)
    viewer._plotter = _FakePlotter()
    viewer._pv = SimpleNamespace(Color=_FakeColor, PolyData=_FakePolyData)
    viewer._config = SimpleNamespace(show_labels=False)
    viewer._uav_cloud = _FakeCloud()
    viewer._uav_actor = SimpleNamespace(SetVisibility=lambda _value: None)
    snapshot = SimpleNamespace(
        uavs=(
            SimpleNamespace(uid=1, position=(0.0, 0.0, 0.0), soc=0.1, emergency_landing=False, energy_alert="low_soc"),
            SimpleNamespace(uid=2, position=(1.0, 0.0, 0.0), soc=0.2, emergency_landing=True, energy_alert="emergency"),
        ),
        recent_conflict_uids=(),
    )
    viewer._update_uavs(snapshot)

    labels = [label for batch, _ in viewer._plotter.labels for label in batch]
    assert any(label.startswith("LOW SOC UID=1") for label in labels)
    assert any(label.startswith("EMERGENCY UID=2") for label in labels)


def test_interpolation_turns_off_for_long_source_gap_and_plan_change() -> None:
    previous = SimpleNamespace(
        time=0.0,
        uavs=(SimpleNamespace(uid=4, phase="ring", position=(0.0, 0.0, 0.0), trajectory_plan_id="p1"),),
    )
    long_gap = SimpleNamespace(
        time=2.0,
        uavs=(SimpleNamespace(uid=4, phase="ring", position=(100.0, 0.0, 0.0), trajectory_plan_id="p1"),),
    )
    changed_plan = SimpleNamespace(
        time=0.1,
        uavs=(SimpleNamespace(uid=4, phase="ring", position=(100.0, 0.0, 0.0), trajectory_plan_id="p2"),),
    )

    assert _interpolated_uav_positions(previous, long_gap, 0.5) == ((100.0, 0.0, 0.0),)
    assert _interpolated_uav_positions(previous, changed_plan, 0.5) == ((100.0, 0.0, 0.0),)


def test_process_consumes_final_frame_without_dropping_order_and_pumps_idle(monkeypatch) -> None:
    class FakeViewer:
        instances = []

        def __init__(self, _initial, _config):
            self.paused = False
            self.playback_speed = 0.0
            self.closed = False
            self.source_steps: list[int] = []
            self.pumps = 0
            self.rendered_positions: list[tuple[int, tuple[tuple[float, float, float], ...]]] = []
            self.final_rendered_before_ack = False
            self.idle_after_ack = 0
            self.instances.append(self)

        def update(self, snapshot, *, uav_positions=None, render=True):
            self.source_steps.append(snapshot.step_count)
            return True

        def update_dynamic(self, snapshot, positions, *, render=True):
            self.source_steps.append(snapshot.step_count)
            return True

        def update_interpolated_positions(self, snapshot, positions):
            self.pumps += 1
            self.rendered_positions.append((snapshot.step_count, positions))
            if snapshot.step_count == 6:
                if not final_consumed.value:
                    self.final_rendered_before_ack = True
                else:
                    self.idle_after_ack += 1
                    if self.idle_after_ack >= 1:
                        self.closed = True
            return True

        def wait_until_resumed(self, _stop_event=None):
            return True

        def close(self):
            self.closed = True

    monkeypatch.setattr(viewer_module, "Terminal3DViewer", FakeViewer)
    snapshot = lambda step: SimpleNamespace(
        time=float(step),
        step_count=step,
        rings=(),
        gates=(),
        uavs=(SimpleNamespace(
            uid=1,
            phase="ring",
            position=(float(step), 0.0, 0.0),
            trajectory_plan_id="plan-1",
        ),),
    )
    initial = snapshot(0)
    frames = queue.Queue(maxsize=viewer_module._PROCESS_BUFFER_CAPACITY)
    ready = queue.Queue(maxsize=1)
    status = queue.Queue(maxsize=2)
    stop = threading.Event()
    paused = SimpleNamespace(value=False)
    playback_speed = SimpleNamespace(value=0.0)
    stop_requested = SimpleNamespace(value=False)
    final_consumed = SimpleNamespace(value=False)
    worker = threading.Thread(
        target=_buffered_process_render_main,
        args=(frames, stop, ready, status, paused, playback_speed, stop_requested, initial, SimpleNamespace(), final_consumed),
        daemon=True,
    )
    worker.start()
    assert ready.get(timeout=1.0)["ready"]
    for step in range(1, 7):
        frames.put((snapshot(step), time.perf_counter(), step == 6), timeout=1.0)
    deadline = time.perf_counter() + 4.0
    while not stop_requested.value and time.perf_counter() < deadline:
        time.sleep(0.03)
    if not stop_requested.value:
        stop.set()
    worker.join(timeout=2.0)

    assert final_consumed.value
    assert not worker.is_alive()
    assert stop_requested.value
    assert FakeViewer.instances[0].source_steps == list(range(1, 7))
    assert FakeViewer.instances[0].final_rendered_before_ack
    assert FakeViewer.instances[0].idle_after_ack >= 1


def test_spawn_observer_finish_delivers_final_frame_offscreen(monkeypatch) -> None:
    monkeypatch.setenv("PYVISTA_OFF_SCREEN", "true")
    env = TerminalParallelEnv(SimConfig(
        arrival_model="finite_cohort",
        fixed_arrival_schedule=(0.0,),
        total_requested_uavs=1,
        max_simulation_seconds=20.0,
    ))
    env.reset(31)
    initial = build_world_render_snapshot(env, trajectory_max_points=20)
    observer = BufferedProcessTerminalViewerObserver(
        initial,
        VisualizationConfig(frame_stride=1, trajectory_max_points=20),
    )
    try:
        assert observer.wait_ready(timeout=20.0), observer.error
        env.step()
        before_finish = (
            env.now,
            tuple((uid, tuple(uav.position), uav.energy_remaining) for uid, uav in sorted(env.active.items())),
        )
        observer(env)
        assert observer.finish(env, wait_for_close=False)
        assert observer.error is None
        after_finish = (
            env.now,
            tuple((uid, tuple(uav.position), uav.energy_remaining) for uid, uav in sorted(env.active.items())),
        )
        assert after_finish == before_finish
    finally:
        observer.close()
    assert observer.error is None
    assert observer.stop_requested
