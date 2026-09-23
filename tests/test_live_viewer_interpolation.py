from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from terminal_marl.visualization import pyvista_viewer
from terminal_marl.visualization.config import VisualizationConfig


def _snapshot(
    *positions: tuple[float, float, float],
    phase: str = "cruise",
    plan_id: str | None = None,
    point_count: int = 0,
    ring_radius: float | None = None,
):
    rings = ()
    if ring_radius is not None:
        rings = (
            SimpleNamespace(
                role="outer",
                radius=ring_radius,
                height=100.0,
                target_radius=ring_radius,
                target_height=100.0,
                capacity=10,
            ),
        )
    return SimpleNamespace(
        rings=rings,
        gates=(),
        uavs=tuple(
            SimpleNamespace(
                uid=index,
                phase=phase,
                position=position,
                ring_role="outer",
                trajectory_plan_id=plan_id,
                trajectory_points=tuple((float(point), 0.0, 0.0) for point in range(point_count)),
            )
            for index, position in enumerate(positions)
        )
    )


def test_interpolation_only_moves_matching_uid_and_phase() -> None:
    previous = _snapshot((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    current = SimpleNamespace(
        uavs=(
            SimpleNamespace(uid=0, phase="cruise", position=(10.0, 0.0, 0.0)),
            SimpleNamespace(uid=1, phase="landing", position=(20.0, 0.0, 0.0)),
            SimpleNamespace(uid=2, phase="cruise", position=(30.0, 0.0, 0.0)),
        )
    )

    result = pyvista_viewer._interpolated_uav_positions(previous, current, 0.5)

    assert result == ((5.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0))
    assert previous.uavs[0].position == (0.0, 0.0, 0.0)
    assert pyvista_viewer._interpolated_uav_positions(previous, current, 2.0)[0] == (10.0, 0.0, 0.0)


def test_source_identity_ignores_plan_point_count_but_tracks_ring_geometry() -> None:
    initial = _snapshot((0.0, 0.0, 0.0), plan_id="plan-a", point_count=2, ring_radius=600.0)
    point_count_changed = _snapshot((1.0, 0.0, 0.0), plan_id="plan-a", point_count=3, ring_radius=600.0)
    ring_changed = _snapshot((1.0, 0.0, 0.0), plan_id="plan-a", point_count=3, ring_radius=610.0)

    assert pyvista_viewer._source_identity(initial) == pyvista_viewer._source_identity(point_count_changed)
    assert pyvista_viewer._source_identity(initial) != pyvista_viewer._source_identity(ring_changed)


def test_async_observer_renders_intermediate_positions_without_full_rebuild(monkeypatch) -> None:
    initial = _snapshot((0.0, 0.0, 0.0), plan_id="plan-a", point_count=2, ring_radius=600.0)
    target = _snapshot((12.0, 0.0, 0.0), plan_id="plan-a", point_count=3, ring_radius=600.0)
    ring_target = _snapshot((18.0, 0.0, 0.0), plan_id="plan-a", point_count=3, ring_radius=610.0)
    full_updates: list[tuple[object, tuple[tuple[float, float, float], ...] | None]] = []
    dynamic_updates: list[tuple[object, tuple[tuple[float, float, float], ...] | None]] = []
    intermediate_updates: list[tuple[tuple[float, float, float], ...]] = []
    full_event = threading.Event()
    intermediate_event = threading.Event()

    class FakeViewer:
        paused = False
        playback_speed = 0.0
        closed = False

        def __init__(self, _initial, _config):
            pass

        def update(self, snapshot, *, uav_positions=None, render=True):
            full_updates.append((snapshot, uav_positions))
            full_event.set()
            return True

        def update_dynamic(self, snapshot, positions, *, render=True):
            dynamic_updates.append((snapshot, positions))
            full_event.set()
            return True

        def update_interpolated_positions(self, _snapshot, positions):
            intermediate_updates.append(positions)
            intermediate_event.set()
            return True

        def wait_until_resumed(self, _stop_event=None):
            return True

        def close(self):
            self.closed = True

    monkeypatch.setattr(pyvista_viewer, "Terminal3DViewer", FakeViewer)
    source_snapshots = iter((target, ring_target))
    monkeypatch.setattr(
        pyvista_viewer,
        "build_world_render_snapshot",
        lambda *_args, **_kwargs: next(source_snapshots),
    )
    observer = pyvista_viewer.AsyncTerminalViewerObserver(initial, VisualizationConfig(frame_stride=1))
    try:
        assert observer.wait_ready()
        observer._last_enqueued_at = time.perf_counter() - 0.25
        observer(SimpleNamespace(step_count=1))
        assert full_event.wait(1.0)
        assert full_updates == []
        assert dynamic_updates == [(target, ((0.0, 0.0, 0.0),))]
        assert intermediate_event.wait(1.0)
        full_event.clear()
        observer(SimpleNamespace(step_count=2))
        assert full_event.wait(1.0)
    finally:
        observer.close()

    assert intermediate_updates
    assert any(0.0 < positions[0][0] < 12.0 for positions in intermediate_updates)
    assert all(0.0 <= positions[0][0] <= 18.0 for positions in intermediate_updates)
    assert len(full_updates) == 1
    assert full_updates[0][0] is ring_target
