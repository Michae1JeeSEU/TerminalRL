from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("pyvista")
vtk_rendering = pytest.importorskip("vtkmodules.vtkRenderingCore")

from terminal_marl.visualization.config import VisualizationConfig
from terminal_marl.visualization.pyvista_viewer import Terminal3DViewer, _vertex_cells
from terminal_marl.visualization.schema import RingRenderState, UAVRenderState, WorldRenderSnapshot


def _uav(uid: int, position: tuple[float, float, float], *, alert: str = "normal") -> UAVRenderState:
    x, y, z = position
    return UAVRenderState(
        uid=uid,
        arrival_id=uid,
        phase="ring_outer",
        ring_role="outer",
        position=position,
        velocity=(1.0, 0.0, 0.0),
        speed=1.0,
        soc={"normal": 0.75, "low_soc": 0.44, "emergency": 0.32}[alert],
        energy_remaining=750.0,
        energy_capacity=1000.0,
        mission_type="passenger",
        base_priority=1,
        assigned_gate=0,
        assigned_lz=None,
        trajectory_plan_id=f"plan-{uid}",
        merge_reservation_id=None,
        committed=True,
        trajectory_points=(position, (x + 20.0, y + 10.0, z + 5.0)),
        dynamic_priority=1.0,
        last_release_decision=None,
        last_safety_decision=None,
        emergency_landing=alert == "emergency",
        energy_alert=alert,
    )


def _uavs(uids: tuple[int, ...] | list[int]) -> tuple[UAVRenderState, ...]:
    result = []
    for index, uid in enumerate(uids):
        column = index % 6
        row = index // 6
        result.append(_uav(uid, ((column - 2.5) * 90.0, (row - 2.0) * 90.0, 100.0)))
    return tuple(result)


def _snapshot(
    uavs: tuple[UAVRenderState, ...], *, time: float = 0.0, step_count: int = 0
) -> WorldRenderSnapshot:
    return WorldRenderSnapshot(
        time=time,
        step_count=step_count,
        rings=(RingRenderState("outer", 600.0, 100.0, 600.0, 100.0, 30, len(uavs)),),
        gates=(),
        landing_zones=(),
        uavs=uavs,
        external_queue_size=0,
        service_count=0,
        completed_count=0,
        transition_active=False,
        transition_progress=0.0,
        telemetry={},
        recent_conflict_uids=(),
    )


@pytest.fixture
def offscreen(monkeypatch):
    monkeypatch.setenv("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    monkeypatch.setattr(pv, "OFF_SCREEN", True)
    yield pv


def _display_coordinates(renderer, point: tuple[float, float, float]) -> tuple[float, float]:
    renderer.SetWorldPoint(*point, 1.0)
    renderer.WorldToDisplay()
    x, y, _ = renderer.GetDisplayPoint()
    return float(x), float(y)


def _pick_point(viewer: Terminal3DViewer, point: tuple[float, float, float]):
    picker = vtk_rendering.vtkPointPicker()
    picker.SetTolerance(0.05)
    renderer = viewer._plotter.renderer
    x, y = _display_coordinates(renderer, point)
    assert picker.Pick(x, y, 0.0, renderer)
    return picker


def test_uav_cloud_vertex_cells_survive_shrink_empty_refill(offscreen) -> None:
    full = _snapshot(_uavs(tuple(range(1, 31))))
    viewer = Terminal3DViewer(full, VisualizationConfig())
    actor = viewer._uav_actor
    cloud = viewer._uav_cloud
    try:
        for step, uavs in enumerate((full.uavs[:11], (), full.uavs), start=1):
            assert viewer.update(_snapshot(tuple(uavs), time=float(step), step_count=step), render=False)
            expected = max(1, len(uavs))
            assert viewer._uav_actor is actor
            assert viewer._uav_cloud is cloud
            assert cloud.n_points == expected
            assert cloud.n_verts == expected
            np.testing.assert_array_equal(np.asarray(cloud.verts).ravel(), _vertex_cells(expected))
            assert actor.GetVisibility() == (1 if uavs else 0)
    finally:
        viewer.close()


def test_real_point_picker_selects_uid_and_respects_uav_pick_list(offscreen, capsys) -> None:
    uavs = _uavs(tuple(range(1, 31)))
    snapshot = _snapshot(uavs)
    viewer = Terminal3DViewer(snapshot, VisualizationConfig(show_labels=True))
    try:
        assert viewer.update(snapshot, render=True)
        picker = getattr(getattr(viewer._plotter, "iren", None), "picker", None)
        assert picker is not None
        assert picker.GetPickFromList() == 1
        pick_list = picker.GetPickList()
        assert pick_list.GetNumberOfItems() == 1
        assert viewer._same_vtk_object(pick_list.GetItemAsObject(0), viewer._uav_actor)
        assert all(actor.GetPickable() == 0 for actor in viewer._label_actors.values())

        for uid in (4, 11, 29):
            index = next(index for index, uav in enumerate(uavs) if uav.uid == uid)
            point_picker = _pick_point(viewer, uavs[index].position)
            assert point_picker.GetPointId() == index
            viewer._select_uav(np.asarray(uavs[index].position), point_picker)
            assert viewer._selected_uid == uid
        output = capsys.readouterr().out
        assert "[terminal-viewer-selection]" in output
        assert "UID: 29" in output

        kept = tuple(uav for uav in uavs if uav.uid in {4, 11, 29})
        shrunk = _snapshot(kept, time=1.0, step_count=1)
        assert viewer.update(shrunk, render=True)
        point_picker = _pick_point(viewer, kept[2].position)
        assert point_picker.GetPointId() == 2
        viewer._select_uav(np.asarray(kept[2].position), point_picker)
        assert viewer._selected_uid == 29
    finally:
        viewer.close()


def test_selected_trajectory_style_and_interpolated_head(offscreen) -> None:
    uavs = _uavs((4, 11))
    snapshot = _snapshot(uavs)
    viewer = Terminal3DViewer(snapshot, VisualizationConfig(show_trajectories=True))
    try:
        assert viewer.update(snapshot, render=True)
        assert viewer._trajectory_actors["trajectory-4"].GetProperty().GetLineWidth() == pytest.approx(1.5)
        assert viewer._trajectory_actors["trajectory-11"].GetProperty().GetLineWidth() == pytest.approx(1.5)
        viewer._select_uav(
            np.asarray(uavs[0].position),
            SimpleNamespace(
                GetActor=lambda: viewer._uav_actor,
                GetDataSet=lambda: viewer._uav_cloud,
                GetPointId=lambda: 0,
            ),
        )
        assert viewer._trajectory_actors["trajectory-4"].GetProperty().GetLineWidth() == pytest.approx(3.0)
        assert viewer._trajectory_actors["trajectory-11"].GetProperty().GetLineWidth() == pytest.approx(1.5)

        displayed = ((40.0, -20.0, 110.0), (-40.0, 20.0, 115.0))
        assert viewer.update_interpolated_positions(snapshot, displayed)
        np.testing.assert_allclose(viewer._trajectory_meshes["trajectory-4"].points[0], displayed[0])
        np.testing.assert_allclose(viewer._trajectory_meshes["trajectory-11"].points[0], displayed[1])
    finally:
        viewer.close()


def test_interpolated_positions_reuse_cached_actors_and_render_once(offscreen) -> None:
    uavs = (
        _uav(4, (-100.0, 0.0, 100.0)),
        _uav(11, (0.0, 0.0, 100.0), alert="low_soc"),
        _uav(29, (100.0, 0.0, 100.0), alert="emergency"),
    )
    snapshot = _snapshot(uavs)
    viewer = Terminal3DViewer(
        snapshot,
        VisualizationConfig(show_labels=True, show_trajectories=True),
    )
    try:
        assert viewer.update(snapshot, render=True)
        label_actor_ids = {name: id(actor) for name, actor in viewer._label_actors.items()}
        label_dataset_ids = {name: id(dataset) for name, dataset in viewer._label_datasets.items()}
        trajectory_mesh_ids = {name: id(mesh) for name, mesh in viewer._trajectory_meshes.items()}
        rgb_before = np.array(viewer._uav_cloud["rgb"], copy=True)
        displayed = ((-80.0, 10.0, 100.0), (20.0, 10.0, 100.0), (120.0, 10.0, 100.0))
        with patch.object(viewer, "_update_uavs", wraps=viewer._update_uavs) as update_uavs_spy:
            with patch.object(viewer._plotter, "update", wraps=viewer._plotter.update) as render_spy:
                assert viewer.update_interpolated_positions(snapshot, displayed)
        assert update_uavs_spy.call_count == 0
        assert render_spy.call_count == 1
        assert {name: id(actor) for name, actor in viewer._label_actors.items()} == label_actor_ids
        assert {name: id(dataset) for name, dataset in viewer._label_datasets.items()} == label_dataset_ids
        assert {name: id(mesh) for name, mesh in viewer._trajectory_meshes.items()} == trajectory_mesh_ids
        np.testing.assert_array_equal(viewer._uav_cloud["rgb"], rgb_before)
    finally:
        viewer.close()
