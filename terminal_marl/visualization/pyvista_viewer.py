"""PyVista implementation of the optional, snapshot-only terminal viewer."""
from __future__ import annotations

from dataclasses import replace
import multiprocessing as mp
import os
import time
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import TYPE_CHECKING

import numpy as np

from .config import VisualizationConfig
from .schema import WorldRenderSnapshot
from .snapshot import build_world_render_snapshot

if TYPE_CHECKING:
    from ..envs.parallel_env import TerminalParallelEnv


_RING_COLORS = {"outer": "steelblue", "middle": "darkorange", "inner": "seagreen"}
_UAV_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#4d4d4d", "#8c9d16", "#17becf", "#393b79", "#637939",
    "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354",
    "#756bb1", "#636363", "#6b6ecf", "#9c9ede", "#bd9e39", "#ad494a",
)
_LZ_COLORS = {"free": "seagreen", "reserved": "gold", "occupied": "tomato"}
_PLAYBACK_OPTIONS = ("0.25x", "0.5x", "1x", "2x", "5x", "10x", "max")
_PLAYBACK_VALUES = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "5x": 5.0, "10x": 10.0, "max": 0.0}
_INTERPOLATION_HZ = 30.0
_INTERPOLATION_PERIOD = 1.0 / _INTERPOLATION_HZ
_MIN_INTERPOLATION_SECONDS = _INTERPOLATION_PERIOD
_MAX_INTERPOLATION_SECONDS = 0.75
_PROCESS_BUFFER_CAPACITY = 4
_PROCESS_BUFFER_PREFILL = 3
_PROCESS_PREFILL_TIMEOUT_SECONDS = 0.75
_SOURCE_GAP_SECONDS = 1.01
_EVENT_PAGE_SIZE = 6


def _event_value(event, name: str, default=None):
    """Read event fields from the frozen render contract or test doubles."""
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def _terminal_events(snapshot: WorldRenderSnapshot) -> tuple[object, ...]:
    """Return the cumulative event stream while keeping old test snapshots valid."""
    return tuple(getattr(snapshot, "terminal_events", ()) or ())


def _event_key(event) -> tuple[object, ...]:
    """Stable deduplication key for cumulative terminal outcome records."""
    return (
        _event_value(event, "uid"),
        _event_value(event, "arrival_id"),
        _event_value(event, "outcome", ""),
        _event_value(event, "reason", ""),
        float(_event_value(event, "time", 0.0) or 0.0),
    )


def _event_outcome(event) -> str:
    return str(_event_value(event, "outcome", "")).lower()


def _uav_alert(uav) -> str:
    """Normalize the authoritative alert field for renderer/test compatibility."""
    if bool(getattr(uav, "emergency_landing", False)):
        return "emergency"
    alert = str(getattr(uav, "energy_alert", "normal") or "normal").lower()
    return alert if alert in {"normal", "low_soc", "emergency"} else "normal"


def _uav_color(uid: int, alert: str) -> str:
    if alert == "emergency":
        return "red"
    if alert == "low_soc":
        return "darkorange"
    return _identity_color(uid)


def _interpolated_uav_positions(
    previous: WorldRenderSnapshot,
    current: WorldRenderSnapshot,
    alpha: float,
    *,
    previous_positions: tuple[tuple[float, float, float], ...] | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Return visual-only positions for ``current`` without mutating snapshots.

    A point moves only when the same UID remains in the same phase in both
    source snapshots.  New aircraft and phase transitions use the current
    source position immediately, so interpolation never invents a path
    across a lifecycle boundary.  ``alpha`` is deliberately clamped to avoid
    extrapolating beyond the latest source frame.
    """
    clamped = min(1.0, max(0.0, float(alpha)))
    if previous_positions is None:
        previous_positions = tuple(uav.position for uav in previous.uavs)
    previous_by_uid = {
        uav.uid: (position, uav.phase, getattr(uav, "trajectory_plan_id", None))
        for uav, position in zip(previous.uavs, previous_positions)
    }
    previous_time = getattr(previous, "time", None)
    current_time = getattr(current, "time", None)
    source_gap = None
    if previous_time is not None and current_time is not None:
        source_gap = float(current_time) - float(previous_time)
    positions: list[tuple[float, float, float]] = []
    for uav in current.uavs:
        previous_state = previous_by_uid.get(uav.uid)
        previous_plan = previous_state[2] if previous_state is not None else None
        current_plan = getattr(uav, "trajectory_plan_id", None)
        if (
            previous_state is None
            or previous_state[1] != uav.phase
            or (source_gap is not None and source_gap > _SOURCE_GAP_SECONDS)
            or previous_plan != current_plan
        ):
            positions.append(tuple(float(value) for value in uav.position))
            continue
        start = np.asarray(previous_state[0], dtype=float)
        end = np.asarray(uav.position, dtype=float)
        point = start + (end - start) * clamped
        positions.append((float(point[0]), float(point[1]), float(point[2])))
    return tuple(positions)


def _source_identity(snapshot: WorldRenderSnapshot) -> tuple[object, ...]:
    """Return the structural identity which requires a full scene refresh."""
    rings = tuple(
        (
            ring.role,
            ring.radius,
            ring.height,
            ring.target_radius,
            ring.target_height,
            ring.capacity,
        )
        for ring in getattr(snapshot, "rings", ())
    )
    gates = tuple(
        (gate.gate_id, gate.spawn_position, gate.gate_position)
        for gate in getattr(snapshot, "gates", ())
    )
    uavs = tuple(
        (
            uav.uid,
            uav.phase,
            getattr(uav, "ring_role", None),
            getattr(uav, "trajectory_plan_id", None),
        )
        for uav in snapshot.uavs
    )
    return rings, gates, uavs


def _snapshot_for_process(snapshot: WorldRenderSnapshot) -> WorldRenderSnapshot:
    """Copy the telemetry mapping into a spawn/pickle-friendly plain dict."""
    return replace(snapshot, telemetry=dict(snapshot.telemetry))


def _identity_color(uid: int) -> str:
    """Stable UAV/trajectory color shared across the complete lifecycle."""
    return _UAV_COLORS[int(uid) % len(_UAV_COLORS)]


def _trajectory_signature(uav) -> tuple[str | None, int]:
    """Topology identity used to rebuild a line when its committed plan changes."""
    return uav.trajectory_plan_id, len(uav.trajectory_points)


def _vertex_cells(point_count: int) -> np.ndarray:
    """Return VTK vertex cells for exactly ``point_count`` points.

    ``PolyData.points = ...`` changes the point array but does not change the
    cell topology.  The viewer keeps a one-point hidden placeholder for an
    empty scene, and therefore has to update the vertex cells whenever the
    number of visible UAVs changes.
    """
    count = max(1, int(point_count))
    return np.column_stack(
        (np.ones(count, dtype=np.int64), np.arange(count, dtype=np.int64))
    ).ravel()


class Terminal3DViewer:
    """Interactive 3D renderer which consumes immutable world snapshots only."""

    def __init__(self, initial: WorldRenderSnapshot, config: VisualizationConfig, *, off_screen: bool = False) -> None:
        try:
            import pyvista as pv
        except ImportError as error:  # pragma: no cover - depends on local optional package
            raise RuntimeError(
                "Visualization dependencies are not installed. "
                "Install requirements-visualization.txt and retry."
            ) from error
        self._pv = pv
        self._config = config
        plotter_options = {"off_screen": True} if off_screen else {}
        self._plotter = pv.Plotter(window_size=(1280, 860), title="Terminal MARL 3D Viewer", **plotter_options)
        self._plotter.set_background("white")
        self._ring_meshes: dict[str, object] = {}
        self._target_ring_meshes: dict[str, object] = {}
        self._trajectory_names: set[str] = set()
        self._trajectory_meshes: dict[str, object] = {}
        self._trajectory_actors: dict[str, object] = {}
        self._trajectory_signatures: dict[str, tuple[str | None, int]] = {}
        self._lz_actors: dict[int, object] = {}
        self._terminal_event_keys: set[tuple[object, ...]] = set()
        self._terminal_events: list[object] = []
        self._event_page = 0
        self._event_markers: dict[tuple[object, ...], object] = {}
        self._label_datasets: dict[str, object] = {}
        self._label_actors: dict[str, object] = {}
        self._label_signatures: dict[str, tuple[str, ...]] = {}
        self._label_uids: dict[str, tuple[int, ...]] = {}
        self._text_cache: dict[str, str] = {}
        self._displayed_uav_positions: dict[int, tuple[float, float, float]] = {}
        self._source_gap_warning: str | None = None
        # PyVista rejects zero-length point arrays during construction.  A
        # hidden one-point backing mesh keeps the actor persistent even when
        # the simulation temporarily has no airborne UAVs.
        self._uav_cloud = self._pv.PolyData(np.zeros((1, 3), dtype=float))
        self._uav_cloud.verts = _vertex_cells(1)
        self._uav_cloud["rgb"] = np.zeros((1, 3), dtype=np.uint8)
        self._uav_point_count = 0
        self._uav_actor = None
        self._latest_snapshot = initial
        self._selected_uid: int | None = None
        self._paused = False
        self._playback_speed = config.playback_speed
        self._shown = False
        self._closed = False
        self._build_static_scene(initial)
        self._add_interaction_controls()
        self.update(initial)

    @staticmethod
    def _circle(radius: float, height: float, samples: int = 121) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, samples)
        return np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.full(samples, height)))

    def _line_mesh(self, points: tuple[tuple[float, float, float], ...] | np.ndarray, *, close: bool = False):
        return self._pv.lines_from_points(np.asarray(points, dtype=float), close=close)

    def _build_static_scene(self, snapshot: WorldRenderSnapshot) -> None:
        for ring in snapshot.rings:
            mesh = self._line_mesh(self._circle(ring.radius, ring.height), close=True)
            self._ring_meshes[ring.role] = mesh
            self._plotter.add_mesh(mesh, color=_RING_COLORS[ring.role], line_width=4, pickable=False, name=f"ring-{ring.role}")
            if self._config.show_target_geometry:
                target = self._line_mesh(self._circle(ring.target_radius, ring.target_height), close=True)
                self._target_ring_meshes[ring.role] = target
                self._plotter.add_mesh(target, color="gray", opacity=0.34, line_width=1, pickable=False, name=f"target-ring-{ring.role}")
        for gate in snapshot.gates:
            points = (gate.spawn_position, gate.gate_position)
            self._plotter.add_mesh(self._line_mesh(points), color="slategray", line_width=4, pickable=False, name=f"gate-{gate.gate_id}")
            self._plotter.add_point_labels(
                np.asarray([gate.gate_position]), [f"Gate {gate.gate_id}"], point_size=0,
                font_size=12, text_color="black", shape=None, always_visible=True,
                show_points=False, pickable=False, render=False,
                name=f"gate-label-{gate.gate_id}",
            )
        self._plotter.add_axes(color="black", x_color="firebrick", y_color="forestgreen", z_color="navy")
        self._plotter.add_text("", position="upper_right", font_size=11, color="black", name="hud")
        self._plotter.add_text(
            "UAV identity\nLOW SOC = orange\nEMERGENCY = red\nfailed_energy = red x\n[ / ]: event history",
            position=(10, 620), font_size=10, color="black", name="legend",
        )
        self._plotter.add_text("Right-click a UAV point to inspect", position=(10, 735), font_size=10, color="black", name="pick-help")
        for lz in snapshot.landing_zones:
            actor = self._plotter.add_mesh(self._pv.Cube(center=lz.position, x_length=45, y_length=45, z_length=6),
                                           color=_LZ_COLORS.get(lz.status, "white"), opacity=0.9, pickable=False,
                                           name=f"lz-{lz.lz_id}")
            self._lz_actors[lz.lz_id] = actor
        self._uav_actor = self._plotter.add_mesh(self._uav_cloud, scalars="rgb", rgb=True, point_size=15,
                                                 render_points_as_spheres=True, pickable=True, render=False,
                                                 name="uavs")
        self._update_landing_zone_labels(snapshot)
        self._update_pad_status_panel(snapshot)
        self._update_event_panel()

    @staticmethod
    def _speed_label(speed: float) -> str:
        if speed <= 0.0:
            return "max"
        return min(_PLAYBACK_OPTIONS[:-1], key=lambda item: abs(_PLAYBACK_VALUES[item] - speed))

    def _add_interaction_controls(self) -> None:
        self._plotter.add_checkbox_button_widget(
            self._set_paused, value=False, position=(10, 795), size=28, border_size=2,
            color_on="tomato", color_off="lightgray", background_color="white",
        )
        self._plotter.add_text("Pause", position=(45, 800), font_size=11, color="black", name="pause-label")
        self._plotter.add_text("Speed", position=(145, 800), font_size=11, color="black", name="speed-label")
        self._plotter.add_text_slider_widget(
            self._set_playback_speed, list(_PLAYBACK_OPTIONS),
            value=float(_PLAYBACK_OPTIONS.index(self._speed_label(self._playback_speed))),
            pointa=(0.18, 0.93), pointb=(0.43, 0.93), color="black", style="modern",
        )
        # Keep the page controls small and discoverable.  Older/fake plotters
        # used by unit tests may not implement key registration.
        try:
            self._plotter.add_key_event("[", self._previous_event_page)
            self._plotter.add_key_event("]", self._next_event_page)
        except (AttributeError, TypeError):  # pragma: no cover - test doubles
            pass
        self._plotter.enable_point_picking(
            self._select_uav, left_clicking=False, picker="point", show_message=False,
            show_point=False, tolerance=0.025, use_picker=True,
        )
        picker = getattr(getattr(self._plotter, "iren", None), "picker", None)
        if picker is not None and hasattr(picker, "PickFromListOn"):
            picker.PickFromListOn()
            if hasattr(picker, "InitializePickList"):
                picker.InitializePickList()
            if hasattr(picker, "AddPickList"):
                picker.AddPickList(self._uav_actor)

    def _set_paused(self, value: bool) -> None:
        self._paused = bool(value)

    def _set_playback_speed(self, value: str) -> None:
        self._playback_speed = _PLAYBACK_VALUES[str(value)]

    def _set_event_page(self, page: int) -> None:
        page_count = max(1, int(np.ceil(len(self._terminal_events) / _EVENT_PAGE_SIZE)))
        self._event_page = min(max(0, int(page)), page_count - 1)
        self._update_event_panel()

    def _previous_event_page(self) -> None:
        self._set_event_page(self._event_page - 1)

    def _next_event_page(self) -> None:
        self._set_event_page(self._event_page + 1)

    @staticmethod
    def _uav_uid_from_point_id(snapshot: WorldRenderSnapshot, point_id: int) -> int | None:
        """Map PyVista's current point-cloud index to snapshot identity."""
        return snapshot.uavs[point_id].uid if 0 <= point_id < len(snapshot.uavs) else None

    @staticmethod
    def _same_vtk_object(left, right) -> bool:
        if left is None or right is None:
            return False
        if left is right:
            return True
        try:
            return left.GetAddressAsString("") == right.GetAddressAsString("")
        except (AttributeError, TypeError, ValueError):
            return False

    def _picker_matches_uav_cloud(self, picker) -> bool:
        """Accept point picks only when VTK reports the UAV actor/dataset."""
        if picker is None or self._uav_actor is None or self._uav_cloud is None:
            return False
        actor = None
        try:
            actor = picker.GetActor()
        except AttributeError:
            pass
        if self._same_vtk_object(actor, self._uav_actor):
            return True
        if actor is not None:
            try:
                mapper = actor.GetMapper()
                dataset = mapper.GetInput() if mapper is not None else None
            except AttributeError:
                dataset = None
            if self._same_vtk_object(dataset, self._uav_cloud):
                return True
        try:
            dataset = picker.GetDataSet()
        except AttributeError:
            dataset = None
        return self._same_vtk_object(dataset, self._uav_cloud)

    def _select_uav(self, picked_point: np.ndarray, picker) -> None:
        """Select a point in the current UAV cloud, never an obscuring mesh."""
        if not self._latest_snapshot.uavs or not self._picker_matches_uav_cloud(picker):
            return
        try:
            point_id = int(picker.GetPointId())
        except (AttributeError, TypeError, ValueError):
            return
        point_count = getattr(self, "_uav_point_count", len(self._latest_snapshot.uavs))
        if not 0 <= point_id < min(point_count, len(self._latest_snapshot.uavs)):
            return
        selected_uid = self._uav_uid_from_point_id(self._latest_snapshot, point_id)
        if selected_uid is None:
            return
        self._selected_uid = selected_uid
        self._update_selected_uav_panel()
        self._update_trajectory_styles()
        print(f"[terminal-viewer-selection]\n{self._selected_uav_panel_text()}", flush=True)

    def _selected_uav(self):
        return next((uav for uav in self._latest_snapshot.uavs if uav.uid == self._selected_uid), None)

    def _selected_terminal_event(self):
        if self._selected_uid is None:
            return None
        for event in reversed(self._terminal_events):
            if _event_value(event, "uid") == self._selected_uid:
                return event
        return None

    def _selected_uav_panel_text(self) -> str:
        uav = self._selected_uav()
        if uav is None:
            event = self._selected_terminal_event()
            if event is None:
                text = "Selected UAV: —"
            else:
                text = (
                    "Selected terminal event\n"
                    f"UID: {_event_value(event, 'uid', '—')}\n"
                    f"Time: {float(_event_value(event, 'time', 0.0)):.1f} s\n"
                    f"Outcome: {_event_value(event, 'outcome', '—')}\n"
                    f"Reason: {_event_value(event, 'reason', '—')}\n"
                    f"SOC: {float(_event_value(event, 'soc', 0.0)):.0%}\n"
                    f"Energy: {float(_event_value(event, 'energy_remaining', 0.0)):.0f}\n"
                    f"Phase: {_event_value(event, 'phase', '—')}"
                )
        else:
            energy = f"{uav.energy_remaining:.0f}/{uav.energy_capacity:.0f} ({uav.soc:.0%})"
            alert = _uav_alert(uav)
            rescue_line = f"Emergency: {uav.rescue_status}\n" if getattr(uav, "rescue_status", None) else ""
            text = (
                f"Selected UAV\n"
                f"UID: {uav.uid}\n"
                f"Arrival ID: {uav.arrival_id}\n"
                f"Phase: {uav.phase}\n"
                f"Ring: {uav.ring_role or '—'}\n"
                f"Speed: {uav.speed:.1f} m/s\n"
                f"Energy: {energy}\n"
                f"Energy alert: {alert}\n"
                f"{rescue_line}"
                f"Mission: {uav.mission_type}\n"
                f"Base priority: {uav.base_priority}\n"
                f"Dynamic priority: {uav.dynamic_priority:.2f}\n"
                f"Gate: {uav.assigned_gate}\n"
                f"Landing zone: {uav.assigned_lz if uav.assigned_lz is not None else '—'}\n"
                f"Trajectory plan: {uav.trajectory_plan_id or '—'}\n"
                f"Merge reservation: {uav.merge_reservation_id or '—'}\n"
                f"Last release: {uav.last_release_decision or '—'}\n"
                f"Last safety: {uav.last_safety_decision or '—'}"
            )
        return text

    def _update_selected_uav_panel(self) -> None:
        text = self._selected_uav_panel_text()
        text_cache = getattr(self, "_text_cache", {})
        self._text_cache = text_cache
        if text_cache.get("selected-uav") == text:
            return
        self._plotter.add_text(text, position="lower_right", font_size=10, color="black", name="selected-uav")
        text_cache["selected-uav"] = text

    def _update_rings(self, snapshot: WorldRenderSnapshot) -> None:
        for ring in snapshot.rings:
            self._ring_meshes[ring.role].points = self._circle(ring.radius, ring.height)
            if self._config.show_target_geometry:
                self._target_ring_meshes[ring.role].points = self._circle(ring.target_radius, ring.target_height)

    def _update_landing_zones(self, snapshot: WorldRenderSnapshot) -> None:
        for lz in snapshot.landing_zones:
            actor = self._lz_actors[lz.lz_id]
            actor.GetProperty().SetColor(self._pv.Color(_LZ_COLORS.get(lz.status, "white")).float_rgb)

    @staticmethod
    def _landing_zone_label(lz) -> str:
        return f"P{lz.lz_id}"

    def _update_landing_zone_labels(self, snapshot: WorldRenderSnapshot) -> None:
        points = np.asarray(
            [(float(lz.position[0]), float(lz.position[1]), float(lz.position[2]) + 9.0)
             for lz in snapshot.landing_zones],
            dtype=float,
        )
        labels = [self._landing_zone_label(lz) for lz in snapshot.landing_zones]
        self._update_label_group(
            "lz-labels", points, labels, font_size=9, text_color="black",
        )

    def _remove_label_group(self, name: str) -> None:
        for actor_name in (name, f"{name}-labels", f"{name}-points"):
            self._plotter.remove_actor(actor_name, render=False)
        getattr(self, "_label_datasets", {}).pop(name, None)
        getattr(self, "_label_actors", {}).pop(name, None)
        getattr(self, "_label_uids", {}).pop(name, None)

    def _update_label_group(
        self,
        name: str,
        points,
        labels,
        *,
        font_size: int,
        text_color: str,
        uids: tuple[int, ...] | None = None,
    ) -> None:
        """Update cached label data without recreating unchanged label actors."""
        label_values = tuple(str(label) for label in labels)
        label_signatures = getattr(self, "_label_signatures", {})
        self._label_signatures = label_signatures
        if not label_values:
            if name in label_signatures or name in getattr(self, "_label_actors", {}):
                self._remove_label_group(name)
            label_signatures[name] = ()
            getattr(self, "_label_uids", {}).pop(name, None)
            return

        point_array = np.asarray(points, dtype=float).reshape((-1, 3))
        if len(point_array) != len(label_values):
            raise ValueError("label points and labels must have equal lengths")
        label_datasets = getattr(self, "_label_datasets", {})
        label_actors = getattr(self, "_label_actors", {})
        label_uids = getattr(self, "_label_uids", {})
        self._label_datasets = label_datasets
        self._label_actors = label_actors
        self._label_uids = label_uids
        dataset = label_datasets.get(name)
        if dataset is not None and label_signatures.get(name) == label_values:
            dataset.points = point_array
            label_uids[name] = tuple(uids or ())
            return

        self._remove_label_group(name)
        dataset = self._pv.PolyData(point_array)
        dataset["labels"] = np.asarray(label_values, dtype=str)
        actor = self._plotter.add_point_labels(
            dataset,
            "labels",
            point_size=0,
            font_size=font_size,
            text_color=text_color,
            shape=None,
            always_visible=True,
            show_points=False,
            pickable=False,
            render=False,
            name=name,
        )
        label_datasets[name] = dataset
        label_actors[name] = actor
        label_signatures[name] = label_values
        label_uids[name] = tuple(uids or ())

    @staticmethod
    def _landing_zone_status_line(lz) -> str:
        if lz.occupied_by is not None:
            return f"P{lz.lz_id}: SERVICE UID={lz.occupied_by}"
        if lz.reserved_by is not None:
            return f"P{lz.lz_id}: RESERVED UID={lz.reserved_by}"
        return f"P{lz.lz_id}: FREE"

    def _update_pad_status_panel(self, snapshot: WorldRenderSnapshot) -> None:
        lines = ["PAD STATUS"]
        lines.extend(self._landing_zone_status_line(lz) for lz in snapshot.landing_zones)
        text = "\n".join(lines)
        text_cache = getattr(self, "_text_cache", {})
        self._text_cache = text_cache
        if text_cache.get("pad-status") == text:
            return
        self._plotter.add_text(text, position=(10, 430), font_size=9, color="black", name="pad-status")
        text_cache["pad-status"] = text

    @staticmethod
    def _is_attention_event(event) -> bool:
        return _event_outcome(event) not in {
            "",
            "normal",
            "completed",
            "complete",
            "success",
            "landed",
            "touchdown",
        }

    def _event_line(self, event) -> str:
        uid = _event_value(event, "uid")
        uid_text = "—" if uid is None else str(uid)
        return (
            f"t={float(_event_value(event, 'time', 0.0)):.1f} "
            f"UID={uid_text} {_event_value(event, 'outcome', 'unknown')} "
            f"{_event_value(event, 'reason', '')} "
            f"SOC={float(_event_value(event, 'soc', 0.0)):.0%}"
        ).strip()

    def _add_failed_energy_marker(self, event, key: tuple[object, ...]) -> None:
        position = _event_value(event, "position")
        if position is None or key in self._event_markers:
            return
        center = np.asarray(position, dtype=float)
        span = 14.0
        p1 = center + np.asarray((-span, -span, 0.0))
        p2 = center + np.asarray((span, span, 0.0))
        p3 = center + np.asarray((-span, span, 0.0))
        p4 = center + np.asarray((span, -span, 0.0))
        index = len(self._event_markers)
        uid = _event_value(event, "uid", "—")
        actors = (
            self._plotter.add_mesh(
                self._line_mesh((tuple(p1), tuple(p2))),
                color="red",
                line_width=5,
                pickable=False,
                name=f"failed-energy-marker-{index}-a",
            ),
            self._plotter.add_mesh(
                self._line_mesh((tuple(p3), tuple(p4))),
                color="red",
                line_width=5,
                pickable=False,
                name=f"failed-energy-marker-{index}-b",
            ),
            self._plotter.add_point_labels(
                np.asarray([center + np.asarray((0.0, 0.0, 12.0))], dtype=float),
                [f"UID={uid} FAILED ENERGY"],
                point_size=0,
                font_size=9,
                text_color="red",
                shape=None,
                always_visible=True,
                show_points=False,
                pickable=False,
                render=False,
                name=f"failed-energy-marker-{index}-label",
            ),
        )
        self._event_markers[key] = actors

    def _record_terminal_events(self, snapshot: WorldRenderSnapshot) -> None:
        changed = False
        for event in _terminal_events(snapshot):
            if not self._is_attention_event(event):
                continue
            key = _event_key(event)
            if key in self._terminal_event_keys:
                continue
            self._terminal_event_keys.add(key)
            self._terminal_events.append(event)
            changed = True
            outcome = _event_outcome(event)
            if outcome == "failed_energy":
                self._add_failed_energy_marker(event, key)
            # The event stream is cumulative, so this is emitted exactly once
            # per terminal event even when intermediate frames are repeated.
            print(f"[terminal-viewer-event] {self._event_line(event)}", flush=True)
        if changed:
            self._set_event_page(self._event_page)

    def _update_event_panel(self) -> None:
        total = len(self._terminal_events)
        page_count = max(1, int(np.ceil(total / _EVENT_PAGE_SIZE)))
        self._event_page = min(max(0, self._event_page), page_count - 1)
        start = self._event_page * _EVENT_PAGE_SIZE
        visible = list(reversed(self._terminal_events))[start:start + _EVENT_PAGE_SIZE]
        if visible:
            lines = [self._event_line(event) for event in visible]
            text = f"Terminal events {total} (page {self._event_page + 1}/{page_count})\n" + "\n".join(lines)
        else:
            text = "Terminal events 0"
        self._plotter.add_text(text, position="lower_left", font_size=9, color="black", name="terminal-events")

    def _update_uavs(
        self,
        snapshot: WorldRenderSnapshot,
        positions: tuple[tuple[float, float, float], ...] | None = None,
    ) -> None:
        def update_cloud(points: np.ndarray, colors: np.ndarray, actual_count: int) -> None:
            previous_count = getattr(self, "_uav_point_count", None)
            if previous_count != actual_count:
                self._uav_cloud.points = points
                self._uav_cloud.verts = _vertex_cells(actual_count)
            else:
                self._uav_cloud.points = points
            self._uav_cloud["rgb"] = colors
            self._uav_point_count = actual_count

        if not snapshot.uavs:
            update_cloud(
                np.zeros((1, 3), dtype=float),
                np.zeros((1, 3), dtype=np.uint8),
                0,
            )
            self._displayed_uav_positions = {}
            if self._uav_actor is not None:
                self._uav_actor.SetVisibility(False)
            self._update_label_group(
                "uav-labels", (), (), font_size=10, text_color="black", uids=(),
            )
            self._update_label_group(
                "uav-alert-labels-emergency", (), (), font_size=10, text_color="red", uids=(),
            )
            self._update_label_group(
                "uav-alert-labels-low_soc", (), (), font_size=10, text_color="darkorange", uids=(),
            )
            return
        if self._uav_actor is not None:
            self._uav_actor.SetVisibility(True)
        if positions is None:
            positions = tuple(uav.position for uav in snapshot.uavs)
        if len(positions) != len(snapshot.uavs):
            raise ValueError("positions must contain one point per UAV in snapshot order")
        points = np.asarray(positions, dtype=float)
        colors = np.empty((len(snapshot.uavs), 3), dtype=np.uint8)
        normal_points: list[tuple[float, float, float]] = []
        normal_labels: list[str] = []
        normal_uids: list[int] = []
        emergency_points: list[tuple[float, float, float]] = []
        emergency_labels: list[str] = []
        emergency_uids: list[int] = []
        low_soc_points: list[tuple[float, float, float]] = []
        low_soc_labels: list[str] = []
        low_soc_uids: list[int] = []
        recent_conflicts = set(getattr(snapshot, "recent_conflict_uids", ()) or ())
        for index, uav in enumerate(snapshot.uavs):
            alert = _uav_alert(uav)
            color = "magenta" if alert == "normal" and uav.uid in recent_conflicts else _uav_color(uav.uid, alert)
            colors[index] = self._pv.Color(color).int_rgb
            point = tuple(float(value) for value in points[index])
            if alert == "normal":
                if self._config.show_labels:
                    normal_points.append(point)
                    normal_labels.append(str(uav.uid))
                    normal_uids.append(uav.uid)
            elif alert == "emergency":
                emergency_points.append(point)
                rescue = getattr(uav, "rescue_status", None)
                emergency_labels.append(f"EMERGENCY{(' ' + rescue) if rescue else ''} UID={uav.uid} SOC={uav.soc:.0%}")
                emergency_uids.append(uav.uid)
            else:
                low_soc_points.append(point)
                low_soc_labels.append(f"LOW SOC UID={uav.uid} SOC={uav.soc:.0%}")
                low_soc_uids.append(uav.uid)
        update_cloud(points, colors, len(snapshot.uavs))
        self._displayed_uav_positions = {
            uav.uid: tuple(float(value) for value in points[index])
            for index, uav in enumerate(snapshot.uavs)
        }
        self._update_label_group(
            "uav-labels", normal_points, normal_labels, font_size=10, text_color="black",
            uids=tuple(normal_uids),
        )
        self._update_label_group(
            "uav-alert-labels-emergency", emergency_points, emergency_labels,
            font_size=10, text_color="red", uids=tuple(emergency_uids),
        )
        self._update_label_group(
            "uav-alert-labels-low_soc", low_soc_points, low_soc_labels,
            font_size=10, text_color="darkorange", uids=tuple(low_soc_uids),
        )

    def _update_uav_positions_only(
        self,
        snapshot: WorldRenderSnapshot,
        positions: tuple[tuple[float, float, float], ...],
    ) -> None:
        """Move existing UAV points/labels without rebuilding their content."""
        if len(positions) != len(snapshot.uavs):
            raise ValueError("positions must contain one point per UAV in snapshot order")
        if not snapshot.uavs:
            points = np.zeros((1, 3), dtype=float)
            actual_count = 0
            self._displayed_uav_positions = {}
            if self._uav_actor is not None:
                self._uav_actor.SetVisibility(False)
        else:
            points = np.asarray(positions, dtype=float)
            actual_count = len(snapshot.uavs)
            if self._uav_actor is not None:
                self._uav_actor.SetVisibility(True)
            self._displayed_uav_positions = {
                uav.uid: tuple(float(value) for value in points[index])
                for index, uav in enumerate(snapshot.uavs)
            }
        previous_count = getattr(self, "_uav_point_count", None)
        if previous_count != actual_count:
            self._uav_cloud.points = points
            self._uav_cloud.verts = _vertex_cells(actual_count)
        else:
            self._uav_cloud.points = points
        self._uav_point_count = actual_count

        by_uid = self._displayed_uav_positions
        for name, uids in getattr(self, "_label_uids", {}).items():
            dataset = getattr(self, "_label_datasets", {}).get(name)
            if dataset is None or not uids or not all(uid in by_uid for uid in uids):
                continue
            dataset.points = np.asarray([by_uid[uid] for uid in uids], dtype=float)

    def _update_trajectories(self, snapshot: WorldRenderSnapshot) -> None:
        if not self._config.show_trajectories:
            for name in tuple(self._trajectory_names):
                self._plotter.remove_actor(name, render=False)
            self._trajectory_names.clear()
            self._trajectory_meshes.clear()
            self._trajectory_actors.clear()
            self._trajectory_signatures.clear()
            return
        visible = set()
        for uav in snapshot.uavs:
            if len(uav.trajectory_points) < 2:
                continue
            name = f"trajectory-{uav.uid}"
            visible.add(name)
            points = np.asarray(uav.trajectory_points, dtype=float)
            displayed = self._displayed_uav_positions.get(uav.uid)
            if displayed is not None and len(points) > 0:
                points = points.copy()
                points[0] = np.asarray(displayed, dtype=float)
            signature = _trajectory_signature(uav)
            if self._trajectory_signatures.get(name) != signature:
                if name in self._trajectory_meshes:
                    self._plotter.remove_actor(name, render=False)
                mesh = self._line_mesh(points)
                self._trajectory_meshes[name] = mesh
                self._trajectory_actors[name] = self._plotter.add_mesh(
                    mesh,
                    color=_identity_color(uav.uid),
                    line_width=1.5,
                    opacity=0.58,
                    pickable=False,
                    render=False,
                    name=name,
                )
                self._trajectory_signatures[name] = signature
            else:
                self._trajectory_meshes[name].points = points
        for name in self._trajectory_names - visible:
            self._plotter.remove_actor(name, render=False)
            self._trajectory_meshes.pop(name, None)
            self._trajectory_actors.pop(name, None)
            self._trajectory_signatures.pop(name, None)
        self._trajectory_names = visible
        self._update_trajectory_styles()

    def _update_trajectory_heads(self, snapshot: WorldRenderSnapshot) -> None:
        """Move only the displayed endpoint during an interpolation frame."""
        if not self._config.show_trajectories:
            return
        for uav in snapshot.uavs:
            name = f"trajectory-{uav.uid}"
            mesh = self._trajectory_meshes.get(name)
            displayed = self._displayed_uav_positions.get(uav.uid)
            if mesh is None or displayed is None or mesh.n_points < 1:
                continue
            points = mesh.points.copy()
            points[0] = np.asarray(displayed, dtype=float)
            mesh.points = points

    def _update_trajectory_styles(self) -> None:
        for name, actor in self._trajectory_actors.items():
            uid = int(name.rsplit("-", 1)[-1])
            selected = uid == self._selected_uid
            actor.GetProperty().SetLineWidth(3.0 if selected else 1.5)
            actor.GetProperty().SetOpacity(0.95 if selected else 0.58)

    def _update_hud(self, snapshot: WorldRenderSnapshot) -> None:
        occupancy = "  ".join(f"{ring.role[0].upper()} {ring.occupancy}/{ring.capacity}" for ring in snapshot.rings)
        telemetry = snapshot.telemetry
        emergency_count = sum(_uav_alert(uav) == "emergency" for uav in snapshot.uavs)
        low_soc_count = sum(_uav_alert(uav) == "low_soc" for uav in snapshot.uavs)
        failed_energy_count = sum(
            _event_outcome(event) == "failed_energy" for event in _terminal_events(snapshot)
        )
        hard_separation = telemetry.get("hard_separation_violations", telemetry.get("hard_separation", 0))
        conflict = "  ".join(
            f"{key}={int(value)}" for key, value in (
                ("hard_separation", hard_separation),
                ("nmac", telemetry.get("nmac", 0)),
                ("collisions", telemetry.get("collisions", 0)),
            )
        )
        transition = f" transition={snapshot.transition_progress:.0%}" if snapshot.transition_active else ""
        gap = f"\n{self._source_gap_warning}" if self._source_gap_warning else ""
        text = (
            f"t = {snapshot.time:.1f} s{transition}\n"
            f"active = {len(snapshot.uavs)}  queue = {snapshot.external_queue_size}\n"
            f"service = {snapshot.service_count}  completed = {snapshot.completed_count}\n"
            f"emergency = {emergency_count}  low_soc = {low_soc_count}  failed_energy = {failed_energy_count}\n"
            f"{occupancy}\n{conflict}{gap}"
        )
        text_cache = getattr(self, "_text_cache", {})
        self._text_cache = text_cache
        if text_cache.get("hud") == text:
            return
        self._plotter.add_text(text, position="upper_right", font_size=11, color="black", name="hud")
        text_cache["hud"] = text

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def playback_speed(self) -> float:
        return self._playback_speed

    def wait_until_resumed(self, stop_event: Event | None = None) -> bool:
        """Keep the GUI responsive while deliberately holding future ticks."""
        while self.paused and not self.closed and (stop_event is None or not stop_event.is_set()):
            try:
                self._plotter.update()
            except RuntimeError:
                self._closed = True
            time.sleep(0.03)
        return not self.closed and (stop_event is None or not stop_event.is_set())

    @property
    def closed(self) -> bool:
        return self._closed or bool(getattr(self._plotter, "_closed", False))

    def update(
        self,
        snapshot: WorldRenderSnapshot,
        *,
        uav_positions: tuple[tuple[float, float, float], ...] | None = None,
        render: bool = True,
    ) -> bool:
        """Redraw the latest snapshot; return False when the window is closed."""
        if self.closed:
            return False
        previous_time = getattr(self._latest_snapshot, "time", None)
        current_time = getattr(snapshot, "time", None)
        if previous_time is not None and current_time is not None:
            gap = float(current_time) - float(previous_time)
            self._source_gap_warning = (
                f"Source gap {gap:.1f}s; interpolation off"
                if gap > _SOURCE_GAP_SECONDS else None
            )
        self._latest_snapshot = snapshot
        self._record_terminal_events(snapshot)
        self._update_rings(snapshot)
        self._update_landing_zones(snapshot)
        self._update_landing_zone_labels(snapshot)
        self._update_pad_status_panel(snapshot)
        self._update_uavs(snapshot, uav_positions)
        self._update_trajectories(snapshot)
        self._update_hud(snapshot)
        self._update_selected_uav_panel()
        if not render:
            return not self.closed
        try:
            if not self._shown:
                self._plotter.show(auto_close=False, interactive_update=True)
                self._shown = True
            self._plotter.update()
        except RuntimeError:  # PyVista raises this after a user closes the window.
            self._closed = True
        return not self.closed

    def update_interpolated_positions(
        self,
        snapshot: WorldRenderSnapshot,
        positions: tuple[tuple[float, float, float], ...],
    ) -> bool:
        """Render only UAV positions between two complete source snapshots.

        Rings, landing zones, trajectories, HUD text, and selection state are
        left untouched.  This keeps intermediate frames cheap while the
        source snapshot remains the authoritative metadata used by picking.
        """
        if self.closed:
            return False
        self._update_uav_positions_only(snapshot, positions)
        self._update_trajectory_heads(snapshot)
        try:
            self._plotter.update()
        except RuntimeError:
            self._closed = True
        return not self.closed

    def update_dynamic(
        self,
        snapshot: WorldRenderSnapshot,
        positions: tuple[tuple[float, float, float], ...],
        *,
        render: bool = True,
    ) -> bool:
        """Refresh dynamic source data without rebuilding static scene actors."""
        if self.closed:
            return False
        previous_time = getattr(self._latest_snapshot, "time", None)
        current_time = getattr(snapshot, "time", None)
        if previous_time is not None and current_time is not None:
            gap = float(current_time) - float(previous_time)
            self._source_gap_warning = (
                f"Source gap {gap:.1f}s; interpolation off"
                if gap > _SOURCE_GAP_SECONDS else None
            )
        self._latest_snapshot = snapshot
        self._record_terminal_events(snapshot)
        self._update_landing_zones(snapshot)
        self._update_landing_zone_labels(snapshot)
        self._update_pad_status_panel(snapshot)
        self._update_uavs(snapshot, positions)
        self._update_trajectories(snapshot)
        self._update_hud(snapshot)
        self._update_selected_uav_panel()
        if not render:
            return not self.closed
        try:
            self._plotter.update()
        except RuntimeError:
            self._closed = True
        return not self.closed

    def close(self) -> None:
        if not self.closed:
            self._plotter.close()
        self._closed = True


class ViewerStepObserver:
    """Latest-frame, non-blocking adapter from simulation to rendering."""

    def __init__(self, viewer: Terminal3DViewer, config: VisualizationConfig) -> None:
        self.viewer = viewer
        self.config = config
        self.stop_requested = False
        self._frames: Queue = Queue(maxsize=1)
        self._closed = Event()
        self._worker = Thread(target=self._render_loop, daemon=True, name="terminal-viewer")
        self._worker.start()

    def _render_loop(self) -> None:
        while not self._closed.is_set():
            try:
                snapshot = self._frames.get(timeout=0.05)
            except Empty:
                continue
            self.stop_requested = not self.viewer.update(snapshot)

    def close(self) -> None:
        self._closed.set()
        self._worker.join(timeout=1.0)

    def __call__(self, env: "TerminalParallelEnv") -> None:
        if env.step_count % self.config.frame_stride:
            return
        snapshot = build_world_render_snapshot(env, trajectory_max_points=self.config.trajectory_max_points)
        try:
            self._frames.put_nowait(snapshot)
        except Full:
            try:
                self._frames.get_nowait()
            except Empty:
                pass
            try:
                self._frames.put_nowait(snapshot)
            except Full:
                pass


class AsyncTerminalViewerObserver:
    """Own the complete VTK lifecycle on one dedicated render thread."""

    def __init__(self, initial: WorldRenderSnapshot, config: VisualizationConfig) -> None:
        self.config = config
        self.stop_requested = False
        self.error: Exception | None = None
        # These are intentionally advisory UI controls.  The simulation
        # thread never waits for rendering, but an operator can explicitly
        # request a paced or paused live playback.
        self.paused = False
        self.playback_speed = config.playback_speed
        self._initial = initial
        self._frames: Queue = Queue(maxsize=1)
        self._closed = Event()
        self._ready = Event()
        self._last_source = initial
        self._last_source_identity = _source_identity(initial)
        self._last_enqueued_at: float | None = None
        self._worker = Thread(target=self._render_loop, daemon=True, name="terminal-viewer")
        self._worker.start()

    @staticmethod
    def _transition_duration(previous_enqueued_at: float | None, enqueued_at: float) -> float:
        if previous_enqueued_at is None:
            return _MIN_INTERPOLATION_SECONDS
        elapsed = max(0.0, enqueued_at - previous_enqueued_at)
        return min(_MAX_INTERPOLATION_SECONDS, max(_MIN_INTERPOLATION_SECONDS, elapsed))

    def _drain_latest(self):
        """Take the newest queued source frame at one render-tick boundary."""
        try:
            latest = self._frames.get_nowait()
        except Empty:
            return None
        while True:
            try:
                latest = self._frames.get_nowait()
            except Empty:
                return latest

    def _render_loop(self) -> None:
        viewer = None
        transition: tuple[
            WorldRenderSnapshot,
            WorldRenderSnapshot,
            tuple[tuple[float, float, float], ...],
            float,
            float,
        ] | None = None
        try:
            viewer = Terminal3DViewer(self._initial, self.config)
            self._ready.set()
            next_tick = time.perf_counter()
            while not self._closed.is_set():
                now = time.perf_counter()
                if now < next_tick:
                    if self._closed.wait(next_tick - now):
                        break
                    continue
                next_tick += _INTERPOLATION_PERIOD
                if next_tick < now:
                    next_tick = now

                if viewer.paused:
                    self.paused = True
                    self.playback_speed = viewer.playback_speed
                    self.stop_requested = not viewer.wait_until_resumed(self._closed)
                    self.paused = viewer.paused
                    self.playback_speed = viewer.playback_speed
                    continue

                queued = self._drain_latest()
                now = time.perf_counter()
                displayed_positions = None
                previous_transition = transition
                if previous_transition is not None:
                    previous, current, previous_positions, started_at, duration = previous_transition
                    alpha = (now - started_at) / duration
                    displayed_positions = _interpolated_uav_positions(
                        previous,
                        current,
                        alpha,
                        previous_positions=previous_positions,
                    )

                if queued is not None:
                    if isinstance(queued, tuple) and len(queued) == 2 and isinstance(queued[1], float):
                        snapshot, enqueued_at = queued
                    else:  # pragma: no cover - compatibility for direct queue users
                        snapshot, enqueued_at = queued, now
                    previous = self._last_source
                    previous_positions = displayed_positions
                    if previous_positions is None:
                        previous_positions = tuple(uav.position for uav in previous.uavs)
                    start_positions = _interpolated_uav_positions(
                        previous,
                        snapshot,
                        0.0,
                        previous_positions=previous_positions,
                    )
                    source_identity = _source_identity(snapshot)
                    if source_identity != self._last_source_identity:
                        source_updated = viewer.update(
                            snapshot,
                            uav_positions=start_positions,
                            render=False,
                        )
                    else:
                        source_updated = viewer.update_dynamic(snapshot, start_positions, render=False)
                    self.stop_requested = not source_updated
                    if self.stop_requested:
                        continue
                    duration = self._transition_duration(self._last_enqueued_at, enqueued_at)
                    transition = (
                        previous,
                        snapshot,
                        previous_positions,
                        now,
                        duration,
                    )
                    self._last_source = snapshot
                    self._last_source_identity = source_identity
                    self._last_enqueued_at = enqueued_at
                    first_alpha = min(1.0, _INTERPOLATION_PERIOD / duration)
                    first_positions = _interpolated_uav_positions(
                        previous,
                        snapshot,
                        first_alpha,
                        previous_positions=previous_positions,
                    )
                    self.stop_requested = not viewer.update_interpolated_positions(snapshot, first_positions)
                    if first_alpha >= 1.0:
                        transition = None
                elif previous_transition is not None:
                    previous, current, previous_positions, started_at, duration = previous_transition
                    self.stop_requested = not viewer.update_interpolated_positions(current, displayed_positions)
                    if (now - started_at) / duration >= 1.0:
                        transition = None
                self.paused = viewer.paused
                self.playback_speed = viewer.playback_speed
        except Exception as error:  # surfaced to rollout/qualification callers
            self.error = error
            self.stop_requested = True
            self._ready.set()
        finally:
            if viewer is not None:
                viewer.close()

    def wait_ready(self, timeout: float = 10.0) -> bool:
        self._ready.wait(timeout)
        return self._ready.is_set() and self.error is None

    def close(self) -> None:
        self._closed.set()
        self._worker.join(timeout=2.0)

    def __call__(self, env: "TerminalParallelEnv") -> None:
        if env.step_count % self.config.frame_stride:
            return
        snapshot = build_world_render_snapshot(env, trajectory_max_points=self.config.trajectory_max_points)
        queued = (snapshot, time.perf_counter())
        try:
            self._frames.put_nowait(queued)
        except Full:
            try:
                self._frames.get_nowait()
            except Empty:
                pass
            try:
                self._frames.put_nowait(queued)
            except Full:
                pass


def _process_transition_seconds(
    previous: WorldRenderSnapshot,
    current: WorldRenderSnapshot,
    playback_speed: float,
) -> float:
    """Map the UI playback rate to one source-frame transition duration."""
    if playback_speed > 0.0:
        simulated_delta = max(float(current.time) - float(previous.time), 1e-3)
        return max(_INTERPOLATION_PERIOD, simulated_delta / playback_speed)
    return _INTERPOLATION_PERIOD


def _put_process_status(status_queue, payload: dict[str, object]) -> None:
    try:
        status_queue.put_nowait(payload)
    except Full:
        pass


def _buffered_process_render_main(
    frame_queue,
    stop_event,
    ready_queue,
    status_queue,
    paused_value,
    playback_speed_value,
    stop_requested_value,
    initial: WorldRenderSnapshot,
    config: VisualizationConfig,
    final_consumed_value=None,
) -> None:
    """Child-process entry point; all VTK objects stay inside this process."""
    viewer = None
    ready_sent = False
    try:
        viewer = Terminal3DViewer(initial, config)
        ready_queue.put({"ready": True, "error": None})
        ready_sent = True
        source_buffer: list[tuple[WorldRenderSnapshot, float, bool]] = []
        displayed_snapshot = initial
        displayed_positions = tuple(uav.position for uav in initial.uavs)
        active_transition = None
        playback_started = False
        prefill_started_at: float | None = None
        last_source_identity = _source_identity(initial)
        next_tick = time.perf_counter()
        while not stop_event.is_set():
            now = time.perf_counter()
            if now < next_tick:
                if stop_event.wait(next_tick - now):
                    break
                continue
            next_tick += _INTERPOLATION_PERIOD
            if next_tick < now:
                next_tick = now

            paused_value.value = bool(viewer.paused)
            playback_speed_value.value = float(viewer.playback_speed)
            if viewer.paused:
                resumed = viewer.wait_until_resumed(stop_event)
                paused_value.value = bool(viewer.paused)
                playback_speed_value.value = float(viewer.playback_speed)
                if not resumed:
                    stop_requested_value.value = True
                    break
                continue
            if viewer.closed:
                stop_requested_value.value = True
                break

            # Keep source order exact.  The producer applies back-pressure
            # when this bounded presentation buffer is full; dropping its
            # oldest item would make a terminal event disappear from the UI.
            while len(source_buffer) < _PROCESS_BUFFER_CAPACITY:
                try:
                    queued = frame_queue.get_nowait()
                except Empty:
                    break
                if isinstance(queued, tuple) and len(queued) == 3:
                    queued_snapshot, enqueued_at, is_final = queued
                else:  # compatibility with direct queue users/tests
                    queued_snapshot, enqueued_at = queued
                    is_final = False
                source_buffer.append((queued_snapshot, float(enqueued_at), bool(is_final)))
                if prefill_started_at is None:
                    prefill_started_at = now
            if not playback_started and source_buffer:
                if len(source_buffer) >= _PROCESS_BUFFER_PREFILL:
                    playback_started = True
                elif (
                    prefill_started_at is not None
                    and now - prefill_started_at >= _PROCESS_PREFILL_TIMEOUT_SECONDS
                ):
                    playback_started = True

            if active_transition is None and playback_started and source_buffer:
                target, _enqueued_at, is_final = source_buffer.pop(0)
                previous = displayed_snapshot
                previous_positions = displayed_positions
                start_positions = _interpolated_uav_positions(
                    previous,
                    target,
                    0.0,
                    previous_positions=previous_positions,
                )
                source_identity = _source_identity(target)
                if source_identity != last_source_identity:
                    source_updated = viewer.update(
                        target,
                        uav_positions=start_positions,
                        render=False,
                    )
                    last_source_identity = source_identity
                else:
                    source_updated = viewer.update_dynamic(
                        target,
                        start_positions,
                        render=False,
                    )
                if not source_updated:
                    stop_requested_value.value = True
                    break
                active_transition = (
                    previous,
                    target,
                    previous_positions,
                    now,
                    _process_transition_seconds(previous, target, playback_speed_value.value),
                    is_final,
                )

            if active_transition is not None:
                previous, target, previous_positions, started_at, duration, is_final = active_transition
                alpha = min(1.0, max(0.0, (now - started_at) / duration))
                displayed_positions = _interpolated_uav_positions(
                    previous,
                    target,
                    alpha,
                    previous_positions=previous_positions,
                )
                if not viewer.update_interpolated_positions(target, displayed_positions):
                    stop_requested_value.value = True
                    break
                displayed_snapshot = target
                if alpha >= 1.0:
                    active_transition = None
                    if is_final and final_consumed_value is not None:
                        final_consumed_value.value = True
            elif playback_started and not source_buffer:
                # Even after the terminal frame, keep pumping VTK so the
                # window remains responsive to close and keyboard events.
                if not viewer.update_interpolated_positions(displayed_snapshot, displayed_positions):
                    stop_requested_value.value = True
                    break
            paused_value.value = bool(viewer.paused)
            playback_speed_value.value = float(viewer.playback_speed)
    except Exception as error:  # pragma: no cover - exercised in child process
        stop_requested_value.value = True
        _put_process_status(status_queue, {"kind": "error", "error": repr(error)})
        if not ready_sent:
            try:
                ready_queue.put({"ready": False, "error": repr(error)})
            except Full:
                pass
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception as error:  # pragma: no cover - defensive child cleanup
                stop_requested_value.value = True
                _put_process_status(status_queue, {"kind": "error", "error": repr(error)})


class BufferedProcessTerminalViewerObserver:
    """Spawn-safe viewer with a bounded presentation buffer.

    The simulation process builds immutable snapshots and enqueues them in
    order with short back-pressure waits when the bounded buffer is full.  The
    child process owns every PyVista/VTK object and interpolates at a fixed
    visual cadence.
    """

    def __init__(self, initial: WorldRenderSnapshot, config: VisualizationConfig) -> None:
        self.config = config
        self._closed = False
        self._ready = False
        self._error: Exception | None = None
        self._context = mp.get_context("spawn")
        self._frames = self._context.Queue(maxsize=_PROCESS_BUFFER_CAPACITY)
        self._ready_queue = self._context.Queue(maxsize=1)
        self._status_queue = self._context.Queue(maxsize=2)
        self._stop_event = self._context.Event()
        self._paused_value = self._context.Value("b", False)
        self._playback_speed_value = self._context.Value("d", float(config.playback_speed))
        self._stop_requested_value = self._context.Value("b", False)
        self._final_consumed_value = self._context.Value("b", False)
        self._process = self._context.Process(
            target=_buffered_process_render_main,
            args=(
                self._frames,
                self._stop_event,
                self._ready_queue,
                self._status_queue,
                self._paused_value,
                self._playback_speed_value,
                self._stop_requested_value,
                _snapshot_for_process(initial),
                config,
                self._final_consumed_value,
            ),
            name="terminal-marl-viewer",
        )
        self._process_started = False
        try:
            self._process.start()
            self._process_started = True
        except Exception as error:
            self._error = error
            self._stop_requested_value.value = True

    def _refresh_status(self) -> None:
        if self._closed:
            return
        while True:
            try:
                status = self._status_queue.get_nowait()
            except Empty:
                break
            if status.get("kind") == "error":
                self._error = RuntimeError(str(status.get("error", "viewer process failed")))
                self._stop_requested_value.value = True
        if self._process_started and not self._closed:
            try:
                exitcode = self._process.exitcode
            except AssertionError:  # pragma: no cover - defensive startup race
                exitcode = None
            if exitcode is not None and not self._stop_requested_value.value:
                self._stop_requested_value.value = True
                if self._error is None:
                    self._error = RuntimeError(
                        f"viewer process exited unexpectedly (exitcode={exitcode})"
                    )

    @property
    def paused(self) -> bool:
        self._refresh_status()
        return bool(self._paused_value.value)

    @property
    def playback_speed(self) -> float:
        self._refresh_status()
        return float(self._playback_speed_value.value)

    @property
    def stop_requested(self) -> bool:
        self._refresh_status()
        return self._closed or bool(self._stop_requested_value.value)

    @property
    def error(self) -> Exception | None:
        self._refresh_status()
        return self._error

    def wait_ready(self, timeout: float = 10.0) -> bool:
        if self._ready:
            return self.error is None
        if not self._process_started:
            return False
        try:
            result = self._ready_queue.get(timeout=timeout)
        except Empty:
            if not self._process.is_alive() and self._error is None:
                self._error = RuntimeError("viewer process exited before becoming ready")
            return False
        self._ready = True
        if not bool(result.get("ready", False)):
            self._error = RuntimeError(str(result.get("error", "viewer process failed to start")))
            self._stop_requested_value.value = True
            return False
        return self.error is None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._process_started:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
        for queue in (self._frames, self._ready_queue, self._status_queue):
            try:
                # A feeder can still own queued snapshots after the child has
                # exited.  Waiting for it can block forever when the pipe no
                # longer has a reader, so detach every queue before closing.
                queue.cancel_join_thread()
                queue.close()
            except (AssertionError, AttributeError, OSError, ValueError):
                pass

    def _enqueue(self, snapshot: WorldRenderSnapshot, *, final: bool = False) -> bool:
        """Put a frame in order, waiting briefly when the bounded queue is full."""
        queued = (snapshot, time.perf_counter(), bool(final))
        while not self._closed:
            if self.stop_requested or self.error is not None:
                return False
            try:
                self._frames.put(queued, timeout=0.05)
                return True
            except Full:
                self._refresh_status()
                if self.stop_requested or self.error is not None:
                    return False
        return False

    def finish(self, env: "TerminalParallelEnv", *, wait_for_close: bool = True) -> bool:
        """Deliver the terminal frame, then keep a normal GUI open for review."""
        if self._closed or self.stop_requested:
            return False
        snapshot = _snapshot_for_process(
            build_world_render_snapshot(env, trajectory_max_points=self.config.trajectory_max_points)
        )
        if not self._enqueue(snapshot, final=True):
            return False
        while not self._closed and not self.stop_requested and not self.error:
            if self._final_consumed_value.value or not self._process.is_alive():
                break
            time.sleep(0.03)
        if not wait_for_close or os.environ.get("PYVISTA_OFF_SCREEN", "").lower() in {"1", "true", "yes"}:
            return bool(self._final_consumed_value.value)
        # A normal interactive run stays visible until the operator closes
        # it.  Closing the window sets stop_requested in the child process.
        while not self._closed and not self.stop_requested and not self.error and self._process.is_alive():
            time.sleep(0.05)
        return bool(self._final_consumed_value.value)

    def __call__(self, env: "TerminalParallelEnv") -> None:
        if self._closed or self.stop_requested:
            return
        if env.step_count % self.config.frame_stride:
            return
        snapshot = _snapshot_for_process(
            build_world_render_snapshot(env, trajectory_max_points=self.config.trajectory_max_points)
        )
        # Keep the frame order exact so cumulative terminal events cannot be
        # hidden by replacing an older source snapshot.
        self._enqueue(snapshot)
