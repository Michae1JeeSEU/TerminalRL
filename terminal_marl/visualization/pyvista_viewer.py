"""PyVista implementation of the optional, snapshot-only terminal viewer."""
from __future__ import annotations

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


def _identity_color(uid: int) -> str:
    """Stable UAV/trajectory color shared across the complete lifecycle."""
    return _UAV_COLORS[int(uid) % len(_UAV_COLORS)]


def _trajectory_signature(uav) -> tuple[str | None, int]:
    """Topology identity used to rebuild a line when its committed plan changes."""
    return uav.trajectory_plan_id, len(uav.trajectory_points)


class Terminal3DViewer:
    """Interactive 3D renderer which consumes immutable world snapshots only."""

    def __init__(self, initial: WorldRenderSnapshot, config: VisualizationConfig) -> None:
        try:
            import pyvista as pv
        except ImportError as error:  # pragma: no cover - depends on local optional package
            raise RuntimeError(
                "Visualization dependencies are not installed. "
                "Install requirements-visualization.txt and retry."
            ) from error
        self._pv = pv
        self._config = config
        self._plotter = pv.Plotter(window_size=(1280, 860), title="Terminal MARL 3D Viewer")
        self._plotter.set_background("white")
        self._ring_meshes: dict[str, object] = {}
        self._target_ring_meshes: dict[str, object] = {}
        self._trajectory_names: set[str] = set()
        self._trajectory_meshes: dict[str, object] = {}
        self._trajectory_actors: dict[str, object] = {}
        self._trajectory_signatures: dict[str, tuple[str | None, int]] = {}
        self._lz_actors: dict[int, object] = {}
        # PyVista rejects zero-length point arrays during construction.  A
        # hidden one-point backing mesh keeps the actor persistent even when
        # the simulation temporarily has no airborne UAVs.
        self._uav_cloud = self._pv.PolyData(np.zeros((1, 3), dtype=float))
        self._uav_cloud["rgb"] = np.zeros((1, 3), dtype=np.uint8)
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
                font_size=12, text_color="black", shape=None, always_visible=True, name=f"gate-label-{gate.gate_id}",
            )
        self._plotter.add_axes(color="black", x_color="firebrick", y_color="forestgreen", z_color="navy")
        self._plotter.add_text("", position="upper_right", font_size=11, color="black", name="hud")
        self._plotter.add_text("Right-click a UAV point to inspect", position=(10, 765), font_size=10, color="black", name="pick-help")
        for lz in snapshot.landing_zones:
            actor = self._plotter.add_mesh(self._pv.Cube(center=lz.position, x_length=45, y_length=45, z_length=6),
                                           color=_LZ_COLORS.get(lz.status, "white"), opacity=0.9, pickable=False,
                                           name=f"lz-{lz.lz_id}")
            self._lz_actors[lz.lz_id] = actor
        self._uav_actor = self._plotter.add_mesh(self._uav_cloud, scalars="rgb", rgb=True, point_size=15,
                                                 render_points_as_spheres=True, pickable=True, name="uavs")

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
        self._plotter.add_text("Speed", position=(112, 800), font_size=11, color="black", name="speed-label")
        self._plotter.add_text_slider_widget(
            self._set_playback_speed, list(_PLAYBACK_OPTIONS),
            value=float(_PLAYBACK_OPTIONS.index(self._speed_label(self._playback_speed))),
            pointa=(0.18, 0.93), pointb=(0.43, 0.93), color="black", style="modern",
        )
        self._plotter.enable_point_picking(
            self._select_uav, left_clicking=False, picker="point", show_message=False,
            show_point=False, tolerance=0.025, use_picker=True,
        )

    def _set_paused(self, value: bool) -> None:
        self._paused = bool(value)

    def _set_playback_speed(self, value: str) -> None:
        self._playback_speed = _PLAYBACK_VALUES[str(value)]

    @staticmethod
    def _uav_uid_from_point_id(snapshot: WorldRenderSnapshot, point_id: int) -> int | None:
        """Map PyVista's current point-cloud index to snapshot identity."""
        return snapshot.uavs[point_id].uid if 0 <= point_id < len(snapshot.uavs) else None

    def _select_uav(self, picked_point: np.ndarray, picker) -> None:
        """Select a point in the current UAV cloud, never an obscuring mesh."""
        if not self._latest_snapshot.uavs or self._uav_cloud is None:
            return
        point_id = int(picker.GetPointId())
        self._selected_uid = self._uav_uid_from_point_id(self._latest_snapshot, point_id)
        self._update_selected_uav_panel()
        self._update_trajectory_styles()

    def _selected_uav(self):
        return next((uav for uav in self._latest_snapshot.uavs if uav.uid == self._selected_uid), None)

    def _update_selected_uav_panel(self) -> None:
        uav = self._selected_uav()
        if uav is None:
            text = "Selected UAV: —"
        else:
            energy = f"{uav.energy_remaining:.0f}/{uav.energy_capacity:.0f} ({uav.soc:.0%})"
            text = (
                f"Selected UAV\n"
                f"UID: {uav.uid}\n"
                f"Arrival ID: {uav.arrival_id}\n"
                f"Phase: {uav.phase}\n"
                f"Ring: {uav.ring_role or '—'}\n"
                f"Speed: {uav.speed:.1f} m/s\n"
                f"Energy: {energy}\n"
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
        self._plotter.add_text(text, position="lower_right", font_size=10, color="black", name="selected-uav")

    def _update_rings(self, snapshot: WorldRenderSnapshot) -> None:
        for ring in snapshot.rings:
            self._ring_meshes[ring.role].points = self._circle(ring.radius, ring.height)
            if self._config.show_target_geometry:
                self._target_ring_meshes[ring.role].points = self._circle(ring.target_radius, ring.target_height)

    def _update_landing_zones(self, snapshot: WorldRenderSnapshot) -> None:
        for lz in snapshot.landing_zones:
            actor = self._lz_actors[lz.lz_id]
            actor.GetProperty().SetColor(self._pv.Color(_LZ_COLORS.get(lz.status, "white")).float_rgb)

    def _update_uavs(self, snapshot: WorldRenderSnapshot) -> None:
        if not snapshot.uavs:
            self._uav_cloud.points = np.zeros((1, 3), dtype=float)
            self._uav_cloud["rgb"] = np.zeros((1, 3), dtype=np.uint8)
            if self._uav_actor is not None:
                self._uav_actor.SetVisibility(False)
            self._plotter.remove_actor("uav-labels", render=False)
            return
        if self._uav_actor is not None:
            self._uav_actor.SetVisibility(True)
        points = np.asarray([uav.position for uav in snapshot.uavs], dtype=float)
        colors = np.empty((len(snapshot.uavs), 3), dtype=np.uint8)
        # An aircraft keeps this deterministic identity color throughout its
        # entire life cycle.  Phase is conveyed by the selected-state panel.
        for index, uav in enumerate(snapshot.uavs):
            color = "magenta" if uav.uid in snapshot.recent_conflict_uids else _identity_color(uav.uid)
            colors[index] = self._pv.Color(color).int_rgb
        self._uav_cloud.points = points
        self._uav_cloud["rgb"] = colors
        self._plotter.remove_actor("uav-labels", render=False)
        if self._config.show_labels:
            self._plotter.add_point_labels(points, [str(uav.uid) for uav in snapshot.uavs], point_size=0,
                                           font_size=10, text_color="black", shape=None, always_visible=True, name="uav-labels")

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
            signature = _trajectory_signature(uav)
            if self._trajectory_signatures.get(name) != signature:
                if name in self._trajectory_meshes:
                    self._plotter.remove_actor(name, render=False)
                mesh = self._line_mesh(points)
                self._trajectory_meshes[name] = mesh
                self._trajectory_actors[name] = self._plotter.add_mesh(
                    mesh,
                    color=_identity_color(uav.uid),
                    line_width=2,
                    opacity=0.58,
                    pickable=False,
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

    def _update_trajectory_styles(self) -> None:
        for name, actor in self._trajectory_actors.items():
            uid = int(name.rsplit("-", 1)[-1])
            selected = uid == self._selected_uid
            actor.GetProperty().SetLineWidth(5 if selected else 2)
            actor.GetProperty().SetOpacity(0.95 if selected else 0.58)

    def _update_hud(self, snapshot: WorldRenderSnapshot) -> None:
        occupancy = "  ".join(f"{ring.role[0].upper()} {ring.occupancy}/{ring.capacity}" for ring in snapshot.rings)
        telemetry = snapshot.telemetry
        conflict = "  ".join(
            f"{key}={int(telemetry.get(key, 0))}" for key in ("hard_separation", "nmac", "collisions")
        )
        transition = f" transition={snapshot.transition_progress:.0%}" if snapshot.transition_active else ""
        text = (
            f"t = {snapshot.time:.1f} s{transition}\n"
            f"active = {len(snapshot.uavs)}  queue = {snapshot.external_queue_size}\n"
            f"service = {snapshot.service_count}  completed = {snapshot.completed_count}\n"
            f"{occupancy}\n{conflict}"
        )
        self._plotter.add_text(text, position="upper_right", font_size=11, color="black", name="hud")

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def playback_speed(self) -> float:
        return self._playback_speed

    def wait_until_resumed(self) -> bool:
        """Keep the GUI responsive while deliberately holding future ticks."""
        while self.paused and not self.closed:
            try:
                self._plotter.update()
            except RuntimeError:
                self._closed = True
            time.sleep(0.03)
        return not self.closed

    @property
    def closed(self) -> bool:
        return self._closed or bool(getattr(self._plotter, "_closed", False))

    def update(self, snapshot: WorldRenderSnapshot) -> bool:
        """Redraw the latest snapshot; return False when the window is closed."""
        if self.closed:
            return False
        self._latest_snapshot = snapshot
        self._update_rings(snapshot)
        self._update_landing_zones(snapshot)
        self._update_uavs(snapshot)
        self._update_trajectories(snapshot)
        self._update_hud(snapshot)
        self._update_selected_uav_panel()
        try:
            if not self._shown:
                self._plotter.show(auto_close=False, interactive_update=True)
                self._shown = True
            self._plotter.render()
            self._plotter.update()
        except RuntimeError:  # PyVista raises this after a user closes the window.
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
        self._worker = Thread(target=self._render_loop, daemon=True, name="terminal-viewer")
        self._worker.start()

    def _render_loop(self) -> None:
        viewer = None
        try:
            viewer = Terminal3DViewer(self._initial, self.config)
            self._ready.set()
            while not self._closed.is_set():
                try:
                    snapshot = self._frames.get(timeout=0.05)
                except Empty:
                    if viewer.paused:
                        self.paused = True
                        self.playback_speed = viewer.playback_speed
                        self.stop_requested = not viewer.wait_until_resumed()
                        self.paused = viewer.paused
                        self.playback_speed = viewer.playback_speed
                    continue
                self.stop_requested = not viewer.update(snapshot)
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
