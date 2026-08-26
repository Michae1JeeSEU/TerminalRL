"""PyVista implementation of the optional, snapshot-only terminal viewer."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from .config import VisualizationConfig
from .schema import WorldRenderSnapshot
from .snapshot import build_world_render_snapshot

if TYPE_CHECKING:
    from ..envs.parallel_env import TerminalParallelEnv


_PHASE_COLORS = {
    "inbound": "deepskyblue",
    "ring_outer": "royalblue",
    "transfer_o2m": "mediumpurple",
    "ring_middle": "darkorange",
    "transfer_m2i": "gold",
    "ring_inner": "limegreen",
    "final": "crimson",
}
_LZ_COLORS = {"free": "seagreen", "reserved": "gold", "occupied": "tomato"}


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
        self._plotter.set_background("midnightblue")
        self._ring_meshes: dict[str, object] = {}
        self._target_ring_meshes: dict[str, object] = {}
        self._trajectory_names: set[str] = set()
        self._shown = False
        self._closed = False
        self._build_static_scene(initial)
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
            self._plotter.add_mesh(mesh, color="white", line_width=3, name=f"ring-{ring.role}")
            if self._config.show_target_geometry:
                target = self._line_mesh(self._circle(ring.target_radius, ring.target_height), close=True)
                self._target_ring_meshes[ring.role] = target
                self._plotter.add_mesh(target, color="lightgray", opacity=0.28, line_width=1, name=f"target-ring-{ring.role}")
        for gate in snapshot.gates:
            points = (gate.spawn_position, gate.gate_position)
            self._plotter.add_mesh(self._line_mesh(points), color="cyan", line_width=4, name=f"gate-{gate.gate_id}")
            self._plotter.add_point_labels(
                np.asarray([gate.gate_position]), [f"Gate {gate.gate_id}"], point_size=0,
                font_size=12, shape=None, always_visible=True, name=f"gate-label-{gate.gate_id}",
            )
        self._plotter.show_axes()
        self._plotter.add_text("", position="upper_right", font_size=11, name="hud")

    def _update_rings(self, snapshot: WorldRenderSnapshot) -> None:
        for ring in snapshot.rings:
            self._ring_meshes[ring.role].points = self._circle(ring.radius, ring.height)
            if self._config.show_target_geometry:
                self._target_ring_meshes[ring.role].points = self._circle(ring.target_radius, ring.target_height)

    def _update_landing_zones(self, snapshot: WorldRenderSnapshot) -> None:
        for lz in snapshot.landing_zones:
            name = f"lz-{lz.lz_id}"
            self._plotter.remove_actor(name, render=False)
            self._plotter.add_mesh(
                self._pv.Cube(center=lz.position, x_length=45, y_length=45, z_length=6),
                color=_LZ_COLORS.get(lz.status, "white"), opacity=0.9, name=name,
            )

    def _update_uavs(self, snapshot: WorldRenderSnapshot) -> None:
        self._plotter.remove_actor("uavs", render=False)
        if not snapshot.uavs:
            return
        points = np.asarray([uav.position for uav in snapshot.uavs], dtype=float)
        cloud = self._pv.PolyData(points)
        colors = np.asarray(
            [(255, 0, 255) if uav.uid in snapshot.recent_conflict_uids else (255, 255, 255)
             for uav in snapshot.uavs], dtype=np.uint8,
        )
        # Phase remains available through the labels; a conflict event has
        # visual precedence so it can be spotted without recomputing safety.
        for index, uav in enumerate(snapshot.uavs):
            if uav.uid not in snapshot.recent_conflict_uids:
                color = _PHASE_COLORS.get(uav.phase, "white")
                colors[index] = self._pv.Color(color).int_rgb
        cloud["rgb"] = colors
        self._plotter.add_mesh(cloud, scalars="rgb", rgb=True, point_size=15, render_points_as_spheres=True, name="uavs")
        self._plotter.remove_actor("uav-labels", render=False)
        if self._config.show_labels:
            self._plotter.add_point_labels(points, [str(uav.uid) for uav in snapshot.uavs], point_size=0,
                                           font_size=10, shape=None, always_visible=True, name="uav-labels")

    def _update_trajectories(self, snapshot: WorldRenderSnapshot) -> None:
        for name in self._trajectory_names:
            self._plotter.remove_actor(name, render=False)
        self._trajectory_names.clear()
        if not self._config.show_trajectories:
            return
        for uav in snapshot.uavs:
            if len(uav.trajectory_points) < 2:
                continue
            name = f"trajectory-{uav.uid}"
            self._plotter.add_mesh(self._line_mesh(uav.trajectory_points), color=_PHASE_COLORS.get(uav.phase, "white"),
                                   line_width=2, opacity=0.75, name=name)
            self._trajectory_names.add(name)

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
        self._plotter.add_text(text, position="upper_right", font_size=11, name="hud")

    @property
    def closed(self) -> bool:
        return self._closed or bool(getattr(self._plotter, "_closed", False))

    def update(self, snapshot: WorldRenderSnapshot) -> bool:
        """Redraw the latest snapshot; return False when the window is closed."""
        if self.closed:
            return False
        self._update_rings(snapshot)
        self._update_landing_zones(snapshot)
        self._update_uavs(snapshot)
        self._update_trajectories(snapshot)
        self._update_hud(snapshot)
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
    """Frame-strided adapter from post-tick environment callbacks to a viewer."""

    def __init__(self, viewer: Terminal3DViewer, config: VisualizationConfig) -> None:
        self.viewer = viewer
        self.config = config
        self.stop_requested = False

    def __call__(self, env: "TerminalParallelEnv") -> None:
        if env.step_count % self.config.frame_stride:
            return
        self.stop_requested = not self.viewer.update(build_world_render_snapshot(env))
        if not self.stop_requested and self.config.playback_speed > 0.0:
            time.sleep(env.config.dt / self.config.playback_speed)
