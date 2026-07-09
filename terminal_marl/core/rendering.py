
from __future__ import annotations

from typing import Iterable, Optional
import numpy as np
import matplotlib.pyplot as plt

from .config import SimConfig
from .baseline import gate_positions
from .uav import UAVState


def _circle_xyz(radius: float, height: float, n: int = 240):
    theta = np.linspace(0, 2 * np.pi, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(theta, fill_value=height, dtype=float)
    return x, y, z


def plot_snapshot(config: SimConfig, aircraft: Iterable[UAVState], save_path: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for ring in config.rings:
        x, y, z = _circle_xyz(ring.radius, ring.height)
        ax.plot(x, y, z, linewidth=1.5, label=f"Ring {ring.index}")

    x, y, z = _circle_xyz(config.approach_threshold_radius, config.gate_height)
    ax.plot(x, y, z, linestyle="--", linewidth=1.0, label="Approach threshold")

    gates = gate_positions(config)
    ax.scatter([g[0] for g in gates], [g[1] for g in gates], [g[2] for g in gates], s=50, marker="^", label="Gates")
    ax.scatter([0], [0], [0], s=80, marker="s", label="Vertiport")

    for ac in aircraft:
        hist = np.asarray(ac.history)
        if len(hist) > 1:
            ax.plot(hist[:, 0], hist[:, 1], hist[:, 2], linewidth=1.0)
        ax.scatter([ac.position[0]], [ac.position[1]], [ac.position[2]], s=35)
        ax.text(ac.position[0], ac.position[1], ac.position[2] + 20.0, f"{ac.uid}")

    lim = config.approach_threshold_radius + config.spawn_margin + 150
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, max(650.0, config.spawn_altitude + 40.0))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Terminal Area MARL Snapshot")
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180)
    else:
        plt.show()
