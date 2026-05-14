"""Generate static summary of Phase 3a landing with key frames."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from pathlib import Path

from src.config import EnvConfig
from src.env.terminal_area_env import TerminalAreaEnv
from src.train_phase3a import Phase3aWrapper
from src.baselines.heuristic_policy import HeuristicPolicy


def record_keyframes(n_background: int = 10, seed: int = 42):
    config = EnvConfig()
    config.num_uavs = 0
    config.max_uavs = 25
    config.sim.max_steps = 400

    env = TerminalAreaEnv(config)
    wrapper = Phase3aWrapper(env, n_background=n_background)
    policy = HeuristicPolicy(config)
    obs, _ = wrapper.reset(seed=seed)

    # Key frames: step 0 (entry), step 10, step 20, step 30, step 40 (landed)
    key_steps = [0, 10, 20, 30, 40]
    keyframes = {}
    step = 0

    while True:
        if step in key_steps:
            uavs = wrapper.unwrapped.uav_manager.uavs
            keyframes[step] = {
                "positions": np.array([u.position.copy() for u in uavs]),
                "ring_ids": np.array([u.ring_id for u in uavs]),
                "statuses": np.array([u.status.value for u in uavs]),
                "controlled": np.array([u.is_controlled for u in uavs]),
            }

        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = wrapper.step(action)
        step += 1
        if terminated or truncated:
            if step not in keyframes:
                uavs = wrapper.unwrapped.uav_manager.uavs
                keyframes[step] = {
                    "positions": np.array([u.position.copy() for u in uavs]),
                    "ring_ids": np.array([u.ring_id for u in uavs]),
                    "statuses": np.array([u.status.value for u in uavs]),
                    "controlled": np.array([u.is_controlled for u in uavs]),
                }
            break

    return keyframes, env.airspace


def main():
    keyframes, airspace = record_keyframes(n_background=10, seed=42)
    radii = airspace.radii
    heights = airspace.heights

    titles = {
        0: "Step 0: Entry UAV spawns at Ring 1 entry point",
        10: "Step 10: Descending through Ring 1 → Ring 2",
        20: "Step 20: Approaching Ring 3",
        30: "Step 30: On Ring 3, initiating landing",
        40: "Step 40: Landed at landing zone!",
    }

    colors = ['#3498db', '#f39c12', '#2ecc71']
    ring_labels = ['Ring 1\n(R=4000, H=800m)', 'Ring 2\n(R=2500, H=500m)', 'Ring 3\n(R=1000, H=200m)']

    n_frames = len(keyframes)
    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 6, 5.5), dpi=100)
    if n_frames == 1:
        axes = [axes]

    for ax, (s, title) in zip(axes, titles.items()):
        if s not in keyframes:
            continue
        frame = keyframes[s]
        positions = frame["positions"]
        ring_ids = frame["ring_ids"]
        statuses = frame["statuses"]
        controlled = frame["controlled"]

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')

        # Draw rings
        for j, (r, c, label) in enumerate(zip(radii, colors, ring_labels)):
            circle = Circle((0, 0), r, fill=False, color=c, linewidth=1.5, alpha=0.4, linestyle='--')
            ax.add_patch(circle)
            if j == 0:
                ax.text(r * 0.75, r * 0.15, label, fontsize=6, color=c, ha='center')

        # Plot UAVs
        for i, (pos, rid, status, ctrl) in enumerate(zip(positions, ring_ids, statuses, controlled)):
            if np.all(pos == 0):
                continue
            x, y = pos[0], pos[1]

            if ctrl:
                color = '#e74c3c'
                size = 120
                marker = 'D'
                edgecolor = 'darkred'
                lw = 1.5
                zord = 10
            else:
                size = 50
                marker = 'o'
                edgecolor = 'none'
                lw = 0
                zord = 5
                if rid == 1:
                    color = '#3498db'
                elif rid == 2:
                    color = '#f39c12'
                else:
                    color = '#2ecc71'

            if status == 2:
                color = '#27ae60'
                size = 200
                marker = '*'
                edgecolor = 'darkgreen'
                lw = 2

            ax.scatter(x, y, c=color, s=size, marker=marker,
                      edgecolors=edgecolor, linewidth=lw, alpha=0.9, zorder=zord)

        # Landing zones
        m = 4
        for lz in range(m):
            angle_lz = 2 * np.pi * lz / m
            lz_x = radii[2] * 0.5 * np.cos(angle_lz)
            lz_y = radii[2] * 0.5 * np.sin(angle_lz)
            ax.scatter(lz_x, lz_y, c='#8e44ad', marker='s', s=50, alpha=0.4, zorder=3)

        ax.set_xlim(-radii[0] * 1.15, radii[0] * 1.15)
        ax.set_ylim(-radii[0] * 1.15, radii[0] * 1.15)

    # Shared legend below
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c',
               markersize=10, markeredgecolor='darkred', label='Entry UAV (Controlled)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=8, label='Background UAV'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#27ae60',
               markersize=12, label='Landed'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#8e44ad',
               markersize=8, label='Landing Zone'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9, framealpha=0.9)

    fig.suptitle("Phase 3a: Single UAV Landing with 10 Background Circling UAVs (Heuristic Policy)",
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])

    Path("results").mkdir(parents=True, exist_ok=True)
    output_path = "results/phase3a_keyframes.png"
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
