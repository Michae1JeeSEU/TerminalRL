"""Visualize Phase 3a: single UAV landing with background circling UAVs."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from src.config import EnvConfig
from src.env.terminal_area_env import TerminalAreaEnv
from src.train_phase3a import Phase3aWrapper
from src.baselines.heuristic_policy import HeuristicPolicy


def record_episode(n_background: int = 10, seed: int = 42):
    """Record an episode with heuristic policy, tracking all positions."""
    config = EnvConfig()
    config.num_uavs = 0
    config.max_uavs = 25
    config.sim.max_steps = 400

    env = TerminalAreaEnv(config)
    wrapper = Phase3aWrapper(env, n_background=n_background)
    policy = HeuristicPolicy(config)

    obs, _ = wrapper.reset(seed=seed)

    frames = []
    step = 0
    while True:
        # Record current state
        uavs = wrapper.unwrapped.uav_manager.uavs
        frame = {
            "step": step,
            "positions": np.array([u.position.copy() for u in uavs]),
            "ring_ids": np.array([u.ring_id for u in uavs]),
            "statuses": np.array([u.status.value for u in uavs]),
            "controlled": np.array([u.is_controlled for u in uavs]),
            "angles": np.array([u.angle for u in uavs]),
        }
        frames.append(frame)

        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = wrapper.step(action)
        step += 1

        if terminated or truncated:
            # Record final state
            uavs = wrapper.unwrapped.uav_manager.uavs
            frame = {
                "step": step,
                "positions": np.array([u.position.copy() for u in uavs]),
                "ring_ids": np.array([u.ring_id for u in uavs]),
                "statuses": np.array([u.status.value for u in uavs]),
                "controlled": np.array([u.is_controlled for u in uavs]),
                "angles": np.array([u.angle for u in uavs]),
            }
            frames.append(frame)
            break

    print(f"Episode: {step} steps, landed={info.get('landed', 0)}, conflicts={info.get('conflicts', 0)}")
    return frames, env.airspace


def create_animation(frames, airspace, n_background: int, output_path: str):
    """Create top-down animation of the episode (smaller file size)."""
    radii = airspace.radii
    heights = airspace.heights

    # Subsample: show every 2nd frame
    frames_sub = frames[::2]
    n_frames = len(frames_sub)
    print(f"Creating animation with {n_frames} frames (subsampled from {len(frames)})...")

    # Pre-compute trajectory of controlled UAV
    entry_positions_all = []
    for f in frames:
        for i, controlled in enumerate(f["controlled"]):
            if controlled:
                entry_positions_all.append(f["positions"][i])
                break
    entry_positions_all = np.array(entry_positions_all)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
    colors = ['#3498db', '#f39c12', '#2ecc71']
    ring_labels = ['Ring 1 (R=4000, H=800)', 'Ring 2 (R=2500, H=500)', 'Ring 3 (R=1000, H=200)']

    def update(frame_idx):
        ax.clear()

        frame = frames_sub[frame_idx]
        positions = frame["positions"]
        ring_ids = frame["ring_ids"]
        statuses = frame["statuses"]
        controlled = frame["controlled"]
        step = frame["step"]

        ax.set_title(f"Phase 3a: Single UAV Landing — Step {step} (t={step*2}s)", fontsize=14, fontweight='bold')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')

        # Draw rings
        for j, (r, c, label) in enumerate(zip(radii, colors, ring_labels)):
            circle = Circle((0, 0), r, fill=False, color=c, linewidth=1.5, alpha=0.5, linestyle='--')
            ax.add_patch(circle)
            ax.text(r * 0.82, -150, label, fontsize=8, color=c, ha='right')

        # Plot background UAVs
        for i, (pos, rid, status, ctrl) in enumerate(zip(positions, ring_ids, statuses, controlled)):
            if np.all(pos == 0):
                continue
            x, y = pos[0], pos[1]

            if ctrl:  # Entry UAV
                color = '#e74c3c'
                size = 100
                marker = 'D'
                edgecolor = 'darkred'
                lw = 1.5
                zord = 10
            else:
                size = 40
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

            if status == 2 and ctrl:  # LANDED entry
                color = '#27ae60'
                size = 150
                marker = '*'
                edgecolor = 'darkgreen'
                lw = 2

            ax.scatter(x, y, c=color, s=size, marker=marker,
                      edgecolors=edgecolor, linewidth=lw, alpha=0.9, zorder=zord)

        # Draw entry UAV trajectory up to current frame (in full-res)
        full_idx = frame_idx * 2  # approximate mapping to original frames
        if full_idx < len(entry_positions_all):
            traj = entry_positions_all[:full_idx + 1]
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1], color='#e74c3c', linewidth=2.5, alpha=0.7, zorder=4)

        # Landing zones
        m = 4
        for lz in range(m):
            angle_lz = 2 * np.pi * lz / m
            lz_x = radii[2] * 0.55 * np.cos(angle_lz)
            lz_y = radii[2] * 0.55 * np.sin(angle_lz)
            ax.scatter(lz_x, lz_y, c='#8e44ad', marker='s', s=60, alpha=0.5, zorder=3)

        # Entry points
        for ea in [0, np.pi/2, np.pi, 3*np.pi/2]:
            ep_x = radii[0] * np.cos(ea)
            ep_y = radii[0] * np.sin(ea)
            ax.scatter(ep_x, ep_y, c='gray', marker='x', s=30, alpha=0.4)

        ax.set_xlim(-radii[0] * 1.15, radii[0] * 1.15)
        ax.set_ylim(-radii[0] * 1.15, radii[0] * 1.15)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c',
                   markersize=10, markeredgecolor='darkred', label='Entry UAV'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
                   markersize=8, label='BG Ring 1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12',
                   markersize=8, label='BG Ring 2'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
                   markersize=8, label='BG Ring 3'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#27ae60',
                   markersize=12, label='Landed'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#8e44ad',
                   markersize=8, label='Landing Zone'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.85)

        # Status
        n_bg = sum(1 for c in controlled if not c)
        ax.text(0.02, 0.98, f"BG UAVs: {n_bg} | Step: {step}/400 | dt=2s",
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        return []

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=150, blit=False, repeat=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=8)
    ani.save(output_path, writer=writer)
    print(f"Animation saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    n_bg = 10
    frames, airspace = record_episode(n_background=n_bg, seed=42)
    create_animation(frames, airspace, n_bg, "results/phase3a_demo.gif")
    print("Done!")
