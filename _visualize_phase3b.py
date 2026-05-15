"""Visualize Phase 3b: all UAVs controlled, clear-the-airspace landing."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from pathlib import Path

from src.config import EnvConfig
from src.env.terminal_area_env import TerminalAreaEnv
from src.env.uav import UAVStatus
from src.train_phase3b import Phase3bWrapper
from src.env.wrappers import PerUAVDiscreteWrapper
from stable_baselines3 import PPO


def record_episode(n_uavs: int = 8, seed: int = 42):
    """Record an episode with PPO policy, tracking all positions."""
    config = EnvConfig()
    config.num_uavs = n_uavs
    config.max_uavs = 15  # must match model's observation space (from training)
    config.sim.max_steps = 800
    config.uav_inflow_rate = 0.0

    env = TerminalAreaEnv(config)
    wrapper = Phase3bWrapper(env, n_uavs=n_uavs)
    disc = PerUAVDiscreteWrapper(wrapper)  # auto-computes continuous params

    model = PPO.load("logs/ppo_phase3b.zip")

    obs, _ = disc.reset(seed=seed)

    frames = []
    step = 0
    while True:
        uavs = env.uav_manager.uavs
        frame = {
            "step": step,
            "positions": np.array([u.position.copy() for u in uavs]),
            "ring_ids": np.array([u.ring_id for u in uavs]),
            "statuses": np.array([u.status.value for u in uavs]),
            "angles": np.array([u.angle for u in uavs]),
        }
        frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = disc.step(action)
        step += 1

        if terminated or truncated:
            uavs = wrapper.unwrapped.uav_manager.uavs
            frame = {
                "step": step,
                "positions": np.array([u.position.copy() for u in uavs]) if uavs else np.empty((0, 3)),
                "ring_ids": np.array([u.ring_id for u in uavs]) if uavs else np.array([]),
                "statuses": np.array([u.status.value for u in uavs]) if uavs else np.array([]),
                "angles": np.array([u.angle for u in uavs]) if uavs else np.array([]),
            }
            frames.append(frame)
            break

    collisions = info.get("conflicts", 0)
    near = info.get("near_misses", 0)
    print(f"Episode: {step} steps, landed={info.get('landed', 0)}, "
          f"collisions={collisions}, near_misses={near}")
    return frames, env.airspace, env.landing_zones


def create_animation(frames, airspace, landing_zones, n_uavs: int, output_path: str):
    """Create top-down animation of all UAVs clearing the airspace."""
    radii = airspace.radii

    # Subsample to keep file size reasonable
    step_skip = max(1, len(frames) // 60)
    frames_sub = frames[::step_skip]
    n_frames = len(frames_sub)
    print(f"Creating animation with {n_frames} frames (subsampled from {len(frames)})...")

    ring_colors = ['#3498db', '#f39c12', '#2ecc71']
    ring_labels = ['Ring 1 (R=4000, H=800)', 'Ring 2 (R=2500, H=500)', 'Ring 3 (R=1000, H=200)']

    # Assign a unique color per UAV
    uav_colors = plt.cm.tab10(np.linspace(0, 1, n_uavs))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)

    def update(frame_idx):
        ax.clear()

        frame = frames_sub[frame_idx]
        positions = frame["positions"]
        ring_ids = frame["ring_ids"]
        statuses = frame["statuses"]
        step = frame["step"]

        total_time = step * 2  # dt=2s
        mins, secs = divmod(total_time, 60)
        ax.set_title(f"Phase 3b: All-UAV Controlled Landing — Step {step} ({int(mins)}m{secs:.0f}s)",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')

        # Draw rings
        for r, c, label in zip(radii, ring_colors, ring_labels):
            circle = Circle((0, 0), r, fill=False, color=c, linewidth=1.5, alpha=0.4, linestyle='--')
            ax.add_patch(circle)
            ax.text(r * 0.82, -150, label, fontsize=8, color=c, ha='right')

        # Draw UAVs
        landed_count = 0
        cruising_count = 0
        descending_count = 0
        for i, (pos, rid, status) in enumerate(zip(positions, ring_ids, statuses)):
            if np.all(pos == 0):
                continue
            x, y = pos[0], pos[1]
            color = uav_colors[i % len(uav_colors)]

            if status == UAVStatus.LANDED:  # LANDED
                marker = '*'
                size = 180
                lw = 1.5
                zord = 9
                landed_count += 1
            elif status == UAVStatus.DESCENDING:  # DESCENDING
                marker = 'v'
                size = 80
                lw = 0.8
                zord = 8
                descending_count += 1
            else:  # CRUISING
                marker = 'o'
                size = 60
                lw = 0.5
                zord = 6
                cruising_count += 1

            ax.scatter(x, y, c=[color], s=size, marker=marker,
                      edgecolors='black', linewidth=lw, alpha=0.9, zorder=zord)

        # Landing zones
        for lz_idx, lz in enumerate(landing_zones):
            ax.scatter(lz[0], lz[1], c='#8e44ad', marker='s', s=60, alpha=0.5, zorder=3)
            ax.annotate(str(lz_idx), (lz[0] + 30, lz[1] + 30), fontsize=7, color='#8e44ad', alpha=0.6)

        # Entry points
        for ea in [0, np.pi/2, np.pi, 3*np.pi/2]:
            ep_x = radii[0] * np.cos(ea)
            ep_y = radii[0] * np.sin(ea)
            ax.scatter(ep_x, ep_y, c='gray', marker='x', s=30, alpha=0.3)

        ax.set_xlim(-radii[0] * 1.15, radii[0] * 1.15)
        ax.set_ylim(-radii[0] * 1.15, radii[0] * 1.15)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                   label='Cruising'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='gray',
                   markersize=8, markeredgecolor='black', markeredgewidth=0.8,
                   label='Descending'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#27ae60',
                   markersize=12, markeredgecolor='darkgreen', markeredgewidth=1.5,
                   label='Landed'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#8e44ad',
                   markersize=8, label='Landing Zone'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.85)

        # Status bar
        status_text = (f"Active: {cruising_count + descending_count} | "
                      f"Cruising: {cruising_count} | Descending: {descending_count} | "
                      f"Landed: {landed_count}")
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        return []

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=120, blit=False, repeat=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=8)
    ani.save(output_path, writer=writer)
    print(f"Animation saved to {output_path} ({n_frames} frames, 8 fps)")
    plt.close(fig)


if __name__ == "__main__":
    n = 8
    print(f"Recording Phase 3b episode ({n} UAVs, all controlled)...")
    frames, airspace, landing_zones = record_episode(n_uavs=n, seed=42)
    create_animation(frames, airspace, landing_zones, n, "results/phase3b_demo.gif")
    print("Done!")
