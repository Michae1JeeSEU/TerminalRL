"""Visualize N=30 Phase 3b landing with collision diagnostics."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from pathlib import Path
import json

from src.config import EnvConfig
from src.env.terminal_area_env import TerminalAreaEnv
from src.env.uav import UAVStatus
from src.train_phase3b import Phase3bWrapper
from src.env.wrappers import PerUAVDiscreteWrapper
from stable_baselines3 import PPO


def record_episode_with_collisions(n_uavs: int = 30, seed: int = 42):
    """Record an episode tracking collisions per step."""
    config = EnvConfig()
    config.num_uavs = n_uavs
    config.max_uavs = 100
    config.sim.max_steps = 1200
    config.uav_inflow_rate = 0.0
    config.safety_enabled = True

    env = TerminalAreaEnv(config)
    wrapper = Phase3bWrapper(env, n_uavs=n_uavs)
    disc = PerUAVDiscreteWrapper(wrapper)

    model = PPO.load("logs/best_model_phase3b/best_model.zip")

    obs, _ = disc.reset(seed=seed)

    frames = []
    all_collisions = {}  # step -> list of (id1, id2) pairs
    step = 0
    while True:
        uavs = env.uav_manager.uavs
        # Check current collisions
        current_collisions = env.conflict_detector.check_all(uavs)
        # Check near-misses
        current_near = env.conflict_detector.check_near_misses(uavs)

        frame = {
            "step": step,
            "positions": np.array([u.position.copy() for u in uavs]),
            "ring_ids": np.array([u.ring_id for u in uavs]),
            "statuses": np.array([u.status.value for u in uavs]),
            "angles": np.array([u.angle for u in uavs]),
            "uav_ids": np.array([u.id for u in uavs]),
            "batteries": np.array([u.battery for u in uavs]),
            "collisions": current_collisions,
            "near_misses": current_near,
            "n_active": len(uavs),
        }
        frames.append(frame)
        if current_collisions:
            all_collisions[step] = current_collisions

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = disc.step(action)
        step += 1

        if terminated or truncated:
            uavs = wrapper.unwrapped.uav_manager.uavs
            frame = {"step": step, "positions": np.empty((0,3)), "ring_ids": np.array([]),
                     "statuses": np.array([]), "angles": np.array([]), "uav_ids": np.array([]),
                     "batteries": np.array([]), "collisions": [], "near_misses": [],
                     "n_active": 0, "final": True}
            frames.append(frame)
            break

    landed = info.get("landed", 0)
    failed = info.get("failed", 0)
    total_collisions = sum(len(c) for c in all_collisions.values())
    print(f"Episode: {step} steps, landed={landed}, failed={failed}, "
          f"total_collisions={total_collisions}, collision_steps={len(all_collisions)}")
    for s, cols in all_collisions.items():
        print(f"  Step {s}: {cols}")
    return frames, env.airspace, env.landing_zones, all_collisions


def create_diagnostic_animation(frames, airspace, landing_zones, n_uavs: int,
                                all_collisions: dict, output_path: str):
    """Create top-down animation with collision and near-miss markings."""
    radii = airspace.radii

    # Subsample but keep all collision frames
    collision_steps = set(all_collisions.keys())
    step_skip = max(1, len(frames) // 100)
    # Ensure collision-adjacent frames are included
    keep_indices = set()
    for i, f in enumerate(frames):
        if i % step_skip == 0:
            keep_indices.add(i)
        # Always keep collision frames and 2 frames before/after
        if f["step"] in collision_steps:
            for offset in range(-3, 4):
                idx = i + offset
                if 0 <= idx < len(frames):
                    keep_indices.add(idx)

    frames_sub = [frames[i] for i in sorted(keep_indices)]
    n_frames = len(frames_sub)
    print(f"Animation: {n_frames} frames (from {len(frames)}, kept all collision-adjacent)")

    ring_colors = ['#3498db', '#f39c12', '#2ecc71']
    ring_labels = ['Ring 1 (4000m, 800m)', 'Ring 2 (2500m, 500m)', 'Ring 3 (1000m, 200m)']

    uav_colors = plt.cm.tab20(np.linspace(0, 1, n_uavs))

    fig, ax = plt.subplots(figsize=(12, 12), dpi=80)
    plt.subplots_adjust(bottom=0.08)

    def update(frame_idx):
        ax.clear()

        frame = frames_sub[frame_idx]
        positions = frame["positions"]
        ring_ids = frame["ring_ids"]
        statuses = frame["statuses"]
        step = frame["step"]
        collisions = frame.get("collisions", [])
        near_misses = frame.get("near_misses", [])

        # Collect colliding UAV IDs for highlighting
        colliding_ids = set()
        for id1, id2 in collisions:
            colliding_ids.add(id1)
            colliding_ids.add(id2)
        near_ids = set()
        for id1, id2 in near_misses:
            near_ids.add(id1)
            near_ids.add(id2)

        total_time = step * 1.0  # dt=1s
        mins, secs = divmod(total_time, 60)
        title = f"Phase 3b: N=30 All-UAV Landing — Step {step} ({int(mins)}m{secs:.0f}s)"
        if collisions:
            title += f"  COLLISION! {collisions}"
        ax.set_title(title, fontsize=14, fontweight='bold',
                     color='red' if collisions else 'black')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')

        # Draw rings
        for r, c, label in zip(radii, ring_colors, ring_labels):
            circle = Circle((0, 0), r, fill=False, color=c, linewidth=1.5, alpha=0.35, linestyle='--')
            ax.add_patch(circle)
            ax.text(r * 0.85, -180, label, fontsize=7, color=c, ha='right', alpha=0.7)

        # Draw UAVs
        landed_count = 0
        cruising_count = 0
        descending_count = 0
        dead_count = 0
        for i, (pos, rid, status, uid) in enumerate(
            zip(positions, ring_ids, statuses, frame["uav_ids"])):
            if np.all(pos == 0):
                continue
            x, y, z = pos[0], pos[1], pos[2]
            color = uav_colors[uid % len(uav_colors)]

            # Check if this UAV is involved in collision or near-miss
            is_colliding = uid in colliding_ids
            is_near = uid in near_ids

            if is_colliding:
                # Red X for colliding UAVs
                ax.scatter(x, y, c='red', s=250, marker='X', edgecolors='darkred',
                          linewidth=2.5, alpha=1.0, zorder=20)
            elif status == UAVStatus.LANDED:
                marker = '*'; size = 150; lw = 1.2; zord = 9
                ax.scatter(x, y, c=[color], s=size, marker=marker,
                          edgecolors='black', linewidth=lw, alpha=0.9, zorder=zord)
                landed_count += 1
            elif status == UAVStatus.DESCENDING:
                marker = 'v'; size = 65; lw = 0.7; zord = 8
                ax.scatter(x, y, c=[color], s=size, marker=marker,
                          edgecolors='black', linewidth=lw, alpha=0.85, zorder=zord)
                descending_count += 1
            else:
                marker = 'o'; size = 50; lw = 0.5; zord = 6
                ax.scatter(x, y, c=[color], s=size, marker=marker,
                          edgecolors='black', linewidth=lw, alpha=0.8, zorder=zord)
                cruising_count += 1

            # Near-miss highlight
            if is_near and not is_colliding:
                ax.scatter(x, y, c='none', s=200, marker='o', edgecolors='yellow',
                          linewidth=2.0, alpha=0.7, zorder=7)

            # Battery indicator for dead UAVs
            if frame["batteries"][i] <= 0:
                dead_count += 1

        # Landing zones
        for lz_idx, lz in enumerate(landing_zones):
            ax.scatter(lz[0], lz[1], c='#8e44ad', marker='s', s=50, alpha=0.4, zorder=3)
            ax.annotate(str(lz_idx), (lz[0] + 50, lz[1] + 50), fontsize=6, color='#8e44ad', alpha=0.5)

        # Entry points
        for ea in [0, np.pi/2, np.pi, 3*np.pi/2]:
            ep_x = radii[0] * np.cos(ea)
            ep_y = radii[0] * np.sin(ea)
            ax.scatter(ep_x, ep_y, c='gray', marker='x', s=25, alpha=0.25)

        ax.set_xlim(-radii[0] * 1.2, radii[0] * 1.2)
        ax.set_ylim(-radii[0] * 1.2, radii[0] * 1.2)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=7, markeredgecolor='black', markeredgewidth=0.5,
                   label='Cruising'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='gray',
                   markersize=7, markeredgecolor='black', markeredgewidth=0.7,
                   label='Descending'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#27ae60',
                   markersize=10, markeredgecolor='darkgreen', markeredgewidth=1.2,
                   label='Landed'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
                   markersize=10, markeredgecolor='darkred', markeredgewidth=2,
                   label='COLLISION'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markersize=8, markeredgecolor='yellow', markeredgewidth=2,
                   label='Near-miss (<50m)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.85)

        # Status bar
        status_text = (f"Active: {cruising_count + descending_count} | "
                      f"Cruising: {cruising_count} | Descending: {descending_count} | "
                      f"Landed: {landed_count}")
        if dead_count > 0:
            status_text += f" | DEAD: {dead_count}"
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Collision log at bottom
        if collisions:
            ax.text(0.5, 0.02, f"COLLISION: {collisions}", transform=ax.transAxes,
                    fontsize=11, color='red', fontweight='bold',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))

        return []

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False, repeat=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=10)
    ani.save(output_path, writer=writer)
    print(f"Animation saved to {output_path} ({n_frames} frames, 10 fps)")
    plt.close(fig)


if __name__ == "__main__":
    n = 30
    seed = 0
    print(f"Recording N={n} with seed={seed}...")
    frames, airspace, lzs, collisions = record_episode_with_collisions(n_uavs=n, seed=seed)
    create_diagnostic_animation(frames, airspace, lzs, n, collisions,
                                f"results/phase3b_n30_diagnostic_seed{seed}.gif")

    # Also save collision summary as JSON
    collision_summary = {
        "n_uavs": n, "seed": seed, "total_steps": len(frames),
        "collision_events": {str(k): v for k, v in collisions.items()},
        "total_collisions": sum(len(v) for v in collisions.values()),
    }
    json.dump(collision_summary, open(f"results/phase3b_n30_collision_summary_seed{seed}.json", "w"), indent=2)
    print(f"Collision summary saved.")
    print("Done!")
