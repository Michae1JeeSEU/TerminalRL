"""Quick collision diagnostic across multiple seeds."""
import numpy as np
import json, sys
from src.config import EnvConfig
from src.env.terminal_area_env import TerminalAreaEnv
from src.train_phase3b import Phase3bWrapper
from src.env.wrappers import PerUAVDiscreteWrapper
from stable_baselines3 import PPO


def diagnose_seed(n: int, seed: int, model):
    config = EnvConfig()
    config.num_uavs = n
    config.max_uavs = 100
    config.sim.max_steps = 1200
    config.uav_inflow_rate = 0.0
    config.safety_enabled = True
    env = TerminalAreaEnv(config)
    wrapper = Phase3bWrapper(env, n_uavs=n)
    disc = PerUAVDiscreteWrapper(wrapper)
    obs, _ = disc.reset(seed=seed)

    collisions = []
    step = 0
    while True:
        uavs = env.uav_manager.uavs
        current = env.conflict_detector.check_all(uavs)
        if current:
            collisions.append((step, list(current)))
            for id1, id2 in current:
                u1 = next(u for u in uavs if u.id == id1)
                u2 = next(u for u in uavs if u.id == id2)
                dist = float(np.linalg.norm(u1.position - u2.position))
                print(f"  COLLISION Step {step}: UAV{id1} pos={u1.position} ring={u1.ring_id} status={u1.status.name} battery={u1.battery:.1f} | UAV{id2} pos={u2.position} ring={u2.ring_id} status={u2.status.name} battery={u2.battery:.1f} | dist={dist:.1f}m")
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = disc.step(action)
        step += 1
        if terminated or truncated:
            break

    landed = info.get('landed', 0)
    failed = info.get('failed', 0)
    total_col = sum(len(c) for _, c in collisions)
    print(f"  Result: {step} steps, landed={landed}, failed={failed}, total_collisions={total_col}")
    return step, landed, failed, total_col, collisions


if __name__ == "__main__":
    n = 30
    model = PPO.load("logs/best_model_phase3b/best_model.zip")

    # Initial placement analysis
    print("=== Initial placement analysis (seed=0) ===")
    config = EnvConfig()
    config.num_uavs = n
    config.max_uavs = 100
    config.sim.max_steps = 1200
    env = TerminalAreaEnv(config)
    wrapper = Phase3bWrapper(env, n_uavs=n)
    disc = PerUAVDiscreteWrapper(wrapper)
    obs, _ = disc.reset(seed=0)
    uavs = env.uav_manager.uavs
    print(f"UAV count: {len(uavs)}")
    positions = np.array([u.position for u in uavs])
    for i in range(len(uavs)):
        for j in range(i+1, len(uavs)):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if dist < 80:
                ui, uj = uavs[i], uavs[j]
                print(f"  CLOSE PAIR: UAV{ui.id} ring={ui.ring_id} angle={np.rad2deg(ui.angle):.1f}deg | UAV{uj.id} ring={uj.ring_id} angle={np.rad2deg(uj.angle):.1f}deg | dist={dist:.1f}m (SAME RING={ui.ring_id==uj.ring_id})")

    # Run diagnostics for multiple seeds
    print("\n=== Multi-seed collision diagnostic ===")
    all_results = {}
    for seed in [0, 1, 2, 42]:
        print(f"\n--- Seed {seed} ---")
        step, landed, failed, total_col, collisions = diagnose_seed(n, seed, model)
        all_results[str(seed)] = {
            "steps": step, "landed": landed, "failed": failed,
            "collisions": total_col, "events": [(s, c) for s, c in collisions]
        }
        if collisions:
            early = [c for c in collisions if c[0] <= 5]
            if early:
                print(f"  WARNING: {len(early)} collision(s) in first 5 steps!")

    json.dump(all_results, open("results/phase3b_n30_diagnostic_multi_seed.json", "w"), indent=2)
    print("\nResults saved.")
