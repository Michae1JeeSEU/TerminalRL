"""Multi-seed multi-scale Phase 3b evaluation with collision/failure diagnosis."""
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from src.config import EnvConfig
from src.env.terminal_area_env import TerminalAreaEnv
from src.env.uav import UAVStatus
from src.train_phase3b import Phase3bWrapper
from src.env.wrappers import PerUAVDiscreteWrapper
from src.baselines.heuristic_policy import HeuristicPolicy


def run_diagnostic_episode(policy, config, n_uavs, seed, policy_name="PPO",
                           record_collision_detail=False):
    """Run one episode and return detailed stats, optionally with per-collision data."""
    env = TerminalAreaEnv(config)
    wrapper = Phase3bWrapper(env, n_uavs=n_uavs)

    if policy_name == "PPO":
        disc = PerUAVDiscreteWrapper(wrapper)
        obs, _ = disc.reset(seed=seed)
    else:
        obs, _ = wrapper.reset(seed=seed)

    step = 0
    collisions = 0
    near_misses = 0
    collision_details = []  # list of (step, uav1_id, uav2_id, dist, pos1, pos2, status1, status2)

    while True:
        if policy_name == "PPO":
            action, _states = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = disc.step(action)
        else:
            action = policy.predict(obs)
            obs, _, terminated, truncated, info = wrapper.step(action)

        # Check for collisions in current step
        step_conflicts = info.get("conflicts", 0)
        step_near = info.get("near_misses", 0)
        collisions += step_conflicts
        near_misses += step_near

        if record_collision_detail and step_conflicts > 0:
            uavs = env.uav_manager.uavs
            for i in range(len(uavs)):
                for j in range(i + 1, len(uavs)):
                    if uavs[i].status == UAVStatus.LANDED or uavs[j].status == UAVStatus.LANDED:
                        continue
                    if uavs[i].is_dead or uavs[j].is_dead:
                        continue
                    dist = float(np.linalg.norm(uavs[i].position - uavs[j].position))
                    if dist < config.uav.safe_separation:
                        collision_details.append({
                            "step": step,
                            "uav1_id": uavs[i].id,
                            "uav2_id": uavs[j].id,
                            "distance": round(dist, 2),
                            "pos1": [round(x, 1) for x in uavs[i].position.tolist()],
                            "pos2": [round(x, 1) for x in uavs[j].position.tolist()],
                            "ring1": uavs[i].ring_id,
                            "ring2": uavs[j].ring_id,
                            "status1": uavs[i].status.name,
                            "status2": uavs[j].status.name,
                            "priority1": round(uavs[i].compute_priority(), 2),
                            "priority2": round(uavs[j].compute_priority(), 2),
                        })

        step += 1
        if terminated or truncated:
            break

    landed = info.get("landed", 0)
    failed = info.get("failed", 0)
    timed_out = (truncated or step >= config.sim.max_steps) and landed + failed < n_uavs

    return {
        "seed": seed,
        "landed": landed,
        "failed": failed,
        "timed_out": timed_out,
        "steps": step,
        "collisions": collisions,
        "near_misses": near_misses,
        "collision_details": collision_details,
    }


def evaluate_multi_seed(policy, config, n_uavs, num_seeds, policy_name="PPO",
                        record_detail_seeds=None):
    """Run multiple seeds and aggregate results."""
    if record_detail_seeds is None:
        record_detail_seeds = set()

    results = []
    for s in range(num_seeds):
        record_detail = s in record_detail_seeds
        r = run_diagnostic_episode(policy, config, n_uavs, seed=s,
                                   policy_name=policy_name,
                                   record_collision_detail=record_detail)
        results.append(r)

    landed_list = [r["landed"] for r in results]
    failed_list = [r["failed"] for r in results]
    collision_list = [r["collisions"] for r in results]
    near_list = [r["near_misses"] for r in results]
    steps_list = [r["steps"] for r in results]

    landing_rates = [l / max(n_uavs, 1) for l in landed_list]

    eps_with_collisions = sum(1 for c in collision_list if c > 0)
    eps_with_failures = sum(1 for f in failed_list if f > 0)

    return {
        "n_uavs": n_uavs,
        "num_seeds": num_seeds,
        "avg_landed": round(float(np.mean(landed_list)), 2),
        "std_landed": round(float(np.std(landed_list)), 2),
        "avg_landing_rate": round(float(np.mean(landing_rates)) * 100, 1),
        "avg_failed": round(float(np.mean(failed_list)), 2),
        "avg_collisions": round(float(np.mean(collision_list)), 2),
        "avg_near_misses": round(float(np.mean(near_list)), 1),
        "avg_steps": round(float(np.mean(steps_list)), 0),
        "eps_with_collisions": eps_with_collisions,
        "eps_with_failures": eps_with_failures,
        "eps_with_timeout": sum(1 for r in results if r["timed_out"]),
        "total_collisions_all_eps": int(sum(collision_list)),
        "per_seed_results": results,
    }


def main():
    num_seeds = 20
    n_list = [15, 20, 25, 30, 35, 40]
    record_detail_seeds = {0}  # record collision detail for seed 0

    print(f"Phase 3b Multi-Seed Multi-Scale Diagnostic")
    print(f"Seeds per N: {num_seeds}, N values: {n_list}")
    print("=" * 70)

    # Load PPO model
    model_path = "logs/best_model_phase3b/best_model.zip"
    if not Path(model_path).exists():
        model_path = "logs/ppo_phase3b.zip"
    model = PPO.load(model_path)
    print(f"Loaded model: {model_path}")

    all_results = {}
    collision_summary = defaultdict(list)

    for n in n_list:
        config = EnvConfig()
        config.num_uavs = n
        config.max_uavs = 100
        config.sim.max_steps = 1200
        config.uav_inflow_rate = 0.0
        config.safety_enabled = True

        # PPO
        print(f"\n--- N={n} PPO ---")
        t0 = time.time()
        ppo_res = evaluate_multi_seed(model, config, n, num_seeds, "PPO",
                                      record_detail_seeds)
        elapsed = time.time() - t0

        print(f"  Landed: {ppo_res['avg_landed']:.1f}/{n} ({ppo_res['avg_landing_rate']:.0f}%) "
              f"+- {ppo_res['std_landed']:.1f}")
        print(f"  Failed: {ppo_res['avg_failed']:.1f}, Collisions: {ppo_res['avg_collisions']:.2f}")
        print(f"  Near-misses: {ppo_res['avg_near_misses']:.1f}, Steps: {ppo_res['avg_steps']:.0f}")
        print(f"  Eps with collisions: {ppo_res['eps_with_collisions']}/{num_seeds}")
        print(f"  Eps with failures: {ppo_res['eps_with_failures']}/{num_seeds}")
        print(f"  Eps with timeout: {ppo_res['eps_with_timeout']}/{num_seeds}")
        print(f"  Total collisions across all episodes: {ppo_res['total_collisions_all_eps']}")
        print(f"  Time: {elapsed:.1f}s")

        # Collect collision details
        for r in ppo_res["per_seed_results"]:
            if r["collision_details"]:
                collision_summary[str(n)].append(r["collision_details"])

        # Heuristic
        print(f"--- N={n} Heuristic ---")
        t0 = time.time()
        heuristic = HeuristicPolicy(config)
        heur_res = evaluate_multi_seed(heuristic, config, n, num_seeds, "Heuristic")
        elapsed = time.time() - t0

        print(f"  Landed: {heur_res['avg_landed']:.1f}/{n} ({heur_res['avg_landing_rate']:.0f}%) "
              f"+- {heur_res['std_landed']:.1f}")
        print(f"  Failed: {heur_res['avg_failed']:.1f}, Collisions: {heur_res['avg_collisions']:.2f}")
        print(f"  Near-misses: {heur_res['avg_near_misses']:.1f}, Steps: {heur_res['avg_steps']:.0f}")
        print(f"  Eps with collisions: {heur_res['eps_with_collisions']}/{num_seeds}")
        print(f"  Eps with failures: {heur_res['eps_with_failures']}/{num_seeds}")
        print(f"  Total collisions across all episodes: {heur_res['total_collisions_all_eps']}")
        print(f"  Time: {elapsed:.1f}s")

        all_results[str(n)] = {"ppo": ppo_res, "heuristic": heur_res}

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'N':<6} {'PPO Landed':<14} {'PPO Rate':<10} {'PPO Coll':<10} "
          f"{'Heur Landed':<14} {'Heur Rate':<10} {'Heur Coll':<10}")
    print("-" * 80)
    for n in n_list:
        ppo = all_results[str(n)]["ppo"]
        heur = all_results[str(n)]["heuristic"]
        print(f"{n:<6} {ppo['avg_landed']:<14.1f} {ppo['avg_landing_rate']:<10.1f}% "
              f"{ppo['avg_collisions']:<10.2f} "
              f"{heur['avg_landed']:<14.1f} {heur['avg_landing_rate']:<10.1f}% "
              f"{heur['avg_collisions']:<10.2f}")

    # Collision diagnosis
    if collision_summary:
        print("\n" + "=" * 80)
        print("COLLISION DIAGNOSIS (from PPO episodes with recorded details)")
        print("=" * 80)
        for n in n_list:
            sn = str(n)
            if sn not in collision_summary:
                continue
            all_collisions = []
            for ep_collisions in collision_summary[sn]:
                all_collisions.extend(ep_collisions)

            if not all_collisions:
                continue

            print(f"\nN={n}: {len(all_collisions)} total collision events recorded")

            # Analyze ring distribution
            ring_pairs = defaultdict(int)
            status_pairs = defaultdict(int)
            for c in all_collisions:
                rp = f"R{c['ring1']}-R{c['ring2']}"
                ring_pairs[rp] += 1
                sp = f"{c['status1']}-{c['status2']}"
                status_pairs[sp] += 1

            print(f"  Ring pairs: {dict(ring_pairs)}")
            print(f"  Status pairs: {dict(status_pairs)}")

            # Show first 3 detailed collision events
            for i, c in enumerate(all_collisions[:3]):
                print(f"  Event {i+1}: step={c['step']}, dist={c['distance']}m")
                print(f"    UAV{c['uav1_id']}: R{c['ring1']} {c['status1']} pos={c['pos1']} pri={c['priority1']}")
                print(f"    UAV{c['uav2_id']}: R{c['ring2']} {c['status2']} pos={c['pos2']} pri={c['priority2']}")

    # Save to JSON
    Path("results").mkdir(exist_ok=True)

    # Strip per_seed_results for JSON (too large)
    save_results = {}
    for n_str, v in all_results.items():
        save_results[n_str] = {
            "ppo": {k: v2 for k, v2 in v["ppo"].items() if k != "per_seed_results"},
            "heuristic": {k: v2 for k, v2 in v["heuristic"].items() if k != "per_seed_results"},
        }
    save_results["collision_diagnosis"] = {
        str(n): [c for ep_c in details for c in ep_c]  # flatten per-episode
        for n, details in collision_summary.items()
    }

    json.dump(save_results, open("results/phase3b_diagnostic.json", "w"), indent=2)
    print(f"\nDiagnostic results saved to results/phase3b_diagnostic.json")


if __name__ == "__main__":
    main()
