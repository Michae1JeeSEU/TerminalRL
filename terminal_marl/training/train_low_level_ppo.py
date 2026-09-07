"""A5 shared low-level PPO curriculum entry point."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import numpy as np
from ..algo import RolloutBuffer, SharedPPO, Transition
from ..core.checkpoints import save_checkpoint
from ..core.config import SimConfig
from ..core.low_level_action import normalized_to_indices
from ..core.observations import A5_OBSERVATION_SCHEMA, LOW_OBS_V2, observation_schema_for_version
from ..envs.parallel_env import TerminalParallelEnv

@dataclass(frozen=True)
class CurriculumStage:
    name: str; episodes: int; requested_uavs: int; arrival_rate: float; reset_mode: str = "flow_from_empty"; demand_model: str = "finite_cohort"

DEFAULT_CURRICULUM = (
    CurriculumStage("sanity_1", 12, 1, 1.0), CurriculumStage("sanity_3", 12, 3, .20),
    CurriculumStage("warm_start_10", 16, 10, .10, "warm_start_rings"),
    CurriculumStage("flow_20", 24, 20, .03), CurriculumStage("variable_bursty", 24, 20, .06, "flow_from_empty", "bursty_wave"),
)

def _stage_config(base: SimConfig, stage: CurriculumStage) -> SimConfig:
    schedule = tuple(0.0 for _ in range(stage.requested_uavs)) if stage.reset_mode == "warm_start_rings" else ()
    return replace(base, experiment_mode="fixed_low_rl", high_level_enabled=False, reset_mode=stage.reset_mode, warm_start_uavs=stage.requested_uavs if stage.reset_mode == "warm_start_rings" else 0, total_requested_uavs=stage.requested_uavs, arrival_model=stage.demand_model, fixed_arrival_schedule=schedule, arrival_rate_per_second=stage.arrival_rate, max_simulation_seconds=max(base.max_simulation_seconds, 2500.0))

def _episode(policy: SharedPPO, config: SimConfig, seed: int, buffer: RolloutBuffer | None, deterministic: bool = False) -> dict:
    env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
    episode_reward = 0.0; action_values=[]; reward_components=[]
    for _ in range(int(config.max_simulation_seconds / config.dt)):
        ids = list(observations)
        if not ids:
            next_observations, _, _, _, info = env.step({})
            observations = next_observations
            if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs: break
            continue
        actor_obs = np.stack([observations[key] for key in ids])
        if hasattr(policy, "act_details"):
            batch = policy.act_details(actor_obs, actor_obs, deterministic=deterministic)
            actions, log_probs, value = batch.normalized, batch.joint_log_prob, batch.value
        else:  # archived rule adapters expose normalized V3 values only
            actions, log_probs, value = policy.act(actor_obs, actor_obs, deterministic=deterministic)
            categories = normalized_to_indices(actions)
            from types import SimpleNamespace
            batch = SimpleNamespace(categories=categories, normalized=actions, joint_log_prob=log_probs,
                                    head_log_prob=np.zeros_like(actions, dtype=np.float32), value=value)
        action_map = {key: action for key, action in zip(ids, actions)}
        next_observations, rewards, terminated, truncated, info = env.step(action_map)
        for index, key in enumerate(ids):
            reward = float(rewards.get(key, 0.0)); done = bool(terminated.get(key, False) or truncated.get(key, False))
            trace = info.get("action_traces", {}).get(key, {})
            teacher_action = trace.get("teacher_corrected_action")
            applied_action = trace.get("shield_applied_action") or trace.get("execution_applied_action")
            if buffer is not None: buffer.add(Transition(key, actor_obs[index], actor_obs, batch.categories[index], float(log_probs[index]), value, reward, done,
                                                          None if teacher_action is None else normalized_to_indices(np.asarray(teacher_action, np.float32)),
                                                          None if applied_action is None else np.asarray(applied_action, np.float32),
                                                          bool(trace.get("teacher_used", False)), bool(trace.get("shield_used", False)),
                                                          actions[index], batch.head_log_prob[index]))
            episode_reward += reward
        action_values.extend(actions.tolist()); reward_components.extend(info["reward_components"].values())
        observations = next_observations
        if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs: break
    components = {name: float(sum(x.get(name, 0.0) for x in reward_components)) for name in ("time", "progress", "release", "touchdown", "failure", "total")}
    completion_delays = [u.completion_time-u.request_time for u in env.completed_uavs if u.completion_time is not None]
    actions_array = np.asarray(action_values, dtype=float).reshape(-1, 3) if action_values else np.zeros((0, 3))
    action_summary = {f"action_{name}_{stat}": float(getattr(actions_array[:, index], stat)()) if len(actions_array) else 0.0 for index, name in enumerate(("release", "speed_residual", "merge_time_residual")) for stat in ("mean", "std")}
    return {"reward": episode_reward, "requested": len(env.ledger.requested), "completed": sum(x.value=="completed" for x in env.ledger.outcomes.values()), "throughput": sum(x.value=="completed" for x in env.ledger.outcomes.values())/max(env.now, 1.), "delay": float(np.mean(completion_delays)) if completion_delays else float("inf"), "energy": float(np.mean([u.energy_consumed for u in env.completed_uavs])) if env.completed_uavs else 0., "fairness": float(np.std(completion_delays)) if completion_delays else 0., "actions": action_values, "reward_components": components, **action_summary, **env.telemetry}

def train_shared_low_level(base: SimConfig | None = None, *, curriculum=DEFAULT_CURRICULUM, output_dir: str | Path = "runs/a5", seed: int = 0, update_every: int = 4) -> tuple[SharedPPO, list[dict]]:
    # Historical API intentionally remains a V1 baseline entry point.  A5-v2
    # training will use a separate teacher-annealed driver after this
    # architecture closure is verified.
    base = replace(base or SimConfig(), observation_schema_version="phaseA_low_obs_v1_41d")
    policy = SharedPPO(A5_OBSERVATION_SCHEMA.dimension); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    history=[]; buffer=RolloutBuffer(); episode_index=0
    for stage in curriculum:
        config=_stage_config(base, stage)
        for _ in range(stage.episodes):
            row=_episode(policy, config, seed+episode_index, buffer); row.update(stage=stage.name, episode=episode_index); history.append(row); episode_index+=1
            if episode_index % update_every == 0: row.update({f"ppo_{k}":v for k,v in policy.update(buffer).items()})
    if buffer.by_agent: history[-1].update({f"ppo_{k}":v for k,v in policy.update(buffer).items()})
    with (output / "training_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in history: handle.write(json.dumps({k:v for k,v in row.items() if k!="actions"}, default=lambda x: x.tolist())+"\n")
    save_checkpoint(output / "fixed_low_rl.pt", policy, base, curriculum=[asdict(x) for x in curriculum])
    return policy, history

if __name__ == "__main__": train_shared_low_level()
