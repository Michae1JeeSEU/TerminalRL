"""A6 high-level PPO training, strict frozen-low loading, and CRN gate evaluation."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..algo.high_level_ppo import HighLevelPPO, HighRolloutBuffer, HighTransition
from ..core.checkpoints import checkpoint_metadata, validate_checkpoint_metadata
from ..core.config import SimConfig
from ..core.uav import TerminalOutcome
from ..envs.high_level_wrapper import DynamicHighLevelWrapper, FrozenLowLevelPolicy, RuleLowLevelAdapter
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import DynamicRulePolicy


class _LearnedHighAdapter:
    def __init__(self, policy: HighLevelPPO) -> None: self.policy = policy
    def act(self, observation: np.ndarray) -> np.ndarray: return self.policy.act(observation, deterministic=True)[0]


class _DynamicRuleAdapter:
    def __init__(self) -> None: self.policy = DynamicRulePolicy()
    def act(self, observation: np.ndarray, wrapper: DynamicHighLevelWrapper) -> np.ndarray:
        return self.policy.act(observation, wrapper._context())


def _metrics(base: TerminalParallelEnv) -> dict[str, float]:
    completed=[u for u in base.completed_uavs if u.completion_time is not None]
    delays=[u.completion_time-u.request_time for u in completed]
    return {
        "requested": float(len(base.ledger.requested)), "completed": float(len(completed)),
        "throughput": float(len(completed)/max(1.,base.now)), "delay": float(np.mean(delays)) if delays else float("inf"),
        "energy": float(np.mean([u.energy_consumed for u in completed])) if completed else 0.,
        "fairness": float(np.std(delays)) if delays else 0., **{key:float(value) for key,value in base.telemetry.items()},
    }


def rollout_high(
    policy: HighLevelPPO, config: SimConfig, seed: int, low_level, buffer: HighRolloutBuffer | None = None, *, deterministic: bool = False,
) -> dict:
    wrapper=DynamicHighLevelWrapper(config,low_level); observation,_=wrapper.reset(seed); total=0.; components=[]; actions=[]
    while True:
        action,log_prob,value=policy.act(observation,deterministic=deterministic)
        next_observation,reward,terminated,truncated,info=wrapper.step(action)
        if buffer is not None: buffer.add(HighTransition(observation,action,log_prob,value,reward,terminated or truncated))
        total+=reward;components.append(info["reward_components"]);actions.append(info["high_action"]);observation=next_observation
        if terminated or truncated: break
    result=_metrics(wrapper.base); result.update({"reward":float(total),"reward_components":{key:float(sum(item[key] for item in components)) for key in HighLevelRewardKeys},"actions":actions,"geometry_action_count":float(sum(any(item["structural_mask"]) for item in actions)),"low_level_checkpoint_hash":getattr(low_level,"checkpoint_hash",None)})
    return result

HighLevelRewardKeys=("service_completion","external_queue","delay","safety","energy","reconfiguration","total")


def train_high_level(
    config: SimConfig, *, low_level, output_dir: str | Path, episodes: int = 24, seed: int = 0, update_every: int = 4,
) -> tuple[HighLevelPPO,list[dict]]:
    if not config.high_level_enabled: raise ValueError("A6 high training requires a dynamic high-level config")
    policy=HighLevelPPO(DynamicHighLevelWrapper.observation_dim); buffer=HighRolloutBuffer(); rows=[]; output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    for episode in range(episodes):
        row=rollout_high(policy,config,seed+episode,low_level,buffer);row["episode"]=episode
        if (episode+1)%update_every==0: row.update({f"ppo_{key}":value for key,value in policy.update(buffer).items()})
        rows.append(row)
    if buffer.rows: rows[-1].update({f"ppo_{key}":value for key,value in policy.update(buffer).items()})
    metadata=checkpoint_metadata(config,policy_kind="high_level",low_level_checkpoint_hash=getattr(low_level,"checkpoint_hash",None))
    torch.save({"metadata":metadata,"model":policy.model.state_dict(),"optimizer":policy.optimizer.state_dict(),"episodes":episodes},output/"high_level.pt")
    with (output/"training_metrics.jsonl").open("w",encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row,default=str)+"\n")
    return policy,rows


def load_high_level(path: str | Path, config: SimConfig, *, expected_low_hash: str | None = None) -> HighLevelPPO:
    payload=torch.load(path,map_location="cpu",weights_only=False);metadata=payload.get("metadata",{});validate_checkpoint_metadata(metadata,config)
    if metadata.get("policy_kind")!="high_level": raise ValueError("checkpoint is not a high-level PPO checkpoint")
    if expected_low_hash is not None and metadata.get("low_level_checkpoint_hash")!=expected_low_hash.upper(): raise ValueError("high-level checkpoint was not trained with the requested frozen low-level hash")
    policy=HighLevelPPO(DynamicHighLevelWrapper.observation_dim);policy.model.load_state_dict(payload["model"]);policy.model.eval();return policy


def _fixed_rule(config: SimConfig, seed: int) -> dict:
    env=TerminalParallelEnv(replace(config,experiment_mode="fixed_rule",high_level_enabled=False));env.reset(seed);env.run_rule_only();result=_metrics(env);result["geometry_action_count"]=0.;return result


def evaluate_a61_dynamic_rule_safety(base: SimConfig, seeds: list[int]) -> dict:
    """Pre-training A6.1 safety gate using only the protected rule baseline."""
    rows=[]
    for seed in seeds:
        config=replace(base,experiment_mode="dynamic_rule",high_level_enabled=True)
        wrapper=DynamicHighLevelWrapper(config,RuleLowLevelAdapter()); observation,_=wrapper.reset(seed); policy=DynamicRulePolicy()
        while True:
            observation,_,terminated,truncated,info=wrapper.step(policy.act(observation,wrapper._context()))
            if terminated or truncated: break
        row=_metrics(wrapper.base); row.update({"geometry_action_count":float(wrapper.base.telemetry["geometry_updates"]),"hard_separation_violations":wrapper.coordinator.telemetry["hard_separation_violations"],"shield":wrapper.coordinator.history}); rows.append(row)
    summary={key:float(np.mean([row[key] for row in rows])) for key in rows[0] if isinstance(rows[0][key],(int,float))}
    checks={"zero_collision":summary["collisions"]==0.,"zero_unique_nmac":summary["nmac"]==0.,"zero_hard_separation":summary["hard_separation_violations"]==0.,"nonzero_geometry":summary["geometry_action_count"]>0.}
    return {"passed":all(checks.values()),"checks":checks,"summary":summary,"per_seed":rows}


def evaluate_a6_gate(high_rule: HighLevelPPO, high_frozen: HighLevelPPO, base: SimConfig, *, frozen_checkpoint: str | Path, frozen_hash: str, seeds: list[int]) -> dict:
    configurations={"fixed_rule":replace(base,experiment_mode="fixed_rule",high_level_enabled=False),"dynamic_rule":replace(base,experiment_mode="dynamic_rule",high_level_enabled=True),"learned_high_rule_low":replace(base,experiment_mode="learned_high_rule_low",high_level_enabled=True),"learned_high_frozen_low":replace(base,experiment_mode="learned_high_frozen_low",high_level_enabled=True)}
    output={}; per_seed={}
    for name,config in configurations.items():
        rows=[]
        for seed in seeds:
            if name=="fixed_rule": row=_fixed_rule(config,seed)
            else:
                low=RuleLowLevelAdapter() if name!="learned_high_frozen_low" else FrozenLowLevelPolicy(frozen_checkpoint,config,frozen_hash)
                wrapper=DynamicHighLevelWrapper(config,low); observation,_=wrapper.reset(seed)
                while True:
                    if name=="dynamic_rule": action=DynamicRulePolicy().act(observation,wrapper._context())
                    else: action=(high_rule if name=="learned_high_rule_low" else high_frozen).act(observation,deterministic=True)[0]
                    observation,_,terminated,truncated,_=wrapper.step(action)
                    if terminated or truncated: break
                row=_metrics(wrapper.base);row["geometry_action_count"]=float(wrapper.base.telemetry["geometry_updates"]);row["action_telemetry"]=wrapper.action_history;row["low_level_checkpoint_hash"]=getattr(low,"checkpoint_hash",None)
            rows.append(row)
        per_seed[name]=rows
        numeric={key:[row[key] for row in rows] for key in rows[0] if isinstance(rows[0][key],(int,float))}
        output[name]={key:float(np.mean(value)) for key,value in numeric.items()}
    fixed=output["fixed_rule"]
    learned=output["learned_high_frozen_low"]
    checks={"geometry_active":all(output[name]["geometry_action_count"]>0 for name in ("dynamic_rule","learned_high_rule_low","learned_high_frozen_low")),"safety_not_worse":learned["collisions"]<=fixed["collisions"] and learned["nmac"]<=fixed["nmac"],"improves_delay_or_throughput":learned["delay"]<fixed["delay"] or learned["throughput"]>fixed["throughput"],"frozen_hash_recorded":all(row.get("low_level_checkpoint_hash")==frozen_hash.upper() for row in per_seed["learned_high_frozen_low"])}
    return {"passed":all(checks.values()),"checks":checks,"summary":output,"per_seed":per_seed}
