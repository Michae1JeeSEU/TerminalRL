"""Offline-only qualification of the V4 joint Teacher/action semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..algo.joint_ppo import JointPPO
from ..core.low_level_joint_action import JointAction
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .train_low_v2_ppo import l1_full_teacher_config
from .v4_joint_teacher import teacher_joint_label


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _teacher_joint(env: TerminalParallelEnv, uav) -> JointAction:
    """Compatibility alias for the shared training-only V4 Teacher query."""
    return teacher_joint_label(env, uav)


def run(output: str | Path, *, states: int = 32) -> dict:
    """Collect deterministic legal states and run an offline CE-only probe."""
    rows=[]; deterministic=True
    for seed in range(states):
        config=l1_full_teacher_config(medium_rate=.001575, reset_mode="warm_start_rings", demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8, horizon_seconds=20.0)
        config=replace(config, action_schema_version="low_action_schema_v4_joint_discrete", teacher_release_label_enabled=False)
        env=TerminalParallelEnv(config); observations,_=env.reset(seed)
        state_hash=_hash(SafetyTraceRecorder._state(env)); labels={}
        for uid, observation in observations.items():
            label=_teacher_joint(env, env.active[int(uid)]); labels[uid]=label.index
            rows.append({"observation":observation.tolist(),"label":label.index,"state_hash":state_hash,"uid":uid})
        replay=TerminalParallelEnv(config); replay_obs,_=replay.reset(seed)
        for uid in replay_obs:
            deterministic = deterministic and labels[uid] == _teacher_joint(replay, replay.active[int(uid)]).index
    observations=np.asarray([row["observation"] for row in rows],np.float32); labels=np.asarray([row["label"] for row in rows],np.int64)
    torch.manual_seed(2401); policy=JointPPO(observations.shape[1]); optimizer=torch.optim.Adam(policy.model.parameters(),lr=3e-4)
    x=torch.as_tensor(observations); y=torch.as_tensor(labels)
    with torch.no_grad(): initial=float(F.cross_entropy(policy.model.actor_distribution(x).logits,y))
    for _ in range(128):
        loss=F.cross_entropy(policy.model.actor_distribution(x).logits,y); optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad(): final=float(F.cross_entropy(policy.model.actor_distribution(x).logits,y)); accuracy=float((policy.model.actor_distribution(x).logits.argmax(-1)==y).float().mean())
    report={"stage":"V4_offline_joint_qualification","evidence_type":"newly_run","ppo_called":False,"states":states,"rows":len(rows),
            "same_full_state_same_teacher_joint_label":deterministic,"unique_labels":sorted(set(int(x) for x in labels)),
            "joint_ce_initial":initial,"joint_ce_final":final,"joint_probe_accuracy":accuracy,
            "teacher_joint_label_determinism":"PASS" if deterministic else "FAIL",
            "offline_joint_supervised_probe":"PASS" if final < initial else "FAIL"}
    destination=Path(output); destination.mkdir(parents=True,exist_ok=True); (destination/"v4_joint_offline_qualification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def _main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",required=True); parser.add_argument("--states",type=int,default=32)
    args=parser.parse_args(); print(json.dumps(run(args.output,states=args.states),indent=2))


if __name__ == "__main__": _main()
