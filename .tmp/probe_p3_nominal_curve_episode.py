"""Frozen previous actor on repaired physics; diagnostic migration, no qualification."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from terminal_marl.algo.trajectory_set_ppo import TrajectorySetActorCritic
from terminal_marl.configs.p3_v25_course import build_config,PROFILE
from terminal_marl.training.p3_v25_runtime import load_actor
from terminal_marl.training.p3_v25_rollout import collect_episode
from terminal_marl.training.p3_v25_task_student import blockers
from terminal_marl.training.p3_v25_imitation_qualification import sha
from terminal_marl.training.p3_v3_bc_learnability import write

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
torch.set_num_threads(1)
source='5ca0c9dc229b7cf750e3d55db611d68fad690093a7da5e9e7d1ab7ea98ad7883'
assert sha(a.checkpoint)==source
cfg=build_config('dense24_fixed');seed,policy_seed=892854,929854
if a.smoke:
    cfg=replace(cfg,total_requested_uavs=3,warm_start_uavs=3,fixed_arrival_schedule=(0.,)*3)
    seed,policy_seed=892862,929862
a.output.mkdir(parents=True,exist_ok=False)
write(a.output/'protocol.json',dict(profile=PROFILE,protocol='p3_nominal_curve_frozen_source_counterfactual_20261008',
    seed=seed,policy_seed=policy_seed,source_sha256=source,smoke=a.smoke,
    source_training_profile='p3_v25_neutral_course_committed_merge_20261008_v1',
    purpose='diagnostic migration only; old actor unqualified on new profile',STUDENT_QUALIFIED=False,PPO_ALLOWED=False))
actor,_=load_actor(a.checkpoint)
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(92001);model=TrajectorySetActorCritic(actor).requires_grad_(False)
_,result=collect_episode(model,cfg,seed=seed,policy_seed=policy_seed,output=a.output/'episode',
    max_steps=2 if a.smoke else 1800,wall_seconds=180 if a.smoke else 3600)
issues=blockers(result,smoke=a.smoke)
assert sha(a.checkpoint)==source
write(a.output/'result.json',dict(status='BLOCKED' if issues else 'SMOKE_OK' if a.smoke else 'COUNTERFACTUAL_COMPLETE',
    source_sha256=source,rollout=result,blockers=issues,STUDENT_QUALIFIED=False,PPO_ALLOWED=False))
print('blockers',issues)
raise SystemExit(1 if issues else 0)
