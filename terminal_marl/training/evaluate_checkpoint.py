"""Bounded common-seed A5 checkpoint evaluation, separate from training."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from ..algo import SharedPPO
from ..core.checkpoints import load_checkpoint
from ..core.config import SimConfig
from ..core.observations import A5_OBSERVATION_SCHEMA
from .evaluate_low_level import compare_fixed_rule_and_low_rl

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",required=True);parser.add_argument("--output",required=True);parser.add_argument("--seeds",type=int,default=3);parser.add_argument("--uavs",type=int,default=10);parser.add_argument("--max-seconds",type=float,default=1500.)
    a=parser.parse_args();config=SimConfig(experiment_mode="fixed_low_rl",arrival_model="finite_cohort",total_requested_uavs=a.uavs,arrival_rate_per_second=.03,max_simulation_seconds=a.max_seconds)
    policy=SharedPPO(A5_OBSERVATION_SCHEMA.dimension);load_checkpoint(a.checkpoint,policy,config)
    result=compare_fixed_rule_and_low_rl(policy,config,list(range(a.seeds)))
    compact={k:v for k,v in result.items() if k!="per_seed"};Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(compact,indent=2),encoding="utf-8");print(json.dumps(compact,indent=2))
if __name__=="__main__":main()
