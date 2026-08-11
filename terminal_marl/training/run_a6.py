"""Command-line A6 staged high-level training/evaluation entry point."""
from __future__ import annotations
import argparse
from pathlib import Path
from dataclasses import replace
from .high_level_a6 import evaluate_a6_gate, evaluate_a61_dynamic_rule_safety, load_high_level, train_high_level
from ..core.config import SimConfig
from ..envs.high_level_wrapper import FrozenLowLevelPolicy, RuleLowLevelAdapter

A5_HASH = "2D02DEA1FCCA0886A4B3B4C2FCDAB162833226DE2D5735F5A717770B239DAF4C"

def config(mode: str, args) -> SimConfig:
    return SimConfig(experiment_mode=mode, high_level_enabled=True, arrival_model="finite_cohort", total_requested_uavs=args.uavs, arrival_rate_per_second=args.rate, max_simulation_seconds=args.max_seconds)

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("rule", "frozen", "gate", "safety"));parser.add_argument("--output",required=True);parser.add_argument("--frozen-checkpoint");parser.add_argument("--rule-checkpoint");parser.add_argument("--frozen-high-checkpoint");parser.add_argument("--episodes",type=int,default=24);parser.add_argument("--uavs",type=int,default=10);parser.add_argument("--rate",type=float,default=.03);parser.add_argument("--max-seconds",type=float,default=900.);parser.add_argument("--seeds",type=int,default=3)
    args=parser.parse_args(); output=Path(args.output)
    if args.stage=="rule":
        _,rows=train_high_level(config("learned_high_rule_low",args),low_level=RuleLowLevelAdapter(),output_dir=output,episodes=args.episodes)
        print({"stage":"rule","episodes":len(rows),"checkpoint":str(output/"high_level.pt")})
    elif args.stage=="frozen":
        if not args.frozen_checkpoint: parser.error("--frozen-checkpoint is required for frozen stage")
        low=FrozenLowLevelPolicy(args.frozen_checkpoint,config("learned_high_frozen_low",args),A5_HASH)
        _,rows=train_high_level(config("learned_high_frozen_low",args),low_level=low,output_dir=output,episodes=args.episodes)
        print({"stage":"frozen","episodes":len(rows),"low_level_checkpoint_hash":low.checkpoint_hash,"checkpoint":str(output/"high_level.pt")})
    elif args.stage=="gate":
        if not all((args.frozen_checkpoint,args.rule_checkpoint,args.frozen_high_checkpoint)): parser.error("gate requires --frozen-checkpoint --rule-checkpoint --frozen-high-checkpoint")
        base=config("learned_high_frozen_low",args); rule=load_high_level(args.rule_checkpoint,replace(base,experiment_mode="learned_high_rule_low")); frozen=load_high_level(args.frozen_high_checkpoint,base,expected_low_hash=A5_HASH)
        result=evaluate_a6_gate(rule,frozen,base,frozen_checkpoint=args.frozen_checkpoint,frozen_hash=A5_HASH,seeds=list(range(args.seeds)))
        output.mkdir(parents=True,exist_ok=True);(output/"a6_gate.json").write_text(__import__("json").dumps(result,indent=2,default=str),encoding="utf-8");print(__import__("json").dumps({k:v for k,v in result.items() if k!="per_seed"},indent=2,default=str))
    else:
        result=evaluate_a61_dynamic_rule_safety(config("dynamic_rule",args),list(range(args.seeds)))
        output.mkdir(parents=True,exist_ok=True);(output/"a61_safety_gate.json").write_text(__import__("json").dumps(result,indent=2,default=str),encoding="utf-8");print(__import__("json").dumps({k:v for k,v in result.items() if k!="per_seed"},indent=2,default=str))
if __name__=="__main__": main()
