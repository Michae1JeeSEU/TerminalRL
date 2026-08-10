"""Command-line A5 train → common-seed evaluation → gate report entry point."""
from __future__ import annotations
import argparse, json
from dataclasses import replace
from pathlib import Path
from .evaluate_low_level import assess_a5_gate, compare_fixed_rule_and_low_rl
from .train_low_level_ppo import DEFAULT_CURRICULUM, train_shared_low_level
from ..core.config import SimConfig

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",required=True); parser.add_argument("--seeds",type=int,default=10); parser.add_argument("--episodes-scale",type=float,default=1.0)
    args=parser.parse_args(); curriculum=tuple(replace(s,episodes=max(1,round(s.episodes*args.episodes_scale))) for s in DEFAULT_CURRICULUM)
    base=SimConfig(experiment_mode="fixed_low_rl",max_simulation_seconds=2500.)
    policy,history=train_shared_low_level(base,curriculum=curriculum,output_dir=args.output)
    evaluation=SimConfig(experiment_mode="fixed_low_rl",arrival_model="finite_cohort",total_requested_uavs=20,arrival_rate_per_second=.03,max_simulation_seconds=2500.)
    comparison=compare_fixed_rule_and_low_rl(policy,evaluation,list(range(args.seeds)))
    gate=assess_a5_gate(comparison,history)
    Path(args.output).mkdir(parents=True,exist_ok=True)
    with (Path(args.output)/"a5_gate.json").open("w",encoding="utf-8") as handle:json.dump(gate,handle,indent=2,allow_nan=False)
    print(json.dumps(gate,indent=2,allow_nan=False));return 0 if gate["passed"] else 2
if __name__=="__main__":raise SystemExit(main())
