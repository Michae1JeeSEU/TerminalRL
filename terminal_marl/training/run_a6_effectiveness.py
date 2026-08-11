from __future__ import annotations
import argparse
from .a6_effectiveness import run_matrix, write_report

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--rule-checkpoint",required=True);parser.add_argument("--frozen-high-checkpoint",required=True);parser.add_argument("--frozen-low-checkpoint",required=True);parser.add_argument("--output",required=True);parser.add_argument("--seeds",type=int,default=3);parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args(); report=write_report(run_matrix(rule_checkpoint=args.rule_checkpoint,frozen_high_checkpoint=args.frozen_high_checkpoint,frozen_low_checkpoint=args.frozen_low_checkpoint,seeds=list(range(args.seeds)),workers=args.workers),args.output);print({"safety_passed":report["safety_passed"],"output":args.output,"workers":args.workers})
if __name__=="__main__":main()
