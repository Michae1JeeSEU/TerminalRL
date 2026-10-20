#!/usr/bin/env python3
"""Prepare/run/resume/status/report the frozen Stage2 pair. No implicit install."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze-provenance","prepare","run","resume","status","report"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--preparation-wall-seconds",type=float,default=0.,help="measured earlier tests/precompute; first prepare only")
    parser.add_argument("--target",type=float,default=3600.)
    parser.add_argument("--provenance",type=Path,help="JSON frozen locally; server never invokes Git")
    args=parser.parse_args()
    from terminal_marl.core.rotating_slots.stage2_campaign import (prepare_campaign,run_campaign,
        report_campaign,load_json,CampaignBudget,common_boundary,freeze_provenance)
    if args.action=="freeze-provenance": result=freeze_provenance(args.output)
    elif args.action=="prepare": result=prepare_campaign(args.output,args.preparation_wall_seconds,provenance_path=args.provenance)
    elif args.action in ("run","resume"):
        if args.preparation_wall_seconds: parser.error("credit preparation during prepare only")
        result=run_campaign(args.output,target=args.target)
    elif args.action=="report": result=report_campaign(args.output)
    else:
        result=dict(campaign=load_json(args.output/"campaign.json"),
            common_completed_time_s=common_boundary(args.output),elapsed_s=CampaignBudget(args.output/"budget.json").elapsed)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.action in ("run","resume") and load_json(args.output/"campaign.json").get("status")=="INVALID_STOP":
        raise SystemExit(1)

if __name__=="__main__": main()
