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
    parser.add_argument("--service-profile",choices=("baseline","service_half_v1"),
                        help="explicit frozen service scenario for freeze-provenance/prepare")
    parser.add_argument("--campaign-wall-budget-seconds",type=float,help="first prepare only; default 36000")
    parser.add_argument("--archive-reserve-seconds",type=float,help="first prepare only; default 1800")
    args=parser.parse_args()
    if args.action!="prepare" and (args.campaign_wall_budget_seconds is not None or args.archive_reserve_seconds is not None):
        parser.error("campaign budget limits may only be supplied during first prepare")
    if args.action not in ("freeze-provenance","prepare") and args.service_profile is not None:
        parser.error("service profile is frozen during prepare; run/resume reads campaign.json")
    from terminal_marl.core.rotating_slots.stage2_campaign import (prepare_campaign,run_campaign,
        report_campaign,load_json,CampaignBudget,common_boundary,freeze_provenance)
    if args.action=="freeze-provenance": result=freeze_provenance(args.output,args.service_profile or "baseline")
    elif args.action=="prepare": result=prepare_campaign(args.output,args.preparation_wall_seconds,
        provenance_path=args.provenance,service_profile=args.service_profile or "baseline",
        campaign_wall_budget_seconds=args.campaign_wall_budget_seconds,archive_reserve_seconds=args.archive_reserve_seconds)
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
