#!/usr/bin/env python3
"""Prepare/run/resume/status/report the frozen Stage2 pair. No implicit install."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze-provenance","prepare","branch","run","resume","status","report"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--parent-output",type=Path,help="read-only v6 parent; branch only")
    parser.add_argument("--from-time",type=float,help="complete common 60s checkpoint; branch only")
    parser.add_argument("--preparation-wall-seconds",type=float,default=0.,help="measured earlier tests/precompute; first prepare only")
    parser.add_argument("--target",type=float,default=3600.)
    parser.add_argument("--provenance",type=Path,help="JSON frozen locally; server never invokes Git")
    parser.add_argument("--service-profile",choices=("baseline","service_half_v1"),
                        help="explicit frozen service scenario for freeze-provenance/prepare")
    parser.add_argument("--scenario-profile",choices=("legacy_busy_v5","stress36_rate180_v1","diagnostic18_rate120_v1","medium24_rate150_v1"))
    parser.add_argument("--rescue-rule-profile",choices=("legacy_v5","rescue_opportunity_v6","rescue_need_v7"))
    parser.add_argument("--cpus",type=int,nargs=2,metavar=("METHOD1_CPU","METHOD2_CPU"),
                        help="two distinct physical-core representatives; run/resume only, frozen on first run")
    parser.add_argument("--campaign-wall-budget-seconds",type=float,help="new prepare/branch allocation; defaults prepare 36000, branch 21600")
    parser.add_argument("--archive-reserve-seconds",type=float,help="new prepare/branch allocation; default 1800")
    args=parser.parse_args()
    if args.action not in ("prepare","branch") and (args.campaign_wall_budget_seconds is not None or args.archive_reserve_seconds is not None):
        parser.error("campaign budget limits may only be supplied during first prepare/branch")
    if args.action=="branch":
        if args.parent_output is None or args.from_time is None or args.provenance is None:
            parser.error("branch requires --parent-output, --from-time and --provenance")
    elif args.parent_output is not None or args.from_time is not None:
        parser.error("--parent-output and --from-time apply to branch only")
    if args.action not in ("freeze-provenance","prepare") and any(value is not None for value in
            (args.service_profile,args.scenario_profile,args.rescue_rule_profile)):
        parser.error("profiles are frozen during prepare; run/resume reads campaign.json")
    if args.cpus is not None and args.action not in ("run","resume"):
        parser.error("CPU selection applies to run/resume only")
    from terminal_marl.core.rotating_slots.stage2_campaign import (prepare_campaign,run_campaign,
        report_campaign,load_json,CampaignBudget,common_boundary,freeze_provenance,branch_campaign)
    profiles=dict(scenario_profile=args.scenario_profile or "legacy_busy_v5",rescue_rule_profile=args.rescue_rule_profile or "legacy_v5")
    if args.action=="freeze-provenance": result=freeze_provenance(args.output,args.service_profile or "baseline",**profiles)
    elif args.action=="prepare": result=prepare_campaign(args.output,args.preparation_wall_seconds,
        provenance_path=args.provenance,service_profile=args.service_profile or "baseline",
        campaign_wall_budget_seconds=args.campaign_wall_budget_seconds,archive_reserve_seconds=args.archive_reserve_seconds,**profiles)
    elif args.action=="branch": result=branch_campaign(args.parent_output,args.output,args.from_time,
        provenance_path=args.provenance,preparation_wall_seconds=args.preparation_wall_seconds,
        campaign_wall_budget_seconds=args.campaign_wall_budget_seconds,archive_reserve_seconds=args.archive_reserve_seconds)
    elif args.action in ("run","resume"):
        if args.preparation_wall_seconds: parser.error("credit preparation during prepare only")
        result=run_campaign(args.output,target=args.target,**({"cpus":args.cpus} if args.cpus is not None else {}))
    elif args.action=="report": result=report_campaign(args.output)
    else:
        result=dict(campaign=load_json(args.output/"campaign.json"),
            common_completed_time_s=common_boundary(args.output),elapsed_s=CampaignBudget(args.output/"budget.json").elapsed)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.action in ("run","resume") and load_json(args.output/"campaign.json").get("status")=="INVALID_STOP":
        raise SystemExit(1)

if __name__=="__main__": main()
