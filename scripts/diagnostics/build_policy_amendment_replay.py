"""Build one disclosed deterministic amendment fixture, never a formal matrix."""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,POLICY_ID
from terminal_marl.core.rotating_slots.demand import ArrivalRequest,ArrivalManifest
from terminal_marl.core.rotating_slots.driver import DeterministicDriver
from terminal_marl.core.rotating_slots.replay import ReplayBundle
from terminal_marl.core.rotating_slots.audit import audit_replay
from terminal_marl.core.rotating_slots.qualification import provenance


def build(output, *, code_commit, command):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    launch=output.parent/(output.name+"_launch.json")
    if launch.exists(): raise FileExistsError(launch)
    launch_record={"started_at_utc":datetime.now(timezone.utc).isoformat(),"command":command,
                   "policy_id":POLICY_ID,"formal_evidence":False,"output_identity":code_commit,
                   "script_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
                   "source_provenance_at_start":provenance(command)}
    launch.write_text(json.dumps(launch_record,indent=2),encoding="utf-8")
    requests=(ArrivalRequest("entry_first",0.,0,.9,"short"),
              ArrivalRequest("entry_waiting",.1,0,.9,"short"))
    w=RotatingSlotWorld(manifest=ArrivalManifest(requests))
    w.seed_service("pad0_initial_service",0,remaining=10.)
    w.seed_hold("inner_rescue","inner",0,.30,emergency=True)
    w.seed_hold("middle_background","middle",14,.9)
    driver=DeterministicDriver()
    def admission_only(snapshot,port):
        # Ordinary ring departures are intentionally withheld in this fixture.
        # Actual automatic emergency monitoring/handling stays enabled.
        driver(replace(snapshot,aircraft=()),port)
    start=time.perf_counter()
    w.advance(.1,admission_only)
    assert "entry_waiting" not in w.aircraft and w.metrics()["upstream_waiting"]==1
    rescue_plan=w.active["inner_rescue"]
    assert rescue_plan.kind=="FINAL_EMERGENCY" and rescue_plan.start_time==0.
    assert rescue_plan.end_time-13.5>10.
    print(json.dumps({"stage":"initial_wait_and_emergency_departure","time":w.time}),flush=True)
    w.advance(rescue_plan.end_time+.2,admission_only)
    assert w.aircraft["entry_waiting"].admitted and w.aircraft["entry_waiting"].admission_time>.1
    assert w.aircraft["inner_rescue"].touchdown_time is not None and not w.failed
    archive=ReplayBundle.write(output,world=w,code_commit=code_commit,command=command,
                              assertions=[{"conservation":w.metrics()["conservation"]}])
    audit=audit_replay(archive.path,output.parent/(output.name+"_audit.json"))
    summary={"policy_id":POLICY_ID,"formal_evidence":False,"command":command,"launch_evidence":str(launch),
        "fixture":"two same-gate requests + initial pad0 service + inner rescue + middle hold; admission-only callback",
        "replay":str(archive.path),"audit_status":audit["status"],"audit_issues":audit["issue_count"],
        "simulated_seconds":w.time,"wall_seconds":time.perf_counter()-start,
        "second_request":w.aircraft["entry_waiting"].to_dict(),"rescue_plan":rescue_plan.to_dict(),
        "rescue":w.aircraft["inner_rescue"].to_dict(),"metrics":w.metrics()}
    (output.parent/(output.name+"_summary.json")).write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if audit["status"]!="PASS": raise AssertionError(audit["issues"])
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="artifacts/rotating_slots/upstream_policy_20261018/small_multi_aircraft_replay")
    parser.add_argument("--code-commit",default="new-policy-working-tree-not-f67")
    args=parser.parse_args(argv)
    command=" ".join([sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)])
    result=build(args.output,code_commit=args.code_commit,command=command)
    print(json.dumps({k:result[k] for k in ("replay","audit_status","audit_issues","simulated_seconds","wall_seconds")}))
    return 0


if __name__=="__main__": raise SystemExit(main())
