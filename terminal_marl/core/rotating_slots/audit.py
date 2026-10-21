"""Independent replay ledger audit using the shared frozen numerical kernels.

No live world is imported or trusted. Archive hashes are an integrity precheck;
physical frames, energy deltas, event costs and conservation are recomputed.
This is not an independent empirical oracle for the research energy surrogate.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import math

from .config import load_config
from .energy import evaluate_power, integrate_energy
from .replay import ReplayBundle
from .safety import validate_plan, validate_separation
from .trajectory import TrajectoryPlan


AIRBORNE=frozenset(("INBOUND","HOLD_OUTER","HOLD_MIDDLE","HOLD_INNER","TRANSFER_O_M",
    "TRANSFER_M_I","FINAL_NORMAL","FINAL_EMERGENCY","VERTICAL_DESCENT"))
PHYSICAL=AIRBORNE|{"LANDED_SERVICE"}
FAILURES=frozenset(("ENERGY_FAILURE","EXECUTED_HARD_SEPARATION_VIOLATION"))
TERMINAL=FAILURES|{"CLEARED","REROUTED","CENSORED_DUE_TO_FAILURE"}
KNOWN=AIRBORNE|TERMINAL|{"LANDED_SERVICE"}
MANEUVERS=frozenset(("INBOUND","TRANSFER","FINAL_NORMAL","FINAL_EMERGENCY"))
FEE={"inner":1200.,"middle":2400.,"outer":3600.}
UPSTREAM_POLICY="upstream_capacity_only_pad_arrival_v1_20261018"
STAGE2_POLICY="rs_s2_loss_continue_throughput_v1_0"
LOSS_STATUS="ENERGY_MISSION_FAILED"
LOSS_REASONS=frozenset(("BATTERY_DEPLETED","PEAK_BUDGET_EXHAUSTED"))


def _rows(path):
    with Path(path).open("rb") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _near(a,b,absolute,relative=0.):
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=max(absolute,relative*abs(float(b)))
    except (TypeError,ValueError,OverflowError):
        return False


def _finite_report(value):
    if isinstance(value,dict): return {key:_finite_report(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)): return [_finite_report(item) for item in value]
    return "NONFINITE" if isinstance(value,float) and not math.isfinite(value) else value


def audit_replay(bundle_path,output=None):
    """Audit a physical ReplayBundle without loading its state history.

    PASS means that the independent recorded ledger is consistent with the
    shared numerical kernels. ``physical_outcome`` separately identifies real
    energy or separation failure. Explicit analytic visual fixtures return
    NOT_PHYSICAL_EPISODE, never physical acceptance. Archives are never changed.
    """
    root=Path(bundle_path).resolve()
    report={"status":"FAIL","issues":[],"issue_count":0,"counts":{},
        "physical_outcome":"UNKNOWN","kernel_scope":"independent state/event ledger; shared frozen trajectory, energy and continuous-separation kernels; not an empirical energy oracle",
        "bundle_path":str(root),"continuous_separation":"NOT_CHECKED"}
    def issue(code,message,**details):
        report["issue_count"]+=1
        if len(report["issues"])<512:
            report["issues"].append({"code":code,"message":message,**details})
    def finish():
        result=_finite_report(report)
        if output is not None:
            destination=Path(output).resolve()
            if destination==root or root in destination.parents:
                raise ValueError("audit output must be outside the immutable replay archive")
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
        return result
    try:
        archive=ReplayBundle.validate(root)
        config=load_config(root/"config.yaml")
        metrics=json.loads((root/"metrics.json").read_bytes())
        if not isinstance(metrics,dict): raise ValueError("metrics must be an object")
        policy=metrics.get("policy_id")
        stage2=policy==STAGE2_POLICY
        upstream_policy=policy in (UPSTREAM_POLICY,STAGE2_POLICY)
        if policy is not None and not upstream_policy:
            issue("UNKNOWN_POLICY","archive declares an unsupported demand/resource policy",policy_id=policy)
        report["policy_id"]=policy or "legacy_immediate_admission_departure_pad"
        if metrics.get("fixture_only"):
            report.update(status="NOT_PHYSICAL_EPISODE",physical_outcome="ANALYTIC_VISUAL_FIXTURE",
                          continuous_separation="NOT_AUDITED",counts={"state_frames":archive.state_count})
            return finish()
        manifest=json.loads((root/"manifest.json").read_bytes())
        requests={str(r["id"]):r for r in manifest.get("requests",())}
        if len(requests)!=len(manifest.get("requests",())):
            issue("MANIFEST_DUPLICATE","manifest request IDs are not unique")
        plans={h:TrajectoryPlan.from_dict(p,config=config) for h,p in json.loads((root/"trajectories.json").read_bytes()).items()}
        events=list(_rows(root/"events.jsonl"))
        loss_archive={}
        if stage2:
            required={"decisions.jsonl","dispatch_timings.jsonl","failure_archive.json",
                      "tentative_predictions.json","stage2_provenance.json"}
            if not required <= archive.index["files"].keys():
                raise ValueError("Stage2 replay is missing content-hashed members")
            if (archive.index.get("experiment_policy_id")!=STAGE2_POLICY or
                    metrics.get("experiment_policy_id")!=STAGE2_POLICY or
                    metrics.get("base_policy_id")!=UPSTREAM_POLICY):
                issue("STAGE2_POLICY","Stage2 archive and effective policy identity disagree")
            if metrics.get("post_failure_model")!="abstract_isolation_no_cascade":
                issue("POST_FAILURE_MODEL","Stage2 must declare its isolation abstraction")
            loss_archive=json.loads((root/"failure_archive.json").read_bytes())
            if not isinstance(loss_archive,dict): raise ValueError("failure archive must be keyed by aircraft ID")
            for aid,record in loss_archive.items():
                if (not isinstance(record,dict) or not isinstance(record.get("last_state"),dict)
                        or record["last_state"].get("id")!=aid):
                    raise ValueError("failure archive lacks an identified last physical state")
                for row in record.get("reservations",()):
                    if (not isinstance(row,dict) or row.get("aircraft_id")!=aid or
                            row.get("end") is not None and not math.isfinite(float(row["end"]))):
                        raise ValueError("failure archive has malformed own resource reservations")
                for key in ("active","soft"):
                    p=record.get(key)
                    if p is not None and (not isinstance(p,dict) or plans.get(p.get("trajectory_hash")) is None
                                          or plans[p["trajectory_hash"]].to_dict()!=p):
                        raise ValueError("failure archive plan differs from immutable trajectory member")
    except (ValueError,KeyError,TypeError,AttributeError,OSError,OverflowError) as exc:
        issue("ARCHIVE_INVALID","archive integrity/schema validation failed",error=str(exc))
        return finish()

    releases={};admissions={};reroutes={};touchdowns={};starts={};emergency_starts={};failures=[];hard_pairs=[]
    losses={};isolations={};invalid_events=[]
    known=KNOWN|{LOSS_STATUS} if stage2 else KNOWN
    terminal=TERMINAL|{LOSS_STATUS} if stage2 else TERMINAL
    last_event=-math.inf
    for event in events:
        try:
            t=float(event["time"]);kind=event["type"];aid=event.get("aircraft_id");details=event.get("details",{})
            if not math.isfinite(t) or t<last_event-1e-9:
                issue("EVENT_TIME","event ledger is not finite/nondecreasing",event=event)
            last_event=max(last_event,t)
            if kind=="request_released":
                if aid in releases: issue("DUPLICATE_RELEASE","request revealed more than once",aircraft_id=aid)
                releases[aid]=t
                if aid not in requests or not _near(t,requests[aid]["release_time"],1e-6):
                    issue("INVALID_RELEASE","release does not match immutable manifest",aircraft_id=aid,time=t)
            elif kind=="admitted":
                if aid in admissions: issue("DUPLICATE_ADMISSION","request admitted twice",aircraft_id=aid)
                admissions[aid]=t
            elif kind=="rerouted":
                if aid in reroutes: issue("DUPLICATE_REROUTE","reroute charged twice",aircraft_id=aid)
                reroutes[aid]=t
                if upstream_policy:
                    evidence=details.get("capacity_evidence",())
                    if details.get("reason")!="OUTER_CAPACITY" or len(evidence)!=37 or {e.get("slot") for e in evidence}!=set(range(37)):
                        issue("CAPACITY_EVIDENCE","new-policy reroute lacks complete outer-capacity evidence",aircraft_id=aid)
                    for entry in evidence:
                        reservations=entry.get("reservations",())
                        if not reservations or any(r.get("end") is not None or r.get("resource")!=["outer",entry.get("slot")] for r in reservations):
                            issue("CAPACITY_EVIDENCE","capacity reroute uses a finite release or wrong resource",aircraft_id=aid)
            elif kind=="PAD_DEADLINE_MISSED" and upstream_policy:
                issue("OBSOLETE_PAD_PROOF","departure-time pad deadline proof is not valid under arrival-time reservations")
            elif kind=="touchdown":
                if aid in touchdowns: issue("DUPLICATE_TOUCHDOWN","aircraft landed twice",aircraft_id=aid)
                touchdowns[aid]=t
            elif kind=="maneuver_started":
                h=details.get("trajectory_hash");plan=plans.get(h)
                if h in starts: issue("DUPLICATE_START","immutable maneuver started twice",trajectory_hash=h)
                starts[h]=t
                if plan is None or plan.aircraft_id!=aid or not _near(t,plan.start_time,1e-6):
                    issue("INVALID_START","maneuver start does not match plan",aircraft_id=aid,trajectory_hash=h)
                elif plan.kind=="FINAL_EMERGENCY":
                    if aid in emergency_starts: issue("DUPLICATE_EMERGENCY_FEE","second emergency start for one visit",aircraft_id=aid)
                    emergency_starts[aid]=(t,plan.source_ring)
            elif kind in FAILURES:
                failures.append((t,aid,kind))
                if stage2:
                    issue("LEGACY_FAILURE_IN_STAGE2","Stage2 uses isolated energy reasons or invalid-stop events",aircraft_id=aid)
            elif stage2 and kind=="aircraft_energy_failure":
                if aid in losses:
                    issue("DUPLICATE_ENERGY_LOSS","energy mission loss charged more than once",aircraft_id=aid)
                reason=details.get("reason")
                if reason not in LOSS_REASONS:
                    issue("LOSS_REASON","unsupported energy mission failure reason",aircraft_id=aid,reason=reason)
                losses[aid]=(t,reason)
            elif stage2 and kind=="failure_isolated":
                if aid in isolations: issue("DUPLICATE_ISOLATION","entity isolated twice",aircraft_id=aid)
                isolations[aid]=(t,details)
            elif stage2 and kind=="invalid_run_stopped":
                invalid_events.append(event)
                issue("INVALID_RUN","recorded run stopped for an execution/software/model error",reason=details.get("reason"),time=t)
            elif kind=="hard_separation_pair":
                hard_pairs.append((t,frozenset((aid,details.get("other_id")))))
        except (KeyError,TypeError,ValueError,AttributeError) as exc:
            issue("EVENT_SCHEMA","malformed event",error=str(exc))
    if len(failures)>1: issue("MULTIPLE_PHYSICAL_FAILURES","episode continued through multiple physical failures")
    stop_failure=failures[0] if failures else None
    if stage2:
        for aid,(t,reason) in losses.items():
            record=loss_archive.get(aid)
            isolated=isolations.get(aid)
            if not isinstance(record,dict) or not _near(record.get("time"),t,1e-7) or record.get("reason")!=reason:
                issue("LOSS_ARCHIVE","permanent failure record disagrees with exact event",aircraft_id=aid)
            if (isolated is None or not _near(isolated[0],t,1e-7)
                    or isolated[1].get("reason")!=reason):
                issue("ISOLATION_EVENT","loss lacks matching same-time isolation",aircraft_id=aid)
            if aid in touchdowns or aid in reroutes:
                issue("LOSS_DOUBLE_OUTCOME","failed task also claimed touchdown/reroute",aircraft_id=aid)
        if set(loss_archive)!=set(losses) or set(isolations)!=set(losses):
            issue("LOSS_IDENTITIES","failure archive, loss events and isolations identify different aircraft")
    if stop_failure:
        report["physical_outcome"]="HARD_SEPARATION_FAILURE" if stop_failure[2]=="EXECUTED_HARD_SEPARATION_VIOLATION" else "ENERGY_FAILURE"

    used=set();initial={};previous=None;last=None;frame_count=0;event_index=0;revealed=set();seen_event_times=set()
    pair_runs={};pair_checks=0;documented_hard=0;energy_intervals=0
    # One additive physical envelope per aircraft, independent of recorded
    # subsequent ledger values. The kernel allocates peak_error_s over the
    # complete plan; summing returned widths does not grant .01s per frame.
    peak_envelopes={};peak_checks=0;peak_max_width=0.;peak_step_mismatches_inside=0;peak_static_checks=0
    try:
        sample_dt=float(archive.index.get("sample_dt_s",.2))
    except (TypeError,ValueError):
        sample_dt=math.nan
    if not math.isfinite(sample_dt) or not 0<sample_dt<=.2:
        issue("SAMPLE_DT","declared execution sample interval is outside (0,.2]",sample_dt=sample_dt)
        sample_dt=.2
    try:
        telemetry_dt=float(archive.index.get("telemetry_dt_s",sample_dt))
    except (TypeError,ValueError):
        telemetry_dt=math.nan
    if not math.isfinite(telemetry_dt) or not sample_dt<=telemetry_dt<=1.:
        issue("TELEMETRY_DT","declared telemetry interval is outside [physics_dt,1]",telemetry_dt=telemetry_dt)
        telemetry_dt=sample_dt
    def plan_for(a,t):
        h=a.get("trajectory_hash")
        p=plans.get(h)
        if p is None:
            issue("MISSING_PLAN","physical aircraft has no authoritative plan",aircraft_id=a.get("id"),time=t,trajectory_hash=h)
            return None
        if p.aircraft_id!=a.get("id"):
            issue("PLAN_OWNER","trajectory belongs to another aircraft",aircraft_id=a.get("id"),trajectory_hash=h)
        if h not in used:
            used.add(h)
            v=validate_plan(p,config=config)
            if not v.accepted:
                issue("PLAN_VALIDATION","used plan is not continuously executable",trajectory_hash=h,status=v.status,reason=v.reason,details=v.details)
            if p.kind in MANEUVERS and h not in starts and p.kind!="INBOUND":
                issue("MISSING_START_EVENT","executed maneuver lacks its start event",trajectory_hash=h)
        return p
    def verify_pose(a,p,t,context="state"):
        k=p.state(t)
        for name,expected,tolerance in (("p",k.p,.001),("v",k.v,1e-6),("a",k.a,1e-6)):
            value=a.get(name,())
            if len(value)!=3 or not all(_near(x,y,tolerance) for x,y in zip(value,expected)) or math.dist(value,expected)>tolerance:
                issue("POSE_MISMATCH",f"{context} {name} disagrees with immutable trajectory",aircraft_id=a.get("id"),time=t,field=name)
    def check_run(run):
        nonlocal pair_checks,documented_hard
        a,b,h1,h2,start,end=run
        pair_checks+=1
        result=validate_separation(plans[h1],plans[h2],start,end,config=config)
        if result.accepted: return
        matching=(stop_failure is not None and stop_failure[2]=="EXECUTED_HARD_SEPARATION_VIOLATION" and
                  start-1e-6<=stop_failure[0]<=end+1e-6 and
                  any(abs(t-stop_failure[0])<=.001 and pair==frozenset((a,b)) for t,pair in hard_pairs))
        if result.status=="UNSAFE" and matching:
            prefix_end=stop_failure[0]-.001
            prefix=validate_separation(plans[h1],plans[h2],start,prefix_end,config=config) if prefix_end>start else None
            if prefix is not None and not prefix.accepted:
                issue("LATE_HARD_FAILURE","violation or UNKNOWN predates recorded failure tolerance",pair=[a,b],details=prefix.details)
            else:
                documented_hard+=1
            return
        issue("SEPARATION_UNKNOWN" if result.status=="UNKNOWN" else "UNREPORTED_HARD_SEPARATION",
              "continuous physical pair run is not certified safe or matched to a hard-failure event",
              pair=[a,b],interval=[start,end],status=result.status,details=result.details)

    try:
        for frame in _rows(root/"states.jsonl"):
            t=float(frame["time"]);frame_count+=1
            peak_step_mismatches={}
            if stage2 and frame.get("failed") and not invalid_events:
                issue("STAGE2_GLOBAL_STOP","isolated energy losses must not stop/censor the whole world",time=t)
            if previous is None and abs(t)>1e-9:
                issue("MISSING_EPISODE_ORIGIN","physical episode recording does not start at t=0",time=t)
            if stop_failure and t>stop_failure[0]+1e-6:
                issue("PHYSICS_AFTER_FAILURE","recorded clock advances after physical failure",time=t)
            while event_index<len(events) and float(events[event_index]["time"])<=t+1e-8:
                event=events[event_index]
                if abs(float(event["time"])-t)<=1e-6: seen_event_times.add(event_index)
                if event.get("type")=="request_released": revealed.add(event.get("aircraft_id"))
                event_index+=1
            rows=frame.get("aircraft",())
            if any(not isinstance(a.get("id"),str) or not isinstance(a.get("admitted"),bool) or not isinstance(a.get("fixture"),bool) for a in rows):
                raise ValueError("physical aircraft require string IDs and explicit admitted/fixture booleans")
            current={str(a["id"]):a for a in rows}
            if len(current)!=len(rows): issue("DUPLICATE_AIRCRAFT","duplicate physical/reference entity in one frame",time=t)
            if {r["id"] for r in frame.get("requests",())}!=revealed:
                issue("REVEALED_REQUESTS","snapshot requests differ from release ledger",time=t)
            if upstream_policy:
                pending={r["id"]:r for r in frame.get("upstream",())}
                if frame.get("policy_id")!=policy or set(pending)!=revealed-set(current):
                    issue("UPSTREAM_LEDGER","external queue does not match revealed unresolved requests",time=t)
                for aid,r in pending.items():
                    immutable=requests.get(aid,{})
                    if any(r.get(k)!=immutable.get(k) for k in ("release_time","gate_id","initial_soc","service_class","vehicle_profile_id")):
                        issue("UPSTREAM_REQUEST_CHANGED","external queue changed immutable request",aircraft_id=aid,time=t)
                    if not _near(r.get("wait_seconds"),t-float(immutable.get("release_time",t)),1e-6):
                        issue("UPSTREAM_WAIT","external waiting duration disagrees with release time",aircraft_id=aid,time=t)
                    expected_status="CENSORED_UPSTREAM" if frame.get("failed") else "WAITING"
                    if r.get("status")!=expected_status:
                        issue("UPSTREAM_STATUS","external queue has wrong censoring state",aircraft_id=aid,time=t)
            for aid,a in current.items():
                status=a.get("status")
                if status not in known: issue("UNKNOWN_STATUS","unrecognized physical lifecycle status",aircraft_id=aid,status=status,time=t)
                if stage2 and status=="CENSORED_DUE_TO_FAILURE":
                    issue("STAGE2_CENSORING","another aircraft was censored after an isolated task loss",aircraft_id=aid,time=t)
                if not 0<=float(a["soc"])<=1:
                    issue("SOC_RANGE","SOC outside physical charge fraction",aircraft_id=aid,time=t)
                if float(a.get("peak_seconds",0))>120+1e-5:
                    issue("PEAK_BUDGET","visit-cumulative peak duration exceeds the physical budget",aircraft_id=aid,time=t)
                if aid not in initial:
                    initial[aid]=a.copy()
                    baseline=float(a.get("peak_seconds",0)) if a.get("fixture") else 0.
                    peak_envelopes[aid]=(baseline,baseline)
                    if not _near(a.get("airborne_time",0),0,1e-6):
                        issue("INITIAL_AIRBORNE_TIME","initial accumulated flight time was not executed in this episode",aircraft_id=aid)
                    if not math.isfinite(float(a.get("energy_j",0))) or float(a.get("energy_j",0))<0 or not 0<=float(a.get("peak_seconds",0))<=120:
                        issue("INITIAL_ENERGY_LEDGER","invalid initial energy/peak baseline",aircraft_id=aid)
                    if a.get("fixture"):
                        if abs(t)>1e-9: issue("LATE_FIXTURE","fixture appeared after episode origin",aircraft_id=aid,time=t)
                    else:
                        request=requests.get(aid)
                        insertion=(admissions if a.get("admitted") else reroutes).get(aid) if upstream_policy else releases.get(aid)
                        if request is None or aid not in releases or insertion is None or not _near(t,insertion,1e-6):
                            issue("UNEXPLAINED_ENTITY","new aircraft is neither initial fixture nor matching demand outcome",aircraft_id=aid,time=t)
                        if upstream_policy and request:
                            if not _near(a.get("release_time"),request["release_time"],1e-6) or not _near(a.get("upstream_wait_seconds"),t-request["release_time"],1e-6):
                                issue("ADMISSION_WAIT","retained release/wait disagrees with event times",aircraft_id=aid,time=t)
                            if a.get("admitted") and not _near(a.get("admission_time"),t,1e-6):
                                issue("ADMISSION_TIME","actual entry time differs from insertion event",aircraft_id=aid,time=t)
                            if not a.get("admitted"):
                                outcomes=[e for e in events if e.get("type")=="rerouted" and e.get("aircraft_id")==aid]
                                for entry in outcomes[0].get("details",{}).get("capacity_evidence",()) if outcomes else ():
                                    for reservation in entry.get("reservations",()):
                                        owner=current.get(reservation.get("aircraft_id"))
                                        if owner is None or owner.get("status") not in AIRBORNE:
                                            issue("CAPACITY_OWNER","capacity evidence owner is not a retained airborne entity",aircraft_id=aid,time=t)
                        if request and not _near(a["soc"],request["initial_soc"],1e-7):
                            issue("INITIAL_SOC","arrival SOC differs from frozen request",aircraft_id=aid,time=t)
                        if not _near(a.get("energy_j",0),0,.05) or not _near(a.get("peak_seconds",0),0,1e-6):
                            issue("INITIAL_ENERGY_LEDGER","ordinary admitted/rerouted demand has unexplained prior energy",aircraft_id=aid)
                        ledger=admissions if a.get("admitted") else reroutes
                        if aid not in ledger or not _near(ledger[aid],t,1e-6):
                            issue("MISSING_OUTCOME_EVENT","request state lacks matching admitted/rerouted event",aircraft_id=aid,time=t)
                if status!="REROUTED":
                    p=plan_for(a,t)
                    pose_time=losses.get(aid,(t,None))[0] if stage2 and status==LOSS_STATUS else t
                    if p is not None: verify_pose(a,p,pose_time)
                if stage2 and status==LOSS_STATUS:
                    loss=losses.get(aid)
                    if loss is None or t<loss[0]-1e-7:
                        issue("LOSS_STATE","isolated terminal state has no preceding exact loss",aircraft_id=aid,time=t)
                    else:
                        lt,reason=loss
                        if a.get("touchdown_time") is not None or a.get("service_end") is not None:
                            issue("LOSS_LANDING","isolated task claims ground completion",aircraft_id=aid,time=t)
                        if reason=="BATTERY_DEPLETED" and not _near(a.get("soc"),0.,1e-7):
                            issue("UNEXPLAINED_BATTERY_LOSS","depleted task retains usable battery",aircraft_id=aid,time=t)
                        if reason=="PEAK_BUDGET_EXHAUSTED" and not _near(a.get("peak_seconds"),120.,1e-5):
                            issue("UNEXPLAINED_PEAK_LOSS","peak failure does not reach the cumulative budget",aircraft_id=aid,time=t)
                        for resource in (*frame.get("slots",()),*frame.get("pads",())):
                            end=resource.get("end")
                            if resource.get("aircraft_id")==aid and (end is None or float(end)>lt+1e-7):
                                issue("LOSS_RESOURCE_RETAINED","failed task still reserves future managed traffic",aircraft_id=aid,time=t)
                        if abs(t-lt)<=1e-7:
                            saved=loss_archive.get(aid,{}).get("last_state",{})
                            for field in ("soc","energy_j","peak_seconds","airborne_time"):
                                if not _near(saved.get(field),a.get(field),1e-6 if field!="energy_j" else .05):
                                    issue("LOSS_LAST_STATE","failure archive does not preserve the failure-time ledger",aircraft_id=aid,field=field)
                            for field in ("p","v","a","jerk"):
                                if saved.get(field)!=a.get(field):
                                    issue("LOSS_LAST_STATE","failure archive does not preserve the last kinematics",aircraft_id=aid,field=field)
                            if saved.get("trajectory_hash")!=a.get("trajectory_hash"):
                                issue("LOSS_LAST_STATE","failure archive changes the last authoritative trajectory",aircraft_id=aid)
                if stage2 and aid in losses and t>=losses[aid][0]-1e-7 and status!=LOSS_STATUS:
                    issue("LOSS_NOT_ISOLATED","failed task remains in another lifecycle state",aircraft_id=aid,time=t)
                if not a.get("admitted") and status!="REROUTED":
                    issue("ADMISSION_FLAG","nonadmitted entity has a physical state",aircraft_id=aid,time=t)
                if status in ("LANDED_SERVICE","CLEARED") and a.get("touchdown_time") is None:
                    issue("GROUND_WITHOUT_TOUCHDOWN","ground lifecycle lacks physical touchdown time",aircraft_id=aid,time=t)
            if previous is not None:
                pt=float(previous["time"]);dt=t-pt
                before={str(a["id"]):a for a in previous.get("aircraft",())}
                if dt>telemetry_dt+1e-6:
                    issue("SAMPLING_GAP","frame interval exceeds declared telemetry maximum",interval=[pt,t])
                for aid,old in before.items():
                    a=current.get(aid)
                    if a is None:
                        issue("DISAPPEARED_AIRCRAFT","recorded entity silently disappears",aircraft_id=aid,time=t);continue
                    if bool(old.get("admitted"))!=bool(a.get("admitted")):
                        issue("ADMISSION_CHANGED","admission identity changed after insertion",aircraft_id=aid,time=t)
                    if bool(old.get("fixture"))!=bool(a.get("fixture")):
                        issue("FIXTURE_CHANGED","fixture identity changed after insertion",aircraft_id=aid,time=t)
                    if upstream_policy and any(old.get(k)!=a.get(k) for k in ("release_time","admission_time","upstream_wait_seconds")):
                        issue("ADMISSION_CHANGED","immutable entry/wait identity changed after insertion",aircraft_id=aid,time=t)
                    if old.get("status") in terminal and old.get("status")!=a.get("status"):
                        issue("TERMINAL_CHANGED","terminal status changed",aircraft_id=aid,time=t)
                    if old.get("emergency",{}).get("declared") and not a.get("emergency",{}).get("declared"):
                        issue("EMERGENCY_WASHED","latched declaration was removed",aircraft_id=aid,time=t)
                    if stage2 and old.get("status")==LOSS_STATUS:
                        for field in ("p","v","a","jerk","soc","energy_j","peak_seconds","trajectory_hash",
                                      "airborne_time","holding_time","transfer_time","final_time","descent_time"):
                            if old.get(field)!=a.get(field):
                                issue("ISOLATED_STATE_CHANGED","failed task changed after isolation",aircraft_id=aid,time=t,field=field)
                    joules=soc_used=peak=0.;peak_width=0.;peak_valid=False
                    if old.get("status") in AIRBORNE:
                        p=plan_for(old,pt)
                        if p is not None:
                            verify_pose(a,p,t,"previous interval endpoint")
                            q=integrate_energy(p,pt,t,max_dt=.2,config=config);energy_intervals+=1
                            if not q.valid:
                                issue("ENERGY_UNKNOWN","shared kernel rejects recorded interval",aircraft_id=aid,interval=[pt,t],reason=q.reason)
                            joules,soc_used,peak=q.joules,q.soc_used,q.peak_seconds
                            peak_width=q.peak_error_s;peak_valid=q.valid
                            if stage2 and aid in losses and abs(t-losses[aid][0])<=1e-7:
                                reason=losses[aid][1]
                                # A positive prefix immediately before the event
                                # must still have capacity; no retroactive delay.
                                probe=max(pt,t-1e-5)
                                prefix=integrate_energy(p,pt,probe,max_dt=.2,config=config)
                                if (reason=="BATTERY_DEPLETED" and float(old["soc"])-prefix.soc_used<=-1e-7
                                        or reason=="PEAK_BUDGET_EXHAUSTED" and
                                        float(old.get("peak_seconds",0))+prefix.peak_seconds>120.+1e-5):
                                    issue("LATE_ENERGY_LOSS","physical exhaustion predates recorded loss time",aircraft_id=aid,time=t)
                            if old.get("trajectory_hash")!=a.get("trajectory_hash") and p.kind in MANEUVERS and t<p.end_time-1e-6:
                                issue("ACTIVE_PLAN_CHANGED","current immutable maneuver changed before completion",aircraft_id=aid,time=t)
                    if not _near(float(a.get("energy_j",0))-float(old.get("energy_j",0)),joules,.05,1e-7):
                        issue("ENERGY_DELTA","recorded joules disagree with integrated physical interval",aircraft_id=aid,interval=[pt,t])
                    if not _near(float(old["soc"])-float(a["soc"]),soc_used,1e-7):
                        issue("SOC_DELTA","SOC change disagrees with integrated physical interval",aircraft_id=aid,interval=[pt,t])
                    peak_delta=float(a.get("peak_seconds",0))-float(old.get("peak_seconds",0))
                    if old.get("status") not in AIRBORNE or dt<=0.:
                        peak_static_checks+=1
                        if not _near(peak_delta,0.,1e-5):
                            issue("PEAK_DELTA","peak ledger changed without a positive airborne interval",aircraft_id=aid,interval=[pt,t])
                    elif peak_valid:
                        lower,upper=peak_envelopes[aid]
                        peak_envelopes[aid]=(lower+peak-peak_width,upper+peak)
                        if not _near(peak_delta,peak,1e-5):
                            peak_step_mismatches[aid]=True
                    expected_air=dt if old.get("status") in AIRBORNE else 0.
                    if not _near(float(a.get("airborne_time",0))-float(old.get("airborne_time",0)),expected_air,1e-6):
                        issue("AIRBORNE_TIME","recorded accumulated airborne time disagrees with physical interval",aircraft_id=aid,interval=[pt,t])
                live={}
                physical=[a for a in before.values() if a.get("status") in PHYSICAL and a.get("trajectory_hash") in plans]
                for a,b in combinations(sorted(physical,key=lambda a:a["id"]),2):
                    pair=(a["id"],b["id"]);hashes=(a["trajectory_hash"],b["trajectory_hash"])
                    run=pair_runs.pop(pair,None)
                    if run is not None and run[2:4]==hashes and abs(run[5]-pt)<=1e-8:
                        live[pair]=(*run[:5],t)
                    else:
                        if run is not None: check_run(run)
                        live[pair]=(pair[0],pair[1],*hashes,pt,t)
                for run in pair_runs.values(): check_run(run)
                pair_runs=live
            for aid,a in current.items():
                lower,upper=peak_envelopes[aid]
                value=float(a.get("peak_seconds",0))
                peak_checks+=1
                peak_max_width=max(peak_max_width,upper-lower)
                within=(all(math.isfinite(x) for x in (value,lower,upper)) and
                        lower-1e-5<=value<=upper+1e-5)
                if not within:
                    issue("PEAK_DELTA","cumulative peak ledger lies outside the shared-kernel numerical envelope",
                          aircraft_id=aid,interval=[float(previous["time"]) if previous is not None else t,t],
                          recorded_peak_s=value,lower_peak_s=lower,upper_peak_s=upper,
                          envelope_width_s=upper-lower)
                elif aid in peak_step_mismatches:
                    peak_step_mismatches_inside+=1
            # Seed singleton runs for initial/new physical pairs so an archive
            # ending on its first frame cannot silently evade separation.
            current_physical=[a for a in current.values() if a.get("status") in PHYSICAL and a.get("trajectory_hash") in plans]
            for a,b in combinations(sorted(current_physical,key=lambda a:a["id"]),2):
                pair=(a["id"],b["id"])
                if pair not in pair_runs:
                    pair_runs[pair]=(pair[0],pair[1],a["trajectory_hash"],b["trajectory_hash"],t,t)
            previous=frame;last=frame
        for run in pair_runs.values(): check_run(run)
    except (KeyError,TypeError,ValueError,AttributeError,OverflowError,OSError) as exc:
        issue("STATE_SCHEMA","cannot interpret physical replay state",error=str(exc))
    report.update(state_frames=frame_count,used_plans=len(used),energy_intervals=energy_intervals,
                  continuous_pair_runs=pair_checks,documented_hard_failure_runs=documented_hard,
                  peak_check_scope="shared-kernel existing additive peak_error_s envelope; precision-switch events not reconstructed",
                  peak_envelope_checks=peak_checks,peak_envelope_max_width_s=peak_max_width,
                  peak_step_mismatch_within_envelope_count=peak_step_mismatches_inside,
                  peak_static_ledger_checks=peak_static_checks)
    if last is None: return finish()
    stop=float(last["time"]);final={a["id"]:a for a in last.get("aircraft",())}
    for i,event in enumerate(events):
        if i not in seen_event_times:
            issue("EVENT_FRAME_MISSING","exact event timestamp has no recorded frame",event_type=event.get("type"),time=event.get("time"))
    for aid,r in requests.items():
        required=float(r["release_time"])<stop-1e-8 or (not stop_failure and float(r["release_time"])<=stop+1e-8)
        if required and aid not in releases: issue("MISSING_RELEASE","manifest request preceding stop was never revealed",aircraft_id=aid)
    for aid,t in releases.items():
        waiting=upstream_policy and aid not in final and aid not in admissions and aid not in reroutes
        if not waiting and (aid not in final or (aid in admissions)==(aid in reroutes)):
            issue("REQUEST_OUTCOME","revealed request does not have exactly one retained outcome",aircraft_id=aid)
    fixtures=[a for a in final.values() if a.get("fixture")]
    ground_ids={aid for aid,a in initial.items() if a.get("fixture") and a.get("status")=="LANDED_SERVICE"} if stage2 else set()
    initial_airborne_ids={aid for aid,a in initial.items() if a.get("fixture") and aid not in ground_ids}
    admitted=[a for a in final.values() if a.get("admitted") and a["id"] not in ground_ids]
    landed=[a for a in admitted if a.get("touchdown_time") is not None]
    failed=[a for a in admitted if a.get("status") in (FAILURES|{LOSS_STATUS} if stage2 else FAILURES)]
    for a in landed:
        aid=a["id"]
        initial_service=initial[aid].get("fixture") and initial[aid].get("status")=="LANDED_SERVICE"
        if not initial_service and (aid not in touchdowns or not _near(touchdowns[aid],a["touchdown_time"],1e-6)):
            issue("LANDING_LEDGER","landed entity lacks matching touchdown event",aircraft_id=aid)
        if a.get("status") not in ("LANDED_SERVICE","CLEARED"):
            issue("LANDING_STATUS","touchdown state has incompatible final status",aircraft_id=aid)
        if stage2 and not a.get("emergency",{}).get("charged") and float(a["soc"])<.25-1e-7:
            issue("NORMAL_LANDING_RESERVE","normal touchdown did not retain the physical reserve",aircraft_id=aid)
        if stage2 and not initial_service:
            plan=plans.get(a.get("trajectory_hash"))
            if (plan is None or plan.kind not in ("FINAL_NORMAL","FINAL_EMERGENCY")
                    or not _near(plan.end_time,a.get("touchdown_time"),1e-7)
                    or plan.pad_id!=a.get("pad_id")):
                issue("FABRICATED_TOUCHDOWN","successful touchdown is not the completed authoritative final",aircraft_id=aid)
    if stop_failure:
        if not _near(stop,stop_failure[0],1e-6): issue("FAILURE_STOP_TIME","episode does not stop at its failure event")
        for a in admitted:
            if a.get("touchdown_time") is None and a.get("status") not in FAILURES|{"CENSORED_DUE_TO_FAILURE"}:
                issue("UNCENSORED_FAILURE","remaining airborne aircraft not marked censored",aircraft_id=a["id"])
        bad=final.get(stop_failure[1])
        if bad is None or bad.get("status")!=stop_failure[2]: issue("FAILURE_LEDGER","failure event does not match retained failed aircraft")
        elif stop_failure[2]=="ENERGY_FAILURE":
            power=evaluate_power(bad["v"],bad["a"],config)
            exhausted=float(bad["soc"])<=1e-7 or float(bad.get("peak_seconds",0))>=120-1e-5
            unavailable=not power.valid or power.current_a>2016 or power.shaft_w>288000
            if not exhausted and not unavailable: issue("UNEXPLAINED_ENERGY_FAILURE","failure has positive usable SOC, peak budget and valid power")
        elif not documented_hard:
            issue("UNEXPLAINED_HARD_FAILURE","hard-failure event has no matching continuously detected physical violation")
    elif failed and not stage2: issue("MISSING_FAILURE_EVENT","physical failure state has no stopping event")
    counts={"planned_manifest":len(requests),"manifest_offered":len(releases),"unreleased":len(set(requests)-set(releases)),
        "fixture_count":len(fixtures),"offered":len(releases)+len(fixtures)-len(ground_ids),"admitted":len(admitted),
        "rerouted":sum(a.get("status")=="REROUTED" for a in final.values()),"physically_landed":len(landed),
        "failed_before_landing":len(failed),
        "airborne_or_censored":sum(a.get("status") in AIRBORNE|{"CENSORED_DUE_TO_FAILURE"} for a in admitted),
        "normal_landed":sum(not a.get("emergency",{}).get("charged") for a in landed),
        "emergency_landed":sum(bool(a.get("emergency",{}).get("charged")) for a in landed),
        "landing_below_reserve":sum(float(a["soc"])<.25 for a in landed),
        "energy_failures":len(losses) if stage2 else sum(a.get("status")=="ENERGY_FAILURE" for a in final.values()),
        "censored":sum(a.get("status")=="CENSORED_DUE_TO_FAILURE" for a in final.values())}
    waiting_ids=set(releases)-set(final) if upstream_policy else set()
    if upstream_policy:
        counts.update(upstream_waiting=len(waiting_ids),censored_upstream=len(waiting_ids) if stop_failure else 0)
    if stage2:
        if {a["id"] for a in failed}!=set(losses):
            issue("LOSS_FINAL_LEDGER","retained failed tasks disagree with failure events")
        first=min((t for t,_ in losses.values()),default=None)
        before=sum(a["touchdown_time"]<first for a in landed) if first is not None else len(landed)
        counts.update(initial_airborne_count=len(initial_airborne_ids),initial_ground_count=len(ground_ids),
            F_energy_total=len(losses),F_battery=sum(reason=="BATTERY_DEPLETED" for _,reason in losses.values()),
            F_peak=sum(reason=="PEAK_BUDGET_EXHAUSTED" for _,reason in losses.values()),
            L_success=len(landed),L_normal=counts["normal_landed"],L_emergency=counts["emergency_landed"],
            N_offered=counts["offered"],N_admitted_total=len(admitted),N_rerouted=counts["rerouted"],
            N_upstream_pending=len(waiting_ids),N_airborne_pending=counts["airborne_or_censored"],
            successful_touchdowns_before_first_failure=before,successful_touchdowns_after_first_failure=len(landed)-before)
    equations=(counts["planned_manifest"]==counts["manifest_offered"]+counts["unreleased"],
               counts["offered"]==counts["admitted"]+counts["rerouted"]+len(waiting_ids),
               counts["admitted"]==counts["physically_landed"]+counts["failed_before_landing"]+counts["airborne_or_censored"])
    report["conservation_equations"]=list(equations)
    if not all(equations): issue("CONSERVATION","independently derived conservation equations fail")
    if metrics.get("conservation")!=all(equations):
        issue("METRIC_CONSERVATION","reported conservation flag disagrees with independently derived equations")
    for key,value in counts.items():
        if key not in metrics or metrics[key]!=value:
            issue("METRIC_COUNT","reported count differs from independent manifest/state ledger",metric=key,expected=value,actual=metrics.get(key))
    expected_costs={"cost_reroute":600.*len(reroutes),
        "cost_emergency":sum(FEE.get(ring,0.) for _,ring in emergency_starts.values()),
        "cost_failure":1000000.*(len(losses) if stage2 else len(failures))}
    if upstream_policy:
        waits=sum((admissions.get(aid,reroutes.get(aid,stop))-release) for aid,release in releases.items())
        expected_costs["cost_upstream_wait"]=waits
        if not _near(metrics.get("upstream_wait_time"),waits,1e-6):
            issue("UPSTREAM_WAIT_METRIC","waiting metric differs from release/outcome event ledger",expected=waits)
    expected_energy=sum(float(a.get("energy_j",0)) for a in final.values())
    expected_airtime=sum(float(a.get("airborne_time",0)) for a in final.values())
    expected_costs.update(energy_j=expected_energy,energy_kwh=expected_energy/3600000,
                         airborne_time=expected_airtime,total_cost=expected_airtime+sum(expected_costs.values()))
    for key,value in expected_costs.items():
        if not _near(metrics.get(key),value,.05,1e-7):
            issue("METRIC_LEDGER","reported metric differs from independently checked state/event ledger",metric=key,expected=value,actual=metrics.get(key))
    if stage2:
        primary=(len(losses),-len(landed),expected_costs["cost_emergency"]+expected_costs["cost_reroute"],
                 expected_energy,expected_airtime+expected_costs.get("cost_upstream_wait",0.))
        for key,value in (("C_legacy",expected_costs["total_cost"]),("E_terminal_J",expected_energy),
                          ("W_unresolved_plus_airborne_seconds",primary[-1])):
            if not _near(metrics.get(key),value,.05,1e-7):
                issue("STAGE2_OBJECTIVE_LEDGER","Stage2 objective component differs from executed ledger",metric=key,expected=value)
        actual=metrics.get("K_episode",())
        if len(actual)!=5 or not all(_near(a,b,.05,1e-7) for a,b in zip(actual,primary)):
            issue("STAGE2_PRIMARY_KEY","reliability/throughput lexicographic key disagrees with executed ledger",expected=primary)
        report["primary_key"]=primary
        if bool(metrics.get("has_energy_losses"))!=bool(losses) or bool(metrics.get("terminated_invalid"))!=bool(invalid_events):
            issue("STAGE2_OUTCOME_FLAGS","loss and invalid-run flags disagree with independent event ledger")
        released_resources=0
        for aid,record in loss_archive.items():
            if not isinstance(record,dict): continue
            lt=losses.get(aid,(math.inf,None))[0]
            own=record.get("reservations",())
            n=sum(r.get("end") is None or float(r["end"])>lt for r in own)
            released_resources+=n
            isolated=isolations.get(aid)
            if isolated and isolated[1].get("resources_released")!=n:
                issue("ISOLATION_RESOURCE_COUNT","released resource count differs from failure-time own reservations",aircraft_id=aid)
        if metrics.get("failure_resources_released")!=released_resources:
            issue("ISOLATION_RESOURCE_COUNT","total resource release count differs from permanent failure archive")
        cohorts={}
        for name,ids in (("initial_airborne",initial_airborne_ids),
                         ("new_requests",set(final)-{a["id"] for a in fixtures})):
            rows=[a for a in admitted if a["id"] in ids]
            cohorts[name]=dict(admitted=len(rows),success=sum(a.get("touchdown_time") is not None for a in rows),
                failed=sum(a["id"] in losses for a in rows),airborne_pending=sum(a.get("status") in AIRBORNE for a in rows),
                upstream_pending=len(waiting_ids) if name=="new_requests" else 0,
                rerouted=len(reroutes) if name=="new_requests" else 0,
                airborne_time_s=sum(float(a.get("airborne_time",0)) for a in rows))
        actual_cohorts=metrics.get("cohorts",{})
        for name,expected in cohorts.items():
            actual=actual_cohorts.get(name,{})
            if any(not _near(actual.get(k),v,1e-6) for k,v in expected.items()):
                issue("COHORT_LEDGER","initial/new-request cohort statistics disagree with physical identities",cohort=name,expected=expected)
        report["cohorts"]=cohorts
    for a in admitted:
        charged=bool(a.get("emergency",{}).get("charged"))
        if charged!=(a["id"] in emergency_starts): issue("EMERGENCY_FEE_LEDGER","charged flag differs from actual emergency start ledger",aircraft_id=a["id"])
    stopped=bool(invalid_events) if stage2 else bool(stop_failure)
    if bool(metrics.get("truncated"))!=stopped or bool(metrics.get("cost_is_lower_bound"))!=bool(stopped or waiting_ids):
        issue("FAILURE_CENSORING_METRIC","truncation/lower-bound flags do not match actual failure ledger")
    if stop_failure and not _near(metrics.get("first_failure_time"),stop_failure[0],1e-6):
        issue("FAILURE_TIME_METRIC","reported failure time differs from the stopping event")
    if stage2 and losses and not _near(metrics.get("first_failure_time"),min(t for t,_ in losses.values()),1e-7):
        issue("FAILURE_TIME_METRIC","reported first task loss time differs from isolated event ledger")
    if not stop_failure and not (stage2 and losses) and metrics.get("first_failure_time") is not None:
        issue("FAILURE_TIME_METRIC","nonfailed episode reports a failure time")
    report["counts"]=counts;report["derived_costs"]=expected_costs
    report["initial_fixture_baselines"]=[{"aircraft_id":aid,"energy_j":a.get("energy_j",0),
        "peak_seconds":a.get("peak_seconds",0),"soc":a.get("soc"),"status":a.get("status")}
        for aid,a in initial.items() if a.get("fixture")]
    if not stop_failure:
        ongoing=counts["unreleased"] or waiting_ids or any(a.get("status") in PHYSICAL for a in admitted)
        report["physical_outcome"]="IN_PROGRESS" if ongoing else "COMPLETE"
    if stage2 and losses:
        report["physical_outcome"]="ENERGY_LOSSES_CONTINUED"
    if stage2 and invalid_events:
        report["physical_outcome"]="INVALID_RUN"
    separation_errors=("SEPARATION_UNKNOWN","UNREPORTED_HARD_SEPARATION","LATE_HARD_FAILURE","UNEXPLAINED_HARD_FAILURE")
    report["continuous_separation"]="FAILED" if any(i["code"] in separation_errors for i in report["issues"]) else "DOCUMENTED_HARD_FAILURE" if documented_hard else "CERTIFIED"
    report["status"]="PASS" if report["issue_count"]==0 else "FAIL"
    return finish()
