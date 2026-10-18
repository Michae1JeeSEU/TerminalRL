"""Bounded additional catalogue fixtures using production operations only."""
from dataclasses import asdict, replace
from itertools import product
import time

from .demand import ArrivalManifest, ArrivalRequest
from .energy import integrate_energy, evaluate_power
from .geometry import slot_state
from .trajectory import generate_final, generate_transfer, generate_inbound, make_stable_plan
from .safety import validate_separation
from .world import AIRBORNE


def cumulative_peak(e):
    """Named earlier-visit peak fixture plus an actually integrated final."""
    from .fixtures import world, seed, feasible, submit, advance
    plan=feasible(e,generate_final,"peak","inner",0,0,0)
    energy=integrate_energy(plan,plan.start_time,plan.end_time)
    e.observe("candidate_peak_energy",asdict(energy))
    e.check(energy.valid and 30 < energy.peak_seconds < 120,
        "candidate contributes a separately sub120s peak interval", "30<peak<120", asdict(energy))
    rejected_world=world(e)
    rejected_world.auto_emergency=False
    seed(e,rejected_world,"peak","inner",0,.9)
    e.action("named_fixture_prior_visit_peak",{"aircraft_id":"peak","peak_seconds":90.,
        "source":"synthetic preexisting terminal-visit accumulation; not claimed executed in this fixture"})
    rejected_world.aircraft["peak"]=replace(rejected_world.aircraft["peak"],peak_seconds=90.)
    result=submit(e,rejected_world,(plan,))
    e.check(result.reason=="PEAK_BUDGET" and not result.accepted and not rejected_world.active,
        "prior90 plus actual candidate exceeds120 and is rejected", "PEAK_BUDGET",asdict(result))
    e.check(rejected_world.aircraft["peak"].peak_seconds==90.,
        "rejected candidate does not reset accumulated peak",90.,rejected_world.aircraft["peak"].peak_seconds)
    executed_world=world(e)
    executed_world.auto_emergency=False
    seed(e,executed_world,"peak","inner",0,.9)
    e.action("named_fixture_prior_visit_peak",{"aircraft_id":"peak","peak_seconds":60.,
        "source":"synthetic prior accumulation; final below is physically executed"})
    executed_world.aircraft["peak"]=replace(executed_world.aircraft["peak"],peak_seconds=60.)
    result=submit(e,executed_world,(plan,))
    e.check(result.accepted,"underbudget accumulated candidate starts",True,asdict(result))
    advance(e,executed_world,plan.end_time)
    actual=executed_world.aircraft["peak"]
    e.check(actual.status=="LANDED_SERVICE" and abs(actual.peak_seconds-(60+energy.peak_seconds))<1e-5,
        "executed final adds peak to prior visit history without reset",60+energy.peak_seconds,
        {"status":actual.status,"peak_seconds":actual.peak_seconds})
    return True


def mixed_conservation(e):
    """One world with physical outcomes plus explicitly censored external demand."""
    from .fixtures import world, seed, feasible, submit, advance
    requests=(ArrivalRequest("rerouted_request",0.,0,.8,"short"),
              ArrivalRequest("unrevealed_request",1000.,2,.8,"long"))
    w=world(e,manifest=ArrivalManifest(requests,seed=11))
    w.auto_emergency=False
    e.action("scripted_statistics_fixture",{"auto_emergency":False,
        "failure_policy":"withhold further commands from named failure aircraft; eligibility monitoring remains active",
        "not_a_research_baseline":True})
    seed(e,w,"normal","inner",0,.8,"short")
    seed(e,w,"emergency","inner",9,.3,"standard",True)
    normal=feasible(e,generate_final,"normal","inner",0,0,0)
    emergency=feasible(e,generate_final,"emergency","inner",9,3,0,emergency=True)
    target_failure=max(normal.end_time,emergency.end_time)+1.
    hold=slot_state("outer",0,0.,w.config)
    failure_soc=evaluate_power(hold.v,hold.a,w.config).battery_w*target_failure/(85*252*3600)
    e.action("named_fixture_failure_SOC",{"soc":failure_soc,"target_failure_time":target_failure,
        "basis":"actual outer hold battery power times latest actual final end plus1s"})
    seed(e,w,"failure","outer",0,failure_soc,"long")
    seed(e,w,"censored","outer",18,.8,"short")
    result=submit(e,w,(normal,emergency))
    e.check(result.accepted,"normal and opposite emergency finals commit atomically",True,asdict(result))
    def scripted(snapshot,wrld):
        if snapshot.time==0. and "rerouted_request" not in wrld.aircraft:
            e.action("reroute",{"request_id":"rerouted_request","reason":"SCRIPTED_CONSERVATION_FIXTURE"})
            result=wrld.reroute("rerouted_request","SCRIPTED_CONSERVATION_FIXTURE")
            e.observe("reroute",asdict(result))
            e.check(not result.accepted and result.reason=="REROUTE_REQUIRES_OUTER_CAPACITY",
                    "arbitrary policy deletion is forbidden; request remains external",False,result.accepted)
    advance(e,w,target_failure+1.,scripted)
    metrics=w.metrics()
    e.check(w.failed and w.aircraft["normal"].touchdown_time is not None and
        w.aircraft["emergency"].touchdown_time is not None,
        "both real landings precede real scripted energy failure",True,
        {"time":w.time,"statuses":{k:a.status for k,a in w.aircraft.items()},"metrics":metrics})
    e.check(abs(w.time-target_failure)<1e-5,"named energy failure occurs1s after latest touchdown",
        target_failure,w.time)
    e.check(metrics["normal_landed"]==1 and metrics["emergency_landed"]==1 and
        metrics["rerouted"]==0 and metrics["censored_upstream"]==1 and metrics["energy_failures"]==1 and metrics["censored"]==1 and metrics["unreleased"]==1,
        "real landings/failure retain censored physical and external populations",[1,1,0,1,1,1,1],
        [metrics[k] for k in ("normal_landed","emergency_landed","rerouted","censored_upstream","energy_failures","censored","unreleased")])
    # Fixture seeds are explicit initial offered demand, separate from the
    # immutable manifest. Include them on both sides of the first equation.
    e.check(metrics["planned_manifest"]==metrics["manifest_offered"]+metrics["unreleased"],
        "planned manifest equals revealed plus unreleased",metrics["planned_manifest"],
        metrics["manifest_offered"]+metrics["unreleased"])
    e.check(metrics["offered"]==metrics["admitted"]+metrics["rerouted"]+metrics["upstream_waiting"],
        "offered equals admitted plus capacity rerouted plus external waiting",metrics["offered"],
        metrics["admitted"]+metrics["rerouted"]+metrics["upstream_waiting"])
    e.check(metrics["admitted"]==metrics["normal_landed"]+metrics["emergency_landed"]+
        metrics["failed_before_landing"]+metrics["airborne_or_censored"],
        "admitted equals physical landings plus failures plus airborne/censored",metrics["admitted"],
        metrics["normal_landed"]+metrics["emergency_landed"]+metrics["failed_before_landing"]+metrics["airborne_or_censored"])
    return True


def search_peak_transfer_pair(e, *, max_seconds=300., max_pairs=100):
    """Bounded search for two actual sub120s peak pieces summing above120s."""
    from .fixtures import candidate, FixtureBlocked
    started=time.monotonic(); pairs=0
    for j,first_duration in product(range(7,13),(100.,120.,140.)):
        if time.monotonic()-started>max_seconds: break
        first=candidate(e,generate_transfer,"actual_peak","outer",0,j,0.,durations=[first_duration])
        if first.plan is None: continue
        q1=integrate_energy(first.plan,0.,first.plan.end_time)
        e.observe("first_actual_peak_segment",{"middle_slot":j,"duration":first_duration,"energy":asdict(q1)})
        if not q1.valid or not 0<q1.peak_seconds<120: continue
        base=j*18//28
        for offset,second_duration in product(range(4,8),(100.,120.,140.)):
            if time.monotonic()-started>max_seconds or pairs>=max_pairs:
                raise FixtureBlocked(f"E07 actual pair bounded NOT_FOUND: pairs={pairs},elapsed={time.monotonic()-started:.3f}s")
            pairs+=1; k=(base+offset)%18
            second=candidate(e,generate_transfer,"actual_peak","middle",j,k,first.plan.end_time,
                source_version=1,durations=[second_duration])
            if second.plan is None: continue
            q2=integrate_energy(second.plan,second.plan.start_time,second.plan.end_time)
            e.observe("actual_peak_pair",{"middle_slot":j,"inner_slot":k,"durations":[first_duration,second_duration],
                "first_energy":asdict(q1),"second_energy":asdict(q2),"peak_sum":q1.peak_seconds+q2.peak_seconds})
            if q2.valid and 0<q2.peak_seconds<120 and q1.peak_seconds+q2.peak_seconds>120:
                e.plans.extend((first.plan,second.plan))
                return first.plan,second.plan,q1,q2
    raise FixtureBlocked(f"E07 actual pair bounded NOT_FOUND: pairs={pairs},elapsed={time.monotonic()-started:.3f}s")


def actual_cumulative_peak(e):
    from .fixtures import world,seed,submit,advance
    first,second,q1,q2=search_peak_transfer_pair(e)
    w=world(e); w.auto_emergency=False
    seed(e,w,"actual_peak","outer",0,.9)
    accepted=submit(e,w,(first,))
    e.check(accepted.accepted,"first real high-peak transfer starts",True,asdict(accepted))
    advance(e,w,first.end_time)
    prior=w.aircraft["actual_peak"].peak_seconds
    e.check(abs(prior-q1.peak_seconds)<1e-5 and 0<prior<120,
        "first executed transfer accumulates its actual integrated peak",q1.peak_seconds,prior)
    rejected=submit(e,w,(second,))
    e.check(q2.peak_seconds<120 and prior+q2.peak_seconds>120 and rejected.reason=="PEAK_BUDGET",
        "second individually under120s segment rejected by visit cumulative budget", "PEAK_BUDGET",
        {"prior_executed":prior,"second_integrated":q2.peak_seconds,"result":asdict(rejected)})
    e.check(w.aircraft["actual_peak"].peak_seconds==prior and "actual_peak" not in w.active,
        "second rejection retains first executed peak history",prior,w.aircraft["actual_peak"].peak_seconds)
    return True


def capacity_above_slots(e, *, max_seconds=540., max_attempts=150):
    """Finite actual admission search; a failed search never earns A05 pass."""
    from .fixtures import world, full_slots, candidate, submit, advance, FixtureBlocked
    started=time.monotonic(); attempts=0
    def bounded():
        if time.monotonic()-started > max_seconds or attempts>=max_attempts:
            raise FixtureBlocked(f"A05 bounded chain search exhausted: attempts={attempts}, elapsed={time.monotonic()-started:.3f}s; inspect observations")
    # Final-vs-other-inner checks prune only continuously certified conflicts;
    # every accepted chain is still checked by the production atomic submit.
    finals=[]
    for inner_slot,pad in product((1,0),range(6)):
        bounded()
        result=candidate(e,generate_final,f"inner_{inner_slot}","inner",inner_slot,pad,0.)
        if result.plan is None:
            continue
        plan=result.plan
        rejection=None
        for j in range(18):
            if j==inner_slot: continue
            other=make_stable_plan(f"inner_{j}","inner",j,end_time=plan.end_time+180.)
            check=validate_separation(plan,other,0.,plan.end_time+180.)
            if not check.accepted:
                rejection={"other_slot":j,**asdict(check)}; break
        e.observe("full_inner_departure_screen",{"inner_slot":inner_slot,"pad":pad,"plan":plan.trajectory_hash,
            "accepted":rejection is None,"first_rejection":rejection})
        if rejection is None: finals.append(plan)
    if not finals:
        raise FixtureBlocked("A05 bounded final departure candidates NOT_FOUND against full inner ring; all actual rejection certificates preserved")
    for final,middle_time,outer_time,slot in product(finals,(.001,5.,10.,15.,20.),(.002,10.,20.,30.,40.),(1,0)):
        bounded(); attempts+=1
        middle=candidate(e,generate_transfer,f"middle_{slot}","middle",slot,final.source_slot,middle_time)
        outer=candidate(e,generate_transfer,f"outer_{slot}","outer",slot,slot,outer_time)
        if middle.plan is None or outer.plan is None: continue
        # A fresh immutable manifest is needed after each attempted real
        # admission, because a rejected request is irrevocably rerouted.
        request=ArrivalRequest("extra_inbound",0.,0,.9,"short")
        w=world(e,manifest=ArrivalManifest((request,),seed=11)); w.auto_emergency=False
        full_slots(e,w)
        chain=(final,middle.plan,outer.plan)
        result=submit(e,w,chain)
        if not result.accepted: continue
        e.plans.extend(chain)
        for duration in (55.,75.,95.):
            bounded()
            inbound=candidate(e,generate_inbound,"extra_inbound",0,slot,0.,durations=[duration])
            if inbound.plan is None: continue
            outcome=[]
            def admit(snapshot,wrld):
                if snapshot.time==0. and "extra_inbound" not in wrld.aircraft:
                    e.action("admit",{"request_id":"extra_inbound","trajectory_hash":inbound.plan.trajectory_hash})
                    outcome.append(wrld.admit("extra_inbound",inbound.plan))
            advance(e,w,.001,admit)
            e.observe("capacity_admission",{"duration":duration,"result":[asdict(r) for r in outcome],"metrics":w.metrics()})
            if outcome and outcome[0].accepted:
                e.plans.append(inbound.plan)
                physical=sum(a.status in AIRBORNE for a in w.aircraft.values())
                e.check(physical==84 and w.metrics()["physical_airborne_count"]==84 and not w.failed,
                    "83 slot aircraft plus actually admitted extra airborne before touchdown",84,w.metrics())
                return True
            break
    raise FixtureBlocked(f"A05 bounded chain search NOT_FOUND after {attempts} actual submissions; raw reasons preserved")


def known_83_plus_inflight(e, *, until=.001):
    """Actual certified construction found by the recorded finite A05 search.

    The 84th body enters through real admission. Initial fixture seeding is
    limited to the 83 distinct legal slots; no total-airborne cap is bypassed.
    """
    from .fixtures import world, full_slots, feasible, submit, advance
    request=ArrivalRequest("extra_inbound",0.,0,.9,"short")
    w=world(e,manifest=ArrivalManifest((request,),seed=11))
    w.auto_emergency=False
    full_slots(e,w)
    final=feasible(e,generate_final,"inner_0","inner",0,5,0.,durations=[35.])
    middle=feasible(e,generate_transfer,"middle_1","middle",1,0,.001,durations=[40.])
    outer=feasible(e,generate_transfer,"outer_1","outer",1,1,.002,durations=[40.])
    inbound=feasible(e,generate_inbound,"extra_inbound",0,1,0.,durations=[55.])
    result=submit(e,w,(final,middle,outer))
    e.check(result.accepted,"certified vacancy chain submitted atomically",True,asdict(result))
    results=[]
    def admit(snapshot,wrld):
        if snapshot.time==0. and "extra_inbound" not in wrld.aircraft:
            e.action("admit",{"request_id":"extra_inbound","trajectory_hash":inbound.trajectory_hash})
            results.append(wrld.admit("extra_inbound",inbound))
    advance(e,w,until,admit)
    metrics=w.metrics()
    e.check(results and results[0].accepted and sum(a.status in AIRBORNE for a in w.aircraft.values())==84
        and metrics["physical_airborne_count"]==84 and metrics["fixture_count"]==83
        and metrics["slot_reservations_active"]==83 and not w.failed,
        "83 legal slot fixtures plus one physically admitted airborne aircraft",84,
        {"admission":[asdict(r) for r in results],"metrics":metrics})
    return w


def execute_case(case,e):
    if case=="E07": return actual_cumulative_peak(e)
    if case=="E14": return mixed_conservation(e)
    if case=="A05":
        known_83_plus_inflight(e)
        return True
    return False
