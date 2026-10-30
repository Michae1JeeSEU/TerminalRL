"""Small nominal-completion and shared service-queue checks; no episode runs."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from terminal_marl.core.rotating_slots.stage2_downstream import (
    Demand, JointDownstream, SERVICE_EVALUATION_PROFILE as PROFILE, ticks, joules)
from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, EvaluationContext, PredictionColumn)
from terminal_marl.core.rotating_slots.stage2_service import service_nominal_completion
from terminal_marl.core.rotating_slots.selective_replanning.frame import build_global_frame
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.trajectory import generate_transfer, generate_final
from terminal_marl.core.rotating_slots.resources import Reservation


def column(aid,kind="WAIT"):
    return PredictionColumn(aid,kind,() if kind=="WAIT" else (NS(trajectory_hash=kind),),(),(0,)*5)


def demand(aid,*,ready=20,baseline=0,pads=(0,1),base=100,rate=0,reserve=100000,available=1000000,
           full=False,real=(),landed=0,unknown=1,duration=20):
    return Demand(aid,full,ready,duration,base,rate,reserve,unknown,0,landed,ready+135,
        None if full else "inner",dict(evaluation_profile=PROFILE,reachable_pads=pads,
            service_baseline_ready=baseline,available_energy_threshold=available,
            explicit_prefix_peak_failure=0,explicit_prefix_j=base*.1,
            estimated_adjustment_wait_j=0.,estimated_adjustment_wait_s=0.,
            estimated_suffix_motion_j=0.,estimated_suffix_motion_s=13.5),real)


def joint(demands,*,fixed=(),calendar=(),order=None,pads=2,baseline=None):
    return JointDownstream(demands,calendar,tuple(order or sorted({d.aircraft_id for d in (*demands.values(),*fixed)})),
        pad_count=pads,fixed_demands=fixed,evaluation_profile=PROFILE,baseline_demands=baseline)


def test_alternative_free_pad_and_each_interval_allocated_once():
    a,b=column("a"),column("b")
    ds={a.signature:demand("a",ready=0),b.signature:demand("b",ready=0)}
    j=joint(ds,calendar=((0,0,100,"ground"),))
    _,rows=j.evaluate((a,b))
    assert rows["a"]["virtual_pad"]==rows["b"]["virtual_pad"]==1
    assert rows["a"]["service_start"]==0.
    assert rows["b"]["service_start"]==2.


def test_real_full_interval_replaces_virtual_and_blocks_once():
    a,b=column("a"),column("b")
    ds={a.signature:demand("a",ready=0,full=True,real=((0,0,30,"a"),),landed=-1),
        b.signature:demand("b",ready=0,pads=(0,))}
    _,rows=joint(ds,pads=1).evaluate((a,b))
    assert rows["a"]["virtual_pad"] is None
    assert rows["b"]["service_start"]==3.


def test_same_window_lower_energy_waiting_wins_without_ring_reward():
    wait,move=column("a"),column("a","MOVE")
    w=demand("a",ready=30,base=100); m=replace(demand("a",ready=10,base=200),ring="middle")
    j=joint({wait.signature:w,move.signature:m},calendar=((0,0,100,"g"),(1,0,100,"h")),baseline=(w,))
    wk,wr=j.evaluate((wait,)); mk,mr=j.evaluate((move,))
    assert wk[3]==mk[3]==0
    assert wr["a"]["service_start"]==mr["a"]["service_start"]==10.
    assert j.key((wait,))<j.key((move,))


def test_real_busy_pads_sufficient_inner_stock_keeps_outer_at_lower_energy():
    """Physical fixtures plus actual O->M quadrature; missing suffix stays nominal.

    Six ground services end at 100 s. Six real INNER holding aircraft then
    consume the first predicted common service window. Neither OUTER choice
    creates an earlier service opportunity, while waiting in OUTER costs less.
    This checks the score and original validators, not a full episode rollout.
    """
    import json
    from terminal_marl.core.rotating_slots.selective_replanning import PlanningScope
    world=RotatingSlotWorld(); world.auto_emergency=False
    world.seed_hold("outer","outer",0,.9)
    for i in range(6): world.seed_hold("inner_"+str(i),"inner",3*i,.9)
    for i in range(6): world.seed_service("ground_"+str(i),i,100.)
    snapshot=world.build_planning_snapshot(); p=provider()
    frame=build_global_frame(snapshot,p,EvaluationContext(snapshot))
    eligible=tuple(aid for aid,a in sorted(snapshot.aircraft.items()) if a.status.startswith("HOLD_"))
    fixed=tuple(aid for aid in sorted(snapshot.aircraft) if aid!="outer")
    scope=PlanningScope(snapshot.version,snapshot.time,eligible,("outer",),fixed,
        actual_k=1,objective_frame_id=frame.identity)
    context=EvaluationContext(snapshot,scope=scope,global_frame=frame)
    keep=p._column(snapshot,snapshot.aircraft["outer"],(),"KEEP",context)
    generated=generate_transfer("outer","outer",0,0,snapshot.time,durations=(40,),config=snapshot.config)
    assert generated.accepted and generated.plan is not None
    plan=generated.plan
    move=p._column(snapshot,snapshot.aircraft["outer"],(plan,),"MOVE",context)
    demands={c.signature:p._describe_column(snapshot,c,context) for c in (keep,move)}
    evaluator=frame.downstream(demands,{"outer"})
    assert all("inner_"+str(i) in evaluator.fixed_demands for i in range(6))
    assert all(frame.order.index("inner_"+str(i))<frame.order.index("outer") for i in range(6))
    keep_key,keep_rows=evaluator.evaluate((keep,)); move_key,move_rows=evaluator.evaluate((move,))
    kr,mr=keep_rows["outer"],move_rows["outer"]
    assert keep_key[:4]==move_key[:4]  # F, -L, R, G all tie.
    assert demands[keep.signature].landed==demands[move.signature].landed==0
    assert kr["normal_reserve_shortfall_j"]==mr["normal_reserve_shortfall_j"]==0.
    assert kr["service_opportunity_delay_s"]==mr["service_opportunity_delay_s"]==0.
    assert kr["service_start"]==mr["service_start"]==233.5
    assert kr["estimated_touchdown"]==mr["estimated_touchdown"]==247.
    assert {keep_rows["inner_"+str(i)]["service_start"] for i in range(6)}=={100.}
    assert kr["terminal_wait_ring"]=="outer" and mr["terminal_wait_ring"]=="middle"
    actual=context.energy_portions(((plan,plan.start_time,plan.end_time),))[0]
    assert mr["explicit_prefix_j"]==pytest.approx(actual)
    assert kr["estimated_suffix_motion_j"]==pytest.approx(actual+mr["estimated_suffix_motion_j"])
    assert kr["total_j"]<mr["total_j"] and evaluator.key((keep,))<evaluator.key((move,))
    checks={}
    for c in (keep,move):
        single=context.check_single(c); batch,_=context.check_batch((c,)); prefix=context.check_commit_prefix((c,))
        assert single.accepted and batch.accepted and prefix.accepted
        checks[c.kind]=dict(single=single.status,batch=batch.status,prefix=prefix.status)
    assert world.build_planning_snapshot()==snapshot
    print(json.dumps(dict(fixture="real_world_seed_hold_and_seed_service",pad_busy_until_s=100.,
        inner_stock=6,service_class="standard",transfer_s=40.,
        service_start_s=kr["service_start"],touchdown_s=kr["estimated_touchdown"],
        keep_energy_j=kr["total_j"],move_energy_j=mr["total_j"],actual_transfer_energy_j=actual,
        keep_wait_s=kr["estimated_wait_seconds_by_ring"]["outer"],
        move_wait_s=mr["estimated_wait_seconds_by_ring"]["middle"],checks=checks),sort_keys=True))


def test_genuine_opportunity_shortfall_and_risk_are_separate():
    wait,move=column("a"),column("a","MOVE")
    w=demand("a",ready=40,baseline=10,base=100,reserve=50)
    m=demand("a",ready=10,baseline=10,base=100,reserve=50,unknown=0)
    j=joint({wait.signature:w,move.signature:m},baseline=(w,))
    wk,wr=j.evaluate((wait,)); mk,mr=j.evaluate((move,))
    assert wk[:3]==mk[:3]==(0,0,50)
    assert wk[3]==30 and mk[3]==0
    assert wr["a"]["uncertainty_risk_flag"] and not mr["a"]["uncertainty_risk_flag"]
    assert not wr["a"]["unknown_risk_flag"]


def test_no_demand_out_of_window_and_known_blocking_masks():
    assert joint({}).evaluate(())[0]==(0,)*6
    a=column("a")
    for d in (demand("a",ready=3000,baseline=2500),demand("a",pads=())):
        key,rows=joint({a.signature:d}).evaluate((a,))
        assert key[3]==0 and not rows["a"]["service_opportunity_eligible"]


def provider():
    return CandidateProvider(planning_profile="downstream_service_v8",evaluation_profile=PROFILE)


def test_nominal_suffix_is_feasible_quadrature_and_never_physical_lower_bound(monkeypatch):
    import terminal_marl.core.rotating_slots.stage2_downstream as downstream
    import terminal_marl.core.rotating_slots.rescue_rules as rescue
    def forbidden(*a,**k): pytest.fail("physical lower bound used as point estimate")
    monkeypatch.setattr(downstream,"normal_energy_lower_bound",forbidden)
    monkeypatch.setattr(rescue,"normal_energy_lower_bound",forbidden)
    world=RotatingSlotWorld(); world.seed_hold("a","middle",0,.9)
    s=world.build_planning_snapshot(); p=provider(); context=EvaluationContext(s)
    # Bypass legacy column materialization, which retains its safety/slack paths.
    c=PredictionColumn("a","WAIT",(),(),(0,)*5)
    d=p._describe_column(s,c,context)
    energy,seconds,metadata=service_nominal_completion(s.config,"middle")
    assert d.diagnostics["estimated_suffix_motion_j"]==pytest.approx(energy)
    assert d.diagnostics["estimated_suffix_motion_s"]==seconds
    assert metadata["primitives"] and all(row["energy_j"]>0 for row in metadata["primitives"])
    assert d.diagnostics["energy_estimate_bound"]=="no_bound_claim" and d.unknown==1
    assert d.landed==0 and d.diagnostics["uncertainty_risk_penalty_wh"]==0.


def test_identical_move_keep_physical_prefix_energy_and_timeline():
    world=RotatingSlotWorld(); world.seed_hold("a","middle",0,.9)
    plan=generate_transfer("a","middle",0,0,0.,durations=(40,),config=world.config).plan
    assert world.submit((plan,)).accepted
    s=world.build_planning_snapshot(); p=provider(); context=EvaluationContext(s)
    keep=p._column(s,s.aircraft["a"],(),"WAIT",context)
    # A candidate repeating the same physical commitment is deduplicated by
    # context.all_plans; it must not charge the transfer twice.
    move=replace(keep,kind="MOVE",plans=(plan,))
    kd=p._describe_column(s,keep,context); md=p._describe_column(s,move,context)
    assert kd.base_energy==md.base_energy
    for field in ("explicit_prefix_j","timeline_explicit_end","timeline_anchor","estimated_suffix_motion_j"):
        assert kd.diagnostics[field]==md.diagnostics[field]


def test_background_frame_uses_same_evaluator_and_common_preselection_baseline():
    world=RotatingSlotWorld()
    for i,aid in enumerate(("a","b")): world.seed_hold(aid,"inner",i,.9)
    s=world.build_planning_snapshot(); p=provider(); context=EvaluationContext(s)
    frame=build_global_frame(s,p,context)
    c=p._column(s,s.aircraft["a"],(),"WAIT",context); d=p._describe_column(s,c,context)
    j=frame.downstream({c.signature:d},{"a"})
    assert len(j.baseline_demands)==2 and set(j.fixed_demands)=={"b"}
    _,rows=j.evaluate((c,))
    assert len(rows)==2
    assert all(row["service_opportunity_delay_s"]>=0 for row in rows.values())


def test_indefinite_pad_is_excluded_and_partial_never_counts_as_landing():
    world=RotatingSlotWorld(); world.seed_hold("a","inner",0,.9)
    s=world.build_planning_snapshot()
    s=replace(s,pads=(Reservation(0,"busy",0.,float("inf")),))
    d=provider()._describe_column(s,PredictionColumn("a","WAIT",(),(),(0,)*5),EvaluationContext(s))
    assert 0 not in d.diagnostics["reachable_pads"] and d.landed==0
    assert d.diagnostics["resource_occupancy_unknown"]
    assert not d.diagnostics["known_blocked_pads"]


def test_existing_cached_reference_does_not_change_missing_suffix_model():
    from terminal_marl.core.rotating_slots.stage2_planning import NormalReference, _node
    world=RotatingSlotWorld(); world.seed_hold("a","inner",0,.9)
    s=world.build_planning_snapshot(); p=provider(); context=EvaluationContext(s)
    c=PredictionColumn("a","WAIT",(),(),(0,)*5)
    before=p._describe_column(s,c,context)
    node=_node(s,s.aircraft["a"])
    final=generate_final("a","inner",0,0,20.,config=s.config).plan
    p.references.move_energy(node,final,s.config)
    key=(s.config.config_hash,node.ring,node.slot,20.,node.version,"a")
    p.references.cache[key]=NormalReference("KNOWN",(final,),duration=final.end_time-final.start_time)
    after=p._describe_column(s,c,context)
    assert before.base_energy==after.base_energy and before.ready==after.ready
    assert after.diagnostics["reference_status"]=="ESTIMATED"


def test_queue_wait_timeline_energy_segments_are_disjoint_and_complete():
    world=RotatingSlotWorld(); world.seed_hold("a","middle",0,.9)
    s=world.build_planning_snapshot(); p=provider(); context=EvaluationContext(s)
    c=p._column(s,s.aircraft["a"],(),"WAIT",context); d=p._describe_column(s,c,context)
    j=joint({c.signature:d},pads=1,calendar=((0,0,2000,"ground"),))
    _,rows=j.evaluate((c,)); diag=rows["a"]; segments=diag["timeline_segments"]
    assert diag["estimated_queue_wait_s"]>0
    assert all(a["end"]==pytest.approx(b["start"],abs=.1) for a,b in zip(segments,segments[1:]))
    assert sum(row["energy_j"] for row in segments)==pytest.approx(diag["component_sum_j"])
    assert diag["component_sum_j"]+diag["base_energy_quantization_j"]==pytest.approx(diag["total_j"])


@pytest.mark.parametrize("selection",(("WAIT","WAIT"),("MOVE","WAIT"),("WAIT","MOVE"),("MOVE","MOVE")))
def test_cp_and_python_exact_six_components(selection):
    cp=pytest.importorskip("ortools.sat.python.cp_model")
    columns=[column(aid,kind) for aid in ("a","b") for kind in ("WAIT","MOVE")]
    ds={c.signature:demand(c.aircraft_id,ready=(30 if c.kind=="WAIT" else 10),
        baseline=0,base=(100 if c.kind=="WAIT" else 130),rate=2,reserve=90,
        pads=((0,1) if c.aircraft_id=="a" else (0,)),
        full=c.aircraft_id=="b" and c.kind=="MOVE",
        real=((0,15,60,"b"),) if c.aircraft_id=="b" and c.kind=="MOVE" else (),
        landed=-int(c.aircraft_id=="b" and c.kind=="MOVE")) for c in columns}
    baseline=(ds[columns[0].signature],ds[columns[2].signature])
    j=joint(ds,calendar=((1,0,40,"g"),),baseline=baseline)
    model=cp.CpModel(); variables={c.signature:model.NewBoolVar(str(c.signature)) for c in columns}
    selected=tuple(next(c for c in columns if c.aircraft_id==aid and c.kind==kind) for aid,kind in zip(("a","b"),selection))
    for c in columns: model.Add(variables[c.signature]==int(c in selected))
    expressions=j.cp_expressions(model,variables,{c.signature:c for c in columns})
    solver=cp.CpSolver(); status=solver.Solve(model)
    assert status in (cp.OPTIMAL,cp.FEASIBLE)
    assert tuple(solver.Value(e) for e in expressions)==j.evaluate(selected)[0]


@pytest.mark.parametrize("case",("selected_blocked","fixed_background","no_demand","negative_full_ready"))
def test_cp_parity_for_masks_background_and_ground_times(case):
    cp=pytest.importorskip("ortools.sat.python.cp_model")
    a=column("a"); d=demand("a",ready=30,baseline=0)
    options={a.signature:d}; fixed=(); baseline=(d,); selected=(a,)
    if case=="selected_blocked": options={a.signature:replace(d,diagnostics=dict(d.diagnostics,reachable_pads=()))}
    elif case=="fixed_background": fixed=(demand("b",ready=10,baseline=0,pads=(0,),base=200,rate=3,reserve=150),)
    elif case=="negative_full_ready":
        d=replace(d,full=True,ready=-20,touchdown=115,hold_per_tick=0,landed=-1)
        options={a.signature:d}; baseline=(d,)
    else: options={}; selected=(); baseline=()
    j=joint(options,fixed=fixed,calendar=((0,0,50,"ground"),),baseline=(*baseline,*fixed))
    model=cp.CpModel(); variables={key:model.NewBoolVar(str(key)) for key in options}
    for v in variables.values(): model.Add(v==1)
    expressions=j.cp_expressions(model,variables,{a.signature:a} if options else {})
    solver=cp.CpSolver(); status=solver.Solve(model)
    assert status in (cp.OPTIMAL,cp.FEASIBLE)
    assert tuple(solver.Value(e) for e in expressions)==j.evaluate(selected)[0]
