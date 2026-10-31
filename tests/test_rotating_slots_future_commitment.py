"""Physical first-action appointments: waiting, publication, execution and replay."""
from dataclasses import replace

import pytest

from terminal_marl.core.rotating_slots.config import thaw
from terminal_marl.core.rotating_slots.future_commitment import record_for_plan, maintain_reservations
from terminal_marl.core.rotating_slots.planning_latency import (
    PlanningTiming, prepare_submission, mark_submission_ready, publish_submission,
    project_visible_snapshot, guard_submission, LatencyAwareScheduler, LOCAL_TRAFFIC_POLICY)
from terminal_marl.core.rotating_slots.stage2_downstream import SERVICE_EVALUATION_PROFILE
from terminal_marl.core.rotating_slots.stage2_planning import EvaluationContext, NormalSchedulerCallback
from terminal_marl.core.rotating_slots.stage2_campaign import (
    save_checkpoint, restore_checkpoint, checkpoint_identity, digest)
from terminal_marl.core.rotating_slots.trajectory import generate_transfer, generate_final, make_stable_plan
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID, EmergencyDependencyConflict


def fixture(start=90.3):
    w=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    w.auto_emergency=False
    w.seed_hold('a','middle',0,.9)
    w.advance(50.289792,False)
    timing=PlanningTiming(50.289792,70.289792,69.289792)
    p=generate_transfer('a','middle',0,0,start,config=w.config).plan
    assert p is not None
    return w,timing,p


def inject_hold(w,aid,ring,slot,soc=.9):
    base=RotatingSlotWorld(config=w.config)
    a=base.seed_hold(aid,ring,slot,soc)
    hold=make_stable_plan(aid,ring,slot,0.,1000.,config=w.config)
    state=hold.state(w.time)
    w.aircraft[aid]=replace(a,p=state.p,v=state.v,a=state.a,jerk=state.jerk,
        trajectory_hash=hold.trajectory_hash)
    w.trajectories[hold.trajectory_hash]=hold.to_dict()
    w.slots.reserve((ring,slot),aid,w.time,float('inf'),kind='occupied')


def prepared(w,timing,plans):
    result,token=prepare_submission(w,plans,timing)
    assert result.accepted,result
    w.advance(max(w.time+1.,w._planning_latency_packets[token]['not_ready_before']+.01),False)
    ready=mark_submission_ready(w,token)
    assert ready.accepted,ready
    w.advance(timing.effective_time,False)
    published=publish_submission(w,token)
    assert published.accepted,published
    return token,published


def test_absolute_future_middle_first_action_waits_starts_and_completes():
    w,timing,p=fixture()
    _,result=prepared(w,timing,(p,))
    r=record_for_plan(w,p)
    assert r['state']=='RESERVED' and r['start_time']==90.3
    assert w.soft['a'] is p and not w.active
    assert result.details['waiting_for_reservation_start']
    assert result.details['delivery_units'][0]['state']=='RESERVED'
    assert r['hold_trajectory_hash'] and r['certificate_interval']==(70.289792,p.end_time)
    projected=project_visible_snapshot(w.build_planning_snapshot(),90.447)
    assert projected.active['a']==p and 'a' not in projected.soft
    assert projected.aircraft['a'].status=='TRANSFER_M_I'
    w.advance(90.447,False)
    assert w.active['a']==p and not w.soft
    assert r['state']=='EXECUTING'
    assert [e['time'] for e in w.events if e['type']=='reservation_started']==[90.3]
    w.advance(p.end_time+.01,False)
    assert r['state']=='COMPLETED'
    assert w.aircraft['a'].ring=='inner'
    assert [e['time'] for e in w.events if e['type']=='reservation_completed']==[p.end_time]


def test_known_appointment_survives_repeated_projection_and_empty_delivery():
    w,timing,p=fixture();prepared(w,timing,(p,))
    for effective in (75.,80.,85.):
        s=project_visible_snapshot(w.build_planning_snapshot(),effective,
            timing=PlanningTiming(w.time,effective,effective-.1))
        assert s.soft['a']==p and s.soft['a'].start_time==90.3
        # No replacement proposal may erase a world-held appointment.
        result,token=prepare_submission(w,(),PlanningTiming(w.time,effective,effective-.1))
        assert not result.accepted and token is None
    assert w.soft['a']==p


def test_unrelated_outer_traffic_reuses_and_preserves_absolute_appointment():
    w,timing,p=fixture();prepared(w,timing,(p,))
    inject_hold(w,'unrelated','outer',18,.9)
    maintain_reservations(w)
    assert w.soft['a']==p and record_for_plan(w,p)['state']=='RESERVED'
    guards=[e for e in w.events if e['type']=='reservation_traffic_guard']
    assert guards[-1]['details']['accepted']
    assert guards[-1]['details']['envelope_proofs']>=1
    w.advance(90.301,False)
    assert w.active['a']==p


def test_waiting_conflict_cannot_advance_or_execute_uncertified_fallback():
    w,timing,p=fixture();prepared(w,timing,(p,))
    # Direct external state injection deliberately bypasses submit/admit's
    # existing joint-continuation check. Conflict already affects the wait.
    inject_hold(w,'intruder','outer',18,.9)
    own=w.aircraft['a']; intruder=w.aircraft['intruder']
    w.aircraft['intruder']=replace(intruder,ring='middle',slot=0,
        p=own.p,v=own.v,a=own.a,jerk=own.jerk,trajectory_hash=own.trajectory_hash)
    before=w.time
    with pytest.raises(EmergencyDependencyConflict,match='RESERVATION_FALLBACK_UNCERTIFIED'):
        w.advance(w.time+.2,False)
    assert w.time==before and 'a' not in w.active
    assert w.soft['a']==p


def test_wait_energy_and_resource_conflicts_reject_before_publication():
    w,timing,p=fixture()
    a=w.aircraft['a']
    action=w._energy(p,p.start_time,p.end_time,fine=True)
    w.aircraft['a']=replace(a,soc=action.soc_used+.000001)
    result,token=prepare_submission(w,(p,),timing)
    assert not result.accepted and token is None
    assert not w.soft and not w.active
    w,timing,p=fixture()
    inject_hold(w,'target_owner','inner',0,.9)
    result,token=prepare_submission(w,(p,),timing)
    assert not result.accepted and token is None
    assert result.reason=='RESOURCE_CONFLICT'


def test_cancelled_or_corrupt_identity_never_uses_legacy_start_path():
    w,timing,p=fixture();prepared(w,timing,(p,))
    r=record_for_plan(w,p)
    r['state']='CANCELLED';r['certificate_valid']=False
    w.advance(90.301,False)
    assert 'a' not in w.active and 'a' not in w.soft
    assert not any(e['type']=='maneuver_started' and e['aircraft_id']=='a' for e in w.events)


def test_transactional_replacement_keeps_old_on_failure_then_publishes_new():
    w,timing,p=fixture(130.3);prepared(w,timing,(p,))
    old=record_for_plan(w,p)
    next_timing=PlanningTiming(w.time,w.time+20.,w.time+19.)
    bad=replace(p,source_version=999,trajectory_hash='')
    rejected,token=prepare_submission(w,(bad,),next_timing)
    assert not rejected.accepted and token is None and w.soft['a']==p
    new=generate_transfer('a','middle',0,0,110.3,config=w.config).plan
    _,result=prepared(w,next_timing,(new,))
    assert old['state']=='CANCELLED' and old['replacement_reservation_id']==result.details['reservation_ids'][0]
    assert w.soft['a']==new and record_for_plan(w,new)['state']=='RESERVED'
    w.advance(110.301,False)
    assert w.active['a']==new


def test_expired_or_unready_replacement_retains_old_absolute_time():
    w,timing,p=fixture(130.3);prepared(w,timing,(p,))
    t=PlanningTiming(w.time,w.time+20.,w.time+19.)
    new=generate_transfer('a','middle',0,0,110.3,config=w.config).plan
    result,token=prepare_submission(w,(new,),t)
    assert result.accepted
    w.advance(t.effective_time,False)
    rejected=publish_submission(w,token)
    assert not rejected.accepted and rejected.reason=='NO_TIMELY_READY_RECEIPT'
    assert w.soft['a']==p and p.start_time==130.3


def test_mixed_immediate_final_and_future_middle_publish_independently():
    w,timing,p=fixture();inject_hold(w,'final','inner',9,.9)
    f=generate_final('final','inner',9,3,timing.effective_time,config=w.config).plan
    assert f is not None
    _,result=prepared(w,timing,(f,p))
    assert {unit['aircraft_id']:unit['state'] for unit in result.details['delivery_units']}=={
        'final':'EXECUTING','a':'RESERVED'}
    assert w.active['final']==f and w.soft['a']==p
    assert record_for_plan(w,f)['state']=='EXECUTING'
    assert record_for_plan(w,p)['state']=='RESERVED'
    w.advance(90.301,False)
    assert w.active['a']==p
    for plan,when in ((f,timing.effective_time),(p,90.3)):
        rid=record_for_plan(w,plan)['reservation_id']
        published=[e for e in w.events if e['type']=='reservation_published' and e['details']['reservation_id']==rid]
        started=[e for e in w.events if e['type']=='reservation_started' and e['details']['reservation_id']==rid]
        assert len(published)==len(started)==1
        assert published[0]['time']==timing.effective_time and started[0]['time']==when


def test_published_certificate_checkpoint_roundtrip_preserves_start_and_events(tmp_path):
    w,timing,p=fixture();prepared(w,timing,(p,))
    def scheduler():
        return LatencyAwareScheduler(NormalSchedulerCallback('energy_slack',planning_profile='downstream_service_v8',
            evaluation_profile=SERVICE_EVALUATION_PROFILE),lead_seconds=20.,traffic_policy=LOCAL_TRAFFIC_POLICY)
    callback=scheduler()
    inject_hold(w,'pending','inner',9,.9)
    pending_timing=PlanningTiming(w.time,w.time+20.,w.time+19.)
    pending_plan=generate_final('pending','inner',9,3,w.time+25.,config=w.config).plan
    pending_result,pending_token=prepare_submission(w,(pending_plan,),pending_timing,traffic_policy=LOCAL_TRAFFIC_POLICY)
    assert pending_result.accepted,pending_result
    identity=checkpoint_identity(dict(source_hash='test',base_config_hash=w.config.config_hash,
        overlay_sha256='test',planning_profile='downstream_service_v8'),digest(w.manifest.to_dict()))
    path=tmp_path/'reserved.json'
    save_checkpoint(path,w,callback,identity)
    loaded=restore_checkpoint(path,scheduler(),identity)
    assert loaded._future_reservations==w._future_reservations
    assert loaded._planning_latency_packets[pending_token]['delivery_units'][0]['state']=='PENDING_DELIVERY'
    loaded.admission_callback=None
    loaded.advance(p.end_time+.01,False)
    assert record_for_plan(loaded,p,include_terminal=True)['state']=='COMPLETED'
    assert [e['time'] for e in loaded.events if e['type']=='reservation_started']==[90.3]
    terminal=record_for_plan(loaded,p,include_terminal=True)
    assert terminal['certificate_snapshot'] is None and terminal['pair_certificates']=={}


def test_future_release_dependencies_are_protected_and_cancel_as_certified_closure():
    w=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID);w.auto_emergency=False
    w.seed_hold('incoming','middle',1,.9);w.seed_hold('landing','inner',0,.9)
    w.advance(50.289792,False)
    timing=PlanningTiming(w.time,w.time+20.,w.time+19.)
    landing=generate_final('landing','inner',0,1,90.3,config=w.config,durations=[50.]).plan
    incoming=generate_transfer('incoming','middle',1,0,90.3,config=w.config,durations=[40.]).plan
    assert landing is not None and incoming is not None
    prepared(w,timing,(landing,incoming))
    assert landing.trajectory_hash in w.dependencies[incoming.trajectory_hash]
    before=dict(w.soft)
    cancelled=w.cancel('landing')
    assert not cancelled.accepted and cancelled.reason=='RESOURCE_DEPENDENCY' and w.soft==before
    # A stale supporting source invalidates the future consumer too. The
    # fallback is jointly checked before either reservation is removed.
    w.aircraft['landing']=replace(w.aircraft['landing'],version=1)
    maintain_reservations(w)
    assert not w.soft and not w.active
    for plan in (landing,incoming):
        record=record_for_plan(w,plan,include_terminal=True)
        assert record['state']=='CANCELLED'
        assert record['terminal_reason']=='RESERVATION_SOURCE_CHANGED'


def test_abnormal_soc_drop_cancels_only_after_finite_wait_energy_certification():
    w,timing,p=fixture();prepared(w,timing,(p,))
    hold=make_stable_plan('a','middle',0,w.time,w.time+20.,config=w.config)
    hold_energy=w._energy(hold,w.time,w.time+20.,fine=True)
    action=w._energy(p,p.start_time,p.end_time,fine=True)
    assert action.soc_used>hold_energy.soc_used
    w.aircraft['a']=replace(w.aircraft['a'],soc=(action.soc_used+hold_energy.soc_used)/2.)
    maintain_reservations(w)
    record=record_for_plan(w,p,include_terminal=True)
    assert record['state']=='CANCELLED' and record['terminal_reason']=='RESERVATION_ENERGY_BOUND_CHANGED'
    assert not w.soft and not w.active


def test_same_hash_reissued_identity_prefers_latest_live_reservation():
    w,timing,p=fixture(130.3);prepared(w,timing,(p,))
    old=record_for_plan(w,p)
    next_timing=PlanningTiming(w.time,w.time+20.,w.time+19.)
    prepared(w,next_timing,(p,))
    current=record_for_plan(w,p,include_terminal=True)
    assert current['reservation_id']!=old['reservation_id']
    assert current['state']=='RESERVED' and old['state']=='CANCELLED'
    assert current['start_time']==old['start_time']==130.3
    w.advance(130.301,False)
    assert w.active['a']==p and current['state']=='EXECUTING'
