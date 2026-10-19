"""Practical numeric reuse and conservative broad-phase regression cases."""
import math
import numpy as np
import pytest
from scipy.optimize import brentq

from terminal_marl.core.rotating_slots import energy as en, safety as sf, trajectory as tr
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.world import _Traffic


def raw_plan(name,segments):
    c=load_config()
    return tr.TrajectoryPlan(name,'INBOUND',segments[0].start_time,segments[-1].end_time,
                             profile_hash=c.profile_hash,config_hash=c.config_hash,segments=segments)


def test_far_endpoints_do_not_hide_midinterval_crossing():
    a=raw_plan('a',(tr.Segment(0.,10.,(200.,-400.),(0.,),(20.,)),))
    b=raw_plan('b',(tr.Segment(0.,10.,(0.,),(0.,),(20.,)),))
    assert math.dist(a.state(0).p,b.state(0).p)>80
    assert math.dist(a.state(10).p,b.state(10).p)>80
    for checker in (sf.validate_separation,sf.validate_separation_original):
        result=checker(a,b,0.,10.)
        assert not result.accepted and result.status=='UNSAFE'


def test_unvalidated_discontinuous_plan_needs_actual_position_bounds():
    a=raw_plan('a',tuple(tr.Segment(t,1.,(radius,),(0.,),(20.,))
                         for t,radius in ((0.,200.),(1.,0.),(2.,200.))))
    b=raw_plan('b',(tr.Segment(0.,3.,(0.,),(0.,),(20.,)),))
    assert not sf.validate_separation(a,b,0.,3.).accepted


@pytest.mark.parametrize('height,accepted',[(80.,True),(math.nextafter(80.,0.),False),(79.9,False)])
def test_exact_eighty_metre_boundary(height,accepted):
    a=raw_plan('a',(tr.Segment(0.,2.,(100.,),(0.,-.1),(0.,)),))
    b=raw_plan('b',(tr.Segment(0.,2.,(100.,),(0.,-.1),(height,)),))
    assert sf.validate_separation(a,b,0.,2.).accepted is accepted


def test_stable_analytic_path_and_generic_traffic_boxes():
    a=tr.make_stable_plan('a','outer',0)
    b=tr.make_stable_plan('b','outer',1)
    result=sf.validate_separation(a,b,0.,1000.)
    assert result.accepted and result.details['method']=='common_rotation_structure_with_roundoff_enclosure'
    ta=_Traffic('a',a,(),0.,1000.); tb=_Traffic('b',b,(),0.,1000.)
    for left,right in ((ta,tb),(a,tb)):
        en.reset_practical_numeric_stats()
        result=sf.validate_separation(left,right,0.,1000.)
        assert result.accepted and result.details['method']=='common_rotation_structure_with_roundoff_enclosure'
        stats=en.practical_numeric_stats()
        assert stats.get('separation_fine_intervals',0)==0
        assert stats.get('separation_box_intervals',0)==0
    future=tr.make_stable_plan('b','outer',0,5.,10.)
    future_traffic=_Traffic('b',b,(future,),0.,10.)
    result=sf.validate_separation(ta,future_traffic,0.,10.)
    assert not result.accepted and result.status=='UNSAFE'
    x=raw_plan('x',(tr.Segment(0.,10.,(100.,),(0.,),(10.,)),))
    y=raw_plan('y',(tr.Segment(0.,10.,(400.,),(0.,),(10.,)),))
    tx=_Traffic('x',x,(),0.,10.); ty=_Traffic('y',y,(),0.,10.)
    en.reset_practical_numeric_stats()
    result=sf.validate_separation(tx,ty,0.,10.)
    assert result.accepted and 80<=result.details['lower_bound_m']<=300
    assert result.details['upper_bound_m']>=300
    assert en.practical_numeric_stats().get('separation_fine_intervals',0)==0


def test_induced_bracket_contains_high_precision_root_and_power_enclosure():
    c=load_config()
    for vp,vn,rhs in [(0.,0.,20.),(2.,-2.,4.),(10.,-1.,80.),(3.,2.,40.)]:
        lo,hi=en._root_bracket(vp,vn,rhs)
        reference=brentq(lambda x:x*math.hypot(vp,vn+x)-rhs,max(0.,-vn),100.,xtol=1e-14)
        assert lo<=reference<=hi
        assert abs(en._root(vp,vn,rhs)*math.hypot(vp,vn+en._root(vp,vn,rhs))-rhs)/rhs<1e-8
    s=tr.descent_segments(0,0.)[0]
    for a,b in ((0.,4.),(.3,.31),(2.,2.1)):
        pb=en.power_bounds(s.intrinsic_bounds(a,b,1),s.intrinsic_bounds(a,b,2),c)
        assert pb is not None
        for t in np.linspace(a,b,11):
            state=s.state(t); p=en.evaluate_power(state.v,state.a,c)
            assert p.valid
            assert pb['battery_w'][0]<=p.battery_w<=pb['battery_w'][1]


def test_shifted_transfer_reuses_numeric_certificate_and_energy():
    sf._PLAN_CERTIFICATES.clear(); sf._NUMERIC_CERTIFICATES.clear(); sf._SEGMENT_CERTIFICATES.clear()
    tr._TRANSFER_FEASIBILITY.clear()
    en._segment_table.cache_clear(); en._segment_primitive.cache_clear(); en._peak_table.cache_clear()
    en.reset_practical_numeric_stats()
    a=tr.generate_transfer('a','outer',0,0,0.)
    assert a.accepted
    ea=en.integrate_energy(a.plan,a.plan.start_time,a.plan.end_time)
    before=en.practical_numeric_stats()
    b=tr.generate_transfer('b','outer',0,0,300.137)
    assert b.accepted
    eb=en.integrate_energy(b.plan,b.plan.start_time,b.plan.end_time)
    after=en.practical_numeric_stats()
    assert a.plan.segments[0].theta[1:]==b.plan.segments[0].theta[1:]
    assert after.get('transfer_feasibility_hits',0)>before.get('transfer_feasibility_hits',0)
    assert b.details['trajectory_hash']==b.plan.trajectory_hash
    assert b.details['profile_hash']==b.plan.profile_hash
    assert after['energy_table_computes']==before['energy_table_computes']
    assert after['peak_table_computes']==before['peak_table_computes']
    assert ea.valid and eb.valid and ea.joules==pytest.approx(eb.joules,rel=1e-12)


def test_shifted_rejected_transfer_does_not_repeat_numeric_validation():
    tr._TRANSFER_FEASIBILITY.clear(); en.reset_practical_numeric_stats()
    a=tr.generate_transfer('a','outer',0,0,0.,durations=(1.,))
    before=en.practical_numeric_stats()
    b=tr.generate_transfer('b','outer',0,0,300.137,source_version=7,durations=(1.,))
    after=en.practical_numeric_stats()
    assert a.status==b.status=='NOT_FOUND'
    assert a.details==b.details
    assert len(a.details['attempts'])==2
    assert before['transfer_feasibility_computes']==2
    assert after['transfer_feasibility_computes']==2
    assert after['transfer_feasibility_hits']==2
    assert after['validate_plan_calls']==before['validate_plan_calls']
    assert after['plan_interval_computes']==before['plan_interval_computes']
    assert all(set(row)=={'duration','reason'} for row in b.details['attempts'])


def test_cumulative_energy_and_peak_are_additive_and_warm():
    p=tr.generate_final('a','inner',0,0,0.).plan
    start,end=p.start_time,p.end_time; mid=start+(end-start)*.417
    whole=en.integrate_energy(p,start,end)
    left=en.integrate_energy(p,start,mid); right=en.integrate_energy(p,mid,end)
    assert whole.valid and left.valid and right.valid
    assert left.joules+right.joules==pytest.approx(whole.joules,rel=1e-12)
    assert left.peak_seconds+right.peak_seconds==pytest.approx(whole.peak_seconds,abs=1e-12)
    en.reset_practical_numeric_stats()
    assert en.integrate_energy(p,mid,end)==right
    assert en.practical_numeric_stats().get('power_array_calls',0)==0
    assert en.practical_numeric_stats().get('peak_table_nodes',0)==0
    # Force a real threshold crossing independently of the frozen continuous limit.
    c=load_config(); s=tr.descent_segments(0,0.)[0]
    sample=[en.evaluate_power(s.state(t).v,s.state(t).a,c).shaft_w for t in (0.,2.,4.)]
    threshold=(min(sample)+max(sample))/2
    total=en._segment_peak(s,0.,4.,threshold,c)
    parts=en._segment_peak(s,0.,1.37,threshold,c)+en._segment_peak(s,1.37,4.,threshold,c)
    assert 0<total<4 and parts==pytest.approx(total,abs=1e-12)


def test_stable_power_reuses_phase_and_duration_independent_value():
    en._stable_power_cached.cache_clear(); en.reset_practical_numeric_stats()
    p=tr.make_stable_plan('a','outer',3,0.,100.)
    a=en.integrate_energy(p,0.,7.1)
    b=en.integrate_energy(p,7.1,100.)
    power=en.stable_power('outer')
    assert a.valid and b.valid
    assert a.joules+b.joules==pytest.approx(power.battery_w*100,rel=1e-12)
    assert en.practical_numeric_stats()['stable_power_computes']==1
