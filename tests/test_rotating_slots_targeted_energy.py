"""Plan-level numerical budgets; allowances are not rigorous quadrature proofs."""
from dataclasses import replace
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from terminal_marl.core.rotating_slots import energy as en, trajectory as tr
from terminal_marl.core.rotating_slots.config import freeze, load_config


class SegmentsOnly:
    def __init__(self, segments): self.segments=tuple(segments)
    def intervals(self,start,end):
        return (s for s in self.segments if s.start_time<end and s.end_time>start)


def custom_config(**energy):
    c=load_config(); data=c.to_dict(); data['energy'].update(energy)
    # Deliberately retain old declared hashes: numerical cache identity must
    # retain the complete immutable config rather than trusting only a digest.
    return replace(c,sections=freeze(data))


_spec=importlib.util.spec_from_file_location('targeted_scalar_reference',
    Path(__file__).resolve().parents[1]/'new10.16'/'energy_reference_v1_0.py')
_reference=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_reference)


def strict_joules(plan,start,end):
    result=0.
    for segment in plan.intervals(start,end):
        a=max(start,segment.start_time); b=min(end,segment.end_time)
        def power(t):
            state=segment.state(t)
            return _reference.power(state.v,state.a)['battery_w']
        result+=quad(power,a,b,epsabs=1e-5,epsrel=2e-12,limit=100)[0]
    return result


def test_global_energy_budget_shared_prefixes_and_precision_identity():
    p=tr.generate_final('a','inner',0,0,0.).plan
    whole=en.integrate_energy(p,0.,p.end_time)
    assert whole.valid and 0<whole.error_j<=max(1.,1e-6*whole.joules)
    shapes=tuple(en._shape(s) for s in p.segments)
    policy=en._plan_numerical_policy(shapes,1.,en._ConfigIdentity(load_config()),False)
    assert policy[2]<=max(1.,1e-6*whole.joules)  # summed actual Gauss4/8 estimates
    cuts=sorted({0.,.031,.137,.371,1.001,p.segments[0].end_time+.371,p.end_time-.137,p.end_time})
    pieces=[en.integrate_energy(p,a,b) for a,b in zip(cuts,cuts[1:])]
    assert all(q.valid for q in pieces)
    assert math.fsum(q.joules for q in pieces)==pytest.approx(whole.joules,abs=1e-7)
    assert math.fsum(q.error_j for q in pieces)==pytest.approx(whole.error_j,abs=1e-12)
    assert math.fsum(q.peak_error_s for q in pieces)==pytest.approx(whole.peak_error_s,abs=1e-12)
    assert math.fsum(q.peak_seconds for q in pieces)==pytest.approx(whole.peak_seconds,abs=1e-12)
    fine=en.integrate_energy(p,0.,p.end_time,fine=True)
    assert fine.valid and fine.error_j<whole.error_j/1000
    assert fine.peak_error_s<=1e-7
    assert abs(whole.joules-fine.joules)<=whole.error_j+fine.error_j
    assert whole.peak_seconds>=fine.peak_seconds-fine.peak_error_s
    en.reset_practical_numeric_stats()
    assert en.integrate_energy(p,0.,p.end_time)==whole
    assert en.integrate_energy(p,0.,p.end_time,fine=True)==fine
    assert en.practical_numeric_stats().get('energy_table_computes',0)==0


@pytest.mark.parametrize('kind',('transfer','final'))
def test_sampled_whole_prefix_and_short_queries_against_strict_reference(kind):
    p=(tr.generate_transfer('a','outer',0,0,0.).plan if kind=='transfer' else
       tr.generate_final('a','inner',0,0,0.).plan)
    end=p.end_time
    intervals=((0.,end),(0.,.137),(.137,.168),(.371,1.137),(end-.137,end))
    for a,b in intervals:
        standard=en.integrate_energy(p,a,b)
        fine=en.integrate_energy(p,a,b,fine=True)
        reference=strict_joules(p,a,b)
        assert standard.valid and fine.valid
        assert abs(standard.joules-reference)<=max(standard.error_j,1e-5)
        assert abs(fine.joules-reference)<=max(fine.error_j,1e-5)
        assert abs(standard.joules-fine.joules)<=max(standard.error_j+fine.error_j,1e-5)


def test_multisegment_peak_upper_measure_and_additive_uncertainty():
    c=load_config(); s=tr.descent_segments(0,0.)[0]
    samples=[en.evaluate_power(s.state(t).v,s.state(t).a,c).shaft_w for t in (0.,2.,4.)]
    threshold=(min(samples)+max(samples))/2
    c=custom_config(reference_motor_continuous_power_kW_each=threshold/16/1000)
    p=SegmentsOnly((s,replace(s,start_time=4.)))
    whole=en.integrate_energy(p,0.,8.,config=c)
    fine=en.integrate_energy(p,0.,8.,config=c,fine=True)
    assert whole.valid and fine.valid
    assert 0<whole.peak_error_s<=.01 and fine.peak_error_s<=1e-7
    assert whole.peak_seconds>=fine.peak_seconds-fine.peak_error_s
    assert whole.peak_seconds-fine.peak_seconds<=whole.peak_error_s+fine.peak_error_s
    pieces=[en.integrate_energy(p,a,b,config=c) for a,b in zip((0.,.371,3.99,4.137),( .371,3.99,4.137,8.))]
    assert math.fsum(q.peak_seconds for q in pieces)==pytest.approx(whole.peak_seconds,abs=1e-12)
    assert math.fsum(q.peak_error_s for q in pieces)==pytest.approx(whole.peak_error_s,abs=1e-12)
    assert math.fsum(q.error_j for q in pieces)==pytest.approx(whole.error_j,abs=1e-12)


def test_configured_root_residual_signed_enclosure_and_fine_keys():
    en.root_cache_clear(); en.reset_practical_numeric_stats()
    for vp,vn,rhs in ((0.,0.,20.),(2.,-2.,4.),(10.,-1.,80.),(3.,2.,40.)):
        tol=load_config().energy.induced_root_relative_residual_tolerance
        root=en._root(vp,vn,rhs,tol=tol)
        assert root is not None
        assert abs(root*math.hypot(vp,vn+root)-rhs)<=tol*max(1.,rhs)
        lo,hi=en._root_bracket(vp,vn,rhs,tol=tol)
        strict=brentq(lambda x:x*math.hypot(vp,vn+x)-rhs,max(0.,-vn),100.,xtol=1e-14)
        assert max(0.,-vn)<=lo<=strict<=hi
        before=en.root_cache_stats()['actual_computes']
        finer=en._root(vp,vn,rhs,tol=1e-12)
        assert en.root_cache_stats()['actual_computes']==before+1
        assert abs(finer*math.hypot(vp,vn+finer)-rhs)<=1e-12*max(1.,rhs)


def test_near_power_limit_refines_without_accepting_known_excess():
    v,a=(21.,0.,0.),(0.,0.,0.)
    current=_reference.power(v,a)['current_a']
    above=custom_config(reference_max_battery_current_A=current*(1+1e-10))
    below=custom_config(reference_max_battery_current_A=current*(1-1e-10))
    en.reset_practical_numeric_stats()
    accepted=en.evaluate_power(v,a,above)
    rejected=en.evaluate_power(v,a,below)
    assert accepted.valid and rejected.status=='POWER_LIMIT'
    assert en.practical_numeric_stats()['power_limit_root_refinements']>=2
    s=tr.Segment(0.,1.,(100.,21.),(0.,),(30.,))
    assert np.isfinite(en._power_array(s,np.array([.1,.7]),above)).all()
    with pytest.raises(ValueError):
        en._power_array(s,np.array([.1,.7]),below)
    p=SegmentsOnly((s,))
    assert en.integrate_energy(p,0.,1.,max_dt=1.,config=above).valid
    assert not en.integrate_energy(p,0.,1.,max_dt=1.,config=below).valid


def test_tables_retain_complete_custom_config_and_explicit_grid_and_fine():
    p=SegmentsOnly(tr.descent_segments(0,0.))
    c=load_config(); different=custom_config(motor_drive_efficiency=.85)
    a=en.integrate_energy(p,0.,13.5,config=c)
    b=en.integrate_energy(p,0.,13.5,config=different)
    assert a.valid and b.valid and a.joules!=b.joules
    assert b.joules/a.joules==pytest.approx(.9/.85,rel=1e-10)
    fine=en.integrate_energy(p,0.,13.5,config=c,fine=True)
    fine_split=en.integrate_energy(p,0.,.371,config=c,fine=True).joules+en.integrate_energy(p,.371,13.5,config=c,fine=True).joules
    assert fine_split==pytest.approx(fine.joules,abs=1e-8)
    with pytest.raises(ValueError): en.integrate_energy(p,0.,1.,max_dt=1.01)
    assert en.integrate_energy(p,0.,1.,max_dt=.2).valid
