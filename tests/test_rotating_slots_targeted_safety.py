"""Targeted motion witnesses and certified short-step distance screens."""
from dataclasses import replace
import math
from unittest.mock import Mock

import pytest
from scipy.optimize import brentq

from terminal_marl.core.rotating_slots import energy as en, safety as sf, trajectory as tr
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.models import ValidationResult


def transfer_with_duration(duration):
    c=load_config(); w=c.airspace.common_angular_speed_radps
    s=tr.Segment(0.,duration,tr.hermite(600.,0.,0.,450.,0.,0.,duration),
                 (0.,-w*duration),tr.hermite(110.,0.,0.,70.,0.,0.,duration))
    return tr._candidate('bad','TRANSFER',0.,'outer',0,'middle',0,None,0,(s,),c)


def test_clear_motion_witness_precedes_all_power_work(monkeypatch):
    plan=transfer_with_duration(1.)
    bounds=Mock(side_effect=AssertionError('unexpected power bounds'))
    point=Mock(side_effect=AssertionError('unexpected point power'))
    monkeypatch.setattr(sf,'power_bounds',bounds); monkeypatch.setattr(sf,'evaluate_power',point)
    en.reset_practical_numeric_stats()
    result=sf.validate_plan(plan,use_cache=False)
    assert result.status=='UNSAFE' and result.reason in ('HORIZONTAL_SPEED','HORIZONTAL_ACCELERATION','HORIZONTAL_JERK')
    assert bounds.call_count==point.call_count==0
    assert en.practical_numeric_stats()['kinematic_witness_rejects']==1
    assert en.practical_numeric_stats()['early_check_calls']>=1


def test_metadata_invalid_precedes_motion_witness(monkeypatch):
    plan=replace(transfer_with_duration(1.),source_slot=1,trajectory_hash='')
    witness=Mock(side_effect=AssertionError('metadata must be checked first'))
    monkeypatch.setattr(sf,'_kinematic_witness',witness)
    result=sf.validate_plan(plan,use_cache=False)
    assert result.status=='INVALID' and result.reason=='SOURCE_STATE_MISMATCH'
    assert witness.call_count==0


def test_clean_samples_do_not_replace_unresolved_interval_certificate(monkeypatch):
    plan=tr.generate_transfer('valid','outer',0,0,0.).plan
    segment=plan.segments[0]
    assert sf._kinematic_witness(plan,segment,tuple(segment.duration*x for x in (0.,.25,.5,.75,1.))) is None
    # A wide/unresolved power enclosure may not be treated as a point witness.
    bounds=Mock(return_value=None); monkeypatch.setattr(sf,'power_bounds',bounds)
    result=sf.validate_plan(plan,use_cache=False)
    assert not result.accepted and result.status=='UNKNOWN'
    assert bounds.call_count>1


def test_verified_far_pair_screen_uses_no_trajectory_bounds(monkeypatch):
    moving=tr.generate_transfer('moving','outer',0,0,0.).plan
    stable=tr.make_stable_plan('stable','outer',18)
    sf._EXECUTION_SPEED_CERTIFICATES.clear(); en.reset_practical_numeric_stats()
    forbidden=Mock(side_effect=AssertionError('screen must not call trajectory bounds'))
    monkeypatch.setattr(tr.TrajectoryPlan,'bounds',forbidden)
    results=sf._execution_distance_screen({'moving':moving,'stable':stable},0.,.2)
    assert set(results)=={('moving','stable')}
    result=results[('moving','stable')]
    actual=math.dist(moving.state(0.).p,stable.state(0.).p)
    assert result.accepted and 80<result.details['lower_bound_m']<actual
    assert result.details['upper_bound_m']>=actual and result.details['witness_time']==0.
    assert forbidden.call_count==0
    computes=en.practical_numeric_stats()['execution_screen_validation_computes']
    sf._execution_distance_screen({'moving':moving,'stable':stable},.2,.4)
    assert en.practical_numeric_stats()['execution_screen_validation_computes']==computes


def test_missing_successful_certificate_does_not_screen(monkeypatch):
    moving=tr.generate_transfer('moving','outer',0,0,0.).plan
    stable=tr.make_stable_plan('stable','outer',18)
    sf._EXECUTION_SPEED_CERTIFICATES.clear()
    validation=Mock(return_value=ValidationResult(False,'UNKNOWN','unverified'))
    monkeypatch.setattr(sf,'validate_plan',validation)
    assert sf._execution_distance_screen({'moving':moving,'stable':stable},0.,.2)=={}
    assert validation.call_count==2


def test_near_boundary_and_future_interval_fall_back_to_continuous_check():
    final=tr.generate_final('final','inner',0,0,0.).plan
    ground=tr.make_ground_plan('ground',0)
    end=final.segments[0].end_time
    start=brentq(lambda t:math.dist(final.state(t).p,ground.state(t).p)-80.,0.,end)
    assert sf._execution_distance_screen({'final':final,'ground':ground},start,start+.2)=={}
    result=sf.validate_separation(final,ground,start,start+.2)
    assert not result.accepted and result.status=='UNSAFE'
    assert sf._execution_distance_screen({'final':final,'ground':ground},0.,end)=={}
    assert not sf.validate_separation(final,ground,0.,final.end_time).accepted
    assert sf._execution_distance_screen({'final':final,'ground':ground},final.end_time+.1,final.end_time+.2)=={}
    assert sf._execution_distance_screen({'final':final,'ground':ground},math.nan,.2)=={}


def test_stable_pairs_retain_analytic_checker():
    a=tr.make_stable_plan('a','outer',0); b=tr.make_stable_plan('b','outer',1)
    assert sf._execution_distance_screen({'a':a,'b':b},0.,.2)=={}
    result=sf.validate_separation(a,b,0.,.2)
    assert result.accepted and result.details['method']=='common_rotation_structure_with_roundoff_enclosure'
