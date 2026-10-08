from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.configs.p3_v25_course import build_config
from terminal_marl.core.config import SimConfig
from terminal_marl.core.priority import PriorityEngine
from terminal_marl.core.safety_filter import SafetyFilter


def pair(distance=84.408036, duration=.2):
    ring = SimpleNamespace(radius=600., height=100.)
    times = np.array([0., duration])
    theta = times*22./ring.radius
    gap = 2*np.arcsin(distance/(2*ring.radius))
    left = np.column_stack((ring.radius*np.cos(theta-gap),
                            ring.radius*np.sin(theta-gap), np.full(2,100.)))
    return ring,times,theta,left


def test_nearby_nominal_holders_have_a_continuous_certificate_without_speed_envelope():
    config=build_config('dense24_fixed');safety=SafetyFilter(config)
    ring,times,theta,left=pair()
    bound=safety._tube_distances_with_margin(left,ring,theta,theta,times,'flight_v3')
    old=SafetyFilter(replace(config,nominal_holder_separation_mode='speed_tube_legacy'))
    legacy=old._tube_distances_with_margin(left,ring,theta,theta,times,'flight_v3')
    assert bound.min()>84.3 and legacy.min()<80.
    # Actual circular separation is constant. The bound must remain below it.
    assert bound[-1] < 84.408036


def test_swept_check_rejects_crossing_despite_safe_endpoints():
    safety=SafetyFilter(build_config('dense24_fixed'))
    ring=SimpleNamespace(radius=600.,height=100.)
    times=np.array([0.,10.]);theta=np.zeros(2)
    left=np.array([[600.,-100.,100.],[600.,100.,100.]])
    distances=safety._tube_distances_with_margin(left,ring,theta,theta,times,'flight_v3')
    assert min(distances[:2])==100. and distances[-1]==0.


def test_curve_margin_rejects_possible_between_sample_excursion():
    safety=SafetyFilter(build_config('dense24_fixed'))
    ring=SimpleNamespace(radius=200.,height=40.)
    values=safety._tube_distances_with_margin(np.array([[280.1,0.,40.]]*2),ring,
        np.zeros(2),np.zeros(2),np.array([0.,1.]),'flight_v3')
    assert values[0]>80. and values[-1]<80.


def test_nonzero_tube_keeps_original_envelope_and_unknown_curve_fails_closed():
    config=build_config('dense24_fixed');safety=SafetyFilter(config)
    old=SafetyFilter(replace(config,nominal_holder_separation_mode='speed_tube_legacy'))
    ring,times,theta,left=pair()
    np.testing.assert_array_equal(
        safety._tube_distances_with_margin(left,ring,theta,theta+.001,times,'flight_v3'),
        old._tube_distances_with_margin(left,ring,theta,theta+.001,times,'flight_v3'))
    assert np.isneginf(safety._tube_distances_with_margin(left,ring,theta,theta,times,None)[:2]).all()


def test_bounded_curves_certify_below_dense_distance():
    safety=SafetyFilter(build_config('dense24_fixed'))
    ring=SimpleNamespace(radius=600.,height=100.)
    rng=np.random.default_rng(71)
    for _ in range(100):
        duration=rng.uniform(.05,1.)
        initial=rng.uniform(-np.pi,np.pi);gap=rng.uniform(-.3,.3)
        speed=rng.uniform(8.,22.);other_speed=rng.uniform(8.,22.)
        radius=rng.uniform(400.,600.)
        times=np.array([0.,duration]);theta=initial+times*other_speed/ring.radius
        angle=initial+gap+times*speed/radius
        left=np.column_stack((radius*np.cos(angle),radius*np.sin(angle),np.full(2,100.)))
        bound=safety._tube_distances_with_margin(left,ring,theta,theta,times,'flight_v3')[-1]
        fine=np.linspace(0.,duration,1001)
        a=initial+gap+fine*speed/radius;b=initial+fine*other_speed/ring.radius
        distances=np.hypot(radius*np.cos(a)-ring.radius*np.cos(b),radius*np.sin(a)-ring.radius*np.sin(b))
        assert bound<=distances.min()+1e-10


def test_total_wait_retains_queue_and_soc_terms_without_random_task_priority():
    config=build_config('dense24_fixed');engine=PriorityEngine(config)
    u=SimpleNamespace(energy_capacity=1500.,energy_remaining=1200.,base_priority=5,
                      external_wait_time=0.,internal_wait_time=300.)
    expected=1+.5*(300./3600.)**1.5
    assert engine.compute(u)==pytest.approx(expected)
    q=SimpleNamespace(request=SimpleNamespace(energy_capacity=1500.,base_priority=1),
                      energy_remaining=1200.,external_wait_time=300.)
    assert engine.compute(q)==pytest.approx(expected)
    u.energy_remaining=300.
    assert engine.compute(u)>expected
    legacy=PriorityEngine(replace(config,priority_wait_mode='external_only_legacy'))
    u.energy_remaining=1200.
    assert legacy.compute(u)==1.


def test_new_modes_explicit_and_old_pickles_keep_class_defaults():
    old=SimConfig()
    assert old.priority_wait_mode=='external_only_legacy'
    assert old.nominal_holder_separation_mode=='speed_tube_legacy'
    del old.priority_wait_mode;del old.nominal_holder_separation_mode
    assert old.priority_wait_mode=='external_only_legacy'
    assert old.nominal_holder_separation_mode=='speed_tube_legacy'
    for field in ('priority_wait_mode','nominal_holder_separation_mode'):
        with pytest.raises(ValueError,match='unsupported'):
            replace(build_config('dense24_fixed'),**{field:'invalid'})
    with pytest.raises(ValueError,match='requires nominal V3'):
        SimConfig(nominal_holder_separation_mode='swept_curve_v1')
