import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from terminal_marl.core.polynomial_trajectory import QuinticSegment
from terminal_marl.core.trajectory_generator import TrajectoryGenerator
from terminal_marl.configs.p3_v25_course import build_config

spec=importlib.util.spec_from_file_location('hull_candidate',Path(__file__).with_name('p3_ring_hold_hull_candidate.py'))
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


def test_acceptance_implies_exact_bounds_for_random_polynomials_without_caching():
    rng=np.random.default_rng(6201);accepted=0
    for index in range(300):
        coefficients=rng.normal(size=(6,3))*10**rng.uniform(-3,2)
        if index%2==0:coefficients[2:]*=.001
        segment=QuinticSegment(0.,10**rng.uniform(-2,2),coefficients)
        exact=segment.bounds()
        segment=QuinticSegment(segment.start_time,segment.end_time,coefficients)
        limits=[exact['max_horizontal_speed']*1.01,exact['max_horizontal_accel']*1.01]
        if candidate.horizontal_hull_certifies((segment,),*limits):
            accepted+=1
            assert exact['max_horizontal_speed']<=limits[0]
            assert exact['max_horizontal_accel']<=limits[1]
        assert segment._bounds_cache is None
        assert not candidate.horizontal_hull_certifies((segment,),exact['max_horizontal_speed']*.99,limits[1])
        assert not candidate.horizontal_hull_certifies((segment,),limits[0],exact['max_horizontal_accel']*.99)
    assert accepted>50


def test_boundary_outward_margin_and_interior_peak_are_not_accepted():
    c=np.zeros((6,3));c[1,0]=2.
    constant=QuinticSegment(0.,1.,c)
    assert not candidate.horizontal_hull_certifies((constant,),2.,1.)
    assert candidate.horizontal_hull_certifies((constant,),2.0001,1.)
    overshoot=QuinticSegment.from_boundary(0.,1.,[0,0,0],[0,0,0],[0,0,0],
                                         [10,0,0],[0,0,0],[0,0,0])
    assert overshoot.bounds()['max_horizontal_speed']>18.
    assert not candidate.horizontal_hull_certifies((overshoot,),10.,1000.)


def test_invalid_hull_falls_back_instead_of_certifying():
    assert not candidate.horizontal_hull_certifies((),1.,1.)
    c=np.ones((6,3))*1e200
    assert not candidate.horizontal_hull_certifies((QuinticSegment(0.,1e-100,c),),1.,1.)


def test_all_coordinate_hulls_enclose_analytic_extrema_and_preserve_decisions():
    generator=TrajectoryGenerator(build_config('dense24_fixed'))
    rng=np.random.default_rng(6202)
    for _ in range(150):
        duration=10**rng.uniform(-1,2)
        coefficients=rng.normal(size=(6,3))*10**rng.uniform(-3,1)
        coefficients[0,2]=50.
        segment=QuinticSegment(0.,duration,coefficients)
        conservative=candidate.hull_bounds((segment,))[0];exact=segment.bounds()
        for key,bound in conservative.items():
            assert bound<=exact[key] if key=='min_altitude' else bound>=exact[key]
        expected=generator.parameterizer._v3_segments_feasible((segment,))
        clean=QuinticSegment(0.,duration,coefficients)
        assert candidate.segments_feasible(generator.parameterizer,(clean,))==expected


@pytest.mark.parametrize('radius,height,speed',[(320.,60.,12.),(520.,100.,20.)])
@pytest.mark.parametrize('target_fraction',[1.,.7])
def test_actual_circular_hold_points_are_exactly_unchanged(radius,height,speed,target_fraction):
    generator=TrajectoryGenerator(build_config('dense24_fixed'))
    ring=SimpleNamespace(radius=radius,height=height,nominal_speed=speed)
    theta=.4;target=speed*target_fraction
    acceleration=-speed**2/radius*np.asarray([np.cos(theta),np.sin(theta),0.])
    required=generator._v3_required_ring_braking_duration(ring,theta,speed,target,initial_acceleration=acceleration)
    for extra in (.5,3.,37.):
        args=(ring,theta,speed,target,10.,required+extra)
        original=generator._v3_ring_hold_points(*args,initial_acceleration=acceleration)
        repaired=candidate.ring_hold_points(generator,*args,initial_acceleration=acceleration)
        assert len(original)==len(repaired)
        for a,b in zip(original,repaired):
            assert a.t==b.t
            for name in ('position','velocity','acceleration'):
                assert np.array_equal(getattr(a,name),getattr(b,name))
