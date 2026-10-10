"""Exact-input reuse must not merge nearby curves or weaken analytic limits."""
import numpy as np
import pytest

from terminal_marl.core import polynomial_trajectory as polynomial


@pytest.fixture(autouse=True)
def clear_reuse():
    polynomial._BOUNDS_REUSE_CACHE.clear()
    yield
    polynomial._BOUNDS_REUSE_CACHE.clear()


def segment():
    return polynomial.QuinticSegment.from_boundary(
        0.,4.,(0.,0.,30.),(2.,1.,0.),(.1,0.,0.),
        (10.,6.,32.),(1.,2.,0.),(0.,.1,0.))


def test_equal_coefficients_and_duration_reuse_independent_of_absolute_time(monkeypatch):
    original=segment()
    expected=polynomial.prime_quintic_bounds([original])[0]
    def forbidden(*args):
        raise AssertionError('identical bounds recomputed')
    monkeypatch.setattr(polynomial,'_batch_real_roots_in_unit_interval',forbidden)
    shifted=polynomial.QuinticSegment(100.,104.,original.coefficients)
    actual=polynomial.prime_quintic_bounds([shifted])[0]
    assert actual==expected
    actual['max_horizontal_speed']=-100
    assert shifted.bounds()==expected  # Public dict cannot mutate shared values.


@pytest.mark.parametrize('change',['duration','coefficient','shape'])
def test_distinct_exact_inputs_cannot_hit_cached_bounds(change,monkeypatch):
    original=segment()
    polynomial.prime_quintic_bounds([original])
    coefficients=original.coefficients.copy()
    end=4.
    if change=='duration':
        end=np.nextafter(end,np.inf)
    elif change=='coefficient':
        coefficients[2,0]=np.nextafter(coefficients[2,0],np.inf)
    changed=polynomial.QuinticSegment(0.,end,coefficients)
    if change=='shape':
        object.__setattr__(changed,'coefficients',changed.coefficients.reshape(18))
    def expected_miss(*args):
        raise RuntimeError('fresh analytic evaluation')
    monkeypatch.setattr(polynomial,'_derivative_coefficients',expected_miss)
    with pytest.raises(RuntimeError,match='fresh analytic evaluation'):
        polynomial.prime_quintic_bounds([changed])


def test_bounded_eviction_changes_cost_only(monkeypatch):
    monkeypatch.setattr(polynomial,'_BOUNDS_REUSE_LIMIT',2)
    original=segment()
    expected=polynomial.prime_quintic_bounds([original])[0]
    for shift in (1.,2.,3.):
        coefficients=original.coefficients.copy();coefficients[0,0]+=shift
        polynomial.prime_quintic_bounds([polynomial.QuinticSegment(0.,4.,coefficients)])
    assert len(polynomial._BOUNDS_REUSE_CACHE)==2
    fresh=polynomial.QuinticSegment(0.,4.,original.coefficients)
    assert polynomial.prime_quintic_bounds([fresh])[0]==expected


def test_cached_and_uncached_batches_match_exactly(monkeypatch):
    rng=np.random.default_rng(892903)
    curves=[rng.normal(size=(6,3)) for _ in range(20)]
    durations=[float(x) for x in rng.uniform(1.,30.,20)]
    def batch():
        return [polynomial.QuinticSegment(0.,d,c) for c,d in zip(curves,durations)]
    with monkeypatch.context() as no_reuse:
        no_reuse.setattr(polynomial,'_BOUNDS_REUSE_LIMIT',0)
        expected=polynomial.prime_quintic_bounds(batch())
    assert not polynomial._BOUNDS_REUSE_CACHE
    assert polynomial.prime_quintic_bounds(batch())==expected
    assert polynomial.prime_quintic_bounds(batch())==expected
