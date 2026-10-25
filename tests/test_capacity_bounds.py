import math

import numpy as np
import pytest

from terminal_marl.analysis.capacity_bounds import (
    finite_cohort_pad_window_upper, kinematic_touchdown_lower_bound,
    loss_adjusted_pad_upper, mandatory_pad_overlap, pad_window_lp_upper,
)
from terminal_marl.core.rotating_slots.capacity_config import make_capacity_config
from terminal_marl.core.rotating_slots.config import load_service_profile
from terminal_marl.core.rotating_slots.trajectory import descent_segments


@pytest.mark.parametrize("durations,pads,expected", [
    ((30., 60., 120.), 6, 271.6981132075472),
    ((60., 120., 240.), 6, 148.4536082474227),
    ((30., 60., 120.), 3, 135.8490566037736),
])
def test_zero_loss_fixed_mix(durations, pads, expected):
    result = loss_adjusted_pad_upper(durations, (.2, .6, .2), pads)
    assert result["offered_rate_upper_per_h"] == pytest.approx(expected)
    assert result["selected_mass"] == pytest.approx([.2, .6, .2])


def test_loss_selects_longest_jobs_to_drop():
    result = loss_adjusted_pad_upper((120., 30., 60.), (.2, .2, .6), epsilon=.05)
    assert result["selected_mass"] == pytest.approx([.15, .2, .6])
    assert result["minimum_work_per_offered_job_s"] == pytest.approx(72.825)
    assert result["offered_rate_upper_per_h"] == pytest.approx(21600 / 72.825)
    # Cross a class breakpoint: all long jobs and part of standard are dropped.
    result = loss_adjusted_pad_upper((30., 60., 120.), (.2, .6, .2), epsilon=.3)
    assert result["selected_mass"] == pytest.approx([.2, .5, 0.])


def test_equal_mean_variance_does_not_change_all_served_bound():
    rates = [loss_adjusted_pad_upper(s, p)["offered_rate_upper_per_h"] for s, p in (
        ((66.,), (1.,)), ((30., 60., 120.), (.2, .6, .2)), ((30., 120.), (.6, .4)))]
    assert rates == pytest.approx([21600 / 79.5] * 3)


@pytest.mark.parametrize("epsilon", [-.01, 1., math.nan, math.inf])
def test_invalid_loss(epsilon):
    with pytest.raises(ValueError):
        loss_adjusted_pad_upper((30.,), (1.,), epsilon=epsilon)


def test_zero_selected_work_is_rejected():
    with pytest.raises(ValueError, match="positive pad work"):
        loss_adjusted_pad_upper((0.,), (1.,), protected_s=0.)


def test_kinematic_baseline_and_capture_match_trajectory():
    config = load_service_profile("service_half_v1")
    result = kinematic_touchdown_lower_bound(config)
    assert [result[key] for key in ("entry_s", "outer_to_middle_s", "middle_to_inner_s",
                                    "final_predescent_s", "protected_descent_s")] == [12., 20., 20., 8., 13.5]
    assert result["touchdown_lower_s"] == 73.5
    assert result["capture_height_m"] == 15.
    assert descent_segments(0, 0., config)[0].z[0] == result["capture_height_m"]


@pytest.mark.parametrize("radius,lower,final", [(250., 73., 7.5), (300., 73.5, 8.), (350., 75.5, 10.)])
def test_kinematic_factory_geometries(radius, lower, final):
    result = kinematic_touchdown_lower_bound(make_capacity_config(inner_radius_m=radius))
    assert result["touchdown_lower_s"] == lower
    assert result["final_predescent_s"] == final
    assert result["inner_radius_m"] == radius


def test_tight_two_job_pad_window_and_dual():
    result = pad_window_lp_upper([0., 0.], [10., 10.], [10., 10.], pads=1,
                                 intervals=[(0., 10.)])
    assert result["lp_objective"] == pytest.approx(1.)
    assert result["relaxed_count_upper"] == pytest.approx(1., abs=1e-7)
    assert result["integer_count_upper"] == 1
    assert result["solver_status"] == 0
    assert result["dual_feasibility_residual"] <= 1e-12
    alpha, z = np.asarray(result["dual_alpha"]), np.asarray(result["dual_z"])
    assert np.all(10. * alpha[0] + z >= 1.)
    assert result["dual_certificate_bound"] == pytest.approx(10 * alpha[0] + z.sum())
    assert result["relaxed_count_upper"] >= result["lp_objective"]


def test_enough_pad_capacity():
    result = pad_window_lp_upper([0., 0.], [10., 10.], [10., 10.], pads=2,
                                 intervals=[(0., 10.)])
    assert result["integer_count_upper"] == 2


def test_sla_impossible_and_touchdown_deadline_mapping():
    assert finite_cohort_pad_window_upper([0., 10.], [30., 60.], sla_s=73.)["integer_count_upper"] == 0
    # At D=L all mandatory work must occupy the exact [60,103.5] pad window.
    result = finite_cohort_pad_window_upper([0., 0.], [30., 30.], sla_s=73.5,
                                            pads=1, intervals=[(60., 103.5)])
    assert result["integer_count_upper"] == 1
    assert result["eligible_jobs"] == 2


def test_mandatory_overlap_matches_deadline_energetic_calculation():
    r, d, b = [0., 2., 4.], [10., 12., 15.], [6., 8., 10.]
    observed = mandatory_pad_overlap(r, d, b, 3., 11.)
    expected = [max(0., work - max(0., 3. - start) - max(0., finish - 11.))
                for start, finish, work in zip(r, d, b)]
    assert observed == pytest.approx(expected)
    assert mandatory_pad_overlap([0.], [10.], [10.], 0., 10.)[0] == 10.
    with pytest.raises(ValueError):
        mandatory_pad_overlap([0.], [9.], [10.], 0., 10.)


def test_interval_subset_is_valid_but_can_be_looser():
    tight = pad_window_lp_upper([0., 0.], [10., 10.], [10., 10.], pads=1,
                                intervals=[(0., 10.), (20., 30.)])
    loose = pad_window_lp_upper([0., 0.], [10., 10.], [10., 10.], pads=1,
                                intervals=[(20., 30.)])
    assert tight["integer_count_upper"] == 1
    assert loose["integer_count_upper"] == 2
    assert loose["relaxed_count_upper"] >= tight["relaxed_count_upper"]


def test_default_grid_includes_fractional_extremes_and_empty_jobs():
    result = pad_window_lp_upper([.25, .25], [10.75, 10.75], [10.5, 10.5], pads=1)
    assert result["interval_count"] == 1
    assert result["integer_count_upper"] == 1
    assert pad_window_lp_upper([], [], [], pads=1)["integer_count_upper"] == 0
    assert pad_window_lp_upper([0.], [9.], [10.], pads=1)["eligible_jobs"] == 0
