from dataclasses import replace

from terminal_marl.core.config import RingRole
from terminal_marl.core.ring_capacity import geometric_ring_capacity
from terminal_marl.training.a611_causal_audit import DemandProfile
from terminal_marl.training.a613_capacity_reaudit import repair_config
from terminal_marl.training.a614_reconfiguration_timing_audit import (
    BURST_RELATIVE_SECONDS,
    PPO_AUTHORIZATION,
    TIMING_SWEEP_SECONDS,
    classify_audit,
    run_dynamic_at_time,
    select_mu0,
)


def _row(time, beneficial):
    return {"condition": f"dynamic_outer_plus1_t{int(time)}", "action_time": float(time), "beneficial_vs_nominal": beneficial}


def test_a614_timing_sweep_is_explicit_and_crn_config_is_identical():
    assert TIMING_SWEEP_SECONDS == (0., 30., 60., 90., 120., 180., 240., 300.)
    assert BURST_RELATIVE_SECONDS == (-180., -120., -60., 0., 60., 180.)
    profile = DemandProfile("crn", ((0., 20., .1),), 20., 1.)
    assert repair_config(profile, dynamic=True).fixed_arrival_schedule == repair_config(profile, dynamic=False).fixed_arrival_schedule


def test_a614_t0_dynamic_reaches_static_outer_capacity_after_transition():
    profile = DemandProfile("t0", ((0., 5., .1),), 5., 20.)
    dynamic = run_dynamic_at_time(repair_config(profile, dynamic=True), 0, 0.)
    static = repair_config(profile, dynamic=False, role=RingRole.OUTER)
    expected = geometric_ring_capacity(static.rings[RingRole.OUTER].radius, static.safe_distance)
    assert dynamic["actual_capacity"]["outer"] == expected == 48
    assert dynamic["timing"]["geometry_effective_time"] is not None


def test_a614_effective_time_and_capacity_use_are_logged():
    profile = DemandProfile("logging", ((0., 5., .1),), 5., 20.)
    row = run_dynamic_at_time(repair_config(profile, dynamic=True), 0, 0.)
    timing = row["timing"]
    assert timing["geometry_requested_time"] == 0.
    assert timing["geometry_applied_time"] == 0.
    assert timing["geometry_effective_time"] >= 20.
    # This very small demand never consumes the 48th slot, which is exactly
    # distinguishable from capacity becoming effective.
    assert timing["first_extra_capacity_used_time"] is None


def test_a614_optional_pre_action_observer_is_29d_and_cannot_change_rollout():
    profile = DemandProfile("observer", ((0., 5., .1),), 5., 20.)
    config = repair_config(profile, dynamic=True)
    canonical = run_dynamic_at_time(config, 0, 0.)
    observed = run_dynamic_at_time(config, 0, 0., observation_capture_time=0.)
    assert len(observed["pre_action_observation"]["observation_29d"]) == 29
    assert observed["metrics"] == canonical["metrics"]


def test_a614_hard_safety_is_unchanged_in_timing_counterfactual():
    profile = DemandProfile("safety", ((0., 5., .1),), 5., 20.)
    row = run_dynamic_at_time(repair_config(profile, dynamic=True), 0, 0.)
    metrics = row["metrics"]
    assert row["safe"]
    assert metrics["collisions"] == metrics["nmac"] == metrics["hard_separation_violations"] == 0
    assert metrics["final_overcapacity_steps"] == 0


def test_a614_warm_start_diagnostic_does_not_require_external_request_ledger_rows():
    profile = DemandProfile("warm_start", ((-20., 0., .2), (0., 5., .1)), 5., 20.)
    config = replace(repair_config(profile, dynamic=True), reset_mode="warm_start_rings", warm_start_uavs=3)
    row = run_dynamic_at_time(config, 0, float("inf"), force_initial_gate=False)
    assert row["safe"] and row["metrics"]["admission_delay_mean"] >= 0.


def test_a614_classification_is_deterministic_and_cannot_authorize_ppo():
    timing = [_row(0, True), _row(60, True), _row(180, False)]
    burst = [_row(120, True)]
    first = classify_audit(timing, burst, static_beneficial=True, materiality_fallback=False)
    assert first == classify_audit(timing, burst, static_beneficial=True, materiality_fallback=False)
    assert first["classification"] == "DECISION_INTERVAL_TOO_SLOW"
    assert first["ppo_authorization"] == PPO_AUTHORIZATION == "STOP"


def test_a614_explicit_mu0_reuses_the_calibration_without_changing_crn_seed():
    assert select_mu0(1, 0.014583333333333334) == (0.014583333333333334, "explicit_preexisting_calibration")
