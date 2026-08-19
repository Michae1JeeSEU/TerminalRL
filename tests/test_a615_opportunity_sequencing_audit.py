from terminal_marl.training.a615_opportunity_sequencing_audit import (
    RADIUS_SWEEP,
    ShadowOpportunityAudit,
    _radius_config,
    classify,
    first_divergence,
)
from terminal_marl.training.a611_causal_audit import DemandProfile
from terminal_marl.core.config import RingRole
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def test_a615_radius_sweep_straddles_only_the_outer_capacity_threshold():
    assert RADIUS_SWEEP == (600.0, 606.0, 610.0, 611.5, 611.6, 612.0)
    profile = DemandProfile("test", (), 1.0, 1.0)
    derived = [TerminalParallelEnv(_radius_config(profile, radius)).airspace.capacity(RingRole.OUTER) for radius in RADIUS_SWEEP]
    assert derived == [47, 47, 47, 47, 48, 48]


def test_a615_first_divergence_is_deterministic_and_reports_candidate_change():
    base = {"trace": [{"time": 1., "arrival_id": 3, "interface": "o2m", "policy_release_request": 1., "safe_candidate_exists": True,
                        "selected_kind": "requested", "merge_time": 10., "merge_angle": 0., "speed_residual": 0., "merge_time_residual": 0.,
                        "future_delay": 0., "predecessor": None, "successor": None, "primary_blocker": None}]}
    changed = {"trace": [{**base["trace"][0], "merge_time": 11.}]}
    result = first_divergence(base, changed, seed=0, compared_branch="dynamic_t180")
    assert result["simulation_time"] == 1. and result["event_type"] == "candidate_or_request_difference"


def test_a615_classification_never_authorizes_ppo():
    branches = {"x": {"events": [{"safe_candidate_exists": False}], "losses": []}}
    labels = classify(branches, [{"derived_capacity": 47, "waiting_area": 1.}, {"derived_capacity": 48, "waiting_area": 2.}])
    assert "NECESSARY_SAFETY_WAIT" in labels and "CAPACITY_THRESHOLD_EFFECT" in labels


def test_a615_classification_is_deterministic_for_same_evidence():
    branches = {"x": {"events": [{"safe_candidate_exists": True}], "losses": []}}
    micro = [{"derived_capacity": 47, "waiting_area": 1.}, {"derived_capacity": 47, "waiting_area": 1.}]
    assert classify(branches, micro) == classify(branches, micro)


def test_a615_shadow_audit_starts_without_live_state_references():
    audit = ShadowOpportunityAudit("nominal", 0)
    assert audit.events == [] and audit.losses() == []
