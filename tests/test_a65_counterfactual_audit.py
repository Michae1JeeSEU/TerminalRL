import numpy as np

from terminal_marl.core.config import SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.training.a65_counterfactual_audit import (
    Candidate,
    NO_OP,
    available_cpus,
    candidate_set,
    summarize,
)


def test_a65_candidates_are_canonical_10d_and_include_real_capacity_thresholds():
    candidates = candidate_set()
    assert candidates[0].name == "no_op"
    assert all(len(candidate.action) == 10 for candidate in candidates)
    assert {candidate.family for candidate in candidates} >= {"NO_OP", "HEADWAY", "GEOMETRY", "SPEED", "COMBINED"}
    for candidate in (item for item in candidates if item.name.endswith("capacity_plus1")):
        airspace = DynamicAirspace(SimConfig())
        update = airspace.request_update(np.asarray(candidate.action, dtype=np.float32), now=0., geometry_gate=True)
        assert any(after > before for before, after in zip(update.capacity_before, update.capacity_after))


def test_a65_cpu_count_and_summary_fail_closed_without_leverage():
    assert available_cpus() >= 1
    stage = {
        "snapshot_restore_noop_deterministic": True,
        "candidates": {"no_op": {"family": "NO_OP"}, "speed": {"family": "SPEED"}},
        "results": {"s0": {
            "no_op": {"safe": True, "metrics": {"completed": 1, "all_request_waiting_area": 1., "unresolved_count": 0., "unresolved_age_p95": 0., "energy_mean": 1., "priority_fairness_std": 1., "service_delay_p95": 1., "service_delay_p99": 1.}, "outcomes": {}, "shield_cost": 0.},
            "speed": {"safe": True, "metrics": {"completed": 1, "all_request_waiting_area": 2., "unresolved_count": 0., "unresolved_age_p95": 0., "energy_mean": 2., "priority_fairness_std": 2., "service_delay_p95": 2., "service_delay_p99": 2.}, "outcomes": {}, "shield_cost": 0., "capacity_changing_updates": 0},
        }},
        "analysis": {"s0": {"best": "no_op", "best_family": "NO_OP", "best_nonzero": "speed", "best_nonzero_family": "SPEED", "best_nonzero_vs_no_op": {"completed": 0, "waiting_area": 1., "energy": 1., "fairness": 1.}, "safe_candidate_rate": 1.}},
    }
    result = summarize(stage)
    assert result["decision"] == "STOP_BEFORE_PPO"
    assert np.array_equal(NO_OP, np.zeros(10, dtype=np.float32))
