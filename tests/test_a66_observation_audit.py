import numpy as np

from terminal_marl.training.a66_observation_audit import (
    MEANINGFUL_WAITING_MARGIN,
    _diverse_select,
    _oracle,
)
from terminal_marl.training.a65_counterfactual_audit import Candidate


def _row(completed: int, waiting: float) -> dict:
    return {"safe": True, "outcomes": {}, "shield_cost": 0., "metrics": {
        "completed": completed, "all_request_waiting_area": waiting, "unresolved_count": 0., "unresolved_age_p95": 0.,
        "service_delay_p95": 1., "service_delay_p99": 1., "energy_mean": 1., "priority_fairness_std": 1.,
    }}


def test_a66_oracle_marks_near_tie_and_meaningful_waiting_margin_explicitly():
    candidates = {"no_op": Candidate("no_op", "NO_OP", (0.,) * 10), "speed": Candidate("speed", "SPEED", (0.,) * 10), "headway": Candidate("headway", "HEADWAY", (0.,) * 10)}
    near = _oracle("s0", {"no_op": _row(1, 100.), "speed": _row(1, 99.), "headway": _row(1, 120.)}, candidates)
    assert near["near_tie"] and not near["meaningful_margin"]
    distinct = _oracle("s1", {"no_op": _row(1, MEANINGFUL_WAITING_MARGIN + 200.), "speed": _row(1, 0.), "headway": _row(1, MEANINGFUL_WAITING_MARGIN + 300.)}, candidates)
    assert distinct["meaningful_margin"] and not distinct["near_tie"]


def test_a66_diverse_selection_is_deterministic_and_bounded():
    pool = []
    for index in range(8):
        pool.append(({"observation_29d": (np.arange(29) * index).tolist(), "snapshot_id": str(index)}, b"x"))
    result = _diverse_select(pool, 4)
    assert len(result) == 4
    assert [item[0]["snapshot_id"] for item in result] == [item[0]["snapshot_id"] for item in _diverse_select(pool, 4)]
