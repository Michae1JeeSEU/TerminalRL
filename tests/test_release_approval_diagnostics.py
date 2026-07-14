"""Release approvals expose enough diagnostics for server deadlock triage."""

import numpy as np

from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint


def test_release_reject_includes_kinematic_and_distance_diagnostics() -> None:
    safety = SafetyFilter(
        safe_distance=10.0,
        boundary=(-100.0, 100.0, -100.0, 100.0, 0.0, 100.0),
        speed_max=20.0,
        acceleration_max=1.0,
    )
    state = {"uid": 1, "position": np.array([0.0, 0.0, 10.0])}
    other = {"uid": 2, "position": np.array([50.0, 0.0, 10.0])}
    plan = TrajectoryPlan(
        uav_id="1",
        mode="outer_to_inner",
        points=[
            TrajectoryPoint(0.0, 0.0, 10.0, 0.0, 0.0),
            TrajectoryPoint(1.0, 0.0, 10.0, 0.0, 1.0),
            TrajectoryPoint(15.0, 0.0, 10.0, 0.0, 2.0),
        ],
        metadata={},
    )

    approval = safety.approve_release(
        state,
        "outer_to_inner",
        [state, other],
        candidate_plan=plan,
    )

    assert not approval["approved"]
    diagnostics = approval["diagnostics"]
    assert diagnostics["current_min_distance"] == 50.0
    assert diagnostics["trajectory_min_distance"] is not None
    kinematics = diagnostics["trajectory"]["diagnostics"]["kinematic_detail"]
    assert kinematics["max_acceleration_observed"] > kinematics["acceleration_max_allowed"]
    assert kinematics["acceleration_violation_segments"]
