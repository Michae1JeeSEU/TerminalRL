"""Tests for safety-gated ring, final, and landing releases."""

from dataclasses import dataclass

import numpy as np

from terminal_marl import ExperimentMode, SimConfig, TerminalParallelEnv
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import FlightPhase


@dataclass
class State:
    uid: int
    position: np.ndarray


def safety() -> SafetyFilter:
    return SafetyFilter(
        safe_distance=1.0,
        boundary=(-10.0, 10.0, -10.0, 10.0, 0.0, 10.0),
        speed_max=3.0,
        acceleration_max=5.0,
        min_segment_duration=0.1,
    )


def safe_plan(uid: str = "1") -> TrajectoryPlan:
    return TrajectoryPlan(
        uav_id=uid,
        mode="ring_transfer",
        points=[
            TrajectoryPoint(0.0, 0.0, 2.0, 0.0, 0.0),
            TrajectoryPoint(1.0, 0.0, 2.0, 0.0, 1.0),
        ],
        metadata={},
    )


def test_outer_to_inner_release_is_approved_when_safe() -> None:
    first = State(1, np.array([0.0, 0.0, 2.0]))
    other = State(2, np.array([8.0, 0.0, 2.0]))
    approval = safety().approve_release(
        first,
        "outer_to_inner",
        [first, other],
        candidate_plan=safe_plan(),
    )

    assert approval["approved"]
    assert approval["decision"] == "RELEASE"


def test_minimum_separation_overrides_release_to_hold() -> None:
    first = State(1, np.array([0.0, 0.0, 2.0]))
    other = State(2, np.array([0.5, 0.0, 2.0]))
    approval = safety().approve_release(
        first,
        "outer_to_inner",
        [first, other],
        candidate_plan=safe_plan(),
    )

    assert not approval["approved"]
    assert approval["decision"] == "HOLD"
    assert "minimum_separation" in approval["violations"]


def test_final_and_landing_occupancy_are_checked() -> None:
    state = State(1, np.array([0.0, 0.0, 2.0]))
    final = safety().approve_release(
        state,
        "inner_to_final",
        [state],
        candidate_plan=safe_plan(),
        final_occupied=True,
    )
    landing = safety().approve_release(
        state,
        "final_to_land",
        [state],
        candidate_plan=safe_plan(),
        landing_pad_occupied=True,
    )

    assert "final_approach_occupied" in final["violations"]
    assert "landing_pad_occupied" in landing["violations"]


def test_unsafe_candidate_trajectory_blocks_release() -> None:
    state = State(1, np.array([0.0, 0.0, 2.0]))
    unsafe = TrajectoryPlan(
        uav_id="1",
        mode="ring_transfer",
        points=[
            TrajectoryPoint(0.0, 0.0, 2.0, 0.0, 0.0),
            TrajectoryPoint(20.0, 0.0, 2.0, 0.0, 0.01),
        ],
        metadata={},
    )
    approval = safety().approve_release(
        state,
        "outer_to_inner",
        [state],
        candidate_plan=unsafe,
    )

    assert not approval["approved"]
    assert approval["filtered_plan"].mode == "safety_override_hover"
    assert any(reason.startswith("trajectory_") for reason in approval["violations"])


def test_environment_records_release_hold_without_deleting_agent() -> None:
    env = TerminalParallelEnv(
        SimConfig(),
        experiment_mode=ExperimentMode.DYNAMIC_RULE,
    )
    env.reset(seed=4)
    first, second = env.aircraft[:2]
    outer = env.airspace.outer_ring
    first.phase = FlightPhase.RING
    first.assigned_ring = outer.index
    first.position = np.array([outer.radius, 0.0, outer.height])
    second.position = first.position + np.array([1.0, 0.0, 0.0])

    approvals = env._prepare_release_approvals({first.uid: True, second.uid: True})

    assert not approvals[first.uid]
    assert env.last_release_approval[first.uid]["decision"] == "HOLD"
    assert first in env.aircraft
    assert not first.finished
