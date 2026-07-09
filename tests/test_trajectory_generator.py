"""Tests for hardware-independent terminal trajectory generation."""

from dataclasses import dataclass
import math

import numpy as np
import pytest

from terminal_marl.core.dynamic_airspace import DynamicRingSpec
from terminal_marl.core.trajectory_generator import TrajectoryGenerator


@dataclass
class LandingPad:
    position: np.ndarray
    approach_height: float


@dataclass
class ExistingUAVState:
    """Minimal shape of the UAVState already used by terminal_marl."""

    uid: int
    position: np.ndarray
    speed: float
    heading: float


@pytest.fixture
def state() -> ExistingUAVState:
    return ExistingUAVState(
        uid=7,
        position=np.array([200.0, 0.0, 180.0]),
        speed=20.0,
        heading=math.pi / 2.0,
    )


@pytest.fixture
def ring() -> DynamicRingSpec:
    return DynamicRingSpec(index=1, radius=200.0, height=180.0, speed=20.0, capacity=2)


@pytest.fixture
def generator() -> TrajectoryGenerator:
    return TrajectoryGenerator(max_speed=50.0, max_acceleration=50.0)


def test_ring_hold_output_point_count(
    generator: TrajectoryGenerator,
    state: ExistingUAVState,
    ring: DynamicRingSpec,
) -> None:
    plan = generator.ring_hold(state, ring, duration=10.0, dt=2.0)
    assert len(plan.points) == 6


def test_ring_hold_times_are_strictly_increasing(
    generator: TrajectoryGenerator,
    state: ExistingUAVState,
    ring: DynamicRingSpec,
) -> None:
    plan = generator.ring_hold(state, ring, duration=10.0, dt=3.0)
    times = np.array([point.t for point in plan.points])
    assert times[0] == 0.0
    assert times[-1] == 10.0
    assert np.all(np.diff(times) > 0.0)


def test_ring_hold_points_stay_on_target_radius(
    generator: TrajectoryGenerator,
    state: ExistingUAVState,
    ring: DynamicRingSpec,
) -> None:
    plan = generator.ring_hold(state, ring, duration=12.0, dt=1.0)
    radii = np.array([math.hypot(point.x, point.y) for point in plan.points])
    assert np.allclose(radii, ring.radius, atol=1e-9)


def test_ring_hold_points_stay_at_target_height(
    generator: TrajectoryGenerator,
    state: ExistingUAVState,
    ring: DynamicRingSpec,
) -> None:
    plan = generator.ring_hold(state, ring, duration=12.0, dt=1.0)
    assert np.allclose([point.z for point in plan.points], ring.height)


def test_ring_hold_yaw_is_finite(
    generator: TrajectoryGenerator,
    state: ExistingUAVState,
    ring: DynamicRingSpec,
) -> None:
    plan = generator.ring_hold(state, ring, duration=12.0, dt=1.0)
    assert all(math.isfinite(point.yaw) for point in plan.points)
    assert not plan.metadata["violations"]


def test_ring_hold_mode_and_uav_id(
    generator: TrajectoryGenerator,
    state: ExistingUAVState,
    ring: DynamicRingSpec,
) -> None:
    plan = generator.ring_hold(state, ring, duration=5.0, dt=1.0)
    assert plan.mode == "ring_hold"
    assert plan.uav_id == "7"
    assert plan.metadata["executable"]


def test_ring_transfer_is_continuous_and_reaches_target(
    generator: TrajectoryGenerator,
    state: ExistingUAVState,
    ring: DynamicRingSpec,
) -> None:
    target = DynamicRingSpec(index=2, radius=350.0, height=300.0, speed=25.0, capacity=6)
    plan = generator.ring_transfer(state, ring, target, duration=20.0, dt=1.0)
    radii = np.array([math.hypot(point.x, point.y) for point in plan.points])
    heights = np.array([point.z for point in plan.points])

    assert plan.mode == "ring_transfer"
    assert radii[0] == pytest.approx(ring.radius)
    assert radii[-1] == pytest.approx(target.radius)
    assert heights[0] == pytest.approx(ring.height)
    assert heights[-1] == pytest.approx(target.height)
    assert np.all(np.diff(radii) >= -1e-9)
    assert np.all(np.diff(heights) >= -1e-9)


def test_final_approach_and_landing_use_pad_targets(
    generator: TrajectoryGenerator,
    state: ExistingUAVState,
) -> None:
    pad = LandingPad(position=np.array([0.0, 0.0, 0.0]), approach_height=30.0)
    approach = generator.final_approach(state, pad, duration=20.0, dt=2.0)
    approach_end = approach.points[-1]
    assert (approach_end.x, approach_end.y, approach_end.z) == pytest.approx((0.0, 0.0, 30.0))

    landing_state = {"uav_id": "uav_7", "position": [0.0, 0.0, 30.0], "heading": 0.0}
    landing = generator.landing(landing_state, pad, duration=10.0, dt=1.0)
    landing_end = landing.points[-1]
    assert landing.mode == "landing"
    assert (landing_end.x, landing_end.y, landing_end.z) == pytest.approx((0.0, 0.0, 0.0))


def test_emergency_hover_is_stationary(
    generator: TrajectoryGenerator,
    state: ExistingUAVState,
) -> None:
    plan = generator.emergency_hover(state, duration=3.0, dt=1.0)
    positions = np.array([[point.x, point.y, point.z] for point in plan.points])
    assert plan.mode == "emergency_hover"
    assert np.allclose(positions, state.position)


def test_non_positive_timing_is_rejected(
    generator: TrajectoryGenerator,
    state: ExistingUAVState,
    ring: DynamicRingSpec,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        generator.ring_hold(state, ring, duration=0.0, dt=1.0)
