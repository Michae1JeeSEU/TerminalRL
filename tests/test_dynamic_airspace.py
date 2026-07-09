"""Tests for dynamic terminal-airspace reconfiguration."""

from dataclasses import dataclass

import numpy as np
import pytest

from terminal_marl.core.dynamic_airspace import (
    AirspaceBounds,
    DynamicAirspace,
    DynamicRingSpec,
)


@dataclass
class ExistingRingSpec:
    """Minimal shape of the RingSpec already used by terminal_marl."""

    index: int
    radius: float
    capacity: int
    height: float


@pytest.fixture
def bounds() -> AirspaceBounds:
    return AirspaceBounds(
        radius_min=100.0,
        radius_max=500.0,
        height_min=50.0,
        height_max=450.0,
        speed_min=8.0,
        speed_max=35.0,
        min_ring_gap=60.0,
        min_height_gap=50.0,
        max_radius_change_per_update=25.0,
        max_height_change_per_update=20.0,
        max_speed_change_per_update=2.0,
        headway_min=2.0,
        headway_max=30.0,
        max_headway_change_per_update=1.5,
    )


@pytest.fixture
def airspace(bounds: AirspaceBounds) -> DynamicAirspace:
    return DynamicAirspace(
        outer_ring=DynamicRingSpec(2, 400.0, 350.0, 28.0, 6),
        inner_ring=DynamicRingSpec(1, 220.0, 180.0, 20.0, 2),
        bounds=bounds,
        ring_headway=7.0,
        final_headway=12.0,
    )


def test_initial_state_is_valid(airspace: DynamicAirspace) -> None:
    assert airspace.validate()
    assert airspace.get_headways() == {"ring": 7.0, "final": 12.0}


def test_random_actions_always_leave_valid_state(airspace: DynamicAirspace) -> None:
    rng = np.random.default_rng(2026)
    for _ in range(500):
        info = airspace.apply_high_level_action(rng.uniform(-2.0, 2.0, size=8))
        assert airspace.validate(), info
        assert info["new_state"].shape == (8,)


def test_extreme_action_is_clipped_and_reported(
    airspace: DynamicAirspace,
    bounds: AirspaceBounds,
) -> None:
    old_state = airspace.state_vector()
    info = airspace.apply_high_level_action(np.full(8, 100.0))

    assert info["clipped"]
    assert "action_bounds" in info["violations"]
    change = np.abs(info["new_state"] - old_state)
    assert change[0] <= bounds.max_radius_change_per_update + 1e-12
    assert change[1] <= bounds.max_radius_change_per_update + 1e-12
    assert change[2] <= bounds.max_height_change_per_update + 1e-12
    assert change[3] <= bounds.max_height_change_per_update + 1e-12
    assert change[4] <= bounds.max_speed_change_per_update + 1e-12
    assert change[5] <= bounds.max_speed_change_per_update + 1e-12
    assert change[6] <= bounds.max_headway_change_per_update + 1e-12
    assert change[7] <= bounds.max_headway_change_per_update + 1e-12


def test_ring_and_height_gaps_are_projected(bounds: AirspaceBounds) -> None:
    airspace = DynamicAirspace(
        outer_ring=DynamicRingSpec(2, 300.0, 250.0, 25.0, 4),
        inner_ring=DynamicRingSpec(
            1,
            300.0 - bounds.min_ring_gap,
            250.0 - bounds.min_height_gap,
            20.0,
            2,
        ),
        bounds=bounds,
    )

    info = airspace.apply_high_level_action(
        np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    )

    assert "min_ring_gap" in info["violations"]
    assert "min_height_gap" in info["violations"]
    assert airspace.outer_ring.radius - airspace.inner_ring.radius >= bounds.min_ring_gap - 1e-9
    assert airspace.outer_ring.height - airspace.inner_ring.height >= bounds.min_height_gap - 1e-9


def test_state_vector_shape_and_order_are_stable(airspace: DynamicAirspace) -> None:
    state = airspace.state_vector()
    assert state.shape == (8,)
    assert np.allclose(state, [400.0, 220.0, 350.0, 180.0, 28.0, 20.0, 7.0, 12.0])


def test_fixed_mode_is_a_no_op(airspace: DynamicAirspace) -> None:
    airspace.dynamic_enabled = False
    old_state = airspace.state_vector()
    info = airspace.apply_high_level_action(np.ones(8))

    assert np.array_equal(airspace.state_vector(), old_state)
    assert np.array_equal(info["old_state"], info["new_state"])
    assert not info["dynamic_enabled"]


def test_existing_ring_specs_can_be_adapted(bounds: AirspaceBounds) -> None:
    existing = [
        ExistingRingSpec(index=1, radius=200.0, capacity=2, height=150.0),
        ExistingRingSpec(index=2, radius=400.0, capacity=6, height=350.0),
    ]
    airspace = DynamicAirspace.from_ring_specs(
        existing,
        bounds=bounds,
        nominal_speeds=[18.0, 28.0],
    )

    converted = airspace.to_ring_specs()
    assert [ring.index for ring in converted] == [1, 2]
    assert [ring.capacity for ring in converted] == [2, 6]
    assert all(hasattr(ring, name) for ring in converted for name in ("radius", "height", "speed"))


def test_invalid_action_shape_is_rejected(airspace: DynamicAirspace) -> None:
    with pytest.raises(ValueError, match="shape"):
        airspace.apply_high_level_action(np.zeros(7))
