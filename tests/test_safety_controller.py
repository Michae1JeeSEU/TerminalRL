"""Tests for SafetyController reflexive collision avoidance."""

import numpy as np
import pytest

from src.config import SafetyConfig, UAVConfig, RingConfig
from src.env.ring_airspace import RingAirspace
from src.env.uav import UAV, UAVStatus, LowLevelAction
from src.env.conflict_detector import ConflictDetector
from src.env.safety_controller import SafetyController, SafetyAdjustment


def make_uav(uid, x, y, z, ring_id=1, base_priority=1, battery=None,
             status=UAVStatus.CRUISING, mission_type=1):
    cfg = UAVConfig()
    uav = UAV(uid, cfg, base_priority=base_priority, mission_type=mission_type)
    if battery is None:
        battery = cfg.max_battery
    uav.reset(np.array([x, y, z], dtype=np.float64), ring_id=ring_id,
              battery=battery, status=status)
    return uav


def make_airspace():
    rc = RingConfig()
    return RingAirspace(rc)


def make_conflict_detector():
    return ConflictDetector(safe_distance=10.0, near_miss_threshold=50.0)


# ---- SafetyAdjustment ----

class TestSafetyAdjustment:
    def test_default_values(self):
        adj = SafetyAdjustment()
        assert adj.speed_mult == 1.0
        assert adj.psi_offset == 0.0
        assert adj.gamma_offset == 0.0
        assert not adj.blocked
        assert adj.active_rules == []

    def test_custom_values(self):
        adj = SafetyAdjustment(speed_mult=0.8, psi_offset=0.3, blocked=True,
                                active_rules=["R1", "R2"])
        assert adj.speed_mult == 0.8
        assert adj.psi_offset == 0.3
        assert adj.blocked
        assert "R1" in adj.active_rules


# ---- SafetyController.compute_adjustment (composition) ----

class TestComposition:
    def test_no_adjustments_returns_default(self):
        result = SafetyController.compute_adjustment([])
        assert result.speed_mult == 1.0
        assert result.psi_offset == 0.0
        assert not result.blocked

    def test_single_adjustment_passthrough(self):
        a = SafetyAdjustment(speed_mult=0.7)
        result = SafetyController.compute_adjustment([a])
        assert result.speed_mult == 0.7

    def test_min_speed_mult_wins(self):
        a1 = SafetyAdjustment(speed_mult=0.85)
        a2 = SafetyAdjustment(speed_mult=0.70)
        a3 = SafetyAdjustment(speed_mult=0.95)
        result = SafetyController.compute_adjustment([a1, a2, a3])
        assert result.speed_mult == 0.70

    def test_largest_psi_wins(self):
        a1 = SafetyAdjustment(psi_offset=0.1)
        a2 = SafetyAdjustment(psi_offset=-0.3)
        a3 = SafetyAdjustment(psi_offset=0.2)
        result = SafetyController.compute_adjustment([a1, a2, a3])
        assert result.psi_offset == -0.3

    def test_blocked_or(self):
        a1 = SafetyAdjustment(blocked=False)
        a2 = SafetyAdjustment(blocked=True)
        result = SafetyController.compute_adjustment([a1, a2])
        assert result.blocked

    def test_rules_collected(self):
        a1 = SafetyAdjustment(active_rules=["R1"])
        a2 = SafetyAdjustment(active_rules=["R2"])
        result = SafetyController.compute_adjustment([a1, a2])
        assert "R1" in result.active_rules
        assert "R2" in result.active_rules


# ---- Rule 1: Same-ring angular separation ----

class TestRule1AngularSeparation:
    @pytest.fixture
    def sc(self):
        return SafetyController(SafetyConfig(), make_airspace(), make_conflict_detector())

    def test_single_uav_no_adjustment(self, sc):
        uav = make_uav(0, 4000, 0, 800, ring_id=1)
        actions = {0: LowLevelAction(control=0)}
        result = sc._rule1_angular_separation([uav])
        assert 0 not in result

    def test_two_uavs_far_apart_no_adjustment(self, sc):
        """Two UAVs at opposite angles on ring 1 (R=4000): arc distance ≈ 12566m."""
        u1 = make_uav(0, 4000, 0, 800, ring_id=1)
        u2 = make_uav(1, -4000, 0, 800, ring_id=1)
        # Force angles
        u1._angle = 0.0
        u2._angle = np.pi
        actions = {0: LowLevelAction(control=0), 1: LowLevelAction(control=0)}
        result = sc._rule1_angular_separation([u1, u2])
        assert 0 not in result
        assert 1 not in result

    def test_two_uavs_close_together_slow_down(self, sc):
        """Two UAVs very close angularly should trigger slowdown.
        Ring 1 R=4000, 50m threshold: safe_angle = 50/4000 = 0.0125 rad.
        Gap of 0.005 rad = 20m arc → slowdown.
        """
        u1 = make_uav(0, 4000, 0, 800, ring_id=1)
        u2 = make_uav(1, 4000, 0, 800, ring_id=1)
        u1._angle = 0.0
        u2._angle = 0.005  # very close
        actions = {0: LowLevelAction(control=0), 1: LowLevelAction(control=0)}
        result = sc._rule1_angular_separation([u1, u2])
        # Both should be slowed since each sees the other as close neighbor
        assert 0 in result
        assert result[0].speed_mult <= 0.95

    def test_descending_uav_ignored(self, sc):
        """Only CRUISING UAVs are considered for angular separation."""
        u1 = make_uav(0, 4000, 0, 800, ring_id=1, status=UAVStatus.CRUISING)
        u2 = make_uav(1, 4000, 10, 800, ring_id=1, status=UAVStatus.DESCENDING)
        u1._angle = 0.0
        u2._angle = 0.005
        actions = {0: LowLevelAction(control=0), 1: LowLevelAction(control=1)}
        result = sc._rule1_angular_separation([u1, u2])
        # Only u1 should be considered (u2 is descender)
        # With 1 circler, no adjustment
        assert 1 not in result


# ---- Rule 2: Circling yields to descending ----

class TestRule2CirclingYieldsDescending:
    @pytest.fixture
    def sc(self):
        return SafetyController(SafetyConfig(), make_airspace(), make_conflict_detector())

    def test_no_descenders_no_adjustment(self, sc):
        u1 = make_uav(0, 4000, 0, 800, ring_id=1)
        u2 = make_uav(1, 4000, 0, 800, ring_id=1)
        actions = {0: LowLevelAction(control=0), 1: LowLevelAction(control=0)}
        result = sc._rule2_circling_yields_descending([u1, u2], actions)
        assert len(result) == 0

    def test_descender_crosses_ring_slows_circler(self, sc):
        """Descender crossing ring 1 altitude: nearby circler should slow."""
        c = make_uav(0, 4000, 0, 800, ring_id=1, status=UAVStatus.CRUISING)
        d = make_uav(1, 4000, 0, 900, ring_id=1, status=UAVStatus.DESCENDING)
        # Descender velocity: straight down
        d.velocity = np.array([0.0, 0.0, -20.0])
        d._angle = 0.0
        c._angle = 0.0
        actions = {0: LowLevelAction(control=0), 1: LowLevelAction(control=1, speed=25.0)}
        result = sc._rule2_circling_yields_descending([c, d], actions)
        # Circler should be slowed because descender crosses ring 1 at same angle
        assert 0 in result
        assert result[0].speed_mult <= sc.config.descend_crossing_slowdown
        assert "R2_circle_yield" in result[0].active_rules

    def test_descender_not_crossing_ring_no_effect(self, sc):
        """Descender moving horizontally at high altitude won't cross ring 2."""
        c = make_uav(0, 2500, 0, 500, ring_id=2, status=UAVStatus.CRUISING)
        d = make_uav(1, 4000, 0, 900, ring_id=1, status=UAVStatus.DESCENDING)
        d.velocity = np.array([20.0, 0.0, 0.0])  # horizontal
        c._angle = 0.0
        d._angle = 0.0
        actions = {0: LowLevelAction(control=0), 1: LowLevelAction(control=1, speed=25.0)}
        result = sc._rule2_circling_yields_descending([c, d], actions)
        # Descender at z=900 moving horizontally, ring 2 at z=500 — no crossing
        assert 0 not in result


# ---- Rule 3: Dual descent divergence ----

class TestRule3DualDescent:
    @pytest.fixture
    def sc(self):
        return SafetyController(SafetyConfig(), make_airspace(), make_conflict_detector())

    def test_two_descenders_converging(self, sc):
        d1 = make_uav(0, 2000, 0, 400, ring_id=2, status=UAVStatus.DESCENDING)
        d2 = make_uav(1, 2030, 0, 400, ring_id=2, status=UAVStatus.DESCENDING)
        d1.velocity = np.array([10.0, 0.0, -5.0])
        d2.velocity = np.array([-10.0, 0.0, -5.0])  # converging
        actions = {0: LowLevelAction(control=1, psi=0.0, speed=15.0),
                   1: LowLevelAction(control=1, psi=np.pi, speed=15.0)}
        result = sc._rule3_dual_descent_divergence([d1, d2], actions)
        # Both should get psi offset
        assert 0 in result or 1 in result

    def test_two_descenders_separating_no_adjustment(self, sc):
        d1 = make_uav(0, 2000, 0, 400, ring_id=2, status=UAVStatus.DESCENDING)
        d2 = make_uav(1, 2030, 0, 400, ring_id=2, status=UAVStatus.DESCENDING)
        d1.velocity = np.array([-10.0, 0.0, -5.0])
        d2.velocity = np.array([10.0, 0.0, -5.0])  # separating
        actions = {0: LowLevelAction(control=1, psi=np.pi, speed=15.0),
                   1: LowLevelAction(control=1, psi=0.0, speed=15.0)}
        result = sc._rule3_dual_descent_divergence([d1, d2], actions)
        assert len(result) == 0

    def test_descenders_far_apart_no_adjustment(self, sc):
        d1 = make_uav(0, 4000, 0, 400, ring_id=2, status=UAVStatus.DESCENDING)
        d2 = make_uav(1, -4000, 0, 400, ring_id=2, status=UAVStatus.DESCENDING)
        # >50m apart, no adjustment
        actions = {0: LowLevelAction(control=1), 1: LowLevelAction(control=1)}
        result = sc._rule3_dual_descent_divergence([d1, d2], actions)
        assert len(result) == 0


# ---- Rule 4: Universal proximity brake ----

class TestRule4UniversalBrake:
    @pytest.fixture
    def sc(self):
        return SafetyController(SafetyConfig(), make_airspace(), make_conflict_detector())

    def test_two_uavs_within_30m_brake(self, sc):
        u1 = make_uav(0, 1000, 0, 400, ring_id=2)
        u2 = make_uav(1, 1020, 0, 400, ring_id=2)
        u1.velocity = np.array([10.0, 0.0, 0.0])
        u2.velocity = np.array([-10.0, 0.0, 0.0])  # converging
        result = sc._rule4_universal_brake([u1, u2])
        assert 0 in result
        assert 1 in result
        assert result[0].speed_mult < 1.0
        assert result[1].speed_mult < 1.0

    def test_far_apart_no_brake(self, sc):
        u1 = make_uav(0, 1000, 0, 400, ring_id=2)
        u2 = make_uav(1, 3000, 0, 400, ring_id=2)
        result = sc._rule4_universal_brake([u1, u2])
        assert len(result) == 0

    def test_landed_uav_ignored(self, sc):
        u1 = make_uav(0, 1000, 0, 400, ring_id=3, status=UAVStatus.CRUISING)
        u2 = make_uav(1, 1020, 0, 400, ring_id=3, status=UAVStatus.LANDED)
        result = sc._rule4_universal_brake([u1, u2])
        assert len(result) == 0


# ---- Rule 5: Emergency override ----

class TestRule5EmergencyOverride:
    @pytest.fixture
    def sc(self):
        return SafetyController(SafetyConfig(), make_airspace(), make_conflict_detector())

    def test_emergency_uav_not_slowed(self, sc):
        e = make_uav(0, 1000, 0, 400, ring_id=3, base_priority=5)
        n = make_uav(1, 1020, 0, 400, ring_id=3, base_priority=1)
        result = sc._rule5_emergency_override([e, n])
        # Emergency UAV gets sentinel adjustment (speed_mult=1.0, no slowdown)
        assert 0 in result
        assert result[0].speed_mult == 1.0
        assert not result[0].blocked
        # Non-emergency nearby should yield
        assert 1 in result
        assert result[1].speed_mult <= sc.config.emergency_yield_slowdown

    def test_critical_battery_is_emergency(self, sc):
        e = make_uav(0, 1000, 0, 400, ring_id=3, base_priority=1,
                      battery=5.0)  # 5% battery < 15% critical
        n = make_uav(1, 1020, 0, 400, ring_id=3, base_priority=3)
        result = sc._rule5_emergency_override([e, n])
        assert 0 in result
        assert result[0].speed_mult == 1.0  # not slowed

    def test_non_emergency_far_away_unaffected(self, sc):
        e = make_uav(0, 1000, 0, 400, ring_id=3, base_priority=5)
        n = make_uav(1, 4000, 0, 800, ring_id=1, base_priority=1)  # far away
        result = sc._rule5_emergency_override([e, n])
        # Emergency still gets its sentinel
        assert 0 in result
        # Non-emergency too far to be affected
        assert 1 not in result


# ---- Priority classification ----

class TestPriorityClassification:
    def test_emergency_beats_high_priority(self):
        e = make_uav(0, 1000, 0, 400, ring_id=3, base_priority=5)
        h = make_uav(1, 1020, 0, 400, ring_id=3, base_priority=3)
        result = SafetyController._classify_priority(e, h)
        assert result is not None
        higher, lower = result
        assert higher.id == 0
        assert lower.id == 1

    def test_high_beats_low(self):
        h = make_uav(0, 1000, 0, 400, ring_id=3, base_priority=3)
        l = make_uav(1, 1020, 0, 400, ring_id=3, base_priority=1)
        result = SafetyController._classify_priority(h, l)
        assert result is not None
        higher, lower = result
        assert higher.id == 0
        assert lower.id == 1

    def test_equal_priority_returns_none(self):
        a = make_uav(0, 1000, 0, 400, ring_id=3, base_priority=1)
        b = make_uav(1, 1020, 0, 400, ring_id=3, base_priority=1)
        result = SafetyController._classify_priority(a, b)
        assert result is None

    def test_both_emergency_equal(self):
        a = make_uav(0, 1000, 0, 400, ring_id=3, base_priority=5)
        b = make_uav(1, 1020, 0, 400, ring_id=3, base_priority=5)
        result = SafetyController._classify_priority(a, b)
        assert result is None

    def test_critical_battery_is_emergency(self):
        e = make_uav(0, 1000, 0, 400, ring_id=3, base_priority=1, battery=5.0)
        n = make_uav(1, 1020, 0, 400, ring_id=3, base_priority=3)
        result = SafetyController._classify_priority(e, n)
        assert result is not None
        higher, _lower = result
        assert higher.id == 0


# ---- apply() method ----

class TestApply:
    @pytest.fixture
    def sc(self):
        return SafetyController(SafetyConfig(), make_airspace(), make_conflict_detector())

    def test_no_adjustment_passthrough(self, sc):
        actions = {0: LowLevelAction(control=1, psi=1.0, speed=25.0, landing_zone=2)}
        adjusted = sc.apply(actions, {})
        assert adjusted[0].control == 1
        assert adjusted[0].psi == 1.0
        assert adjusted[0].speed == 25.0
        assert adjusted[0].landing_zone == 2

    def test_speed_mult_applied(self, sc):
        actions = {0: LowLevelAction(control=0, speed=25.0)}
        adj = {0: SafetyAdjustment(speed_mult=0.85)}
        adjusted = sc.apply(actions, adj)
        assert adjusted[0].speed == 21.25  # 25 * 0.85

    def test_psi_offset_applied(self, sc):
        actions = {0: LowLevelAction(control=1, psi=0.5, speed=25.0)}
        adj = {0: SafetyAdjustment(psi_offset=0.3)}
        adjusted = sc.apply(actions, adj)
        assert adjusted[0].psi == 0.8  # 0.5 + 0.3

    def test_blocked_forces_circle(self, sc):
        actions = {0: LowLevelAction(control=1, psi=0.0, speed=25.0)}
        adj = {0: SafetyAdjustment(blocked=True)}
        adjusted = sc.apply(actions, adj)
        assert adjusted[0].control == 0  # forced CIRCLE


# ---- Full evaluate integration ----

class TestEvaluateIntegration:
    @pytest.fixture
    def sc(self):
        return SafetyController(SafetyConfig(), make_airspace(), make_conflict_detector())

    def test_empty_uavs_returns_empty(self, sc):
        result = sc.evaluate([], {})
        assert len(result) == 0

    def test_single_cruising_uav_no_adjustments(self, sc):
        u = make_uav(0, 4000, 0, 800, ring_id=1)
        actions = {0: LowLevelAction(control=0)}
        result = sc.evaluate([u], actions)
        assert 0 not in result or result[0].speed_mult == 1.0

    def test_multiple_rules_compose(self, sc):
        """Two UAVs close together should trigger rules 1, 4, and possibly others."""
        u1 = make_uav(0, 2500, 0, 500, ring_id=2)
        u2 = make_uav(1, 2510, 0, 500, ring_id=2)
        u1._angle = 0.0
        u2._angle = 0.005
        u1.velocity = np.array([5.0, 0.0, 0.0])
        u2.velocity = np.array([-5.0, 0.0, 0.0])
        actions = {0: LowLevelAction(control=0), 1: LowLevelAction(control=0)}
        result = sc.evaluate([u1, u2], actions)
        assert 0 in result or 1 in result
        # Speed should be reduced (at least rule 1 or 4)
        for uid in result:
            assert result[uid].speed_mult <= 1.0
