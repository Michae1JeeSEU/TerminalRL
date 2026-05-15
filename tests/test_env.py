"""Tests for the terminal area environment modules (revised orbital model)."""

import numpy as np
import pytest

from src.config import EnvConfig, RingConfig, UAVConfig
from src.env.ring_airspace import RingAirspace
from src.env.uav import UAV, UAVStatus, LowLevelAction
from src.env.uav_manager import UAVManager
from src.env.conflict_detector import ConflictDetector
from src.env.priority_engine import PriorityEngine
from src.env.metrics_collector import MetricsCollector
from src.env.terminal_area_env import TerminalAreaEnv


class TestRingAirspace:
    def setup_method(self):
        self.config = RingConfig()
        self.airspace = RingAirspace(self.config)

    def test_initial_state(self):
        assert np.allclose(self.airspace.radii, self.config.default_radii)
        assert np.allclose(self.airspace.heights, self.config.default_heights)

    def test_valid_update(self):
        ok = self.airspace.update(
            np.array([3000., 2000., 1000.]),
            np.array([600., 400., 200.]),
            np.array([20., 15., 10.]),
        )
        assert ok
        assert self.airspace.radii[0] == 3000.

    def test_invalid_radius_ordering_rejected(self):
        ok = self.airspace.update(
            np.array([1000., 2000., 3000.]),
            np.array([600., 400., 200.]),
            np.array([20., 15., 10.]),
        )
        assert not ok

    def test_invalid_height_ordering_rejected(self):
        ok = self.airspace.update(
            np.array([3000., 2000., 1000.]),
            np.array([200., 400., 600.]),
            np.array([20., 15., 10.]),
        )
        assert not ok

    def test_get_ring(self):
        pos = np.array([3500., 0., 700.])
        assert self.airspace.get_ring(pos) == 1

    def test_inner_ring_detection(self):
        pos = np.array([500., 0., 100.])
        assert self.airspace.in_inner_ring(pos)

    def test_outside_all_rings(self):
        pos = np.array([6000., 0., 1200.])
        assert self.airspace.get_ring(pos) == 0

    def test_state_vector(self):
        sv = self.airspace.state_vector
        assert sv.shape == (9,)
        assert np.allclose(sv[:3], self.config.default_radii)

    def test_entry_points(self):
        pts = self.airspace.get_entry_points()
        assert pts.shape == (4, 3)
        R1 = self.config.default_radii[0]
        H1 = self.config.default_heights[0]
        for i, pt in enumerate(pts):
            r = np.linalg.norm(pt[:2])
            assert abs(r - R1) < 1.0
            assert abs(pt[2] - H1) < 1.0

    def test_get_ring_orbit(self):
        r, h = self.airspace.get_ring_orbit(1)
        assert r == self.config.default_radii[0]
        assert h == self.config.default_heights[0]

    def test_random_position_on_ring_stays_on_orbit(self):
        pos = self.airspace.random_position_on_ring(2)
        r = np.linalg.norm(pos[:2])
        assert abs(r - self.config.default_radii[1]) < 1.0
        assert abs(pos[2] - self.config.default_heights[1]) < 1.0


class TestUAV:
    def setup_method(self):
        self.config = UAVConfig()
        self.uav = UAV(0, self.config, base_priority=3)

    def test_initial_state(self):
        self.uav.reset(np.array([3500., 0., 700.]), ring_id=1)
        assert self.uav.status == UAVStatus.CRUISING
        assert self.uav.battery == self.config.max_battery

    def test_circle_motion(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        initial_pos = self.uav.position.copy()
        self.uav.step(LowLevelAction(control=0), ring_speed=25.0,
                       ring_radius=4000.0, ring_height=800.0, dt=2.0)
        assert not np.allclose(self.uav.position, initial_pos)
        assert self.uav.status == UAVStatus.CRUISING

    def test_circle_orbit_lock(self):
        """UAV on CIRCLE must stay at ring orbit (r, h)."""
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        for _ in range(10):
            self.uav.step(LowLevelAction(control=0), ring_speed=25.0,
                           ring_radius=4000.0, ring_height=800.0, dt=2.0)
            r = np.linalg.norm(self.uav.position[:2])
            assert abs(r - 4000.0) < 1e-6
            assert abs(self.uav.position[2] - 800.0) < 1e-6

    def test_descend_decreases_height(self):
        """DESCEND with negative gamma should decrease height."""
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        initial_z = self.uav.position[2]
        action = LowLevelAction(control=1, psi=0.0, gamma=-0.3, speed=25.0)
        self.uav.step(action, ring_speed=0.0, ring_radius=4000.0, ring_height=800.0, dt=2.0)
        assert self.uav.position[2] < initial_z
        assert self.uav.status == UAVStatus.DESCENDING

    def test_descend_3d_motion(self):
        """DESCEND with (psi=0, gamma=0, V) moves in +x direction at speed V."""
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        action = LowLevelAction(control=1, psi=0.0, gamma=0.0, speed=20.0)
        self.uav.step(action, ring_speed=0.0, ring_radius=4000.0, ring_height=800.0, dt=2.0)
        # gamma=0, psi=0 → move in +x, no z change
        assert self.uav.position[0] > 4000.0
        assert abs(self.uav.position[2] - 800.0) < 1e-6

    def test_energy_consumption(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        self.uav.step(LowLevelAction(control=0), ring_speed=25.0,
                       ring_radius=4000.0, ring_height=800.0, dt=10.0)
        assert self.uav.energy_consumed > 0
        assert self.uav.battery < self.config.max_battery

    def test_battery_critical(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        self.uav.battery = self.config.battery_critical * self.config.max_battery * 0.5
        assert self.uav.is_critical_battery

    def test_ring3_landing(self):
        """On ring 3, DESCEND + landing_zone >= 0 triggers continuous landing approach.
        UAV descends toward ground and lands when z <= 1.0m."""
        self.uav.reset(np.array([500., 0., 200.]), ring_id=3)
        action = LowLevelAction(control=1, psi=0.0, gamma=-0.5, speed=25.0, landing_zone=0)
        # Step until landed (continuous approach)
        for _ in range(20):
            self.uav.step(action, ring_speed=15.0, ring_radius=1000.0, ring_height=200.0, dt=2.0)
            if self.uav.status == UAVStatus.LANDED:
                break
        assert self.uav.status == UAVStatus.LANDED
        assert self.uav.landing_zone == 0
        assert self.uav.position[2] == 0.0

    def test_descend_no_landing_without_zone(self):
        """On ring 3, DESCEND without landing_zone should NOT land."""
        self.uav.reset(np.array([500., 0., 200.]), ring_id=3)
        action = LowLevelAction(control=1, psi=0.0, gamma=-0.3, speed=25.0, landing_zone=-1)
        self.uav.step(action, ring_speed=15.0, ring_radius=1000.0, ring_height=200.0, dt=2.0)
        assert self.uav.status == UAVStatus.DESCENDING  # still descending, not landed

    def test_is_dead(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        self.uav.battery = 0.0
        assert self.uav.is_dead

    def test_priority_increases_with_wait(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        p1 = self.uav.compute_priority()
        self.uav.wait_time = 7200.0
        p2 = self.uav.compute_priority()
        assert p2 > p1

    def test_priority_increases_with_low_battery(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        p1 = self.uav.compute_priority()
        self.uav.battery = self.config.battery_critical * self.config.max_battery * 0.3
        p2 = self.uav.compute_priority()
        assert p2 > p1

    def test_angle_property(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        assert abs(self.uav.angle) < 1e-9  # angle=0 at (R, 0, H)


class TestConflictDetector:
    def test_no_conflict(self):
        d = ConflictDetector(50.0)
        u1 = UAV(0, UAVConfig(), 1)
        u2 = UAV(1, UAVConfig(), 1)
        u1.reset(np.array([0., 0., 100.]))
        u2.reset(np.array([100., 0., 100.]))
        assert not d.check_pairwise(u1, u2)

    def test_conflict_detected(self):
        d = ConflictDetector(50.0)
        u1 = UAV(0, UAVConfig(), 1)
        u2 = UAV(1, UAVConfig(), 1)
        u1.reset(np.array([0., 0., 100.]))
        u2.reset(np.array([30., 0., 100.]))
        assert d.check_pairwise(u1, u2)

    def test_landed_uav_excluded(self):
        d = ConflictDetector(50.0)
        u1 = UAV(0, UAVConfig(), 1)
        u2 = UAV(1, UAVConfig(), 1)
        u1.reset(np.array([0., 0., 100.]))
        u1.status = UAVStatus.LANDED
        u2.reset(np.array([0., 0., 100.]))
        assert not d.check_pairwise(u1, u2)

    def test_near_miss_detected(self):
        """Distance in [safe_dist, near_miss_threshold) should be near-miss, not collision."""
        d = ConflictDetector(10.0, near_miss_threshold=50.0)
        u1 = UAV(0, UAVConfig(), 1)
        u2 = UAV(1, UAVConfig(), 1)
        u1.reset(np.array([0., 0., 100.]))
        u2.reset(np.array([30., 0., 100.]))  # dist=30: between 10 and 50
        assert not d.check_pairwise(u1, u2)  # not a collision
        nm = d.check_near_misses([u1, u2])
        assert len(nm) == 1  # but is a near-miss

    def test_no_near_miss_beyond_threshold(self):
        """Distance >= near_miss_threshold should be neither collision nor near-miss."""
        d = ConflictDetector(10.0, near_miss_threshold=50.0)
        u1 = UAV(0, UAVConfig(), 1)
        u2 = UAV(1, UAVConfig(), 1)
        u1.reset(np.array([0., 0., 100.]))
        u2.reset(np.array([60., 0., 100.]))  # dist=60: beyond both thresholds
        assert not d.check_pairwise(u1, u2)
        nm = d.check_near_misses([u1, u2])
        assert len(nm) == 0

    def test_near_miss_excludes_collision(self):
        """Collision pairs should not also appear in near-misses."""
        d = ConflictDetector(10.0, near_miss_threshold=50.0)
        u1 = UAV(0, UAVConfig(), 1)
        u2 = UAV(1, UAVConfig(), 1)
        u1.reset(np.array([0., 0., 100.]))
        u2.reset(np.array([5., 0., 100.]))  # dist=5: collision
        assert d.check_pairwise(u1, u2)
        nm = d.check_near_misses([u1, u2])
        assert len(nm) == 0  # collision excluded from near-misses


class TestPriorityEngine:
    def test_sorting_descending(self):
        engine = PriorityEngine()
        cfg = UAVConfig()
        u1 = UAV(1, cfg, base_priority=1)
        u2 = UAV(2, cfg, base_priority=5)
        u1.reset(np.array([0., 0., 100.]))
        u2.reset(np.array([0., 0., 100.]))
        sorted_uavs = engine.get_sorting_key([u1, u2])
        assert sorted_uavs[0].id == 2

    def test_violation_detection(self):
        engine = PriorityEngine()
        cfg = UAVConfig()
        u1 = UAV(1, cfg, base_priority=5)
        u2 = UAV(2, cfg, base_priority=1)
        u1.reset(np.array([0., 0., 100.]))
        u2.reset(np.array([0., 0., 100.]))
        u2.status = UAVStatus.LANDED
        violations = engine.get_priority_violations([u1, u2])
        assert violations > 0


class TestMetricsCollector:
    def test_episode_summary(self):
        m = MetricsCollector()
        for _ in range(10):
            m.step(total_energy=5.0, avg_wait=10.0, num_conflicts=0,
                   priority_violations=0.0, landed=0, failed=0)
        summary = m.end_episode()
        assert summary.step_count == 10
        assert abs(summary.total_energy - 50.0) < 1e-6


class TestTerminalAreaEnv:
    def test_reset(self):
        config = EnvConfig()
        config.num_uavs = 5
        env = TerminalAreaEnv(config)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert len(env.uav_manager.uavs) == 5

    def test_step_runs(self):
        config = EnvConfig()
        config.num_uavs = 5
        config.sim.max_steps = 50
        env = TerminalAreaEnv(config)
        env.reset(seed=42)

        for _ in range(10):
            n = env.max_n
            m = config.sim.num_landing_zones
            action = {
                "high_level": np.array(
                    [4000., 2500., 1000., 800., 500., 200., 25., 20., 15.]
                ),
                "low_level": {
                    "control": np.zeros(n, dtype=np.int32),
                    "psi": np.zeros(n, dtype=np.float64),
                    "gamma": np.zeros(n, dtype=np.float64),
                    "speed": np.full(n, 25.0, dtype=np.float64),
                    "landing_zone": np.full(n, m, dtype=np.int32),
                },
            }
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, float)
            assert not terminated
            assert not truncated

    def test_high_level_action_applied(self):
        config = EnvConfig()
        config.num_uavs = 5
        env = TerminalAreaEnv(config)
        env.reset(seed=42)

        new_radii = np.array([3500., 2200., 800.])
        n = env.max_n
        m = config.sim.num_landing_zones
        action = {
            "high_level": np.array(
                [3500., 2200., 800., 750., 450., 150., 22., 17., 12.]
            ),
            "low_level": {
                "control": np.zeros(n, dtype=np.int32),
                "psi": np.zeros(n, dtype=np.float64),
                "gamma": np.zeros(n, dtype=np.float64),
                "speed": np.full(n, 25.0, dtype=np.float64),
                "landing_zone": np.full(n, m, dtype=np.int32),
            },
        }
        env.step(action)
        assert np.allclose(env.airspace.radii, new_radii)

    def test_observation_structure(self):
        """Verify local observation has correct structure."""
        config = EnvConfig()
        config.num_uavs = 3
        config.k_nearest = 5
        env = TerminalAreaEnv(config)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (config.max_uavs, env._local_obs_dim)

        # Active UAV rows should have non-zero self features
        # Row 0: sin(θ) or cos(θ) should be non-zero for active UAV
        assert not (abs(obs[0, 9]) < 1e-9 and abs(obs[0, 10]) < 1e-9)

    def test_collision_terminates_episode(self):
        """A collision (dist < safe_separation) must trigger immediate termination."""
        config = EnvConfig()
        config.num_uavs = 2
        config.uav.safe_separation = 10.0
        config.sim.max_steps = 100
        env = TerminalAreaEnv(config)
        env.reset(seed=42)

        # Force two UAVs to collide by setting positions, ring_ids, and angles consistently
        uavs = env.uav_manager.uavs
        for u in uavs:
            u.ring_id = 1  # both on ring 1 (H=800)
        uavs[0].position = np.array([4000., 0., 800.])  # on ring 1 orbit, angle=0
        uavs[0]._angle = 0.0
        uavs[1].position = np.array([4005., 0., 800.])  # 5m apart, angle=0
        uavs[1]._angle = 0.0

        n = env.max_n
        m = config.sim.num_landing_zones
        action = {
            "high_level": np.array([4000., 2500., 1000., 800., 500., 200., 25., 20., 15.]),
            "low_level": {
                "control": np.zeros(n, dtype=np.int32),
                "psi": np.zeros(n, dtype=np.float64),
                "gamma": np.zeros(n, dtype=np.float64),
                "speed": np.full(n, 25.0, dtype=np.float64),
                "landing_zone": np.full(n, m, dtype=np.int32),
            },
        }
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated, "Episode must terminate on collision"
        assert reward < -100, f"Collision reward should be heavily negative, got {reward}"

    def test_inflow_disabled_by_default(self):
        config = EnvConfig()
        config.num_uavs = 3
        config.uav_inflow_rate = 0.0
        config.sim.max_steps = 50
        env = TerminalAreaEnv(config)
        env.reset(seed=42)
        initial_count = len(env.uav_manager.uavs)
        # Run a few steps
        n = env.max_n
        m = config.sim.num_landing_zones
        for _ in range(5):
            action = {
                "high_level": np.array(
                    [4000., 2500., 1000., 800., 500., 200., 25., 20., 15.]
                ),
                "low_level": {
                    "control": np.zeros(n, dtype=np.int32),
                    "psi": np.zeros(n, dtype=np.float64),
                    "gamma": np.zeros(n, dtype=np.float64),
                    "speed": np.full(n, 25.0, dtype=np.float64),
                    "landing_zone": np.full(n, m, dtype=np.int32),
                },
            }
            env.step(action)
        # With no inflow and all CIRCLE, count should not increase beyond initial
        assert len(env.uav_manager.uavs) <= initial_count
