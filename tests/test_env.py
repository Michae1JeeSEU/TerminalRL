"""Smoke tests for the terminal area environment modules."""

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
            np.array([1000., 2000., 3000.]),  # R1 < R2 < R3 (wrong)
            np.array([600., 400., 200.]),
            np.array([20., 15., 10.]),
        )
        assert not ok

    def test_invalid_height_ordering_rejected(self):
        ok = self.airspace.update(
            np.array([3000., 2000., 1000.]),
            np.array([200., 400., 600.]),  # H1 < H2 < H3 (wrong)
            np.array([20., 15., 10.]),
        )
        assert not ok

    def test_get_ring(self):
        # Inside outer ring boundary
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
        self.uav.step(LowLevelAction(control=0), ring_speed=25.0, ring_radius=4000.0, dt=2.0)
        assert not np.allclose(self.uav.position, initial_pos)
        assert self.uav.status == UAVStatus.CRUISING

    def test_hover_no_motion(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        initial_pos = self.uav.position.copy()
        self.uav.step(LowLevelAction(control=1), ring_speed=25.0, ring_radius=4000.0, dt=2.0)
        assert np.allclose(self.uav.position, initial_pos)
        assert self.uav.status == UAVStatus.HOLDING

    def test_descend_decreases_height(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        initial_z = self.uav.position[2]
        self.uav.step(LowLevelAction(control=2), ring_speed=0., ring_radius=4000.0, dt=2.0)
        assert self.uav.position[2] < initial_z
        assert self.uav.status == UAVStatus.DESCENDING

    def test_energy_consumption(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        self.uav.step(LowLevelAction(control=0), ring_speed=25.0, ring_radius=4000.0, dt=10.0)
        assert self.uav.energy_consumed > 0
        assert self.uav.battery < self.config.max_battery

    def test_battery_critical(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        self.uav.battery = self.config.battery_critical * self.config.max_battery * 0.5
        assert self.uav.is_critical_battery

    def test_landing_transition(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        self.uav.step(LowLevelAction(control=2, landing_zone=0),
                       ring_speed=0., ring_radius=4000.0, dt=2.0)
        assert self.uav.status == UAVStatus.LANDED

    def test_priority_increases_with_wait(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        p1 = self.uav.compute_priority()
        self.uav.wait_time = 7200.0  # 2 hours
        p2 = self.uav.compute_priority()
        assert p2 > p1

    def test_priority_increases_with_low_battery(self):
        self.uav.reset(np.array([4000., 0., 800.]), ring_id=1)
        p1 = self.uav.compute_priority()
        self.uav.battery = self.config.battery_critical * self.config.max_battery * 0.3
        p2 = self.uav.compute_priority()
        assert p2 > p1


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


class TestPriorityEngine:
    def test_sorting_descending(self):
        engine = PriorityEngine()
        cfg = UAVConfig()
        u1 = UAV(1, cfg, base_priority=1)
        u2 = UAV(2, cfg, base_priority=5)
        u1.reset(np.array([0., 0., 100.]))
        u2.reset(np.array([0., 0., 100.]))
        sorted_uavs = engine.get_sorting_key([u1, u2])
        assert sorted_uavs[0].id == 2  # higher priority first

    def test_violation_detection(self):
        engine = PriorityEngine()
        cfg = UAVConfig()
        u1 = UAV(1, cfg, base_priority=5)
        u2 = UAV(2, cfg, base_priority=1)
        u1.reset(np.array([0., 0., 100.]))  # high priority, still waiting
        u2.reset(np.array([0., 0., 100.]))
        u2.status = UAVStatus.LANDED        # low priority, already landed
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
        obs, _ = env.reset(seed=42)

        for _ in range(10):
            action = {
                "high_level": np.array(
                    [4000., 2500., 1000., 800., 500., 200., 25., 20., 15.]
                ),
                "low_level": {
                    "control": np.zeros(5, dtype=np.int32),
                    "waypoints": np.zeros((5, 4, 3), dtype=np.float64),
                    "landing_zone": np.zeros(5, dtype=np.int32),
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
        obs, _ = env.reset(seed=42)

        new_radii = np.array([3500., 2200., 800.])
        action = {
            "high_level": np.array(
                [3500., 2200., 800., 750., 450., 150., 22., 17., 12.]
            ),
            "low_level": {
                "control": np.zeros(5, dtype=np.int32),
                "waypoints": np.zeros((5, 4, 3), dtype=np.float64),
                "landing_zone": np.zeros(5, dtype=np.int32),
            },
        }
        env.step(action)
        assert np.allclose(env.airspace.radii, new_radii)
