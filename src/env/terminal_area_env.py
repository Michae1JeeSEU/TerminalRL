from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.config import EnvConfig
from src.env.ring_airspace import RingAirspace
from src.env.uav import UAV, UAVStatus, LowLevelAction
from src.env.uav_manager import UAVManager
from src.env.conflict_detector import ConflictDetector
from src.env.priority_engine import PriorityEngine
from src.env.metrics_collector import MetricsCollector, EpisodeMetrics


class TerminalAreaEnv(gym.Env):
    """Terminal area landing environment with hierarchical actions.

    High-level action (every N steps): airspace config [R1,R2,R3, H1,H2,H3, V1,V2,V3]
    Low-level action (every step): per-UAV control + waypoints + landing zone.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()

        self.airspace = RingAirspace(self.config.ring)
        self.rng = np.random.default_rng()

        self.uav_manager: UAVManager = None  # type: ignore[assignment]
        self.conflict_detector = ConflictDetector(self.config.uav.safe_separation)
        self.priority_engine = PriorityEngine()
        self.metrics = MetricsCollector()

        self.step_count = 0
        self._prev_landed = 0
        self.high_level_interval = int(
            self.config.sim.dt_high / self.config.sim.dt_low
        )  # 900 low-level steps per high-level step

        # Landing zones: fixed positions on ground within inner ring
        self.landing_zones = self._init_landing_zones()

        self._define_spaces()

    def _init_landing_zones(self) -> np.ndarray:
        """Place M landing zones evenly within the inner ring."""
        m = self.config.sim.num_landing_zones
        inner_r = self.config.ring.default_radii[2] * 0.6
        angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
        zones = np.zeros((m, 3), dtype=np.float64)
        zones[:, 0] = inner_r * np.cos(angles)
        zones[:, 1] = inner_r * np.sin(angles)
        return zones

    def _define_spaces(self) -> None:
        m = self.config.sim.num_landing_zones
        k = self.config.sim.num_waypoints
        n = self.config.num_uavs

        self.high_level_action_space = spaces.Box(
            low=np.array([
                self.config.ring.r_min, self.config.ring.r_min, self.config.ring.r_min,
                self.config.ring.h_min, self.config.ring.h_min, self.config.ring.h_min,
                self.config.ring.v_min, self.config.ring.v_min, self.config.ring.v_min,
            ], dtype=np.float64),
            high=np.array([
                self.config.ring.r_max, self.config.ring.r_max, self.config.ring.r_max,
                self.config.ring.h_max, self.config.ring.h_max, self.config.ring.h_max,
                self.config.ring.v_max, self.config.ring.v_max, self.config.ring.v_max,
            ], dtype=np.float64),
        )

        self.low_level_action_space = spaces.Dict({
            "control": spaces.MultiDiscrete([3] * n),       # per-UAV: 0=CIRCLE, 1=HOVER, 2=DESCEND
            "waypoints": spaces.Box(-1.0, 1.0, shape=(n, k, 3), dtype=np.float64),
            "landing_zone": spaces.MultiDiscrete([m] * n),  # per-UAV landing zone index
        })

        # Unified action space for step()
        self.action_space = spaces.Dict({
            "high_level": self.high_level_action_space,
            "low_level": self.low_level_action_space,
        })

        # Observation: [airspace(9) + ring_counts(3) + per-UAV(5×N) + landing_zones(M×3)]
        obs_dim = 9 + 3 + n * 5 + m * 3
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float64
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.airspace = RingAirspace(self.config.ring)
        self.uav_manager = UAVManager(
            self.config.uav, self.config.num_uavs, self.airspace, self.rng
        )
        self.metrics = MetricsCollector()
        self.step_count = 0
        self._prev_landed = 0

        return self._get_obs(), self._get_info()

    def step(self, action: dict) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        dt = self.config.sim.dt_low

        # Apply high-level action at the start of each option interval
        if self.step_count % self.high_level_interval == 1:
            hl = action.get("high_level")
            if hl is not None:
                radii = np.clip(hl[:3], self.config.ring.r_min, self.config.ring.r_max)
                heights = np.clip(hl[3:6], self.config.ring.h_min, self.config.ring.h_max)
                speeds = np.clip(hl[6:9], self.config.ring.v_min, self.config.ring.v_max)
                self.airspace.update(radii, heights, speeds)

        # Apply low-level actions
        ll = action.get("low_level", {})
        uav_actions = self._parse_low_level_actions(ll)
        self.uav_manager.step_all(uav_actions, dt)

        # Remove landed/failed UAVs and spawn replacements
        self.uav_manager.remove_landed()

        # Compute reward and metrics
        reward = self._compute_reward()

        terminated = self.step_count >= self.config.sim.max_steps
        truncated = False

        self.metrics.step(
            total_energy=sum(u.energy_consumed for u in self.uav_manager.uavs),
            avg_wait=self.uav_manager.avg_wait_time,
            num_conflicts=len(self.conflict_detector.check_all(self.uav_manager.uavs)),
            priority_violations=self.priority_engine.get_priority_violations(
                self.uav_manager.uavs
            ),
            landed=self.uav_manager.landed_count,
            failed=self.uav_manager.failed_count,
        )

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _parse_low_level_actions(self, ll: dict) -> dict[int, LowLevelAction]:
        """Convert batched low-level action arrays to per-UAV LowLevelAction dict."""
        actions: dict[int, LowLevelAction] = {}
        n_uavs = len(self.uav_manager.uavs)
        for i, uav in enumerate(self.uav_manager.uavs):
            if i >= n_uavs:
                break
            control = int(ll["control"][i]) if "control" in ll else 0
            waypoints = None
            landing_zone = -1
            if control == 2 and "waypoints" in ll:
                wp = ll["waypoints"][i]
                waypoints = uav.position + wp * self.airspace.radii[0]
            if uav.ring_id == 3 and control == 2 and "landing_zone" in ll:
                landing_zone = int(ll["landing_zone"][i])
            actions[uav.id] = LowLevelAction(
                control=control, waypoints=waypoints, landing_zone=landing_zone
            )
        return actions

    def _compute_reward(self) -> float:
        cfg = self.config
        uavs = self.uav_manager.uavs
        dt = cfg.sim.dt_low
        n = max(len(uavs), 1)

        # Per-step power (incremental, not cumulative)
        power_map = {
            UAVStatus.CRUISING: cfg.uav.power_cruise,
            UAVStatus.HOLDING: cfg.uav.power_hover,
            UAVStatus.DESCENDING: cfg.uav.power_descend,
        }
        step_power = sum(power_map.get(u.status, 0) * dt for u in uavs)
        R_energy = -cfg.w_energy * step_power / n

        R_wait = -cfg.w_wait * self.uav_manager.avg_wait_time / 3600.0

        # Landing bonus: spike reward for each new landing this step
        new_landings = self.uav_manager.landed_count - self._prev_landed
        self._prev_landed = self.uav_manager.landed_count
        R_throughput = cfg.w_throughput * 20.0 * new_landings / max(cfg.num_uavs, 1)

        conflicts = self.conflict_detector.check_all(uavs)
        R_safety = -cfg.w_safety * len(conflicts) * cfg.penalty_conflict

        violations = self.priority_engine.get_priority_violations(uavs)
        R_priority = -cfg.w_priority * violations

        dead = sum(1 for u in uavs if u.is_dead)
        R_battery = -cfg.w_battery * dead

        # Reward shaping: reward DESCENDING UAVs for their progress position
        R_progress = 0.0
        R1 = cfg.ring.default_radii[0]
        H1 = cfg.ring.default_heights[0]
        for u in uavs:
            if u.status == UAVStatus.DESCENDING:
                r = np.linalg.norm(u.position[:2])
                z = u.position[2]
                radial_progress = 1.0 - min(r / R1, 1.0)
                vertical_progress = 1.0 - min(z / H1, 1.0)
                R_progress += 2.0 * (radial_progress + vertical_progress)
        R_progress /= n

        return R_energy + R_wait + R_throughput + R_safety + R_priority + R_battery + R_progress

    def _get_obs(self) -> np.ndarray:
        """Build flat observation vector."""
        n = self.config.num_uavs
        m = self.config.sim.num_landing_zones

        airspace_state = self.airspace.state_vector  # 9
        ring_counts = np.array(self.uav_manager.ring_counts, dtype=np.float64)  # 3

        # Per-UAV state: pad/truncate to config.num_uavs
        uav_states = np.zeros((n, 5), dtype=np.float64)
        state_matrix = self.uav_manager.get_state_matrix()  # active UAVs only
        active = min(len(self.uav_manager.active_uavs), n)
        if active > 0:
            uav_states[:active] = state_matrix[:active]

        # Landing zones
        zones = self.landing_zones.reshape(-1)  # M*3

        return np.concatenate([
            airspace_state,
            ring_counts,
            uav_states.reshape(-1),  # N*5
            zones,                    # M*3
        ])

    def _get_info(self) -> dict:
        return {
            "step": self.step_count,
            "landed": self.uav_manager.landed_count,
            "failed": self.uav_manager.failed_count,
            "active": len(self.uav_manager.active_uavs),
            "conflicts": len(self.conflict_detector.check_all(self.uav_manager.uavs)),
            "avg_wait": self.uav_manager.avg_wait_time,
        }

    def render(self) -> np.ndarray | None:
        """Return a simple text representation (can be extended for matplotlib)."""
        return None

    def get_episode_metrics(self) -> EpisodeMetrics:
        return self.metrics.end_episode()
