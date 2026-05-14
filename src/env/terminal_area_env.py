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
    """Terminal area landing environment with per-UAV local observations.

    High-level action (every N steps): airspace config [R1,R2,R3, H1,H2,H3, V1,V2,V3]
    Low-level action (every step): per-UAV control (CIRCLE/DESCEND) + (psi, gamma, V) + landing_zone.
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
        )

        # Landing zones: fixed positions on ground within inner ring
        self.landing_zones = self._init_landing_zones()

        # Per-UAV local observation dimension
        self.k = self.config.k_nearest
        self.max_n = self.config.max_uavs
        self._local_obs_dim = 9 + 9 + self.k * 6 + 3  # airspace + self(8+can_descend) + neighbors + ring_counts

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
        n = self.max_n

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
            "control": spaces.MultiDiscrete([2] * n),         # 0=CIRCLE, 1=DESCEND
            "psi": spaces.Box(0.0, 2 * np.pi, shape=(n,), dtype=np.float64),
            "gamma": spaces.Box(-np.pi / 2, np.pi / 2, shape=(n,), dtype=np.float64),
            "speed": spaces.Box(5.0, 40.0, shape=(n,), dtype=np.float64),
            "landing_zone": spaces.MultiDiscrete([m + 1] * n),  # 0..m-1 for zone, m = no-op
        })

        self.action_space = spaces.Dict({
            "high_level": self.high_level_action_space,
            "low_level": self.low_level_action_space,
        })

        # Per-UAV local observation matrix
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(n, self._local_obs_dim), dtype=np.float64
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.airspace = RingAirspace(self.config.ring)
        initial_uavs = self.config.num_uavs
        if options and "initial_uavs" in options:
            initial_uavs = options["initial_uavs"]

        self.uav_manager = UAVManager(
            self.config.uav,
            self.config.num_uavs,
            self.airspace,
            max_uavs=self.config.max_uavs,
            uav_inflow_rate=self.config.uav_inflow_rate,
            rng=self.rng,
            initial_uavs=initial_uavs,
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

        # Remove landed/failed UAVs
        self.uav_manager.remove_landed()

        # Inflow: spawn new UAVs according to rate
        self.uav_manager.try_spawn_inflow(dt)

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
        """Convert batched low-level action arrays to per-UAV LowLevelAction dict.

        Non-controlled UAVs always get CIRCLE (control=0).
        """
        actions: dict[int, LowLevelAction] = {}
        m = self.config.sim.num_landing_zones
        for i, uav in enumerate(self.uav_manager.uavs):
            if uav.is_controlled:
                control = int(ll["control"][i]) if "control" in ll else 0
                # Enforce ring-wait constraint: force CIRCLE if not allowed to descend
                if control == 1 and not uav.can_descend():
                    control = 0
                psi = float(ll["psi"][i]) if "psi" in ll else 0.0
                gamma = float(ll["gamma"][i]) if "gamma" in ll else 0.0
                speed = float(ll["speed"][i]) if "speed" in ll else self.config.uav.descent_speed
                lz_raw = int(ll["landing_zone"][i]) if "landing_zone" in ll else m
                landing_zone = lz_raw if lz_raw < m else -1
            else:
                # Background UAV: always CIRCLE
                control = 0
                psi, gamma, speed, landing_zone = 0.0, 0.0, 0.0, -1
            actions[uav.id] = LowLevelAction(
                control=control, psi=psi, gamma=gamma, speed=speed,
                landing_zone=landing_zone,
            )
        return actions

    def _compute_reward(self) -> float:
        cfg = self.config
        uavs = self.uav_manager.uavs
        controlled = [u for u in uavs if u.is_controlled]
        n_ctrl = max(len(controlled), 1)
        dt = cfg.sim.dt_low

        power_map = {
            UAVStatus.CRUISING: cfg.uav.power_cruise,
            UAVStatus.DESCENDING: cfg.uav.power_descend,
        }
        step_power = sum(power_map.get(u.status, cfg.uav.power_cruise) * dt for u in controlled)
        R_energy = -cfg.w_energy * step_power / n_ctrl

        wait_ctrl = sum(u.wait_time for u in controlled) / n_ctrl
        R_wait = -cfg.w_wait * wait_ctrl / 3600.0

        new_landings = self.uav_manager.landed_count - self._prev_landed
        self._prev_landed = self.uav_manager.landed_count
        R_throughput = cfg.w_throughput * 20.0 * new_landings / max(cfg.num_uavs, 1)

        conflicts = self.conflict_detector.check_all(uavs)
        R_safety = -cfg.w_safety * len(conflicts) * cfg.penalty_conflict

        violations = self.priority_engine.get_priority_violations(uavs)
        R_priority = -cfg.w_priority * violations

        dead = sum(1 for u in uavs if u.is_dead)
        R_battery = -cfg.w_battery * dead

        R_progress = 0.0
        R1 = cfg.ring.default_radii[0]
        H1 = cfg.ring.default_heights[0]
        for u in controlled:
            if u.status == UAVStatus.DESCENDING:
                r = np.linalg.norm(u.position[:2])
                z = u.position[2]
                radial_progress = 1.0 - min(r / R1, 1.0)
                vertical_progress = 1.0 - min(z / H1, 1.0)
                R_progress += 2.0 * (radial_progress + vertical_progress)
        R_progress /= n_ctrl

        return (R_energy + R_wait + R_throughput + R_safety
                + R_priority + R_battery + R_progress)

    def _get_obs(self) -> np.ndarray:
        """Build per-UAV local observation matrix of shape (max_n, local_obs_dim)."""
        obs = np.zeros((self.max_n, self._local_obs_dim), dtype=np.float64)
        active = self.uav_manager.active_uavs
        if not active:
            return obs

        # Global airspace state (normalized)
        asv = self.airspace.state_vector  # [R1,R2,R3, H1,H2,H3, V1,V2,V3]
        asv_norm = np.array([
            asv[0] / 5000.0, asv[1] / 5000.0, asv[2] / 5000.0,
            asv[3] / 1000.0, asv[4] / 1000.0, asv[5] / 1000.0,
            asv[6] / 30.0, asv[7] / 30.0, asv[8] / 30.0,
        ])

        R1 = self.airspace.radii[0]
        H1 = self.airspace.heights[0]
        c1, c2, c3 = self.uav_manager.ring_counts
        total_active = len(active)

        # Pre-compute per-UAV features for neighbor distance calculation
        all_features = []
        for u in active:
            r = np.linalg.norm(u.position[:2])
            z = u.position[2]
            all_features.append({
                "uav": u,
                "pos": u.position.copy(),
                "r": r,
                "z": z,
            })

        for i, feat in enumerate(all_features):
            if i >= self.max_n:
                break
            u = feat["uav"]
            pos = feat["pos"]
            r = feat["r"]
            z = feat["z"]

            # Self features: [sin(θ), cos(θ), ring_id/3, r/R1, z/H1, battery/max, priority/5, wait/3600, can_descend]
            theta = u.angle
            self_feat = np.array([
                np.sin(theta), np.cos(theta),
                u.ring_id / 3.0,
                min(r / R1, 1.0),
                min(z / H1, 1.0),
                u.battery / u.config.max_battery,
                u.base_priority / 5.0,
                min(u.wait_time / 3600.0, 1.0),
                float(u.can_descend()),
            ])

            # Find k nearest neighbors (by 3D distance)
            distances = []
            for j, other_feat in enumerate(all_features):
                if j == i or j >= self.max_n:
                    continue
                other_u = other_feat["uav"]
                if other_u.status == UAVStatus.LANDED:
                    continue
                dist = np.linalg.norm(pos - other_feat["pos"])
                distances.append((dist, other_feat))

            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k]

            # Neighbor features: [sin(θ), cos(θ), ring_id/3, r/R1, z/H1, battery/max] each
            neigh_feats = np.zeros(self.k * 6, dtype=np.float64)
            for j, (_, nf) in enumerate(neighbors):
                nu = nf["uav"]
                nr = nf["r"]
                nz = nf["z"]
                ntheta = nu.angle
                offset = j * 6
                neigh_feats[offset:offset + 6] = [
                    np.sin(ntheta), np.cos(ntheta),
                    nu.ring_id / 3.0,
                    min(nr / R1, 1.0),
                    min(nz / H1, 1.0),
                    nu.battery / nu.config.max_battery,
                ]

            # Ring counts (normalized)
            ring_feat = np.array([
                c1 / max(total_active, 1),
                c2 / max(total_active, 1),
                c3 / max(total_active, 1),
            ])

            # Assemble row
            obs[i, :9] = asv_norm
            obs[i, 9:18] = self_feat
            obs[i, 18:18 + self.k * 6] = neigh_feats
            obs[i, 18 + self.k * 6:] = ring_feat

        return obs

    def _get_info(self) -> dict:
        return {
            "step": self.step_count,
            "landed": self.uav_manager.landed_count,
            "failed": self.uav_manager.failed_count,
            "active": len(self.uav_manager.active_uavs),
            "total_active": len(self.uav_manager.uavs),
            "conflicts": len(self.conflict_detector.check_all(self.uav_manager.uavs)),
            "avg_wait": self.uav_manager.avg_wait_time,
        }

    def render(self) -> np.ndarray | None:
        return None

    def get_episode_metrics(self) -> EpisodeMetrics:
        return self.metrics.end_episode()
