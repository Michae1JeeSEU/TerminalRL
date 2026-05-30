from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.config import EnvConfig, MISSION_TYPE_LZ_OCCUPANCY
from src.env.ring_airspace import RingAirspace
from src.env.uav import UAV, UAVStatus, LowLevelAction
from src.env.uav_manager import UAVManager
from src.env.conflict_detector import ConflictDetector
from src.env.priority_engine import PriorityEngine
from src.env.metrics_collector import MetricsCollector, EpisodeMetrics
from src.env.safety_controller import SafetyController


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
        self.conflict_detector = ConflictDetector(
            self.config.uav.safe_separation,
            self.config.uav.near_miss_threshold,
        )
        self.priority_engine = PriorityEngine()
        self.metrics = MetricsCollector()
        self.safety_controller = SafetyController(
            self.config.safety, self.airspace, self.conflict_detector
        )

        self.step_count = 0
        self._prev_landed = 0
        self._last_collision_events = 0
        self._last_descent_blocked_ids: set[int] = set()
        self.high_level_interval = int(
            self.config.sim.dt_high / self.config.sim.dt_low
        )
        self._last_descent_release: dict[int, float] = {}

        # Landing zones: fixed positions on ground within inner ring
        self.landing_zones = self._init_landing_zones()
        self.lz_occupied_until = np.zeros(self.config.sim.num_landing_zones, dtype=np.float64)
        self._lz_reserved_by: dict[int, int] = {}  # lz_idx -> uav_id (active landing approach)
        self._max_lz_occ = max(MISSION_TYPE_LZ_OCCUPANCY.values()) if MISSION_TYPE_LZ_OCCUPANCY else 120.0

        # Per-UAV local observation dimension
        self.k = self.config.k_nearest
        self.max_n = self.config.max_uavs
        self._local_obs_dim = (9 + 9 + self.k * 6 + 3 + self.config.sim.num_landing_zones
                                + 10)  # +10 safety context/intervention features

        self._define_spaces()
        self._reset_descent_release_state()

    def _reset_descent_release_state(self) -> None:
        """Reset per-corridor release timers for rule-based descent clearances."""
        self._last_descent_release = {2: -1e9, 3: -1e9, 4: -1e9}

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
        self.safety_controller = SafetyController(
            self.config.safety, self.airspace, self.conflict_detector
        )
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
        self._last_collision_events = 0
        self._last_descent_blocked_ids = set()
        self._reset_descent_release_state()
        self.lz_occupied_until = np.zeros(self.config.sim.num_landing_zones, dtype=np.float64)
        self._lz_reserved_by = {}

        return self._get_obs(), self._get_info()

    def step(self, action: dict) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        dt = self.config.sim.dt_low
        self._last_collision_events = 0
        self._last_descent_blocked_ids = set()

        # Apply high-level action at the start of each option interval
        if self.step_count % self.high_level_interval == 1:
            hl = action.get("high_level")
            if hl is not None:
                radii = np.clip(hl[:3], self.config.ring.r_min, self.config.ring.r_max)
                heights = np.clip(hl[3:6], self.config.ring.h_min, self.config.ring.h_max)
                speeds = np.clip(hl[6:9], self.config.ring.v_min, self.config.ring.v_max)
                self.airspace.update(radii, heights, speeds)

        pre_collision_penalty = self._mark_collisions_failed()

        # Apply low-level actions
        ll = action.get("low_level", {})
        uav_actions = self._parse_low_level_actions(ll)
        sc = self.safety_controller if self.config.safety_enabled else None
        self.uav_manager.step_all(uav_actions, dt, sc)

        # Update LZ occupancy timers
        self.lz_occupied_until = np.maximum(0.0, self.lz_occupied_until - dt)

        # Clean up reservations for UAVs that are no longer actively descending to LZ
        active_uav_ids = {u.id for u in self.uav_manager.uavs}
        freed_lzs = [lz for lz, uid in self._lz_reserved_by.items() if uid not in active_uav_ids]
        for lz in freed_lzs:
            del self._lz_reserved_by[lz]
        # Also free reservations for UAVs that stopped descending toward their target
        for uav in self.uav_manager.uavs:
            if uav._landing_target is None and uav.landing_zone >= 0:
                if uav.landing_zone in self._lz_reserved_by:
                    if self._lz_reserved_by[uav.landing_zone] == uav.id:
                        del self._lz_reserved_by[uav.landing_zone]
            # Free reservation for landed UAVs (occupancy timer replaces reservation)
            if uav.status == UAVStatus.LANDED and uav.landing_zone >= 0:
                if uav.landing_zone in self._lz_reserved_by:
                    del self._lz_reserved_by[uav.landing_zone]
                occ_time = MISSION_TYPE_LZ_OCCUPANCY.get(uav.mission_type, 30.0)
                self.lz_occupied_until[uav.landing_zone] = occ_time

        # Remove landed/failed UAVs
        self.uav_manager.remove_landed()

        # Inflow: spawn new UAVs according to rate
        self.uav_manager.try_spawn_inflow(dt)

        # Compute reward and metrics
        reward = self._compute_reward()

        # Collision: kill colliding UAVs, penalize, but DON'T terminate
        terminated = False
        collision_penalty = pre_collision_penalty + self._mark_collisions_failed()
        reward += collision_penalty
        truncated = self.step_count >= self.config.sim.max_steps

        self.metrics.step(
            total_energy=sum(u.energy_consumed for u in self.uav_manager.uavs),
            avg_wait=self.uav_manager.avg_wait_time,
            num_conflicts=self._last_collision_events
            + len(self.conflict_detector.check_all(self.uav_manager.uavs)),
            priority_violations=self.priority_engine.get_priority_violations(
                self.uav_manager.uavs
            ),
            landed=self.uav_manager.landed_count,
            failed=self.uav_manager.failed_count,
        )

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _mark_collisions_failed(self) -> float:
        """Mark currently colliding UAVs as failed and return an event penalty."""
        collisions = self.conflict_detector.check_all(self.uav_manager.uavs)
        if len(collisions) > 0:
            self._last_collision_events += len(collisions)
            collision_ids = set()
            for id1, id2 in collisions:
                collision_ids.add(id1)
                collision_ids.add(id2)
            newly_failed = 0
            for u in self.uav_manager.uavs:
                if u.id in collision_ids:
                    if not u.failure_counted:
                        newly_failed += 1
                        u.failure_counted = True
                    u.battery = 0  # mark as dead (removed next remove_landed)
                    # Release LZ reservation held by colliding UAV
                    for lz, uid in list(self._lz_reserved_by.items()):
                        if uid == u.id:
                            del self._lz_reserved_by[lz]
            # Immediate penalty: event + per-UAV
            collision_penalty = -(self.config.penalty_conflict + self.config.penalty_conflict * 0.5 * len(collision_ids))
            self.uav_manager.failed_count += newly_failed
            return collision_penalty
        return 0.0

    def _parse_low_level_actions(self, ll: dict) -> dict[int, LowLevelAction]:
        """Convert batched low-level action arrays to per-UAV LowLevelAction dict.

        Non-controlled UAVs always get CIRCLE (control=0).
        The policy may request DESCEND, but the environment owns the actual
        descent guidance and can reject requests that violate hard spacing.
        """
        actions: dict[int, LowLevelAction] = {}
        m = self.config.sim.num_landing_zones
        for i, uav in enumerate(self.uav_manager.uavs):
            if uav.is_dead:
                actions[uav.id] = LowLevelAction(control=0, speed=1.0)
                continue
            if uav.is_controlled:
                control = int(ll["control"][i]) if "control" in ll else 0
                speed = float(ll["speed"][i]) if "speed" in ll else 1.0
                lz_raw = int(ll["landing_zone"][i]) if "landing_zone" in ll else m
                landing_zone = lz_raw if lz_raw < m else -1

                # Once a UAV has started a descent, let the rule controller finish
                # that segment. If it has reached the next ring, snap it into that
                # ring's orbit and resume metered circling.
                if uav.status == UAVStatus.DESCENDING and uav._landing_target is None:
                    reached_ring = self._reached_transition_ring(uav)
                    if reached_ring is not None:
                        uav.ring_id = reached_ring
                        ring_r, ring_h = self.airspace.get_ring_orbit(uav.ring_id)
                        self._safe_orbit_snap(uav, ring_r, ring_h)
                        uav._prev_ring_id = uav.ring_id
                        control = 0
                    elif uav.ring_id > uav._prev_ring_id:
                        ring_r, ring_h = self.airspace.get_ring_orbit(uav.ring_id)
                        self._safe_orbit_snap(uav, ring_r, ring_h)
                        uav._prev_ring_id = uav.ring_id
                        control = 0
                    else:
                        control = 1

                if uav.status == UAVStatus.DESCENDING and uav._landing_target is not None:
                    control = 1

                # Enforce ring-wait and capacity before a new descent clearance.
                if control == 1 and uav.status != UAVStatus.DESCENDING and not uav.can_descend():
                    control = 0
                if (control == 1 and uav.status != UAVStatus.DESCENDING
                        and uav.ring_id < 3
                        and not uav.is_critical_battery
                        and uav.base_priority < self.config.uav.priority_emergency):
                    if not self._ring_has_capacity(uav.ring_id + 1, exempt_uav_id=uav.id):
                        control = 0

                if control == 1 and uav.ring_id == 3 and uav._landing_target is None:
                    landing_zone = self._select_landing_zone(uav, landing_zone)
                    if landing_zone < 0:
                        control = 0
                    else:
                        uav.landing_zone = landing_zone
                        uav._landing_target = self.landing_zones[landing_zone].copy()
                        self._lz_reserved_by[landing_zone] = uav.id

                if control == 0:
                    ring_r, ring_h = self.airspace.get_ring_orbit(uav.ring_id)
                    if uav.position[2] < ring_h or abs(np.linalg.norm(uav.position[:2]) - ring_r) > 1.0:
                        self._safe_orbit_snap(uav, ring_r, ring_h)
                    speed = float(np.clip(speed, 0.3, 1.3))
                    actions[uav.id] = LowLevelAction(control=0, speed=speed, landing_zone=-1)
                    continue

                actions[uav.id] = self._baseline_descent_action(uav, landing_zone)
            else:
                # Background UAV: always CIRCLE at neutral speed
                control = 0
                actions[uav.id] = LowLevelAction(control=control, speed=1.0)

        self._apply_descent_clearances(actions)
        self._apply_same_ring_spacing(actions)
        return actions

    def _reached_transition_ring(self, uav: UAV) -> int | None:
        """Detect arrival at the next ring before geometry classification flips."""
        if uav.ring_id >= 3:
            return None
        target_ring = min(uav.ring_id + 1, 3)
        target_r, target_h = self.airspace.get_ring_orbit(target_ring)
        r = float(np.linalg.norm(uav.position[:2]))
        radius_tol = max(25.0, self.config.uav.near_miss_threshold)
        height_tol = 30.0
        if r <= target_r + radius_tol and uav.position[2] <= target_h + height_tol:
            return target_ring
        return None

    def _select_landing_zone(self, uav: UAV, requested_lz: int) -> int:
        """Return an available LZ, preferring the requested or nearest free zone."""
        candidates: list[int] = []
        if 0 <= requested_lz < self.config.sim.num_landing_zones:
            candidates.append(requested_lz)
        dists = [
            (float(np.linalg.norm(uav.position[:2] - lz[:2])), idx)
            for idx, lz in enumerate(self.landing_zones)
        ]
        candidates.extend(idx for _dist, idx in sorted(dists))

        seen: set[int] = set()
        for lz in candidates:
            if lz in seen:
                continue
            seen.add(lz)
            blocked = (
                self.lz_occupied_until[lz] > 0 or
                (lz in self._lz_reserved_by and self._lz_reserved_by[lz] != uav.id)
            )
            if not blocked:
                return lz
        return -1

    def _baseline_descent_action(self, uav: UAV, landing_zone: int) -> LowLevelAction:
        """Generate a rule-based descent command; RL no longer controls ψ/γ/V."""
        if uav.ring_id == 3:
            target = uav._landing_target
            if target is None:
                if landing_zone < 0:
                    return LowLevelAction(control=0, speed=1.0)
                target = self.landing_zones[landing_zone]
        else:
            target_ring = min(uav.ring_id + 1, 3)
            target_r, target_h = self.airspace.get_ring_orbit(target_ring)
            lead_angle = max(0.08, self.config.uav.min_ring_spacing / max(target_r, 1.0))
            theta = uav.angle + min(lead_angle, 0.25)
            target = np.array([
                target_r * np.cos(theta),
                target_r * np.sin(theta),
                target_h,
            ], dtype=np.float64)
            landing_zone = -1

        delta = target - uav.position
        horizontal = float(np.linalg.norm(delta[:2]))
        psi = float(np.arctan2(delta[1], delta[0]))
        gamma = float(np.arctan2(delta[2], max(horizontal, 1e-6)))
        gamma = float(np.clip(gamma, -np.deg2rad(35.0), -np.deg2rad(5.0)))
        speed = float(min(self.config.uav.descent_speed, self.config.uav.max_speed))
        return LowLevelAction(
            control=1, psi=psi, gamma=gamma, speed=speed,
            landing_zone=landing_zone,
        )

    def _descent_target_key(self, uav: UAV, action: LowLevelAction) -> int:
        """2/3 represent ring transitions; 4 represents the final/LZ corridor."""
        if uav.ring_id >= 3 or action.landing_zone >= 0 or uav._landing_target is not None:
            return 4
        return min(uav.ring_id + 1, 3)

    def _descent_release_gap(self, target_key: int) -> float:
        """Minimum release interval per descent corridor."""
        if target_key == 4:
            return 22.0
        return 35.0

    def _apply_descent_clearances(self, actions: dict[int, LowLevelAction]) -> None:
        """Meter new descent requests and reject predictively unsafe ones."""
        active_targets: set[int] = set()
        for u in self.uav_manager.uavs:
            if u.status == UAVStatus.DESCENDING and not u.is_dead:
                action = actions.get(u.id, LowLevelAction(control=1))
                active_targets.add(self._descent_target_key(u, action))

        candidates: list[tuple[float, float, UAV, LowLevelAction]] = []
        for u in self.uav_manager.uavs:
            action = actions.get(u.id)
            if action is None or action.control != 1:
                continue
            if u.status == UAVStatus.DESCENDING:
                continue
            candidates.append((-u.compute_priority(), -u.wait_time, u, action))

        current_time = self.step_count * self.config.sim.dt_low
        for _priority, _wait, uav, action in sorted(candidates):
            target_key = self._descent_target_key(uav, action)
            if target_key in active_targets:
                self._block_descent(uav, actions)
                continue
            if current_time - self._last_descent_release[target_key] < self._descent_release_gap(target_key):
                self._block_descent(uav, actions)
                continue
            if not self._descent_is_predictively_safe(uav, action):
                self._block_descent(uav, actions)
                continue

            active_targets.add(target_key)
            self._last_descent_release[target_key] = current_time

    def _block_descent(self, uav: UAV, actions: dict[int, LowLevelAction]) -> None:
        """Convert a rejected descent request back to a safe CIRCLE action."""
        self._last_descent_blocked_ids.add(uav.id)
        for lz, uid in list(self._lz_reserved_by.items()):
            if uid == uav.id:
                del self._lz_reserved_by[lz]
        uav._landing_target = None
        uav.landing_zone = -1
        ring_r, ring_h = self.airspace.get_ring_orbit(uav.ring_id)
        if abs(np.linalg.norm(uav.position[:2]) - ring_r) > 1.0 or uav.position[2] < ring_h:
            self._safe_orbit_snap(uav, ring_r, ring_h)
        actions[uav.id] = LowLevelAction(control=0, speed=1.0)

    def _apply_same_ring_spacing(self, actions: dict[int, LowLevelAction]) -> None:
        """Hard longitudinal control on each ring before movement is applied."""
        for ring_id in (1, 2, 3):
            ring_uavs = [
                u for u in self.uav_manager.uavs
                if u.ring_id == ring_id and u.status == UAVStatus.CRUISING and not u.is_dead
            ]
            if len(ring_uavs) < 2:
                continue
            ring_uavs.sort(key=lambda u: u.angle % (2 * np.pi))
            ring_r = self.airspace.radii[ring_id - 1]
            target_gap = self.config.uav.min_ring_spacing / max(ring_r, 1e-6)
            hard_gap = self.config.uav.near_miss_threshold / max(ring_r, 1e-6)
            for idx, uav in enumerate(ring_uavs):
                action = actions.get(uav.id)
                if action is None or action.control != 0:
                    continue
                ahead = ring_uavs[(idx + 1) % len(ring_uavs)]
                gap = ((ahead.angle % (2 * np.pi)) - (uav.angle % (2 * np.pi))) % (2 * np.pi)
                if gap < hard_gap:
                    action.speed = min(max(action.speed, 0.3), 0.3)
                elif gap < target_gap:
                    ratio = gap / max(target_gap, 1e-6)
                    action.speed = min(max(action.speed, 0.3), max(0.3, 0.35 + 0.55 * ratio))

    def _descent_is_predictively_safe(self, uav: UAV, action: LowLevelAction) -> bool:
        """Reject a descent if straight-line prediction enters protected spacing."""
        horizon = 24.0
        dt = self.config.sim.dt_low
        protected = max(self.config.uav.near_miss_threshold, self.config.uav.min_ring_spacing)

        velocity = self._velocity_from_action(action)
        for t in np.arange(0.0, horizon + 1e-6, dt):
            cand_pos = uav.position + velocity * t
            for other in self.uav_manager.uavs:
                if other.id == uav.id or other.status == UAVStatus.LANDED or other.is_dead:
                    continue
                other_pos = self._predict_position(other, t)
                if float(np.linalg.norm(cand_pos - other_pos)) < protected:
                    return False
        return True

    def _velocity_from_action(self, action: LowLevelAction) -> np.ndarray:
        cos_gamma = np.cos(action.gamma)
        return np.array([
            action.speed * cos_gamma * np.cos(action.psi),
            action.speed * cos_gamma * np.sin(action.psi),
            action.speed * np.sin(action.gamma),
        ], dtype=np.float64)

    def _predict_position(self, uav: UAV, t: float) -> np.ndarray:
        if uav.status == UAVStatus.CRUISING and uav.ring_id > 0:
            ring_r, ring_h = self.airspace.get_ring_orbit(uav.ring_id)
            omega = self.airspace.speeds[uav.ring_id - 1] / max(ring_r, 1e-6)
            theta = uav.angle + omega * t
            return np.array([ring_r * np.cos(theta), ring_r * np.sin(theta), ring_h])
        return uav.position + uav.velocity * t

    def _ring_capacity(self, ring_id: int) -> int:
        """Maximum UAVs allowed on a ring based on ideal spacing."""
        ring_r = self.airspace.radii[ring_id - 1]
        circumference = 2 * np.pi * ring_r
        return max(1, int(circumference / self.config.uav.min_ring_spacing))

    def _ring_has_capacity(self, ring_id: int, exempt_uav_id: int = -1) -> bool:
        """Check if ring has capacity for one more UAV.

        Emergency UAVs always pass. Non-emergency UAVs blocked when ring
        is at >= 90% capacity.
        """
        # Count UAVs currently on the target ring (or descending toward it)
        count = 0
        for u in self.uav_manager.uavs:
            if u.id == exempt_uav_id:
                continue
            if u.status == UAVStatus.LANDED:
                continue
            if u.ring_id == ring_id:
                count += 1
            # Also count UAVs descending from the ring above that will arrive
            elif u.ring_id == ring_id - 1 and u.status == UAVStatus.DESCENDING:
                count += 1

        capacity = self._ring_capacity(ring_id)
        # Allow up to 90% capacity, above that only emergency
        return count < int(capacity * 0.9)

    def _safe_orbit_snap(self, uav: UAV, ring_r: float, ring_h: float) -> None:
        """Snap UAV to ring orbit, avoiding collisions with other UAVs on the same ring.

        Searches for a safe angular slot within ±30° of the UAV's current angle.
        If no safe slot exists, keeps the UAV at its current position.
        """
        safe_dist = self.config.uav.safe_separation * 3.0  # generous buffer
        angle = uav.angle

        # Get positions of other UAVs on the same ring
        same_ring_uavs = [
            other for other in self.uav_manager.uavs
            if other.id != uav.id
            and other.status != UAVStatus.LANDED
            and other.ring_id == uav.ring_id
        ]

        if not same_ring_uavs:
            # No conflicts — snap directly
            uav.position[2] = ring_h
            uav.position[0] = ring_r * np.cos(angle)
            uav.position[1] = ring_r * np.sin(angle)
            return

        # Check if current angle is safe
        test_pos = np.array([ring_r * np.cos(angle), ring_r * np.sin(angle), ring_h])
        if all(np.linalg.norm(test_pos - other.position) >= safe_dist
               for other in same_ring_uavs):
            uav.position[:] = test_pos
            return

        # Search for a safe offset angle
        for offset_deg in range(5, 35, 5):
            offset = np.deg2rad(offset_deg)
            for sign in [1.0, -1.0]:
                test_angle = angle + sign * offset
                test_pos = np.array([ring_r * np.cos(test_angle),
                                    ring_r * np.sin(test_angle), ring_h])
                if all(np.linalg.norm(test_pos - other.position) >= safe_dist
                       for other in same_ring_uavs):
                    uav.position[:] = test_pos
                    uav._angle = test_angle
                    return

        # No safe slot — keep at current position, snap only z to ring height
        # (prevents collision from sudden horizontal jump)
        uav.position[2] = ring_h

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
        R_throughput = cfg.w_throughput * cfg.reward_landing * new_landings

        conflicts = self.conflict_detector.check_all(uavs)
        R_safety = -cfg.w_safety * len(conflicts) * cfg.penalty_conflict

        near_misses = self.conflict_detector.check_near_misses(uavs)
        R_near_miss = (
            -cfg.w_safety * len(near_misses) * cfg.penalty_near_miss / n_ctrl
        )

        # Continuous spacing penalty starts before near-miss range, giving PPO
        # gradient before the reflexive safety layer needs to intervene.
        active_uavs = [u for u in uavs if u.status != UAVStatus.LANDED and not u.is_dead]
        margin_cost = self._safety_margin_cost(active_uavs)
        R_margin = -cfg.w_safety * cfg.uav.safety_margin_penalty_scale * margin_cost / n_ctrl

        # Legacy very-close proximity term remains as an extra sharp penalty.
        R_proximity = 0.0
        proximity_threshold = cfg.uav.proximity_penalty_threshold
        for i in range(len(active_uavs)):
            for j in range(i + 1, len(active_uavs)):
                dist = float(np.linalg.norm(active_uavs[i].position - active_uavs[j].position))
                if dist < proximity_threshold:
                    # Penalty scales as 1/distance — closer = worse
                    R_proximity -= cfg.uav.proximity_penalty_scale / max(dist, 0.1)

        safety_counts = self._safety_intervention_counts()
        intervention_cost = (
            cfg.penalty_safety_rule * safety_counts["active_rules"]
            + cfg.penalty_safety_speed * safety_counts["speed_reduction"]
            + cfg.penalty_safety_block * safety_counts["blocked"]
            + cfg.penalty_descent_block * len(self._last_descent_blocked_ids)
        )
        R_intervention = -cfg.w_safety * intervention_cost / n_ctrl

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
                + R_near_miss + R_margin + R_proximity + R_intervention
                + R_priority + R_battery + R_progress)

    def _safety_margin_cost(self, active_uavs: list[UAV]) -> float:
        """Quadratic pairwise spacing cost for distances below safety margin."""
        threshold = self.config.uav.safety_margin_threshold
        cost = 0.0
        for i in range(len(active_uavs)):
            for j in range(i + 1, len(active_uavs)):
                dist = float(np.linalg.norm(active_uavs[i].position - active_uavs[j].position))
                if dist < threshold:
                    slack = (threshold - dist) / max(threshold, 1e-6)
                    cost += slack * slack
        return cost

    def _safety_intervention_counts(self) -> dict[str, float]:
        """Summarize latest safety-layer interventions for reward/info."""
        active_rules = 0
        speed_reduction = 0.0
        blocked = 0
        for adj in self.uav_manager.last_safety_adjustments.values():
            active_rules += len(getattr(adj, "active_rules", []))
            speed_reduction += max(0.0, 1.0 - float(getattr(adj, "speed_mult", 1.0)))
            blocked += int(bool(getattr(adj, "blocked", False)))
        return {
            "active_rules": float(active_rules),
            "speed_reduction": float(speed_reduction),
            "blocked": float(blocked),
        }

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

            # LZ occupancy (normalized remaining time, 1.0 = occupied)
            lz_feat = np.minimum(self.lz_occupied_until / max(self._max_lz_occ, 1.0), 1.0)
            lz_start = 18 + self.k * 6 + 3
            lz_end = lz_start + self.config.sim.num_landing_zones
            safety_context = self._safety_context_features(u, pos, all_features)

            # Assemble row
            obs[i, :9] = asv_norm
            obs[i, 9:18] = self_feat
            obs[i, 18:18 + self.k * 6] = neigh_feats
            obs[i, 18 + self.k * 6:18 + self.k * 6 + 3] = ring_feat
            obs[i, lz_start:lz_end] = lz_feat
            obs[i, lz_end:lz_end + 6] = safety_context

            # Safety adjustment features (last 4 dims): applied TO this UAV
            safety_start = lz_end + 6
            obs[i, safety_start] = 1.0
            adj = self.uav_manager.last_safety_adjustments.get(u.id)
            if adj is not None:
                obs[i, safety_start] = adj.speed_mult
                obs[i, safety_start + 1] = adj.psi_offset / np.pi
                obs[i, safety_start + 2] = adj.gamma_offset / (np.pi / 2)
                obs[i, safety_start + 3] = float(adj.blocked)

        return obs

    def _safety_context_features(
        self,
        uav: UAV,
        pos: np.ndarray,
        all_features: list[dict],
    ) -> np.ndarray:
        """Direct spacing and intervention features for active safety learning.

        Layout:
        [nearest_3d, front_arc_gap, back_arc_gap, predicted_min_dist,
         last_safety_active, descent_blocked]
        """
        norm = max(self.config.uav.safety_margin_threshold, 1.0)
        nearest = norm
        for feat in all_features:
            other = feat["uav"]
            if other.id == uav.id or other.status == UAVStatus.LANDED or other.is_dead:
                continue
            nearest = min(nearest, float(np.linalg.norm(pos - feat["pos"])))

        front_gap, back_gap = self._same_ring_arc_gaps(uav)
        predicted = self._min_predicted_distance(uav, horizon=12.0)
        adj = self.uav_manager.last_safety_adjustments.get(uav.id)
        safety_active = 0.0
        if adj is not None:
            safety_active = float(
                bool(getattr(adj, "active_rules", []))
                or abs(float(getattr(adj, "speed_mult", 1.0)) - 1.0) > 1e-6
                or bool(getattr(adj, "blocked", False))
            )

        return np.array([
            min(nearest / norm, 1.0),
            min(front_gap / norm, 1.0),
            min(back_gap / norm, 1.0),
            min(predicted / norm, 1.0),
            safety_active,
            float(uav.id in self._last_descent_blocked_ids),
        ], dtype=np.float64)

    def _same_ring_arc_gaps(self, uav: UAV) -> tuple[float, float]:
        """Return forward/backward same-ring arc gaps in meters."""
        if uav.ring_id <= 0:
            inf_gap = self.config.uav.safety_margin_threshold
            return inf_gap, inf_gap
        same_ring = [
            other for other in self.uav_manager.active_uavs
            if other.id != uav.id
            and other.ring_id == uav.ring_id
            and other.status == UAVStatus.CRUISING
        ]
        if not same_ring:
            inf_gap = self.config.uav.safety_margin_threshold
            return inf_gap, inf_gap

        theta = uav.angle % (2 * np.pi)
        ring_r = self.airspace.radii[uav.ring_id - 1]
        forward = []
        backward = []
        for other in same_ring:
            other_theta = other.angle % (2 * np.pi)
            fwd = (other_theta - theta) % (2 * np.pi)
            back = (theta - other_theta) % (2 * np.pi)
            forward.append(fwd * ring_r)
            backward.append(back * ring_r)
        return min(forward), min(backward)

    def _min_predicted_distance(self, uav: UAV, horizon: float) -> float:
        """Minimum predicted distance to any other active UAV over a short horizon."""
        dt = self.config.sim.dt_low
        best = self.config.uav.safety_margin_threshold
        for t in np.arange(0.0, horizon + 1e-6, dt):
            own_pos = self._predict_position(uav, t)
            for other in self.uav_manager.active_uavs:
                if other.id == uav.id:
                    continue
                other_pos = self._predict_position(other, t)
                best = min(best, float(np.linalg.norm(own_pos - other_pos)))
        return best

    def _get_info(self) -> dict:
        return {
            "step": self.step_count,
            "landed": self.uav_manager.landed_count,
            "failed": self.uav_manager.failed_count,
            "active": len(self.uav_manager.active_uavs),
            "total_active": len(self.uav_manager.uavs),
            "conflicts": self._last_collision_events
            + len(self.conflict_detector.check_all(self.uav_manager.uavs)),
            "near_misses": len(self.conflict_detector.check_near_misses(self.uav_manager.uavs)),
            "avg_wait": self.uav_manager.avg_wait_time,
        }

    def render(self) -> np.ndarray | None:
        return None

    def get_episode_metrics(self) -> EpisodeMetrics:
        return self.metrics.end_episode()
