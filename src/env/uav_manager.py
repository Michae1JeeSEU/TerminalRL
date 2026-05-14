import numpy as np

from src.config import UAVConfig
from src.env.ring_airspace import RingAirspace
from src.env.uav import UAV, UAVStatus, LowLevelAction


class UAVManager:
    """Manages a fleet of UAVs: creation, batch stepping, state aggregation."""

    # Initial ring distribution proportions
    _INIT_RING_PROBS = (0.5, 0.3, 0.2)  # ring 1, 2, 3

    # Status distribution per ring for initial UAVs
    _INIT_STATUS_MAP = {
        1: {UAVStatus.CRUISING: 0.8, UAVStatus.HOLDING: 0.2},
        2: {UAVStatus.CRUISING: 0.4, UAVStatus.HOLDING: 0.3, UAVStatus.DESCENDING: 0.3},
        3: {UAVStatus.HOLDING: 0.4, UAVStatus.DESCENDING: 0.6},
    }

    def __init__(self, config: UAVConfig, num_uavs: int, airspace: RingAirspace,
                 rng: np.random.Generator | None = None):
        self.config = config
        self.airspace = airspace
        self.rng = rng or np.random.default_rng()
        self.uavs: list[UAV] = []
        self.landed_count = 0
        self.failed_count = 0
        self._next_id = 0

        # Distribute initial UAVs across rings with random states
        ring_ids = self.rng.choice([1, 2, 3], size=num_uavs, p=self._INIT_RING_PROBS)
        for ring_id in ring_ids:
            self._spawn_one(ring_id=int(ring_id))

    def _spawn_one(self, ring_id: int = 1, battery: float | None = None,
                   wait_time: float | None = None, status: UAVStatus | None = None) -> UAV:
        """Spawn a UAV. For initial distribution, pass ring_id and random state.
        For replacement spawns (default args), spawns on ring 1 boundary with random state."""
        base_priority = self.rng.choice([1, 3, 5], p=[0.5, 0.35, 0.15])
        uav = UAV(self._next_id, self.config, base_priority)
        self._next_id += 1

        if ring_id == 1 or battery is None:
            position = self.airspace.random_position_on_ring(ring_id, self.rng)
        else:
            position = self.airspace.random_position_in_ring(ring_id, self.rng)

        if battery is None:
            battery = self.rng.uniform(
                self.config.battery_critical * self.config.max_battery,
                self.config.max_battery,
            )
        if wait_time is None:
            wait_time = self.rng.uniform(0, 600)
        if status is None:
            statuses, probs = zip(*self._INIT_STATUS_MAP.get(ring_id, {UAVStatus.CRUISING: 1.0}).items())
            status = self.rng.choice(list(statuses), p=list(probs))

        uav.reset(position, ring_id=ring_id, battery=battery,
                  wait_time=wait_time, status=status)
        self.uavs.append(uav)
        return uav

    def remove_landed(self) -> None:
        """Remove landed UAVs and spawn replacements."""
        landed = [u for u in self.uavs if u.status == UAVStatus.LANDED]
        self.landed_count += len(landed)
        failed = [u for u in self.uavs if u.is_dead]
        self.failed_count += len(failed)
        remove_ids = {u.id for u in landed + failed}
        self.uavs = [u for u in self.uavs if u.id not in remove_ids]
        for _ in landed:
            self._spawn_one()

    def step_all(self, actions: dict[int, LowLevelAction], dt: float) -> None:
        """Apply per-UAV actions and advance all UAVs."""
        for uav in self.uavs:
            ring_id = self.airspace.get_ring(uav.position)
            if ring_id == 0:
                ring_id = uav.ring_id
            uav.ring_id = ring_id
            boundary = self.airspace.get_boundary(ring_id)
            action = actions.get(uav.id, LowLevelAction(control=0))
            if action.control == 2:  # DESCEND
                target_ring = min(ring_id + 1, 3)
                if (action.waypoints is not None and uav._waypoints is None
                        and not np.allclose(action.waypoints, uav.position, atol=1.0)):
                    uav.set_descent_path(action.waypoints, target_ring)
            uav.step(action, boundary["speed"], boundary["radius"], dt)
            new_ring = self.airspace.get_ring(uav.position)
            if new_ring != ring_id and new_ring > 0:
                uav.ring_id = new_ring
                uav._waypoints = None  # ring changed, allow new waypoints next step
                uav._waypoint_idx = 0

    @property
    def active_uavs(self) -> list[UAV]:
        return [u for u in self.uavs if u.status != UAVStatus.LANDED]

    def get_state_matrix(self) -> np.ndarray:
        """Return (N, 5) matrix: [x, y, z, battery, base_priority] for all active UAVs."""
        active = self.active_uavs
        if not active:
            return np.empty((0, 5))
        data = []
        for u in active:
            data.append([u.position[0], u.position[1], u.position[2],
                         u.battery, u.base_priority])
        return np.array(data, dtype=np.float64)

    def uavs_by_ring(self, ring_id: int) -> list[UAV]:
        return [u for u in self.active_uavs
                if self.airspace.get_ring(u.position) == ring_id]

    @property
    def ring_counts(self) -> tuple[int, int, int]:
        c1 = len(self.uavs_by_ring(1))
        c2 = len(self.uavs_by_ring(2))
        c3 = len(self.uavs_by_ring(3))
        return c1, c2, c3

    @property
    def avg_wait_time(self) -> float:
        active = self.active_uavs
        if not active:
            return 0.0
        return float(np.mean([u.wait_time for u in self.uavs
                              if u.status != UAVStatus.LANDED]))
