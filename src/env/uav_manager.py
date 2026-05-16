import numpy as np

from src.config import UAVConfig, MISSION_TYPE_WEIGHTS
from src.env.ring_airspace import RingAirspace
from src.env.uav import UAV, UAVStatus, LowLevelAction


class UAVManager:
    """Manages a fleet of UAVs: creation, batch stepping, inflow scheduling."""

    # Initial ring distribution proportions for orbit-based UAVs
    _INIT_RING_PROBS = (0.60, 0.25, 0.15)  # ring 1, 2, 3

    def __init__(self, config: UAVConfig, num_uavs: int, airspace: RingAirspace,
                 max_uavs: int = 100, uav_inflow_rate: float = 0.0,
                 rng: np.random.Generator | None = None,
                 initial_uavs: int | None = None):
        self.config = config
        self.airspace = airspace
        self.rng = rng or np.random.default_rng()
        self.uavs: list[UAV] = []
        self.landed_count = 0
        self.failed_count = 0
        self._next_id = 0
        self.max_uavs = max_uavs

        # Inflow scheduling
        self.uav_inflow_rate = uav_inflow_rate
        if uav_inflow_rate > 0:
            self.inflow_interval = 3600.0 / uav_inflow_rate  # seconds per UAV
        else:
            self.inflow_interval = float('inf')
        self._time_since_last_spawn = 0.0
        self.last_safety_adjustments: dict[int, object] = {}

        # Distribute initial UAVs across ring orbits
        n_initial = initial_uavs if initial_uavs is not None else num_uavs
        if n_initial > 0:
            ring_ids = self.rng.choice([1, 2, 3], size=n_initial, p=self._INIT_RING_PROBS)
            for ring_id in ring_ids:
                self._spawn_one(ring_id=int(ring_id))

    def _spawn_one(self, ring_id: int = 1, battery: float | None = None,
                   wait_time: float | None = None, base_priority: int | None = None,
                   is_entry: bool = False, mission_type: int | None = None) -> UAV:
        """Spawn a UAV.

        Entry spawn (is_entry=True): uses a fixed entry point on ring 1.
        Orbit spawn (is_entry=False): random angle on the ring orbit.
        """
        if base_priority is None:
            base_priority = int(self.rng.choice([1, 3, 5], p=[0.5, 0.35, 0.15]))
        if mission_type is None:
            mission_type = int(self.rng.choice([1, 2, 3, 4], p=MISSION_TYPE_WEIGHTS))
        uav = UAV(self._next_id, self.config, base_priority, mission_type)
        self._next_id += 1

        if ring_id == 1 and is_entry:
            entry_points = self.airspace.get_entry_points()
            position = entry_points[self.rng.integers(0, 4)].copy()
        else:
            position = self.airspace.random_position_on_ring(ring_id, self.rng)

        if battery is None:
            battery = float(self.rng.uniform(
                self.config.battery_critical * self.config.max_battery,
                self.config.max_battery,
            ))
        if wait_time is None:
            wait_time = float(self.rng.uniform(0, 600))

        uav.reset(position, ring_id=ring_id, battery=battery,
                  wait_time=wait_time, status=UAVStatus.CRUISING)
        self.uavs.append(uav)
        return uav

    def remove_landed(self) -> None:
        """Remove landed UAVs. Failed UAVs are also removed."""
        landed = [u for u in self.uavs if u.status == UAVStatus.LANDED]
        self.landed_count += len(landed)
        failed = [u for u in self.uavs if u.is_dead]
        self.failed_count += len(failed)
        remove_ids = {u.id for u in landed + failed}
        self.uavs = [u for u in self.uavs if u.id not in remove_ids]

    def spawn_entry_uav(self) -> int:
        """Spawn a single UAV at a random entry point. Returns its list index."""
        uav = self._spawn_one(ring_id=1, is_entry=True)
        return len(self.uavs) - 1  # index of the new UAV

    def try_spawn_inflow(self, dt: float) -> None:
        """Accumulate time and spawn UAVs according to inflow rate."""
        if self.uav_inflow_rate <= 0:
            return
        self._time_since_last_spawn += dt
        while (self._time_since_last_spawn >= self.inflow_interval
               and len(self.uavs) < self.max_uavs):
            self._spawn_one()
            self._time_since_last_spawn -= self.inflow_interval

    def step_all(self, actions: dict[int, LowLevelAction], dt: float,
                 safety_controller=None) -> None:
        """Apply per-UAV actions and advance all UAVs.

        When safety_controller is provided, it evaluates and adjusts actions
        before each UAV step for reflexive collision avoidance.
        """
        # Update ring assignments
        for uav in self.uavs:
            ring_id = self.airspace.get_ring(uav.position)
            if ring_id == 0:
                ring_id = uav.ring_id
            uav.ring_id = ring_id

        # Reflexive safety: evaluate and adjust actions
        self.last_safety_adjustments: dict[int, object] = {}
        if safety_controller is not None:
            adjustments = safety_controller.evaluate(self.uavs, actions)
            actions = safety_controller.apply(actions, adjustments)
            self.last_safety_adjustments = adjustments

        for uav in self.uavs:
            action = actions.get(uav.id, LowLevelAction(control=0))

            ring_id = uav.ring_id
            if ring_id > 0:
                r_i, h_i = self.airspace.get_ring_orbit(ring_id)
                speed_i = self.airspace.speeds[ring_id - 1]
            else:
                r_i, h_i, speed_i = self.airspace.radii[0], self.airspace.heights[0], self.airspace.speeds[0]

            uav.step(action, speed_i, r_i, h_i, dt)

            # Update ring after step (may have crossed boundary during descent)
            if uav.status != UAVStatus.LANDED:
                new_ring = self.airspace.get_ring(uav.position)
                if new_ring > 0 and new_ring != ring_id:
                    uav._prev_ring_id = ring_id
                    uav.ring_id = new_ring
                    uav.ring_entry_time = uav.wait_time  # reset residency timer

    @property
    def active_uavs(self) -> list[UAV]:
        return [u for u in self.uavs if u.status != UAVStatus.LANDED]

    def get_state_matrix(self) -> np.ndarray:
        """Return (N, 6) matrix: [x, y, z, battery, base_priority, angle] for all active UAVs."""
        active = self.active_uavs
        if not active:
            return np.empty((0, 6))
        data = []
        for u in active:
            data.append([u.position[0], u.position[1], u.position[2],
                         u.battery, u.base_priority, u.angle])
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
