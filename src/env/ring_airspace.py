import numpy as np

from src.config import RingConfig


class RingAirspace:
    """Three-ring terminal area airspace model with hard constraints.

    Hard constraints: R1 > R2 > R3, H1 > H2 > H3.
    """

    def __init__(self, config: RingConfig):
        self.config = config
        self.radii = np.array(config.default_radii, dtype=np.float64)    # [R1, R2, R3]
        self.heights = np.array(config.default_heights, dtype=np.float64) # [H1, H2, H3]
        self.speeds = np.array(config.default_speeds, dtype=np.float64)   # [V1, V2, V3]

    def update(self, radii: np.ndarray, heights: np.ndarray, speeds: np.ndarray) -> bool:
        """Apply new airspace parameters. Returns True if valid (constraints satisfied)."""
        if not self._check_constraints(radii, heights):
            return False
        self.radii = radii.copy()
        self.heights = heights.copy()
        self.speeds = speeds.copy()
        return True

    def _check_constraints(self, radii: np.ndarray, heights: np.ndarray) -> bool:
        """Verify: R1 > R2 > R3, H1 > H2 > H3, all positive."""
        if not (radii[0] > radii[1] > radii[2]):
            return False
        if not (heights[0] > heights[1] > heights[2]):
            return False
        if not (np.all(radii >= self.config.r_min) and np.all(radii <= self.config.r_max)):
            return False
        if not (np.all(heights >= self.config.h_min) and np.all(heights <= self.config.h_max)):
            return False
        return True

    def get_ring(self, position: np.ndarray) -> int:
        """Determine which ring a position belongs to (1=outer, 2=middle, 3=inner, 0=outside)."""
        r = np.linalg.norm(position[:2])
        z = position[2]
        eps = 1.0  # small tolerance for boundary crossing
        for i in range(2, -1, -1):  # inner to outer: assign innermost containing ring
            if r <= self.radii[i] + eps and z <= self.heights[i] + eps:
                return i + 1
        return 0  # outside all rings

    def in_inner_ring(self, position: np.ndarray) -> bool:
        """Check if position is within the inner ring (landing-capable zone)."""
        return self.get_ring(position) == 3

    def random_position_on_ring(self, ring_id: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate a random position on the circumference of a specific ring."""
        if rng is None:
            rng = np.random.default_rng()
        idx = ring_id - 1
        angle = rng.uniform(0, 2 * np.pi)
        r = self.radii[idx]
        h = self.heights[idx]
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return np.array([x, y, h], dtype=np.float64)

    def random_position_in_ring(self, ring_id: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate a random position *inside* a ring's volume.

        Ring 1: r in (R2, R1], z in (H2, H1]  (outer band)
        Ring 2: r in (R3, R2], z in (H3, H2]  (middle band)
        Ring 3: r in (0, R3],   z in (0, H3]  (inner disc)
        """
        if rng is None:
            rng = np.random.default_rng()
        idx = ring_id - 1
        r_outer = self.radii[idx]
        h_upper = self.heights[idx]
        if ring_id == 3:
            r_inner = 0.0
            h_lower = 0.0
        else:
            r_inner = self.radii[idx + 1]
            h_lower = self.heights[idx + 1]
        r = rng.uniform(r_inner, r_outer)
        h = rng.uniform(h_lower, h_upper)
        angle = rng.uniform(0, 2 * np.pi)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return np.array([x, y, h], dtype=np.float64)

    def get_boundary(self, ring_id: int) -> dict:
        """Return the geometric boundary of a ring."""
        idx = ring_id - 1
        return {
            "radius": self.radii[idx],
            "height": self.heights[idx],
            "speed": self.speeds[idx],
        }

    @property
    def state_vector(self) -> np.ndarray:
        """Return the full airspace state as a flat vector [R1,R2,R3, H1,H2,H3, V1,V2,V3]."""
        return np.concatenate([self.radii, self.heights, self.speeds])
