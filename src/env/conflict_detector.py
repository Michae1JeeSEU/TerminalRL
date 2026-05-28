import numpy as np

from src.env.uav import UAV, UAVStatus


class ConflictDetector:
    """Detects pairwise safety conflicts and near-misses between UAVs.

    Collision: 3D distance < safe_distance → immediate termination.
    Near-miss: safe_distance ≤ distance < near_miss_threshold → soft penalty.
    """

    def __init__(self, safe_distance: float, near_miss_threshold: float = 50.0):
        self.safe_distance = safe_distance
        self.near_miss_threshold = near_miss_threshold

    def check_pairwise(self, uav1: UAV, uav2: UAV) -> bool:
        """Return True if uav1 and uav2 are in collision (< safe_distance)."""
        if (uav1.status == UAVStatus.LANDED or uav2.status == UAVStatus.LANDED
                or uav1.is_dead or uav2.is_dead):
            return False
        dist = np.linalg.norm(uav1.position - uav2.position)
        return dist < self.safe_distance

    def check_all(self, uavs: list[UAV]) -> list[tuple[int, int]]:
        """Return list of colliding (id1, id2) pairs (dist < safe_distance)."""
        active = [u for u in uavs if u.status != UAVStatus.LANDED and not u.is_dead]
        conflicts = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if self.check_pairwise(active[i], active[j]):
                    conflicts.append((active[i].id, active[j].id))
        return conflicts

    def check_near_misses(self, uavs: list[UAV]) -> list[tuple[int, int]]:
        """Return list of near-miss (id1, id2) pairs.

        Near-miss: safe_distance ≤ distance < near_miss_threshold.
        Excludes pairs already in collision.
        """
        active = [u for u in uavs if u.status != UAVStatus.LANDED and not u.is_dead]
        pairs = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                u1, u2 = active[i], active[j]
                dist = np.linalg.norm(u1.position - u2.position)
                if self.safe_distance <= dist < self.near_miss_threshold:
                    pairs.append((u1.id, u2.id))
        return pairs

    def compute_penalty(self, uavs: list[UAV], penalty_per_conflict: float) -> float:
        n = len(self.check_all(uavs))
        return float(n) * penalty_per_conflict
