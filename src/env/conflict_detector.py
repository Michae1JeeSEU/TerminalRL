import numpy as np

from src.env.uav import UAV, UAVStatus


class ConflictDetector:
    """Detects pairwise safety conflicts between UAVs."""

    def __init__(self, safe_distance: float):
        self.safe_distance = safe_distance

    def check_pairwise(self, uav1: UAV, uav2: UAV) -> bool:
        """Return True if uav1 and uav2 violate safe separation."""
        if uav1.status == UAVStatus.LANDED or uav2.status == UAVStatus.LANDED:
            return False
        dist = np.linalg.norm(uav1.position - uav2.position)
        return dist < self.safe_distance

    def check_all(self, uavs: list[UAV]) -> list[tuple[int, int]]:
        """Return list of conflicting (id1, id2) pairs."""
        active = [u for u in uavs if u.status != UAVStatus.LANDED]
        conflicts = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if self.check_pairwise(active[i], active[j]):
                    conflicts.append((active[i].id, active[j].id))
        return conflicts

    def compute_penalty(self, uavs: list[UAV], penalty_per_conflict: float) -> float:
        n = len(self.check_all(uavs))
        return float(n) * penalty_per_conflict
