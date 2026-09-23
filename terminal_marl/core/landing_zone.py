"""Independent ground-service resource model; touchdown is not completion."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from .uav import MissionType, MISSION_SERVICE_SECONDS, UAVState

class LZStatus(str, Enum): FREE="free"; RESERVED="reserved"; OCCUPIED="occupied"
@dataclass
class LandingZone:
    lz_id: int; position: np.ndarray; status: LZStatus = LZStatus.FREE
    reserved_by: int | None = None; reservation_start: float | None = None
    expected_release_at: float | None = None
    occupied_by: int | None = None; service_remaining: float = 0.0
    compatible_missions: frozenset[MissionType] = frozenset(MissionType)
@dataclass(frozen=True)
class ReservationResult: approved: bool; lz_id: int | None; reason: str = ""

class LandingZoneManager:
    POSITIONS = ((-80.,-40.,0.), (0.,-40.,0.), (80.,-40.,0.), (-80.,40.,0.), (0.,40.,0.), (80.,40.,0.))
    ASSIGNMENT_MODES = frozenset(("lowest_id", "nearest_available"))

    def __init__(self, assignment_mode: str = "lowest_id") -> None:
        if assignment_mode not in self.ASSIGNMENT_MODES:
            raise ValueError(
                f"unsupported landing-zone assignment_mode {assignment_mode!r}; "
                f"expected one of {sorted(self.ASSIGNMENT_MODES)!r}"
            )
        self.assignment_mode = assignment_mode
        self.zones = [LandingZone(i, np.array(p)) for i,p in enumerate(self.POSITIONS)]

    def available_zones(self, uav: UAVState, now: float = 0.0) -> list[int]:
        return [z.lz_id for z in self.zones if z.status is LZStatus.FREE and uav.mission_type in z.compatible_missions]

    def rank_available_zones(self, uav: UAVState, now: float = 0.0) -> list[int]:
        """Return free, mission-compatible LZ ids in deterministic assignment order.

        ``lowest_id`` is the historical rule and remains the constructor default.
        ``nearest_available`` ranks by 3-D Euclidean distance from the current UAV
        position, with the LZ id as the deterministic tie-breaker.
        """
        available = self.available_zones(uav, now)
        if self.assignment_mode == "lowest_id":
            return sorted(available)

        position = np.asarray(uav.position, dtype=float)
        return sorted(
            available,
            key=lambda lz_id: (
                float(np.linalg.norm(self.zones[lz_id].position - position)),
                lz_id,
            ),
        )

    def choose_rule_zone(self, uav: UAVState, now: float = 0.0) -> int | None:
        ranked = self.rank_available_zones(uav, now)
        return ranked[0] if ranked else None
    def reserve(self, lz_id: int, uav_id: int, now: float) -> ReservationResult:
        z=self.zones[lz_id]
        if z.status is not LZStatus.FREE: return ReservationResult(False,None,"lz_not_free")
        z.status=LZStatus.RESERVED; z.reserved_by=uav_id; z.reservation_start=now; z.expected_release_at=None
        return ReservationResult(True,lz_id)
    def release(self,lz_id:int,uav_id:int,reason:str="") -> None:
        z=self.zones[lz_id]
        if z.status is LZStatus.RESERVED and z.reserved_by==uav_id:
            z.status=LZStatus.FREE; z.reserved_by=None; z.reservation_start=None; z.expected_release_at=None
    def note_committed_landing(self, lz_id: int, uav_id: int, expected_release_at: float) -> bool:
        """Record the conservative release ETA for a matching pad reservation."""
        z = self.zones[lz_id]
        if z.status is not LZStatus.RESERVED or z.reserved_by != uav_id:
            return False
        try:
            eta = float(expected_release_at)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected_release_at must be a finite timestamp") from exc
        if not math.isfinite(eta):
            raise ValueError("expected_release_at must be a finite timestamp")
        if z.reservation_start is not None and eta < float(z.reservation_start):
            raise ValueError("expected_release_at must not precede reservation_start")
        z.expected_release_at = eta
        return True
    def touchdown(self,lz_id:int,uav_id:int,mission:MissionType) -> None:
        z=self.zones[lz_id]
        if z.status is not LZStatus.RESERVED or z.reserved_by!=uav_id: raise ValueError("touchdown without matching reservation")
        z.status=LZStatus.OCCUPIED; z.occupied_by=uav_id; z.reserved_by=None; z.reservation_start=None; z.expected_release_at=None; z.service_remaining=MISSION_SERVICE_SECONDS[mission]
    def step(self,dt:float) -> list[int]:
        complete=[]
        for z in self.zones:
            if z.status is LZStatus.OCCUPIED:
                z.service_remaining=max(0.,z.service_remaining-dt)
                if z.service_remaining==0.:
                    complete.append(z.occupied_by); z.status=LZStatus.FREE; z.occupied_by=None; z.expected_release_at=None
        return [uid for uid in complete if uid is not None]
    def earliest_release_time(self,lz_id:int,now:float=0.0)->float:
        z=self.zones[lz_id]
        if z.status is LZStatus.FREE:
            return now
        if z.status is LZStatus.RESERVED:
            if z.expected_release_at is None:
                return math.inf
            return max(now, z.expected_release_at)
        return now+z.service_remaining
