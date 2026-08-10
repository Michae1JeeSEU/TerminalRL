"""Independent ground-service resource model; touchdown is not completion."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .uav import MissionType, MISSION_SERVICE_SECONDS, UAVState

class LZStatus(str, Enum): FREE="free"; RESERVED="reserved"; OCCUPIED="occupied"
@dataclass
class LandingZone:
    lz_id: int; position: np.ndarray; status: LZStatus = LZStatus.FREE
    reserved_by: int | None = None; reservation_start: float | None = None
    occupied_by: int | None = None; service_remaining: float = 0.0
    compatible_missions: frozenset[MissionType] = frozenset(MissionType)
@dataclass(frozen=True)
class ReservationResult: approved: bool; lz_id: int | None; reason: str = ""

class LandingZoneManager:
    POSITIONS = ((-80.,-40.,0.), (0.,-40.,0.), (80.,-40.,0.), (-80.,40.,0.), (0.,40.,0.), (80.,40.,0.))
    def __init__(self) -> None: self.zones = [LandingZone(i, np.array(p)) for i,p in enumerate(self.POSITIONS)]
    def available_zones(self, uav: UAVState, now: float = 0.0) -> list[int]:
        return [z.lz_id for z in self.zones if z.status is LZStatus.FREE and uav.mission_type in z.compatible_missions]
    def choose_rule_zone(self, uav: UAVState, now: float = 0.0) -> int | None:
        available = self.available_zones(uav, now)
        return min(available) if available else None
    def reserve(self, lz_id: int, uav_id: int, now: float) -> ReservationResult:
        z=self.zones[lz_id]
        if z.status is not LZStatus.FREE: return ReservationResult(False,None,"lz_not_free")
        z.status=LZStatus.RESERVED; z.reserved_by=uav_id; z.reservation_start=now
        return ReservationResult(True,lz_id)
    def release(self,lz_id:int,uav_id:int,reason:str="") -> None:
        z=self.zones[lz_id]
        if z.status is LZStatus.RESERVED and z.reserved_by==uav_id:
            z.status=LZStatus.FREE; z.reserved_by=None; z.reservation_start=None
    def touchdown(self,lz_id:int,uav_id:int,mission:MissionType) -> None:
        z=self.zones[lz_id]
        if z.status is not LZStatus.RESERVED or z.reserved_by!=uav_id: raise ValueError("touchdown without matching reservation")
        z.status=LZStatus.OCCUPIED; z.occupied_by=uav_id; z.reserved_by=None; z.service_remaining=MISSION_SERVICE_SECONDS[mission]
    def step(self,dt:float) -> list[int]:
        complete=[]
        for z in self.zones:
            if z.status is LZStatus.OCCUPIED:
                z.service_remaining=max(0.,z.service_remaining-dt)
                if z.service_remaining==0.:
                    complete.append(z.occupied_by); z.status=LZStatus.FREE; z.occupied_by=None
        return [uid for uid in complete if uid is not None]
    def earliest_release_time(self,lz_id:int,now:float=0.0)->float:
        z=self.zones[lz_id]; return now if z.status is LZStatus.FREE else now+z.service_remaining
