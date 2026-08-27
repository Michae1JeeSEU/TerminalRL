"""Policy-independent external arrival demand generation."""
from __future__ import annotations
import math
import numpy as np
from .config import SimConfig
from .uav import ArrivalRequest, MissionType

class ArrivalProcess:
    def __init__(self, config: SimConfig, seed: int | None = None): self.config=config; self.reset(seed)
    def reset(self, seed: int | None = None) -> None:
        self.rng=np.random.default_rng(seed); self.generated=0; self.index=0; self.wave=0
        self._schedule=tuple(sorted(self.config.fixed_arrival_schedule))
        if self._schedule:
            self.next_time = self._schedule[0]
        elif self.config.arrival_model == "finite_cohort" and self.config.total_requested_uavs:
            self.next_time = 0.0
        else:
            self.next_time = self._next(0.0)
    def exhausted(self)->bool:
        return math.isinf(self.next_time)
    def _next(self,now:float)->float:
        if self.config.total_requested_uavs is not None and self.generated>=self.config.total_requested_uavs:return math.inf
        rate=self.config.arrival_rate_per_second
        if rate<=0:return math.inf
        if self.config.arrival_model=="poisson": return now+float(self.rng.exponential(1/rate))
        if self.config.arrival_model=="bursty_wave":
            gap=(.25 if self.wave%4<3 else 3.25)/rate; self.wave+=1; return now+gap
        return now+1/rate
    def _request(self,time:float)->ArrivalRequest:
        n=self.generated; mission=MissionType(self.rng.choice([m.value for m in MissionType],p=self.config.mission_probabilities)); base=int(self.rng.choice([1,3,5],p=[.50,.35,.15]))
        if mission is MissionType.EMERGENCY: base=5
        if self.config.initial_soc_mode=="fixed": soc=float(self.config.initial_soc)
        else:
            lower = self.config.initial_soc_min if self.config.initial_soc_min is not None else max(self.config.soc_reserve, self.config.soc_warning)
            upper = self.config.initial_soc_max if self.config.initial_soc_max is not None else 1.0
            soc=float(self.rng.uniform(lower, upper))
        self.generated+=1
        return ArrivalRequest(n,time,int(self.rng.choice(self.config.gate_count, p=self.config.gate_probabilities)),mission,base,self.config.energy_capacity,soc)
    def requests_until(self,now:float)->list[ArrivalRequest]:
        result=[]
        if self._schedule:
            while self.index<len(self._schedule) and self._schedule[self.index]<=now+1e-9:
                if self.config.total_requested_uavs is None or self.generated<self.config.total_requested_uavs: result.append(self._request(self._schedule[self.index]))
                self.index+=1
            self.next_time=self._schedule[self.index] if self.index<len(self._schedule) else math.inf
            return result
        while self.next_time<=now+1e-9:
            result.append(self._request(self.next_time)); self.next_time=self._next(self.next_time)
        return result
