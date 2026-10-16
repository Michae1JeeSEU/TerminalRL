from dataclasses import dataclass, field
import math

@dataclass(frozen=True)
class KinematicState:
    p: tuple
    v: tuple = (0.,0.,0.)
    a: tuple = (0.,0.,0.)
    jerk: tuple = (0.,0.,0.)
    def __post_init__(self):
        for key in ('p','v','a','jerk'):
            value = tuple(float(x) for x in getattr(self,key))
            if len(value)!=3: raise ValueError('three coordinates required')
            if not all(math.isfinite(x) for x in value): raise ValueError('nonfinite kinematic state')
            object.__setattr__(self,key,value)
    @property
    def position(self): return self.p
    @property
    def velocity(self): return self.v
    @property
    def acceleration(self): return self.a

@dataclass(frozen=True)
class CandidateResult:
    status: str
    plan: object = None
    reason: str = ''
    details: dict = field(default_factory=dict)
    @property
    def accepted(self): return self.status == 'FEASIBLE'

@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    status: str
    reason: str = ''
    details: dict = field(default_factory=dict)
    @property
    def certificate(self): return self.details
