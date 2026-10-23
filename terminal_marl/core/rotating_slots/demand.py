"""Frozen, replayable demand; only revealed requests enter a world snapshot."""
from dataclasses import asdict, dataclass
from statistics import NormalDist
import random


@dataclass(frozen=True)
class ArrivalRequest:
    id: str
    release_time: float
    gate_id: int
    initial_soc: float
    service_class: str
    vehicle_profile_id: str = "eh216s_reference_pm16_v1_0"

    def __post_init__(self):
        if self.release_time < 0 or self.gate_id not in range(4):
            raise ValueError("invalid release time or gate")
        if not 0 <= self.initial_soc <= 1 or self.service_class not in ("short", "standard", "long"):
            raise ValueError("invalid SOC or service class")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ArrivalManifest:
    requests: tuple = ()
    seed: int | None = None
    rate_per_hour: float | None = None
    arrival_window: float = 3600.0

    def __post_init__(self):
        ordered = tuple(sorted(self.requests, key=lambda r: (r.release_time, r.id)))
        if len({r.id for r in ordered}) != len(ordered):
            raise ValueError("duplicate request id")
        object.__setattr__(self, "requests", ordered)

    @classmethod
    def generate(cls, seed=11, rate_per_hour=60.0, arrival_window=3600.0, *, allow_new_seed=False):
        if (not allow_new_seed and seed not in (11, 23, 47, 71, 101)) or rate_per_hour not in (60, 120, 150, 180):
            raise ValueError("formal diagnostics use the frozen seeds and rates")
        if arrival_window <= 0:
            raise ValueError("arrival window must be positive")
        rng = random.Random(seed)
        normal = NormalDist(0.75, 0.08)
        lower, upper = normal.cdf(0.60), normal.cdf(0.90)
        requests, t = [], 0.0
        while True:
            t += rng.expovariate(rate_per_hour / 3600)
            if t > arrival_window:
                break
            gate = rng.randrange(4)
            soc = normal.inv_cdf(lower + rng.random() * (upper - lower))
            u = rng.random()
            service = "short" if u < .2 else "standard" if u < .8 else "long"
            requests.append(ArrivalRequest(f"req_{len(requests):06d}", t, gate, soc, service))
        return cls(tuple(requests), seed, rate_per_hour, arrival_window)

    def to_dict(self):
        return {"requests": [r.to_dict() for r in self.requests], "seed": self.seed,
                "rate_per_hour": self.rate_per_hour, "arrival_window": self.arrival_window}
