"""Independent demand/SOC streams, true truncated-normal rejection sampling."""
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, pi
from random import Random
from .domain import Aircraft, distance


@dataclass(frozen=True)
class Request:
    aircraft_id: str
    arrival_time: float
    soc: float
    ring: int | None
    theta: float


@dataclass(frozen=True)
class Scenario:
    seed: int
    load: str
    soc_profile: str
    requests: tuple[Request, ...]
    demand_stream_seed: int
    soc_stream_seed: int


def stream_seed(seed, name):
    return int.from_bytes(sha256(f"fixed-terminal-stage1:{seed}:{name}".encode()).digest()[:8], "big")


def truncated_normal(rng, mean, std, lower, upper):
    if not 0 <= lower < upper <= 1 or std <= 0:
        raise ValueError("SOC requires 0<=lower<upper<=1 and positive std")
    for _ in range(100000):
        value = rng.gauss(mean, std)
        if lower <= value <= upper:
            return value
    raise ValueError("rejection sampler exhausted")


def generate_scenario(config, seed=101, load="low", soc_profile="high"):
    from terminal_marl.configs.fixed_terminal_stage1 import LOAD_PROFILES, SOC_PROFILES
    demand_seed, soc_seed = stream_seed(seed, "demand"), stream_seed(seed, "soc")
    demand_rng, soc_rng = Random(demand_seed), Random(soc_seed)
    load_spec, soc_spec = LOAD_PROFILES[load], SOC_PROFILES[soc_profile]
    requests = []

    def add(time, ring, theta):
        soc = truncated_normal(soc_rng, soc_spec.mean, soc_spec.std, soc_spec.lower, soc_spec.upper)
        requests.append(Request(f"uav-{len(requests):04d}", time, soc, ring, theta))

    for i, _ in enumerate(config.rings):
        count = max(1, ceil(config.operational_capacity(i)*load_spec.initial_fraction))
        for j in range(count):
            add(0., i, (2*pi*(j+.37)/count) % (2*pi))
    time = demand_rng.expovariate(1/load_spec.mean_interarrival)
    while time < config.horizon:
        add(time, None, 0.)
        time += demand_rng.expovariate(1/load_spec.mean_interarrival)
    initial = [Aircraft(r.aircraft_id, 0, r.soc, r.soc, r.ring, r.theta)
               for r in requests if r.ring is not None]
    for i, a in enumerate(initial):
        for b in initial[i+1:]:
            if distance(a.position(0, config), b.position(0, config)) < config.protected_distance:
                raise ValueError("initial protected spacing violated")
    return Scenario(seed, load, soc_profile, tuple(requests), demand_seed, soc_seed)
