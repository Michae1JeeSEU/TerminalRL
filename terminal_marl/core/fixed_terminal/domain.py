"""Plain domain records. Units: metres, seconds, normalized SOC.

Kinematic surrogate only: piecewise circular/linear paths have instantaneous
heading changes. Acceleration and flight-controller tracking are not modeled.
"""
from dataclasses import dataclass, field, asdict
from math import asin, cos, sin, pi, floor, sqrt, isfinite

Point = tuple[float, float, float]


def distance(a, b):
    return sqrt(sum((x-y)**2 for x, y in zip(a, b)))


@dataclass(frozen=True)
class Ring:
    radius: float
    height: float


@dataclass(frozen=True)
class EnergyRates:
    hold: float = .0008
    transfer: float = .0012
    descent: float = .0009
    landing: float = .0014
    reserve: float = .06


@dataclass(frozen=True)
class TerminalConfig:
    rings: tuple[Ring, ...] = (Ring(12., 6.), Ring(20., 10.), Ring(28., 14.))
    angular_speed: float = .12
    separation: float = 3.
    tracking_error: float = .5
    temporal_spacing: float = 2.
    merge_window: float = 2.
    merge_spare_slots: int = 1
    transfer_speed: float = 2.
    max_speed: float = 4.
    max_descent: float = 1.
    landing_speed: float = .5
    landing_height: float = .5
    pad_service: float = 5.
    normal_pad: Point = (0., 0., 0.)
    emergency_pad: Point = (-38., 0., 0.)
    horizon: float = 240.
    dt: float = 1.
    continuous_tolerance: float = .005
    energy: EnergyRates = field(default_factory=EnergyRates)

    def __post_init__(self):
        # Copy caller-owned containers before freezing episode geometry.
        object.__setattr__(self, "rings", tuple(self.rings))
        values = (self.angular_speed, self.separation, self.temporal_spacing,
                  self.merge_window, self.transfer_speed, self.max_speed,
                  self.max_descent, self.landing_speed, self.landing_height,
                  self.pad_service, self.horizon, self.dt, self.continuous_tolerance)
        if any(not isfinite(v) or v <= 0 for v in values):
            raise ValueError("finite positive timing and kinematic parameters required")
        if not self.rings or self.tracking_error < 0 or self.merge_spare_slots < 1:
            raise ValueError("rings, tracking margin and merge spare required")
        if any(not isfinite(r.radius+r.height) or r.radius <= 0 or r.height <= self.landing_height for r in self.rings):
            raise ValueError("positive ring radii/heights above landing height required")
        if any(b.radius <= a.radius or b.height <= a.height for a, b in zip(self.rings, self.rings[1:])):
            raise ValueError("radii/heights increase from inner to outer")
        if any(distance((a.radius, 0, a.height), (b.radius, 0, b.height)) < self.protected_distance
               for a, b in zip(self.rings, self.rings[1:])):
            raise ValueError("adjacent rings violate protected spacing")
        if self.rings[-1].radius*self.angular_speed > self.max_speed:
            raise ValueError("holding speed exceeds limit")
        if self.transfer_speed > self.max_speed or self.landing_speed > min(self.max_descent, self.max_speed):
            raise ValueError("route speed exceeds limit")
        if self.normal_pad != (0., 0., 0.) or self.emergency_pad[1:] != (0., 0.):
            raise ValueError("center normal pad and negative-x emergency pad required")
        if len(self.emergency_pad) != 3 or any(not isfinite(v) for v in self.emergency_pad):
            raise ValueError("finite emergency pad coordinates required")
        if self.emergency_pad[0] >= -self.rings[-1].radius-self.protected_distance:
            raise ValueError("emergency pad must lie outside rings")
        if any(not isfinite(v) or v <= 0 for v in asdict(self.energy).values()) or self.energy.reserve >= 1:
            raise ValueError("positive SOC rates and reserve in (0,1) required")
        if any(self.operational_capacity(i) < 2 for i in range(len(self.rings))):
            raise ValueError("insufficient operational capacity")

    @property
    def protected_distance(self):
        return self.separation+2*self.tracking_error

    @property
    def lap_time(self):
        return 2*pi/self.angular_speed

    def geometric_capacity(self, ring):
        r = self.rings[ring].radius
        return 1 if self.separation > 2*r else floor(pi/asin(self.separation/(2*r)))

    def operational_capacity(self, ring):
        r = self.rings[ring].radius
        if self.protected_distance >= 2*r:
            return 0
        angle = 2*asin(self.protected_distance/(2*r))
        angle += self.angular_speed*(self.temporal_spacing+self.merge_window)
        return max(0, floor(2*pi/angle)-self.merge_spare_slots)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    mode: str
    p0: Point
    p1: Point
    ring: int | None = None
    theta0: float = 0.
    omega: float = 0.

    def position(self, time):
        fraction = min(1., max(0., (time-self.start)/(self.end-self.start)))
        if self.mode == "hold":
            theta = self.theta0-self.omega*(min(self.end, max(self.start, time))-self.start)
            radius = (self.p0[0]**2+self.p0[1]**2)**.5
            return radius*cos(theta), radius*sin(theta), self.p0[2]
        return tuple(a+fraction*(b-a) for a, b in zip(self.p0, self.p1))

    @property
    def speed_bound(self):
        return (self.omega*(self.p0[0]**2+self.p0[1]**2)**.5 if self.mode == "hold"
                else distance(self.p0, self.p1)/(self.end-self.start))


@dataclass(frozen=True)
class Reservation:
    resource: str
    start: float
    end: float


@dataclass(frozen=True)
class Intent:
    aircraft_id: str
    # Inner-to-current ring: extra complete laps before leaving each ring.
    wait_cycles: tuple[int, ...] = ()
    emergency: bool = False


@dataclass(frozen=True)
class Plan:
    aircraft_id: str
    emergency: bool
    wait_cycles: tuple[int, ...]
    segments: tuple[Segment, ...]
    reservations: tuple[Reservation, ...]

    @property
    def start(self):
        return self.segments[0].start

    @property
    def end(self):
        return self.segments[-1].end

    def position(self, time):
        for segment in self.segments:
            if time <= segment.end:
                return segment.position(time)
        return self.segments[-1].p1


@dataclass
class Aircraft:
    aircraft_id: str
    arrival_time: float
    initial_soc: float
    soc: float
    ring: int | None
    theta: float
    state: str = "holding"
    plan: Plan | None = None
    completion_time: float | None = None
    admitted_time: float | None = None
    reason: str | None = None
    minimum_soc: float = 1.
    consumed: float = 0.
    unimpeded_duration: float | None = None
    emergency_energy: float = 0.
    emergency_resource_seconds: float = 0.

    def position(self, time, config):
        if self.plan:
            return self.plan.position(time)
        ring = config.rings[self.ring]
        return ring.radius*cos(self.theta), ring.radius*sin(self.theta), ring.height
