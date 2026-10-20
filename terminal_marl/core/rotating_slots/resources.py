"""Half-open resource calendars with explicit local commitment dependencies."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class Reservation:
    resource: tuple | int
    aircraft_id: str
    start: float
    end: float = math.inf
    kind: str = "target"
    plan_hash: str | None = None
    committed: bool = False
    dependencies: tuple = ()

    def overlaps(self, start, end):
        return self.start < end and start < self.end

    def to_dict(self):
        d = asdict(self)
        d["end"] = self.end if math.isfinite(self.end) else None
        return d


class ResourceConflict(ValueError):
    pass


class ResourceDependencyError(ValueError):
    pass


class ResourceCalendar:
    def __init__(self):
        self._entries = []

    @property
    def reservations(self):
        return tuple(self._entries)

    def clone(self):
        result = type(self)()
        result._entries = list(self._entries)
        return result

    def conflicts(self, resource, start, end, aircraft_id=None):
        return tuple(r for r in self._entries if r.resource == resource and
                     r.aircraft_id != aircraft_id and r.overlaps(start, end))

    def reserve(self, resource, aircraft_id, start, end=math.inf, **kwargs):
        if not math.isfinite(start) or end <= start or math.isnan(end):
            raise ValueError("resource intervals must have positive length")
        if self.conflicts(resource, start, end, aircraft_id):
            raise ResourceConflict(f"resource {resource} already reserved")
        r = Reservation(resource, aircraft_id, float(start), float(end), **kwargs)
        self._entries.append(r)
        self._entries.sort(key=lambda x: (str(x.resource), x.start, x.aircraft_id, x.kind))
        return r

    def remove_plan(self, plan_hash, protected=()):
        if plan_hash in protected or any(plan_hash in r.dependencies and r.committed for r in self._entries):
            raise ResourceDependencyError("already-started maneuver depends on this release")
        self._entries = [r for r in self._entries if r.plan_hash != plan_hash]

    def release(self, aircraft_id, resource, end):
        self._entries = [replace(r, end=max(r.start, end)) if r.aircraft_id == aircraft_id and
                         r.resource == resource and r.end > end else r for r in self._entries]
        self._entries = [r for r in self._entries if r.end > r.start]

    def mark_committed(self, plan_hash):
        self._entries = [replace(r, committed=True) if r.plan_hash == plan_hash else r for r in self._entries]

    def isolate(self, aircraft_id, time):
        """End this entity's managed occupancy now, retaining its past intervals."""
        released = sum(r.aircraft_id == aircraft_id and r.end > time for r in self._entries)
        self._entries = [replace(r, end=time) if r.aircraft_id == aircraft_id and r.end > time else r
                         for r in self._entries if r.aircraft_id != aircraft_id or r.start < time]
        return released

    def active(self, resource, time):
        return tuple(r for r in self._entries if r.resource == resource and r.start <= time < r.end)

    def to_dict(self):
        return [r.to_dict() for r in self._entries]

    @contextmanager
    def transaction(self):
        old = list(self._entries)
        try:
            yield self
        except Exception:
            self._entries = old
            raise


class SlotCalendar(ResourceCalendar):
    pass


class PadCalendar(ResourceCalendar):
    pass
