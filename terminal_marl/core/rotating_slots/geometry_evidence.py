"""Bounded, process-local SAFE geometry evidence; never scheduling authority."""
from collections import OrderedDict
import math
import sys

from .config import Config, Section
from .trajectory import Segment, TrajectoryPlan

_MAX_ENTRIES = 4096
_MAX_BYTES = 16 * 1024 * 1024
_VALUES = OrderedDict()
_BYTES = 0
_COUNTS = {}


def count(name, value=1):
    _COUNTS[name] = _COUNTS.get(name, 0) + value


def clear_geometry_evidence_cache():
    """Cold-comparison reset, including process-local observations."""
    global _BYTES
    _VALUES.clear()
    _COUNTS.clear()
    _BYTES = 0


def geometry_evidence_cache_stats():
    return dict(geometry_evidence_counts(), entries=len(_VALUES), estimated_bytes=_BYTES,
                max_entries=_MAX_ENTRIES, max_bytes=_MAX_BYTES)


def geometry_evidence_counts():
    """Pure snapshot suitable for before/after operation-count differences."""
    return {name: _COUNTS.get(name, 0) for name in
            ('hit', 'miss', 'compute', 'analytic', 'fine', 'box', 'bypass',
             'store', 'eviction', 'oversized')}


def _immutable(value):
    if value is None:
        return ('none',)
    if type(value) in (str, bool, int):
        return (type(value).__name__, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError('nonfinite identity')
        return ('float', value.hex())
    if type(value) is tuple:
        return ('tuple', tuple(_immutable(x) for x in value))
    if type(value) is Section:
        return ('section', tuple((k, _immutable(v)) for k, v in value._items))
    raise TypeError('identity is not an immutable supported type')


def config_identity(config):
    """Full frozen contents, including metadata; hashes alone are insufficient."""
    if type(config) is not Config:
        return None
    try:
        return tuple((name, _immutable(getattr(config, name)))
                     for name in Config.__dataclass_fields__)
    except (AttributeError, TypeError):
        return None


def plan_identity(plan):
    if type(plan) is not TrajectoryPlan or any(type(s) is not Segment for s in plan.segments):
        return None
    try:
        # Include bytes represented by every field as well as the checked digest.
        # The constructor verifies the digest, and no duck-typed plan is trusted.
        return tuple((name, tuple(tuple((field, _immutable(getattr(s, field)))
                                       for field in Segment.__dataclass_fields__)
                                 for s in plan.segments)
                      if name == 'segments' else _immutable(getattr(plan, name)))
                     for name in TrajectoryPlan.__dataclass_fields__)
    except (AttributeError, TypeError):
        return None


def fragment_identity(traffic, start, end, traffic_type, identities):
    """Snapshot inclusive endpoints and every actual switch/segment boundary.

    _Traffic itself is mutable. Only its immutable plan bindings for this exact
    absolute fragment are retained; no active-plan or midpoint-only shortcut is
    used. Including the right endpoint prevents a new tail at that endpoint
    from inheriting the previous tail's evidence.
    """
    def identity(plan):
        index = id(plan)
        if index not in identities:
            identities[index] = plan_identity(plan)
        return identities[index]

    direct = type(traffic) is TrajectoryPlan
    if not direct and type(traffic) is not traffic_type:
        return None
    if not direct and (type(traffic.base) is not TrajectoryPlan or
                       type(traffic.plans) is not tuple or
                       any(type(p) is not TrajectoryPlan for p in traffic.plans)):
        return None
    try:
        switches = () if direct else tuple(p.start_time for p in traffic.plans
                                          if start < p.start_time < end)
        cuts = sorted({start, end, *switches})
        at = (lambda t: traffic) if direct else traffic._at
        used = [at(start), at(end)]
        used.extend(at((a+b)/2) for a, b in zip(cuts, cuts[1:]))
        values = tuple(identity(p) for p in used)
        if any(v is None for v in values):
            return None
        boundaries = tuple(tuple((s.start_time, s.end_time) for s in p.segments
                                 if s.start_time <= end and s.end_time >= start)
                           for p in used)
        # Wrapper horizon metadata does not change state/bounds. Normalizing it
        # away permits the identical fragment to be shared by direct plans and
        # hold/current/future wrappers, while every queried time is still keyed.
        return ('fragment', _immutable(tuple(cuts)), values, boundaries)
    except (AttributeError, TypeError, ValueError):
        return None


def evidence_key(a, b, start, end, config_key, max_depth, min_interval, traffic_type, identities):
    if config_key is None:
        return None
    try:
        if type(max_depth) is not int or type(min_interval) not in (float, int):
            return None
        times = _immutable((start, end, max_depth, min_interval))
        left = fragment_identity(a, start, end, traffic_type, identities)
        right = fragment_identity(b, start, end, traffic_type, identities)
        if left is None or right is None:
            return None
        return ('continuous-separation-v1', config_key, times, left, right)
    except (TypeError, ValueError):
        return None


def lookup(key):
    if key is None:
        count('bypass')
        return None
    found = _VALUES.get(key)
    if found is None:
        count('miss')
        return None
    _VALUES.move_to_end(key)
    count('hit')
    return dict(found[0])


def _size(value):
    # Deliberately charge duplicate references too: this overestimates shared
    # keys and avoids retaining unaccounted large plan/config payloads.
    return sys.getsizeof(value) + (sum(_size(x) for x in value) if type(value) is tuple else 0)


def store_safe(key, result):
    global _BYTES
    if key is None or not result.accepted or result.status != 'SAFE':
        return
    # SAFE paths contain only scalar details. Refuse unexpected mutable data.
    if any(type(v) not in (str, bool, int, float) for v in result.details.values()):
        return
    value = tuple(result.details.items())
    cost = 512 + _size(key) + _size(value)
    if cost > _MAX_BYTES:
        count('oversized')
        return
    previous = _VALUES.pop(key, None)
    if previous is not None:
        _BYTES -= previous[1]
    while _VALUES and (len(_VALUES) >= _MAX_ENTRIES or _BYTES + cost > _MAX_BYTES):
        _, (_, old_cost) = _VALUES.popitem(last=False)
        _BYTES -= old_cost
        count('eviction')
    _VALUES[key] = (value, cost)
    _BYTES += cost
    count('store')
