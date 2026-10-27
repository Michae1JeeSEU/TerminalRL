"""Explicit open variables and immutable complete-world bindings."""
from dataclasses import dataclass, asdict
import hashlib


def digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


@dataclass(frozen=True)
class PlanningScope:
    snapshot_id: str
    effective_time: float
    eligible_ids: tuple
    replan_ids: tuple
    fixed_background_ids: tuple
    mandatory_ids: tuple = ()
    nominal_k: int = 8
    actual_k: int = 0
    max_k: int = 12
    selection_reasons: tuple = ()
    relation_digest: str = ""
    objective_frame_id: str = ""
    status: str = "LOCAL"

    def __post_init__(self):
        for name in ("eligible_ids", "replan_ids", "fixed_background_ids", "mandatory_ids"):
            values = tuple(getattr(self, name))
            if len(values) != len(set(values)): raise ValueError("duplicate scope IDs")
            object.__setattr__(self, name, values)
        s, u, b, m = map(set, (self.replan_ids, self.eligible_ids, self.fixed_background_ids, self.mandatory_ids))
        if not s <= u or s & b or not m <= u: raise ValueError("invalid planning scope partition")
        if self.actual_k != len(s): raise ValueError("actual_k does not match open variables")
        if self.status != "MANDATORY_OVERFLOW_KEEP" and not m <= s: raise ValueError("mandatory coverage missing")
        if len(s) > self.max_k and self.status not in ("FULL_H", "MANDATORY_OVERFLOW_FULL"):
            raise ValueError("local scope exceeds max_k")

    @property
    def identity(self): return digest(self)

    def validate(self, snapshot):
        from ..stage2_planning import _decision_aircraft
        u = {a.id for a in _decision_aircraft(snapshot)}
        if self.snapshot_id != snapshot.version or self.effective_time != snapshot.time:
            raise ValueError("scope snapshot binding mismatch")
        if set(self.eligible_ids) != u or set(self.replan_ids) | set(self.fixed_background_ids) != set(snapshot.aircraft):
            raise ValueError("scope does not preserve complete world")
        return self

    def to_dict(self): return asdict(self)
