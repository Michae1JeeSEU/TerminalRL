"""Cheap revealed resource relations; these are never safety certificates."""
from dataclasses import dataclass
from .scope import digest


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    kind: str
    definite: bool = False
    required: bool = False
    resource: tuple = ()


def build_relations(snapshot, eligible_ids, previous_blockers=(), frame=None):
    rows = set()
    # Real release dependencies and occupied next-ring slots are cheap. A slot
    # edge is an opportunity relation, not proof that this target is necessary.
    next_ring = {"outer":"middle", "middle":"inner"}
    for aid in eligible_ids:
        a = snapshot.aircraft[aid]
        for r in snapshot.slots:
            resource = r.resource
            if isinstance(resource, tuple) and resource[0] == next_ring.get(a.ring) and r.aircraft_id != aid:
                rows.add(Relation(aid, r.aircraft_id, "next_ring_occupancy", False, False, resource))
        for p in (snapshot.active.get(aid), snapshot.soft.get(aid)):
            if p is None: continue
            for h in snapshot.dependencies.get(p.trajectory_hash, ()):
                release=next((q for q in snapshot.active.values() if q.trajectory_hash==h),None)
                if release is not None: rows.add(Relation(aid, release.aircraft_id, "committed_release", True))
    # The known pad calendar and shared virtual assignment connect even distant
    # aircraft. Competition is approximate and can never force expansion.
    for aid in eligible_ids:
        for r in snapshot.pads:
            if r.aircraft_id != aid:
                rows.add(Relation(aid,r.aircraft_id,"pad_release",False,False,("pad",r.resource)))
    for i,aid in enumerate(eligible_ids):
        for other in eligible_ids[i+1:]:
            if snapshot.aircraft[aid].ring==snapshot.aircraft[other].ring=="inner":
                rows.add(Relation(aid,other,"normal_pad_competition"))
            if frame is not None:
                assignment=dict(frame.assignment)
                if aid in assignment and assignment.get(aid)==assignment.get(other):
                    rows.add(Relation(aid,other,"virtual_pad_competition",False,False,("pad",assignment[aid])))
    for row in previous_blockers:
        if row.get("snapshot_id") != snapshot.version: continue
        source, target = row["source"], row["target"]
        if source in snapshot.aircraft and target in snapshot.aircraft:
            rows.add(Relation(source, target, "validated_background_blocker", True, True, tuple(row.get("resource", ()))))
    return tuple(sorted(rows, key=lambda r:(r.source, r.target, r.kind, repr(r.resource))))


def relation_digest(relations): return digest(relations)
