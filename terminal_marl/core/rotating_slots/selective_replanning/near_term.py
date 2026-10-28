"""Cheap, resource-specific opportunities, independent of FOLLOW tail scores.

Rows are ordering estimates, never feasibility or safety certificates. No cold
reference calculation or candidate generation is performed here.
"""
import math


def summarize_opportunities(snapshot, eligible, references=None, *, horizon_seconds=240.):
    from ..stage2_planning import _node, remaining_normal_time_lower
    from ..trajectory import ring_config
    nodes={aid:_node(snapshot,a) for aid,a in eligible.items()}
    end=snapshot.time+horizon_seconds
    cache=getattr(references,"cache",{})
    rows={}
    for aid,node in nodes.items():
        key=(snapshot.config.config_hash,node.ring,node.slot,node.time,node.version,aid)
        ref=cache.get(key)
        applicable=bool(ref is not None and ref.status=="KNOWN" and ref.plans and
            (ref.plans[0].source_ring,ref.plans[0].source_slot,ref.plans[0].source_version)==
            (node.ring,node.slot,node.version) and ref.plans[0].start_time>=node.time-1.e-8)
        next_ring={"outer":"middle","middle":"inner","inner":None}[node.ring]
        count=snapshot.config.pads.count if next_ring is None else ring_config(next_ring,snapshot.config).slot_count
        lower=remaining_normal_time_lower(snapshot,node.ring)
        step=max(20.,lower-(remaining_normal_time_lower(snapshot,next_ring) if next_ring else 13.5))
        options=[]
        targets=(ref.plans[0].pad_id if next_ring is None else ref.plans[0].target_slot,) if applicable else range(count)
        for target in targets:
            resource=("pad",target) if next_ring is None else (next_ring,target)
            reservations=snapshot.pads if next_ring is None else snapshot.slots
            occupied=[r for r in reservations if (r.resource==target if next_ring is None else r.resource==resource)
                and r.aircraft_id!=aid and r.end>node.time]
            finite=max((r.end for r in occupied if math.isfinite(r.end)),default=node.time)
            indefinite=[r for r in occupied if not math.isfinite(r.end)]
            members=[]; unknown=False
            for r in indefinite:
                owner=nodes.get(r.aircraft_id)
                # Only this specific owner/slot can supply the potential release.
                if owner is None or (owner.ring,owner.slot)!=resource:
                    unknown=True; break
                owner_lower=remaining_normal_time_lower(snapshot,owner.ring)
                after={"outer":"middle","middle":"inner","inner":None}[owner.ring]
                owner_step=max(20.,owner_lower-(remaining_normal_time_lower(snapshot,after) if after else 13.5))
                finite=max(finite,owner.time+owner_step)
                members.append(r.aircraft_id)
            if unknown: continue
            start=max(node.time,finite)
            arrival=start+step
            if applicable: arrival=max(arrival,ref.plans[0].end_time)
            if arrival>end: continue
            options.append(dict(resource=resource,release_time=finite,earliest_arrival=arrival,
                joint_members=tuple(sorted(set(members))),
                source="applicable_cached_path" if applicable else "shortest_arrival_resource_estimate",
                release_source="potential_owner_maneuver" if members else "finite_calendar_or_free_resource",
                certified=False))
        options.sort(key=lambda row:(row["earliest_arrival"],len(row["joint_members"]),repr(row["resource"])))
        best=options[0] if options else None
        rows[aid]=dict(source=best["source"] if best else "UNKNOWN",unknown=best is None,
            earliest_arrival=None if best is None else best["earliest_arrival"],
            resource=None if best is None else best["resource"],
            joint_members=() if best is None else best["joint_members"],
            options=tuple(options),horizon_end=end,cached_reference_applicable=applicable)
    return rows


def cooperative_seed_groups(opportunities, *, max_groups=3):
    """Up to three deterministic alternate starting groups for same-state audit."""
    if not 1<=max_groups<=3: raise ValueError("at most three diagnostic groups")
    ordered=sorted(opportunities,key=lambda aid:(opportunities[aid]["unknown"],
        opportunities[aid]["earliest_arrival"] if opportunities[aid]["earliest_arrival"] is not None else math.inf,aid))
    candidates=[]
    for aid in ordered:
        members={aid}; frontier=[aid]
        while frontier:
            current=frontier.pop()
            for owner in opportunities[current]["joint_members"]:
                if owner in opportunities and owner not in members:
                    members.add(owner); frontier.append(owner)
        # Follow the release in reverse: a downstream anchor's specific source
        # slot admits one best upstream follower, then its source can admit the
        # next. Alternative targets remain ordering estimates, never hard edges.
        while len(members)<3:
            followers=[]
            for follower,row in opportunities.items():
                if follower in members: continue
                for option in row["options"]:
                    owners=set(option["joint_members"])
                    if owners and owners<=members:
                        followers.append((option["earliest_arrival"],repr(option["resource"]),follower))
            if not followers: break
            members.add(min(followers)[2])
        group=tuple(sorted(members))
        if group not in [row[1] for row in candidates]:
            candidates.append(((len(group)==1,opportunities[aid]["earliest_arrival"] or math.inf,aid),group))
    groups=[]
    for _,group in sorted(candidates):
        groups.append(group)
        if len(groups)>=max_groups: break
    return tuple(groups)
