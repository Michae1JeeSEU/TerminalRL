"""Read-only, cheap deployment features; no simulator or torch imports."""
from collections.abc import Mapping
import math
from ..geometry import slot_state, pad_position

SCHEMA_ID = 'candidate_context_sets_v1_0'
SET_LIMITS = dict(aircraft=256, slots=128, pads=16, reservations=1024)
RINGS = ('outer', 'middle', 'inner')
QUERY_DIM = 66
SET_DIMS = dict(aircraft=84, slots=34, pads=26, reservations=34)

class FeatureInputError(ValueError):
    def __init__(self, reason, detail=''):
        self.reason = reason
        super().__init__(reason + (': '+detail if detail else ''))

def get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)

def row(values):
    out = []
    for item in values:
        value, scale = item if isinstance(item, tuple) else (item, 1.)
        if value is None:
            out.extend((0., 0.)); continue
        try:
            value = float(value)/scale
        except (ValueError, TypeError):
            raise FeatureInputError('INPUT_OOD', 'nonnumeric field') from None
        if not math.isfinite(value):
            raise FeatureInputError('INPUT_OOD', 'nonfinite visible field')
        out.extend((value, 1.))
    return out

def hot(value, choices=RINGS):
    return [float(value == choice) for choice in choices]

def pos(obj):
    p = get(obj, 'p')
    return tuple(p) if p is not None and len(p) == 3 else (None,)*3

def relation(p, source):
    if any(v is None for v in (*p, *source)):
        return [None]*6
    delta = [p[i]-source[i] for i in range(3)]
    angle = math.atan2(p[1], p[0])-math.atan2(source[1], source[0])
    return [*( (v, 900.) for v in delta), (math.sqrt(sum(v*v for v in delta)), 900.), math.sin(angle), math.cos(angle)]

class FeatureEncoder:
    """Fixed physical scales precede a train-fitted normalizer.

    Only allowlisted visible fields and actual materialized nodes are read.
    Identifiers establish relations and never enter numeric inputs.
    """
    def __init__(self, horizon_s=240., energy_capacity_j=None, entity_limits=None):
        if not math.isfinite(horizon_s) or horizon_s <= 0:
            raise ValueError('positive bound horizon required')
        self.horizon_s = float(horizon_s)
        self.energy_capacity_j = energy_capacity_j
        self.entity_limits = dict(SET_LIMITS)
        if entity_limits:
            self.entity_limits.update(entity_limits)

    def encode(self, snapshot, descriptors, observed_search_state=None):
        state = observed_search_state or {}
        cfg, now = get(snapshot, 'config'), float(get(snapshot, 'time', 0.))
        aircraft = get(snapshot, 'aircraft', {})
        aircraft = list(aircraft.values()) if isinstance(aircraft, Mapping) else list(aircraft)
        airspace = get(cfg, 'airspace')
        rings = list(get(airspace, 'rings', ()))
        slots = [(get(r,'id'), i, slot_state(get(r,'id'), i, now, cfg).p)
                 for r in rings for i in range(int(get(r,'slot_count',0)))]
        pc = get(cfg, 'pads')
        pads = [(i,pad_position(i,cfg)) for i in range(int(get(pc,'count',0)))]
        reservations = [('slot', r) for r in get(snapshot, 'slots', ())]+[('pad', r) for r in get(snapshot, 'pads', ())]
        counts = dict(aircraft=len(aircraft), slots=len(slots), pads=len(pads), reservations=len(reservations))
        if any(counts[k] > self.entity_limits[k] for k in counts):
            raise FeatureInputError('ENTITY_LIMIT', str(counts))
        capacity = self.energy_capacity_j
        if capacity is None:
            energy = get(cfg, 'energy'); ah = get(energy, 'reference_capacity_Ah'); voltage = get(energy, 'equivalent_voltage_V')
            capacity = float(ah)*float(voltage)*3600. if ah and voltage else None
        output = []
        for descriptor in descriptors:
            d = descriptor.as_dict() if hasattr(descriptor, 'as_dict') else descriptor
            did, aid = get(d, 'descriptor_id'), get(d, 'aircraft_id')
            own = next((a for a in aircraft if get(a, 'id') == aid), None)
            if own is None or not did:
                raise FeatureInputError('ILLEGAL_DESCRIPTOR_ID')
            parent = get(d, 'parent_prefix_id')
            node = state.get('parents', {}).get(parent) if parent is not None else state.get('nodes', {}).get(get(d, 'source_node_id'))
            if (parent is not None or node is not None) and (not node or not get(node, 'materialized', False)):
                raise FeatureInputError('INPUT_OOD', 'unmaterialized parent')
            sr, tk, ti = get(d, 'source_ring'), get(d, 'target_kind'), get(d, 'target_id')
            tr = RINGS[RINGS.index(sr)+1] if tk == 'next_ring_slot' and sr in RINGS[:-1] else None
            resource = (tr, ti) if tk == 'next_ring_slot' else ti
            soc = get(node,'soc') if node else get(own,'soc')
            if soc is not None and (not math.isfinite(float(soc)) or not 0. <= float(soc) <= 1.):
                raise FeatureInputError('INPUT_OOD','SOC outside [0,1]')
            if float(get(d,'start_offset_s',-1)) < 0 or float(get(d,'duration_s',0)) <= 0:
                raise FeatureInputError('INPUT_OOD','descriptor time')
            source = pos(own)
            if node:
                source = slot_state(get(node,'ring'),get(node,'slot'),get(node,'end_time'),cfg).p
            start = now+float(get(d, 'start_offset_s', 0.)); end = start+float(get(d, 'duration_s', 0.))
            downstream = state.get('downstream',{}).get(get(d,'source_node_id'),{})
            peak = get(node,'peak') if node else get(own,'peak_seconds')
            committed = [p for plans in (get(snapshot,'active',{}),get(snapshot,'soft',{}))
                         for key,value in plans.items() if key == aid
                         for p in (value if isinstance(value,(tuple,list)) else (value,))]
            commitment_end = max((get(p,'end_time',now) for p in committed),default=now)
            query = row([*hot(sr), *hot(tr), float(tk == 'normal_pad'),
                (get(d, 'start_offset_s'), self.horizon_s), (get(d, 'duration_s'), self.horizon_s),
                *hot(get(d, 'construction_stage'), ('progress','reference','alternative')),
                *hot(get(d, 'proposal_origin'), ('current','delayed','resource_aligned','reuse')),
                get(node, 'soc') if node else get(own, 'soc'), get(node, 'depth', 0) if node else 0,
                float(parent is not None), (state.get('attempts'), 2048.), (state.get('materialized'),2048.),
                state.get('remaining_budget_fraction'), *[(v,900.) for v in source],
                (peak,120.), ((120.-peak),120.) if peak is not None else None,
                (get(node,'end_time')-now,self.horizon_s) if node and get(node,'end_time') is not None else None,
                (get(own,'holding_time'),self.horizon_s),
                (now-get(own,'admission_time'),self.horizon_s) if get(own,'admission_time') is not None else None,
                (commitment_end-now,self.horizon_s),
                (get(downstream,'energy_j'),capacity) if capacity and get(downstream,'status') == 'KNOWN' else None,
                (get(downstream,'duration_s'),self.horizon_s) if get(downstream,'status') == 'KNOWN' else None])
            sets = dict(aircraft=[], slots=[], pads=[], reservations=[])
            active, soft = get(snapshot, 'active', {}), get(snapshot, 'soft', {})
            for a in aircraft:
                p, e, identity = pos(a), get(a, 'emergency', {}), get(a, 'id')
                if not 0. <= float(get(a,'soc')) <= 1.:
                    raise FeatureInputError('INPUT_OOD','SOC outside [0,1]')
                sets['aircraft'].append(row([get(a,'soc'), *hot(get(a,'ring')), *[(v,900.) for v in p],
                    *[(v,25.) for v in get(a,'v',(None,)*3)], *[(v,1.5) for v in get(a,'a',(None,)*3)],
                    (get(a,'battery_power_w'),100000.), (get(a,'energy_j'),capacity) if capacity else None,
                    *relation(p,source), float(identity == aid),
                    float((get(a,'ring'),get(a,'slot')) == resource) if isinstance(resource,tuple) else float(get(a,'pad_id') == resource),
                    float(get(a,'ring') == sr),float(get(a,'ring') == tr),float(identity in active),float(identity in soft),
                    float(get(e,'declared',False)),float(get(e,'pending',False)),get(e,'q_e'),get(e,'normal_margin'),
                    (get(e,'rescue_slack_seconds'),self.horizon_s), *hot(get(a,'service_class'),('short','standard','long')),
                    float(get(a,'admitted',True)),float(get(a,'trajectory_hash') is not None),
                    (get(a,'peak_seconds'),120.),(get(a,'holding_time'),self.horizon_s),
                    (now-get(a,'admission_time'),self.horizon_s) if get(a,'admission_time') is not None else None,
                    (get(a,'upstream_wait_seconds'),self.horizon_s),(get(a,'airborne_time'),self.horizon_s)]))
            for ring, index, p in slots:
                matches = [r for kind,r in reservations if kind == 'slot' and tuple(get(r,'resource')) == (ring,index)]
                sets['slots'].append(row([*hot(ring),*[(v,900.) for v in p],*relation(p,source),float((ring,index) == resource),
                    float(ring == sr),float(ring == tr),(len(matches),self.entity_limits['reservations']),float(any(get(r,'committed',False) for r in matches))]))
            for index,p in pads:
                matches = [r for kind,r in reservations if kind == 'pad' and get(r,'resource') == index]
                sets['pads'].append(row([*[(v,900.) for v in p],*relation(p,source),float(tk == 'normal_pad' and index == ti),
                    (len(matches),self.entity_limits['reservations']),float(any(get(r,'committed',False) for r in matches)),float(any(get(r,'aircraft_id') == aid for r in matches))]))
            for kind,r in reservations:
                res = get(r,'resource'); res = tuple(res) if isinstance(res,(tuple,list)) else res
                rs,re = get(r,'start'),get(r,'end'); infinite_end = re is None or re == math.inf; re = None if infinite_end else re; ring = res[0] if isinstance(res,tuple) else None
                sets['reservations'].append(row([float(kind == 'pad'),*hot(ring),float(res == resource and (kind == 'pad') == (tk == 'normal_pad')),
                    float(get(r,'aircraft_id') == aid),float(get(r,'committed',False)),
                    (rs-now,self.horizon_s) if rs is not None else None,(re-now,self.horizon_s) if re is not None else None,
                    (rs-start,self.horizon_s) if rs is not None else None,(re-end,self.horizon_s) if re is not None else None,
                    float(infinite_end),float(ring == sr),float(ring == tr),*hot(get(r,'kind'),('source','target','service'))]))
            if len(query) != QUERY_DIM or any(len(r) != SET_DIMS[k] for k,rows in sets.items() for r in rows):
                raise RuntimeError('feature schema dimensions: '+str((len(query),{k:len(v[0]) if v else 0 for k,v in sets.items()})))
            output.append(dict(descriptor_id=did,schema_id=SCHEMA_ID,query=query,sets=sets,
                query_mask=query[1::2],set_masks={k:[r[1::2] for r in rows] for k,rows in sets.items()}))
        return output



