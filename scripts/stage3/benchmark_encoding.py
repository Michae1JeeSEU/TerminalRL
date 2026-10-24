"""Development-only microbenchmark; never runs episodes or training.

LegacyFeatureEncoder is the bounded pre-optimization encoder from 2f73845,
retained as an independent development oracle, not a second production module.
"""
from collections.abc import Mapping
import math
from terminal_marl.core.rotating_slots.stage3_learning.features import (
    SCHEMA_ID, SET_LIMITS, RINGS, QUERY_DIM, SET_DIMS, FeatureInputError,
    get, row, hot, pos, relation, FeatureEncoder)
from terminal_marl.core.rotating_slots.geometry import slot_state, pad_position

class LegacyFeatureEncoder:
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


def legacy_transform(normalizer, feature):
    """Pre-optimization float64 Python normalization, independent copy path."""
    from copy import deepcopy
    from terminal_marl.core.rotating_slots.stage3_learning.model import _validate_feature
    _validate_feature(feature)
    result = deepcopy(feature)
    for name, rows in [('query', [result['query']]), *result['sets'].items()]:
        for r in rows:
            for j, statistic in enumerate(normalizer.statistics[name]):
                r[2*j] = ((r[2*j]-statistic['mean'])/statistic['std']) if r[2*j+1] else 0.
    return result


def legacy_forward(model, features):
    """Original per-candidate tensor assembly with the identical model modules."""
    import torch
    from terminal_marl.core.rotating_slots.stage3_learning.model import _validate_feature
    if not features:
        return torch.empty((0,3), dtype=torch.float32, device='cpu')
    for feature in features:
        _validate_feature(feature)
    query = torch.tensor([f['query'] for f in features], dtype=torch.float32, device='cpu')
    chunks = [model.query_encoder(query)]
    for name, dim in SET_DIMS.items():
        size = max((len(f['sets'][name]) for f in features), default=0)
        data = torch.zeros((len(features),size,dim), dtype=torch.float32, device='cpu')
        mask = torch.zeros((len(features),size), dtype=torch.bool, device='cpu')
        for i, feature in enumerate(features):
            rows = feature['sets'][name]
            if rows:
                data[i,:len(rows)] = torch.tensor(rows, dtype=torch.float32, device='cpu')
                mask[i,:len(rows)] = True
        chunks.append(model.pool(model.entity_encoders[name](data),mask,SET_LIMITS[name]))
    return model.score_head(torch.cat(chunks,dim=-1))


def legacy_rank(model, normalizer, features):
    import torch
    with torch.inference_mode():
        prediction = legacy_forward(model, [legacy_transform(normalizer, f) for f in features]).tolist()
    ids = [f['descriptor_id'] for f in features]
    scores = {did: r[0] for did,r in zip(ids,prediction)}
    return tuple(sorted(ids,key=lambda did:(-scores[did],ids.index(did)))), scores


def development_fixture(aircraft_count=24, batch_size=16, reservation_count=32):
    """Artificial development input, never a heldout snapshot or episode."""
    from types import SimpleNamespace
    from terminal_marl.core.rotating_slots.config import load_config
    from terminal_marl.core.rotating_slots.world import AircraftState
    from terminal_marl.core.rotating_slots.resources import Reservation
    config = load_config()
    aircraft, descriptors = {}, []
    for i in range(aircraft_count):
        ring = RINGS[i%3]
        count = next(r.slot_count for r in config.airspace.rings if r.id == ring)
        slot = i%count
        point = slot_state(ring,slot,10.,config)
        aid = 'development-aircraft-'+str(i)
        aircraft[aid] = AircraftState(aid,'HOLD_'+ring.upper(),.7+i%3*.05,
            point.p,point.v,point.a,point.jerk,ring=ring,slot=slot)
    for i in range(batch_size):
        aid = list(aircraft)[i%aircraft_count]
        a = aircraft[aid]
        descriptors.append(dict(descriptor_id='development-work-'+str(i),aircraft_id=aid,
            source_node_id='node-'+aid,source_ring=a.ring,source_slot=a.slot,
            target_kind='normal_pad' if a.ring == 'inner' else 'next_ring_slot',
            target_id=i%config.pads.count if a.ring == 'inner' else i%8,
            start_offset_s=float(i%4)*5.,duration_s=30.+i%3,
            parent_prefix_id=None,
            construction_stage=('progress','reference','alternative')[i%3],
            proposal_origin=('current','delayed','resource_aligned','reuse')[i%4]))
    reservations = tuple(Reservation((RINGS[i%3],i%8),list(aircraft)[i%aircraft_count],
        float(i),math.inf if i%5 == 0 else float(i+40),committed=i%2 == 0)
        for i in range(reservation_count))
    snapshot = SimpleNamespace(time=10.,config=config,aircraft=aircraft,active={},soft={},
        slots=reservations,pads=(Reservation(0,list(aircraft)[0],0.,math.inf),))
    return snapshot, descriptors


def development_state(snapshot, step):
    nodes, parents = {}, {}
    for aid,a in snapshot.aircraft.items():
        node = dict(materialized=True,soc=a.soc-.01*(step%3),ring=a.ring,slot=a.slot,
            end_time=10.+float(step%3),depth=step%3,peak=float(step%3))
        nodes['node-'+aid] = node
        parents['parent-'+aid] = dict(node)
    return dict(nodes=nodes,parents=parents,attempts=step,materialized=step//2,
        remaining_budget_fraction=max(0.,1.-step/1000.))


def run_benchmark(iterations=12, batch_size=16, aircraft_count=24, checkpoint=None):
    import copy
    import hashlib
    import time
    import torch
    from terminal_marl.core.rotating_slots.stage3_learning.model import (
        build_model, FeatureNormalizer, LearnedAdvisor, ARCHITECTURE_ID)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        assert torch.get_num_interop_threads() == 1
    snapshot, descriptors = development_fixture(aircraft_count,batch_size)
    for i,descriptor in enumerate(descriptors):
        if i%2:
            descriptor['parent_prefix_id'] = 'parent-'+descriptor['aircraft_id']
    legacy, encoder = LegacyFeatureEncoder(), FeatureEncoder()
    states = [development_state(snapshot,i) for i in range(iterations)]
    oracle_features = [legacy.encode(snapshot,descriptors,state) for state in states]
    optimized_features = [encoder.encode(snapshot,descriptors,state) for state in states]
    assert oracle_features == optimized_features
    torch.manual_seed(20261024)
    model = build_model().eval()
    checkpoint_hash = None
    if checkpoint:
        from pathlib import Path
        path = Path(checkpoint)
        if path.is_dir():
            path = path/'model.pt'
        checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        data = torch.load(path,map_location='cpu',weights_only=True)
        assert data['architecture_id'] == ARCHITECTURE_ID and data['schema_id'] == SCHEMA_ID
        model.load_state_dict(data['state_dict'],strict=True)
        normalizer = FeatureNormalizer.from_dict(data['normalizer'])
    else:
        normalizer = FeatureNormalizer().fit([dict(split='train',features=oracle_features[0])])
    weights = {k:v.clone() for k,v in model.state_dict().items()}
    statistics = copy.deepcopy(normalizer.as_dict())
    advisor = LearnedAdvisor('development-in-memory-only')
    advisor._model, advisor._normalizer, advisor._metadata = model,normalizer,{}
    max_difference = 0.
    for old,new in zip(oracle_features,optimized_features):
        expected_order, expected_scores = legacy_rank(model,normalizer,old)
        actual = advisor.rank(new)
        assert actual.ordered_ids == expected_order
        max_difference = max(max_difference,max((abs(actual.scores[k]-v) for k,v in expected_scores.items()),default=0.))
        assert max_difference <= 1e-7
    # Warmup is excluded. Key construction, output copies and changing query rows
    # are included in encoder timings; rank includes normalization and tensors.
    for state in states[:2]:
        legacy.encode(snapshot,descriptors,state)
        encoder.encode(snapshot,descriptors,state)
    legacy_rank(model,normalizer,oracle_features[0])
    advisor.rank(optimized_features[0])
    def timed(function, inputs):
        started = time.perf_counter()
        for item in inputs:
            function(item)
        return time.perf_counter()-started
    times = dict(
        legacy_encoder_s=timed(lambda state:legacy.encode(snapshot,descriptors,state),states),
        optimized_encoder_s=timed(lambda state:encoder.encode(snapshot,descriptors,state),states),
        legacy_rank_s=timed(lambda features:legacy_rank(model,normalizer,features),oracle_features),
        optimized_rank_s=timed(advisor.rank,optimized_features),
        legacy_end_to_end_s=timed(lambda state:legacy_rank(model,normalizer,legacy.encode(snapshot,descriptors,state)),states),
        optimized_end_to_end_s=timed(lambda state:advisor.rank(encoder.encode(snapshot,descriptors,state)),states))
    assert statistics == normalizer.as_dict()
    assert all(torch.equal(weights[k],v) for k,v in model.state_dict().items())
    if checkpoint:
        assert checkpoint_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(protocol='artificial_development_microbenchmark_v1',baseline_commit='2f738458940eb0f7d02ef369be2a9f483577f961',
        iterations=iterations,descriptors_per_call=batch_size,aircraft=aircraft_count,
        encoder_calls_per_timed_path=iterations,rank_calls_per_timed_path=iterations,
        entity_cache_entries=len(encoder._entity_cache),features_exact=True,order_exact=True,
        max_score_difference=max_difference,weights_unchanged=True,normalizer_unchanged=True,
        checkpoint_sha256=checkpoint_hash,torch_threads=torch.get_num_threads(),timings=times,
        scope='Warm repeated artificial inputs with changing budget/attempts/parent state; includes content keys and copies; excludes episodes, CP, physics and online benefit claims.')


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--aircraft-count',type=int,default=24)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 50 or not 1 <= args.batch_size <= 64 or not 1 <= args.aircraft_count <= 64:
        parser.error('bounded development workload: iterations 1..50, batch 1..64, aircraft 1..64')
    result = run_benchmark(args.iterations,args.batch_size,args.aircraft_count,args.checkpoint)
    rendered = json.dumps(result,indent=2,sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+'\n',encoding='utf8')
    print(rendered)
