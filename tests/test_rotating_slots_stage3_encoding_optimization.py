"""Independent pre-optimization references and bounded cache regressions."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import math

import pytest

from scripts.stage3.benchmark_encoding import (
    LegacyFeatureEncoder, legacy_transform, legacy_forward, legacy_rank,
    development_fixture, development_state)
from terminal_marl.core.rotating_slots.stage3_learning.features import FeatureEncoder, FeatureInputError
from terminal_marl.core.rotating_slots.stage3_learning.model import FeatureNormalizer, build_model, LearnedAdvisor
from terminal_marl.core.rotating_slots.resources import Reservation


def compare(encoder, snapshot, descriptors, state=None):
    expected = LegacyFeatureEncoder(horizon_s=encoder.horizon_s,
        energy_capacity_j=encoder.energy_capacity_j,entity_limits=encoder.entity_limits).encode(snapshot,descriptors,state)
    actual = encoder.encode(snapshot,descriptors,state)
    assert actual == expected
    return actual


def test_dynamic_queries_and_parent_geometry_match_legacy_exactly():
    snapshot, descriptors = development_fixture()
    for i,descriptor in enumerate(descriptors):
        if i%2:
            descriptor['parent_prefix_id'] = 'parent-'+descriptor['aircraft_id']
    encoder = FeatureEncoder()
    prior = None
    for step in (0,1,2,3,25,0):
        state = development_state(snapshot,step)
        state['downstream'] = {descriptors[0]['source_node_id']:
            dict(status='KNOWN',energy_j=1000.+step,duration_s=20.+step)}
        actual = compare(encoder,snapshot,descriptors,state)
        assert actual == FeatureEncoder(cache_enabled=False).encode(snapshot,descriptors,state)
        if prior is not None:
            assert actual[0]['query'] != prior[0]['query']
        prior = actual
    assert len(encoder._entity_cache) <= 256
    # All roots/parents are validated even when entity rows were cached.
    state['parents'][descriptors[1]['parent_prefix_id']]['materialized'] = False
    with pytest.raises(FeatureInputError,match='unmaterialized'):
        encoder.encode(snapshot,descriptors,state)


@pytest.mark.parametrize('change', ['soc','position','velocity','emergency','reservation','active','soft','time','aircraft_order'])
def test_same_mutable_snapshot_content_invalidates(change):
    snapshot, descriptors = development_fixture()
    encoder = FeatureEncoder()
    before = compare(encoder,snapshot,descriptors)
    aid = descriptors[0]['aircraft_id']
    own = snapshot.aircraft[aid]
    if change == 'soc':
        snapshot.aircraft[aid] = replace(own,soc=.51)
    elif change == 'position':
        snapshot.aircraft[aid] = replace(own,p=(50.,60.,70.))
    elif change == 'velocity':
        snapshot.aircraft[aid] = replace(own,v=(1.,2.,3.))
    elif change == 'emergency':
        snapshot.aircraft[aid] = replace(own,emergency=replace(own.emergency,pending=True,normal_margin=.15))
    elif change == 'reservation':
        snapshot.slots += (Reservation(('middle',0),aid,20.,None,committed=True),)
    elif change in ('active','soft'):
        setattr(snapshot,change,{aid:SimpleNamespace(end_time=95.)})
    elif change == 'time':
        snapshot.time = 14.
    else:
        snapshot.aircraft = dict(reversed(list(snapshot.aircraft.items())))
    assert compare(encoder,snapshot,descriptors) != before


def test_mutable_config_and_descriptor_relations_invalidate():
    snapshot, descriptors = development_fixture()
    def namespace(value):
        if isinstance(value,dict):
            return SimpleNamespace(**{k:namespace(v) for k,v in value.items()})
        if isinstance(value,list):
            return [namespace(v) for v in value]
        return value
    snapshot.config = namespace(snapshot.config.to_dict())
    encoder = FeatureEncoder()
    first = compare(encoder,snapshot,descriptors)
    snapshot.config.airspace.rings[0].radius_m += 2.
    assert compare(encoder,snapshot,descriptors) != first
    snapshot.config.pads.layout_radius_m += 1.
    snapshot.config.energy.reference_capacity_Ah += 1.
    compare(encoder,snapshot,descriptors)
    for changes in (dict(target_id=5),dict(start_offset_s=40.),dict(duration_s=50.),
                    dict(aircraft_id=descriptors[3]['aircraft_id']),
                    dict(source_ring='middle',target_kind='normal_pad',target_id=0)):
        compare(encoder,snapshot,[dict(descriptors[0],**changes)])


def test_missing_to_explicit_none_and_invalid_mutation_match_legacy():
    snapshot, descriptors = development_fixture(3,1,0)
    aid = descriptors[0]['aircraft_id']
    own = dict(vars(snapshot.aircraft[aid]))
    del own['admitted']
    snapshot.aircraft[aid] = own
    encoder = FeatureEncoder()
    before = compare(encoder,snapshot,descriptors)
    own['admitted'] = None
    for candidate in (encoder,LegacyFeatureEncoder()):
        with pytest.raises(TypeError):
            candidate.encode(snapshot,descriptors)
    own['admitted'] = False
    assert compare(encoder,snapshot,descriptors) != before
    own.pop('v')
    compare(encoder,snapshot,descriptors)
    own['v'] = None
    for candidate in (encoder,LegacyFeatureEncoder()):
        with pytest.raises(TypeError):
            candidate.encode(snapshot,descriptors)


def test_equal_replacement_does_not_retain_old_mutable_objects():
    snapshot, descriptors = development_fixture(3,2,2)
    aid = descriptors[0]['aircraft_id']
    old = dict(vars(snapshot.aircraft[aid]))
    snapshot.aircraft[aid] = old
    encoder = FeatureEncoder()
    compare(encoder,snapshot,descriptors)
    replacement = deepcopy(snapshot,{id(snapshot.config):snapshot.config})
    compare(encoder,replacement,descriptors)
    old['soc'] = .01
    compare(encoder,replacement,descriptors)
    old_current = dict(vars(replacement.aircraft[descriptors[1]['aircraft_id']]))
    replacement.aircraft[descriptors[1]['aircraft_id']] = old_current
    compare(encoder,replacement,descriptors)
    replacement.aircraft[descriptors[1]['aircraft_id']] = dict(old_current)
    compare(encoder,replacement,descriptors)
    old_current['soc'] = .02
    compare(encoder,replacement,descriptors)


def test_outputs_are_unaliased_and_cache_is_bounded():
    snapshot, descriptors = development_fixture(3,2,2)
    encoder = FeatureEncoder()
    original = compare(encoder,snapshot,descriptors)
    retained = deepcopy(original)
    again = encoder.encode(snapshot,descriptors)
    again[0]['query'][0] = -99.
    again[0]['query_mask'][0] = -99.
    for key in again[0]['sets']:
        if again[0]['sets'][key]:
            again[0]['sets'][key][0][0] = -99.
            again[0]['set_masks'][key][0][0] = -99.
    assert original == retained
    compare(encoder,snapshot,descriptors)
    for offset in range(260):
        encoder.encode(snapshot,[dict(descriptors[0],start_offset_s=float(offset))])
    assert len(encoder._entity_cache) == 256
    encoder.clear_cache()
    assert not encoder._entity_cache and encoder._prepared is None
    compare(encoder,snapshot,descriptors)


def test_empty_missing_infinite_reservations_limits_and_ood():
    snapshot, descriptors = development_fixture(1,1,0)
    snapshot.pads = ()
    snapshot.aircraft[next(iter(snapshot.aircraft))] = replace(
        next(iter(snapshot.aircraft.values())),p=(None,None,None),admission_time=None)
    encoder = FeatureEncoder()
    compare(encoder,snapshot,descriptors)
    snapshot.slots = (Reservation(('outer',0),'a',None,None),
                      Reservation(('middle',0),'b',0.,math.inf))
    compare(encoder,snapshot,descriptors)
    for candidate in (encoder,LegacyFeatureEncoder()):
        candidate.entity_limits['reservations'] = 1
        with pytest.raises(FeatureInputError,match='ENTITY_LIMIT'):
            candidate.encode(snapshot,descriptors)
    snapshot, descriptors = development_fixture(1,1,0)
    encoder = FeatureEncoder()
    compare(encoder,snapshot,descriptors)
    aid = descriptors[0]['aircraft_id']
    for changes in (dict(soc=math.nan),dict(soc=1.1),dict(p=(math.nan,0.,0.))):
        snapshot.aircraft[aid] = replace(snapshot.aircraft[aid],**changes)
        for candidate in (encoder,LegacyFeatureEncoder()):
            with pytest.raises(FeatureInputError,match='INPUT_OOD'):
                candidate.encode(snapshot,descriptors)
        snapshot.aircraft[aid] = replace(snapshot.aircraft[aid],soc=.8,p=(0.,0.,0.))


def test_custom_numeric_arrays_bypass_cache_and_mapping_snapshot_supported():
    np = pytest.importorskip('numpy')
    snapshot, descriptors = development_fixture(1,1,0)
    aid = descriptors[0]['aircraft_id']
    p = np.array([1.,2.,3.])
    snapshot.aircraft[aid] = replace(snapshot.aircraft[aid],p=p)
    encoder = FeatureEncoder()
    compare(encoder,snapshot,descriptors)
    assert encoder._prepared is None
    p[0] = 99.
    compare(encoder,snapshot,descriptors)
    compare(encoder,vars(snapshot),descriptors)


def test_normalizer_forward_and_rank_match_independent_legacy():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    torch.manual_seed(20261024)
    snapshot, descriptors = development_fixture(5,7,4)
    features = FeatureEncoder().encode(snapshot,descriptors,development_state(snapshot,2))
    # Vary entity counts, including zero, and accept tuple row collections.
    for i,f in enumerate(features):
        f['sets'] = {k:tuple(v[:i]) for k,v in f['sets'].items()}
    normalizer = FeatureNormalizer().fit([dict(split='train',features=features)])
    statistics = deepcopy(normalizer.as_dict())
    before = deepcopy(features)
    normalized = [normalizer.transform(f) for f in features]
    assert normalized == [legacy_transform(normalizer,f) for f in features]
    normalized[0]['query'][0] = 999.
    assert features == before
    model = build_model().eval()
    weights = {k:v.clone() for k,v in model.state_dict().items()}
    advisor = LearnedAdvisor('never-loaded')
    advisor._model, advisor._normalizer, advisor._metadata = model,normalizer,{}
    with torch.inference_mode():
        old = legacy_forward(model,[legacy_transform(normalizer,f) for f in features])
        new = model([normalizer.transform(f) for f in features])
    assert torch.allclose(old,new,atol=1e-7,rtol=0.)
    expected_order,scores = legacy_rank(model,normalizer,features)
    actual = advisor.rank(features)
    assert actual.ordered_ids == expected_order
    assert actual.scores == pytest.approx(scores,abs=1e-7,rel=0.)
    assert statistics == normalizer.as_dict()
    assert all(torch.equal(weights[k],v) for k,v in model.state_dict().items())
    # Exactly tied logits retain inherited order.
    for parameter in model.parameters():
        parameter.data.zero_()
    assert advisor.rank(features).ordered_ids == tuple(f['descriptor_id'] for f in features)
    assert model([]).shape == (0,3)


def test_fixed_attempt_scheduler_uses_identical_legacy_and_new_selection(monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_planning as sp
    from terminal_marl.core.rotating_slots.stage3_learning import AdvisorRuntime, RankResult
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = build_model().eval()
    snapshot_fixture, descriptors = development_fixture(3,3,0)
    normalizer = FeatureNormalizer().fit([dict(split='train',features=LegacyFeatureEncoder().encode(snapshot_fixture,descriptors))])
    class LegacyAdvisor:
        def rank(self,features,budget):
            order,scores = legacy_rank(model,normalizer,features)
            return RankResult(order,scores=scores)
    advisor = LearnedAdvisor('never-loaded')
    advisor._model,advisor._normalizer,advisor._metadata = model,normalizer,{}
    monkeypatch.setattr(sp.PlanningBudget,'elapsed',property(lambda self:0.))
    calls = []
    monkeypatch.setattr(sp,'_generate',lambda snapshot,aid,node,target,start,duration:
        (calls.append((aid,target,start,duration)) or None))
    monkeypatch.setattr(sp.ReferenceProvider,'screen',lambda self,snapshot,node,start,budget:
        {(target,duration):None for target in range(6) for duration in range(20,301,5)})
    monkeypatch.setattr(sp.CandidateProvider,'_column',lambda self,s,a,plans,kind,context,*args,**kwargs:
        sp.PredictionColumn(a.id,kind,tuple(plans),(),(0,0,0,0,0)))
    world = RotatingSlotWorld()
    world.seed_hold('i','inner',0,.9)
    snapshot = world.build_planning_snapshot()
    results = []
    for encoder,ranker in ((LegacyFeatureEncoder(),LegacyAdvisor()),(FeatureEncoder(),advisor)):
        calls.clear()
        runtime = AdvisorRuntime('GUIDED_ORDER',advisor=ranker,feature_encoder=encoder)
        budget = sp.PlanningBudget(max_attempts=10,max_per_aircraft=10)
        catalog = sp.CandidateProvider(advisor_runtime=runtime).get_columns(snapshot,budget)
        ranks = [(r['ready_ids'],r['ordered_ids']) for r in runtime.traces if r['record_type'] == 'Rank']
        assert not any(r['record_type'] == 'Fallback' for r in runtime.traces)
        results.append((list(calls),dict(catalog.attempts),[c.kind for c in catalog.columns['i']],ranks))
    assert results[0] == results[1]
    assert len(results[0][0]) == 10
    assert results[0][2] == ['KEEP','WAIT']
