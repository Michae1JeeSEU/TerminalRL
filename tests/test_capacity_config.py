"""Configuration variants retain the full simulator and frozen loader boundary."""
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
import math
from types import SimpleNamespace

import pytest
import yaml

from terminal_marl.core.rotating_slots.capacity_config import make_capacity_config
from terminal_marl.core.rotating_slots.config import (
    CANONICAL_CONFIG_HASH, SERVICE_HALF_CONFIG_HASH, digest, load_config, load_service_profile,
)
from terminal_marl.core.rotating_slots.geometry import pad_position, ring_config, slot_state
from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, PlanningBudget, PlanningNode, ReferenceProvider,
)
from terminal_marl.core.rotating_slots.trajectory import generate_final, screen_normal_templates
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


@pytest.mark.parametrize('radius,slots', ((250., 15), (300., 18), (350., 21)))
@pytest.mark.parametrize('pads', (3, 6))
def test_complete_immutable_config_and_static_layout(radius, slots, pads):
    config = make_capacity_config(inner_radius_m=radius, pad_count=pads,
                                  service_seconds=(20., 40., 80.), service_probabilities=(.1, .7, .2))
    base = load_service_profile('service_half_v1')
    expected = base.to_dict()
    expected['airspace']['rings'][2].update(radius_m=radius, slot_count=slots)
    expected['pads']['count'] = pads
    for name, duration, probability in zip(('short', 'standard', 'long'), (20., 40., 80.), (.1, .7, .2)):
        expected['pads']['service_classes'][name].update(duration_s=duration, probability=probability)
    data = config.to_dict()
    metadata = data.pop('capacity_study')
    assert data == expected
    assert metadata['base_config_hash'] == SERVICE_HALF_CONFIG_HASH
    assert config.service_profile == 'capacity_study_v1'
    assert config.profile_hash == base.profile_hash
    assert config.config_hash == digest(config.to_dict()) != base.config_hash
    raw = json.dumps(config.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert config.raw_hash == sha256(raw).hexdigest()
    assert config.config_hash == make_capacity_config(inner_radius_m=radius, pad_count=pads,
        service_seconds=(20., 40., 80.), service_probabilities=(.1, .7, .2)).config_hash
    with pytest.raises(AttributeError): config.airspace.rings[2].radius_m = 1
    with pytest.raises(FrozenInstanceError): config.config_hash = 'changed'
    config.to_dict()['pads']['count'] = 99
    assert config.pads.count == pads
    rings = config.airspace.rings
    for ring in rings:
        assert 2 * ring.radius_m * math.sin(math.pi / ring.slot_count) >= 100.
        state = slot_state(ring.id, 0, 0., config)
        assert math.dist(state.v, (0., 0., 0.)) <= config.vehicle.max_horizontal_speed_mps
        assert math.dist(state.a, (0., 0., 0.)) <= config.vehicle.max_horizontal_acceleration_norm_mps2
    for first, second in zip(rings, rings[1:]):
        assert math.hypot(first.radius_m-second.radius_m, first.height_agl_m-second.height_agl_m) >= 80.
    assert ring_config('inner', config).slot_count == slots
    positions = [pad_position(pad, config) for pad in range(pads)]
    for pad, position in enumerate(positions):
        angle = 2 * math.pi * pad / pads
        assert position == pytest.approx((100 * math.cos(angle), 100 * math.sin(angle), 0.))
        assert math.dist(position, positions[(pad+1) % pads]) >= 80.
    with pytest.raises(ValueError): pad_position(pads, config)


@pytest.mark.parametrize('kwargs', (
    {'inner_radius_m': 299.}, {'inner_radius_m': float('nan')}, {'pad_count': 4}, {'pad_count': 3.},
    {'service_seconds': (0., 60., 120.)}, {'service_seconds': (30., float('inf'), 120.)},
    {'service_seconds': (30., 60.)}, {'service_probabilities': (-.1, .9, .2)},
    {'service_probabilities': (.2, .6, .3)}, {'service_probabilities': (0., 0., 0.)},
    {'service_probabilities': (.2, float('nan'), .8)},
    {'service_probabilities': (1e308, 1e308, 1e308)},
))
def test_rejects_unapproved_or_invalid_parameters(kwargs):
    with pytest.raises(ValueError): make_capacity_config(**kwargs)


def test_frozen_loaders_still_reject_study_and_yaml_drift(tmp_path):
    assert load_config().config_hash == CANONICAL_CONFIG_HASH
    assert load_service_profile('service_half_v1').config_hash == SERVICE_HALF_CONFIG_HASH
    for name, data in (('study', make_capacity_config().to_dict()),
                       ('drift', load_service_profile('service_half_v1').to_dict())):
        if name == 'drift': data['pads']['service_classes']['short']['duration_s'] = 31.
        path = tmp_path / (name+'.yaml')
        path.write_text(yaml.safe_dump(data), encoding='utf-8')
        with pytest.raises(ValueError): load_config(path)
    with pytest.raises(ValueError): load_service_profile('capacity_study_v1')


def test_three_pad_resources_and_search_targets(monkeypatch):
    config = make_capacity_config(pad_count=3)
    world = RotatingSlotWorld(config=config)
    assert set(world.pad_busy_seconds) == {0, 1, 2}
    for pad in range(3): world.seed_service(str(pad), pad, remaining=2.)
    assert {r.resource for r in world.pads.reservations} == {0, 1, 2}
    with pytest.raises(ValueError): world.seed_service('invalid', 3)
    world.advance(.2, False)
    assert world.pad_busy_seconds == {0: .2, 1: .2, 2: .2}
    calls = []
    monkeypatch.setattr(world, '_reference_pad', lambda ring, slot, pad, *args: (calls.append(pad) or .1, None))
    assert world._reference('inner', 0, 0.) == (.1, None)
    assert calls == [0, 1, 2]
    calls.clear()
    def missing(*args, **kwargs):
        calls.append(args[3])
        return SimpleNamespace(plan=None, status='NOT_FOUND', reason='bounded test')
    monkeypatch.setattr('terminal_marl.core.rotating_slots.world.generate_final', missing)
    aircraft = SimpleNamespace(id='a', ring='inner', slot=0, version=0)
    assert not world._try_final(aircraft, False)
    assert calls == [0, 1, 2]


def test_planning_reference_and_candidate_cursors_use_three_pads(monkeypatch):
    config = make_capacity_config(pad_count=3)
    world = RotatingSlotWorld(config=config, experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_hold('a', 'inner', 0, .9)
    snapshot = world.build_planning_snapshot()
    aircraft = snapshot.aircraft['a']
    node = PlanningNode('inner', 0, 0., .9, 0., 0)
    screen, _ = screen_normal_templates('inner', 0, 0., config)
    assert {target for target, duration in screen} == {0, 1, 2}
    references = ReferenceProvider()
    monkeypatch.setattr(references, 'screen', lambda *args: screen)
    provider = CandidateProvider(reference_provider=references)
    cursors = list(provider._offset_cursors(snapshot, aircraft, node, PlanningBudget(), 0.))
    assert {cursor['target'] for cursor in cursors} == {0, 1, 2}
    # Force a complete finite search without expensive plan construction.
    calls = []
    monkeypatch.setattr(references, 'generate', lambda snapshot, a, node, target, *args: (calls.append(target) or True, None))
    references.get(snapshot, aircraft, node, PlanningBudget())
    assert set(calls) == {0, 1, 2}


def test_three_pad_monitor_interpolation_and_rescue_search(monkeypatch):
    config = make_capacity_config(pad_count=3)
    world = RotatingSlotWorld(config=config, experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_hold('a', 'inner', 0, .9)
    width = 2*math.pi/240
    expected_cells = []
    for pad in range(3):
        phase = (2*math.pi*pad/3 + .035*2.3) % (2*math.pi)
        cell = int(phase/width)
        expected_cells.append(cell)
        world._practical_runtime['table'][f'inner:{cell}'] = [[.01, 90.]]*3
    monkeypatch.setattr(world, '_reference', lambda *args: pytest.fail('covered monitor should interpolate'))
    monkeypatch.setattr(world, '_reference_pad', lambda *args: pytest.fail('unexpected phase cell'))
    assert world._monitor_reference('inner', 0, 2.3, 0, .35, 0.) == .01
    assert len(world._practical_runtime['table']) == len(set(expected_cells)) == 3
    assert world.practical_runtime_stats()['counters']['table_hits'] == 1
    calls = []
    def missing(*args, **kwargs):
        calls.append(args[3])
        return SimpleNamespace(plan=None, status='NOT_FOUND')
    monkeypatch.setattr('terminal_marl.core.rotating_slots.world.generate_final', missing)
    assert world._search_rescue_opportunity(world.aircraft['a']) is None
    assert set(calls) == {0, 1, 2}
    assert calls == [0, 1, 2] * (len(calls)//3)


def test_three_pad_final_and_reference_relative_phase(monkeypatch):
    config = make_capacity_config(pad_count=3)
    world = RotatingSlotWorld(config=config, experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency = False
    world.seed_hold('a', 'inner', 0, .9)
    result = generate_final('a', 'inner', 0, 1, 2.3, config=config)
    assert result.accepted
    plan = result.plan
    assert plan.state(plan.end_time).p == pytest.approx(pad_position(1, config), abs=1e-9)
    assert world.submit((plan,)).accepted
    assert {r.resource for r in world.pads.reservations} == {1}
    bins = []
    monkeypatch.setattr(world, '_canonical_reference_duration', lambda ring, phase_bin: bins.append(phase_bin) or plan.segments[0].duration)
    q, reference = world._reference_pad('inner', 0, 1, 2.3)
    assert q is not None and reference.pad_id == 1
    source = slot_state('inner', 0, 2.3, config).p
    phase = (math.atan2(*pad_position(1, config)[1::-1])-math.atan2(*source[1::-1])) % (2*math.pi)
    assert bins == [int(phase/(2*math.pi)*world.REFERENCE_PHASE_BINS)]


def test_persistent_reference_identity_rejects_other_geometry(tmp_path):
    path = tmp_path / 'reference.json'
    first = RotatingSlotWorld(config=make_capacity_config(pad_count=6), reference_cache_path=path)
    first._practical_runtime['table']['inner:0'] = [[.1, 90.]] * 3
    first.flush_reference_table()
    second = RotatingSlotWorld(config=make_capacity_config(pad_count=3), reference_cache_path=path)
    assert not second._practical_runtime['table']
    assert second.practical_runtime_stats()['counters']['table_identity_rejected'] == 1
