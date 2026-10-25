"""Explicit capacity-study variants of the complete frozen service-half payload.

The static checks here validate the permitted layouts and steady ring motion.
They do not certify any transfer, final approach, or multi-aircraft schedule.
"""
from hashlib import sha256
from itertools import combinations
import json
import math

from .config import Config, Section, digest, load_service_profile


def _triple(values, name, *, positive):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must contain three finite numbers') from None
    if (len(result) != 3 or any(not math.isfinite(value) for value in result)
            or any(value <= 0 if positive else value < 0 for value in result)):
        raise ValueError(f'{name} must contain three finite ' + ('positive' if positive else 'nonnegative') + ' numbers')
    return result


def _validate_static_layout(data):
    hard = data['safety']['hard_separation_m']
    design = data['airspace']['slot_design_min_chord_m']
    omega = data['airspace']['common_angular_speed_radps']
    vehicle = data['vehicle']
    rings = data['airspace']['rings']
    for ring in rings:
        radius = ring['radius_m']
        if 2 * radius * math.sin(math.pi / ring['slot_count']) < design:
            raise ValueError('ring slot chord is below design spacing')
        if radius * omega > vehicle['max_horizontal_speed_mps']:
            raise ValueError('steady ring speed exceeds vehicle constraint')
        if radius * omega**2 > min(vehicle['max_horizontal_acceleration_norm_mps2'],
                                  vehicle['max_lateral_acceleration_mps2']):
            raise ValueError('steady ring acceleration exceeds vehicle constraint')
    for first, second in combinations(rings, 2):
        distance = math.hypot(first['radius_m'] - second['radius_m'],
                              first['height_agl_m'] - second['height_agl_m'])
        if distance < hard:
            raise ValueError('interring minimum is below hard separation')
    pads = data['pads']
    if 2 * pads['layout_radius_m'] * math.sin(math.pi / pads['count']) < hard:
        raise ValueError('pad chord is below hard separation')


def make_capacity_config(*, inner_radius_m=300., pad_count=6,
                         service_seconds=(30., 60., 120.),
                         service_probabilities=(.2, .6, .2)) -> Config:
    """Build an immutable, identified study config without weakening load_config.

    Probabilities must already sum to one; rounding is normalized deterministically.
    The raw hash identifies canonical JSON bytes, which a runner may save separately.
    """
    if isinstance(inner_radius_m, bool) or inner_radius_m not in (250., 300., 350.):
        raise ValueError('inner_radius_m must be one of 250, 300, 350')
    if type(pad_count) is not int or pad_count not in (3, 6):
        raise ValueError('pad_count must be 3 or 6')
    durations = _triple(service_seconds, 'service_seconds', positive=True)
    probabilities = _triple(service_probabilities, 'service_probabilities', positive=False)
    try:
        total = math.fsum(probabilities)
    except OverflowError:
        raise ValueError('service_probabilities must sum to one') from None
    if not math.isclose(total, 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError('service_probabilities must sum to one')
    probabilities = tuple(value / total for value in probabilities)
    base = load_service_profile('service_half_v1')
    data = base.to_dict()
    radius = float(inner_radius_m)
    count = math.floor(math.pi / math.asin(data['airspace']['slot_design_min_chord_m'] / (2 * radius)))
    inner = next(ring for ring in data['airspace']['rings'] if ring['id'] == 'inner')
    inner.update(radius_m=radius, slot_count=count)
    data['pads']['count'] = pad_count
    for name, duration, probability in zip(('short', 'standard', 'long'), durations, probabilities):
        data['pads']['service_classes'][name].update(duration_s=duration, probability=probability)
    data['capacity_study'] = dict(schema_version='capacity_study_v1',
        base_service_profile='service_half_v1', base_config_hash=base.config_hash,
        inner_radius_m=radius, inner_slot_count=count, pad_count=pad_count,
        service_seconds=list(durations), service_probabilities=list(probabilities))
    _validate_static_layout(data)
    raw = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    semantic_hash = digest(data)
    return Config(Section(data), sha256(raw).hexdigest(), semantic_hash, base.profile_hash,
                  f'capacity_study_v1/{semantic_hash}.json')
