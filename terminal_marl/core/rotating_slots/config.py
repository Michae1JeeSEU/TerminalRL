"""Frozen FT-RS-S1-1.0 configuration, independent of the legacy model."""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import math
from functools import lru_cache
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / 'new10.16' / 'fixed_terminal_rotating_slots_stage1_config_v1_0.yaml'
# Semantic digest of the supplied normative payload (comments/format excluded).
CANONICAL_CONFIG_HASH = '0c7cce0c0c019d50f7bba5aecd5e8340925d0d16867f8f06eb51e89102343d53'
SERVICE_HALF_CONFIG = DEFAULT_CONFIG.with_name('fixed_terminal_rotating_slots_stage2_service_half_v1.yaml')
SERVICE_HALF_CONFIG_HASH = '5582b671279c0162dd3f6f72da12368a5b2ecffd12d97f5d59eaf945a47ecf87'
SERVICE_PROFILE_PATHS = {'baseline': DEFAULT_CONFIG, 'service_half_v1': SERVICE_HALF_CONFIG}

def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

class Section(Mapping):
    __slots__ = ('_items',)
    def __init__(self, data):
        object.__setattr__(self, '_items', tuple((k, freeze(v)) for k, v in data.items()))
    def __setattr__(self, key, value):
        raise AttributeError('configuration is immutable')
    def __getitem__(self, key):
        for k, v in self._items:
            if k == key: return v
        raise KeyError(key)
    def __iter__(self): return (k for k, _ in self._items)
    def __len__(self): return len(self._items)
    def __getattr__(self, key):
        try: return self[key]
        except KeyError: raise AttributeError(key) from None
    def to_dict(self): return {k: thaw(v) for k, v in self._items}

def freeze(value):
    if isinstance(value, dict): return Section(value)
    if isinstance(value, (list, tuple)): return tuple(freeze(v) for v in value)
    return value

def thaw(value):
    if isinstance(value, Section): return value.to_dict()
    if isinstance(value, tuple): return [thaw(v) for v in value]
    return value

@dataclass(frozen=True)
class Config:
    sections: Section
    raw_hash: str
    config_hash: str
    profile_hash: str
    path: str
    @property
    def service_profile(self):
        study = self.sections.get('capacity_study')
        if (isinstance(study, Section) and study.get('schema_version') == 'capacity_study_v1'
                and study.get('base_config_hash') == SERVICE_HALF_CONFIG_HASH):
            return 'capacity_study_v1'
        return 'service_half_v1' if self.config_hash == SERVICE_HALF_CONFIG_HASH else 'baseline'
    def __getattr__(self, key): return getattr(self.sections, key)
    def __getitem__(self, key): return self.sections[key]
    def to_dict(self): return self.sections.to_dict()

def _parse_config(path):
    path = Path(path or DEFAULT_CONFIG)
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    expected = {
      'schema_version': 'rotating_slots_stage1_v1_0',
      'vehicle.profile_id': 'eh216s_reference_pm16_v1_0',
      'vehicle.fixed_flight_mass_kg':620., 'vehicle.rotor_count':16,
      'vehicle.rotor_diameter_m':1.575, 'vehicle.max_horizontal_speed_mps':25.,
      'vehicle.max_horizontal_acceleration_norm_mps2':1.5,
      'vehicle.max_vertical_acceleration_mps2':1., 'vehicle.max_horizontal_jerk_mps3':1.,
      'vehicle.max_vertical_jerk_mps3':.5, 'vehicle.max_climb_speed_mps':2.,
      'vehicle.max_descent_speed_mps':2.,
      'airspace.boundary_radius_m':900., 'airspace.common_angular_speed_radps':.035,
      'airspace.rotation_direction':'clockwise', 'safety.hard_separation_m':80.,
      'safety.allow_undershoot_m':0., 'safety.unknown_is_rejected':True,
      'energy.profile_id':'eh216s_lit_proxy_v1_0', 'energy.reference_capacity_Ah':252.,
      'energy.equivalent_voltage_V':85., 'energy.air_density_kgpm3':1.225,
      'energy.effective_horizontal_drag_area_m2':2.11,
      'energy.effective_vertical_drag_area_m2':1.47, 'energy.coaxial_chi':1.,
      'energy.profile_solidity':.055, 'energy.profile_mean_drag_coefficient':.0089,
      'energy.profile_Fp':.97, 'energy.profile_tip_speed_mps':120.48,
      'energy.motor_drive_efficiency':.9, 'energy.minimum_axial_descent_ratio':-.28,
      'pads.count':6, 'pads.layout_radius_m':100.,
      'pads.vertical_descent_profile.total_duration_s':13.5,
    }
    for name, value in expected.items():
        found = data
        try:
            for part in name.split('.'): found = found[part]
        except (KeyError, TypeError): raise ValueError('missing canonical field: '+name) from None
        if found != value: raise ValueError('noncanonical FT-RS-S1-1.0 field: '+name)
    rings = [(r['id'], r['radius_m'], r['height_agl_m'], r['slot_count'],r['initial_phase_rad']) for r in data['airspace']['rings']]
    if rings != [('outer',600.,110.,37,0.),('middle',450.,70.,28,0.),('inner',300.,30.,18,0.)]:
        raise ValueError('noncanonical ring geometry')
    def finite(v):
        if isinstance(v,dict): return all(finite(x) for x in v.values())
        if isinstance(v,list): return all(finite(x) for x in v)
        return not isinstance(v,float) or math.isfinite(v)
    if not finite(data): raise ValueError('nonfinite configuration')
    if digest(data) not in (CANONICAL_CONFIG_HASH, SERVICE_HALF_CONFIG_HASH):
        raise ValueError('configuration differs from frozen FT-RS-S1-1.0 payload or authorized service_half_v1 variant')
    return Config(Section(data),sha256(raw).hexdigest(),digest(data),digest({k:data[k] for k in ('vehicle','energy','coordinates')}),str(path.resolve()))

@lru_cache(maxsize=1)
def _default_config(): return _parse_config(DEFAULT_CONFIG)

def load_config(path=None):
    # Explicit files are always read again so editing a supplied path cannot
    # silently return a previously cached canonical configuration.
    return _default_config() if path is None else _parse_config(path)


def load_service_profile(profile='baseline'):
    """Select one complete frozen payload; the canonical default is unchanged."""
    if profile not in SERVICE_PROFILE_PATHS:
        raise ValueError('unknown service profile: '+str(profile))
    return load_config(SERVICE_PROFILE_PATHS[profile])


def service_duration(config, service_class):
    return float(config.pads.service_classes[service_class].duration_s)
