"""Bounded streaming online telemetry, separate from portable collection traces.

Call summary() for serializable aggregate counters, cost_counts() for evaluation
cost fields, coverage() for the last lightweight snapshot coverage. It accepts
either full collection events or already projected online events identically.
No feature arrays, search states, columns or descriptors are retained.
"""
from collections import Counter
import math


_DROP = {'FeaturesBeforeQuery','WorkDescriptor','ColumnAncestry'}
_FIELDS = {
    'FactualResult': ('descriptor_id','outcome','censored','wall_s','cpu_s','physical_attempts','features_before_query_ref'),
    'SingleVehicleEvidence': ('descriptor_id','outcome','wall_s','cpu_s','validity_label_mask','energy_cache_hit'),
    'ColumnValidationCost': ('descriptor_id','features_before_query_ref','wall_s','cpu_s','cache_hit','accepted','validation_status','cost_scope'),
    'Rank': ('policy',), 'Fallback': ('reason','remaining_budget','error_type'),
    'ReuseEvidence': ('physical_attempts',),
    'SnapshotStart': ('snapshot_id','time','mode','deterministic_only'),
    'Coverage': ('snapshot_id','counts','unserved_aircraft_count','guided_stages','geometry_screening','geometry_timing'),
    'PlanningSelection': ('snapshot_id','certified','selected_descriptor_ids','selected_column_ids','objective','submitted_trajectory_hashes','commit_status'),
}


def lightweight_event(row):
    """Project before copying; unknown verbose events never enter online trace."""
    kind = row.get('record_type')
    if kind in _DROP or kind not in _FIELDS:
        return None
    return dict(record_type=kind, **{k:row[k] for k in _FIELDS[kind] if k in row})


class OnlineObserver:
    def __init__(self):
        self._counts = Counter()
        self._costs = Counter()
        self._coverage = {}
        self._fallbacks = Counter()
        self._selection = None

    def record(self,row):
        row = lightweight_event(row)
        if row is None:
            return
        kind = row['record_type']
        self._counts[kind] += 1
        if kind == 'FactualResult':
            self._costs['descriptor_queries'] += 1
            self._costs['physical_attempts'] += row.get('physical_attempts',0)
            field = 'generation_wall_s'
        elif kind == 'SingleVehicleEvidence':
            field = 'energy_wall_s'
            self._counts['outcome:'+str(row.get('outcome'))] += 1
        elif kind == 'ColumnValidationCost':
            field = 'column_validation_wall_s'
        else:
            field = None
        if field:
            value = row.get('wall_s')
            if isinstance(value,(int,float)) and math.isfinite(value) and value >= 0:
                self._costs[field] += value
                self._counts[field+':observed'] += 1
        if kind == 'Rank':
            self._counts['rank:'+str(row.get('policy'))] += 1
        elif kind == 'Coverage':
            self._coverage = dict(row)
        elif kind == 'Fallback':
            self._fallbacks[str(row.get('reason'))] += 1
        elif kind == 'PlanningSelection':
            self._selection = dict(row)

    __call__ = record

    def cost_counts(self):
        return {k:self._costs[k] for k in ('descriptor_queries','physical_attempts','generation_wall_s','energy_wall_s','column_validation_wall_s')}

    def coverage(self):
        return dict(self._coverage)

    def summary(self):
        return dict(schema='stage3_online_telemetry_v2',counts=dict(self._counts),costs=self.cost_counts(),
            coverage=self.coverage(),fallback_counts=dict(self._fallbacks),last_selection=self._selection)
