"""Actual P3 mission/priority/SOC task on the repaired executable physics.

The integrated-repair profile remains task neutral for historical evidence.
This explicit profile is shared by fresh data, qualification and P3 learning.
"""
from dataclasses import replace

from .p3_integrated_repair import build_config as foundation_config

PROFILE = 'p3_v25_priority_soc_committed_merge_20261008_v1'
SCENARIOS = ('flow6', 'dense24', 'flow12')


def build_config(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f'unknown P3 scenario: {scenario}')
    requests = {'flow6': 6, 'dense24': 24, 'flow12': 12}[scenario]
    warm = 12 if scenario == 'dense24' else 0
    config = foundation_config(requests=requests, warm_start=warm,
                               high_soc=False, forecast_merge_gap=True)
    schedule = config.fixed_arrival_schedule if warm else tuple(10.+60.*i for i in range(requests))
    return replace(config, task_priority_enabled=True,
        priority_mode='random_135', initial_priority=None,
        initial_soc=None, initial_soc_mode='uniform', initial_soc_min=.55, initial_soc_max=.90,
        mission_probabilities=(.45,.25,.10,.20),
        release_arbitration_mode='downstream_priority',
        low_level_priority_time_weight=1., low_level_soc_urgency_time_weight=2.,
        low_level_soc_urgency_threshold=.65, fixed_arrival_schedule=schedule,
        max_simulation_seconds=1800. if warm else schedule[-1]+900.)
