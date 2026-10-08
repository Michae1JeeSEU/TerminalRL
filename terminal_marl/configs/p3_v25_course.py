"""Current task-neutral P3 course on the repaired executable physics.

Mission controls physical service duration; it does not grant release priority.
The three conditions isolate SOC and dense heterogeneous service occupancy.
"""
from dataclasses import replace

from .p3_integrated_repair import build_config as foundation_config

PROFILE = 'p3_v25_neutral_course_nominal_curve_total_wait_20261008_v2'
SCENARIOS = ('flow6_mixed', 'dense24_fixed', 'dense24_mixed')


def build_config(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f'unknown P3 course scenario: {scenario}')
    dense = scenario != 'flow6_mixed'
    config = foundation_config(requests=24 if dense else 6,
        warm_start=12 if dense else 0, high_soc=False, forecast_merge_gap=True)
    schedule = config.fixed_arrival_schedule if dense else tuple(10.+60.*i for i in range(6))
    return replace(config, task_priority_enabled=False,
        priority_wait_mode='total_wait_v1', nominal_holder_separation_mode='swept_curve_v1',
        low_level_priority_time_weight=0., low_level_soc_urgency_time_weight=2.,
        initial_soc_mode='fixed' if scenario=='dense24_fixed' else 'uniform',
        initial_soc=.8 if scenario=='dense24_fixed' else None,
        initial_soc_min=.55, initial_soc_max=.90,
        fixed_arrival_schedule=schedule,
        max_simulation_seconds=1800. if dense else schedule[-1]+900.)
