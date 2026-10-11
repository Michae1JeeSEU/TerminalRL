"""Frozen candidate registration for the same-authority closed-loop validation."""
from __future__ import annotations

from terminal_marl.configs.arrival_restructure import EXTERNAL_AUTHORITY, MAX_TICKS, MAX_WALL_SECONDS

PROFILE = PROTOCOL_VERSION = 'arrival_closed_loop_validation_v1'
CONTROLLERS = ('downstream_soc_external', 'energy_slack', 'rolling_reorder')
SMOKE_SEEDS = {'flow6_mixed': 896191, 'dense24_fixed': 896192, 'dense24_mixed': 896193}
BRIDGE_SEEDS = {'flow6_mixed': 896101, 'dense24_fixed': 896102, 'dense24_mixed': 896103}
REPLICATE_COHORTS = (
    ('flow6_mixed', 896201), ('dense24_fixed', 896211), ('dense24_fixed', 896212),
    ('dense24_mixed', 896221), ('dense24_mixed', 896222),
)
SEARCH_SETTINGS = dict(lookahead_steps=4, max_orders=4, replan_interval=10, planning_wall_seconds=60.)
COMPARISON_BOUNDARY = ('Three traditional controllers share external_dispatch_v1 authority; '
    'bridge records legacy pilot seeds for descriptive comparison only, without automatic equivalence assertions.')


def registration():
    return dict(protocol=PROTOCOL_VERSION, controllers=list(CONTROLLERS), control_authority=EXTERNAL_AUTHORITY,
        smoke_seeds=SMOKE_SEEDS.copy(), smoke_ticks=2, smoke_native_replay_required=True,
        bridge_seeds=BRIDGE_SEEDS.copy(), bridge_controller=CONTROLLERS[0],
        bridge_legacy_controller='rule_nominal_k3', bridge_equivalence_asserted=False,
        replicate_cohorts=[dict(scenario=s, seed=n) for s, n in REPLICATE_COHORTS],
        bridge_arms=3, replicate_arms=15, suite_arms=18, max_ticks=MAX_TICKS,
        wall_seconds=MAX_WALL_SECONDS, native_time_cap='unchanged scenario configuration',
        search_settings=SEARCH_SETTINGS.copy(), q_consumed=False, ppo_allowed=False)


def build_specs(mode, output, source_commit):
    from pathlib import Path
    if mode not in ('smoke', 'bridge', 'replicate', 'suite'):
        raise ValueError(f'unsupported validation mode: {mode}')
    rows = []
    if mode == 'smoke':
        rows.extend(('smoke', s, n, CONTROLLERS[0]) for s, n in SMOKE_SEEDS.items())
    if mode in ('bridge', 'suite'):
        rows.extend(('bridge', s, n, CONTROLLERS[0]) for s, n in BRIDGE_SEEDS.items())
    if mode in ('replicate', 'suite'):
        rows.extend(('replicate', s, n, c) for s, n in REPLICATE_COHORTS for c in CONTROLLERS)
    return [dict(mode=phase, scenario=scenario, seed=seed, controller=controller,
        control_authority=EXTERNAL_AUTHORITY, source_commit=source_commit,
        output=str(Path(output) / f'{phase}_{scenario}_{seed}_{controller}'),
        max_ticks=2 if phase == 'smoke' else MAX_TICKS, wall_seconds=MAX_WALL_SECONDS,
        profile=PROFILE, protocol_version=PROTOCOL_VERSION, registration=registration(),
        comparison_boundary=COMPARISON_BOUNDARY,
        concurrency_budget=dict(smoke_workers=3, suite_workers=18, total_max=18, nested_pools=False),
        protocol_revision_reason='same-authority traditional closed-loop validation; no RL')
        for phase, scenario, seed, controller in rows]
