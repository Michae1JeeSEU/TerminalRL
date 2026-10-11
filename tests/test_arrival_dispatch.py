"""Detached dispatcher estimates, order-only planning, and native integration."""
from copy import deepcopy
import json
import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.core import arrival_dispatch as dispatch
from terminal_marl.core.config import SimConfig
from terminal_marl.core.landing_zone import LZStatus
from terminal_marl.core.merge_scheduler import MergeReservation
from terminal_marl.core.trajectory_time_parameterization import TrajectoryInfeasibleError
from terminal_marl.core.uav import ArrivalRequest, QueuedArrivalState, MissionType, FlightPhase, TerminalOutcome
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def make_env():
    return TerminalParallelEnv(SimConfig(
        reset_mode='warm_start_rings', warm_start_uavs=3, warm_start_steady_motion=True,
        fixed_arrival_schedule=(0., 0., 0., 1000.), total_requested_uavs=4,
        initial_soc_mode='fixed', initial_soc=.8,
        release_arbitration_mode='external_dispatch_v1', runtime_proposal_shortlist_k=3,
        action_schema_version='low_action_schema_v4_2_hierarchical'), seed=77)


def test_downstream_external_matches_native_tuple_on_same_state_without_mutation(monkeypatch):
    from terminal_marl.core.config import RingRole
    env = make_env()
    env.config.task_priority_enabled = True
    template = deepcopy(env.active[0])
    # Each adjacent pair exercises a distinct component of the tuple key.
    cases = [
        (RingRole.INNER, .8, 1, 0.),
        (RingRole.MIDDLE, .6, 1, 0.),
        (RingRole.OUTER, .6, 1, 0.),
        (None, .6, 1, 0.),
        (RingRole.INNER, .7, 1, 0.),
        (RingRole.INNER, .7, 3, 0.),
        (RingRole.INNER, .7, 3, 20.),
        (RingRole.INNER, .7, 3, 20.),
    ]
    env.config.priority_alpha = 0.  # Isolate wait tie-breaking from priority.
    env.active = {}
    for uid, (role, soc, base, wait) in enumerate(cases):
        uav = deepcopy(template)
        uav.uid = uid; uav.ring_role = role
        uav.energy_remaining = soc * uav.energy_capacity
        uav.base_priority = base; uav.internal_wait_time = wait
        uav.dynamic_priority = -999.
        env.active[uid] = uav
    env.active[6].committed_plan = SimpleNamespace(plan_id='committed')
    before = pickle.dumps(env)
    def forbidden(*args, **kwargs):
        pytest.fail('pure rule must not query energy budgets or clone rollouts')
    monkeypatch.setattr(dispatch, '_energy_rows', forbidden)
    monkeypatch.setattr(dispatch, '_clone_for_planning', forbidden)
    decision = dispatch.downstream_soc_external_decision(env)
    assert decision.order == (6, 7, 5, 4, 0, 1, 2, 3)
    assert set(decision.order) == set(env.active)
    assert pickle.dumps(env) == before
    native = deepcopy(env)
    native.config.release_arbitration_mode = 'downstream_soc_priority'
    for uav in native.active.values():
        native.priority.update(uav, native.now)
    assert decision.order == tuple(uid for uid, _ in native._release_processing_order())
    # Same-state tuple equality does not equate pre/post-admission tick behavior.
    assert decision.diagnostics['ordering_state'] == 'pre_step_active'
    assert decision.diagnostics['legacy_stepwise_equivalence'] is False
    actions, shortlists = dispatch.nominal_requests(env)
    assert {k:v.tolist() for k,v in decision.actions.items()} == {k:v.tolist() for k,v in actions.items()}
    assert {k:[v.tolist() for v in vs] for k,vs in decision.shortlists.items()} == {k:[v.tolist() for v in vs] for k,vs in shortlists.items()}
    assert decision.actions['6'][0] == decision.actions['3'][0] == 0.
    json.dumps(decision.diagnostics, allow_nan=False)
    env.active[7].energy_remaining = .1 * template.energy_capacity
    assert dispatch.downstream_soc_external_decision(env).order[0] == 7


def test_normal_remaining_routes_change_equal_soc_urgency_without_live_mutation():
    env = make_env()
    before = pickle.dumps(env)
    budgets = [dispatch.normal_route_budget(env, u) for u in env.active.values()]
    assert budgets[0] > budgets[1] > budgets[2] > 0
    assert dispatch.energy_slack_order(env) == (0, 1, 2)
    assert pickle.dumps(env) == before


def test_nominal_requests_only_release_eligible_and_keep_k3():
    env = make_env()
    env.active[1].safety_retry_after = 10
    env.active[2].committed_plan = SimpleNamespace(plan_id='committed')
    actions, shortlists = dispatch.nominal_requests(env)
    assert tuple(actions) == ('0', '1', '2')
    assert actions['0'].tolist() == [1., 0., 0.]
    assert actions['1'].tolist() == actions['2'].tolist() == [0., 0., 0.]
    assert [a.tolist() for a in shortlists['0']] == [list(a) for a in dispatch.NOMINAL_K3]
    assert set(shortlists) == {'0'}
    actions, shortlists = dispatch.nominal_requests(env, [])
    assert all(a[0] == 0 for a in actions.values()) and shortlists == {}
    with pytest.raises(ValueError, match='unknown active'):
        dispatch.nominal_requests(env, [99])


def test_nonholders_follow_holders_and_wait_breaks_ties(monkeypatch):
    env = make_env()
    monkeypatch.setattr(dispatch, '_normal_route_budget_details', lambda e, u: {'required_energy': 10.})
    env.active[1].internal_wait_time = 20
    env.active[0].phase = FlightPhase.INBOUND
    env.active[0].ring_role = None
    assert dispatch.energy_slack_order(env) == (1, 2, 0)


def test_no_route_is_explicit_json_null_and_known_infeasible_only(monkeypatch):
    env = make_env()
    for z in env.lz_manager.zones:
        z.compatible_missions = frozenset()
    assert math.isinf(dispatch.normal_route_budget(env, env.active[0]))
    d = dispatch.ArrivalDispatcher().decide(env)
    assert all(row['required_energy'] is None and row['status'] == 'no_route'
               for row in d.diagnostics['energy_rows'].values())
    json.dumps(d.diagnostics, allow_nan=False)
    def infeasible(*a, **kw):
        raise TrajectoryInfeasibleError('test unavailable')
    monkeypatch.setattr(dispatch.TrajectoryGenerator, 'ring_transfer', infeasible)
    assert math.isinf(dispatch.normal_route_budget(env, env.active[0]))
    def bug(*a, **kw):
        raise RuntimeError('unexpected bug')
    monkeypatch.setattr(dispatch.TrajectoryGenerator, 'ring_transfer', bug)
    with pytest.raises(RuntimeError, match='unexpected bug'):
        dispatch.normal_route_budget(env, env.active[0])


def test_pad_eta_cost_and_cache_invalidation(monkeypatch):
    env = make_env()
    inner = env.active[2]
    base = dispatch.normal_route_budget(env, inner)
    for zone in env.lz_manager.zones:
        zone.status = LZStatus.RESERVED
        zone.reserved_by = 999
        zone.expected_release_at = 100
    assert dispatch.normal_route_budget(env, inner) > base
    controller = dispatch.ArrivalDispatcher()
    assert controller.decide(env).diagnostics['cache_refreshed']
    env.step_count = 1
    d = controller.decide(env)
    assert not d.diagnostics['cache_refreshed']
    assert d.diagnostics['cache_age_ticks'] == 1
    assert d.diagnostics['native_budget_queries'] == 0
    env.lz_manager.zones[0].expected_release_at = 110
    assert controller.decide(env).diagnostics['replan_reason'] == 'resource_change'
    env.scheduler.reservations['test'] = MergeReservation('test', 0, 'outer', 'middle', 50, None, None, None)
    assert controller.decide(env).diagnostics['cache_refreshed']
    env.active[0].committed_plan = SimpleNamespace(plan_id='new')
    assert controller.decide(env).diagnostics['cache_refreshed']
    env.step_count += 10
    assert controller.decide(env).diagnostics['replan_reason'] == 'interval'


def test_clone_freezes_only_hidden_arrivals_retains_observed_queue_and_commits():
    env = make_env()
    request = ArrivalRequest(98, 0., 0, MissionType.STANDARD, 1, 1500., .8)
    queued = QueuedArrivalState(request, 1200.)
    env.admission.queue.items.append(queued)
    env.ledger.request(98)
    env.active[0].committed_plan = SimpleNamespace(plan_id='committed')
    before = pickle.dumps(env)
    clone = dispatch._clone_for_planning(env)
    assert clone.arrivals._schedule == () and clone.arrivals.next_time == math.inf
    assert clone.admission.queue.items[0].request == request
    assert clone.admission.queue.items[0] is not queued
    assert clone.active[0].committed_plan.plan_id == 'committed'
    assert pickle.dumps(env) == before


def test_native_rolling_unchanged_live_state_hidden_future_and_rng():
    env = make_env()
    before = pickle.dumps(env)
    controller = dispatch.ArrivalDispatcher('rolling_reorder', lookahead_steps=2, max_orders=3)
    decision = controller.decide(env)
    assert pickle.dumps(env) == before
    altered = deepcopy(env)
    altered.arrivals._schedule = (0., 1., 2., 3.)
    altered.arrivals.next_time = 0.
    altered.arrivals.rng = np.random.default_rng(999)
    other = dispatch.ArrivalDispatcher('rolling_reorder', lookahead_steps=2, max_orders=3).decide(altered)
    assert decision.order == other.order
    assert [r.get('score') for r in decision.diagnostics['candidate_results']] == [r.get('score') for r in other.diagnostics['candidate_results']]
    assert len(decision.diagnostics['candidate_results']) <= 3
    assert decision.diagnostics['native_rollout_steps'] <= 6
    json.dumps(decision.diagnostics, allow_nan=False)
    result = env.step(decision.actions, proposal_shortlists=decision.shortlists, release_order=decision.order)
    assert len(result) == 5 and env.step_count == 1


def test_timeout_invalid_fallback_and_deterministic_tie(monkeypatch):
    env = make_env()
    def timeout(e, order, horizon, deadline):
        return {'status': 'timeout', 'order': order}
    monkeypatch.setattr(dispatch, '_evaluate_order', timeout)
    result = dispatch.ArrivalDispatcher('rolling_reorder').decide(env)
    assert result.diagnostics['fallback']
    assert result.order == dispatch.energy_slack_order(env)
    def equal(e, order, horizon, deadline):
        return {'status': 'evaluated', 'order': order, 'score': (1, 0, 0, 0, 0)}
    monkeypatch.setattr(dispatch, '_evaluate_order', equal)
    result = dispatch.ArrivalDispatcher('rolling_reorder').decide(env)
    assert result.order == dispatch.energy_slack_order(env)
    assert len(dispatch._candidate_orders(result.order, env, 1)) == 1
    def bug(*a, **kw):
        raise RuntimeError('native bug')
    monkeypatch.setattr(dispatch, '_evaluate_order', bug)
    with pytest.raises(RuntimeError, match='native bug'):
        dispatch.ArrivalDispatcher('rolling_reorder').decide(env)


def test_native_bad_outcomes_counted_individually_and_rejected(monkeypatch):
    env = make_env()
    def bad_step(clone, *a, **kw):
        for uid in (0, 1):
            uav = clone.active.pop(uid)
            clone.ledger.resolve(uav.arrival_id, TerminalOutcome.FAILED_ENERGY)
        return {}, {}, {}, {}, {}
    monkeypatch.setattr(TerminalParallelEnv, 'step', bad_step)
    result = dispatch.ArrivalDispatcher('rolling_reorder', max_orders=1, lookahead_steps=1).decide(env)
    candidate = result.diagnostics['candidate_results'][0]
    assert candidate['status'] == 'invalid_native_outcome'
    assert candidate['failed_energy'] == 2
    assert result.diagnostics['fallback']

def test_native_collision_integrity_and_programming_errors_are_visible(monkeypatch):
    env = make_env()
    def collision_step(clone, *a, **kw):
        clone.separation_monitor.telemetry['collisions'] += 1
        clone.ledger.request(999)  # Orphan unresolved request is an integrity issue.
        return {}, {}, {}, {}, {}
    monkeypatch.setattr(TerminalParallelEnv, 'step', collision_step)
    result = dispatch.ArrivalDispatcher('rolling_reorder', max_orders=1, lookahead_steps=1).decide(env)
    row = result.diagnostics['candidate_results'][0]
    assert row['separation_deltas']['collisions'] == 1
    assert row['integrity_errors'] == ['unresolved_ledger_mismatch']
    assert result.diagnostics['fallback']
    def bug(*a, **kw):
        raise RuntimeError('native code error')
    monkeypatch.setattr(TerminalParallelEnv, 'step', bug)
    with pytest.raises(RuntimeError, match='native code error'):
        dispatch.ArrivalDispatcher('rolling_reorder', max_orders=1).decide(env)


def test_wall_deadline_records_all_unrun_candidates_and_falls_back():
    result = dispatch.ArrivalDispatcher('rolling_reorder', planning_wall_seconds=1e-9).decide(make_env())
    assert result.diagnostics['fallback']
    assert all(r['status'] == 'timeout' for r in result.diagnostics['candidate_results'])
    assert result.diagnostics['native_rollout_steps'] == 0

def test_nominal_contract_matches_existing_collector_even_for_native_rejections():
    from terminal_marl.scripts.collect_p3_v3_sparse_expert import rule_tape
    env = make_env()
    env.active[0].energy_remaining = 0.
    env.future_release_reservations[0] = {'pending': True}
    env.active[1].safety_retry_after = 10.
    env.active[2].phase = FlightPhase.INBOUND  # Contract uses ring_role, not phase.
    expected = rule_tape(env)
    actions, shortlists = dispatch.nominal_requests(env)
    assert {k: v.tolist() for k, v in actions.items()} == expected['actions']
    assert {k: [v.tolist() for v in rows] for k, rows in shortlists.items()} == expected['shortlists']


def test_checked_executed_hash_mismatch_invalidates_candidate(monkeypatch):
    def mismatch(clone, *a, **kw):
        clone.telemetry['checked_executed_plan_hash_mismatches'] += 1
        return {}, {}, {}, {}, {}
    monkeypatch.setattr(TerminalParallelEnv, 'step', mismatch)
    result = dispatch.ArrivalDispatcher('rolling_reorder', max_orders=1, lookahead_steps=1).decide(make_env())
    row = result.diagnostics['candidate_results'][0]
    assert row['checked_executed_plan_hash_mismatches_delta'] == 1
    assert row['status'] == 'invalid_native_outcome' and result.diagnostics['fallback']

def test_pad_delay_advances_detached_holder_before_final(monkeypatch):
    env = make_env()
    for zone in env.lz_manager.zones:
        zone.status = LZStatus.RESERVED
        zone.reserved_by = 999
        zone.expected_release_at = 100.
    before = pickle.dumps(env)
    observed = []
    original = dispatch.TrajectoryGenerator.final_approach
    def record(self, holder, zone, reservation, now, **kw):
        observed.append((now, holder.position.copy()))
        return original(self, holder, zone, reservation, now, **kw)
    monkeypatch.setattr(dispatch.TrajectoryGenerator, 'final_approach', record)
    assert math.isfinite(dispatch.normal_route_budget(env, env.active[2]))
    assert len(observed) == 6
    assert all(now == 100. for now, position in observed)
    assert all(not np.array_equal(position, env.active[2].position) for now, position in observed)
    assert pickle.dumps(env) == before

def test_current_inner_state_not_rotated_again_since_ring_entry(monkeypatch):
    env = make_env()
    env.now = 120.
    inner = env.active[2]
    assert inner.phase_entry_time < env.now
    before = pickle.dumps(env)
    observed = []
    original = dispatch.TrajectoryGenerator.final_approach
    def record(self, holder, zone, reservation, now, **kw):
        observed.append((now, holder.position.copy()))
        return original(self, holder, zone, reservation, now, **kw)
    monkeypatch.setattr(dispatch.TrajectoryGenerator, 'final_approach', record)
    assert math.isfinite(dispatch.normal_route_budget(env, inner))
    assert observed and all(now == env.now for now, position in observed)
    assert all(np.array_equal(position, inner.position) for now, position in observed)
    assert pickle.dumps(env) == before
