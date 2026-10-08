"""Read-only saved-label audit and six predeclared candidate witnesses.

No reset, step, learner, relabelling or certificate bypass. The trace observes
the real SafetyFilter call; dense distances are a grid-sensitivity cross-check
of its prediction model, not an independent proof of all future traffic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import time

import numpy as np

from terminal_marl.core.config import RingRole
from terminal_marl.core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from terminal_marl.core.merge_scheduler import LowLevelIntent, RepairCandidate
from terminal_marl.core.observations import LOW_OBS_V2_4, _slot_vector, build_low_observation_v2_4
from terminal_marl.core.safety_trace import _digest, _json
from terminal_marl.core.topology_neighbors import TopologyNeighborSelector
from terminal_marl.envs.parallel_env import WorldSnapshot
from terminal_marl.scripts.diagnose_p3_direct_contract import canonical, state_record
from terminal_marl.training.p3_v3_expert_dataset import load_sparse_training_episode
from terminal_marl.training.v4_1_joint_teacher import STATE_ONLY_JOINT_CATALOGUE_V4_1

CASES = ((893001, 'p3_v3_i0_20261004/train_01', 59),
         (893101, 'p3_v3_i0_v2_20261004/dense_high_launch2', 613),
         (893201, 'p3_v3_i0_v3_20261007/dense_mixed', 374))
# First is the output-path smoke and is never re-executed by the remaining run.
CANDIDATES = ((10, 12), (13, 12), (4, 12), (4, 14), (19, 12), (19, 13))
SLOT_NAMES = ('same_ring_leader', 'same_ring_follower', 'target_ring_predecessor',
              'target_ring_successor', 'nearest_predicted_conflict')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, result):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(_json(result), stream, ensure_ascii=False, indent=2, allow_nan=False)


def audit_prefix(label):
    rows = label['teacher_audit']
    assert 1 <= len(rows) <= 26
    assert [r['joint_index'] for r in rows] == [c.index for c in STATE_ONLY_JOINT_CATALOGUE_V4_1[:len(rows)]]
    for i, row in enumerate(rows):
        assert row['rank'] == i and row['decoded'] == STATE_ONLY_JOINT_CATALOGUE_V4_1[i].as_dict()
        assert row['feasible'] is (i == len(rows)-1)
    assert rows[-1]['joint_index'] == label['teacher_joint_index']
    assert label['release_intents_checked'] == sum(r['joint_index'] > 0 for r in rows)
    assert label['candidate_plan_calls'] >= sum('final_lz_pairs_checked' not in r and r['reason'] == 'certified' for r in rows)
    cert = label['selected_certificate']
    if label['teacher_gate']:
        assert rows[-1]['reason'] == 'certified'
        assert cert and cert['trajectory_hash'] == cert['checked_hash']
    else:
        assert len(rows) == 26 and rows[-1]['reason'] == 'wait_legal' and cert is None
    return dict(gate=label['teacher_gate'], selected_rank=len(rows)-1,
                rejected=Counter(r['reason'] for r in rows[:-1]))


def audit_labels(root):
    totals, ranks, rejected, reports = Counter(), Counter(), Counter(), []
    for seed, relative, expected in CASES:
        episode = root/relative/f'seed_{seed}'
        batch = load_sparse_training_episode(episode, allowed_train_seeds=(seed,))
        assert len(batch.decision_ids) == expected
        rows = {r['decision_id']: r for r in map(json.loads, (episode/'rows.jsonl').read_text().splitlines()) if r['actor_mask']}
        labels = [read(episode/e['label_file']) for e in read(episode/'annotation_result.json')['entries']]
        assert set(rows) == set(batch.decision_ids) == {l['decision_id'] for l in labels}
        count = Counter()
        for label in labels:
            audit = audit_prefix(label)
            count['RELEASE' if audit['gate'] else 'WAIT'] += 1
            ranks[str(audit['selected_rank'])] += 1
            rejected.update(audit['rejected'])
            row = rows[label['decision_id']]
            count['behavior_gate_differs'] += int((row['behavior_request'][0] > 0) != bool(audit['gate']))
        totals.update(count)
        reports.append(dict(seed=seed, counts=count, loader=batch.summary,
                            annotation_sha256=sha(episode/'annotation_result.json')))
    return dict(status='LABEL_AUDIT_COMPLETE', counts=totals, selected_rank=ranks,
                rejected_reasons=rejected, episodes=reports, new_teacher_queries=0,
                environment_resets=0, environment_steps=0, parameter_updates=0)


def grid(start, end, dt):
    values = np.arange(start, end+1e-9, dt)
    if values[-1] < end:
        values = np.append(values, end)
    return values


def minimum(values, times):
    values = np.asarray(values)
    i = int(np.argmin(values))
    return dict(distance=float(values[i]), index=i, time=float(times[min(i, len(times)-1)]))


def state_delta(before, after):
    """Only known candidate-query counters may change in a private clone."""
    a, b = dict(before), dict(after)
    ta, tb = a.pop('telemetry'), b.pop('telemetry')
    assert a == b, 'candidate changed physical/resource state'
    changes = {k: tb.get(k, 0)-ta.get(k, 0) for k in set(ta)|set(tb) if tb.get(k) != ta.get(k)}
    assert set(changes) <= {'repair_candidates_checked', 'trajectory_infeasible_candidates',
        'energy_reachability_checks', 'energy_reachability_negative_budgets'}
    assert all(v >= 0 for v in changes.values())
    return changes


class NativeWitness:
    """Observe native checks without replacing their numeric return values.

    Caller-local lookup is deliberately confined to this diagnostic. It fails
    closed if the SafetyFilter loop structure changes, rather than guessing a
    blocker UID or silently modifying production safety code.
    """
    def __init__(self, safety):
        self.safety, self.events = safety, []
        self.track = safety._track_distances_with_margin
        self.tube = safety._tube_distances_with_margin

    def install(self):
        def track(left, right, times, ld, rd):
            frame = inspect.currentframe().f_back
            if frame.f_code.co_name == '_tube_distances_with_margin':
                # The opt-in nominal-curve certificate calls this helper
                # inside the outer tube check; the tube event owns its UID.
                return self.track(left, right, times, ld, rd)
            assert frame.f_code.co_name == 'filter_plan'
            local = frame.f_locals
            result = self.track(left, right, times, ld, rd)
            stage = 'post' if times is local.get('post_samples') or times is local.get('nominal_post_times') else 'plan'
            point, swept = result
            event = dict(kind='track', uid=int(local['other_id']), stage=stage,
                point=minimum(point, times), swept=minimum(swept, times[:-1]) if len(swept) else None,
                start=float(times[0]), end=float(times[-1]), samples=len(times),
                left_descriptor=ld, right_descriptor=rd)
            self.events.append(event)
            return result

        def tube(points, ring, lower, upper, times, descriptor):
            frame = inspect.currentframe().f_back
            assert frame.f_code.co_name == 'filter_plan'
            local = frame.f_locals
            result = self.tube(points, ring, lower, upper, times, descriptor)
            point = result[:len(times)]
            intervals = result[len(times):]
            self.events.append(dict(kind='tube', uid=int(local['other_id']),
                stage='post' if times is local.get('post_times') else 'plan',
                point=minimum(point, times), interval_bound=minimum(intervals, times[:-1]) if len(intervals) else None,
                start=float(times[0]), end=float(times[-1]), samples=len(times),
                max_angular_width=float(np.max(upper-lower)), descriptor=descriptor))
            return result
        self.safety._track_distances_with_margin = track
        self.safety._tube_distances_with_margin = tube

    def restore(self):
        self.safety._track_distances_with_margin = self.track
        self.safety._tube_distances_with_margin = self.tube


def observation_context(env, ego):
    selector = TopologyNeighborSelector(env.airspace, now=env.now, cpa_horizon=env.config.shield_horizon_seconds)
    slots = {name: getattr(selector, name)(ego, env.active.values()) for name in SLOT_NAMES}
    report = {name: dict(**asdict(slot), encoded=_slot_vector(slot, env.config)) for name, slot in slots.items()}
    # Report intended *relative* gaps alongside the preserved historical wire
    # values; do not change any observation or feed this correction to a model.
    target = {RingRole.OUTER: RingRole.MIDDLE, RingRole.MIDDLE: RingRole.INNER}.get(ego.ring_role)
    if target is not None:
        source, ring = env.airspace.actual_ring(ego.ring_role), env.airspace.actual_ring(target)
        horizon = abs(source.radius-ring.radius)/max(1., ring.nominal_speed)+6.
        angle = float(ego.ring_progress if ego.ring_progress is not None else np.arctan2(ego.position[1], ego.position[0]))
        expected = (angle+ring.nominal_speed*horizon/ring.radius) % (2*np.pi)
        for name in ('target_ring_predecessor', 'target_ring_successor'):
            slot = slots[name]
            if slot.present:
                other = env.active[slot.uid]
                predicted = (float(other.ring_progress)+ring.nominal_speed*horizon/ring.radius) % (2*np.pi)
                gap = ((predicted-expected) % (2*np.pi)) * ring.radius if name.endswith('predecessor') else -((expected-predicted) % (2*np.pi))*ring.radius
                report[name].update(relative_path_gap=gap, relative_gap_encoded=float(np.clip(gap/(4*env.config.safe_distance), -1, 1)),
                                    baseline_merge_horizon=horizon)
    return report


def dense_event(env, plan, snapshot, event, slots):
    uid = event['uid']
    other = env.active[uid]
    dt = min(env.config.physics_dt, env.config.safety_sampling_dt)/4
    times = grid(event['start'], event['end'], dt)
    safety = env.safety
    left = np.asarray(plan.sample_positions(times) if event['stage'] == 'plan' else safety._post_plan_positions(plan, snapshot, times))
    if other.committed_plan is not None:
        right = safety._committed_positions(other.committed_plan, snapshot, times)
        model = 'committed_canonical_then_nominal_holder_or_stationary_endpoint'
    else:
        assert env.config.holding_speed_contract == 'nominal_until_release' or env.config.max_speed_residual_mps == 0
        right = np.asarray(safety._holder_positions(other, snapshot, times))
        model = 'finite_response_nominal_holder'
    distances = np.linalg.norm(left-right, axis=1)
    m = minimum(distances, times)
    at = m['index']
    clip = slice(max(0, at-2), min(len(times), at+3))
    result = dict(uid=uid, phase=other.phase.value, ring_role=getattr(other.ring_role, 'value', None),
        committed=other.committed_plan is not None, prediction_model=model,
        visible_slots=[name for name, value in slots.items() if value['uid'] == uid],
        initial_position=other.position, initial_velocity=other.velocity,
        current_distance=float(np.linalg.norm(other.position-env.active[plan.uav_id].position)),
        dense_dt=dt, dense_minimum=m, ego_at_min=left[at], other_at_min=right[at],
        dense_trace=dict(times=times[clip], distances=distances[clip], ego=left[clip], other=right[clip]))
    if other.committed_plan is not None:
        op = other.committed_plan
        result.update(other_plan=dict(start=op.start_time, end=op.end_time, mode=op.mode,
            target_phase=op.metadata.get('target_phase'), hash=op.trajectory_hash,
            endpoint=op.points[-1].position, endpoint_velocity=op.points[-1].velocity))
    return result


def candidate_witness(root, index):
    started = time.monotonic()
    window, proposal_index = CANDIDATES[index]
    episode = root/CASES[0][1]/'seed_893001'
    entry = read(episode/'snapshots'/f'window_{window:02d}.json')
    label = read(episode/'labels'/f'window_{window:02d}.json')
    row = entry['row']
    path = episode/'snapshots'/entry['snapshot']
    assert Path(entry['snapshot']).name == entry['snapshot'] and sha(path) == entry['snapshot_sha256']
    env = pickle.loads(path.read_bytes())  # only the above known, hash-checked local run artifact
    assert _digest(canonical(state_record(env))) == row['pre_state_hash'] == label['pre_state_hash']
    ego = env.active[int(row['uid'])]
    assert np.array_equal(build_low_observation_v2_4(ego, env), np.asarray(row['observation'], dtype=np.float32))
    assert ego.ring_role is RingRole.MIDDLE and row['actor_mask']
    assert not env.config.teacher_enabled and env.config.scheduler_mode == 'execution'
    before = canonical(state_record(env))
    slots = observation_context(env, ego)
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    action = JOINT_ACTION_CATALOG_V4_1[proposal_index+1]
    trace = NativeWitness(env.safety)
    trace.install()
    try:
        proposal, plan, reason = env._candidate_plan(ego, RepairCandidate('teacher_context_diagnostic', LowLevelIntent(*action.normalized), 0.), RingRole.INNER, snapshot, None)
    finally:
        trace.restore()
    counter_delta = state_delta(before, canonical(state_record(env)))
    assert plan is not None and proposal is not None
    prefix = [r for r in label['teacher_audit'] if r['joint_index'] == proposal_index+1]
    assert len(prefix) == 1
    expected = prefix[0]
    assert expected['feasible'] == (reason is None)
    assert expected['reason'] == (reason or 'certified')
    certificate_equal = None
    if label['teacher_proposal'] == proposal_index:
        certificate_equal = plan.trajectory_hash == label['selected_certificate']['trajectory_hash']
        assert certificate_equal and env.safety.checked_trajectory_hashes[plan.plan_id] == plan.trajectory_hash
    enriched = []
    for event in trace.events:
        enriched.append(dict(**event, dense=dense_event(env, plan, snapshot, event, slots)))
    values = {name: row['observation'][s.start:s.stop] for name, s in LOW_OBS_V2_4.slices.items()}
    result = dict(status='CANDIDATE_WITNESS_COMPLETE', index=index, decision_id=row['decision_id'],
        tick=row['tick'], uid=row['uid'], proposal_index=proposal_index, request=action.normalized,
        original_teacher_gate=label['teacher_gate'], original_teacher_proposal=label['teacher_proposal'],
        native_reason=reason or 'certified', original_reason_equal=True,
        selected_certificate_equal=certificate_equal, snapshot_sha256=sha(path),
        observation_equal=True, physical_resource_state_unchanged=True, query_counter_delta=counter_delta,
        observation=values, slots=slots,
        config={k:getattr(env.config,k) for k in ('safe_distance','physics_dt','safety_sampling_dt','shield_horizon_seconds',
            'holding_speed_contract','max_speed_residual_mps','max_merge_time_residual_seconds')},
        plan=dict(start=plan.start_time, end=plan.end_time, duration=plan.end_time-plan.start_time,
            hash=plan.trajectory_hash, integrity=plan.verify_trajectory_integrity(),
            dynamics_errors=env.safety.dynamics.validate_trajectory(plan), metadata=plan.metadata,
            merge_time=proposal.merge_time, merge_angle=proposal.merge_angle,
            start_velocity=plan.points[0].velocity, end_velocity=plan.points[-1].velocity),
        events=enriched, elapsed_seconds=time.monotonic()-started,
        candidate_calls=1, teacher_scans=0, environment_steps=0, environment_resets=0, parameter_updates=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-index', type=int, choices=range(6))
    args = parser.parse_args()
    assert not args.output.exists()
    result = audit_labels(args.root) if args.candidate_index is None else candidate_witness(args.root, args.candidate_index)
    write(args.output, result)
    print(json.dumps(dict(status=result['status'], output=str(args.output))))


if __name__ == '__main__':
    main()
