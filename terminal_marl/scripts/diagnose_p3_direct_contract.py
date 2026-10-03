"""Three fixed trusted development snapshots; one request and two replay copies each.

This is a no-gradient interface preflight, not student qualification. The policy
RNG is new and fixed, not the unavailable RNG of the original snapshot run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import torch

from terminal_marl.algo.service_context_migration_v4_2 import EXPANDED_INPUT_WEIGHT_NAMES
from terminal_marl.algo.service_context_runtime_v4_2 import initialize_service_context_policy
from terminal_marl.configs.p3_integrated_repair import build_config, load_legacy_source
from terminal_marl.core.low_level_joint_action_v4_2 import HierarchicalActionV42, hierarchical_from_normalized_v4_2
from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest, _json
from terminal_marl.training.student_rollout_v4_2 import _student_request_details, validate_student_config


CASES = {
    'A': ('4d4a6961d8be636158a8db25c9681d4e086523c97d6730595cacdfeddbc1be9f', 891701),
    'B': ('9caa7b67cd358c52b127d4261601eca65909c57ec7a07101151c4efe32a34427', 891702),
    'C': ('236adbd8bac2b48834d185e74915c805a5bf17e9d6d20e2cf935ee97162392d3', 891703),
}
CHECKPOINT_SHA = '3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab'
ATOL = 1e-6  # Existing service-context migration contract; rtol is zero.
SAFETY_KEYS = (
    'collisions', 'nmac', 'hard_separation_violations', 'operational_buffer_entries',
    'shield_fail_closed', 'joint_safety_fail_closed', 'runtime_traffic_interventions',
    'checked_executed_plan_hash_mismatches',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(_json(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf8')


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def canonical(value):
    # Use the existing replay contract: opaque UUID spelling is irrelevant;
    # trajectory content hashes, values, ranks and timings remain compared.
    return _json(SafetyTraceRecorder._replay_state_value(value))


def state_record(env):
    plans = {}
    energy = {}
    for uid, uav in sorted(env.active.items()):
        energy[str(uid)] = dict(soc=uav.soc, remaining=uav.energy_remaining,
                                path_length=uav.path_length)
        plan = uav.committed_plan
        if plan is not None:
            checked = env.safety.checked_trajectory_hashes.get(plan.plan_id)
            require(checked == plan.trajectory_hash, f'unchecked active trajectory uid={uid}')
            plans[str(uid)] = dict(trajectory_hash=plan.trajectory_hash, checked_hash=checked,
                                   mode=plan.mode, start=plan.start_time, end=plan.end_time)
    return dict(state=SafetyTraceRecorder._state(env), energy=energy, plans=plans,
                requested=sorted(env.ledger.requested), outcomes=dict(env.ledger.outcomes),
                unresolved=sorted(env.ledger.unresolved), telemetry=dict(env.telemetry),
                service=sorted(env.service_uavs), queue_size=len(env.admission.queue.items),
                energy_outcomes=env.energy_outcomes,
                external_queue_uav_seconds=env.external_queue_uav_seconds)


def check_requests(policy, env, observations, details):
    uids, world, mask, actions, gates, proposals, gate_lp, proposal_lp, likelihoods, shortlists = details
    require(uids == sorted(observations), 'UID order differs from observation order')
    require(set(uids) == {str(uid) for uid in env.active}, 'UID coverage differs from active UAVs')
    require(world.shape == (len(uids), 90) and np.isfinite(world).all(), 'invalid 90D world')
    require(np.array_equal(world, np.array([observations[uid] for uid in uids], dtype=np.float32)),
            'world rows differ from observations')
    require(set(actions) == set(uids) == set(likelihoods), 'action/likelihood UID mismatch')
    affected = set(env.airspace.affected_interfaces())
    with torch.no_grad():
        gd, pd = policy.model.distributions(torch.as_tensor(world, device=policy.device))
    gate_logits, proposal_logits = gd.logits.cpu().numpy(), pd.logits.cpu().numpy()
    rows = []
    for index, uid in enumerate(uids):
        uav = env.active[int(uid)]
        interface = {'outer': 'o2m', 'middle': 'm2i', 'inner': 'i2f'}.get(
            getattr(uav.ring_role, 'value', None))
        eligible = (uav.committed_plan is None and uav.phase.value in
                    {'ring_outer', 'ring_middle', 'ring_inner'} and
                    int(uid) not in env.future_release_reservations and interface is not None and
                    uav.soc > 0 and env.now >= uav.safety_retry_after and interface not in affected)
        require(bool(mask[index]) == eligible, f'forced-WAIT mask mismatch uid={uid}')
        gate, proposal = int(gates[index]), int(proposals[index])
        require(gate in (0, 1) and (eligible or gate == 0), f'invalid gate uid={uid}')
        require((gate == 0 and proposal == -1) or (gate == 1 and 0 <= proposal < 25),
                f'conditional proposal sentinel mismatch uid={uid}')
        decoded = HierarchicalActionV42(gate, None if gate == 0 else proposal)
        require(np.array_equal(actions[uid], decoded.normalized), f'action decode mismatch uid={uid}')
        expected_gate = float(gate_logits[index, gate]) if eligible else 0.0
        expected_proposal = None if gate == 0 else float(proposal_logits[index, proposal])
        expected_joint = expected_gate + (expected_proposal or 0.0)
        expected = dict(gate=expected_gate, proposal=expected_proposal, joint=expected_joint)
        for key, value in expected.items():
            actual = likelihoods[uid][key]
            require(actual is None if value is None else
                    actual is not None and abs(actual-value) <= ATOL,
                    f'requested {key} likelihood mismatch uid={uid}')
        require(abs(float(gate_lp[index])-expected_gate) <= ATOL, 'gate array mismatch')
        require(np.isnan(proposal_lp[index]) if gate == 0 else
                abs(float(proposal_lp[index])-expected_proposal) <= ATOL, 'proposal array mismatch')
        ranks = []
        if gate == 1:
            require(uid in shortlists and len(shortlists[uid]) == 3, 'missing K3 shortlist')
            ranks = [hierarchical_from_normalized_v4_2(action).proposal for action in shortlists[uid]]
            ranked = sorted(range(25), key=lambda p: (-float(proposal_logits[index, p]), p))
            require(ranks == [proposal] + [p for p in ranked if p != proposal][:2],
                    f'K3 sampled rank-1/conditional ranking mismatch uid={uid}')
        else:
            require(uid not in shortlists, 'WAIT has a shortlist')
        rows.append(dict(uid=uid, actor=eligible, gate=gate, proposal=None if gate == 0 else proposal,
                         likelihood=likelihoods[uid], shortlist_proposals=ranks,
                         gate_logits=gate_logits[index], proposal_logits=proposal_logits[index]))
    require(set(shortlists) == {uid for uid, gate in zip(uids, gates) if gate == 1}, 'extra shortlist UID')
    return dict(uids=uids, world=world, rows=rows, actions=actions, shortlists=shortlists,
                counts=dict(Counter('RELEASE' if row['gate'] else 'actor_WAIT' if row['actor']
                                    else 'forced_WAIT' for row in rows)))


def check_migration(source, target, world):
    original = source.model.state_dict()
    for key, value in target.model.state_dict().items():
        if key in EXPANDED_INPUT_WEIGHT_NAMES:
            require(torch.equal(value[:, :88], original[key]), f'prefix weight changed: {key}')
            require(torch.count_nonzero(value[:, 88:]).item() == 0, f'nonzero appended weights: {key}')
        else:
            require(torch.equal(value, original[key]), f'copied tensor changed: {key}')
    obs = torch.as_tensor(world, dtype=torch.float32)
    mask = torch.ones((1, len(world)), dtype=torch.bool)
    with torch.no_grad():
        sg, sp = source.model.distributions(obs[:, :88])
        tg, tp = target.model.distributions(obs)
        sv = source.model.critic(obs[None, :, :88], mask)
        tv = target.model.critic(obs[None], mask)
    differences = {name: float(torch.max(torch.abs(a-b))) for name, a, b in
                   [('gate_logits', sg.logits, tg.logits), ('proposal_logits', sp.logits, tp.logits),
                    ('critic', sv, tv)]}
    require(all(np.isfinite(v) and v <= ATOL for v in differences.values()),
            f'88/90 output difference exceeds fixed tolerance: {differences}')
    return dict(max_abs_differences=differences, atol=ATOL, rtol=0, copied_tensors_exact=True)


def execute_tape(env, tape):
    actions = {uid: np.array(row, dtype=np.float32) for uid, row in tape['actions'].items()}
    shortlists = {uid: tuple(np.array(row, dtype=np.float32) for row in rows)
                 for uid, rows in tape['shortlists'].items()}
    before = dict(env.telemetry)
    outputs = env.step(actions, shortlists)
    observations, rewards, _, _, info = outputs
    require(all(np.isfinite(obs).all() for obs in observations.values()), 'nonfinite next observation')
    require(all(np.isfinite(r) for r in rewards.values()) and np.isfinite(info['team_reward']),
            'nonfinite reward')
    for key in SAFETY_KEYS:
        require(key in before and key in env.telemetry, f'missing safety counter {key}')
        require(env.telemetry[key] == before[key] == 0, f'safety counter nonzero: {key}')
    for uid, trace in info['action_traces'].items():
        if uid not in actions:  # Admission can add a UAV during this tick.
            continue
        require(np.array_equal(trace['requested_low_action'], actions[uid]), 'executed request differs from tape')
        require(not trace['teacher_used'], 'unexpected teacher substitution')
        rank = trace['runtime_shortlist_rank']
        if rank is not None:
            require(uid in shortlists and 1 <= rank <= 3, 'invalid executed shortlist rank')
            require(np.array_equal(trace['execution_applied_action'], shortlists[uid][rank-1]),
                    'executed action differs from selected K3 rank')
    return canonical(dict(outputs=outputs, after=state_record(env),
                          safety_delta={key: env.telemetry[key]-before[key] for key in SAFETY_KEYS}))


def run(case, input_root, checkpoint, output):
    started = time.monotonic()
    result = dict(case=case, status='ERROR', error=None, PPO_ALLOWED=False,
                  qualification='NOT_QUALIFIED', trained=False, optimizer_created=False,
                  logical_decision_ticks=0, env_step_calls=0, policy_requests=0,
                  policy_seed=CASES[case][1], original_policy_rng_reconstructed=False)
    try:
        snapshot = Path(input_root) / f'{case}_source' / 'snapshot_400.pkl'
        result.update(snapshot=str(snapshot), snapshot_sha256=sha256(snapshot),
                      checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        require(result['snapshot_sha256'] == CASES[case][0], 'unregistered snapshot hash')
        require(result['checkpoint_sha256'] == CHECKPOINT_SHA, 'unregistered checkpoint hash')
        data = snapshot.read_bytes()
        env = pickle.loads(data)  # Only the hash-bound trusted local experiment above.
        expected = build_config(requests=12 if case == 'A' else 24,
                                warm_start=6 if case == 'A' else 12, high_soc=case != 'C')
        require(asdict(env.config) == asdict(expected), 'snapshot config differs from fixed case')
        require(env.step_count == 400 and env.now == 400.0, 'snapshot is not tick 400')
        validate_student_config(env.config)
        source, payload = load_legacy_source(checkpoint, env.config)
        policy, migration = initialize_service_context_policy(source, source_checkpoint_metadata=payload['metadata'])
        require(not hasattr(source, 'optimizer') and not hasattr(policy, 'optimizer'), 'optimizer present')
        require(not any(p.requires_grad for p in policy.model.parameters()), 'target is not frozen')
        before_weights = [tensor_digest(source.model), tensor_digest(policy.model)]
        before = canonical(state_record(env))
        observations = env._observations()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(CASES[case][1])
            details = _student_request_details(policy, env, observations, deterministic=False, shortlist_k=3)
        result['policy_requests'] = 1
        tape = check_requests(policy, env, observations, details)
        result['migration_check'] = check_migration(source, policy, details[1])
        result['migration'] = migration
        require(before == canonical(state_record(env)), 'request construction changed physical state')
        write_json(output / 'request_tape.json', tape)
        tape = json.loads((output / 'request_tape.json').read_text(encoding='utf8'))
        result['request_counts'] = tape['counts']
        copies = [pickle.loads(data), pickle.loads(data)]
        replay = []
        for index, copy in enumerate(copies):
            require(before == canonical(state_record(copy)), 'initial replay state differs')
            result['env_step_calls'] += 1
            replay.append(execute_tape(copy, tape))
            write_json(output / f'replay_{index+1}.json', replay[-1])
        require(replay[0] == replay[1], 'independent one-tick replay differs; inspect both records')
        require(before_weights == [tensor_digest(source.model), tensor_digest(policy.model)], 'weights mutated')
        require(sha256(checkpoint) == CHECKPOINT_SHA, 'checkpoint file changed')
        result.update(status='PASS', logical_decision_ticks=1, weights_unchanged=True,
                      replay_equal=True, replay_sha256=_digest(replay[0]),
                      actual_ranks={uid: trace['runtime_shortlist_rank'] for uid, trace in
                                    replay[0]['outputs'][4]['action_traces'].items()},
                      safety_delta=replay[0]['safety_delta'],
                      checked_executed_samples_delta=copies[0].telemetry['checked_executed_plan_samples']-
                          env.telemetry['checked_executed_plan_samples'])
    except Exception as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    result['elapsed_seconds'] = time.monotonic()-started
    write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=CASES)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve earlier evidence')
    args.output.mkdir(parents=True)
    result = run(args.case, args.input_root, args.checkpoint, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
