"""Bounded, in-sample 90D BC diagnostic. Not formal I1 or PPO admission.

Only three complete train episodes are allowed. No environment, teacher,
checkpoint loading, audit/Q access, model selection, or deployment happens.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from terminal_marl.algo.joint_ppo_v4_2 import HierarchicalActorCentralCriticV42, hierarchical_bc_loss
from terminal_marl.training.p3_v3_expert_dataset import load_sparse_training_episode, OBSERVATION_SCHEMA

PROTOCOL = 'p3_v3_90d_bc_in_sample_20261007'
BALANCED_PROTOCOL = 'p3_v3_90d_bc_gate_balance_20261007'
DETACHED_PROTOCOL = 'p3_v3_90d_bc_proposal_detach_20261007'
SEPARATE_PROTOCOL = 'p3_v3_90d_bc_separate_encoders_20261007'
GATE_WEIGHTS = (1046 / (2 * 817), 1046 / (2 * 229))
REFERENCE_CASE = 'p3_v3_bc_learnability_20261007/fit'
CASES = (
    (893001, 'p3_v3_i0_20261004/train_01', 59),
    (893101, 'p3_v3_i0_v2_20261004/dense_high_launch2', 613),
    (893201, 'p3_v3_i0_v3_20261007/dense_mixed', 374),
)
MODEL_SEEDS = (91701, 91702, 91703)
UPDATES, BATCH_SIZE, LEARNING_RATE = 1000, 128, 3e-4
FIT_WALL_SECONDS = 300


class SeparateEncodersDiagnosticV42(HierarchicalActorCentralCriticV42):
    """Offline diagnostic only; preserve the shared model's initial outputs/RNG."""

    def __init__(self, observation_dim: int, hidden: int = 96):
        super().__init__(observation_dim, hidden)
        self.proposal_encoder = deepcopy(self.actor_encoder)

    def distributions(self, observations):
        gate = self.gate_head(self.actor_encoder(observations))
        proposal = self.proposal_head(self.proposal_encoder(observations))
        return torch.distributions.Categorical(logits=gate), torch.distributions.Categorical(logits=proposal)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


@dataclass
class Data:
    x: torch.Tensor
    gate: torch.Tensor
    proposal: torch.Tensor
    ids: tuple[str, ...]
    contexts: list[dict]
    inputs: list[dict]


def check_complete(result):
    require(result['status'] == 'COLLECTION_OK' and result['error'] is None and not result['blockers'],
            'only complete collection episodes are allowed')
    flow, annotation = result['rollout'], result['annotation']
    require(flow['terminal'] and not flow['censored'] and flow['error'] is None,
            'incomplete environment cycle')
    require(flow['selection_overflow'] == flow['unresolved'] == flow['active'] == flow['queue'] == flow['service'] == 0,
            'collection contains unresolved or overflow rows')
    require(set(flow['outcome_counts']) <= {'completed', 'emergency_landed'}
            and sum(flow['outcome_counts'].values()) == flow['requested'], 'invalid business terminal')
    require(annotation['status'] == 'COMPLETED' and annotation['selected'] == annotation['labelled']
            and all(e['status'] == 'OK' for e in annotation['entries']), 'incomplete annotation')
    for key in ('hard_separation_violations', 'nmac', 'collisions', 'checked_executed_plan_hash_mismatches'):
        require(flow['telemetry'][key] == 0, f'unsafe collection: {key}')


def load_inputs(root, *, cases=CASES):
    """No model/optimizer construction. Preserve episode identities and row order."""
    arrays, gates, proposals, ids, contexts, inputs = [], [], [], [], [], []
    for seed, relative, expected in cases:
        case = Path(root)/relative
        result, protocol = read(case/'result.json'), read(case/'protocol.json')
        require(result['seed'] == protocol['seed'] == seed, 'input seed mismatch')
        check_complete(result)
        batch = load_sparse_training_episode(case/f'seed_{seed}', allowed_train_seeds={seed})
        require(batch.summary == result['dataset_check'], 'loader differs from collection counts')
        require(len(batch.decision_ids) == expected, 'frozen episode row count mismatch')
        require(batch.summary['actor_eligible_rows'] == batch.summary['gate_supervision_rows']
                == batch.summary['valid_labels'] == expected, 'not every actor has exactly one label')
        raw = [json.loads(line) for line in (case/f'seed_{seed}'/'rows.jsonl').read_text(encoding='utf8').splitlines()]
        eligible, prior = [], {}
        for row in raw:
            old = prior.get(row['uid'])
            if row['actor_mask']:
                eligible.append(row['decision_id'])
                contexts.append(dict(seed=seed, interface=row['interface'], low_soc=row['soc'] < .55,
                    fast_inner=row['interface'] == 'i2f' and row['speed'] >= 15,
                    busy_lz=row['busy_lz'] > 0, after_retry=bool(old and old['retry_active'])))
            prior[row['uid']] = row
        require(tuple(eligible) == batch.decision_ids, 'actor row order differs from loader')
        arrays.append(batch.observations); gates.append(batch.gate_target); proposals.append(batch.proposal_target)
        ids.extend(batch.decision_ids)
        inputs.append(dict(seed=seed, relative_case=relative, collection_protocol=protocol['protocol'],
            summary=batch.summary, outcomes=result['rollout']['outcome_counts']))
    require(len(ids) == len(set(ids)), 'duplicate input decision IDs')
    return Data(torch.from_numpy(np.concatenate(arrays)), torch.from_numpy(np.concatenate(gates)),
        torch.from_numpy(np.concatenate(proposals)), tuple(ids), contexts, inputs)


def classification(gate_logits, proposal_logits, gate, proposal):
    """Conditional proposal metrics use teacher RELEASE, regardless of predicted gate."""
    n = len(gate)
    if not n:
        return dict(rows=0, gate=None, proposal=None, joint_accuracy=None)
    pred, pp = gate_logits.argmax(-1), proposal_logits.argmax(-1)
    confusion = [[int(((gate == y) & (pred == p)).sum()) for p in (0, 1)] for y in (0, 1)]
    recall = [confusion[y][y]/sum(confusion[y]) if sum(confusion[y]) else None for y in (0, 1)]
    release = gate == 1
    n_release = int(release.sum())
    conditional = None
    if n_release:
        top3 = proposal_logits[release].topk(3, dim=-1).indices
        conditional = dict(rows=n_release, top1=float((pp[release] == proposal[release]).float().mean()),
            top3=float((top3 == proposal[release, None]).any(-1).float().mean()),
            target_counts=dict(Counter(proposal[release].tolist())),
            prediction_counts=dict(Counter(pp[release].tolist())),
            per_class_recall={str(k): float((pp[proposal == k] == k).float().mean())
                              for k in sorted(set(proposal[release].tolist()))})
    correct = (pred == gate) & ((gate == 0) | (pp == proposal))
    return dict(rows=n, gate=dict(accuracy=float((pred == gate).float().mean()), confusion=confusion,
        WAIT_recall=recall[0], RELEASE_recall=recall[1],
        balanced_accuracy=sum(recall)/2 if all(x is not None for x in recall) else None,
        prediction_RELEASE=int((pred == 1).sum()), prediction_WAIT=int((pred == 0).sum())),
        proposal=conditional, joint_accuracy=float(correct.float().mean()))


def context_masks(data):
    groups = {'overall': torch.ones(len(data.gate), dtype=torch.bool)}
    for seed in sorted({r['seed'] for r in data.contexts}):
        groups[f'episode_{seed}'] = torch.tensor([r['seed'] == seed for r in data.contexts])
    for key in ('o2m', 'm2i', 'i2f'):
        groups[key] = torch.tensor([r['interface'] == key for r in data.contexts])
    for key in ('low_soc', 'fast_inner', 'busy_lz', 'after_retry'):
        groups[key] = torch.tensor([r[key] for r in data.contexts])
    return groups


@torch.no_grad()
def measure(model, data):
    model.eval()
    g, p = model.distributions(data.x)
    loss, pieces = hierarchical_bc_loss(g.logits, p.logits, data.gate, data.proposal)
    return dict(loss=float(loss), gate_loss=float(pieces['gate']), proposal_loss=float(pieces['proposal']),
        groups={name: classification(g.logits[mask], p.logits[mask], data.gate[mask], data.proposal[mask])
                for name, mask in context_masks(data).items()})


def gate_auc(scores, labels):
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    difference = positive[:, None] - negative[None, :]
    return float(((difference > 0).double() + .5 * (difference == 0).double()).mean())


@torch.no_grad()
def measure_auc(model, data):
    model.eval()
    gate, _ = model.distributions(data.x)
    score = gate.logits[:, 1] - gate.logits[:, 0]
    return {name: gate_auc(score[mask], data.gate[mask]) for name, mask in context_masks(data).items()}


def training_distributions(model, observations, detach_proposal_encoder=False):
    if not detach_proposal_encoder:
        return model.distributions(observations)
    encoded = model.actor_encoder(observations)
    return (torch.distributions.Categorical(logits=model.gate_head(encoded)),
            torch.distributions.Categorical(logits=model.proposal_head(encoded.detach())))


def baseline(data):
    return dict(all_WAIT_accuracy=float((data.gate == 0).double().mean()),
                all_WAIT_balanced_accuracy=.5 if set(data.gate.tolist()) == {0, 1} else None,
                proposal_majority_class=Counter(data.proposal[data.gate == 1].tolist()).most_common(1)[0])


def training_loss(gate_logits, proposal_logits, gate, proposal, gate_weights=None):
    loss, pieces = hierarchical_bc_loss(gate_logits, proposal_logits, gate, proposal)
    if gate_weights is None:
        return loss
    weights = gate_logits.new_tensor(gate_weights)
    # Fixed full-dataset weights, divided by batch ROW count, not weight sum.
    ce = torch.nn.functional.cross_entropy(gate_logits, gate, reduction='none')
    return (ce * weights[gate]).mean() + pieces['proposal']


@torch.no_grad()
def objective(model, data, gate_weights):
    g, p = model.distributions(data.x)
    return float(training_loss(g.logits, p.logits, data.gate, data.proposal, gate_weights))


def reference_results(root, data):
    """Reuse the completed unweighted arm; never refit it or load its model."""
    case = Path(root)/REFERENCE_CASE
    result, inputs, protocol = (read(case/f'{name}.json') for name in ('result', 'inputs', 'protocol'))
    require(result['status'] == 'FIT_COMPLETE' and result['protocol'] == PROTOCOL,
            'missing completed unweighted reference')
    require(inputs['decision_ids'] == list(data.ids) and inputs['episodes'] == data.inputs,
            'reference dataset differs')
    require(protocol['model_seeds'] == list(MODEL_SEEDS) and protocol['updates'] == UPDATES
            and protocol['batch_size'] == BATCH_SIZE and protocol['learning_rate'] == LEARNING_RATE,
            'reference settings differ')
    require(inputs['counts'] == {'rows': 1046, 'WAIT': 817, 'RELEASE': 229}, 'reference counts differ')
    require([r['seed'] for r in result['seeds']] == list(MODEL_SEEDS), 'reference seeds differ')
    for r in result['seeds']:
        require(r['status'] == 'FIT_COMPLETE' and r['updates'] == UPDATES and r['critic_unchanged'],
                'reference seed incomplete')
    return {r['seed']: r for r in result['seeds']}


def fit(data, out, seed, *, updates=UPDATES, wall_seconds=FIT_WALL_SECONDS,
        gate_weights=None, protocol=PROTOCOL, reference_initial=None, detach_proposal_encoder=False,
        separate_encoders=False):
    """CPU-only fixed-budget BC. No use of audit/Q or execution trace as targets."""
    require(sum((gate_weights is not None, detach_proposal_encoder, separate_encoders)) <= 1,
            'only one diagnostic factor allowed')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.manual_seed(seed)
    model = SeparateEncodersDiagnosticV42(90) if separate_encoders else HierarchicalActorCentralCriticV42(90)
    for name, param in model.named_parameters():
        param.requires_grad_(not name.startswith('critic_'))
    params = [p for p in model.parameters() if p.requires_grad]
    added_parameters = sum(p.numel() for p in model.proposal_encoder.parameters()) if separate_encoders else 0
    parameter_counts = dict(actor=sum(p.numel() for p in params), added_proposal_encoder=added_parameters,
        reference_actor=sum(p.numel() for p in params)-added_parameters,
        critic=sum(p.numel() for p in model.parameters() if not p.requires_grad))
    critic_before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith('critic_')}
    actor_before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('critic_')}
    initial = measure(model, data)
    if reference_initial is not None:
        # JSON object keys are strings; Counter keys in freshly measured metrics
        # are integers. Compare the persisted representation without float tolerances.
        require(json.loads(json.dumps(initial, allow_nan=False)) == reference_initial,
                'paired initialization metrics differ')
    initial_objective = objective(model, data, gate_weights)
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)
    permutation, offset = torch.randperm(len(data.x), generator=generator), 0
    maximum_grad = 0.
    for update in range(1, updates+1):
        require(time.monotonic()-start < wall_seconds, 'fit wall budget exhausted')
        if offset == len(data.x):
            permutation, offset = torch.randperm(len(data.x), generator=generator), 0
        ids = permutation[offset:offset+BATCH_SIZE]; offset += len(ids)
        model.train(); optimizer.zero_grad(set_to_none=True)
        g, p = training_distributions(model, data.x[ids], detach_proposal_encoder)
        loss = training_loss(g.logits, p.logits, data.gate[ids], data.proposal[ids], gate_weights)
        require(bool(torch.isfinite(loss)), 'nonfinite BC loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, .5, error_if_nonfinite=True)
        maximum_grad = max(maximum_grad, float(norm))
        optimizer.step()
        if update == 1 or update % 100 == 0:
            write(out/'progress.json', dict(update=update, budget=updates, seed=seed,
                batch_loss=float(loss.detach()), elapsed_seconds=time.monotonic()-start))
    require(all(torch.isfinite(p).all() for p in model.parameters()), 'nonfinite parameters')
    require(all(torch.equal(v, model.state_dict()[k]) for k, v in critic_before.items()), 'critic changed')
    final = measure(model, data)
    changed = any(not torch.equal(v, model.state_dict()[k]) for k,v in actor_before.items())
    result = dict(status='FIT_COMPLETE', seed=seed, updates=updates, initial=initial, final=final,
        actor_changed=changed, critic_unchanged=True, maximum_preclip_gradient_norm=maximum_grad,
        detach_proposal_encoder=detach_proposal_encoder, separate_encoders=separate_encoders,
        parameter_counts=parameter_counts,
        actor_module_changed={prefix: any(not torch.equal(v, model.state_dict()[k])
            for k, v in actor_before.items() if k.startswith(prefix))
            for prefix in ('actor_encoder', 'gate_head', 'proposal_head') + (('proposal_encoder',) if separate_encoders else ())},
        final_gate_auc=measure_auc(model, data) if detach_proposal_encoder or separate_encoders else None,
        gate_weights=gate_weights, initial_training_objective=initial_objective,
        final_training_objective=objective(model, data, gate_weights),
        reference_initial_equal=True if reference_initial is not None else None,
        elapsed_seconds=time.monotonic()-start, evaluation='in_sample_only', student_qualified=False,
        I0_QUALIFIED=False, PPO_ALLOWED=False)
    require(result['elapsed_seconds'] <= wall_seconds, 'fit wall budget exhausted')
    # Fresh diagnostic artifact only; never overwrite or select by endpoint score.
    torch.save(dict(protocol=protocol, gate_weights=gate_weights, detach_proposal_encoder=detach_proposal_encoder,
        separate_encoders=separate_encoders, parameter_counts=parameter_counts,
        observation_schema=OBSERVATION_SCHEMA, observation_dim=90,
        action_schema='low_action_schema_v4_2_hierarchical', model_class=type(model).__name__,
        model_state=model.state_dict(), model_seed=seed, train_seeds=[x['seed'] for x in data.inputs],
        diagnostic_only=True, qualified=False, PPO_ALLOWED=False), out/'final_diagnostic.pt')
    write(out/'result.json', result)
    return result


def synthetic_data():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(16, 90, generator=generator)
    gates = torch.arange(16) % 2
    proposals = torch.where(gates == 1, torch.arange(16) % 25, -1)
    contexts = [dict(seed=0, interface=('o2m','m2i','i2f')[i%3],
        low_soc=False, fast_inner=False, busy_lz=False, after_retry=False) for i in range(16)]
    return Data(x, gates, proposals, tuple(f'synthetic:{i}' for i in range(16)), contexts, [])


def run(root, output, mode, *, gate_balance=False, detach_proposal_encoder=False, separate_encoders=False):
    require(mode in {'prepare', 'smoke', 'fit'}, 'unknown mode')
    require(sum((gate_balance, detach_proposal_encoder, separate_encoders)) <= 1, 'only one diagnostic factor allowed')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    protocol = (SEPARATE_PROTOCOL if separate_encoders else DETACHED_PROTOCOL if detach_proposal_encoder
                else BALANCED_PROTOCOL if gate_balance else PROTOCOL)
    weights = GATE_WEIGHTS if gate_balance else None
    settings = dict(protocol=protocol, mode=mode, model_seeds=list(MODEL_SEEDS), updates=UPDATES,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, gradient_clip=.5,
        optimizer='Adam', loss='gate CE + teacher-RELEASE conditional proposal CE; unweighted',
        evaluation='in_sample_initial_and_fixed_final_only', model='HierarchicalActorCentralCriticV42',
        observation_dim=90, hidden=96, critic_frozen=True, device='cpu', fit_wall_seconds=FIT_WALL_SECONDS,
        I0_QUALIFIED=False, PPO_ALLOWED=False, checkpoint_selection=False, environment_resets=0,
        teacher_queries=0, audit_Q_accessed=False)
    if gate_balance:
        settings.update(gate_weights=list(weights), gate_reduction='fixed-weight CE sum / batch row count',
            loss='fixed-class-weight gate CE + unchanged teacher-RELEASE conditional proposal CE',
            reported_loss='unweighted, identical definition to reference', reference=REFERENCE_CASE)
    if detach_proposal_encoder:
        settings.update(detach_proposal_encoder=True, gate_weights=None,
            reference=REFERENCE_CASE, proposal_head_trainable=True,
            gradient_clipping='unchanged joint actor norm clip .5; proposal head still included',
            final_gate_auc='pairwise logit-score ranking; ties half; missing class null')
    if separate_encoders:
        settings.update(separate_encoders=True, detach_proposal_encoder=False, gate_weights=None,
            model='SeparateEncodersDiagnosticV42', reference=REFERENCE_CASE,
            encoder_initialization='deep copy initial actor_encoder after original model initialization; no extra RNG',
            gradient_clipping='joint actor norm clip .5 including both encoders and heads',
            final_gate_auc='pairwise logit-score ranking; ties half; missing class null')
    if mode == 'smoke':
        settings.update(model_seeds=[17], updates=2, synthetic_only=True)
    write(output/'protocol.json', settings)
    report = dict(status='ERROR', mode=mode, protocol=protocol, I0_QUALIFIED=False, PPO_ALLOWED=False)
    try:
        data = synthetic_data() if mode == 'smoke' else load_inputs(root)
        reference = reference_results(root, data) if (gate_balance or detach_proposal_encoder or separate_encoders) and mode != 'smoke' else None
        counts = dict(rows=len(data.x), WAIT=int((data.gate==0).sum()), RELEASE=int((data.gate==1).sum()))
        write(output/'inputs.json', dict(episodes=data.inputs, counts=counts, baselines=baseline(data),
            decision_ids=list(data.ids)))
        if mode == 'prepare':
            report.update(status='PREPARED', optimizer_created=False, trained=False, counts=counts)
        else:
            report['seeds'] = []
            seeds, updates = ((17,), 2) if mode == 'smoke' else (MODEL_SEEDS, UPDATES)
            for seed in seeds:
                report['seeds'].append(fit(data, output/f'seed_{seed}', seed, updates=updates,
                    gate_weights=weights, protocol=protocol,
                    reference_initial=reference[seed]['initial'] if reference else None,
                    detach_proposal_encoder=detach_proposal_encoder, separate_encoders=separate_encoders))
                write(output/'progress.json', dict(completed_seeds=[r['seed'] for r in report['seeds']]))
            report.update(status='SMOKE_OK' if mode == 'smoke' else 'FIT_COMPLETE', trained=True,
                optimizer_created=True, synthetic_only=mode=='smoke', counts=counts)
    except Exception as exc:
        report.update(status='ERROR', error=dict(type=type(exc).__name__, message=str(exc)))
    write(output/'result.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('prepare','smoke','fit'), required=True)
    variant = p.add_mutually_exclusive_group()
    variant.add_argument('--gate-balance', action='store_true', help='fixed paired gate-weight diagnostic')
    variant.add_argument('--detach-proposal-encoder', action='store_true', help='stop proposal gradient into encoder only')
    variant.add_argument('--separate-encoders', action='store_true', help='independent trainable gate/proposal encoders, same initial weights')
    args = p.parse_args()
    result = run(args.root, args.output, args.mode, gate_balance=args.gate_balance,
                 detach_proposal_encoder=args.detach_proposal_encoder, separate_encoders=args.separate_encoders)
    print(json.dumps(dict(status=result['status'], mode=args.mode)))
    return int(result['status'] == 'ERROR')


if __name__ == '__main__':
    raise SystemExit(main())
