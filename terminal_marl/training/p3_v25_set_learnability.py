"""Fixed old/new-information comparison on already consumed train states.

No rollout, teacher query, audit/Q, threshold search or checkpoint selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..algo.trajectory_set_actor import TrajectorySetActor
from ..algo.joint_ppo_v4_2 import hierarchical_bc_loss
from ..core.observations import LOW_OBS_V2_4
from ..core.trajectory_context_observation import VERSION
from .p3_v25_context_dataset import load_context_dataset
from .p3_v3_bc_learnability import (Data, load_inputs, classification, context_masks,
    gate_auc, baseline, require, write, MODEL_SEEDS, UPDATES, BATCH_SIZE, LEARNING_RATE)

PROTOCOL = 'p3_v25_same_network_information_pair_20261008'
RECONSTRUCTION = 'p3_v3_observation_v25_20261008/full'
ARMS = ('old_information', 'new_information')
WALL_SECONDS = 300


@dataclass
class PairedInputs:
    original: Data
    new: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def tensors(self, arm):
        if arm == 'new_information':
            return self.new
        require(arm == 'old_information', 'unknown information arm')
        ego, traffic, mask = self.new
        old_ego = torch.zeros_like(ego)
        old_ego[:, :90] = self.original.x
        # No leakage through number of neighbors, padding or absent-branch bias.
        return old_ego, torch.zeros_like(traffic), torch.zeros_like(mask)


def load_paired(root):
    original = load_inputs(root)
    rebuilt = load_context_dataset(Path(root)/RECONSTRUCTION, allowed_train_seeds={893001, 893101, 893201})
    require(len(rebuilt.decision_ids) == 1046 and set(original.ids) == set(rebuilt.decision_ids),
            'old/new decision sets differ')
    # Join by stable IDs; annotation and raw orders need not coincide.
    by_id = {did: i for i, did in enumerate(original.ids)}
    order = torch.tensor([by_id[did] for did in rebuilt.decision_ids])
    original = Data(original.x[order], original.gate[order], original.proposal[order],
        rebuilt.decision_ids, [original.contexts[i] for i in order.tolist()], original.inputs)
    require(np.array_equal(original.gate.numpy(), rebuilt.gate_target)
            and np.array_equal(original.proposal.numpy(), rebuilt.proposal_target), 'labels changed')
    new = tuple(torch.from_numpy(a) for a in rebuilt.padded())
    preserved = torch.ones(90, dtype=torch.bool)
    for name in ('target_ring_predecessor', 'target_ring_successor'):
        preserved[LOW_OBS_V2_4.slices[name]] = False
    require(torch.equal(new[0][:, :90][:, preserved], original.x[:, preserved]),
            'unintended changes to original observation fields')
    require(len(original.gate) == 1046 and int(original.gate.sum()) == 229, 'frozen counts changed')
    return PairedInputs(original, new)


def digest_state(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def measure(model, tensors, data):
    model.eval()
    gate, proposal = model(*tensors)
    loss, pieces = hierarchical_bc_loss(gate, proposal, data.gate, data.proposal)
    masks = context_masks(data)
    result = dict(loss=float(loss), gate_loss=float(pieces['gate']), proposal_loss=float(pieces['proposal']),
        groups={k: classification(gate[m], proposal[m], data.gate[m], data.proposal[m]) for k, m in masks.items()},
        auc={k: gate_auc((gate[:, 1]-gate[:, 0])[m], data.gate[m]) for k, m in masks.items()})
    predictions = [dict(decision_id=did, release_probability=float(gate[i].softmax(-1)[1]),
                        gate=int(gate[i].argmax()), proposal=int(proposal[i].argmax()),
                        proposal_top3=proposal[i].topk(3).indices.tolist()) for i, did in enumerate(data.ids)]
    return result, predictions


@torch.no_grad()
def forward_check(paired):
    torch.manual_seed(MODEL_SEEDS[0])
    model = TrajectorySetActor()
    before = digest_state(model.state_dict())
    x = tuple(t[:16] for t in paired.new)
    direct = model(*x)
    permuted = model(x[0], x[1].flip(1), x[2].flip(1))
    singleton = tuple(torch.cat([model(*(t[i:i+1] for t in x))[j] for i in range(len(x[0]))]) for j in (0, 1))
    for values in (permuted, singleton):
        for a, b in zip(direct, values):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    old = paired.tensors('old_information')
    require(torch.equal(old[0][:, :90], paired.original.x) and not old[0][:, 90:].any()
            and not old[1].any() and not old[2].any(), 'old-information leakage')
    require(before == digest_state(model.state_dict()), 'forward changed parameters')
    return dict(status='FORWARD_OK', parameters=sum(p.numel() for p in model.parameters()),
                parameter_updates=0, old_information_exact=True, permutation_and_batch_checks=True)


def fit_arm(paired, out, arm, seed, initial_state, *, updates=UPDATES, wall_seconds=WALL_SECONDS):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = TrajectorySetActor()
    model.load_state_dict(initial_state)
    initial_hash = digest_state(model.state_dict())
    require(initial_hash == digest_state(initial_state), 'paired weights differ')
    data, tensors = paired.original, paired.tensors(arm)
    initial, _ = measure(model, tensors, data)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)
    permutation, offset = torch.randperm(len(data.gate), generator=generator), 0
    batch_hash = hashlib.sha256()
    for update in range(1, updates+1):
        require(time.monotonic()-started < wall_seconds, 'fit wall budget exhausted')
        if offset == len(data.gate):
            permutation, offset = torch.randperm(len(data.gate), generator=generator), 0
        indices = permutation[offset:offset+BATCH_SIZE]; offset += len(indices)
        batch_hash.update(indices.numpy().tobytes())
        model.train(); optimizer.zero_grad(set_to_none=True)
        gate, proposal = model(*(t[indices] for t in tensors))
        loss, _ = hierarchical_bc_loss(gate, proposal, data.gate[indices], data.proposal[indices])
        require(bool(torch.isfinite(loss)), 'nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 250 == 0:
            write(out/'progress.json', dict(update=update, loss=float(loss.detach()), elapsed_seconds=time.monotonic()-started))
    final, predictions = measure(model, tensors, data)
    changed = {prefix: any(not torch.equal(v, initial_state[k]) for k, v in model.state_dict().items()
                          if k.startswith(prefix)) for prefix in ('gate_encoder', 'proposal_encoder', 'gate_head', 'proposal_head')}
    require(all(changed.values()), 'an actor module did not update')
    torch.save(dict(model=model.state_dict(), observation_schema=VERSION, protocol=PROTOCOL,
                    arm=arm, seed=seed, updates=updates, diagnostic_only=True), out/'diagnostic_final.pt')
    write(out/'predictions.json', predictions)
    result = dict(status='FIT_COMPLETE', arm=arm, seed=seed, updates=updates,
        initial_state_hash=initial_hash, final_state_hash=digest_state(model.state_dict()),
        batch_order_hash=batch_hash.hexdigest(), changed=changed, initial=initial, final=final,
        elapsed_seconds=time.monotonic()-started)
    write(out/'result.json', result)
    return result


def synthetic():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(16, 90, generator=generator)
    gate = torch.arange(16) % 2
    proposal = torch.where(gate.bool(), torch.arange(16) % 25, -1)
    contexts = [dict(seed=17, interface=('o2m', 'm2i', 'i2f')[i % 3],
                    low_soc=False, fast_inner=False, busy_lz=False, after_retry=False) for i in range(16)]
    original = Data(x, gate, proposal, tuple(f'synthetic:{i}' for i in range(16)), contexts, [])
    ego = torch.cat((x, torch.randn(16, 15, generator=generator)), -1)
    traffic = torch.randn(16, 4, 50, generator=generator)
    mask = torch.arange(4)[None, :] < (torch.arange(16) % 5)[:, None]
    return PairedInputs(original, (ego, traffic, mask))


def run(root, output, mode):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    try:
        paired = synthetic() if mode == 'smoke' else load_paired(root)
        write(output/'protocol.json', dict(protocol=PROTOCOL, observation_schema=VERSION, arms=ARMS,
            model_seeds=MODEL_SEEDS, updates=UPDATES, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
            wall_seconds_per_arm=WALL_SECONDS, optimizer='Adam', gradient_clip=.5,
            loss='unweighted gate CE + conditional proposal CE on teacher RELEASE',
            synthetic_only=mode == 'smoke', I0_QUALIFIED=False, STUDENT_QUALIFIED=False, PPO_ALLOWED=False))
        write(output/'inputs.json', dict(decision_ids=paired.original.ids, episodes=paired.original.inputs,
            rows=len(paired.original.gate), RELEASE=int(paired.original.gate.sum()),
            baseline=baseline(paired.original)))
        checked = forward_check(paired)
        write(output/'forward.json', checked)
        results = []
        if mode != 'prepare':
            for seed in ((17,) if mode == 'smoke' else MODEL_SEEDS):
                torch.manual_seed(seed)
                initial_state = deepcopy(TrajectorySetActor().state_dict())
                pair = [fit_arm(paired, output/f'{seed}_{arm}', arm, seed, initial_state,
                               updates=2 if mode == 'smoke' else UPDATES) for arm in ARMS]
                require(pair[0]['initial_state_hash'] == pair[1]['initial_state_hash']
                        and pair[0]['batch_order_hash'] == pair[1]['batch_order_hash'], 'unpaired fit')
                results.extend(pair)
        result = dict(status='PREPARED' if mode == 'prepare' else 'FIT_COMPLETE', mode=mode,
            protocol=PROTOCOL, forward=checked, fits=results, environment_steps=0, teacher_queries=0,
            parameter_updates=0 if mode == 'prepare' else sum(r['updates'] for r in results))
        write(output/'result.json', result)
        return result
    except Exception as error:
        write(output/'result.json', dict(status='ERROR', mode=mode, error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare', 'smoke', 'fit'], required=True)
    args = parser.parse_args()
    result = run(args.root, args.output, args.mode)
    print(json.dumps(dict(status=result['status'], mode=result['mode'], parameter_updates=result['parameter_updates'])))


if __name__ == '__main__':
    main()
