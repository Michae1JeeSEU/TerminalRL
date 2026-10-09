"""Join offline pair supervision by decision ID; keep whole cycles isolated."""
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .p3_v25_imitation_qualification import sha
from .p3_v3_bc_learnability import read, require
from .p3_neighbor_witness_data import PROTOCOL, SUPPLEMENTAL_PROTOCOL, TRAFFIC_REASONS
from . import p3_v25_course_data as old
from . import p3_v26_course_data as independent
from . import p3_v26_supplemental_data as supplemental


@dataclass
class WitnessTargets:
    state_index: torch.Tensor
    proposal: torch.Tensor
    pad: torch.Tensor
    blocked: torch.Tensor
    known: torch.Tensor
    scene_blocked: torch.Tensor
    scene_known: torch.Tensor
    seeds: torch.Tensor

    def take(self,indices):
        return WitnessTargets(*(getattr(self,key)[indices] for key in self.__dataclass_fields__))


def load_witness_targets(directory,paired):
    directory=Path(directory);protocol=read(directory/'protocol.json');report=read(directory/'result.json')
    protocol_name=protocol['protocol']
    require(protocol_name in (PROTOCOL,SUPPLEMENTAL_PROTOCOL) and not protocol['smoke'] and report['status']=='COMPLETE'
        and sha(directory/'witnesses.jsonl')==report['witness_sha256'],'incomplete or wrong witness data')
    expected=(set(seed for seed,_ in old.TRAIN_CASES)|set(seed for seed,_ in independent.TRAIN_CASES)
        if protocol_name==PROTOCOL else set(seed for seed,_ in supplemental.TRAIN_CASES))
    declared={item['seed'] for item in protocol['inputs']}
    require(declared==expected,'witness source seed set differs')
    rows=[json.loads(s) for s in (directory/'witnesses.jsonl').read_text().splitlines()]
    require(len(rows)==report['states'] and [r['decision_id'] for r in rows]==protocol['selected_ids'],'witness row set differs')
    require({row['seed'] for row in rows}<=expected,'witness seed outside declared source group')
    index={did:i for i,did in enumerate(paired.original.ids)};width=paired.new[1].shape[1]
    values=[]
    for row in rows:
        did=row['decision_id'];require(did in index,'witness outside train data')
        i=index[did];n=len(row['traffic_uids'])
        require(row['seed']==paired.original.contexts[i]['seed'] and
            torch.allclose(torch.tensor(row['ego']),paired.new[0][i],rtol=1e-6,atol=1e-7) and
            torch.allclose(torch.tensor(row['traffic']).reshape(-1,50),paired.new[1][i,:n],rtol=1e-6,atol=1e-7)
            and n==int(paired.new[2][i].sum()),'witness numeric input mismatch')
        require(row['teacher_gate']==int(paired.original.gate[i]) and
            row['teacher_joint']==(int(paired.original.proposal[i])+1 if row['teacher_gate'] else 0), 'original label mismatch')
        for candidate in row['candidates']:
            for dest in candidate['destinations']:
                blocked=torch.zeros(width);known=torch.zeros(width,dtype=torch.bool)
                pairs=dest['pairs']
                if pairs:
                    require([p['uid'] for p in pairs]==row['traffic_uids'],'pair order differs')
                    for j,pair in enumerate(pairs):
                        if pair['blocked'] is not None:known[j]=True;blocked[j]=float(pair['blocked'])
                reason=dest['reason']
                scene_known=dest['feasible'] or not reason.startswith(TRAFFIC_REASONS)
                values.append((i,candidate['joint']-1,0 if dest['pad'] is None else dest['pad'],
                    blocked,known,float(not dest['feasible']),scene_known,row['seed']))
    require(bool(values),'no native witness targets')
    return WitnessTargets(*(torch.stack([v[k] for v in values]) if k in (3,4) else
        torch.tensor([v[k] for v in values]) for k in range(8)))


def combine_witness_targets(parts):
    require(bool(parts),'no witness target parts')
    require(all(isinstance(part,WitnessTargets) for part in parts),'invalid witness target part')
    seen=set()
    for part in parts:
        require(len(part.state_index)==len(part.proposal)==len(part.pad)==len(part.blocked)==len(part.known)
            ==len(part.scene_blocked)==len(part.scene_known)==len(part.seeds),'witness target lengths differ')
        for key in zip(part.state_index.tolist(),part.proposal.tolist(),part.pad.tolist()):
            require(tuple(key) not in seen,'duplicate witness target')
            seen.add(tuple(key))
    return WitnessTargets(*(torch.cat([getattr(part,key) for part in parts])
        for key in WitnessTargets.__dataclass_fields__))


def auxiliary_loss(pair_logits,scene_logits,target):
    # Balance native blocked/safe pairs; null outcomes never acquire a target.
    losses=[]
    for positive in (False,True):
        selected=target.known & (target.blocked.bool()==positive)
        if selected.any():losses.append(F.binary_cross_entropy_with_logits(pair_logits[selected],target.blocked[selected]))
    pair_loss=torch.stack(losses).mean() if losses else pair_logits.sum()*0.
    scene_loss=(F.binary_cross_entropy_with_logits(scene_logits[target.scene_known],target.scene_blocked[target.scene_known])
        if target.scene_known.any() else scene_logits.sum()*0.)
    return pair_loss+.25*scene_loss
