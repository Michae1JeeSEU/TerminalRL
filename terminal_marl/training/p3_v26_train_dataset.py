"""Six whole neutral train cycles; no development-smoke or Q input route."""
import json

import numpy as np
import torch

from ..core.trajectory_context_observation import pad_trajectory_context
from .p3_v26_course_data import TRAIN_CASES,validated_case,decode_resource_row
from .p3_v26_resource_inputs import load_resource_train
from .p3_v25_set_learnability import PairedInputs
from .p3_v3_expert_dataset import DATA_SCHEMA
from .p3_v3_bc_learnability import Data,require


def load_independent_train(directories):
    require(len(directories)==len(TRAIN_CASES),'all three independent train cycles required')
    observations,raw,gates,proposals,ids,contexts,inputs=[],[],[],[],[],[],[]
    seen=set()
    for index,(directory,(seed,scenario)) in enumerate(zip(directories,TRAIN_CASES),1):
        path,report=validated_case(directory,index);count=0
        for line in path.read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            require(row['seed']==seed and row['context']['seed']==seed and
                row['decision_id']==f"{DATA_SCHEMA}:{seed}:{row['tick']}:{row['uid']}" and
                row['decision_id'] not in seen,'wrong or duplicate train identity')
            obs,gate,proposal=decode_resource_row(row)
            observations.append(obs);raw.append(row['original_observation']);gates.append(gate)
            proposals.append(proposal);ids.append(row['decision_id']);contexts.append(row['context'])
            seen.add(row['decision_id']);count+=1
        require(count==report['resource_reconstruction']['rows']==report['reconstruction']['rows']>0,
            'incomplete resource row set')
        inputs.append(dict(seed=seed,scenario=scenario,rows=count,partition='train',
            sha256=report['resource_reconstruction']['sha256']))
    raw=np.asarray(raw,dtype=np.float32)
    require(raw.shape==(len(ids),90) and np.isfinite(raw).all(),'invalid original raw observations')
    original=Data(torch.from_numpy(raw),torch.tensor(gates),torch.tensor(proposals),tuple(ids),contexts,inputs)
    return PairedInputs(original,tuple(torch.from_numpy(a) for a in pad_trajectory_context(observations)))


def combine_train(parts):
    require(bool(parts),'no train parts')
    ids=tuple(did for part in parts for did in part.original.ids)
    require(len(ids)==len(set(ids)),'duplicate training decisions')
    require(all(t['partition']=='train' for p in parts for t in p.original.inputs),'non-train input')
    require(all(p.new[0].shape==(len(p.original.ids),159) for p in parts),'wrong resource schema')
    width=max(p.new[1].shape[1] for p in parts)
    traffic=torch.zeros(len(ids),width,50);mask=torch.zeros(len(ids),width,dtype=torch.bool)
    offset=0
    for part in parts:
        n=len(part.original.ids);w=part.new[1].shape[1]
        traffic[offset:offset+n,:w]=part.new[1];mask[offset:offset+n,:w]=part.new[2];offset+=n
    original=Data(torch.cat([p.original.x for p in parts]),torch.cat([p.original.gate for p in parts]),
        torch.cat([p.original.proposal for p in parts]),ids,
        [c for p in parts for c in p.original.contexts],[i for p in parts for i in p.original.inputs])
    return PairedInputs(original,(torch.cat([p.new[0] for p in parts]),traffic,mask))


def load_six_cycle_train(original_directories,independent_directories):
    return combine_train([load_resource_train(original_directories),load_independent_train(independent_directories)])
