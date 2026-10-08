"""Rebuild resource observations from verified original train pre-states only."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from ..core.resource_context_observation import ResourceContextCache
from .p3_v25_course_data import load_train, TRAIN_CASES
from .p3_v25_set_learnability import PairedInputs


def load_resource_train(directories):
    paired=load_train(directories)
    by_id={did:i for i,did in enumerate(paired.original.ids)}
    result=[None]*len(by_id)
    for directory,(seed,_) in zip(directories,TRAIN_CASES):
        episode=Path(directory)/f'seed_{seed}'
        cached_name=None
        for entry in json.loads((episode/'selected.json').read_text(encoding='utf8')):
            row=entry['row'];i=by_id[row['decision_id']]
            name=entry['snapshot']
            if name!=cached_name:
                blob=(episode/'snapshots'/name).read_bytes()
                if hashlib.sha256(blob).hexdigest()!=entry['snapshot_sha256']:
                    raise ValueError('train snapshot changed')
                env=pickle.loads(blob);cache=ResourceContextCache(env);cached_name=name
                cached_sha=entry['snapshot_sha256']
            if entry['snapshot_sha256']!=cached_sha:
                raise ValueError('inconsistent snapshot reference')
            obs=cache.build(env.active[int(row['uid'])])
            # Windows/Linux BLAS rotation can differ at roundoff (including
            # cancellation near zero). Validate numerically but retain the
            # archived prefix/traffic bytes, not a recomputed replacement.
            if (not np.allclose(obs.ego[:105],paired.new[0][i].numpy(),rtol=1e-6,atol=1e-7) or
                not np.allclose(obs.traffic,paired.new[1][i][paired.new[2][i]].numpy(),rtol=1e-6,atol=1e-7)):
                ego_diff=np.flatnonzero(obs.ego[:105]!=paired.new[0][i].numpy()).tolist()
                previous=paired.new[1][i][paired.new[2][i]].numpy()
                raise ValueError(f'original observation changed at {row["decision_id"]}: '
                    f'ego indices={ego_diff}; traffic shapes={obs.traffic.shape}/{previous.shape}; '
                    f'max ego difference={np.max(np.abs(obs.ego[:105]-paired.new[0][i].numpy()))}')
            if result[i] is not None:raise ValueError('duplicate train decision')
            result[i]=np.concatenate((paired.new[0][i].numpy(),obs.ego[105:]))
    if any(v is None for v in result):raise ValueError('missing train decision')
    return PairedInputs(paired.original,(torch.from_numpy(np.stack(result)),*paired.new[1:]))
