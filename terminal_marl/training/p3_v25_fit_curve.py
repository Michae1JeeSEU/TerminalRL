"""Fixed 5000-update new-information curve; only final output is a candidate."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from ..algo.trajectory_set_actor import TrajectorySetActor
from .p3_v25_set_learnability import (load_paired, synthetic, forward_check, fit_arm,
    MODEL_SEEDS, write, require)

PROTOCOL = 'p3_v25_fixed_fit_curve_20261008'


def run(root, output, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    try:
        paired = synthetic() if smoke else load_paired(root)
        reference = None if smoke else json.loads((root/'p3_v25_set_fit_20261008/fit/result.json').read_text())
        if reference is not None:
            require(reference['status'] == 'FIT_COMPLETE', 'reference incomplete')
            original_ids = json.loads((root/'p3_v25_set_fit_20261008/fit/inputs.json').read_text())['decision_ids']
            require(original_ids == list(paired.original.ids), 'reference IDs differ')
        write(output/'protocol.json', dict(protocol=PROTOCOL, smoke=smoke,
            seeds=[17] if smoke else MODEL_SEEDS, updates=3 if smoke else 5000,
            measurement_updates=[2,3] if smoke else [1000,2500,5000],
            data_rows=len(paired.original.ids), loss='unchanged unweighted hierarchical BC',
            wall_seconds_per_seed=300, checkpoint_selection='fixed_final', PPO_ALLOWED=False))
        write(output/'forward.json', forward_check(paired))
        fits=[]
        for seed in ([17] if smoke else MODEL_SEEDS):
            torch.manual_seed(seed)
            initial = deepcopy(TrajectorySetActor().state_dict())
            if smoke:
                prefix = fit_arm(paired, output/'smoke_prefix', 'new_information', seed, initial, updates=2)
                prefix = json.loads(json.dumps(prefix))
            else:
                prefix = next(f for f in reference['fits'] if f['seed']==seed and f['arm']=='new_information')
            result = fit_arm(paired, output/f'seed_{seed}', 'new_information', seed, initial,
                updates=3 if smoke else 5000, measurement_updates=(2,3) if smoke else (1000,2500,5000),
                reference_prefix=prefix, protocol=PROTOCOL)
            require(result['curve'][0]['reference_prefix_exact'], 'missing paired prefix check')
            fits.append(result)
            write(output/'progress.json', dict(completed_seeds=len(fits), seed=seed, updates=result['updates']))
        result = dict(status='CURVE_COMPLETE', protocol=PROTOCOL, smoke=smoke, fits=fits,
                      environment_steps=0, teacher_queries=0)
        write(output/'result.json', result)
        return result
    except Exception as error:
        write(output/'result.json', dict(status='ERROR', error=repr(error)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    result=run(a.root,a.output,a.smoke)
    print(json.dumps(dict(status=result['status'], smoke=a.smoke)))


if __name__=='__main__': main()
