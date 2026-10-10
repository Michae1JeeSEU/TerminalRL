"""Full saved-record checks for the primary's neutral student review; no rollout."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from terminal_marl.configs.p3_v25_course import PROFILE,SCENARIOS,build_config
from terminal_marl.training.p3_v25_course_data import task_settings
from terminal_marl.training.p3_v25_rollout import load_steps
from terminal_marl.training.p3_v25_task_imitation import frozen_models
from terminal_marl.training.p3_v25_task_student import PROTOCOL,blockers

p=argparse.ArgumentParser()
p.add_argument('case',choices=['smoke',*SCENARIOS])
p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);assert not out.exists()
base=Path('D:/7_codespace/training_outputs/terminal/p3_v25_neutral_course_20261008')
root=base/'student_initial'/a.case
read=lambda p:json.loads(p.read_text(encoding='utf8'))
protocol=read(root/'protocol.json');report=read(root/'result.json')
ep=root/'episode';runtime=read(ep/'protocol.json');result=read(ep/'result.json')
smoke=a.case=='smoke';scenario='dense24_mixed' if smoke else a.case
index=0 if smoke else SCENARIOS.index(scenario)+1
assert protocol['protocol']==report['protocol']==PROTOCOL
assert protocol['profile']==report['profile']==PROFILE
assert protocol['seed']==runtime['seed']==892852+index
assert protocol['policy_seed']==runtime['policy_seed']==929852+index
assert protocol['scenario']==report['scenario']==scenario
assert protocol['smoke'] is smoke and not protocol['deterministic']
assert not protocol['teacher_enabled'] and not runtime['teacher_enabled']
assert protocol['parameter_updates']==report['parameter_updates']==0
manifest=frozen_models(base/'fit_initial');source=manifest['checkpoints'][0]
assert protocol['source']==report['source']==source
assert source['seed']==91901 and source['sha256']=='5ca0c9dc229b7cf750e3d55db611d68fad690093a7da5e9e7d1ab7ea98ad7883'
assert report['rollout']==result and report['status']==('SMOKE_OK' if smoke else 'DEVELOPMENT_COMPLETE')
assert not report['blockers'] and not blockers(result,smoke=smoke)
config=build_config(scenario)
if smoke:config=replace(config,total_requested_uavs=3,warm_start_uavs=3,fixed_arrival_schedule=(0.,)*3)
assert {key:runtime['physics_config'][key] for key in task_settings(config)}==task_settings(config)
steps=load_steps(ep/'steps.pt')
requests=[json.loads(line) for line in (ep/'requests.jsonl').read_text(encoding='utf8').splitlines()]
assert len(steps)==len(requests)==result['ticks'] and len(steps)>0
assert sum(row.reward for row in steps)==result['reward']
actors=releases=shield=0
for tick,(row,record) in enumerate(zip(steps,requests)):
    assert record['tick']==tick and record['reward']==row.reward
    assert not row.teacher_used and row.stochastic_behavior
    uids=sorted(record['actor_mask'],key=int)
    assert np.array_equal(row.actor_mask,[record['actor_mask'][uid] for uid in uids])
    assert len(uids)==len(row.ego)
    for i,uid in enumerate(uids):
        action=record['tape']['actions'][uid]
        trace=record['action_traces'][uid]
        assert not trace['teacher_used'] and trace['requested_low_action']==action
        assert bool(action[0])==bool(row.requested_gate[i])
        if row.requested_gate[i]:
            proposal=int(row.requested_proposal[i]);grid=(-1.,-.5,0.,.5,1.)
            assert action==[1.,grid[proposal//5],grid[proposal%5]]
        else:assert action==[0.,0.,0.]
        shield+=int(row.actor_mask[i] and trace['shield_used'])
    actors+=int(row.actor_mask.sum());releases+=int(row.requested_gate.sum())
assert result['counts']['actor']==actors and result['counts']['RELEASE']==releases
assert result['counts']['shield_actor']==shield
assert result['shield_actor_fraction']==shield/max(1,actors)
assert all(not row.episode_end for row in steps[:-1]) and steps[-1].episode_end
assert steps[-1].terminated is not smoke
assert abs(sum(result['reward_components'].values())-result['reward'])<=1e-7
assert all(item['energy_remaining']>=0 and item['soc']>=0 for item in result['energy_outcomes'])
summary=dict(status='FULL_STUDENT_ANALYSIS_COMPLETE',case=a.case,source=source,
    ticks=len(steps),actors=actors,release=releases,wait=actors-releases,
    outcomes=result['outcome_counts'],shield=shield,telemetry=result['telemetry'],
    task_metrics=result['task_metrics'],reward=result['reward'],reward_components=result['reward_components'],
    terminal_soc_min=min((item['soc'] for item in result['energy_outcomes']),default=None),
    result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
    elapsed_seconds=result['elapsed_seconds'],replay_exact=result['replay_exact'])
out.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=summary['status'],case=a.case,ticks=len(steps),actors=actors,output=str(out))))
