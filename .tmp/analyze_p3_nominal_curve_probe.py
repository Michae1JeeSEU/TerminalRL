"""Use the already checked record assertions on the explicit diagnostic migration."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from terminal_marl.configs.p3_v25_course import build_config,PROFILE
from terminal_marl.training.p3_v25_course_data import task_settings
from terminal_marl.training.p3_v25_rollout import load_steps
from terminal_marl.training.p3_v25_task_student import blockers
p=argparse.ArgumentParser();p.add_argument('case',choices=('smoke','source_dense_fixed'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output;assert not out.exists();smoke=a.case=='smoke'
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_nominal_curve_validation_20261008')/a.case
ep=root/'episode';read=lambda path:json.loads(path.read_text(encoding='utf8'))
protocol=read(root/'protocol.json');report=read(root/'result.json');runtime=read(ep/'protocol.json');result=read(ep/'result.json')
assert report['status']==('SMOKE_OK' if smoke else 'COUNTERFACTUAL_COMPLETE')
assert result==report['rollout'] and not report['blockers'] and not blockers(result,smoke=smoke)
assert protocol['profile']==PROFILE and protocol['source_training_profile']=='p3_v25_neutral_course_committed_merge_20261008_v1'
assert runtime['seed']==protocol['seed']==(892862 if smoke else 892854)
assert runtime['policy_seed']==protocol['policy_seed']==(929862 if smoke else 929854)
assert not runtime['teacher_enabled'] and not runtime['deterministic']
source=dict(sha256='5ca0c9dc229b7cf750e3d55db611d68fad690093a7da5e9e7d1ab7ea98ad7883',seed=91901)
assert protocol['source_sha256']==report['source_sha256']==source['sha256']
config=build_config('dense24_fixed')
if smoke:config=replace(config,total_requested_uavs=3,warm_start_uavs=3,fixed_arrival_schedule=(0.,)*3)
assert {k:runtime['physics_config'][k] for k in task_settings(config)}==task_settings(config)
body=Path('.tmp/analyze_p3_v25_student.py').read_text(encoding='utf8')
exec(compile(body[body.index('steps=load_steps'):],__file__,'exec'))
