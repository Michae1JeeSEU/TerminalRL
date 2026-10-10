from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.training import p3_neighbor_witness_data as data


def _label():
    return dict(
        pre_state_hash='state-hash',
        teacher_gate=1,
        teacher_joint_index=12,
        teacher_audit=[
            dict(joint_index=13, feasible=False, reason='nominal_rejected'),
            dict(joint_index=12, feasible=True, reason='certified'),
        ],
    )


def test_requested_joints_preserve_original_teacher_audit_order_and_default_route():
    label=_label()
    assert data.candidate_indices(label)==[13,12]
    assert data.candidate_indices(label,requested_joints=(12,13))==[13,12]
    assert data.candidate_indices(label,requested_joints=[12,13])==[13,12]


def test_wait_default_route_keeps_original_prefix_semantics():
    label=dict(
        teacher_gate=0,
        teacher_joint_index=0,
        teacher_audit=[
            *[dict(joint_index=index, feasible=False, reason='nominal_rejected') for index in range(1,26)],
            dict(joint_index=0, feasible=True, reason='certified'),
        ],
    )
    assert data.candidate_indices(label)==[13,1]


@pytest.mark.parametrize('requested',(
    [],
    [True],
    [12,12],
    [0],
    [26],
    [11],
))
def test_requested_joints_reject_invalid_or_unknown_suffix_before_planner(tmp_path,monkeypatch,requested):
    label=_label()
    with pytest.raises(ValueError):data.candidate_indices(label,requested_joints=requested)
    planner_calls=[]
    monkeypatch.setattr(data,'pickle',SimpleNamespace(loads=lambda payload: planner_calls.append(payload)))
    with pytest.raises(ValueError):
        data.witness_state(tmp_path,{}, {}, label, remaining_calls=1,requested_joints=requested)
    assert planner_calls==[]


def test_witness_state_traverses_only_requested_prefix_subset(tmp_path,monkeypatch):
    label=_label();planner_calls=[]
    class Env:
        now=0.
        airspace=object()
        config=SimpleNamespace(task_priority_enabled=False)
        def __init__(self):
            self.active={1:SimpleNamespace(uid=1,soc=1.)}
        def _candidate_plan(self,*args):
            planner_calls.append(args[1])
            return object(),SimpleNamespace(trajectory_hash='plan-hash',start_time=0.,end_time=1.),None
    env=Env()
    monkeypatch.setattr(data,'sha',lambda path:'snapshot-sha')
    monkeypatch.setattr(data.pickle,'loads',lambda payload:env)
    monkeypatch.setattr(data,'canonical',lambda value:value)
    monkeypatch.setattr(data,'state_record',lambda value:{})
    monkeypatch.setattr(data,'_digest',lambda value:'state-hash')
    monkeypatch.setattr(data,'state_delta',lambda before,after:{})
    monkeypatch.setattr(data,'_release_context',lambda env,ego:('middle','o2m'))
    monkeypatch.setattr(data,'global_block_reason',lambda *args:None)
    monkeypatch.setattr(data,'WorldSnapshot',lambda *args:None)
    monkeypatch.setattr(data,'pair_check',lambda *args:{'approved':True,'blocked':False})
    observation=SimpleNamespace(ego=np.zeros(105),traffic=np.zeros((0,50)),traffic_uids=[])
    monkeypatch.setattr(data,'build_low_observation_v2_6',lambda ego,env:observation)
    (tmp_path/'snapshots').mkdir()
    snapshot=tmp_path/'snapshots'/'snapshot.pkl';snapshot.write_bytes(b'fixture')
    row=dict(pre_state_hash='state-hash',decision_id='d',seed=1,tick=0,uid=1,
        ego=[0.]*105,traffic=[],traffic_mask=[],context={'interface':'o2m'})
    entry=dict(snapshot=snapshot.name,snapshot_sha256='snapshot-sha')
    result=data.witness_state(tmp_path,entry,row,label,remaining_calls=2,requested_joints=(12,))
    assert [item['joint'] for item in result['candidates']]==[12]
    assert result['plan_calls']==1 and len(planner_calls)==1
