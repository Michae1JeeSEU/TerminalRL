import json
from types import SimpleNamespace

import pytest

from terminal_marl.training import p3_neighbor_witness_data as data


def test_selection_is_whole_cycle_stratified_and_bounded():
    rows=[dict(partition='train',actor_mask=True,context={'interface':interface},
        teacher_gate=gate,tick=t,uid='2',decision_id=f'{interface}:{gate}:{t}')
        for interface in ('o2m','m2i','i2f') for gate in (0,1) for t in range(20)]
    selected=data.stratified(list(reversed(rows)))
    assert len(selected)==24 and len({r['decision_id'] for r in selected})==24
    assert sorted({r['tick'] for r in selected})==[0,6,12,19]
    with pytest.raises(ValueError):data.stratified([dict(rows[0],partition='Q')])


def test_unknown_candidates_are_never_supervised_as_rejections():
    assert data.candidate_indices(dict(teacher_joint_index=13,teacher_gate=1,
        teacher_audit=[{'joint_index':13}]))==[13]
    assert data.candidate_indices(dict(teacher_joint_index=0,teacher_gate=0,
        teacher_audit=[{'joint_index':13},{'joint_index':1},{'joint_index':0}]))==[13,1]
    with pytest.raises(ValueError):data.candidate_indices(dict(teacher_joint_index=2,teacher_gate=1,
        teacher_audit=[{'joint_index':13}]))


def test_output_is_written_only_after_witness_and_preserves_failure(tmp_path,monkeypatch):
    row=dict(decision_id='train:1',context={'interface':'o2m'})
    monkeypatch.setattr(data,'selected_inputs',lambda _: ([(None,None,row,None)],[]))
    def witness(*args,**kw):
        assert (tmp_path/'ok/protocol.json').exists()
        return dict(decision_id='train:1',plan_calls=1,pairs=[{'blocked':None}])
    monkeypatch.setattr(data,'witness_state',witness)
    result=data.collect([],tmp_path/'ok')
    assert result['status']=='COMPLETE' and result['states']==1
    saved=json.loads((tmp_path/'ok/witnesses.jsonl').read_text())
    assert saved['pairs'][0]['blocked'] is None
    with pytest.raises(FileExistsError):data.collect([],tmp_path/'ok')
    def failed(*args,**kwargs):raise RuntimeError('native mismatch')
    monkeypatch.setattr(data,'witness_state',failed)
    result=data.collect([],tmp_path/'failed')
    assert result['status']=='INCOMPLETE' and 'native mismatch' in result['error']
    assert (tmp_path/'failed/protocol.json').exists()
