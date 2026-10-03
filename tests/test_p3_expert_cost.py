import json
import pickle

import pytest

from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.core.safety_trace import _digest
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts import diagnose_p3_expert_cost as cost


def test_reconstruction_checks_each_step_and_preserves_source(tmp_path):
    env=TerminalParallelEnv(build_config(requests=3,warm_start=3,high_soc=True),seed=892902)
    data=pickle.dumps(env)
    steps=[]
    for tick in range(2):
        pre=_digest(cost.canonical(cost.state_record(env)))
        tape=dict(actions={str(uid):[0.,0.,0.] for uid in env.active},shortlists={})
        live=cost.execute_tape(env,tape)
        steps.append(dict(tick=tick,pre_state_hash=pre,tape=tape,
            post_state_hash=_digest(live['after']),action_traces=live['outputs'][4]['action_traces']))
    reconstructed,times,_=cost.reconstruct(data,steps,1,tmp_path)
    assert reconstructed.step_count==1 and len(times)==1
    assert pickle.loads(data).step_count==0
    steps[0]['post_state_hash']='bad'
    with pytest.raises(ValueError,match='post-state differs'):
        cost.reconstruct(data,steps,1,tmp_path)


def test_real_profile_has_serializable_components_and_no_live_mutation(tmp_path):
    env=TerminalParallelEnv(build_config(requests=3,warm_start=3,high_soc=True),seed=892902)
    data=pickle.dumps(env)
    query=cost.profile_query(env,'0',tmp_path)
    assert query['joint_index']>0 and query['candidates']
    assert {x['component'] for x in query['components']}=={'ring_transfer','filter_plan'}
    assert query['profile_top'] and (tmp_path/'teacher.prof').exists()
    assert pickle.dumps(env)==data
    json.dumps(query,allow_nan=False)


def test_wrong_hash_rejected_before_pickle_load(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'first_before.pkl').write_bytes(b'not trusted pickle')
    report=cost.run(source,tmp_path/'output',smoke=True)
    assert report['status']=='ERROR'
    assert report['error']['message']=='input hash differs: first_before.pkl'
