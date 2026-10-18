"""Real worker lifecycle tests; Linux evidence requires actual Linux execution."""
from hashlib import sha256
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest

from terminal_marl.core.rotating_slots.performance.provenance import R0_COMMIT, R0_SOURCE_SHA256, run_reference

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'artifacts/rotating_slots/performance_reference_r0'

PRELUDE=r'''
import sys
sys.dont_write_bytecode=True
import hashlib,json,os,pathlib,time
root=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2])
sys.path.insert(0,str(root))
from terminal_marl.core.rotating_slots import energy,trajectory,safety
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.performance import parallel
core=root/'terminal_marl/core/rotating_slots'
config=load_config()
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in core.glob('*.py')}
semantic={'policy_id':parallel.POLICY_ID,'effective_rules_hash':'test-rules',
          'candidate_policy_hash':'test-candidates','numerical_policy_hash':'test-numerics'}
def pool(workers):
    return parallel.ReferencePool(workers,config=config,expected_core_root=core,
        expected_core_manifest=manifest,semantic_identity=semantic,
        process_budget=workers+1,core_budget=workers,exclusive_process_accounting=True,kernel='OPT')
def descendants(pid):
    result=set(); pending=[pid]
    while pending:
        current=pending.pop()
        path=pathlib.Path('/proc')/str(current)/'task'/str(current)/'children'
        for child in map(int,path.read_text().split()):
            if child not in result: result.add(child); pending.append(child)
    return result
def write(value):
    output.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf8')
'''


def run_child(code,output,*extra,timeout=180):
    command=[sys.executable,'-I','-B','-c',PRELUDE+code,str(ROOT),str(output),*map(str,extra)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
    output.with_suffix('.stdout.log').write_text(result.stdout,encoding='utf8')
    output.with_suffix('.stderr.log').write_text(result.stderr,encoding='utf8')
    output.with_suffix('.process.json').write_text(json.dumps({'command':command,'returncode':result.returncode},indent=2)+'\n',encoding='utf8')
    assert result.returncode==0,result.stderr
    return json.loads(output.read_text(encoding='utf8'))


@pytest.mark.parametrize('workers',[1,2,4,6])
def test_real_startup_all_workers_and_inherited_caches(workers,tmp_path):
    code=r'''
workers=int(sys.argv[3])
trajectory.poly_bounds((1.,2.),0.,1.)
stable=trajectory.make_stable_plan('warm','inner',0,0.,1.,config=config)
assert energy._integrable_plan(stable).accepted
energy._segment_table(1.,(100.,),(0.,),(15.,),.2,config.profile_hash)
assert energy.evaluate_power((0.,0.,0.),(0.,0.,0.),config).valid
before=parallel._worker_cache_snapshot()
mandatory=('trajectory.poly_bounds','energy._integrable_plan','energy._segment_table',
           'safety._NUMERIC_CERTIFICATES','safety._PLAN_CERTIFICATES')
assert all(before[name].get('currsize',before[name].get('entries'))>0 for name in mandatory)
if before['energy.root_cache']['present']: assert before['energy.root_cache']['entries']>0
if before['trajectory._BERNSTEIN_METADATA']['present'] and before['trajectory._BERNSTEIN_METADATA'].get('enabled',True):
    assert before['trajectory._BERNSTEIN_METADATA']['entries']>0
before_children=descendants(os.getpid()) if sys.platform=='linux' else None
assert before_children in (None,set())
with pool(workers) as executor:
    summary=executor.summary()
    observed=summary['observed_worker_startups']
    registered=set(summary['worker_pids'])
    assert len(observed)==workers and {r['pid'] for r in observed}==registered
    assert summary['worker_startup_coverage_complete'] is True
    for row in observed:
        assert row['expected_parent_pid']==os.getpid() and row['actual_parent_pid']==os.getpid()
        assert row['dont_write_bytecode'] is True
        assert row['isolated'] is True
        assert row['canonical_phase_bins']==24
        assert all(v.get('currsize',v.get('entries'))==0 for v in row['cache_reset']['after'].values())
        for key,old in row['cache_reset']['before'].items():
            for flag in ('enabled','policy','max_entries','max_bytes'):
                if flag in old: assert row['cache_reset']['after'][key][flag]==old[flag]
        if sys.platform=='linux':
            assert row['parent_death']['status']=='BOUND_TO_EXPECTED_PARENT'
            assert all(row['cache_reset']['before'][name].get('currsize',row['cache_reset']['before'][name].get('entries'))>0 for name in mandatory)
        else: assert row['parent_death']['status']=='UNSUPPORTED_PLATFORM'
    actual=descendants(os.getpid()) if sys.platform=='linux' else None
    if actual is not None:
        assert actual==registered
        assert 1+len(actual)==1+workers
        assert not any(b'resource_tracker' in pathlib.Path('/proc',str(pid),'cmdline').read_bytes() for pid in actual)
    assert parallel._worker_cache_snapshot()==before
    running={'registered_worker_pids':sorted(registered),'actual_descendants':None if actual is None else sorted(actual),
             'strict_one_plus_workers_proved':sys.platform=='linux','start_method':executor.start_method}
after=parallel._worker_cache_snapshot()
assert after==before
if sys.platform=='linux': assert descendants(os.getpid())==set()
write({'workers':workers,'parent_cache_before':before,'parent_cache_after':after,'running':running,
       'summary':executor.summary(),'linux_process_proof':'OBSERVED' if sys.platform=='linux' else 'SKIPPED_UNSUPPORTED_PLATFORM'})
'''
    row=run_child(code,tmp_path/'startup.json',workers)
    assert len(row['summary']['observed_worker_startups'])==workers
    assert row['summary']['closed_and_joined'] is True
    assert row['parent_cache_before']==row['parent_cache_after']


def test_reconstruction_matches_independent_r0_and_retains_core_bytes(tmp_path):
    expected=run_reference(REFERENCE,code=r'''
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
value=RotatingSlotWorld()._reference('inner',0,.125,0)
query_result={'soc_hex':None if value[0] is None else value[0].hex(),
              'plan':None if value[1] is None else value[1].to_dict()}
''',timeout=180)
    assert expected['reference_commit']==R0_COMMIT and expected['source_sha256']==R0_SOURCE_SHA256
    oracle=tmp_path/'oracle.json'; oracle.write_text(json.dumps(expected,indent=2)+'\n',encoding='utf8')
    # This driver selects frozen R0 before importing its core. Current helper
    # modules are appended to that package, as in the existing real R0 test.
    code=r'''
import sys
sys.dont_write_bytecode=True
import hashlib,json,pathlib
reference,helper,output=map(pathlib.Path,sys.argv[1:4])
sys.path.insert(0,str(reference))
import terminal_marl.core.rotating_slots as package
package.__path__.append(str(helper))
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.performance.parallel import ReferencePool,install_parallel_reference,POLICY_ID
core=reference/'terminal_marl/core/rotating_slots'
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in core.glob('*.py')}
semantic={'policy_id':POLICY_ID,'effective_rules_hash':'test-rules','candidate_policy_hash':'test-candidates','numerical_policy_hash':'test-numerics'}
rows=[]
for _ in range(2):
    world=RotatingSlotWorld(); config=load_config()
    with ReferencePool(2,config=config,expected_core_root=core,expected_core_manifest=manifest,
        semantic_identity=semantic,process_budget=3,core_budget=2,exclusive_process_accounting=True) as executor:
        with install_parallel_reference(world,executor): value=world._reference('inner',0,.125,0)
    rows.append({'soc_hex':None if value[0] is None else value[0].hex(),
                 'plan':None if value[1] is None else value[1].to_dict(),
                 'summary':executor.summary()})
assert manifest=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in core.glob('*.py')}
output.write_text(json.dumps(rows,sort_keys=True,indent=2)+'\n',encoding='utf8')
'''
    output=tmp_path/'reconstruction.json'
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(REFERENCE),
        str(ROOT/'terminal_marl/core/rotating_slots'),str(output)],capture_output=True,text=True,timeout=240)
    (tmp_path/'reconstruction.stderr.log').write_text(result.stderr,encoding='utf8')
    assert result.returncode==0,result.stderr
    rows=json.loads(output.read_text())
    assert len(rows)==2
    for row in rows:
        assert {k:row[k] for k in ('soc_hex','plan')}==expected['query_result']
        assert row['summary']['closed_and_joined'] and row['summary']['worker_startup_coverage_complete']


@pytest.mark.skipif(sys.platform!='linux',reason='real PDEATHSIG/process inventory requires Linux')
def test_real_owner_death_exits_exact_workers_unrelated_sentinel_survives(tmp_path):
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        pytest.skip('identity-safe owned orphan cleanup requires Linux pidfd support')
    ready=tmp_path/'owner_ready.json'
    code=PRELUDE+r'''
executor=pool(2)
summary=executor.summary(); owned=set(summary['worker_pids'])
assert descendants(os.getpid())==owned
assert {r['pid'] for r in summary['observed_worker_startups']}==owned
write({'owner_pid':os.getpid(),'worker_pids':sorted(owned),'summary':summary})
while True: time.sleep(.05)
'''
    command=[sys.executable,'-I','-B','-c',code,str(ROOT),str(ready)]
    sentinel=subprocess.Popen([sys.executable,'-I','-B','-c','import sys,time;sys.dont_write_bytecode=True;time.sleep(60)'],
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    owner=None; descriptors={}; observations={}
    try:
        owner=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        limit=time.monotonic()+40
        while not ready.exists() and owner.poll() is None and time.monotonic()<limit: time.sleep(.02)
        assert ready.exists(),owner.communicate(timeout=2) if owner.poll() is not None else 'owner startup did not complete'
        row=json.loads(ready.read_text()); assert row['owner_pid']==owner.pid
        descriptors={pid:os.pidfd_open(pid) for pid in row['worker_pids']}
        assert len(descriptors)==2 and sentinel.poll() is None
        owner.kill(); stdout,stderr=owner.communicate(timeout=10)
        assert owner.returncode==-signal.SIGKILL,(stdout,stderr)
        limit=time.monotonic()+5
        while time.monotonic()<limit:
            exited=[pid for pid,fd in descriptors.items() if select.select([fd],[],[],0)[0]]
            if len(exited)==len(descriptors): break
            time.sleep(.02)
        assert len(exited)==len(descriptors)
        assert sentinel.poll() is None
        observations={'owner_pid':owner.pid,'owner_returncode':owner.returncode,
            'owned_worker_pids':sorted(descriptors),'kernel_exit_observed_by_pidfd':sorted(exited),
            'unrelated_sentinel_pid':sentinel.pid,'unrelated_sentinel_survived':True,
            'strict_parent_plus_workers_at_readiness':True,'linux_kernel_test':'OBSERVED',
            'pidfd_proves_exit_not_reaping_by_this_test':True}
        (tmp_path/'parent_death.json').write_text(json.dumps(observations,indent=2)+'\n',encoding='utf8')
    finally:
        for fd in descriptors.values():
            try: signal.pidfd_send_signal(fd,signal.SIGKILL)
            except ProcessLookupError: pass
            os.close(fd)
        for process in (owner,sentinel):
            if process is not None and process.poll() is None: process.kill()
            if process is not None: process.communicate(timeout=10)


def test_deadline_helper_identity_and_source_before_after(tmp_path):
    code=r'''
identity=parallel._implementation_identity()
assert set(identity)=={'parallel.py','process_metrics.py','deadline.py'}
helper=pathlib.Path(parallel.__file__).parent
before={name:hashlib.sha256((helper/name).read_bytes()).hexdigest() for name in identity}
with pool(1) as executor:
    assert executor.summary()['identity']['implementation_files_sha256']==identity
assert before=={name:hashlib.sha256((helper/name).read_bytes()).hexdigest() for name in identity}
assert manifest=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in core.glob('*.py')}
write({'helper_identity':identity,'helper_source_before_after_equal':True,'core_source_before_after_equal':True,
       'startup':executor.summary()['observed_worker_startups']})
'''
    row=run_child(code,tmp_path/'source_identity.json')
    assert row['helper_source_before_after_equal'] and row['core_source_before_after_equal']
