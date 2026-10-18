"""One frozen Linux joint pytest invocation; no remote launch at import time."""
import sys
sys.dont_write_bytecode=True
import argparse
import ast
import base64
import contextlib
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
import traceback

PROJECT=Path('/home/user/projects/terminal')
OUTPUT=Path('/home/user/training_outputs/terminal/rotating_slots_perf_v1/final_pass_linux_tests_20261019a')
DRIVER='artifacts/rotating_slots/final_pass_linux_tests_20261019a/run_tests.py'
LAUNCHER='artifacts/rotating_slots/final_pass_linux_tests_20261019a/run_linux_tests.ps1'
TESTS=tuple('tests/test_rotating_slots_performance_'+name+'.py' for name in (
    'bernstein_precompute','root_reuse','peak_reuse','energy_queries','p2',
    'cartesian_component','execution_reuse','windows','paired','parallel',
    'worker_lifecycle','deadline','runner'))+('tests/test_rotating_slots_physics.py',)
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT')

def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path):
    result=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): result.update(block)
    return result.hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())

def verify_request(encoded,root):
    request=json.loads(base64.b64decode(encoded,validate=True).decode('utf8'))
    assert request['schema']=='final_pass_linux_tests_source_v1'
    assert re.fullmatch('[0-9a-f]{40}',request['implementation_commit'])
    assert request['tests']==list(TESTS) and request['driver_relative']==DRIVER
    assert {DRIVER,LAUNCHER,*TESTS}<=request['files'].keys()
    actual={}
    for relative,expected in request['files'].items():
        path=(root/relative).resolve()
        assert not Path(relative).is_absolute() and path.is_relative_to(root)
        assert re.fullmatch('[0-9a-f]{64}',expected)
        actual[relative]=sha(path)
    assert actual==request['files'],'local/server source bytes differ'
    assert actual[DRIVER]==request['driver_sha256']
    return request,actual

def environment():
    result={'utc':utc(),'pid':os.getpid(),'ppid':os.getppid(),'platform':sys.platform,
        'load_average':list(os.getloadavg()),'allowed_logical_cpus':sorted(os.sched_getaffinity(0)),
        'nice':os.getpriority(os.PRIO_PROCESS,0),'uname':list(os.uname())}
    for name in ('loadavg','meminfo'):
        try: result['proc_'+name]=Path('/proc',name).read_text()
        except OSError as error: result['proc_'+name+'_error']=repr(error)
    return result

@contextlib.contextmanager
def redirect_logs(stdout,stderr):
    # Capture Python and native/fork-child FD output without a supervising process.
    saved=[os.dup(1),os.dup(2)]
    try:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(stdout.fileno(),1); os.dup2(stderr.fileno(),2)
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr): yield
    finally:
        stdout.flush(); stderr.flush()
        os.dup2(saved[0],1); os.dup2(saved[1],2)
        for fd in saved: os.close(fd)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request-base64',required=True)
    parser.add_argument('--local-check',action='store_true')
    args=parser.parse_args()
    if args.local_check:
        root=Path(__file__).resolve().parents[3]
        request,actual=verify_request(args.request_base64,root)
        ast.parse(Path(__file__).read_text(encoding='utf8'))
        assert len(TESTS)==14 and all((root/path).is_file() for path in TESTS)
        print(json.dumps({'status':'LOCAL_STATIC_CHECK_PASSED','source_hash_count':len(actual),
            'test_file_count':len(TESTS),'driver_sha256':request['driver_sha256'],
            'implementation_commit':request['implementation_commit'],'pytest_executed':False,
            'server_executed':False,'output_created':False})); return 0
    assert sys.platform=='linux' and sys.flags.isolated and sys.dont_write_bytecode
    assert not OUTPUT.exists(),'refusing existing final-pass output'
    OUTPUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); code=2; pytest_exit=None; preflight=False; failure=None
    metadata={'schema':'final_pass_linux_tests_run_v1','started_utc':utc(),
        'original_python_argv':list(sys.orig_argv),'driver_argv':list(sys.argv),
        'project':str(PROJECT),'output':str(OUTPUT),'tests':list(TESTS),
        'collection_count_source':'actual pytest collection hook; no fixed case-count target',
        'isolated':bool(sys.flags.isolated),'dont_write_bytecode':sys.dont_write_bytecode,
        'pytest_started':False,
        'estimated_duration_seconds':[180,300],'estimate_uncertain':True,
        'hard_timeout_enforced':False,'server_git_invoked':False}
    provenance=None; deployment=None; reference=None; expected=None
    with (OUTPUT/'stdout.log').open('x',encoding='utf8') as stdout,(OUTPUT/'stderr.log').open('x',encoding='utf8') as stderr:
        with redirect_logs(stdout,stderr):
            try:
                os.chdir(PROJECT)
                for name in THREADS: os.environ[name]='1'
                os.environ['RUN_WINDOW_INTEGRATION']='1'
                current=os.getpriority(os.PRIO_PROCESS,0)
                assert current<=10,'inherited nice already exceeds 10'
                if current<10: os.nice(10-current)
                metadata.update(environment_start=environment(),native_threads={k:os.environ[k] for k in THREADS},
                    run_window_integration=os.environ['RUN_WINDOW_INTEGRATION'])
                request,expected=verify_request(args.request_base64,PROJECT)
                metadata['implementation_commit']=request['implementation_commit']
                write(OUTPUT/'source_gate.json',{'request':request,'actual_source_sha256':expected,'sha_match':True})
                sys.path.insert(0,str(PROJECT)); sys.path.insert(0,str(PROJECT/'terminal_marl/core/rotating_slots'))
                from performance import provenance as imported_provenance
                provenance=imported_provenance
                assert not any(n=='terminal_marl' or n.startswith('terminal_marl.') for n in sys.modules),'core imported before preflight'
                reference=provenance._verify_export(PROJECT/'artifacts/rotating_slots/performance_reference_r0')
                assert len(reference['actual_source_files_sha256'])==24
                extras=list(request['files'])
                deployment=provenance.implementation_fingerprint(PROJECT,extra_paths=extras,modules={})
                deployment['implementation_commit']=request['implementation_commit']
                verified=provenance.verify_deployment(PROJECT,deployment,supplied_commit=request['implementation_commit'])
                write(OUTPUT/'source_preflight.json',{'deployment':deployment,'verified_deployment':verified,'reference':reference})
                preflight=True
                pytest_argv=[*TESTS,'-q','-x','-p','no:cacheprovider','--basetemp',str(OUTPUT/'pytest')]
                metadata['pytest_argv']=pytest_argv
                write(OUTPUT/'pytest_argv.json',pytest_argv)
                import pytest
                class CollectionEvidence:
                    def pytest_collection_finish(self,session):
                        write(OUTPUT/'pytest_collection.json',{'collected_tests':len(session.items),
                            'nodeids':[item.nodeid for item in session.items]})
                    def pytest_sessionfinish(self,session,exitstatus):
                        metadata['pytest_session_exitstatus']=int(exitstatus)
                        metadata['pytest_testsfailed']=session.testsfailed
                metadata['pytest_started']=True
                pytest_exit=int(pytest.main(pytest_argv,plugins=[CollectionEvidence()]))
                code=pytest_exit
            except BaseException as error:
                failure={'type':type(error).__name__,'message':str(error)}; code=2; traceback.print_exc()
            finally:
                # Verify again even after pytest failure; preserve its original exit.
                if provenance is not None and deployment is not None and reference is not None:
                    finish={}
                    try:
                        gate_after={name:sha(PROJECT/name) for name in expected}
                        finish.update(source_gate_sha256=gate_after,source_unchanged=gate_after==expected)
                        after=provenance._source_files(PROJECT,extra_paths=extras,modules={})[0]
                        reference_after=provenance._verify_export(PROJECT/'artifacts/rotating_slots/performance_reference_r0',reference)
                        finish.update(source_files_sha256=after,source_manifest_sha256=provenance.digest(after),
                            source_inventory_scope='test harness file inventory with explicit extras and modules={}; not production mixed-core import proof',
                            source_unchanged=gate_after==expected and after==deployment['actual_source_files_sha256'],reference=reference_after)
                        assert after==deployment['actual_source_files_sha256'],'test harness source inventory changed'
                        assert provenance.digest(after)==deployment['implementation_sha256'],'test harness source digest changed'
                        assert gate_after==expected,'source gate changed during tests'
                    except BaseException as error:
                        metadata['source_finish_error']={'type':type(error).__name__,'message':str(error)}
                        finish['verification_error']=metadata['source_finish_error']
                        code=2; traceback.print_exc()
                    write(OUTPUT/'source_finish.json',finish)
                metadata.update(ended_utc=utc(),wall_seconds=time.monotonic()-started,
                    exit_code=code,pytest_exit_code=pytest_exit,preflight_passed=preflight,failure=failure,
                    environment_finish=environment())
    write(OUTPUT/'run_metadata.json',metadata)
    with (OUTPUT/'numericexitcode.txt').open('x',encoding='utf8') as stream: stream.write(str(code)+'\n')
    print(json.dumps({'status':'TESTS_FINISHED' if code==0 else 'FAILED','output':str(OUTPUT),
        'exit_code':code,'pytest_exit_code':pytest_exit,'started_utc':metadata['started_utc'],'ended_utc':metadata['ended_utc']}))
    return code

if __name__=='__main__': raise SystemExit(main())
