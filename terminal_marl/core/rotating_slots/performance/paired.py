"""Serial AB/BA/AB common-window component gate; never full qualification."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from . import provenance
from .checkpoint import CheckpointIdentity, atomic_write_json
from .corpus import WINDOWS
from .process_metrics import OwnedProcessMetrics

IDS = tuple(row[0] for row in WINDOWS)
ORDER = ((1,"R0"),(1,"OPT"),(2,"OPT"),(2,"R0"),(3,"R0"),(3,"OPT"))
MEMBERS = ("states.jsonl","events.jsonl","rejections.jsonl","trajectories.json","manifest.json","metrics.json")
THREADS = ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS")
BUDGET = 7200.
HISTORICAL_POLICY = 'historical_noise_retry_v1'
REPORT_ONLY_POLICY = 'report_only_single_group_v1'


class PairedError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise PairedError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def finite(value, *, positive=False):
    return type(value) in (int,float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def plan(windows=IDS, group=1):
    require(group in (1,2), "only initial group and one noise retry are supported")
    require(len(windows) == len(set(windows)) and set(windows) <= set(IDS) and windows,
            "unknown/duplicate/empty window selection")
    return [{"window":window,"group":group,"repeat":repeat,"kernel":kernel,
             "relative_case":f"windows/{window}/group-{group}/repeat-{repeat}/{kernel}/attempt"}
            for window in IDS if window in windows for repeat,kernel in ORDER]


def noise(values):
    require(len(values)==3 and all(finite(x,positive=True) for x in values), "three positive finite wall values required")
    return max(values)/min(values) > 1.25


def aggregate(groups, *, full_coverage):
    """Pure aggregation of already verified selected groups; no best-run choice."""
    rows=[]
    for window in IDS:
        if window not in groups:
            continue
        raw=groups[window]
        r0=[r["rollout_wall_s"] for r in raw if r["kernel"]=="R0"]
        opt=[r["rollout_wall_s"] for r in raw if r["kernel"]=="OPT"]
        noisy=noise(r0) or noise(opt)
        b,o=statistics.median(r0),statistics.median(opt)
        cpu={k:statistics.median([r["rollout_parent_cpu_s"] for r in raw if r["kernel"]==k]) for k in ("R0","OPT")}
        full_cpu={}
        for field in ('collector_parent_cpu_s','joined_child_cpu_s','process_tree_cpu_s'):
            full_cpu[field]={k:statistics.median([r[field] for r in raw if r['kernel']==k])
                            if all(finite(r.get(field)) for r in raw if r['kernel']==k) else None for k in ('R0','OPT')}
        paired=[next(r["rollout_wall_s"] for r in raw if r["kernel"]=="R0" and r["repeat"]==i)/
                next(r["rollout_wall_s"] for r in raw if r["kernel"]=="OPT" and r["repeat"]==i) for i in (1,2,3)]
        rows.append({"window":window,"R0_raw_wall_s":r0,"OPT_raw_wall_s":opt,
                     "R0_median_wall_s":b,"OPT_median_wall_s":o,"R0_p95_wall_s":max(r0),"OPT_p95_wall_s":max(opt),
                     "nearest_rank_p95_sample_size":3,"R0_max_wall_s":max(r0),"OPT_max_wall_s":max(opt),
                     "paired_wall_ratios_by_repeat":paired,"median_wall_speedup":b/o,
                     "rollout_parent_cpu_medians_s":cpu,
                     "full_lifecycle_cpu_medians_s":full_cpu,
                     "full_lifecycle_cpu_speedup":full_cpu['process_tree_cpu_s']['R0']/full_cpu['process_tree_cpu_s']['OPT']
                        if full_cpu['process_tree_cpu_s']['OPT'] is not None and full_cpu['process_tree_cpu_s']['OPT']>0 and full_cpu['process_tree_cpu_s']['R0'] is not None else None,
                     "rollout_parent_cpu_speedup":cpu["R0"]/cpu["OPT"] if cpu["OPT"]>0 else None,
                     "per_window_no_regression":o<=max(1.1*b,b+.05),"status":"NOISY" if noisy else "STABLE"})
    geometric=math.exp(sum(math.log(r["median_wall_speedup"]) for r in rows)/len(rows)) if rows else None
    passed=(full_coverage and len(rows)==6 and all(r["status"]=="STABLE" and r["per_window_no_regression"] for r in rows)
            and geometric>=2.)
    return {"component_gate":"PASS" if passed else "NOT_PASSED","geometric_median_wall_speedup":geometric,
            "required_geometric_speedup":2.,"windows":rows,
            "performance_status":"NOT_QUALIFIED","human_status":"HUMAN_PENDING",
            "scope":"six 10sim-second serial component only; remaining contract requirements are not qualified"}


def system_snapshot(logical_cpu=None):
    result={"utc":utc(),"logical_cpu":logical_cpu,"platform":sys.platform,
            "native_threads":{name:os.environ.get(name) for name in THREADS}}
    if sys.platform=="linux":
        result.update(load_average=list(os.getloadavg()),allowed_logical_cpus=sorted(os.sched_getaffinity(0)),
                      inherited_nice=os.getpriority(os.PRIO_PROCESS,0))
        result["memory_kib"]={line.split(':',1)[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                              if line.startswith(('MemTotal:','MemAvailable:','SwapTotal:','SwapFree:'))}
        if logical_cpu is not None:
            directory=Path(f'/sys/devices/system/cpu/cpu{logical_cpu}/topology')
            result["cpu_topology"]={name:(directory/name).read_text().strip() for name in
                                    ('physical_package_id','core_id','thread_siblings_list')}
    return result


# Same child/guard structure as the reviewed diagnostic launcher: fresh -I/-B,
# one owned child, no signal callback, no core import until the real window CLI.
CHILD = r'''
import sys
sys.dont_write_bytecode=True
import json,os,pathlib,runpy,time
request=json.loads(sys.argv[1])
for name in request['thread_variables']: os.environ[name]='1'
linux=sys.platform=='linux'
environment={'pid':os.getpid(),'isolated':bool(sys.flags.isolated),'dont_write_bytecode':sys.dont_write_bytecode,
 'thread_environment':{n:os.environ[n] for n in request['thread_variables']},
 'core_modules_before_cli':[n for n in sys.modules if n=='terminal_marl' or n.startswith('terminal_marl.')]}
if linux:
 requested=set(request['cpu_affinity'])
 if not requested<=os.sched_getaffinity(0): raise RuntimeError('requested CPU affinity is unavailable')
 os.sched_setaffinity(0,requested)
 old=os.getpriority(os.PRIO_PROCESS,0)
 if old>10: raise RuntimeError('inherited nice exceeds 10')
 if old<10: os.nice(10-old)
 environment.update(affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0))
 sys.path.insert(0,str(pathlib.Path(request['project'])/'terminal_marl/core/rotating_slots'))
 from performance.deadline import LinuxWallDeadline
else:
 if request['kind'] in ('window','runner'): raise RuntimeError('real kernel collectors require Linux')
 environment.update(affinity=None,nice=None,linux_deadline='SKIPPED_UNSUPPORTED_PLATFORM')
pathlib.Path(request['environment']).write_text(json.dumps(environment,indent=2)+'\n',encoding='utf8')
def work():
 if request['kind'] in ('window','runner'):
  sys.argv=request['cli_argv']
  try: runpy.run_path(sys.argv[0],run_name='__main__')
  except SystemExit as exc:
   if exc.code is None: return 0
   if isinstance(exc.code,int): return exc.code
   print(str(exc.code),file=sys.stderr); return 1
  return 0
 if request['kind']=='deadline':
  if not linux: return 0
  time.sleep(5.); raise RuntimeError('deadline did not terminate child')
 time.sleep(.1)
 return 7 if request['kind']=='nonzero' else 0
if linux:
 with LinuxWallDeadline(request['budget'],request['deadlines']): code=work()
else: code=work()
raise SystemExit(code)
'''


def classify(returncode, pid, records, ended):
    matching=[r for r in records if r.get('state')=='ARMED' and r.get('pid')==pid and
              r.get('expiry_signal')==getattr(signal,'SIGALRM',14) and
              finite(r.get('deadline_monotonic_seconds')) and ended>=r['deadline_monotonic_seconds']]
    if returncode==-getattr(signal,'SIGALRM',14) and len(matching)==1:
        return 'PERFORMANCE_TIMEOUT'
    return 'EXIT_ZERO' if returncode==0 else 'IMPLEMENTATION_OR_STARTUP_ERROR'


def _collector_inputs(project, logical_cpu, cpu_affinity, kind, cli_argv):
    require(kind in ('window','runner','success','nonzero','deadline'), 'unknown collector kind')
    affinity = [logical_cpu] if cpu_affinity is None else list(cpu_affinity)
    require(affinity and all(type(value) is int and value >= 0 for value in affinity)
            and len(affinity) == len(set(affinity)), 'invalid CPU affinity list')
    affinity = sorted(affinity)
    output = None
    if kind in ('window','runner'):
        entry = 'run_rotating_slots_performance_windows.py' if kind == 'window' else 'run_rotating_slots_performance.py'
        require(type(cli_argv) is list and len(cli_argv) >= 2 and all(type(value) is str for value in cli_argv),
                'known CLI argv required')
        require(Path(cli_argv[0]).resolve() == (Path(project)/'scripts/server'/entry).resolve()
                and cli_argv[1] == ('window' if kind == 'window' else 'traffic'), 'collector requires exact known kernel CLI')
        require(cli_argv.count('--output') == 1, 'exact kernel output path required')
        index = cli_argv.index('--output')
        require(index+1 < len(cli_argv), 'missing kernel output value')
        output = str(Path(cli_argv[index+1]).resolve()/'result.json')
    return affinity, output


def _kernel_accounting(kind, returncode, result_path):
    """Completion evidence for descendants; exit zero alone proves no joins."""
    if kind not in ('window','runner'):
        return {'complete': True, 'workers': 0, 'kernel_resources': None,
                'reason': 'known bounded stdlib fixture creates no descendants'}
    observed = {'complete': False, 'workers': None, 'kernel_resources': None,
                'reason': 'kernel completion/join evidence unavailable'}
    try:
        result = read(result_path)
        require(isinstance(result, dict), 'invalid kernel result')
        workers = result.get('inner_workers')
        require(type(workers) is int and 0 <= workers <= 6, 'invalid kernel worker observation')
        container = result.get('resources') or {}
        require(isinstance(container, dict), 'invalid kernel resource observation')
        resources = container.get('resources', {}) if kind == 'runner' else container
        pool = container if kind == 'runner' else result.get('parallel') or {}
        require(isinstance(resources, dict) and isinstance(pool, dict), 'invalid kernel pool/resource observation')
        observed.update(workers=workers, kernel_resources=resources,
                        pool_closed_and_joined=pool.get('closed_and_joined') if workers else True)
        observed['complete'] = (returncode == 0 and result.get('failure') is None and
            resources.get('cpu_complete') is True and
            (not workers or pool.get('closed_and_joined') is True))
        observed['reason'] = ('successful own result confirms joined workers and complete kernel resources'
                              if observed['complete'] else 'exit/error or incomplete pool/resource evidence')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        observed['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    return observed


def collect(logs, *, project, logical_cpu=None, cpu_affinity=None, cli_argv=None, kind="window", budget=BUDGET, python=None):
    """One owned child from spawn through join; never invent a failed spawn exit."""
    affinity, result_path = _collector_inputs(project, logical_cpu, cpu_affinity, kind, cli_argv)
    logs=Path(logs); logs.mkdir(parents=True,exist_ok=False)
    start_cpu=time.process_time(); started=time.monotonic()
    request={"kind":kind,"project":str(project),"logical_cpu":logical_cpu,"cpu_affinity":affinity,"cli_argv":cli_argv,"budget":budget,
             "thread_variables":THREADS,"environment":str(logs/'runtime_environment.json'),"deadlines":str(logs/'deadlines')}
    command=[str(python or sys.executable),'-I','-B','-c',CHILD,json.dumps(request)]
    record={"command":command,"started_utc":utc(),"pid":None,"returncode":None,"status":"LAUNCHING",
            # Legacy no_pool describes this collector, not the owned kernel.
            "no_pool":True,"collector_uses_pool":False,"owned_kernel_child_count":1,"guard_wall_budget_s":budget,
            "external_collector_process_count":1,"requested_cpu_affinity":affinity,"kernel_result_path":result_path}
    atomic_write_json(logs/'collector.json',record)
    metrics=OwnedProcessMetrics(exclusive=True,rss_series_path=logs/'rss_samples.jsonl')
    child=None; joined=False
    try:
        with (logs/'stdout.log').open('xb') as out,(logs/'stderr.log').open('xb') as err:
            try:
                child=subprocess.Popen(command,stdout=out,stderr=err)
                record.update(pid=child.pid,status='RUNNING')
                atomic_write_json(logs/'collector.json',record)
                metrics.register([child.pid])
                record['returncode']=child.wait(); joined=True
            except OSError as exc:
                record['launch_error']={"type":type(exc).__name__,"message":str(exc)}
                if child is None: record['no_child_exitcode']=True
                else: record['returncode']=child.wait(); joined=True
    finally:
        # No pool or unrelated child is created/reaped during this context.
        if child is not None and not joined:
            child.wait(); joined=True
        report=metrics.finish(joined=joined or child is None)
    ended=time.monotonic()
    kernel = _kernel_accounting(kind, record['returncode'], result_path)
    observed_cpu = report['joined_children_cpu_seconds']
    complete = report['cpu_complete'] and kernel['complete']
    # Linux joined rusage propagates descendants reaped by the kernel. Windows
    # held kernel handles do not; add only its separately verified joined workers.
    full_cpu = observed_cpu if complete else None
    if complete and os.name == 'nt' and kernel['workers']:
        worker_cpu = kernel['kernel_resources'].get('joined_children_cpu_seconds')
        complete = finite(worker_cpu)
        full_cpu = observed_cpu + worker_cpu if complete else None
    report = dict(report, owned_pid_cpu_complete=report['cpu_complete'], cpu_complete=bool(complete),
                  descendant_cpu_complete=bool(complete),
                  descendant_accounting_note=kernel['reason'])
    guards=[read(path) for path in sorted((logs/'deadlines').glob('deadline_*.json'))]
    record.update(finished_utc=utc(),finished_monotonic_s=ended,full_child_wall_s=ended-started,
                  owned_kernel_child_count=int(child is not None),
                  collector_parent_cpu_s=time.process_time()-start_cpu,resources=report,guard_records=guards,
                  guard_scope='kernel child CLI; wrapper setup precedes arming; no other process is signalled',
                  joined_child_cpu_s=full_cpu, observed_reaped_child_cpu_lower_bound_s=observed_cpu,
                  process_tree_cpu_s=(time.process_time()-start_cpu+full_cpu) if full_cpu is not None else None,
                  kernel_completion_evidence=kernel, kernel_resources=kernel['kernel_resources'],
                  inner_worker_count=kernel['workers'], kernel_process_count=1 if child is not None else 0,
                  kernel_and_workers_process_count=(0 if child is None else
                      1+kernel['workers'] if kernel['workers'] is not None else None),
                  cpu_scope='extra collector plus joined kernel; descendants complete only with own successful pool/resource proof; observed reaped CPU retained on incomplete paths',
                  rss_scope='collector+kernel PID sweep and lifetime HWM estimates, not strict bounds; kernel+worker samples separate, no unmatched peaks summed')
    record['status']=classify(record['returncode'],record['pid'],guards,ended)
    if record['returncode'] is not None:
        (logs/'numericexitcode.txt').write_text(str(record['returncode'])+'\n',encoding='utf8')
    record['evidence_files_sha256']={path.relative_to(logs).as_posix():sha(path) for path in sorted(logs.rglob('*'))
                                    if path.is_file() and path.name!='collector.json'}
    record['collector_parent_cpu_s']=time.process_time()-start_cpu
    record['process_tree_cpu_s']=(record['collector_parent_cpu_s']+full_cpu) if full_cpu is not None else None
    atomic_write_json(logs/'collector.json',record)
    return record


def guarded_completion(record):
    rows=record.get('guard_records',[])
    require(len(rows)==1 and rows[0].get('state')=='DISARMED' and rows[0].get('pid')==record['pid'] and
            rows[0].get('expiry_signal')==getattr(signal,'SIGALRM',14) and rows[0].get('budget_seconds')==BUDGET,
            'missing/mismatched completed kernel deadline record')


def window_result(case, job, campaign, record, *, require_final_accounting=True):
    """Check persisted authoritative-window outputs, without a second simulation."""
    case=Path(case)
    logs=case.parent/'collector'
    require(record.get('returncode')==0 and record.get('status')=='COMPLETED_WINDOW', 'nonzero/incomplete collector result')
    require((logs/'numericexitcode.txt').read_text().strip()=='0', 'missing/nonzero actual numeric exit artifact')
    require(record.get('evidence_files_sha256')=={p.relative_to(logs).as_posix():sha(p) for p in sorted(logs.rglob('*'))
            if p.is_file() and p.name!='collector.json'}, 'collector evidence bytes changed')
    guarded_completion(record)
    require(record['guard_records']==[read(p) for p in sorted((logs/'deadlines').glob('deadline_*.json'))], 'kernel guard evidence changed')
    environment=read(logs/'runtime_environment.json')
    require(environment.get('pid')==record['pid'] and environment.get('isolated') is True and
            environment.get('dont_write_bytecode') is True and environment.get('affinity')==[campaign['logical_cpu']] and
            environment.get('nice')==10 and environment.get('thread_environment')=={name:'1' for name in THREADS},
            'child affinity/nice/native/isolated environment differs')
    require(environment.get('core_modules_before_cli')==[], 'core imported before child CLI bootstrap')
    resources=record.get('resources',{})
    require(resources.get('cpu_complete') is True and resources.get('errors')==[] and resources.get('worker_pids')==[record['pid']] and
            finite(record.get('collector_parent_cpu_s')) and finite(record.get('joined_child_cpu_s')) and
            finite(record.get('process_tree_cpu_s')), 'incomplete full child/collector CPU accounting')
    require(record['process_tree_cpu_s']>=record['collector_parent_cpu_s']+record['joined_child_cpu_s'], 'CPU total omits owned child')
    require(finite(resources.get('sampled_peak_simultaneous_rss_bytes')) and
            resources.get('rss_sample_count',0)>0 and resources.get('rss_full_series_path') and
            (logs/'rss_samples.jsonl').is_file(), 'RSS samples missing')
    if require_final_accounting:
        require(finite(record.get('collector_validation_parent_cpu_s')) and
                finite(record.get('collector_parent_cpu_including_validation_s')) and finite(record.get('process_tree_cpu_including_validation_s')) and
                record['collector_parent_cpu_including_validation_s']==record['collector_parent_cpu_s']+record['collector_validation_parent_cpu_s'] and
                record['process_tree_cpu_including_validation_s']==record['collector_parent_cpu_including_validation_s']+record['joined_child_cpu_s'],
                'post-join collector CPU omitted/mislabeled')
    result=read(case/'result.json'); identity=read(case/'input_identity.json')
    frozen=campaign['frozen_windows'][job['window']]
    require(record.get('result_sha256')==sha(case/'result.json') and
            record.get('input_identity_sha256')==sha(case/'input_identity.json'), 'result/input identity hash changed')
    require(result.get('schema')=='ft_rs_common_window_v1_0' and result.get('status')=='COMPLETED_WINDOW' and
            result.get('failure') is None and result.get('window')==job['window'] and result.get('duration_s')==10 and
            result.get('mode')=='benchmark' and result.get('inner_workers')==0 and
            result.get('execution_backend')==job['kernel']+'_SERIAL', 'invalid physical window/backend/mode completion')
    require(result.get('corpus_sha256')==campaign['corpus_sha256'] and result.get('workload')==frozen['workload'] and
            result.get('original_advance_target')==frozen['cursor']['original_target'] and
            result.get('outer_phase')==frozen['cursor']['outer_phase'], 'frozen workload/original invocation changed')
    source=result['source_identity']
    require(source.get('kernel')==job['kernel'] and source.get('runner_execution_sha256')==campaign['target_execution_sha256'][job['kernel']] and
            source.get('semantic_fingerprint')==campaign['semantic_fingerprint'] and source.get('implementation_commit')==campaign['implementation_commit'],
            'runtime implementation/semantics changed')
    require(identity.get('corpus_sha256')==campaign['corpus_sha256'] and identity.get('window')==job['window'] and
            identity.get('selected_cut')==frozen['cut'] and identity.get('source_identity')==frozen['identity'] and
            identity.get('source_cursor')==frozen['cursor'] and identity.get('runtime_cursor')==frozen['cursor'] and
            identity.get('target_identity')==dict(frozen['identity'],implementation_sha256=campaign['target_execution_sha256'][job['kernel']]),
            'checkpoint cut/identity/cursor changed')
    cold=result.get('cold_cache_evidence',{})
    required_cache={'trajectory.poly_bounds','energy._integrable_plan','energy._segment_table',
                    'safety._NUMERIC_CERTIFICATES','safety._PLAN_CERTIFICATES'}
    observed_cache=cold.get('module_numerical_caches',{})
    optional_root=observed_cache.get('energy._ROOT_CACHE')
    optional_peak=observed_cache.get('energy._PEAK_CACHE')
    optional_metadata=observed_cache.get('trajectory._BERNSTEIN_METADATA')
    require(result.get('cache_mode')=='cold' and cold.get('all_named_acceleration_caches_empty') is True and
            cold.get('module_caches_cleared') is False and
            set(cold.get('world_entries',{}))=={'_q_cache','_energy_cache','_reference_templates'} and
            all(v==0 for v in cold['world_entries'].values()) and required_cache<=set(observed_cache) and
            set(observed_cache)<=required_cache|{'energy._ROOT_CACHE','energy._PEAK_CACHE','trajectory._BERNSTEIN_METADATA'} and
            all(v.get('currsize',v.get('entries'))==0 for v in observed_cache.values()) and
            (optional_root is None or (optional_root.get('entries')==0 and type(optional_root.get('present')) is bool)) and
            (optional_peak is None or (type(optional_peak.get('entries')) is int and optional_peak['entries']==0
                                       and type(optional_peak.get('present')) is bool)) and
            (optional_metadata is None or (type(optional_metadata.get('entries')) is int and optional_metadata['entries']==0
                                           and type(optional_metadata.get('present')) is bool)),
            'cold-cache evidence missing/nonempty/unknown')
    start=result.get('start_sim_time'); stop=result.get('actual_stop_sim_time'); delta=result.get('actual_sim_delta_s')
    require(all(finite(v) for v in (start,stop,delta)) and float(start).hex()==float(frozen['cut']['actual_sim_time']).hex() and
            stop>=start+10. and float(delta).hex()==float(stop-start).hex() and result.get('nominal_stop_sim_time')==start+10. and
            result.get('simulation_deferral_s')==stop-(start+10.), 'window timeline/cut/deferral invalid')
    require(result.get('physical_outcome',{}).get('failed') is False and
            result.get('physical_outcome',{}).get('episode_complete') is False, 'window mistaken for episode/failure')
    timing=result.get('timing',{})
    require(finite(timing.get('rollout_wall_s'),positive=True) and finite(timing.get('rollout_parent_process_cpu_s')),
            'missing rollout wall/real process CPU')
    hashes={name:sha(case/'replay'/name) for name in MEMBERS}
    require(hashes==record.get('replay_sha256') and sha(case/'replay/replay.json')==record.get('replay_index_sha256'), 'replay bytes changed')
    require(record.get('window_time_hex')=={name:float(result[name]).hex() for name in
            ('start_sim_time','actual_stop_sim_time','actual_sim_delta_s')}, 'persisted exact timeline changed')
    return dict(job,rollout_wall_s=timing['rollout_wall_s'],rollout_parent_cpu_s=timing['rollout_parent_process_cpu_s'],
                collector_parent_cpu_s=record.get('collector_parent_cpu_including_validation_s',record['collector_parent_cpu_s']),joined_child_cpu_s=record['joined_child_cpu_s'],
                process_tree_cpu_s=record.get('process_tree_cpu_including_validation_s',record['process_tree_cpu_s']),replay_sha256=hashes,window_time_hex=record['window_time_hex'])


def group_result(root, window, group, campaign):
    directory=Path(root)/f'windows/{window}/group-{group}'
    metadata=read(directory/'group.json')
    expected=plan((window,),group)
    require(metadata.get('window')==window and metadata.get('group')==group and
            metadata.get('jobs')==expected, 'group order/coverage changed')
    require(all(isinstance(metadata.get(name),dict) and metadata[name].get('utc') and
            len(metadata[name].get('load_average',[]))==3 and metadata[name].get('memory_kib')
            for name in ('load_before','load_after')), 'load/memory recheck evidence missing')
    require(set(directory.rglob('collector.json'))=={(Path(root)/j['relative_case']).parent/'collector/collector.json' for j in expected},
            'missing/duplicate/unexpected collector coverage')
    rows=[]
    for job in expected:
        case=Path(root)/job['relative_case']; record=read(case.parent/'collector/collector.json')
        require(record.get('job')==job,'collector job identity/order changed')
        rows.append(window_result(case,job,campaign,record))
    baseline=rows[0]
    require(all(r['replay_sha256']==baseline['replay_sha256'] and r['window_time_hex']==baseline['window_time_hex'] for r in rows),
            'semantic/timeline difference within common-state group')
    noisy=any(noise([r['rollout_wall_s'] for r in rows if r['kernel']==k]) for k in ('R0','OPT'))
    require(metadata.get('status')==('NOISY' if noisy else 'COMPLETED_GROUP'),'group noise/status changed')
    return rows,noisy


def summarize(root):
    root=Path(root); campaign=read(root/'campaign.json')
    require(campaign.get('schema')=='ft_rs_serial_paired_campaign_v1' and campaign.get('duration_s')==10 and
            campaign.get('repeat_order')==[[r,k] for r,k in ORDER], 'campaign schema/protocol changed')
    policy=campaign.get('measurement_policy',HISTORICAL_POLICY)
    require(policy in (HISTORICAL_POLICY,REPORT_ONLY_POLICY),'unknown saved measurement policy')
    report_only=policy==REPORT_ONLY_POLICY
    selected=campaign.get('windows',[])
    require(selected==[w for w in IDS if w in selected] and len(selected)==len(set(selected)) and selected,
            'unknown/duplicate/reordered campaign windows')
    groups={}; retained={}; errors=[]
    if campaign.get('status')!='COMPLETED_PAIRED': errors.append({'message':'campaign did not complete without collector error'})
    all_group_paths=set(root.glob('windows/*/group-*/group.json'))
    expected_paths=set()
    for window in selected:
        try:
            rows,noisy=group_result(root,window,1,campaign)
            expected_paths.add(root/f'windows/{window}/group-1/group.json')
            retained[window]=[{'group':1,'status':'NOISY' if noisy else 'COMPLETED_GROUP','raw':rows}]
            if noisy and not report_only:
                retry,still_noisy=group_result(root,window,2,campaign)
                expected_paths.add(root/f'windows/{window}/group-2/group.json')
                retained[window].append({'group':2,'status':'NOISY' if still_noisy else 'COMPLETED_GROUP','raw':retry})
                # Exactly the retry group is used, even when slower; no min/best selection.
                require(all(r['replay_sha256']==rows[0]['replay_sha256'] and r['window_time_hex']==rows[0]['window_time_hex'] for r in retry),
                        'noise retry changed common-state semantics/timeline')
                groups[window]=retry
            else:
                groups[window]=rows
        except (OSError,ValueError,KeyError,TypeError) as exc:
            errors.append({'window':window,'type':type(exc).__name__,'message':str(exc)})
    if all_group_paths != expected_paths:
        errors.append({'message':'missing/extra group coverage','actual':[str(p.relative_to(root)) for p in sorted(all_group_paths)],
                       'expected':[str(p.relative_to(root)) for p in sorted(expected_paths)]})
    all_rows=[row for attempts in retained.values() for entry in attempts for row in entry['raw']]
    child_sum=sum(row['joined_child_cpu_s'] for row in all_rows)
    if not (campaign.get('full_child_cpu_complete') is True and campaign.get('started_kernel_child_count')==len(all_rows) and
            finite(campaign.get('collector_process_lifetime_cpu_s')) and finite(campaign.get('collector_run_interval_cpu_s')) and
            campaign.get('full_joined_kernel_children_cpu_s')==child_sum and
            campaign.get('total_process_tree_cpu_s')==campaign['collector_process_lifetime_cpu_s']+child_sum):
        errors.append({'message':'campaign full process-lifetime CPU/child coverage missing or inconsistent'})
    output=aggregate(groups,full_coverage=not errors and not campaign.get('diagnostic_subset') and selected==list(IDS)) if groups else {
        'component_gate':'NOT_PASSED','performance_status':'NOT_QUALIFIED','human_status':'HUMAN_PENDING','windows':[]}
    if report_only:
        output['historical_component_gate']=output['component_gate']
        output['component_gate']='REPORT_ONLY'
    if errors: output['component_gate']='INVALID_OR_INCOMPLETE'
    output.update(schema='ft_rs_serial_paired_summary_v1',campaign=str(root),errors=errors,retained_groups=retained,
                  measurement_policy=policy,report_only=report_only,
                  amended_scope_status=('INTEGRITY_COMPLETE_REPORT_ONLY' if not errors else 'INVALID_OR_INCOMPLETE') if report_only else None,
                  noise_is_advisory=report_only,
                  selected_group_policy=('initial AB/BA/AB group only; noise advisory; no retry' if report_only else
                      'initial unless NOISY; then mandatory whole-group retry exactly once; retain both'),
                  campaign_status=campaign.get('status'),diagnostic_subset=bool(campaign.get('diagnostic_subset')),
                  full_expected_window_count=6,full_expected_initial_job_count=36,
                  source_campaign_sha256=sha(root/'campaign.json'))
    return output


def preflight(args, project):
    from . import windows
    from .contract import preflight as contract_preflight
    require(sys.platform=='linux' and sys.flags.isolated and sys.dont_write_bytecode, 'paired requires fresh Linux -I -B process')
    require(not any(n=='terminal_marl' or n.startswith('terminal_marl.') for n in sys.modules), 'core imported into collector')
    require(not args.output.exists(), 'new campaign output only; refusing overwrite')
    require(len(args.implementation_commit)==40 and all(c in '0123456789abcdef' for c in args.implementation_commit), 'exact commit label required')
    corpus=read(args.corpus_manifest)
    for protected in (args.corpus_manifest.resolve().parent,args.reference.resolve(),
                      *(Path(source['source_attempt']).resolve() for source in corpus['sources'].values())):
        require(not args.output.resolve().is_relative_to(protected) and not protected.is_relative_to(args.output.resolve()),
                'campaign output must be separate from frozen sources/reference')
    for name in IDS: windows.validate_manifest(corpus,args.corpus_sha256,name,10)
    deployment=provenance.verify_deployment(project,read(args.deployment_manifest),supplied_commit=args.implementation_commit)
    reference=provenance._verify_export(args.reference)
    require(len(reference['actual_source_files_sha256'])==24,'fixed R0 file count changed')
    contract_preflight(project/'rotating_slots_performance_contract_v1_0',project)
    frozen_semantics=provenance.semantic_fingerprint(args.reference)
    provenance.require_same_semantics(frozen_semantics,corpus['semantic_fingerprint'])
    helper_sha=sha(project/'terminal_marl/core/rotating_slots/performance/checkpoint.py')
    targets={}
    for kernel,path in (('R0',args.r0_migration_approval),('OPT',args.opt_migration_approval)):
        approval=read(path); pairs=approval.get('migrations',[])
        target_keys={provenance.digest(pair['target_identity']) for pair in pairs}
        require(len(target_keys)==1,'approval must identify exactly one runtime target')
        target=CheckpointIdentity(**pairs[0]['target_identity'])
        target_meta={'observation_original_source_sha256':approval['target_observation_original_sha256'],
                     'observation_generated_source_sha256':approval['target_observation_generated_sha256']}
        for row in corpus['windows']:
            windows.validate_approval(approval,source=CheckpointIdentity(**row['validation']['identity']),target=target,
                corpus_digest=args.corpus_sha256,source_metadata=corpus['sources'][row['case']]['source_identity'],
                target_metadata=target_meta,checkpoint_helper_sha256=helper_sha)
        targets[kernel]=target.implementation_sha256
    snapshot=system_snapshot(args.logical_cpu)
    require(args.logical_cpu in snapshot['allowed_logical_cpus'] and snapshot['inherited_nice']<=10,'CPU/nice request unavailable')
    return corpus,targets,{'system':snapshot,'deployment':deployment,'reference':reference,
        'target_runtime_verification':'fresh window CLI bootstrap checks actual target before restore/advance',
        'input_files_sha256':{str(p.resolve()):sha(p) for p in (args.corpus_manifest,args.deployment_manifest,
            args.r0_migration_approval,args.opt_migration_approval)}}


def window_command(args, project, job, case):
    """Production command only; tests can replace this narrow orchestration seam."""
    return [str(project/'scripts/server/run_rotating_slots_performance_windows.py'),'window',
        '--corpus-manifest',str(args.corpus_manifest.resolve()),'--corpus-sha256',args.corpus_sha256,
        '--window',job['window'],'--duration','10','--kernel',job['kernel'],'--mode','benchmark',
        '--reference',str(args.reference.resolve()),'--deployment-manifest',str(args.deployment_manifest.resolve()),
        '--implementation-commit',args.implementation_commit,'--migration-approval',str(
            (args.r0_migration_approval if job['kernel']=='R0' else args.opt_migration_approval).resolve()),'--output',str(case)]


def paired(args, project):
    started_cpu=time.process_time(); joined_cpu=0.; observed_cpu=0.; joined_cpu_complete=True; launched_children=0
    report_only=bool(getattr(args,'report_only',False))
    for name in THREADS: os.environ[name]='1'
    corpus,targets,evidence=preflight(args,project)
    selected=tuple(w for w in IDS if args.diagnostic_windows is None or w in args.diagnostic_windows)
    require(args.diagnostic_windows is None or len(args.diagnostic_windows)==len(set(args.diagnostic_windows)), 'duplicate diagnostic window')
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    campaign={'schema':'ft_rs_serial_paired_campaign_v1','status':'RUNNING','started_utc':utc(),'command':list(sys.argv),
        'duration_s':10,'windows':list(selected),'diagnostic_subset':args.diagnostic_windows is not None,
        'measurement_policy':REPORT_ONLY_POLICY if report_only else HISTORICAL_POLICY,
        'repeat_order':[[r,k] for r,k in ORDER],'logical_cpu':args.logical_cpu,'implementation_commit':args.implementation_commit,
        'corpus_sha256':args.corpus_sha256,'semantic_fingerprint':corpus['semantic_fingerprint'],'target_execution_sha256':targets,
        'preflight':evidence,'frozen_windows':{row['window']:{'cut':row['cut'],'cursor':row['validation']['cursor'],
            'identity':row['validation']['identity'],'workload':corpus['sources'][row['case']]['workload']} for row in corpus['windows']},
        'performance_status':'NOT_QUALIFIED','human_status':'HUMAN_PENDING'}
    atomic_write_json(root/'campaign.json',campaign)
    failure=None
    try:
        for window in selected:
            for group in ((1,) if report_only else (1,2)):
                jobs=plan((window,),group); directory=root/f'windows/{window}/group-{group}'
                metadata={'window':window,'group':group,'jobs':jobs,'status':'RUNNING','load_before':system_snapshot(args.logical_cpu)}
                atomic_write_json(directory/'group.json',metadata)
                rows=[]
                for job in jobs:
                    case=root/job['relative_case']
                    # Each slot has sibling collector/ and new attempt/ paths;
                    # no directory for the actual window output is precreated.
                    logs=case.parent/'collector'
                    argv=window_command(args,project,job,case)
                    record=collect(logs,project=project,logical_cpu=args.logical_cpu,cli_argv=argv)
                    validation_cpu=time.process_time()
                    record['job']=job
                    if record['pid'] is not None: launched_children+=1
                    if record.get('joined_child_cpu_s') is not None: joined_cpu+=record['joined_child_cpu_s']
                    if finite(record.get('observed_reaped_child_cpu_lower_bound_s')):
                        observed_cpu+=record['observed_reaped_child_cpu_lower_bound_s']
                    if record['pid'] is not None and not record['resources'].get('cpu_complete'): joined_cpu_complete=False
                    if record['status']=='EXIT_ZERO' and (case/'result.json').is_file():
                        result=read(case/'result.json'); record['status']=result.get('status','IMPLEMENTATION_FAILURE')
                        record['result_sha256']=sha(case/'result.json')
                        record['input_identity_sha256']=sha(case/'input_identity.json')
                        if record['status']=='COMPLETED_WINDOW':
                            record['replay_sha256']={name:sha(case/'replay'/name) for name in MEMBERS}
                            record['replay_index_sha256']=sha(case/'replay/replay.json')
                            record['window_time_hex']={name:float(result[name]).hex() for name in
                                ('start_sim_time','actual_stop_sim_time','actual_sim_delta_s')}
                    elif record['status']=='EXIT_ZERO': record['status']='MISSING_RESULT'
                    try:
                        atomic_write_json(logs/'collector.json',record)
                        row=window_result(case,job,campaign,record,require_final_accounting=False)
                        require(not rows or (row['replay_sha256']==rows[0]['replay_sha256'] and row['window_time_hex']==rows[0]['window_time_hex']),
                                'semantic/timeline difference between paired jobs')
                        rows.append(row)
                    finally:
                        record['collector_validation_parent_cpu_s']=time.process_time()-validation_cpu
                        record['collector_parent_cpu_including_validation_s']=record['collector_parent_cpu_s']+record['collector_validation_parent_cpu_s']
                        record['process_tree_cpu_including_validation_s']=record['collector_parent_cpu_including_validation_s']+record['joined_child_cpu_s'] if record['joined_child_cpu_s'] is not None else None
                        atomic_write_json(logs/'collector.json',record)
                noisy=any(noise([r['rollout_wall_s'] for r in rows if r['kernel']==k]) for k in ('R0','OPT'))
                metadata.update(status='NOISY' if noisy else 'COMPLETED_GROUP',load_after=system_snapshot(args.logical_cpu))
                atomic_write_json(directory/'group.json',metadata)
                if not noisy: break
        campaign['status']='COMPLETED_PAIRED'
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc)}
        campaign['status']='STOPPED_ON_ERROR'
    finally:
        parent_lifetime_cpu=time.process_time()
        campaign.update(finished_utc=utc(),failure=failure,collector_process_lifetime_cpu_s=parent_lifetime_cpu,
                        collector_run_interval_cpu_s=parent_lifetime_cpu-started_cpu,
                        full_joined_kernel_children_cpu_s=joined_cpu if joined_cpu_complete else None,
                        observed_joined_children_cpu_lower_bound_s=observed_cpu,
                        full_child_cpu_complete=joined_cpu_complete,started_kernel_child_count=launched_children,
                        total_process_tree_cpu_s=parent_lifetime_cpu+joined_cpu if joined_cpu_complete else None,
                        total_cpu_scope='time.process_time absolute at final accounting snapshot: collector process startup/imports included; plus every full joined serial kernel child. Final campaign/summary JSON serialization and stdout emission occur after this snapshot and are excluded; run interval and rollout CPU are separate')
        atomic_write_json(root/'campaign.json',campaign)
    output=summarize(root)
    atomic_write_json(root/'summary.json',output)
    return output


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    commands=result.add_subparsers(dest='command',required=True)
    command=commands.add_parser('paired')
    for name in ('corpus-manifest','reference','deployment-manifest','r0-migration-approval','opt-migration-approval','output'):
        command.add_argument('--'+name,type=Path,required=True)
    command.add_argument('--corpus-sha256',required=True)
    command.add_argument('--implementation-commit',required=True)
    command.add_argument('--logical-cpu',type=int,required=True)
    command.add_argument('--diagnostic-windows',nargs='+',choices=IDS)
    command.add_argument('--report-only',action='store_true',help='one AB/BA/AB group; performance and noise are observations')
    command=commands.add_parser('summarize')
    command.add_argument('--campaign',type=Path,required=True)
    command.add_argument('--output',type=Path,help='optional new summary JSON; never overwrite')
    return result


def main(argv=None, *, helper_root=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='paired':
            output=paired(args,Path(helper_root or Path(__file__).resolve().parents[4]))
        else:
            require(args.output is None or not args.output.exists(),'summary output exists')
            output=summarize(args.campaign)
            if args.output is not None: atomic_write_json(args.output,output)
        print(json.dumps(output,allow_nan=False))
        return 1 if output.get('errors') else 0
    except Exception as exc:
        print(json.dumps({'status':'REJECTED','error_type':type(exc).__name__,'error':str(exc),
                          'performance_status':'NOT_QUALIFIED','human_status':'HUMAN_PENDING'}))
        return 2
