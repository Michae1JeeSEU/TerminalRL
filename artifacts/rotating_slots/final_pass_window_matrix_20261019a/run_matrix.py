"""Fixed window matrices; artifact preparation, no automatic backend choice."""
import sys
sys.dont_write_bytecode=True
import argparse,ast,json,math,os,statistics,time
from hashlib import sha256
from pathlib import Path

PROJECT=Path('/home/user/projects/terminal')
OUTPUTS=Path('/home/user/training_outputs/terminal')
IDS=('W71-A','W71-B','W71-C','W60','W120','W180')
WORKERS=(1,2,4,6)
def require(ok,message):
    if not ok: raise ValueError(message)
def read(path): return json.loads(Path(path).read_bytes())
def save(path,value):
    with Path(path).open('x',encoding='utf8') as f: json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
def plan(mode,workers=None):
    require(mode in ('parallel','sustained') and (mode=='parallel' or workers in (0,*WORKERS)),'explicit supported sustained backend required')
    return [dict(window=w,workers=n,repeat=r,duration=10 if mode=='parallel' else 60,kernel='OPT',
        relative_case=f'windows/{w}/workers-{n}/repeat-{r}/attempt')
        for w in IDS for n in (WORKERS if mode=='parallel' else (workers,)) for r in ((1,) if mode=='parallel' else (1,2,3))]
def command(a,j,case):
    argv=[str(PROJECT/'scripts/server/run_rotating_slots_performance_windows.py'),'window']
    for flag,value in (('corpus-manifest',a.corpus_manifest),('corpus-sha256',a.corpus_sha256),('window',j['window']),
        ('duration',j['duration']),('kernel','OPT'),('mode','benchmark'),('reference',a.reference),
        ('deployment-manifest',a.deployment_manifest),('implementation-commit',a.implementation_commit),
        ('migration-approval',a.opt_migration_approval),('inner-workers',j['workers']),
        ('core-budget',max(1,j['workers'])),('process-budget',j['workers']+1),('output',case)):
        argv.extend(('--'+flag,str(value)))
    return argv
def aggregate(rows,baseline,mode):
    table=[]
    for n in (0,*WORKERS):
        windows=[]; ratios=[]
        for w in IDS:
            raw=([r for r in baseline[w] if r['kernel']=='OPT'] if n==0 and mode=='parallel' else
                 [r for r in rows if r['window']==w and r['workers']==n and r['status']=='COMPLETED_WINDOW'])
            if not raw: continue
            med=lambda key: statistics.median([r[key] for r in raw]) if all(type(r.get(key)) in (int,float) for r in raw) else None
            wall=med('rollout_wall_s'); serial=statistics.median(r['rollout_wall_s'] for r in baseline[w] if r['kernel']=='OPT')
            ratio=serial/wall if mode=='parallel' else None
            if ratio is not None: ratios.append(ratio)
            windows.append(dict(window=w,n=len(raw),duration_s=10 if mode=='parallel' else 60,rollout_median_s=wall,
                nominal_duration_s=10 if mode=='parallel' else 60,actual_sim_delta_s=[r['actual_sim_delta_s'] for r in raw],
                actual_wall_per_sim_median_s=statistics.median(r['rollout_wall_s']/r['actual_sim_delta_s'] for r in raw),
                actual_e2e_wall_per_sim_median_s=statistics.median(r['e2e_wall_s']/r['actual_sim_delta_s'] for r in raw),
                e2e_median_s=med('e2e_wall_s'),kernel_parent_workers_cpu_median_s=med('kernel_parent_workers_cpu_s'),
                collector_cpu_median_s=med('collector_cpu_s'),collector_plus_kernel_cpu_median_s=med('collector_plus_kernel_cpu_s'),serial_OPT_rollout_ratio=ratio))
        expected=3 if n==0 and mode=='parallel' else 1 if mode=='parallel' else 3
        table.append(dict(backend=n,windows=windows,valid=len(windows)==6 and all(w['n']==expected for w in windows),
            geometric_actual_wall_per_sim_s=math.exp(sum(math.log(w['actual_wall_per_sim_median_s']) for w in windows)/len(windows)) if windows else None,
            actual_wall_per_sim_covered_windows=len(windows),
            geometric_serial_OPT_rollout_ratio=math.exp(sum(map(math.log,ratios))/len(ratios)) if ratios else None,
            ratio_window_count=len(ratios),sampling_note='n=1 per parallel window; no empirical tail inference' if mode=='parallel' and n else
            'initial three serial samples retained' if mode=='parallel' else 'three fixed 60s repetitions; no 10s/60s speed ratio'))
    return dict(table=table,automatic_backend_choice=None,performance_threshold=None)
def baseline(a,p):
    campaign=read(a.serial_campaign/'campaign.json'); summary=p.summarize(a.serial_campaign)
    require(summary['report_only'] and not summary['errors'] and campaign['status']=='COMPLETED_PAIRED' and campaign['windows']==list(IDS),'complete six-window report-only serial campaign required')
    require(campaign['implementation_commit']==a.implementation_commit and campaign['corpus_sha256']==a.corpus_sha256,'serial source/corpus mismatch')
    groups=summary['retained_groups']; result={}
    for w in IDS:
        require(len(groups[w])==1 and groups[w][0]['group']==1,'initial three trials only; no retry selection')
        result[w]=groups[w][0]['raw']
        for r in result[w]:
            rec=read((a.serial_campaign/r['relative_case']).parent/'collector/collector.json')
            physical=read(a.serial_campaign/r['relative_case']/'result.json')
            r.update(e2e_wall_s=rec['full_child_wall_s'],kernel_parent_workers_cpu_s=r['joined_child_cpu_s'],
                nominal_duration_s=physical['duration_s'],actual_sim_delta_s=physical['actual_sim_delta_s'],
                collector_cpu_s=r['collector_parent_cpu_s'],collector_plus_kernel_cpu_s=r['process_tree_cpu_s'])
    return campaign,result
def validate(a,j,case,rec,p,campaign,targets):
    r=read(case/'result.json'); inp=read(case/'input_identity.json'); frozen=campaign['frozen_windows'][j['window']]
    require(rec['returncode']==0 and rec['status']=='EXIT_ZERO' and r['status']=='COMPLETED_WINDOW' and r['failure'] is None,'physical/implementation failure or incomplete window')
    require(r['pid']==rec['pid'] and r['window']==j['window'] and r['duration_s']==j['duration'] and r['inner_workers']==j['workers'] and r['mode']=='benchmark','job/result mismatch')
    require(r['execution_backend']==('OPT_PARALLEL' if j['workers'] else 'OPT_SERIAL') and r['physical_outcome']['failed'] is False,'backend/physical outcome mismatch')
    source=r['source_identity']; target=targets['OPT']
    require(source['kernel']=='OPT' and source['runner_execution_sha256']==target and source['implementation_commit']==a.implementation_commit and source['semantic_fingerprint']==campaign['semantic_fingerprint'],'runtime source/semantics changed')
    require(r['corpus_sha256']==a.corpus_sha256 and r['workload']==frozen['workload'] and r['original_advance_target']==frozen['cursor']['original_target'] and r['outer_phase']==frozen['cursor']['outer_phase'],'frozen invocation changed')
    require(inp['selected_cut']==frozen['cut'] and inp['source_identity']==frozen['identity'] and inp['source_cursor']==inp['runtime_cursor']==frozen['cursor'] and inp['target_identity']==dict(frozen['identity'],implementation_sha256=target),'input cut/identity/cursor changed')
    require(inp['corpus_sha256']==a.corpus_sha256 and inp['window']==j['window'],'input identity mismatch')
    cold=r['cold_cache_evidence']; require(r['cache_mode']=='cold' and cold['all_named_acceleration_caches_empty'] and cold['module_caches_cleared'] is False,'cold-start proof missing')
    require(all(v==0 for v in cold['world_entries'].values()) and all(v.get('currsize',v.get('entries'))==0 for v in cold['module_numerical_caches'].values()),'cold caches nonempty')
    guards=rec['guard_records']; env=read(case.parent/'collector/runtime_environment.json')
    require(len(guards)==1 and guards[0]['state']=='DISARMED' and guards[0]['pid']==rec['pid'] and guards[0]['budget_seconds']==a.budget,'completed guard mismatch')
    require(env['pid']==rec['pid'] and env['isolated'] and env['dont_write_bytecode'] and env['nice']==10 and env['affinity']==sorted(a.cpus[:max(1,j['workers'])]) and env['core_modules_before_cli']==[],'child execution environment mismatch')
    require(env['thread_environment']=={n:'1' for n in p.THREADS} and rec['resources']['cpu_complete'] and rec['kernel_completion_evidence']['complete'] and r['resources']['cpu_complete'],'full successful kernel/worker CPU proof missing')
    if j['workers']:
        pool=r['parallel']; starts=pool['observed_worker_startups']
        require(pool['closed_and_joined'] and pool['worker_startup_coverage_complete'] and len(starts)==j['workers'] and {s['pid'] for s in starts}==set(pool['worker_pids']),'worker startup/join coverage missing')
        require(all(s['expected_parent_pid']==s['actual_parent_pid']==r['pid'] and s['parent_death']['status']=='BOUND_TO_EXPECTED_PARENT' and s['isolated'] and s['dont_write_bytecode'] and all(v.get('currsize',v.get('entries'))==0 for snap in (s['cache_reset']['after'],s['cache_after_world_construction']) for v in snap.values()) for s in starts),'worker parent/cold reset proof missing')
    start,stop,delta=(r[k] for k in ('start_sim_time','actual_stop_sim_time','actual_sim_delta_s'))
    require(float(start).hex()==float(frozen['cut']['actual_sim_time']).hex() and stop>=start+j['duration'] and float(delta).hex()==float(stop-start).hex() and r['nominal_stop_sim_time']==start+j['duration'] and r['simulation_deferral_s']==stop-(start+j['duration']),'exact timeline/deferral mismatch')
    times={k:float(r[k]).hex() for k in ('start_sim_time','actual_stop_sim_time','actual_sim_delta_s')}
    evidence=[case/'result.json',case/'input_identity.json',case.parent/'collector/collector.json',*sorted((case.parent/'collector/deadlines').glob('deadline_*.json')),*[case/'replay'/n for n in p.MEMBERS]]
    rss=lambda value:{k:value.get(k) for k in ('sampled_peak_rss_sweep_sum_bytes','lifetime_hwm_sum_estimate_bytes','rss_full_series_path','rss_sample_count','rss_is_strict_bound','errors')}
    require(p.finite(r['timing']['rollout_wall_s'],positive=True) and p.finite(rec['full_child_wall_s'],positive=True) and all(p.finite(x) for x in (r['timing']['rollout_parent_process_cpu_s'],rec['joined_child_cpu_s'],rec['collector_parent_cpu_s'],rec['process_tree_cpu_s'])),'observed timing/CPU missing or nonnumeric')
    return dict(j,status='COMPLETED_WINDOW',replay_sha256={n:p.sha(case/'replay'/n) for n in p.MEMBERS},window_time_hex=times,
        nominal_duration_s=j['duration'],actual_sim_delta_s=delta,
        rollout_wall_s=r['timing']['rollout_wall_s'],e2e_wall_s=rec['full_child_wall_s'],rollout_parent_cpu_s=r['timing']['rollout_parent_process_cpu_s'],
        kernel_parent_workers_cpu_s=rec['joined_child_cpu_s'],collector_cpu_s=rec['collector_parent_cpu_s'],collector_plus_kernel_cpu_s=rec['process_tree_cpu_s'],
        rss_estimates_separate={'collector_kernel':rss(rec['resources']),'kernel_workers':rss(r['resources'])},evidence_sha256={str(x):p.sha(x) for x in evidence})
def run(a,p,v):
    require(sys.platform=='linux' and sys.flags.isolated and sys.dont_write_bytecode,'fresh Linux -I -B required')
    require(not a.output.exists() and a.output.is_relative_to(OUTPUTS),'new training_outputs path required')
    require(len(a.cpus)==6 and len(set(a.cpus))==6 and set(a.cpus)<=os.sched_getaffinity(0),'six explicit available physical P-core CPU choices required')
    topology={n:{k:(Path(f'/sys/devices/system/cpu/cpu{n}/topology')/k).read_text().strip() for k in ('physical_package_id','core_id','thread_siblings_list')} for n in a.cpus}
    require(len({(r['physical_package_id'],r['core_id']) for r in topology.values()})==6,'CPU list repeats physical cores')
    dep=read(a.deployment_manifest); relative=Path(__file__).resolve().relative_to(PROJECT).as_posix()
    require(relative in dep.get('extra_paths',[]),'matrix script must be explicit finalized deployment.extra_paths')
    verified=v.verify_deployment(PROJECT,dep,supplied_commit=a.implementation_commit); ref=v._verify_export(a.reference)
    campaign,serial=baseline(a,p); targets={}
    for kernel,path in (('R0',a.r0_migration_approval),('OPT',a.opt_migration_approval)):
        identities=[x['target_identity'] for x in read(path)['migrations']]
        require(len({v.digest(x) for x in identities})==1,'migration target not unique')
        targets[kernel]=identities[0]['implementation_sha256']
    require(targets==campaign['target_execution_sha256'],'serial and requested target execution differ')
    corpus=read(a.corpus_manifest)
    for w in IDS:
        from performance.windows import validate_manifest
        validate_manifest(corpus,a.corpus_sha256,w,10 if a.mode=='parallel' else 60)
    protected=[a.reference,a.serial_campaign,a.corpus_manifest.parent,*[Path(x['source_attempt']).resolve() for x in corpus['sources'].values()]]
    require(all(not a.output.is_relative_to(x) and not x.is_relative_to(a.output) for x in protected),'output overlaps frozen/baseline input')
    jobs=plan(a.mode,a.inner_workers); a.output.mkdir(parents=True); rows=[]; status='RUNNING'; error=None
    inputs={str(x):p.sha(x) for x in (a.corpus_manifest,a.deployment_manifest,a.r0_migration_approval,a.opt_migration_approval,a.serial_campaign/'campaign.json')}
    meta=dict(schema='fixed_window_matrix_v1',mode=a.mode,jobs=jobs,matrix_argv=list(sys.orig_argv),source_args={k:str(x) if isinstance(x,Path) else x for k,x in vars(a).items()},serial_baseline_raw=serial,input_sha256=inputs,driver_sha256=p.sha(__file__),topology=topology,started_utc=p.utc(),repetitions_per_window_backend=1 if a.mode=='parallel' else 3,automatic_backend_choice=None)
    save(a.output/'matrix.json',meta)
    save(a.output/'source_preflight.json',dict(deployment=verified,reference=ref))
    try:
        for j in jobs:
            case=a.output/j['relative_case']; case.parent.mkdir(parents=True)
            rec=p.collect(case.parent/'collector',project=PROJECT,cpu_affinity=a.cpus[:max(1,j['workers'])],cli_argv=command(a,j,case),kind='window',budget=a.budget,python=a.python)
            began=time.process_time()
            if rec['status']=='PERFORMANCE_TIMEOUT': row=dict(j,status='PERFORMANCE_TIMEOUT',collector_record=str(case.parent/'collector/collector.json'),full_cpu_complete=False,e2e_wall_s=rec['full_child_wall_s'],observed_reaped_child_cpu_lower_bound_s=rec['observed_reaped_child_cpu_lower_bound_s'],evidence_sha256={str(case.parent/'collector/collector.json'):p.sha(case.parent/'collector/collector.json')})
            else:
                row=validate(a,j,case,rec,p,campaign,targets)
                same=serial[j['window']][0] if a.mode=='parallel' else next((r for r in rows if r['window']==j['window'] and r['status']=='COMPLETED_WINDOW'),None)
                if same: require(row['replay_sha256']==same['replay_sha256'] and row['window_time_hex']==same['window_time_hex'],'semantic replay/timeline difference')
            row['collector_validation_parent_cpu_s']=time.process_time()-began
            save(case.parent/'row.json',row); rows.append(row)
        status='FIXED_MATRIX_FINISHED_WITH_TIMEOUTS' if any(r['status']=='PERFORMANCE_TIMEOUT' for r in rows) else 'FIXED_MATRIX_FINISHED'
    except BaseException as exc:
        status='STOPPED_ON_INTEGRITY_OR_PHYSICS_ERROR'; error={'type':type(exc).__name__,'message':str(exc),'failed_job':j}
    finally:
        try:
            v.verify_deployment(PROJECT,dep,supplied_commit=a.implementation_commit); v._verify_export(a.reference,ref)
            require(all(p.sha(x)==s for x,s in inputs.items()),'input bytes changed')
            save(a.output/'source_finish.json',dict(source_unchanged=True,driver_sha256=p.sha(__file__)))
        except BaseException as exc: status='STOPPED_ON_SOURCE_ERROR'; error={'type':type(exc).__name__,'message':str(exc)}
        child_complete=len(rows)==len(jobs) and all(r['status']=='COMPLETED_WINDOW' for r in rows)
        save(a.output/'completion.json',dict(status=status,error=error,rows=rows,finished_utc=p.utc(),collector_process_cpu_through_final_snapshot_s=time.process_time(),
            full_kernel_children_cpu_complete=child_complete,full_kernel_children_cpu_s=sum(r['kernel_parent_workers_cpu_s'] for r in rows) if child_complete else None,
            full_60s_R0_OPT_oracle='separate pending comparison' if a.mode=='sustained' else '10s serial R0/OPT oracle verified',
            timing_scope='matrix collector CPU through final snapshot, final write/print tail unobserved; do not sum with per-job collector CPU',**aggregate(rows,serial,a.mode)))
    display(aggregate(rows,serial,a.mode)); return 0 if status=='FIXED_MATRIX_FINISHED' else 1 if status=='FIXED_MATRIX_FINISHED_WITH_TIMEOUTS' else 2
def display(report):
    print('backend | valid | windows | geometric OPT-serial 10s rollout ratio')
    for row in report['table']: print(row['backend'],'|',row['valid'],'|',len(row['windows']),'|',row['geometric_serial_OPT_rollout_ratio'])
def self_check(a):
    fake=argparse.Namespace(corpus_manifest='/SYNTHETIC/corpus.json',corpus_sha256='1'*64,reference='/SYNTHETIC/reference',deployment_manifest='/SYNTHETIC/deployment.json',implementation_commit='0'*40,opt_migration_approval='/SYNTHETIC/OPT.json')
    parallel=plan('parallel'); sustained=plan('sustained',2)
    require(len(parallel)==24 and len(sustained)==18 and len({x['relative_case'] for x in parallel})==24,'matrix shape')
    require([(x['workers'],x['repeat']) for x in parallel[:4]]==[(1,1),(2,1),(4,1),(6,1)] and [x['repeat'] for x in sustained[:3]]==[1,2,3],'matrix order')
    commands=[command(fake,j,OUTPUTS/'SYNTHETIC_NOT_EXECUTED'/j['relative_case']) for j in parallel+sustained]
    require(all(x[0]==str(PROJECT/'scripts/server/run_rotating_slots_performance_windows.py') and x[1]=='window' and '--output' in x for x in commands),'fixed argv paths')
    serial={w:[dict(window=w,kernel=k,nominal_duration_s=10,actual_sim_delta_s=10.,rollout_wall_s=10.,e2e_wall_s=11.,kernel_parent_workers_cpu_s=3.,collector_cpu_s=1.,collector_plus_kernel_cpu_s=4.) for k in ('R0','OPT') for _ in range(3)] for w in IDS}
    rows=[dict(j,status='COMPLETED_WINDOW',nominal_duration_s=10,actual_sim_delta_s=10.,rollout_wall_s=5.,e2e_wall_s=6.,kernel_parent_workers_cpu_s=3.,collector_cpu_s=1.,collector_plus_kernel_cpu_s=4.) for j in parallel]
    report=aggregate(rows,serial,'parallel'); require(all(t['valid'] for t in report['table']) and math.isclose(report['table'][1]['geometric_serial_OPT_rollout_ratio'],2.),'synthetic aggregation')
    long_rows=[dict(j,status='COMPLETED_WINDOW',nominal_duration_s=60,actual_sim_delta_s=(60.2,60.4,60.6)[j['repeat']-1],rollout_wall_s=(12.04,6.04,6.06)[j['repeat']-1],e2e_wall_s=13.,kernel_parent_workers_cpu_s=3.,collector_cpu_s=1.,collector_plus_kernel_cpu_s=4.) for j in sustained]
    long_report=aggregate(long_rows,serial,'sustained'); chosen=next(t for t in long_report['table'] if t['backend']==2)
    require(chosen['valid'] and chosen['actual_wall_per_sim_covered_windows']==6 and math.isclose(chosen['geometric_actual_wall_per_sim_s'],.1) and all(math.isclose(w['actual_wall_per_sim_median_s'],.1) and w['nominal_duration_s']==60 and all(d>60 for d in w['actual_sim_delta_s']) for w in chosen['windows']),'synthetic deferral: median per-sample wall/actual-sim, not nominal or ratio of medians')
    ast.parse(Path(__file__).read_text()); path=a.self_check_output or Path(__file__).with_name('self_check.json')
    save(path,dict(status='SELF_CHECK_PASSED',script=str(Path(__file__).resolve()),script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),synthetic_only=True,physics_executed=False,server_executed=False,parallel_jobs=24,sustained_jobs=18,commands=commands,synthetic_aggregate=report,synthetic_sustained_aggregate=long_report))
    print(json.dumps(dict(status='SELF_CHECK_PASSED',report=str(path),parallel_jobs=24,sustained_jobs=18,physics_executed=False))); return 0
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('plan','self-check','parallel','sustained','summarize'))
    for name in ('corpus-manifest','reference','deployment-manifest','r0-migration-approval','opt-migration-approval','serial-campaign','output','self-check-output'): parser.add_argument('--'+name,type=Path)
    for name in ('corpus-sha256','implementation-commit'): parser.add_argument('--'+name)
    parser.add_argument('--cpus',type=int,nargs='+'); parser.add_argument('--inner-workers',type=int,choices=(0,*WORKERS))
    parser.add_argument('--budget',type=float,default=1800.); parser.add_argument('--python',default=sys.executable); parser.add_argument('--plan-for',choices=('parallel','sustained'),default='parallel')
    a=parser.parse_args()
    if a.mode=='self-check': return self_check(a)
    if a.mode=='summarize':
        meta=read(a.output/'matrix.json'); done=read(a.output/'completion.json')
        sys.path.insert(0,str(PROJECT/'terminal_marl/core/rotating_slots')); from performance.paired import sha
        for row in done['rows']:
            for path,value in row.get('evidence_sha256',{}).items(): require(sha(path)==value,'saved result/replay evidence changed')
        report=aggregate(done['rows'],meta['serial_baseline_raw'],meta['mode']); print(json.dumps(dict(status=done['status'],**report),indent=2)); display(report); return 0
    require(all(getattr(a,k) is not None for k in ('corpus_manifest','corpus_sha256','reference','deployment_manifest','implementation_commit','r0_migration_approval','opt_migration_approval','serial_campaign','cpus','output')),'all explicit input/CPU/output arguments required')
    for name in ('output','corpus_manifest','reference','deployment_manifest','r0_migration_approval','opt_migration_approval','serial_campaign'): setattr(a,name,getattr(a,name).resolve())
    require(math.isfinite(a.budget) and a.budget>0,'positive finite wall budget required')
    if a.mode=='plan':
        jobs=plan(a.plan_for,a.inner_workers); print(json.dumps(dict(jobs=jobs,commands=[command(a,j,a.output/j['relative_case']) for j in jobs],output_exists=a.output.exists(),executed=False),indent=2)); return 0
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','OMP_THREAD_LIMIT'): os.environ[n]='1'
    old=os.getpriority(os.PRIO_PROCESS,0); require(old<=10,'inherited nice exceeds10')
    if old<10: os.nice(10-old)
    sys.path.insert(0,str(PROJECT/'terminal_marl/core/rotating_slots')); from performance import paired,provenance
    require(tuple(paired.IDS)==IDS,'frozen window order changed')
    return run(a,paired,provenance)
if __name__=='__main__': raise SystemExit(main())
