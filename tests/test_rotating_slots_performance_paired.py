"""Paired protocol/collector tests; synthetic summary fixtures are not traffic evidence."""
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys

import pytest

ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent/'rotating_slots_performance_contract_v1_0').is_dir())
sys.path.insert(0,str(ROOT/'terminal_marl/core/rotating_slots'))
from performance import paired as p


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def test_plan_fixed_order_coverage_and_repeat_pairing():
    jobs=p.plan()
    assert len(jobs)==36 and [j['window'] for j in jobs[::6]]==list(p.IDS)
    assert [(j['repeat'],j['kernel']) for j in jobs[:6]]==list(p.ORDER)
    assert len({j['relative_case'] for j in jobs})==36
    assert [j['window'] for j in p.plan(('W180','W71-A'))[::6]]==['W71-A','W180']
    with pytest.raises(p.PairedError): p.plan(('W60','W60'))
    with pytest.raises(p.PairedError): p.plan(('unknown',))
    with pytest.raises(p.PairedError): p.plan(group=3)


def raw(r0,opt):
    return [{'kernel':k,'repeat':i,'rollout_wall_s':(r0 if k=='R0' else opt)[i-1],
             'rollout_parent_cpu_s':(r0 if k=='R0' else opt)[i-1]*.8} for i,k in p.ORDER]


def test_noise_boundary_and_exact_thresholds():
    assert not p.noise([1.,1.25,1.]) and p.noise([1.,1.2500001,1.])
    groups={w:raw([1.]*3,[.5]*3) for w in p.IDS}
    result=p.aggregate(groups,full_coverage=True)
    assert result['component_gate']=='PASS' and result['geometric_median_wall_speedup']==2.
    assert all(row['R0_p95_wall_s']==row['R0_max_wall_s']==1. for row in result['windows'])
    assert result['performance_status']=='NOT_QUALIFIED' and result['human_status']=='HUMAN_PENDING'
    groups={w:raw([4.]*3,[1.]*3) for w in p.IDS}
    groups[p.IDS[0]]=raw([1.]*3,[1.1]*3)
    assert p.aggregate(groups,full_coverage=True)['component_gate']=='PASS'
    groups[p.IDS[0]]=raw([1.]*3,[1.1000001]*3)
    assert p.aggregate(groups,full_coverage=True)['component_gate']=='NOT_PASSED'
    assert p.aggregate({p.IDS[0]:raw([10.]*3,[1.]*3)},full_coverage=False)['component_gate']=='NOT_PASSED'


def fixture_campaign(root, *, windows=p.IDS, diagnostic=False):
    cursor={'original_target':3600.,'phase':'loop_start','outer_phase':'arrival_window','shim_source_sha256':'c'*64}
    frozen={w:{'cut':{'actual_sim_time':300.,'checkpoint_sha256':'d'*64},'cursor':cursor,
               'identity':{'implementation_sha256':'s'*64,'schema':'fixture_only'},'workload':{'fixture':'synthetic-summary-only'}} for w in p.IDS}
    campaign={'schema':'ft_rs_serial_paired_campaign_v1','duration_s':10,'repeat_order':[[r,k] for r,k in p.ORDER],
        'windows':list(windows),'diagnostic_subset':diagnostic,'status':'COMPLETED_PAIRED',
        'corpus_sha256':'a'*64,'logical_cpu':0,'implementation_commit':'b'*40,'semantic_fingerprint':{'fixture':True},
        'target_execution_sha256':{'R0':'r'*64,'OPT':'o'*64},'frozen_windows':frozen}
    write(root/'campaign.json',campaign)
    return campaign


def synthetic_result(campaign, job, wall):
    frozen=campaign['frozen_windows'][job['window']]
    cache={'all_named_acceleration_caches_empty':True,'module_caches_cleared':False,
           'world_entries':{'_q_cache':0,'_energy_cache':0,'_reference_templates':0},
           'module_numerical_caches':{name:{'currsize':0} for name in ('trajectory.poly_bounds','energy._integrable_plan',
                'energy._segment_table','safety._NUMERIC_CERTIFICATES','safety._PLAN_CERTIFICATES')}}
    return {'schema':'ft_rs_common_window_v1_0','fixture':'synthetic-orchestration-only-no-physics',
        'status':'COMPLETED_WINDOW','failure':None,'window':job['window'],'duration_s':10,'mode':'benchmark',
        'inner_workers':0,'execution_backend':job['kernel']+'_SERIAL','corpus_sha256':campaign['corpus_sha256'],
        'workload':frozen['workload'],'original_advance_target':3600.,'outer_phase':'arrival_window',
        'source_identity':{'kernel':job['kernel'],'runner_execution_sha256':campaign['target_execution_sha256'][job['kernel']],
            'semantic_fingerprint':campaign['semantic_fingerprint'],'implementation_commit':campaign['implementation_commit']},
        'cache_mode':'cold','cold_cache_evidence':cache,'start_sim_time':300.,'actual_stop_sim_time':310.,
        'actual_sim_delta_s':10.,'nominal_stop_sim_time':310.,'simulation_deferral_s':0.,
        'physical_outcome':{'failed':False,'episode_complete':False},
        'timing':{'rollout_wall_s':wall,'rollout_parent_process_cpu_s':wall*.8}}


def fixture_group(root,campaign,window,group=1,r0=(2.,2.,2.),opt=(.5,.5,.5)):
    jobs=p.plan((window,),group)
    load={'utc':'fixture-only','load_average':[0.,0.,0.],'memory_kib':{'MemAvailable':1}}
    metadata={'window':window,'group':group,'jobs':jobs,'load_before':load,'load_after':load,
              'status':'NOISY' if p.noise(r0) or p.noise(opt) else 'COMPLETED_GROUP'}
    write(root/f'windows/{window}/group-{group}/group.json',metadata)
    for job in jobs:
        case=root/job['relative_case']; logs=case.parent/'collector'; logs.mkdir(parents=True)
        frozen=campaign['frozen_windows'][window]
        identity={'corpus_sha256':campaign['corpus_sha256'],'window':window,'selected_cut':frozen['cut'],
            'source_identity':frozen['identity'],'target_identity':dict(frozen['identity'],implementation_sha256=campaign['target_execution_sha256'][job['kernel']]),
            'source_cursor':frozen['cursor'],'runtime_cursor':frozen['cursor']}
        wall=(r0 if job['kernel']=='R0' else opt)[job['repeat']-1]
        result=synthetic_result(campaign,job,wall)
        write(case/'result.json',result); write(case/'input_identity.json',identity)
        for name in (*p.MEMBERS,'replay.json'):
            target=case/'replay'/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(b'synthetic summary fixture\n')
        env={'pid':1234,'isolated':True,'dont_write_bytecode':True,'affinity':[0],'nice':10,
             'thread_environment':{name:'1' for name in p.THREADS},'core_modules_before_cli':[]}
        write(logs/'runtime_environment.json',env)
        guard={'state':'DISARMED','pid':1234,'expiry_signal':getattr(signal,'SIGALRM',14),'budget_seconds':p.BUDGET}
        write(logs/'deadlines/deadline_fixture.json',guard)
        for name,data in (('rss_samples.jsonl','{}\n'),('numericexitcode.txt','0\n'),('stdout.log','fixture\n'),('stderr.log','')):
            (logs/name).write_text(data,encoding='utf8')
        parent=.01; child=.1; post=.001
        record={'job':job,'pid':1234,'returncode':0,'status':'COMPLETED_WINDOW','guard_records':[guard],
            'collector_parent_cpu_s':parent,'joined_child_cpu_s':child,'process_tree_cpu_s':parent+child,
            'collector_validation_parent_cpu_s':post,'collector_parent_cpu_including_validation_s':parent+post,
            'process_tree_cpu_including_validation_s':parent+post+child,
            'resources':{'cpu_complete':True,'errors':[],'worker_pids':[1234],'sampled_peak_simultaneous_rss_bytes':1,
                'rss_sample_count':1,'rss_full_series_path':str(logs/'rss_samples.jsonl')},
            'result_sha256':p.sha(case/'result.json'),'input_identity_sha256':p.sha(case/'input_identity.json'),
            'replay_sha256':{name:p.sha(case/'replay'/name) for name in p.MEMBERS},
            'replay_index_sha256':p.sha(case/'replay/replay.json'),
            'window_time_hex':{name:float(result[name]).hex() for name in ('start_sim_time','actual_stop_sim_time','actual_sim_delta_s')}}
        record['evidence_files_sha256']={path.relative_to(logs).as_posix():p.sha(path) for path in sorted(logs.rglob('*')) if path.is_file()}
        write(logs/'collector.json',record)
    records=[p.read(path) for path in root.glob('windows/**/collector/collector.json')]
    child_sum=sum(record['joined_child_cpu_s'] for record in records)
    campaign.update(full_child_cpu_complete=True,started_kernel_child_count=len(records),
        collector_process_lifetime_cpu_s=100.,collector_run_interval_cpu_s=10.,
        full_joined_kernel_children_cpu_s=child_sum,total_process_tree_cpu_s=100.+child_sum)
    write(root/'campaign.json',campaign)


def test_complete_persisted_coverage_and_component_only(tmp_path):
    campaign=fixture_campaign(tmp_path)
    for w in p.IDS: fixture_group(tmp_path,campaign,w)
    result=p.summarize(tmp_path)
    assert result['errors']==[] and result['component_gate']=='PASS'
    assert len(result['windows'])==6 and result['full_expected_initial_job_count']==36
    assert result['performance_status']=='NOT_QUALIFIED'


def test_noisy_whole_group_retry_never_picks_faster_group(tmp_path):
    campaign=fixture_campaign(tmp_path)
    for w in p.IDS:
        fixture_group(tmp_path,campaign,w,r0=(2.,2.,3.) if w==p.IDS[0] else (2.,2.,2.))
    fixture_group(tmp_path,campaign,p.IDS[0],2,r0=(6.,6.,6.),opt=(2.,2.,2.))
    result=p.summarize(tmp_path)
    assert result['errors']==[] and result['retained_groups'][p.IDS[0]][0]['status']=='NOISY'
    assert result['windows'][0]['R0_median_wall_s']==6.
    assert result['windows'][0]['OPT_median_wall_s']==2.


def test_second_noisy_group_cannot_pass(tmp_path):
    campaign=fixture_campaign(tmp_path)
    for w in p.IDS: fixture_group(tmp_path,campaign,w,r0=(2.,2.,3.) if w==p.IDS[0] else (2.,2.,2.))
    fixture_group(tmp_path,campaign,p.IDS[0],2,r0=(6.,6.,9.),opt=(2.,2.,2.))
    result=p.summarize(tmp_path)
    assert result['errors']==[] and result['component_gate']=='NOT_PASSED'
    assert result['windows'][0]['status']=='NOISY'


@pytest.mark.parametrize('mutation',['missing','duplicate','replay','profile','cpu','environment','guard','timeline','cold','stopped'])
def test_persisted_missing_duplicate_tamper_errors_cannot_pass(tmp_path,mutation):
    campaign=fixture_campaign(tmp_path,windows=('W71-A',),diagnostic=True)
    fixture_group(tmp_path,campaign,'W71-A')
    job=p.plan(('W71-A',))[0]; case=tmp_path/job['relative_case']; collector=case.parent/'collector/collector.json'
    if mutation=='missing': (case/'result.json').unlink()
    elif mutation=='duplicate': write(case.parent.parent/'EXTRA/collector/collector.json',{})
    elif mutation=='replay': (case/'replay/states.jsonl').write_bytes(b'changed')
    elif mutation=='stopped': campaign['status']='STOPPED_ON_ERROR'; write(tmp_path/'campaign.json',campaign)
    elif mutation in ('profile','timeline','cold'):
        result=p.read(case/'result.json')
        if mutation=='profile': result['mode']='profile'
        if mutation=='timeline': result['actual_stop_sim_time']=310.2
        if mutation=='cold': result['cold_cache_evidence']['world_entries']['_energy_cache']=1
        write(case/'result.json',result)
        record=p.read(collector); record['result_sha256']=p.sha(case/'result.json'); write(collector,record)
    else:
        record=p.read(collector)
        if mutation=='cpu': record['joined_child_cpu_s']=None
        if mutation=='environment':
            env=p.read(case.parent/'collector/runtime_environment.json'); env['affinity']=[1]
            write(case.parent/'collector/runtime_environment.json',env)
        if mutation=='guard': record['guard_records'][0]['pid']=99
        write(collector,record)
    result=p.summarize(tmp_path)
    assert result['component_gate']=='INVALID_OR_INCOMPLETE' and result['errors']


def test_diagnostic_subset_cannot_pass_and_extra_retry_rejected(tmp_path):
    campaign=fixture_campaign(tmp_path,windows=('W71-A',),diagnostic=True)
    fixture_group(tmp_path,campaign,'W71-A')
    assert p.summarize(tmp_path)['component_gate']=='NOT_PASSED'
    fixture_group(tmp_path,campaign,'W71-A',2)
    assert p.summarize(tmp_path)['component_gate']=='INVALID_OR_INCOMPLETE'


@pytest.mark.parametrize('case',['absent','present-empty','not-present-empty','warm','unknown'])
def test_known_optional_root_cache_strict_compatibility(tmp_path,case):
    campaign=fixture_campaign(tmp_path,windows=('W71-A',),diagnostic=True)
    fixture_group(tmp_path,campaign,'W71-A')
    for job in p.plan(('W71-A',)):
        attempt=tmp_path/job['relative_case']; result=p.read(attempt/'result.json')
        cache=result['cold_cache_evidence']['module_numerical_caches']
        if case=='present-empty': cache['energy._ROOT_CACHE']={'present':True,'entries':0}
        if case=='not-present-empty': cache['energy._ROOT_CACHE']={'present':False,'entries':0}
        if case=='warm': cache['energy._ROOT_CACHE']={'present':True,'entries':1}
        if case=='unknown': cache['unreviewed.cache']={'entries':0}
        write(attempt/'result.json',result)
        record_path=attempt.parent/'collector/collector.json'; record=p.read(record_path)
        record['result_sha256']=p.sha(attempt/'result.json'); write(record_path,record)
    result=p.summarize(tmp_path)
    assert bool(result['errors']) is (case in ('warm','unknown'))


def test_guard_classification_requires_exact_armed_signal_pid_and_deadline():
    alarm=getattr(signal,'SIGALRM',14)
    record={'state':'ARMED','pid':123,'expiry_signal':alarm,'deadline_monotonic_seconds':2.}
    assert p.classify(-alarm,123,[record],3.)=='PERFORMANCE_TIMEOUT'
    assert p.classify(-alarm,124,[record],3.)=='IMPLEMENTATION_OR_STARTUP_ERROR'
    assert p.classify(-alarm,123,[record],1.)=='IMPLEMENTATION_OR_STARTUP_ERROR'
    assert p.classify(7,123,[record],3.)=='IMPLEMENTATION_OR_STARTUP_ERROR'


@pytest.mark.parametrize('kind,code',[('success',0),('nonzero',7)])
def test_real_small_collector_child_isolated_native_one_full_cpu(tmp_path,kind,code):
    cpu=min(__import__('os').sched_getaffinity(0)) if sys.platform=='linux' else 0
    result=p.collect(tmp_path/'collector',project=ROOT,logical_cpu=cpu,kind=kind,budget=3.)
    assert result['returncode']==code and (tmp_path/'collector/numericexitcode.txt').read_text().strip()==str(code)
    assert result['resources']['cpu_complete'] and result['joined_child_cpu_s']>=0
    assert result['process_tree_cpu_s']==result['collector_parent_cpu_s']+result['joined_child_cpu_s']
    assert result['resources']['rss_sample_count']>=1
    assert 'upper bound' in result['rss_scope'] and 'lower bounds' in result['rss_scope']
    env=p.read(tmp_path/'collector/runtime_environment.json')
    assert env['isolated'] and env['dont_write_bytecode'] and env['core_modules_before_cli']==[]
    assert all(value=='1' for value in env['thread_environment'].values())


def test_failed_spawn_has_no_fabricated_numeric_exit(tmp_path):
    result=p.collect(tmp_path/'collector',project=ROOT,logical_cpu=0,kind='success',python=tmp_path/'missing_python',budget=3.)
    assert result['returncode'] is None and result['no_child_exitcode']
    assert not (tmp_path/'collector/numericexitcode.txt').exists()


@pytest.mark.skipif(sys.platform!='linux',reason='requires actual Linux SIGALRM kernel timer')
def test_real_small_collector_kernel_deadline(tmp_path):
    cpu=min(__import__('os').sched_getaffinity(0))
    result=p.collect(tmp_path/'collector',project=ROOT,logical_cpu=cpu,kind='deadline',budget=.15)
    assert result['status']=='PERFORMANCE_TIMEOUT' and result['returncode']==-signal.SIGALRM
    assert result['resources']['cpu_complete']


@pytest.mark.skipif(sys.platform!='linux',reason='real orchestration needs Linux affinity and kernel deadlines')
def test_real_six_child_orchestration_persistence_and_offline_summarize(tmp_path,monkeypatch):
    """Real controller/children; synthetic emitter does not exercise physics/CLI admission."""
    from argparse import Namespace
    import os
    template_root=tmp_path/'templates'
    campaign=fixture_campaign(template_root,windows=('W71-A',),diagnostic=True)
    fixture_group(template_root,campaign,'W71-A')
    frozen=campaign['frozen_windows']['W71-A']
    corpus={'semantic_fingerprint':campaign['semantic_fingerprint'],
            'windows':[{'window':'W71-A','case':'fixture','cut':frozen['cut'],
                        'validation':{'cursor':frozen['cursor'],'identity':frozen['identity']}}],
            'sources':{'fixture':{'workload':frozen['workload']}}}
    monkeypatch.setattr(p,'preflight',lambda *args:(corpus,campaign['target_execution_sha256'],{'scope':'synthetic-orchestration-only'}))
    emitter=tmp_path/'synthetic_emitter.py'
    emitter.write_text('''import sys
sys.dont_write_bytecode=True
import json,pathlib
source,target=map(pathlib.Path,sys.argv[1:3])
assert not target.exists()
target.mkdir(parents=True)
for name in ("result.json","input_identity.json"):
 (target/name).write_bytes((source/name).read_bytes())
(target/"replay").mkdir()
for path in (source/"replay").iterdir(): (target/"replay"/path.name).write_bytes(path.read_bytes())
''',encoding='utf8')
    def command(args,project,job,case):
        source=template_root/job['relative_case']
        return [str(emitter),str(source),str(case)]
    monkeypatch.setattr(p,'window_command',command)
    for name in p.THREADS: monkeypatch.setenv(name,'1')
    args=Namespace(output=tmp_path/'actual-controller',logical_cpu=min(os.sched_getaffinity(0)),
                   diagnostic_windows=['W71-A'],implementation_commit=campaign['implementation_commit'],
                   corpus_sha256=campaign['corpus_sha256'])
    # Rebind fixture expected affinity to the actual explicitly selected CPU.
    # The real child wrapper produces that environment; emitter writes no logs.
    result=p.paired(args,ROOT)
    assert result['errors']==[] and result['component_gate']=='NOT_PASSED' and result['diagnostic_subset']
    persisted=p.read(args.output/'campaign.json')
    assert persisted['status']=='COMPLETED_PAIRED' and persisted['started_kernel_child_count']==6
    assert persisted['full_child_cpu_complete'] and persisted['total_process_tree_cpu_s']>=persisted['full_joined_kernel_children_cpu_s']
    assert p.summarize(args.output)['errors']==[]
    collectors=[p.read(path) for path in args.output.glob('windows/**/collector/collector.json')]
    assert len(collectors)==6 and all(row['returncode']==0 and row['resources']['cpu_complete'] for row in collectors)
    assert all(row['guard_records'][0]['state']=='DISARMED' for row in collectors)
    assert all(row['command'][1:3]==['-I','-B'] for row in collectors)
