"""Four fixed 600s evaluations of the existing online-v2 model, without fitting.

All algorithm imports come from frozen96c. This driver owns only preparation,
absolute world.advance boundaries, subprocess budgets, saved evidence and summary.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).resolve()
AGGREGATOR = DRIVER.with_name('summarize_online_v2.py')
CORE_COMMIT = '96c00d49ce8ee9fb7f570e88cec545bfb6a46a5d'
CORE_HASH = '2afe5eec33f403479dc548dcc106d004a15c3329cc8b591113391b9fadfc7e3f'
MODEL_HASH = 'dba25b3393a5580bdc3971d9adf7d6e3cff6360404fdb08bc8fbb6f0b36e3b48'
NORMALIZER_HASH = '1e06b066dad2d657acbe16de6706b180dd4f4ed8ba1bc106cefcc2a3d4f4e7be'
ARMS = ('heuristic_reduced', 'learned_reduced')
CASES = dict(medium=dict(root_id='s3r23',seed=7324,scenario_profile='medium24_rate150_v1',rate_per_hour=150,cpu=12,initial_airborne=24,requests=141),
             high=dict(root_id='s3r24',seed=7325,scenario_profile='stress36_rate180_v1',rate_per_hour=180,cpu=14,initial_airborne=36,requests=167))
CAPS = dict(preflight_tests=900., evaluation=45000., archive=900.)
TOTAL, COMPUTE_STOP, JOB_CAP = 46800., 45900., 21600.
THREAD_KEYS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')


def require(condition,message):
    if not condition:raise ValueError(message)


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_hash(path):
    value=sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def load_aggregator(path=AGGREGATOR):
    spec=importlib.util.spec_from_file_location('_online_v2_600_aggregate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def frozen_dependencies(checkout):
    """Called only after CLI parsing, before any project import."""
    checkout=Path(checkout).resolve();require(checkout.is_dir(),'frozen checkout not ready')
    os.chdir(checkout);sys.path.insert(0,str(checkout))
    os.environ.update({k:'1' for k in THREAD_KEYS});os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['PYTHONPATH']=str(checkout)+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    from terminal_marl.core.rotating_slots.stage3_learning import campaign, evaluation
    from terminal_marl.core.rotating_slots.stage3_learning.data import restore_root
    from terminal_marl.core.rotating_slots.stage3_learning.online_observer import OnlineObserver
    from terminal_marl.core.rotating_slots.stage2_campaign import atomic_json,digest,shared_fifo_admission,episode_metrics,save_checkpoint
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    from terminal_marl.core.rotating_slots.replay import ReplayBundle
    require(campaign.ROOT.resolve()==checkout and Path(evaluation.__file__).resolve().is_relative_to(checkout),
            'project modules imported outside frozen checkout')
    return SimpleNamespace(campaign=campaign,evaluation=evaluation,restore_root=restore_root,OnlineObserver=OnlineObserver,
        atomic_json=atomic_json,digest=digest,shared_fifo_admission=shared_fifo_admission,episode_metrics=episode_metrics,
        save_checkpoint=save_checkpoint,NormalSchedulerCallback=NormalSchedulerCallback,ReplayBundle=ReplayBundle,
        checkout=checkout,aggregate=load_aggregator().aggregate_decisions)


def matrix():return [(case,arm) for case in CASES for arm in ARMS]


def verify_core(frozen,d):
    require(frozen['source_commit']==CORE_COMMIT and frozen['source_hash']==CORE_HASH,'unexpected frozen core identity')
    require(len(frozen['source_files'])==85,'expected original 85-file frozen inventory')
    d.campaign.verify_sources(frozen)


def verify_model(parent):
    require(file_hash(parent/'model/model.pt')==MODEL_HASH,'frozen model hash mismatch')
    require(file_hash(parent/'model/feature_normalizer.json')==NORMALIZER_HASH,'frozen normalizer hash mismatch')
    card=load(parent/'model/model_card.json');binding=load(parent/'run_binding.generated.json')
    require(binding['source_hash']==CORE_HASH and binding['fingerprint']['source_commit']==CORE_COMMIT,'model execution binding changed')
    require(card['model_frozen'] and card['model_hash']==MODEL_HASH and card['parameter_count']==119555 and card['seed']==20261023,
            'unexpected model card')
    require(card['fingerprint']==binding['fingerprint'],'model card/binding fingerprint mismatch')
    require(load(parent/'model/feature_normalizer.json')['schema_id']==binding['fingerprint']['feature_schema_id'],
            'normalizer schema mismatch')
    return binding


def prepare(args,d):
    output=Path(args.output).resolve();model_parent=Path(args.model_parent).resolve();initial_parent=Path(args.initial_parent).resolve()
    require(not output.exists(),'fresh output directory required; no retry or reset')
    require(not output.is_relative_to(model_parent) and not output.is_relative_to(initial_parent),'output must be separate from input archives')
    frozen=load(model_parent/'source_freeze.json');verify_core(frozen,d);binding=verify_model(model_parent)
    require(binding['source_files']==frozen['source_files'],'model binding/frozen inventory mismatch')
    for parent in (model_parent,initial_parent):require(not parent.is_relative_to(output),'output cannot contain an input archive')
    output.mkdir(parents=True);d.atomic_json(output/'prepare.started.json',dict(started_at=d.campaign.utc_now()))
    try:
        (output/'model').mkdir()
        for name in ('model.pt','feature_normalizer.json','model_card.json'):
            shutil.copyfile(model_parent/'model'/name,output/'model'/name)
        shutil.copyfile(model_parent/'run_binding.generated.json',output/'run_binding.generated.json')
        shutil.copyfile(model_parent/'source_freeze.json',output/'source_freeze.json')
        roots={}
        for case,spec in CASES.items():
            original=initial_parent/'roots'/spec['root_id'];row=load(original/'root.json')
            manifest=load(original/'manifest.json');initial=load(original/'initial_state.json');checkpoint=load(original/'initial.checkpoint.json')
            require(all(row[k]==spec[k] for k in ('root_id','seed','scenario_profile')),'frozen root metadata mismatch')
            require(manifest['seed']==spec['seed'] and manifest['rate_per_hour']==spec['rate_per_hour'] and manifest['arrival_window']==3600. and
                    len(manifest['requests'])==spec['requests'],'frozen manifest mismatch')
            require(d.digest(manifest)==row['manifest_hash'] and d.digest(initial)==row['initial_state_hash'],'root content digest mismatch')
            payload=checkpoint['payload'];require(checkpoint['sha256']==d.digest(payload) and payload['identity']==row['identity'] and
                payload['time']==0. and payload['sink'] is None,'t0 checkpoint identity/content mismatch')
            require(sum(a['status'].startswith('HOLD_') for a in initial['aircraft'])==spec['initial_airborne'] and
                    sum(a['status']=='LANDED_SERVICE' for a in initial['aircraft'])==3,'initial aircraft counts changed')
            directory=output/'roots'/case;directory.mkdir(parents=True)
            for name in ('initial.checkpoint.json','manifest.json','initial_state.json'):
                shutil.copyfile(original/name,directory/name)
            row=deepcopy(row);row.update(root_id=case,ancestry_root_id=spec['root_id'],snapshot_time_s=0.,
                checkpoint_path=str(directory/'initial.checkpoint.json'),raw_snapshot_path=str(directory/'initial_state.json'))
            d.atomic_json(directory/'root.json',row)
            roots[case]=dict(spec,manifest_hash=row['manifest_hash'],initial_state_hash=row['initial_state_hash'],
                releases_through_60=sum(r['release_time']<=60. for r in manifest['requests']),
                releases_through_600=sum(r['release_time']<=600. for r in manifest['requests']))
        scope=dict(schema='stage3_online_v2_600_evaluation_v1',target_sim_seconds=600,boundary_seconds=5,checkpoint_seconds=60,
            telemetry='existing practical runtime 1s plus event/junction frames',cases=roots,arms=list(ARMS),formal_jobs=4,
            ordering='H then L serial per case; two cases parallel',job_wall_cap_s=JOB_CAP,
            budgets=dict(total_s=TOTAL,compute_stop_s=COMPUTE_STOP,phase_caps=CAPS),
            model_parent_readonly=str(model_parent),initial_parent_readonly=str(initial_parent),
            core_checkout=str(d.checkout),core_commit=CORE_COMMIT,core_source_hash=CORE_HASH,
            core_inventory_count=len(frozen['source_files']),driver_path=str(DRIVER),driver_sha256=file_hash(DRIVER),
            aggregator_path=str(AGGREGATOR),aggregator_sha256=file_hash(AGGREGATOR),
            model_sha256=MODEL_HASH,normalizer_sha256=NORMALIZER_HASH,model_fingerprint=deepcopy(binding['fingerprint']),
            original_model_protocol_id=binding['fingerprint']['evaluation_protocol_id'],
            interpretation='new user-authorized evaluation scope; original model fingerprint unchanged',
            no_training_or_labeling=True,independent_generalization='NOT_ESTABLISHED',
            physical_confound='medium/high differ in initial airborne counts as well as arrival rate')
        scope['evaluation_scope_id']=d.digest(scope)
        scope['prepared_file_sha256']={str(p.relative_to(output)):file_hash(p) for p in (output/'model').iterdir()}
        for p in (output/'roots').glob('*/*'):scope['prepared_file_sha256'][str(p.relative_to(output))]=file_hash(p)
        scope['prepared_file_sha256']['run_binding.generated.json']=file_hash(output/'run_binding.generated.json')
        scope['prepared_file_sha256']['source_freeze.json']=file_hash(output/'source_freeze.json')
        d.atomic_json(output/'evaluation_scope.json',scope)
        ledger=d.campaign.CampaignLedger(output/'budget_ledger.json')
        ledger.data.update(total_s=TOTAL,compute_stop_s=COMPUTE_STOP,phase_caps=dict(CAPS));ledger.persist();ledger.pause()
        d.atomic_json(output/'prepare.finished.json',dict(status='COMPLETED',evaluation_scope_id=scope['evaluation_scope_id']))
        return scope
    except Exception as exc:
        d.atomic_json(output/'prepare.finished.json',dict(status='FAILED',error_type=type(exc).__name__,error=str(exc)));raise


def verify(output,d):
    scope=load(output/'evaluation_scope.json')
    require(load(output/'prepare.finished.json')['status']=='COMPLETED','preparation incomplete')
    verify_core(load(output/'source_freeze.json'),d)
    require(Path(scope['core_checkout']).resolve()==d.checkout,'wrong isolated checkout')
    require(file_hash(DRIVER)==scope['driver_sha256'] and file_hash(AGGREGATOR)==scope['aggregator_sha256'],'driver/dependency changed after preparation')
    require(scope['target_sim_seconds']==600 and scope['boundary_seconds']==5 and scope['checkpoint_seconds']==60 and
        scope['job_wall_cap_s']==JOB_CAP and scope['budgets']==dict(total_s=TOTAL,compute_stop_s=COMPUTE_STOP,phase_caps=CAPS),'600s scope or budgets changed')
    require(scope['evaluation_scope_id']==d.digest({k:v for k,v in scope.items() if k not in ('evaluation_scope_id','prepared_file_sha256')}),'evaluation scope digest changed')
    require(scope['arms']==list(ARMS) and set(scope['cases'])==set(CASES) and scope['formal_jobs']==4,'evaluation matrix changed')
    require(all(all(scope['cases'][case][k]==v for k,v in spec.items()) for case,spec in CASES.items()),'fixed case metadata changed')
    for name,expected in scope['prepared_file_sha256'].items():require(file_hash(output/name)==expected,'prepared input changed: '+name)
    binding=verify_model(output);require(binding['fingerprint']==scope['model_fingerprint'],'model fingerprint changed')
    require((output/'budget_ledger.json').is_file(),'existing new-run ledger required; no reset')
    ledger=d.campaign.CampaignLedger(output/'budget_ledger.json')
    require(ledger.data['phase_caps']==CAPS and ledger.data['total_s']==TOTAL and ledger.data['compute_stop_s']==COMPUTE_STOP,'new-run ledger caps changed')
    return scope,binding,ledger


def run_job(output,case,arm,d,*,started=None):
    started=time.perf_counter() if started is None else started;cpu_started=time.process_time()
    scope,binding,_=verify(output,d);spec=CASES[case]
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{spec['cpu']})
    directory=output/'evaluation'/case/arm;directory.mkdir(parents=True)
    d.atomic_json(directory/'job.started.json',dict(started_at=d.campaign.utc_now(),pid=os.getpid(),case=case,arm=arm))
    row=load(output/'roots'/case/'root.json')
    cold=d.NormalSchedulerCallback('rh_cp_sat',planning_profile='downstream_service_v8')
    world=d.restore_root(row,cold);world.configure_practical_runtime(True)
    observer=d.OnlineObserver();setup={}
    scheduler=d.evaluation._scheduler(output,arm,binding,observer,case,setup)
    world.admission_callback=d.shared_fifo_admission
    holding,restore_observer=d.evaluation.observe_executed_holding(world)
    progress_path=output/'progress'/case/(arm+'.json');completed=0
    actual_scope=dict(evaluation_scope_id=scope['evaluation_scope_id'],case=case,cpu=spec['cpu'],target_sim_seconds=600,
                      core_commit=CORE_COMMIT,core_source_hash=CORE_HASH,driver_sha256=scope['driver_sha256'],aggregator_sha256=scope['aggregator_sha256'])
    def progress(phase):
        d.atomic_json(progress_path,dict(case=case,arm=arm,pid=os.getpid(),phase=phase,sim_time_s=world.time,
            completed_boundary_s=completed,target_sim_seconds=600,elapsed_wall_s=time.perf_counter()-started,
            callbacks=len(scheduler.records),observed_at=d.campaign.utc_now()))
    def normal(snapshot,port):return scheduler(port)
    provenance=deepcopy(binding['execution_provenance']);provenance['new_evaluation_scope']=actual_scope
    last=None
    try:
        d.save_checkpoint(directory/'checkpoint_0000.json',world,scheduler,row['identity']);progress('advance')
        for boundary in range(5,601,5):
            world.advance(float(boundary),normal,settle_only_at_end=True)
            if abs(world.time-boundary)>1.e-7 or world.failed or getattr(world,'terminated_invalid',False):break
            completed=boundary
            last=dict(status='COMPLETE_BOUNDARY',case=case,root_id=case,arm=arm,time=world.time,quality=d.episode_metrics(world),
                holding_energy_by_ring=dict(values_j=dict(holding),status='OBSERVED_EXECUTOR_ENERGY_DELTAS'),
                costs=d.evaluation._costs(scheduler,observer,time.perf_counter()-started,time.process_time()-cpu_started,
                    sum(r.get('dispatch_wall_s',0.) for r in world.dispatch_timings),setup_timings=setup,dispatch_records=world.dispatch_timings),
                dispatch_timings=world.dispatch_timings,shared_manifest_hash=row['manifest_hash'],initial_state_hash=row['initial_state_hash'],
                cpu=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                execution_provenance=provenance,evaluation_scope=actual_scope)
            if boundary%60==0:d.save_checkpoint(directory/f'checkpoint_{boundary:04d}.json',world,scheduler,row['identity'],boundary_summary=last)
            d.atomic_json(directory/f'boundary_{boundary:04d}.json',last);progress('boundary_saved')
        terminal=d.evaluation.terminal_observation(world);d.atomic_json(directory/'terminal_observation.json',terminal)
        if world.time%60!=0 or world.failed or getattr(world,'terminated_invalid',False):
            d.save_checkpoint(directory/'checkpoint_terminal.json',world,scheduler,row['identity'])
        result=deepcopy(last) if last else dict(case=case,root_id=case,arm=arm,time=0.,status='EVALUATION_PARTIAL')
        result.update(status='COMPLETE' if completed==600 else 'EVALUATION_PARTIAL',terminal_observation=terminal,
            evaluation_scope=actual_scope,execution_provenance=provenance)
        result['terminal_costs']=d.evaluation._costs(scheduler,observer,time.perf_counter()-started,time.process_time()-cpu_started,
            sum(r.get('dispatch_wall_s',0.) for r in world.dispatch_timings),setup_timings=setup,dispatch_records=world.dispatch_timings)
        progress('replay_export')
        record=world.to_dict();record.update(decisions=scheduler.records,stage3_trace=[],stage2_provenance=binding,
            tentative_predictions=[dict(aircraft_id=aid,as_of=world.time,plans=[p.to_dict() for p in plans]) for aid,plans in scheduler.provider.previous.items()])
        d.ReplayBundle.write(directory/'replay',record=record,code_commit=CORE_COMMIT,config_path=world.config.path,
                            command=str(DRIVER),seed=spec['seed'],eager=False)
        result['replay_path']=str(directory/'replay')
        result['all_callbacks']=d.aggregate(scheduler.records,result['terminal_costs'],world.dispatch_timings)
        d.atomic_json(directory/'result.json',result)
        d.atomic_json(directory/'job.finished.json',dict(status=result['status'],time=world.time,completed_boundary_s=completed,
            full_job_wall_s=time.perf_counter()-started,finished_at=d.campaign.utc_now()))
        progress('complete' if completed==600 else 'partial')
        return result
    except Exception as exc:
        d.atomic_json(directory/'job.finished.json',dict(status='FAILED',error_type=type(exc).__name__,error=str(exc),
            sim_time_s=world.time,completed_boundary_s=completed));raise
    finally:restore_observer()


def job_command(output,checkout,case,arm):
    return [sys.executable,str(DRIVER),'job','--output',str(output),'--checkout',str(checkout),'--case',case,'--arm',arm]


def run(output,d):
    with d.campaign.campaign_lock(output):
        scope,_,ledger=verify(output,d)
        require(not (output/'run.started.json').exists(),'run already attempted; no retry/reset')
        require(ledger.data.get('paused_at') is not None and ledger.data.get('active_phase') is None and not ledger.data['jobs'],'fresh paused evaluation ledger required')
        resource=d.campaign.resource_probe()
        require(all(spec['cpu'] in resource.get('free_physical_cores',[]) for spec in CASES.values()),'fixed CPU12/14 not both free; return to primary')
        d.atomic_json(output/'run.started.json',dict(started_at=d.campaign.utc_now(),resource_probe=resource,evaluation_scope_id=scope['evaluation_scope_id']))
        stop=threading.Event();receipts=[];errors=[]
        def one_case(case):
            values=[]
            for arm in ARMS:
                if stop.is_set():break
                require(ledger.remaining('evaluation')>=JOB_CAP+1.,'full equal 21600s arm cap unavailable; return to primary')
                receipt=d.campaign.run_subprocess(ledger,'evaluation',case+':'+arm,job_command(output,d.checkout,case,arm),output/'logs',cap=JOB_CAP,parallel=True)
                values.append(receipt)
                if receipt.get('timeout_s')!=JOB_CAP or receipt['status']!='completed' or receipt.get('exit_code')!=0:
                    stop.set();break
                result=load(output/'evaluation'/case/arm/'result.json')
                if result['status']!='COMPLETE':stop.set();break
            return values
        try:
            with ledger.phase('evaluation'):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures={pool.submit(one_case,case):case for case in CASES}
                    for future in as_completed(futures):
                        try:receipts.extend(future.result())
                        except Exception as exc:stop.set();errors.append(dict(case=futures[future],error_type=type(exc).__name__,error=str(exc)))
            complete=len(receipts)==4 and not stop.is_set() and not errors
            result=dict(status='COMPLETED' if complete else 'INCOMPLETE_RETURN_TO_PRIMARY',receipts=receipts,errors=errors,
                finished_at=d.campaign.utc_now(),no_automatic_retry=True)
            d.atomic_json(output/'run.finished.json',result)
            if not complete:raise RuntimeError('four-cell run incomplete; retain evidence and return to primary')
            return result
        finally:ledger.pause()


def summarize(output,d):
    scope,_,ledger=verify(output,d)
    require(ledger.data.get('paused_at') is not None and ledger.data.get('active_phase') is None,'summarize only after parent paused')
    require(not (output/'summary.json').exists(),'summary already published')
    cases={}
    for case in CASES:
        paths={arm:output/'evaluation'/case/arm for arm in ARMS}
        boundaries={arm:{int(p.stem.split('_')[-1]):p for p in path.glob('boundary_*.json')} for arm,path in paths.items()}
        common=max(set(boundaries[ARMS[0]])&set(boundaries[ARMS[1]]),default=0)
        rows={arm:load(boundaries[arm][common]) for arm in ARMS} if common else {}
        terminals={arm:load(path/'terminal_observation.json') if (path/'terminal_observation.json').exists() else dict(status='NOT_PUBLISHED') for arm,path in paths.items()}
        aggregates={}
        for arm,path in paths.items():
            result=load(path/'result.json') if (path/'result.json').exists() else {}
            decisions=path/'replay/decisions.jsonl'
            if decisions.exists():
                with decisions.open(encoding='utf8') as stream:records=[json.loads(line) for line in stream if line.strip()]
                aggregates[arm]=d.aggregate(records,result.get('terminal_costs',result.get('costs',{})),result.get('dispatch_timings',[]))
            else:aggregates[arm]=result.get('all_callbacks')
        trends={arm:[dict(time=t,quality=load(p)['quality']) for t,p in sorted(values.items()) if t%60==0] for arm,values in boundaries.items()}
        cases[case]=dict(common_completed_boundary_s=common,rows=rows,terminal_observations=terminals,all_callbacks=aggregates,
            quality_differences=d.evaluation.quality_differences(rows[ARMS[0]]['quality'],rows[ARMS[1]]['quality']) if rows else None,
            trends_every_60s=trends,status='COMPLETE' if common==600 else 'EVALUATION_PARTIAL')
    result=dict(schema='stage3_online_v2_600_summary_v1',evaluation_scope=scope,cases=cases,
        ledger=dict(phase_used=ledger.data['phase_used'],elapsed_s=ledger.elapsed),
        interpretation='One H/L pair per case; no independent generalization or statistical significance established; cases also differ in initial traffic.')
    d.atomic_json(output/'summary.json',result);return result


def main(argv=None):
    started=time.perf_counter();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run','job','summarize'))
    parser.add_argument('--output',type=Path,default=Path('/home/user/training_outputs/terminal/stage3_online_v2_600s_20261026'))
    parser.add_argument('--checkout',type=Path,default=PROJECT/'.worktrees/stage3-online-v2-frozen96c')
    parser.add_argument('--model-parent',type=Path,default=Path('/home/user/training_outputs/terminal/stage3_online_v2_20261025'))
    parser.add_argument('--initial-parent',type=Path,default=Path('/home/user/training_outputs/terminal/stage3_learning_v1_0_20261023'))
    parser.add_argument('--case',choices=tuple(CASES));parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args(argv)
    if args.stage=='job' and (args.case is None or args.arm is None):parser.error('job requires --case and --arm')
    output=args.output.resolve();d=frozen_dependencies(args.checkout)
    if args.stage=='prepare':result=prepare(args,d)
    elif args.stage=='run':result=run(output,d)
    elif args.stage=='job':result=run_job(output,args.case,args.arm,d,started=started)
    else:result=summarize(output,d)
    print(json.dumps(dict(stage=args.stage,status=result.get('status','PREPARED'),output=str(output)),sort_keys=True))


if __name__=='__main__':main()
