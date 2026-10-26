"""Driver-only serialization tests; no world physics, solver or model inference."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import threading
import shutil

import pytest

from scripts.stage3 import run_online_v2_600 as driver


def atomic(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding='utf8')


def digest(value):return sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Scheduler:
    def __init__(self):self.records=[];self.provider=SimpleNamespace(previous={})
    def __call__(self,port):
        self.records.append(dict(time=port.time,wall_s=.1,attempts={'a':1},timings={},catalog_rounds=[],prefix_reuse={}))


class World:
    def __init__(self,fail_at=None):
        self.time=0.;self.failed=False;self.terminated_invalid=False;self.dispatch_timings=[]
        self.config=SimpleNamespace(path='unchanged-config.yaml');self.steps=[];self.fail_at=fail_at
    def configure_practical_runtime(self,enabled):assert enabled is True
    def advance(self,boundary,normal,*,settle_only_at_end):
        assert settle_only_at_end is True
        normal(None,self);self.time=boundary;self.steps.append(boundary)
        self.dispatch_timings.append(dict(time=boundary,dispatch_wall_s=.2))
        if boundary==self.fail_at:self.terminated_invalid=True
    def to_dict(self):return dict(states=[dict(time=0.),dict(time=self.time)],events=[],trajectories={})


def job_deps(tmp_path,world,monkeypatch):
    output=tmp_path/'output';output.mkdir()
    row=dict(identity={'original':'kept'},manifest_hash='manifest',initial_state_hash='initial')
    for case in driver.CASES:atomic(output/'roots'/case/'root.json',dict(row,root_id=case))
    binding=dict(fingerprint={'original':'fingerprint'},execution_provenance={'source_commit':driver.CORE_COMMIT})
    scope=dict(evaluation_scope_id='new-600',driver_sha256='driver',aggregator_sha256='agg')
    monkeypatch.setattr(driver,'verify',lambda output,d:(scope,binding,None))
    monkeypatch.setattr(driver.os,'sched_setaffinity',lambda pid,cpus:None,raising=False)
    scheduler=Scheduler();seen=[];saved=[];restored=[]
    def build(output,arm,binding_arg,observer,case,setup):
        seen.append((arm,deepcopy(binding_arg),case));return scheduler
    def costs(scheduler,observer,wall,cpu,dispatch,**kwargs):
        return dict(full_online_wall_s=wall,full_dispatch_wall_s=dispatch,planning_latency=dict(count=len(scheduler.records)))
    def quality(world):return dict(L_normal=0,L_success=0,F_energy_total=0,L_emergency=0,cost_emergency=0.,cost_reroute=0.,energy_j=world.time,airborne_time=world.time,upstream_wait_time=0.,K_episode=[0,0,0.,world.time,world.time],cohorts={})
    def checkpoint(path,world,scheduler,identity,**kwargs):
        saved.append((Path(path).name,world.time,identity));atomic(path,dict(time=world.time,identity=identity))
    class Replay:
        @staticmethod
        def write(path,*,record,**kwargs):
            path.mkdir()
            (path/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in record['decisions']))
            atomic(path/'replay.json',dict(code_commit=kwargs['code_commit']))
    d=SimpleNamespace(restore_root=lambda row,cold:world,NormalSchedulerCallback=lambda *a,**k:object(),OnlineObserver=lambda:object(),
        shared_fifo_admission=object(),episode_metrics=quality,save_checkpoint=checkpoint,ReplayBundle=Replay,atomic_json=atomic,
        campaign=SimpleNamespace(utc_now=lambda:'UTC'),
        evaluation=SimpleNamespace(_scheduler=build,_costs=costs,observe_executed_holding=lambda w:({},lambda:restored.append(True)),
            terminal_observation=lambda w:dict(status='OBSERVED',observed_time_s=w.time,failed=w.failed,terminated_invalid=w.terminated_invalid,quality=quality(w))),
        aggregate=lambda records,costs,dispatch:dict(callbacks=len(records),dispatches=len(dispatch)))
    return output,d,seen,saved,restored


def test_600_loop_serializes_120_boundaries_and_only_60s_checkpoints(tmp_path,monkeypatch):
    world=World();output,d,seen,saved,restored=job_deps(tmp_path,world,monkeypatch)
    result=driver.run_job(output,'medium','heuristic_reduced',d)
    base=output/'evaluation/medium/heuristic_reduced'
    assert world.steps==[float(t) for t in range(5,601,5)]
    assert len(list(base.glob('boundary_*.json')))==120
    assert [t for _,t,_ in saved]==list(range(0,601,60))
    assert not (base/'checkpoint_terminal.json').exists()
    assert result['status']=='COMPLETE' and result['time']==600.
    assert result['all_callbacks']==dict(callbacks=120,dispatches=120)
    assert driver.load(base/'terminal_observation.json')['observed_time_s']==600.
    assert driver.load(output/'progress/medium/heuristic_reduced.json')['phase']=='complete'
    assert driver.load(base/'job.finished.json')['status']=='COMPLETE' and restored==[True]
    assert seen[0][1]['fingerprint']=={'original':'fingerprint'}


def test_partial_keeps_last_completed_boundary_and_independent_terminal(tmp_path,monkeypatch):
    world=World(fail_at=35.);output,d,_,saved,restored=job_deps(tmp_path,world,monkeypatch)
    result=driver.run_job(output,'high','learned_reduced',d)
    base=output/'evaluation/high/learned_reduced'
    assert result['status']=='EVALUATION_PARTIAL' and result['time']==30.
    assert result['terminal_observation']['observed_time_s']==35.
    assert result['terminal_observation']['terminated_invalid'] is True
    assert not (base/'boundary_0035.json').exists()
    assert driver.load(base/'checkpoint_terminal.json')['time']==35.
    assert result['all_callbacks']['callbacks']==7 and len(saved)==2 and restored==[True]


def test_fixed_matrix_equal_arm_cap_and_new_campaign_bounds():
    assert driver.matrix()==[('medium','heuristic_reduced'),('medium','learned_reduced'),('high','heuristic_reduced'),('high','learned_reduced')]
    assert driver.CASES['medium']['cpu']==12 and driver.CASES['high']['cpu']==14
    assert driver.JOB_CAP==21600. and driver.CAPS==dict(preflight_tests=900.,evaluation=45000.,archive=900.)
    assert (driver.COMPUTE_STOP,driver.TOTAL)==(45900.,46800.)
    assert driver.job_command(Path('out'),Path('frozen'),'high','learned_reduced')[-4:]==['--case','high','--arm','learned_reduced']


def test_same_case_arms_receive_same_identity_and_original_fingerprint(tmp_path,monkeypatch):
    identities=[];bindings=[]
    for arm in driver.ARMS:
        where=tmp_path/arm;where.mkdir();world=World(fail_at=5.)
        output,d,seen,saved,_=job_deps(where,world,monkeypatch)
        driver.run_job(output,'medium',arm,d)
        identities.append(saved[0][2]);bindings.append(seen[0][1]['fingerprint'])
    assert identities[0]==identities[1]=={'original':'kept'}
    assert bindings[0]==bindings[1]=={'original':'fingerprint'}


def test_model_metadata_preflight_does_not_load_or_infer(tmp_path,monkeypatch):
    (tmp_path/'model').mkdir()
    (tmp_path/'model/model.pt').write_bytes(b'weights-only-copy')
    atomic(tmp_path/'model/feature_normalizer.json',dict(schema_id='v2'))
    fp={'feature_schema_id':'v2','evaluation_protocol_id':'original-60-unchanged','source_commit':driver.CORE_COMMIT}
    atomic(tmp_path/'model/model_card.json',dict(model_frozen=True,model_hash=driver.file_hash(tmp_path/'model/model.pt'),parameter_count=119555,seed=20261023,fingerprint=fp))
    atomic(tmp_path/'run_binding.generated.json',dict(fingerprint=fp,source_hash=driver.CORE_HASH))
    monkeypatch.setattr(driver,'MODEL_HASH',driver.file_hash(tmp_path/'model/model.pt'))
    monkeypatch.setattr(driver,'NORMALIZER_HASH',driver.file_hash(tmp_path/'model/feature_normalizer.json'))
    assert driver.verify_model(tmp_path)['fingerprint']==fp


class Ledger:
    def __init__(self,path):
        self.path=Path(path)
        self.data=driver.load(path) if self.path.exists() else dict(total_s=0.,compute_stop_s=0.,phase_caps={},phase_used={},paused_at=None,active_phase=None,jobs={})
    def persist(self):atomic(self.path,self.data)
    def pause(self):self.data['paused_at']=1.;self.data['active_phase']=None;self.persist()
    def remaining(self,phase):return self.data['phase_caps'][phase]
    @contextmanager
    def phase(self,name):
        self.data['paused_at']=None;self.data['active_phase']={'phase':name};self.persist()
        try:yield
        finally:self.data['active_phase']=None;self.persist()
    @property
    def elapsed(self):return 0.


def prepared_fixture(tmp_path,monkeypatch):
    model=tmp_path/'model_parent';initial=tmp_path/'initial_parent';checkout=tmp_path/'frozen';checkout.mkdir()
    (model/'model').mkdir(parents=True)
    (model/'model/model.pt').write_bytes(b'existing-frozen-weights')
    atomic(model/'model/feature_normalizer.json',dict(schema_id='v2'))
    monkeypatch.setattr(driver,'MODEL_HASH',driver.file_hash(model/'model/model.pt'))
    monkeypatch.setattr(driver,'NORMALIZER_HASH',driver.file_hash(model/'model/feature_normalizer.json'))
    fp=dict(source_commit=driver.CORE_COMMIT,feature_schema_id='v2',evaluation_protocol_id='ORIGINAL_PROTOCOL_UNCHANGED')
    atomic(model/'model/model_card.json',dict(model_frozen=True,model_hash=driver.MODEL_HASH,parameter_count=119555,seed=20261023,fingerprint=fp))
    inventory={str(i):'original' for i in range(85)}
    binding=dict(fingerprint=fp,source_hash=driver.CORE_HASH,source_files=inventory,execution_provenance=dict(source_commit=driver.CORE_COMMIT))
    atomic(model/'run_binding.generated.json',binding)
    atomic(model/'source_freeze.json',dict(source_commit=driver.CORE_COMMIT,source_hash=driver.CORE_HASH,source_files=inventory))
    for case,spec in driver.CASES.items():
        base=initial/'roots'/spec['root_id']
        manifest=dict(seed=spec['seed'],rate_per_hour=spec['rate_per_hour'],arrival_window=3600.,requests=[dict(release_time=float(i)) for i in range(spec['requests'])])
        state=dict(aircraft=[dict(status='HOLD_INNER') for _ in range(spec['initial_airborne'])]+[dict(status='LANDED_SERVICE') for _ in range(3)],slots=[],pads=[])
        row=dict(root_id=spec['root_id'],seed=spec['seed'],scenario_profile=spec['scenario_profile'],identity={'original':case},manifest_hash=digest(manifest),initial_state_hash=digest(state))
        payload=dict(identity=row['identity'],time=0.,sink=None)
        for name,value in [('manifest.json',manifest),('initial_state.json',state),('root.json',row),('initial.checkpoint.json',dict(payload=payload,sha256=digest(payload)))]:atomic(base/name,value)
    checked=[]
    @contextmanager
    def lock(output):yield
    campaign=SimpleNamespace(CampaignLedger=Ledger,utc_now=lambda:'UTC',verify_sources=lambda value:checked.append(value),
        resource_probe=lambda:dict(free_physical_cores=[12,14]),campaign_lock=lock)
    d=SimpleNamespace(campaign=campaign,checkout=checkout,atomic_json=atomic,digest=digest)
    args=SimpleNamespace(output=tmp_path/'new_output',model_parent=model,initial_parent=initial)
    before={p:p.read_bytes() for parent in (model,initial) for p in parent.rglob('*') if p.is_file()}
    scope=driver.prepare(args,d)
    return args.output,d,scope,before,checked


def test_prepare_copies_only_required_inputs_preserves_fingerprint_and_serializes_new_scope(tmp_path,monkeypatch):
    output,d,scope,before,checked=prepared_fixture(tmp_path,monkeypatch)
    assert all(p.read_bytes()==value for p,value in before.items())
    assert scope['model_fingerprint']['evaluation_protocol_id']=='ORIGINAL_PROTOCOL_UNCHANGED'
    assert scope['target_sim_seconds']==600 and scope['core_inventory_count']==85 and len(checked)==1
    assert len(list((output/'model').iterdir()))==3
    assert len(list((output/'roots').glob('*/*')))==8
    assert not (output/'training_dataset.json').exists()
    verified,binding,ledger=driver.verify(output,d)
    assert verified['evaluation_scope_id']==scope['evaluation_scope_id']
    assert binding['fingerprint']==scope['model_fingerprint']
    assert ledger.data['phase_caps']==driver.CAPS and ledger.data['paused_at'] is not None
    with pytest.raises(ValueError,match='fresh output'):driver.prepare(SimpleNamespace(output=output,model_parent=Path('unused'),initial_parent=Path('unused')),d)


def test_parent_runs_exactly_four_equal_cap_jobs_parallel_cases_serial_arms(tmp_path,monkeypatch):
    output,d,scope,before,_=prepared_fixture(tmp_path,monkeypatch)
    barrier=threading.Barrier(2);seen=[];mutex=threading.Lock()
    def subprocess(ledger,phase,job_id,command,directory,*,cap,parallel):
        assert phase=='evaluation' and cap==21600. and parallel is True
        case,arm=job_id.split(':')
        if arm=='heuristic_reduced':barrier.wait(timeout=5)
        with mutex:seen.append((case,arm,cap))
        atomic(output/'evaluation'/case/arm/'result.json',dict(status='COMPLETE'))
        receipt=dict(status='completed',exit_code=0,timeout_s=cap,job_id=job_id)
        ledger.data['jobs'][job_id]=receipt;ledger.persist();return receipt
    d.campaign.run_subprocess=subprocess
    result=driver.run(output,d)
    assert result['status']=='COMPLETED' and len(seen)==4
    for case in driver.CASES:assert [arm for c,arm,_ in seen if c==case]==list(driver.ARMS)
    assert Ledger(output/'budget_ledger.json').data['paused_at'] is not None
    assert all(p.read_bytes()==value for p,value in before.items())
    with pytest.raises(ValueError,match='already attempted'):driver.run(output,d)


def test_parent_refuses_reduced_cap_without_launching_any_job(tmp_path,monkeypatch):
    output,d,_,_,_=prepared_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(Ledger,'remaining',lambda self,phase:21600.)
    launched=[];d.campaign.run_subprocess=lambda *a,**k:launched.append(True)
    with pytest.raises(RuntimeError,match='incomplete'):driver.run(output,d)
    assert not launched
    assert driver.load(output/'run.finished.json')['status']=='INCOMPLETE_RETURN_TO_PRIMARY'


def test_summary_serializes_common_boundary_terminal_all_callbacks_and_saved_trends(tmp_path,monkeypatch):
    h=tmp_path/'h';h.mkdir();output,d,_,_,_=job_deps(h,World(),monkeypatch)
    driver.run_job(output,'medium','heuristic_reduced',d)
    l=tmp_path/'l';l.mkdir();other,d2,_,_,_=job_deps(l,World(fail_at=35.),monkeypatch)
    driver.run_job(other,'medium','learned_reduced',d2)
    shutil.copytree(other/'evaluation/medium/learned_reduced',output/'evaluation/medium/learned_reduced')
    ledger=SimpleNamespace(data=dict(paused_at=1.,active_phase=None,phase_used={}),elapsed=0.)
    monkeypatch.setattr(driver,'verify',lambda output,d:({}, {}, ledger))
    d.evaluation.quality_differences=lambda h,l:dict(normal_delta=l['L_normal']-h['L_normal'])
    before={p:p.read_bytes() for p in output.rglob('*') if p.is_file()}
    result=driver.summarize(output,d)
    medium=result['cases']['medium']
    assert medium['common_completed_boundary_s']==30 and medium['status']=='EVALUATION_PARTIAL'
    assert medium['terminal_observations']['learned_reduced']['observed_time_s']==35.
    assert medium['all_callbacks']['heuristic_reduced']['callbacks']==120
    assert medium['all_callbacks']['learned_reduced']['callbacks']==7
    assert [r['time'] for r in medium['trends_every_60s']['heuristic_reduced']]==list(range(60,601,60))
    assert result['cases']['high']['common_completed_boundary_s']==0
    assert driver.load(output/'summary.json')['cases']['medium']['rows']['heuristic_reduced']['time']==30.
    assert all(p.read_bytes()==value for p,value in before.items())
