"""Focused protocol and telemetry checks; no formal model fit or solver campaign."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.stage3 import run_online_v2 as runner
from terminal_marl.core.rotating_slots.stage2_campaign import atomic_json,digest,load_json
from terminal_marl.core.rotating_slots.stage3_learning.evaluation import _costs
from terminal_marl.core.rotating_slots.stage3_learning.online_observer import OnlineObserver


def test_matrix_is_two_states_three_callbacks_and_one_closed_pair():
    protocol=dict(development_case_ids=["original_t0","encoding_t30"],closed_loop_case_id="original_t0")
    assert runner.matrix(protocol,"dev")==[(r,a) for r in protocol["development_case_ids"] for a in runner.ARMS]
    assert runner.matrix(protocol,"evaluate")==[("original_t0",a) for a in runner.ARMS[1:]]
    cfg=runner.config()
    assert (cfg["candidate_seconds"],cfg["max_per_aircraft"],cfg["max_attempts"])==(6.,32,1024)
    assert (cfg["solver_seconds"],cfg["hint_seconds"],cfg["soft_seconds"],cfg["watchdog_seconds"])==(8.,3.,30.,60.)
    assert cfg["heldout_signal_measurement"] is False
    assert cfg["formal_training_runs_max"]==1 and cfg["max_epochs"]==30


def test_mid_roots_keep_original_ancestry_split_and_actual_profiles():
    roots=[dict(spec) for spec in runner.config()["mid_snapshots"]]
    assert [r["root_id"] for r in runner.validate_mid_roots(dict(roots=roots))]==["s3r01","s3r02","s3r17"]
    wrong=deepcopy(roots);wrong[-1]["split"]="test"
    with pytest.raises(ValueError,match="manifest mismatch"):runner.validate_mid_roots(dict(roots=wrong))


def test_acquisition_progress_records_completed_callbacks_and_lifecycle(tmp_path):
    world=SimpleNamespace(time=0.);calls=[];ticks=iter(range(20));marker=object()
    def scheduler(port):
        calls.append(port)
        return marker
    path=tmp_path/"acquisition_progress/root.json"
    callback,publish=runner.acquisition_monitor(world,scheduler,path,clock=lambda:float(next(ticks)))
    initial=load_json(path)
    assert initial["stage"]=="acquisition" and initial["completed_callbacks"]==0
    world.time=5.
    assert callback(object(),"port") is marker
    observed=load_json(path)
    assert observed["time_s"]==5. and observed["completed_callbacks"]==1
    assert observed["last_callback_wall_s"]==1. and observed["elapsed_wall_s"]>0.
    assert observed["requested_sim_seconds"]==30. and observed["pid"]>0 and calls==["port"]
    world.time=30.;publish(stage="labeling");assert load_json(path)["stage"]=="labeling"
    publish(stage="complete",preference_pairs=2)
    assert load_json(path)["completed_callbacks"]==1 and load_json(path)["preference_pairs"]==2


def test_checkpoint_copy_keeps_time_identity_and_bytes_without_touching_input(tmp_path):
    source=tmp_path/"old.json";target=tmp_path/"new.json"
    payload=dict(time=30.,identity=dict(source_hash="OLD_INITIAL_IDENTITY"),world={"physical":"unchanged"},sink=None)
    atomic_json(source,dict(payload=payload,sha256=digest(payload)))
    original=source.read_bytes()
    provenance=runner.copy_checkpoint(source,target,30.)
    assert source.read_bytes()==target.read_bytes()==original
    assert provenance["identity"]==payload["identity"] and provenance["original_time_s"]==30.
    with pytest.raises(ValueError,match="time mismatch"):runner.copy_checkpoint(source,target,0.)
    payload["sink"]={"path":"readonly-state-stream"};atomic_json(source,dict(payload=payload,sha256=digest(payload)))
    with pytest.raises(ValueError,match="stream_copy"):runner.copy_checkpoint(source,target,30.)


def test_online_costs_use_stats_without_full_records_or_extra_features():
    observer=OnlineObserver()
    observer(dict(record_type="FeaturesBeforeQuery",features=object()))
    observer(dict(record_type="FactualResult",descriptor_id="d",wall_s=.2,physical_attempts=2))
    observer(dict(record_type="SingleVehicleEvidence",descriptor_id="d",wall_s=.3,outcome="SINGLE_VALID"))
    observer(dict(record_type="ColumnValidationCost",descriptor_id="d",wall_s=.4))
    result=_costs(SimpleNamespace(records=[dict(wall_s=1.)]),observer,2.,1.,None)
    assert not hasattr(observer,"records")
    assert result["physical_attempts"]==2 and result["descriptor_queries"]==1
    assert result["generation_wall_s"]==.2 and result["energy_wall_s"]==.3 and result["column_validation_wall_s"]==.4
    assert result["planning_latency"]["p95_s"]==1.


def test_cold_search_projection_preserves_entire_non_scheduler_payload(tmp_path):
    from terminal_marl.core.rotating_slots.stage2_campaign import Stage2Codec
    class Cold:
        def state_dict(self):return dict(mode="OFF",records=[],previous={})
        def load_state_dict(self,state):raise ValueError("Stage3 callback binding mismatch")
    payload=dict(time=30.,identity=dict(source_hash="OLD"),sink=None,boundary_summary={"observed":"old"},
        scheduler=Stage2Codec().encode(dict(mode="GUIDED_LIMITED",budget_scale=.5)),
        world=dict(active=["committed"],soft=["dependent"],energy_j=1729.,dependencies={"child":"parent"},extra_field="preserve all"))
    source=tmp_path/"old.json";target=tmp_path/"initial.checkpoint.json"
    atomic_json(source,dict(payload=payload,sha256=digest(payload)));old_bytes=source.read_bytes()
    result=runner.copy_checkpoint(source,target,30.,scheduler=Cold(),allow_cold_projection=True)
    actual=load_json(target)["payload"]
    assert result["cold_search_projection"] is True
    assert {k:v for k,v in actual.items() if k!="scheduler"}=={k:v for k,v in payload.items() if k!="scheduler"}
    assert source.read_bytes()==(tmp_path/"source.checkpoint.json").read_bytes()==old_bytes
    assert load_json(target)["sha256"]==digest(actual)


def test_closed_comparison_uses_last_common_completed_boundary(tmp_path):
    for arm,times in zip(runner.ARMS[1:],((5,10,15),(5,10))):
        directory=tmp_path/"evaluation/closed_loop/original_t0"/arm;directory.mkdir(parents=True)
        for t in times:atomic_json(directory/f"boundary_{t:04d}.json",dict(time=t,arm=arm))
    row=runner.closed_comparison(tmp_path,dict(closed_loop_case_id="original_t0"),[])
    assert row["common_completed_boundary_s"]==10 and [r["time"] for r in row["rows"]]==[10,10]
    assert row["status"]=="EVALUATION_PARTIAL"


def test_formal_evaluation_requires_primary_review_before_stage_claim(tmp_path,monkeypatch):
    class Ledger:
        def pause(self):pass
    atomic_json(tmp_path/"online_v2_checks.json",dict(passed=True))
    monkeypatch.setattr(runner,"verify",lambda output:({}, {}, Ledger()))
    with pytest.raises(ValueError,match="primary review"):runner.run_stage(tmp_path,"evaluate")
    assert not (tmp_path/"evaluate.started.json").exists()


def test_train_after_gate_refused_before_guard(tmp_path,monkeypatch):
    atomic_json(tmp_path/"online_v2_checks.json",dict(passed=True))
    atomic_json(tmp_path/"heldout_access.json",dict(access="DEVELOPMENT_SEEN"))
    monkeypatch.setattr(runner,"verify",lambda output:({}, {}, object()))
    with pytest.raises(ValueError,match="no training"):runner.run_stage(tmp_path,"train")
    assert not (tmp_path/"train.started.json").exists()


def repair_fixture(tmp_path,monkeypatch):
    """Tiny saved artifacts only; no fit, generator, trajectory, validator or CP."""
    from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_V2
    output=tmp_path/'output';output.mkdir()
    cfg=dict(mid_snapshots=[dict(root_id='r',split='train')])
    monkeypatch.setattr(runner,'config',lambda:cfg)
    path='terminal_marl/core/rotating_slots/stage3_learning/learning.py'
    oldfiles={path:'oldhash'};newfiles={path:'newhash'}
    oldfreeze=dict(source_files=oldfiles,source_hash=digest(oldfiles),source_commit='oldcommit',workspace_diff_hash='clean')
    newfreeze=dict(source_files=newfiles,source_hash=digest(newfiles),source_commit='newcommit',workspace_diff_hash='clean')
    pilotfp=dict(objective_id='o',source_commit='pilot',workspace_diff_hash='clean')
    fp=dict(objective_id='o',source_commit='oldcommit',workspace_diff_hash='clean')
    binding=dict(source_files=oldfiles,source_hash=oldfreeze['source_hash'],fingerprint=fp,
        execution_provenance=dict(old_training_fingerprint=pilotfp,source_commit='oldcommit',source_hash=oldfreeze['source_hash']))
    atomic_json(tmp_path/'pilot.json',dict(fingerprint=pilotfp,contexts=[],preference_pairs=[]))
    atomic_json(output/'split_manifest.json',dict(roots=[]))
    protocol=dict(config=cfg,source_hash=oldfreeze['source_hash'],training_dataset_input=str(tmp_path/'pilot.json'),
        inherited_split_sha256=runner.file_hash(output/'split_manifest.json'),development_case_ids=['unchanged'],closed_loop_case_id='unchanged')
    for name,value in [('source_freeze.json',oldfreeze),('run_binding.generated.json',binding),('online_v2_protocol.json',protocol),('collect.finished.json',dict(status='COMPLETED'))]:atomic_json(output/name,value)
    fact=dict(record_type='FactualResult',descriptor_id='d',context_id='n',ancestry_root_id='r',fingerprint=fp,features_before_query_ref='q',
        outcome='UNKNOWN',wall_s=.2,cost_complete=True,cost_scope='GENERATOR_ENERGY_TERMINAL_VALIDATION_V2')
    labels=dict(root_id='r_t30',ancestry_root_id='r',context_id='s',split='train',fingerprint=fp,facts=[fact],reference_solutions=[],conditional_attempts=[],
        incremental_contribution_evidence=[],incremental_labeled_descriptors=0,preference_pairs=[],candidate_timer_regime='existing')
    atomic_json(output/'labels/r_t30/labels.json',labels)
    raw=output/'raw/r_t30';raw.mkdir(parents=True)
    descriptor=dict(descriptor_id='d',source_node_id='n')
    feature=dict(descriptor_id='d',schema_id=SCHEMA_V2,query=[0.,1.]*33,sets={})
    import json
    (raw/'descriptors.jsonl').write_text(json.dumps(descriptor)+'\n')
    (raw/'features.jsonl').write_text(json.dumps(dict(descriptor_id='d',features_before_query_ref='q',features=feature))+'\n')
    (raw/'facts.jsonl').write_text(json.dumps(fact)+'\n')
    runner.build_dataset(output,binding,protocol)
    ledger=runner.campaign.CampaignLedger(output/'budget_ledger.json');ledger.pause()
    atomic_json(tmp_path/'newfreeze.json',newfreeze)
    verified=[]
    monkeypatch.setattr(runner.campaign,'verify_sources',lambda value:verified.append(value))
    return output,tmp_path/'newfreeze.json',verified


def test_pretrain_repair_archives_exact_bytes_reindexes_and_rebinds_without_new_jobs(tmp_path,monkeypatch):
    output,newfreeze,verified=repair_fixture(tmp_path,monkeypatch)
    names=('source_freeze.json','run_binding.generated.json','online_v2_protocol.json','training_dataset.json','dataset_index.json')
    before={name:(output/name).read_bytes() for name in names}
    raw_before={p:p.read_bytes() for p in (output/'raw').rglob('*.jsonl')}
    labels=(output/'labels/r_t30/labels.json').read_bytes()
    ledger_before=load_json(output/'budget_ledger.json')
    result=runner.repair_pretrain(output,newfreeze)
    assert result['status']=='COMPLETED' and result['eligible_train_groups']==1 and result['eligible_cost_only_groups']==1
    assert len(verified)==1 and verified[0]['source_commit']=='newcommit'
    archive=output/'implementation_repair_pretrain'
    assert all((archive/name).read_bytes()==value for name,value in before.items())
    assert all(p.read_bytes()==value for p,value in raw_before.items())
    assert (output/'labels/r_t30/labels.json').read_bytes()==labels
    binding=load_json(output/'run_binding.generated.json');protocol=load_json(output/'online_v2_protocol.json')
    assert binding['fingerprint']['source_commit']=='newcommit' and binding['source_hash']==protocol['source_hash']
    assert binding['execution_provenance']['collection_source']['source_commit']=='oldcommit'
    assert binding['execution_provenance']['source_commit']=='newcommit' and binding['execution_provenance']['source_hash']==binding['source_hash']
    fresh=load_json(output/'training_dataset.json')['contexts'][0]['facts'][0]
    assert fresh['fingerprint']['source_commit']=='newcommit' and fresh['source_fingerprint']['source_commit']=='oldcommit'
    ledger=load_json(output/'budget_ledger.json')
    assert ledger['paused_at']==ledger_before['paused_at'] and ledger['jobs']==ledger_before['jobs']
    assert ledger['phase_caps']==ledger_before['phase_caps'] and ledger['phase_used']['data_and_labels']>0
    assert not (output/'train.started.json').exists() and not (output/'formal_training.guard.json').exists()
    assert (archive/'repair_receipt.json').is_file()
    with pytest.raises(ValueError,match='already attempted'):runner.repair_pretrain(output,newfreeze)


@pytest.mark.parametrize('blocked',('train.started.json','formal_training.guard.json','model','heldout_access.json','running','unpaused','incomplete','unapproved'))
def test_pretrain_repair_refuses_started_training_live_jobs_and_other_changes(tmp_path,monkeypatch,blocked):
    output,newfreeze,_=repair_fixture(tmp_path,monkeypatch)
    if blocked in ('train.started.json','formal_training.guard.json','heldout_access.json'):atomic_json(output/blocked,{})
    elif blocked=='model':(output/'model').mkdir()
    elif blocked in ('running','unpaused'):
        ledger=load_json(output/'budget_ledger.json')
        if blocked=='running':ledger['jobs']['unexpected']=dict(status='running',pid=123)
        else:ledger['paused_at']=None
        atomic_json(output/'budget_ledger.json',ledger)
    elif blocked=='incomplete':atomic_json(output/'collect.finished.json',dict(status='FAILED'))
    else:
        frozen=load_json(newfreeze);frozen['source_files']={'physics.py':'changed'};frozen['source_hash']=digest(frozen['source_files']);atomic_json(newfreeze,frozen)
    old={p:p.read_bytes() for p in output.glob('*.json')}
    with pytest.raises(ValueError):runner.repair_pretrain(output,newfreeze)
    assert all(p.read_bytes()==value for p,value in old.items())
    assert not (output/'implementation_repair_pretrain').exists()
