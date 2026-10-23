#!/usr/bin/env python3
"""Read-only specification checks; never imports the repo or starts training.

Usage: python validate_package.py
Optional: --write-report PATH writes the document-level check report.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
try:
    import yaml
except ImportError:
    raise SystemExit('PyYAML is required for document checks; no packages installed automatically.')

ROOT = Path(__file__).resolve().parent

def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def validate() -> dict:
    cfg = yaml.safe_load((ROOT/'stage3_learning_config_v1_0.yaml').read_text(encoding='utf-8'))
    ds = yaml.safe_load((ROOT/cfg['dataset_config_file']).read_text(encoding='utf-8'))
    cat = json.loads((ROOT/cfg['targeted_checks_file']).read_text(encoding='utf-8'))
    schema = json.loads((ROOT/cfg['data_schema_file']).read_text(encoding='utf-8'))
    text = (ROOT/cfg['contract_file']).read_text(encoding='utf-8')
    checks: dict[str, bool] = {}
    def ck(name: str, condition: bool) -> None:
        checks[name] = bool(condition)
    ck('formal_id_and_status', cfg['contract_id']=='FT-RS-S3-INITIAL-1.0' and cfg['status']=='FORMAL_FOR_IMPLEMENTATION' and cfg['contract_id'] in text)
    ck('decisions_all_approved', set(cfg['approval']['decisions'])=={f'D{i}' for i in range(1,8)} and all(x['approved'] for x in cfg['approval']['decisions'].values()))
    ck('no_unresolved_research_decisions', cfg['execution']['unresolved_research_decisions']==[])
    ck('authorized_but_not_run', cfg['execution']['training_authorized'] and cfg['execution']['guided_pilot_authorized'] and not any(cfg['execution'][k] for k in ['implementation_already_run','model_already_trained','simulation_already_run']))
    ck('off_default_no_autorun', cfg['execution']['default_advisor_mode']=='OFF' and not cfg['execution']['campaign_autorun_on_import'])
    ck('training_not_rl_or_world_model', not cfg['scope']['online_rl'] and not cfg['scope']['world_model'] and cfg['training']['formal_runs_max']==1)
    ck('cp_objective_and_tail_unchanged', not cfg['scope']['change_cp_objective'] and not cfg['scope']['tail_model_training'] and not cfg['scope']['tail_model_inference'])
    ck('physics_and_safety_preserved', cfg['scope']['final_authoritative_validation_required'] and not cfg['scope']['learned_safety_authorization'] and not cfg['scope']['physical_rules_changed'] and not cfg['scope']['emergency_rules_changed'])
    ck('single_cpu_set_model', cfg['model']['family']=='candidate_conditioned_set_mlp' and cfg['model']['device']=='cpu' and cfg['model']['entity_encoder_hidden']==[64,64])
    ck('guarded_reduced_budget', cfg['proposal']['budget']['reduced_scale']==0.5 and cfg['proposal']['budget']['deterministic_finish_fraction']==0.2 and not cfg['proposal']['budget']['full_budget_reset_on_fallback'])
    ck('coverage_and_pre_materialization', cfg['proposal']['coverage']['preserve_keep_wait'] and cfg['proposal']['parent_must_be_materialized'] and not cfg['proposal']['full_generated_catalog_before_inference'])
    ck('labels_masked_not_pseudo_negatives', not cfg['training']['unknown_is_negative'] and not cfg['training']['unselected_is_negative'] and cfg['training']['validity_label_mask_required'])
    ck('four_hour_total', cfg['compute']['campaign_wall_seconds']==14400 and sum(cfg['compute']['phase_caps_seconds'].values())==14400)
    ck('archive_reserve', cfg['compute']['compute_stop_seconds']+cfg['compute']['archive_reserve_seconds']==14400 and cfg['compute']['archive_reserve_seconds']==900)
    ck('no_budget_reset_or_nested_workers', not cfg['compute']['resume_resets_budget'] and not cfg['compute']['nested_parallelism'] and cfg['compute']['per_process_threads']==1 and cfg['compute']['max_parallel_processes']==2)
    roots=ds['roots']; counts=Counter(r['split'] for r in roots)
    ck('dataset_24_unique_roots', len(roots)==24 and len({r['root_id'] for r in roots})==24 and len({r['preferred_seed'] for r in roots})==24)
    ck('splits_16_4_4', dict(counts)=={'train':16,'validation':4,'test':4} and ds['split_targets']==dict(counts))
    ck('seed71_development_only', all(r['preferred_seed']!=71 for r in roots) and ds['development']['seed_71_role']=='DEVELOPMENT_ONLY' and not ds['development']['formal_training_use'] and not ds['development']['formal_test_use'])
    pilot=next((r for r in roots if r['root_id']==ds['short_pilot']['preferred_root_id']),{})
    ck('pilot_is_predeclared_medium_test', pilot.get('split')=='test' and pilot.get('scenario_profile')=='medium24_rate150_v1' and pilot.get('preferred_seed')==ds['short_pilot']['preferred_seed'])
    ck('same_budget_control_and_60s', cfg['evaluation']['arms']==['reference_full','heuristic_reduced','learned_reduced'] and cfg['evaluation']['closed_loop_arms']==['heuristic_reduced','learned_reduced'] and cfg['evaluation']['closed_loop_sim_seconds']==ds['short_pilot']['sim_seconds']==60)
    ck('no_grid_search_or_test_tuning', not cfg['training']['grid_search'] and not cfg['training']['automatic_retraining'] and not cfg['training']['test_based_tuning'] and not cfg['evaluation']['after_test_retrain'])
    ck('quality_tradeoff_not_approved', cfg['evaluation']['quality_degradation_allowance']=='NOT_AUTHORIZED' and cfg['evaluation']['performance_pass_threshold']=='NONE')
    ck('twelve_targeted_tests_not_run', cat['count']==len(cat['checks'])==12 and len({r['id'] for r in cat['checks']})==12 and all(r['execution_status']=='NOT_RUN' for r in cat['checks']))
    ck('schemas_cover_four_record_types', set(schema['$defs'])=={'Fingerprint','WorkDescriptor','FactualResult','PreferencePair','RunBinding'} and len(schema['oneOf'])==4)
    ck('contract_sections_preserved', [int(x) for x in re.findall(r'^## (\d+)\.',text,re.M)]==list(range(21)))
    ck('no_obsolete_approval_gate', all(s not in text for s in ['DISCUSSION_DRAFT','D1—D7确认后','经用户确认才启用','由D4—D7确认范围','等待用户确认']))
    ck('required_files_exist', all((ROOT/f).is_file() for f in [cfg['contract_file'],cfg['dataset_config_file'],cfg['data_schema_file'],cfg['targeted_checks_file'],'IMPLEMENTATION_ENTRY.md','README.md','CHANGELOG.md']))
    ck('claim_not_fresh_repository_audit', not cfg['provenance']['fresh_repository_review_performed_for_this_package'] and '不代表再次读取了当前仓库' in text)
    ck('engineering_values_documented', all(v in text for v in ['0.5倍','14400秒','180秒','每快照最多8次','学习率0.001','16个train','4个validation','4个test']))
    try:
        import jsonschema
        jsonschema.Draft202012Validator.check_schema(schema)
        ck('json_schema_meta_validation', True)
    except ImportError:
        # Optional library; report accurately instead of marking it passed.
        pass
    manifest_path = ROOT/'FILES_SHA256.json'
    if manifest_path.exists():
        mf=json.loads(manifest_path.read_text(encoding='utf-8'))
        ck('payload_hashes', all((ROOT/name).is_file() and digest(ROOT/name)==value for name,value in mf['files'].items()))
        ck('manifest_complete', set(mf['files'])=={f.name for f in ROOT.iterdir() if f.is_file() and f.name!='FILES_SHA256.json'})
    return {'scope':'document/config/schema internal consistency only; not repository tests, training or simulation',
            'contract_id':cfg['contract_id'],'result':'PASS' if all(checks.values()) else 'FAIL',
            'check_count':len(checks),'checks':checks,
            'implementation_status':'NOT_RUN','training_status':'NOT_RUN','simulation_status':'NOT_RUN',
            'source_draft_sha256':cfg['provenance']['input_draft_sha256']}

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report',type=Path)
    args=parser.parse_args()
    try:
        result=validate()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'result':'FAIL','error':str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2
    rendered=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.write_report:
        dest=args.write_report.resolve()
        if (ROOT/'FILES_SHA256.json').exists() and dest.parent==ROOT:
            raise SystemExit('Refusing to modify a sealed package. Write reports outside the package.')
        dest.write_text(rendered,encoding='utf-8')
    print(rendered,end='')
    return 0 if result['result']=='PASS' else 1

if __name__=='__main__':
    raise SystemExit(main())
