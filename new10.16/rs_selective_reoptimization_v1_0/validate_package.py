#!/usr/bin/env python3
"""Validate this specification bundle only; never run a simulator or training job."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
try:
    import yaml
    import jsonschema
except ImportError as exc:
    raise SystemExit('Package validation requires PyYAML and jsonschema; no research job was run.') from exc

BASE = Path(__file__).resolve().parent
EXCLUDED_HASH_FILES = {'MANIFEST.json', 'PACKAGE_VALIDATION.json'}

def load_json(name: str):
    return json.loads((BASE / name).read_text(encoding='utf-8'))

def load_yaml(name: str):
    return yaml.safe_load((BASE / name).read_text(encoding='utf-8'))

def main() -> int:
    checks: list[dict] = []
    def check(name: str, condition: bool, detail: str = '') -> None:
        checks.append({'name': name, 'passed': bool(condition), 'detail': detail})
        if not condition:
            raise ValueError(f'{name}: {detail}')
    try:
        config = load_yaml('selective_reoptimization_config_v1_0.yaml')
        contract_id = config['contract_id']
        check('contract_identity', contract_id == 'FT-RS-SELECTIVE-1.0')
        text = (BASE/'STAGE3R_SELECTIVE_REOPTIMIZATION_CONTRACT_v1_0.md').read_text(encoding='utf-8')
        check('contract_and_source_binding', contract_id in text and config['source']['review_commit'] in text)
        check('no_implementation_claim', config['status'] == 'SPECIFICATION_NOT_IMPLEMENTED' and config['verification']['results_status'] == 'NOT_RUN')
        check('budget_total', sum(config['campaign']['phase_caps_s'].values()) == config['campaign']['total_wall_cap_s'] == 36000)
        check('phase_gates', config['campaign']['require_phase_review'] and not config['campaign']['auto_advance'])
        check('scope_limits', config['scope']['nominal_k'] == 8 and config['scope']['hard_max_k'] == 12 and config['scope']['full_snapshot_retained'])
        check('safety_invariants', config['preserve']['checked_equals_executed'] and config['objective']['background_demands_visible'] and not config['scope']['allow_cancel_committed'])
        check('model_boundary', not config['model']['reuse_old_weights'] and config['search']['old_descriptor_ranker'] == 'OFF' and config['model']['batch_forwards_per_task'] == 1)
        timing = config['timing']
        check('timing_headroom', timing['lead_s'] - timing['preparation_margin_s'] - timing['search_handoff_reserve_s'] > 0)
        check('C_not_silent', not config['phase_C']['C2_default_enabled'] and config['phase_C']['C2_leads_s'] == [20.0, 10.0])
        for phase in 'ABC':
            file = load_yaml(f'phase_{phase}_v1_0.yaml')
            check(f'phase_{phase}_mapping', file['contract_id'] == contract_id and file['settings'] == config[f'phase_{phase}'] and file['phase'] == phase)
        data = load_yaml('dataset_plan_v1_0.yaml')
        seeds = [set(data['new_roots'][k]) for k in ('train', 'validation', 'test')]
        check('split_seed_disjoint', all(not (seeds[i] & seeds[j]) for i in range(3) for j in range(i+1,3)))
        check('root_ancestry_rules', data['root_ancestry_is_split_unit'] and data['same_root_branches_same_split'] and data['old_pair_labels_for_selector'] == 'PROHIBITED')
        tests = load_json('targeted_tests_v1_0.json')
        check('test_catalog_count', tests['test_count'] == len(tests['tests']) == 32)
        check('test_catalog_identity', len({r['id'] for r in tests['tests']}) == 32 and all(r['status']=='NOT_RUN' for r in tests['tests']))
        schema = load_json('records.schema.json')
        jsonschema.Draft202012Validator.check_schema(schema)
        sample = load_json('record_example.json')
        jsonschema.validate(sample, schema)
        check('record_schema_and_example', True)
        s = sample['scope']
        check('example_semantic_scope', set(s['selected_ids']) <= set(s['eligible_ids']) and set(s['mandatory_ids']) <= set(s['selected_ids']) and not set(s['selected_ids']) & set(s['background_ids']) and s['actual_k'] == len(s['selected_ids']))
        check('example_not_data', sample['status'] == 'EXAMPLE_NOT_RUN')
        features = load_json('feature_spec_v1_0.json')
        check('feature_boundaries', not features['raw_ids_as_numeric_model_inputs'] and not features['graph_is_safety_certificate'])
        manifest = load_json('MANIFEST.json')
        for name, expected in manifest['files'].items():
            actual = hashlib.sha256((BASE/name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Hash mismatch: {name}')
        check('manifest_hashes', True, f"{len(manifest['files'])} substantive files checked")
        report = {'status':'PASS_DOCUMENT_PACKAGE_ONLY', 'checks_passed':len(checks), 'checks':checks,
                  'simulator_executed':False, 'solver_executed':False, 'model_trained':False,
                  'repository_modified':False, 'research_tests_executed':0,
                  'scope':'Configuration, documentation, schema, example and file integrity only.'}
        (BASE/'PACKAGE_VALIDATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:report[k] for k in ['status','checks_passed','research_tests_executed','simulator_executed','model_trained']},ensure_ascii=False))
        return 0
    except Exception as exc:
        report = {'status':'FAIL_DOCUMENT_PACKAGE', 'checks':checks, 'error':str(exc), 'research_tests_executed':0}
        (BASE/'PACKAGE_VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(str(exc), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
