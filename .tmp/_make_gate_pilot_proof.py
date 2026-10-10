import hashlib
import json
from pathlib import Path

out_root = Path(r'D:\7_codespace\training_outputs\terminal\p3_followup_20260930\gate_release_control')
ref_root = Path(r'D:\7_codespace\training_outputs\terminal\p3_followup_20260930\forced_wait_frozen')
pilot = json.loads((out_root / 'pilot.json').read_text(encoding='utf-8'))
pair = json.loads((out_root / 'pairs' / 'pair_0001.json').read_text(encoding='utf-8'))
control = json.loads((out_root / 'arms' / 'pair_0001_control.json').read_text(encoding='utf-8'))
reference = json.loads((ref_root / 'arms' / 'pair_0001_legacy.json').read_text(encoding='utf-8'))
preflight = json.loads((out_root / 'preflight.json').read_text(encoding='utf-8'))
source_binding = preflight['runtime']['source_binding']

def identity_digest(rows):
    payload = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def metrics(arm):
    cm = arm.get('control_metrics', arm)
    return {
        k: cm.get(k)
        for k in ('status', 'generated', 'admitted', 'completed', 'resolved', 'normal', 'emergency', 'failed', 'diverted', 'unresolved', 'wait_seconds', 'request_elapsed_total_seconds', 'makespan_seconds', 'energy_total_consumed', 'reward_total', 'raw_reward_excluding_rejection_penalty', 'release_rejection_penalty')
    }

checks = pair['reference_control_checks']
control_identity = control['request_identity']
reference_identity = reference['request_identity']
proof = {
    'phase': 'pilot',
    'status': pilot['status'],
    'pair_status': pair['status'],
    'pair': pair['pair'],
    'PPO_ALLOWED': pilot['PPO_ALLOWED'],
    'off_policy': pilot['off_policy'],
    'training': pair['training'],
    'censored': control['censored'],
    'technical_error': control['error'],
    'control_metrics': metrics(pair),
    'reference_metrics': metrics({'control_metrics': pair['reference_metrics']}),
    'complete_scene_delta': pair['complete_scene_delta'],
    'penalty_split': {
        'control': {
            'reward_total': control['reward_total'],
            'raw_reward_excluding_rejection_penalty': control['raw_reward_excluding_rejection_penalty'],
            'release_rejection_penalty': control['release_rejection_penalty'],
        },
        'reference': {
            'reward_total': reference['reward_total'],
            'raw_reward_excluding_rejection_penalty': reference['raw_reward_excluding_rejection_penalty'],
            'release_rejection_penalty': reference['release_rejection_penalty'],
        },
    },
    'request_identity': {
        'control_count': len(control_identity),
        'reference_count': len(reference_identity),
        'control_unique': control['request_identity_unique'],
        'reference_unique': reference['request_identity_unique'],
        'ids': [int(row['arrival_id']) for row in control_identity],
        'exact_equal': control_identity == reference_identity,
        'control_sha256': identity_digest(control_identity),
        'reference_sha256': identity_digest(reference_identity),
        'pair_check': checks['request_identity_match'],
    },
    'ledger_energy_wait': {
        'energy_coverage': control['energy_ledger']['coverage'],
        'unique_energy_rows': control['energy_ledger']['coverage']['unique_arrival_ids'],
        'ledger': control['ledger'],
        'wait_integrity': control['wait_integrity'],
        'wait_metrics': control['wait_metrics'],
    },
    'source_model_rng_safety': {
        'model_digest_before': control['model_digest_before'],
        'model_digest_after': control['model_digest_after'],
        'model_unchanged': control['model_unchanged'],
        'caller_rng_unchanged': control['caller_rng_unchanged'],
        'safety': control['safety'],
        'reference_control_checks': checks,
    },
    'control_intervention': {
        'name': control['control']['name'],
        'forced_release': control['control']['forced_release'],
        'canonical_wait': control['control']['canonical_wait'],
        'mask_counts': control['mask_counts'],
        'logprob_recorded': control['control']['logprob_recorded'],
    },
    'novel_rows': {
        'failed_arrival_ids': pair['new_failed_arrival_ids'],
        'diverted_arrival_ids': pair['new_diverted_arrival_ids'],
        'unresolved_arrival_ids': pair['new_unresolved_arrival_ids'],
        'failure_comparison_known': pair['failure_comparison_known'],
    },
    'binding': {
        'preflight_status': preflight['status'],
        'preflight_runner_sha256': pilot['preflight_runner_sha256'],
        'source_checkpoint_sha256': source_binding['binding']['checkpoint_sha256'],
        'warmup_checkpoint_sha256': source_binding['warmup']['checkpoint_sha256'],
        'source_model_digest': source_binding['source_model_digest'],
        'pair_ids': preflight['control']['pair_ids'],
        'proposal_sampling': preflight['control']['proposal_sampling'],
        'shortlist': preflight['control']['shortlist'],
        'max_steps': preflight['control']['max_steps'],
        'runtime_k': preflight['runtime']['config_snapshot']['runtime_proposal_shortlist_k'],
        'mask_mode': preflight['runtime']['config_snapshot']['student_decision_mask_mode'],
        'cap_seconds': preflight['runtime']['config_snapshot']['max_simulation_seconds'],
    },
    'artifacts': {
        'pilot': str(out_root / 'pilot.json'),
        'pair': str(out_root / 'pairs' / 'pair_0001.json'),
        'control_arm': str(out_root / 'arms' / 'pair_0001_control.json'),
        'reference_arm': str(ref_root / 'arms' / 'pair_0001_legacy.json'),
    },
}
Path(r'D:\7_codespace\terminal\.tmp\p3_gate_control_pilot_proof.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({
    'proof': str(Path(r'D:\7_codespace\terminal\.tmp\p3_gate_control_pilot_proof.json')),
    'status': proof['status'],
    'pair_status': proof['pair_status'],
    'request_count': proof['request_identity']['control_count'],
    'identity_exact': proof['request_identity']['exact_equal'],
    'control_outcomes': proof['control_metrics'],
    'reference_outcomes': proof['reference_metrics'],
    'checks': proof['source_model_rng_safety']['reference_control_checks'],
}, ensure_ascii=False, sort_keys=True))
