"""Fixed one-pass traffic orchestration artifact; no numerical acceptance gate."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

PRIORITY = ((60, 23), (120, 11), (180, 71))
CASES = PRIORITY + tuple((r, s) for r in (60, 120, 180) for s in (11, 23, 47, 71, 101)
                         if (r, s) not in PRIORITY)
MEMBERS = ('states.jsonl', 'events.jsonl', 'rejections.jsonl', 'trajectories.json', 'manifest.json', 'metrics.json')
WORKERS = (0, 1, 2, 4, 6)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def exact(value):
    if type(value) is float:
        return {'float.hex': value.hex()}
    if isinstance(value, dict):
        return {k: exact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [exact(v) for v in value]
    return value


def jobs():
    return [{'index': i * 2 + j, 'rate': rate, 'seed': seed, 'kernel': kernel,
             'relative_slot': f'jobs/rate-{rate}/seed-{seed}/{kernel}'}
            for i, (rate, seed) in enumerate(CASES) for j, kernel in enumerate(('R0', 'OPT'))]


def validate_inputs(args):
    require(len(args.implementation_commit) == 40 and all(c in '0123456789abcdef' for c in args.implementation_commit),
            'exact implementation commit required')
    require(args.cpus and len(set(args.cpus)) == len(args.cpus) and all(type(x) is int and x >= 0 for x in args.cpus),
            'nonempty distinct CPU list required')
    require(len(args.cpus) >= max(1, args.r0_workers, args.opt_workers), 'CPU list smaller than requested worker count')
    require(finite(args.hard_budget) and args.hard_budget > 0, 'positive finite per-run hard budget required')
    require(args.project.is_dir() and args.reference.is_dir() and args.deployment_manifest.is_file(), 'missing explicit input')


def command(args, job, attempt):
    workers = args.r0_workers if job['kernel'] == 'R0' else args.opt_workers
    return [str(args.project / 'scripts/server/run_rotating_slots_performance.py'), 'traffic',
            '--reference', str(args.reference), '--deployment-manifest', str(args.deployment_manifest),
            '--implementation-commit', args.implementation_commit, '--kernel', job['kernel'],
            '--rate', str(job['rate']), '--seed', str(job['seed']), '--mode', 'benchmark',
            '--inner-workers', str(workers), '--core-budget', str(len(args.cpus)),
            '--process-budget', str(workers + 1), '--wall-budget', str(args.hard_budget),
            '--cache-mode', 'cold', '--output', str(attempt)]


def helpers(project):
    sys.path.insert(0, str(project / 'terminal_marl/core/rotating_slots'))
    from performance import paired, provenance
    return paired, provenance


def file_inventory(args, provenance):
    """Outer file-only check. Real runner imports remain the runtime authority."""
    deployment = read(args.deployment_manifest)
    require(deployment.get('implementation_commit') == args.implementation_commit, 'deployment commit differs')
    expected = deployment.get('actual_source_files_sha256')
    require(expected and provenance.digest(expected) == deployment.get('implementation_sha256'), 'invalid deployment map')
    actual, _ = provenance._source_files(args.project, extra_paths=deployment.get('extra_paths', ()), modules={})
    require(actual == expected, 'outer file inventory changed')
    provenance._verify_export(args.reference)
    return {'scope': 'file-inventory check only; no outer runtime-import proof',
            'implementation_sha256': provenance.digest(actual),
            'deployment_file_sha256': sha(args.deployment_manifest), 'driver_sha256': sha(__file__)}


# A fresh validator imports the actual selected core only after the unchanged
# runner bootstrap. This is ReplayBundle.validate, never a fabricated validator.
VALIDATOR = r'''
import sys
sys.dont_write_bytecode=True
import json,pathlib
request=json.loads(sys.argv[1])
project=pathlib.Path(request['project'])
sys.path.insert(0,str(project/'terminal_marl/core/rotating_slots'))
from performance import runner
args=runner.parser().parse_args(['preflight','--reference',request['reference'],
 '--deployment-manifest',request['deployment_manifest'],'--implementation-commit',request['implementation_commit'],
 '--kernel',request['kernel']])
runtime=runner.bootstrap(args,project)
module=runtime['replay_module']
expected=runtime['core_root']/'terminal_marl/core/rotating_slots/replay.py'
assert pathlib.Path(module.__file__).resolve()==expected.resolve()
archive=module.ReplayBundle.validate(request['replay'])
runner._verify_finish(runtime,args)
print(json.dumps({'api':'ReplayBundle.validate','module_path':str(pathlib.Path(module.__file__).resolve()),
 'state_count':archive.state_count,'time_bounds':archive.time_bounds,'config_hash':archive.index['config_hash'],
 'semantic_fingerprint':runtime['source']['semantic_fingerprint'],
 'runner_execution_sha256':runtime['source']['runner_execution_sha256']},allow_nan=False))
'''


def validate_replay(args, job, slot, result):
    request = {name: str(getattr(args, name)) for name in ('project', 'reference', 'deployment_manifest', 'implementation_commit')}
    request.update(kernel=job['kernel'], replay=str(slot / 'attempt/replay'))
    started = time.monotonic()
    process = subprocess.run([sys.executable, '-I', '-B', '-c', VALIDATOR, json.dumps(request)], capture_output=True)
    (slot / 'replay_validation.stdout.log').write_bytes(process.stdout)
    (slot / 'replay_validation.stderr.log').write_bytes(process.stderr)
    (slot / 'replay_validation.numericexitcode.txt').write_text(str(process.returncode) + '\n', encoding='utf8')
    require(process.returncode == 0, 'actual replay validator failed; see raw logs')
    observed = json.loads(process.stdout)
    require(observed['semantic_fingerprint'] == result['source_identity']['semantic_fingerprint'] and
            observed['runner_execution_sha256'] == result['source_identity']['runner_execution_sha256'], 'validator/runtime identity differs')
    first, last = observed['time_bounds']
    require(float(first).hex() == 0.0.hex() and float(last).hex() == float(result['physical_outcome']['sim_time']).hex(),
            'replay does not cover exact from-zero result stop')
    return dict(observed, wall_s=time.monotonic() - started,
                cost_scope='post-collector validation subprocess; wall in driver total; CPU/RSS not in kernel collector totals')


def result_fields(result, job, workers):
    require(result['schema'] == 'ft_rs_performance_attempt_v1_0' and result['subcommand'] == 'traffic', 'not a traffic result')
    require(result['workload'] == {'kind': 'full_traffic', 'rate_per_hour': job['rate'], 'seed': job['seed'],
            'arrival_window_s': 3600., 'external_phase_targets': [3600., 5400.]}, 'traffic workload differs')
    require(result['resume_source'] is None and result['target_sim_time'] == 5400., 'not a fresh from-zero full target')
    require(result['inner_workers'] == workers and result['execution_backend'] == job['kernel'] +
            ('_PARALLEL' if workers else '_SERIAL') and result['mode'] == 'benchmark', 'kernel/backend/mode differs')
    require(result['source_identity']['kernel'] == job['kernel'] and result['failure'] is None, 'kernel or implementation error')
    require(result['status'] == result['stop_reason'] and result['status'] in
            ('COMPLETED_EPISODE', 'PHYSICAL_FAILURE', 'PERFORMANCE_TIMEOUT'), 'unexpected terminal status')
    physical = result['physical_outcome']
    require(type(physical['failed']) is bool and type(physical['episode_complete']) is bool and finite(physical['sim_time']),
            'invalid physical outcome')
    if result['status'] == 'PHYSICAL_FAILURE':
        require(physical['failed'] and not physical['episode_complete'] and finite(physical['failure_time']) and
                physical['failure_time'] <= physical['sim_time'], 'physical failure lacks actual terminal metadata')
    elif result['status'] == 'COMPLETED_EPISODE':
        require(not physical['failed'] and physical['failure_time'] is None and physical['episode_complete'] and
                physical['sim_time'] >= 5400., 'incomplete episode')
    else:
        require(not physical['failed'] and not physical['episode_complete'], 'timeout is not completed physics')
    return result['status']


def observe(args, paired, job, root):
    slot = root / job['relative_slot']
    workers = args.r0_workers if job['kernel'] == 'R0' else args.opt_workers
    started = time.monotonic()
    record = paired.collect(slot / 'collector', project=args.project, cpu_affinity=args.cpus,
                            kind='runner', cli_argv=command(args, job, slot / 'attempt'), budget=args.hard_budget)
    row = dict(job, observed_utc=utc(), outcome='INTEGRITY_ERROR', completed_physics=False,
               collector=record, collector_json_sha256=sha(slot / 'collector/collector.json'))
    try:
        environment = read(slot / 'collector/runtime_environment.json')
        require(environment['isolated'] is True and environment['dont_write_bytecode'] is True and
                environment['core_modules_before_cli'] == [] and environment['affinity'] == sorted(args.cpus) and
                environment['nice'] == 10 and environment['thread_environment'] == {n: '1' for n in paired.THREADS}, 'child environment differs')
        if record['status'] == 'PERFORMANCE_TIMEOUT':
            row['outcome'] = 'HARD_TIMEOUT'
        else:
            require(record['returncode'] == 0 and record['status'] == 'EXIT_ZERO', 'kernel startup/implementation error')
            guards = record['guard_records']
            require(len(guards) == 1 and guards[0].get('state') == 'DISARMED' and guards[0].get('pid') == record['pid']
                    and guards[0].get('budget_seconds') == args.hard_budget, 'missing actual completed hard-budget guard')
            result = read(slot / 'attempt/result.json')
            status = result_fields(result, job, workers)
            require(result['source_identity']['implementation_commit'] == args.implementation_commit, 'result commit differs')
            require(record['resources']['cpu_complete'] is True, 'successful result lacks complete joined CPU evidence')
            row.update(outcome='SOFT_BUDGET_STOP' if status == 'PERFORMANCE_TIMEOUT' else status,
                       completed_physics=status in ('PHYSICAL_FAILURE', 'COMPLETED_EPISODE'),
                       result_json_sha256=sha(slot / 'attempt/result.json'), stop_time_hex=float(result['physical_outcome']['sim_time']).hex(),
                       physical_outcome_exact=exact(result['physical_outcome']), stop_reason=result['stop_reason'],
                       semantic_fingerprint=result['source_identity']['semantic_fingerprint'],
                       kernel_timing=result['timing'], kernel_resources=result['resources'],
                       replay_validation=validate_replay(args, job, slot, result),
                       replay_sha256={name: sha(slot / 'attempt/replay' / name) for name in MEMBERS})
    except Exception as exc:
        row.update(outcome='INTEGRITY_ERROR', completed_physics=False,
                   error={'type': type(exc).__name__, 'message': str(exc)})
    row.update(driver_job_wall_s=time.monotonic() - started,
               timing_scope='driver job includes collector, validation and output checks; kernel timing spans retained without summing nested spans',
               rss_scope='collector+kernel samples and kernel+worker samples separate estimates, no unmatched peak sums')
    write(slot / 'observed.json', row)
    return row


def summarize_rows(rows):
    expected = jobs()
    require(len({r['index'] for r in rows}) == len(rows), 'duplicate observed jobs')
    require(all({key: row[key] for key in ('index', 'rate', 'seed', 'kernel', 'relative_slot')} == expected[row['index']]
                for row in rows), 'unknown/mismatched job')
    pairs, differences = [], []
    for rate, seed in CASES:
        values = {r['kernel']: r for r in rows if (r['rate'], r['seed']) == (rate, seed)}
        if len(values) != 2 or not all(r['completed_physics'] for r in values.values()):
            continue
        fields = ('replay_sha256', 'stop_time_hex', 'physical_outcome_exact', 'stop_reason', 'semantic_fingerprint')
        different = [name for name in fields if values['R0'][name] != values['OPT'][name]]
        pair = {'rate': rate, 'seed': seed, 'differences': different}
        def ratio(r0, opt):
            return r0 / opt if finite(r0) and finite(opt) and opt > 0 else None
        def rollout(row):
            spans = [s for s in row.get('kernel_timing', {}).get('spans', []) if s['name'] == 'rollout_and_record_flush']
            return spans[0]['wall_s'] if len(spans) == 1 else None
        a, b = values['R0'], values['OPT']
        pair['observed_R0_over_OPT'] = {
            'collector_child_full_wall': ratio(a.get('collector', {}).get('full_child_wall_s'), b.get('collector', {}).get('full_child_wall_s')),
            'kernel_end_to_end_wall': ratio(a.get('kernel_timing', {}).get('end_to_end_wall_s'), b.get('kernel_timing', {}).get('end_to_end_wall_s')),
            'kernel_rollout_and_record_flush_wall': ratio(rollout(a), rollout(b)),
            'collector_plus_kernel_process_tree_cpu': ratio(a.get('collector', {}).get('process_tree_cpu_s'), b.get('collector', {}).get('process_tree_cpu_s'))}
        pairs.append(pair)
        if different:
            differences.append(pair)
    return {'report_status': 'REPORT_ONLY', 'expected_cases': 15, 'expected_kernel_jobs': 30,
            'observed_kernel_jobs': len(rows), 'covered_cases': len({(r['rate'], r['seed']) for r in rows}),
            'missing_jobs': [job for job in expected if job['index'] not in {r['index'] for r in rows}],
            'outcomes': {name: sum(r['outcome'] == name for r in rows) for name in
                         ('COMPLETED_EPISODE', 'PHYSICAL_FAILURE', 'HARD_TIMEOUT', 'SOFT_BUDGET_STOP', 'INTEGRITY_ERROR')},
            'complete_pairs': pairs, 'complete_pair_count': len(pairs), 'differences': differences,
            'timeouts': [r['index'] for r in rows if r['outcome'] in ('HARD_TIMEOUT', 'SOFT_BUDGET_STOP')],
            'integrity_errors': [r for r in rows if r['outcome'] == 'INTEGRITY_ERROR'],
            'numeric_thresholds': 'observations only; no speed/wall/CPU PASS threshold',
            'human_status': 'HUMAN_PENDING'}


def summarize(root):
    root = Path(root)
    campaign = read(root / 'run.json')
    require(campaign['plan'] == jobs(), 'saved fixed plan differs')
    require(set(root.glob('jobs/**/observed.json')) <= {root / j['relative_slot'] / 'observed.json' for j in jobs()},
            'unexpected observed job')
    rows = []
    for job in jobs():
        slot = root / job['relative_slot']
        if not (slot / 'observed.json').exists():
            continue
        row = read(slot / 'observed.json')
        require(sha(slot / 'collector/collector.json') == row['collector_json_sha256'], 'collector bytes changed')
        if 'result_json_sha256' in row:
            require(sha(slot / 'attempt/result.json') == row['result_json_sha256'], 'result bytes changed')
        for name, digest in row.get('replay_sha256', {}).items():
            require(sha(slot / 'attempt/replay' / name) == digest, 'replay bytes changed')
        rows.append(row)
    return dict(summarize_rows(rows), run_status=campaign['status'], driver=campaign,
                rows=rows, cpu_scope='row collectors include owned kernel/workers only when joined proof complete; driver CPU and post-validation subprocess CPU are separate; no aggregate full-CPU claim')


def run(args):
    require(sys.platform == 'linux' and sys.flags.isolated and sys.dont_write_bytecode, 'formal run requires fresh Linux -I -B')
    validate_inputs(args)
    require(set(args.cpus) <= os.sched_getaffinity(0), 'requested CPU affinity unavailable')
    require(not args.output.exists(), 'new output only; no retries/resume/overwrite')
    paired, provenance = helpers(args.project)
    before = file_inventory(args, provenance)
    root = args.output
    require(not root.is_relative_to(args.project), 'formal output must be outside the synchronized project tree')
    require(not root.is_relative_to(args.reference) and not args.reference.is_relative_to(root), 'output overlaps reference')
    root.mkdir(parents=True, exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    campaign = {'schema': 'one_pass_traffic_artifact_v1', 'started_utc': utc(), 'status': 'RUNNING', 'plan': jobs(),
                'inputs': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'file_inventory_before': before, 'policy': 'one fixed pass, no retries/reseed/tuning/resume; timeout continues; integrity stops'}
    write(root / 'run.json', campaign)
    rows = []
    try:
        for job in jobs():
            require(file_inventory(args, provenance) == before, 'outer input/driver inventory changed')
            row = observe(args, paired, job, root)
            rows.append(row)
            require(file_inventory(args, provenance) == before, 'outer input/driver inventory changed')
            partial = summarize_rows(rows)
            write(root / 'summary.json', dict(partial, run_status='RUNNING'))
            if row['outcome'] == 'INTEGRITY_ERROR' or partial['differences']:
                raise ValueError('integrity/semantic difference; remaining fixed jobs not launched')
        campaign['status'] = 'FINISHED_ONE_PASS'
    except Exception as exc:
        campaign.update(status='STOPPED_ON_INTEGRITY_ERROR', error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        try:
            campaign['file_inventory_after'] = file_inventory(args, provenance)
            require(campaign['file_inventory_after'] == before, 'outer input/driver inventory changed at finish')
        except Exception as exc:
            campaign.update(status='STOPPED_ON_INTEGRITY_ERROR', final_inventory_error={'type': type(exc).__name__, 'message': str(exc)})
        campaign.update(finished_utc=utc(), driver_end_to_end_wall_s=time.monotonic() - started,
                        driver_run_interval_parent_cpu_s=time.process_time() - cpu,
                        cpu_scope='driver parent only; nested collector CPU and post-validation subprocess not added; final JSON/stdout excluded')
        write(root / 'run.json', campaign)
    output = summarize(root)
    write(root / 'summary.json', output)
    return output


def self_check():
    """Pure synthetic protocol checks; deliberately no collector/core execution."""
    from argparse import Namespace
    plan = jobs()
    assert len(CASES) == len(set(CASES)) == 15 and CASES[:3] == PRIORITY
    assert len(plan) == 30 and [r['kernel'] for r in plan[:6]] == ['R0', 'OPT'] * 3
    args = Namespace(project=Path('/project'), reference=Path('/reference'), deployment_manifest=Path('/deployment.json'),
                     implementation_commit='a' * 40, cpus=[1, 2], r0_workers=0, opt_workers=2, hard_budget=7200.)
    argv = command(args, plan[1], Path('/output'))
    assert argv[1] == 'traffic' and '--checkpoint' not in argv and argv[argv.index('--inner-workers') + 1] == '2'
    base = {'schema': 'ft_rs_performance_attempt_v1_0', 'subcommand': 'traffic', 'workload': {
        'kind': 'full_traffic', 'rate_per_hour': 60, 'seed': 23, 'arrival_window_s': 3600., 'external_phase_targets': [3600., 5400.]},
        'resume_source': None, 'target_sim_time': 5400., 'inner_workers': 0, 'execution_backend': 'R0_SERIAL',
        'mode': 'benchmark', 'source_identity': {'kernel': 'R0'}, 'failure': None,
        'status': 'PHYSICAL_FAILURE', 'stop_reason': 'PHYSICAL_FAILURE',
        'physical_outcome': {'failed': True, 'episode_complete': False, 'failure_time': 3.125, 'sim_time': 3.125}}
    assert result_fields(base, plan[0], 0) == 'PHYSICAL_FAILURE'
    rows = [dict(job, outcome='HARD_TIMEOUT', completed_physics=False) for job in plan]
    assert summarize_rows(rows)['timeouts'] == list(range(30))
    for row in rows[:2]:
        row.update(outcome='PHYSICAL_FAILURE', completed_physics=True, replay_sha256={n: 'synthetic' for n in MEMBERS},
                   stop_time_hex=3.125.hex(), physical_outcome_exact=exact(base['physical_outcome']),
                   stop_reason='PHYSICAL_FAILURE', semantic_fingerprint={'synthetic': True})
    assert summarize_rows(rows)['complete_pair_count'] == 1 and not summarize_rows(rows)['differences']
    rows[1]['stop_time_hex'] = 3.126.hex()
    assert summarize_rows(rows)['differences'][0]['differences'] == ['stop_time_hex']
    return {'status': 'SELF_CHECK_COMPLETE', 'synthetic_only': True, 'planned_cases': 15, 'planned_kernel_jobs': 30,
            'physics_executed': False, 'collector_executed': False, 'replay_validation_executed': False}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    modes = result.add_subparsers(dest='mode', required=True)
    for name in ('plan', 'run'):
        command_parser = modes.add_parser(name)
        for option in ('project', 'reference', 'deployment-manifest', 'output'):
            command_parser.add_argument('--' + option, type=lambda s: Path(s).resolve(), required=True)
        command_parser.add_argument('--implementation-commit', required=True)
        command_parser.add_argument('--r0-workers', type=int, choices=WORKERS, required=True)
        command_parser.add_argument('--opt-workers', type=int, choices=WORKERS, required=True)
        command_parser.add_argument('--cpus', type=int, nargs='+', required=True)
        command_parser.add_argument('--hard-budget', type=float, default=7200.)
    command_parser = modes.add_parser('summarize')
    command_parser.add_argument('--campaign', type=Path, required=True)
    modes.add_parser('self-check')
    return result


def main():
    args = parser().parse_args()
    try:
        require(sys.flags.isolated and sys.dont_write_bytecode, 'launch with -I -B')
        if args.mode == 'self-check':
            output = self_check()
        elif args.mode == 'plan':
            validate_inputs(args)
            output = {'status': 'PLAN_ONLY', 'plan': [dict(j, command=command(args, j, args.output / j['relative_slot'] / 'attempt')) for j in jobs()],
                      'driver_sha256': sha(__file__), 'formal_jobs_launched': 0}
        elif args.mode == 'summarize':
            output = summarize(args.campaign)
        else:
            output = run(args)
        print(json.dumps(output, allow_nan=False))
        return 1 if output.get('run_status') == 'STOPPED_ON_INTEGRITY_ERROR' or output.get('integrity_errors') or output.get('differences') else 0
    except Exception as exc:
        print(json.dumps({'status': 'REJECTED', 'error': {'type': type(exc).__name__, 'message': str(exc)}}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
