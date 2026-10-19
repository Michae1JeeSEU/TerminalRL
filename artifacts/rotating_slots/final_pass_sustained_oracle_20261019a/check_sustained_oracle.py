"""Fixed six fresh R0 60sim windows versus the three retained OPT6 runs."""
import sys
sys.dont_write_bytecode = True

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import time

PROJECT = PurePosixPath('/home/user/projects/terminal')
BASE = PurePosixPath('/home/user/training_outputs/terminal/rotating_slots_perf_v1')
OUTPUT = BASE / 'final_pass_sustained_oracle_20261019a'
OPT = BASE / 'final_pass_sustained_20261019c'
SERIAL = BASE / 'final_pass_paired_20261019a'
REFERENCE = PROJECT / 'artifacts/rotating_slots/performance_reference_r0'
DEPLOYMENT = PROJECT / 'artifacts/rotating_slots/final_pass_deployment_20261019a/deployment.json'
CORPUS = BASE / 'corpus_frozen_retention_20261018a/corpus_manifest.json'
COMMIT = '4623e38d9b13238f6ca886229c603a5626285698'
CORPUS_SHA = '91d35ee0e74d0b0082d3c30d5c5ce6134b1e8d37d4da213166f01221156ee9b2'
DEPLOYMENT_SHA = '85298b6a7dbe8610abf97f2103a9009edbba1d192a5e72f1786b286c113f4f5b'
IDS = ('W71-A', 'W71-B', 'W71-C', 'W60', 'W120', 'W180')
MEMBERS = ('states.jsonl', 'events.jsonl', 'rejections.jsonl', 'trajectories.json', 'manifest.json', 'metrics.json')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT')
BUDGET = 7200
PYTHON = '/home/user/envs/terminal_env/bin/python'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_thread_observation(observed, recorded_names):
    """Six collector fields are observed; seven variables are set at launch."""
    require(tuple(recorded_names) == THREADS[:-1], 'collector thread observation policy differs')
    require(observed == dict.fromkeys(recorded_names, '1'), 'recorded thread environment differs')
    return True


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def jobs():
    return [dict(window=w, kernel='R0', workers=0, duration=60,
                 relative_case=f'windows/{w}/R0/attempt') for w in IDS]


def approval(kernel):
    return DEPLOYMENT.parent / (kernel + '_migration_approval.json')


def command(job):
    values = [('corpus-manifest', CORPUS), ('corpus-sha256', CORPUS_SHA), ('window', job['window']),
        ('duration', 60), ('kernel', 'R0'), ('mode', 'benchmark'), ('reference', REFERENCE),
        ('deployment-manifest', DEPLOYMENT), ('implementation-commit', COMMIT),
        ('migration-approval', approval('R0')), ('inner-workers', 0), ('core-budget', 1),
        ('process-budget', 1), ('output', OUTPUT / job['relative_case'])]
    return [str(PROJECT / 'scripts/server/run_rotating_slots_performance_windows.py'), 'window',
            *[str(item) for key, value in values for item in ('--' + key, value)]]


def inventory(v):
    dep = read(DEPLOYMENT)
    require(sha(DEPLOYMENT) == DEPLOYMENT_SHA and dep['implementation_commit'] == COMMIT, 'deployment identity differs')
    expected = dep['actual_source_files_sha256']
    require(len(expected) == 523 and v.digest(expected) == dep['implementation_sha256'], 'fixed map invalid')
    actual, _ = v._source_files(Path(PROJECT), extra_paths=dep.get('extra_paths', ()), modules={})
    require(actual == expected, 'outer fixed file inventory changed')
    relative = Path(__file__).resolve().relative_to(PROJECT).as_posix()
    require(relative not in expected and relative not in dep.get('extra_paths', ()), 'oracle driver unexpectedly bound')
    ref = v._verify_export(Path(REFERENCE))
    return dict(scope='outer file-only modules={} inventory; real window retains strict runtime checks',
        source_map_sha256=v.digest(actual), reference=ref, driver_sha256=sha(__file__))


def checked_window(case, rec, window, kernel, workers, campaign, targets, expected_source, p):
    """Same window invariants, with explicit R0/OPT identities and worker policies."""
    case = Path(case)
    result = read(case / 'result.json')
    inp = read(case / 'input_identity.json')
    frozen = campaign['frozen_windows'][window]
    require(rec['returncode'] == 0 and rec['status'] == 'EXIT_ZERO' and result['status'] == 'COMPLETED_WINDOW'
        and result['failure'] is None and result['physical_outcome']['failed'] is False, 'window not physically/structurally complete')
    require(result['pid'] == rec['pid'] and result['window'] == window and result['duration_s'] == 60
        and result['inner_workers'] == workers and result['mode'] == 'benchmark'
        and result['execution_backend'] == ('R0_SERIAL' if kernel == 'R0' else 'OPT_PARALLEL'), 'result invocation differs')
    source = result['source_identity']
    require(source['kernel'] == kernel and source['runner_execution_sha256'] == targets[kernel]
        and source['implementation_commit'] == COMMIT and source['semantic_fingerprint'] == campaign['semantic_fingerprint'], 'source/semantic identity differs')
    for key in ('actual_core_files_sha256', 'helper_execution_files_sha256', 'observation_original_source_sha256', 'observation_generated_source_sha256'):
        require(source[key] == expected_source[key], 'selected core/helper/shim changed: ' + key)
    core = result['final_imported_module_paths']['core']
    core_root = REFERENCE if kernel == 'R0' else PROJECT
    require(core and all(Path(path).is_relative_to(core_root) for path in core.values()) and
        core['terminal_marl.core.rotating_slots.replay'] == str(core_root / 'terminal_marl/core/rotating_slots/replay.py'), 'actual core module path differs')
    require(result['corpus_sha256'] == CORPUS_SHA and result['workload'] == frozen['workload']
        and result['original_advance_target'] == frozen['cursor']['original_target']
        and result['outer_phase'] == frozen['cursor']['outer_phase'], 'frozen invocation changed')
    require(inp['corpus_sha256'] == CORPUS_SHA and inp['window'] == window and
        inp['selected_cut'] == frozen['cut'] and inp['source_identity'] == frozen['identity'] and
        inp['source_cursor'] == inp['runtime_cursor'] == frozen['cursor'] and
        inp['target_identity'] == dict(frozen['identity'], implementation_sha256=targets[kernel]), 'cut/migration/cursor identity differs')
    cold = result['cold_cache_evidence']
    require(result['cache_mode'] == 'cold' and cold['all_named_acceleration_caches_empty']
        and cold['module_caches_cleared'] is False and all(x == 0 for x in cold['world_entries'].values())
        and all(x.get('currsize', x.get('entries')) == 0 for x in cold['module_numerical_caches'].values()), 'cold cache evidence missing')
    guards = rec['guard_records']
    env = read(case.parent / 'collector/runtime_environment.json')
    affinity = [0] if kernel == 'R0' else [0, 2, 4, 6, 8, 10]
    require(len(guards) == 1 and guards[0]['state'] == 'DISARMED' and guards[0]['pid'] == rec['pid']
        and guards[0]['budget_seconds'] == BUDGET, 'guard completion differs')
    require(env['pid'] == rec['pid'] and env['isolated'] and env['dont_write_bytecode'] and env['nice'] == 10
        and env['affinity'] == affinity and env['core_modules_before_cli'] == []
        and check_thread_observation(env['thread_environment'], p.THREADS), 'fresh process environment differs')
    require(rec['resources']['cpu_complete'] and rec['kernel_completion_evidence']['complete']
        and result['resources']['cpu_complete'], 'successful joined CPU proof incomplete')
    if workers:
        pool = result['parallel']
        starts = pool['observed_worker_startups']
        require(pool['closed_and_joined'] and pool['worker_startup_coverage_complete'] and len(starts) == workers
            and {s['pid'] for s in starts} == set(pool['worker_pids']), 'worker startup/join incomplete')
        require(all(s['expected_parent_pid'] == s['actual_parent_pid'] == result['pid']
            and s['parent_death']['status'] == 'BOUND_TO_EXPECTED_PARENT' and s['isolated'] and s['dont_write_bytecode']
            and all(x.get('currsize', x.get('entries')) == 0 for snap in
                (s['cache_reset']['after'], s['cache_after_world_construction']) for x in snap.values()) for s in starts), 'worker parent/cold proof differs')
    else:
        require(result['parallel'] is None and result['timing']['serial_no_workers'] is True, 'R0 must have no pool')
    start, stop, delta = (result[k] for k in ('start_sim_time', 'actual_stop_sim_time', 'actual_sim_delta_s'))
    require(float(start).hex() == float(frozen['cut']['actual_sim_time']).hex() and stop >= start + 60
        and float(delta).hex() == float(stop - start).hex() and result['nominal_stop_sim_time'] == start + 60
        and result['simulation_deferral_s'] == stop - (start + 60), 'actual timeline/deferral differs')
    require(all(p.finite(x, positive=True) for x in (result['timing']['rollout_wall_s'], rec['full_child_wall_s']))
        and all(p.finite(x) for x in (rec['joined_child_cpu_s'], rec['collector_parent_cpu_s'])), 'actual timings unavailable')
    require({'output_replay_bundle_and_validate', 'source_and_deployment_finish_verify'} <=
        {span['name'] for span in result['timing']['spans']}, 'original window validation/finish span missing')
    index = read(case / 'replay/replay.json')
    require(index['config_hash'] == campaign['semantic_fingerprint']['config_hash'] and
        index['config_bytes_sha256'] == campaign['semantic_fingerprint']['physical_config_bytes_sha256'], 'archive config differs')
    archive_files = [case / 'replay/replay.json']
    for name, expected in index['files'].items():
        require(Path(name).name == name and sha(case / 'replay' / name) == expected, 'retained replay index/content differs: ' + name)
        archive_files.append(case / 'replay' / name)
    evidence = [case / 'result.json', case / 'input_identity.json', case / 'cold_cache_report.json',
        case.parent / 'collector/collector.json', case.parent / 'collector/runtime_environment.json',
        *sorted((case.parent / 'collector/deadlines').glob('deadline_*.json')),
        *archive_files]
    return dict(window=window, kernel=kernel, workers=workers, status='COMPLETED_WINDOW',
        replay=str(case / 'replay'), window_time_hex={k: float(result[k]).hex() for k in
            ('start_sim_time', 'actual_stop_sim_time', 'actual_sim_delta_s')},
        replay_sha256={name: sha(case / 'replay' / name) for name in MEMBERS},
        replay_validation_scope='ReplayBundle.validate executed by unchanged selected-core window CLI',
        replay_module_path=core['terminal_marl.core.rotating_slots.replay'], cold=True,
        guard=guards[0], cpu_complete=True, rollout_wall_s=result['timing']['rollout_wall_s'],
        full_child_wall_s=rec['full_child_wall_s'], joined_kernel_cpu_s=rec['joined_child_cpu_s'],
        collector_cpu_s=rec['collector_parent_cpu_s'], kernel_resources=result['resources'],
        collector_resources=rec['resources'], evidence_sha256={str(f): sha(f) for f in evidence})


def three_opt(rows):
    require(len(rows) == 3 and sorted(r['repeat'] for r in rows) == [1, 2, 3]
        and all(r['status'] == 'COMPLETED_WINDOW' for r in rows), 'three completed OPT repetitions required')
    require(all(r['window_time_hex'] == rows[0]['window_time_hex'] and r['replay_sha256'] == rows[0]['replay_sha256'] for r in rows), 'OPT repetitions differ')


def equal_members(left, right):
    comparisons = []
    for name in MEMBERS:
        a, b = Path(left) / name, Path(right) / name
        ah, bh = hashlib.sha256(), hashlib.sha256()
        same = True
        with a.open('rb') as af, b.open('rb') as bf:
            while True:
                x, y = af.read(1024 * 1024), bf.read(1024 * 1024)
                ah.update(x); bh.update(y)
                same = same and x == y
                if not x and not y:
                    break
        comparisons.append(dict(member=name, R0_sha256=ah.hexdigest(), OPT_sha256=bh.hexdigest(), bytes_equal=same))
    return comparisons


def compare_one(r0, opts):
    three_opt(opts)
    pairs = []
    for opt in opts:
        pairs.append(dict(OPT_repeat=opt['repeat'], exact_timeline_equal=r0['window_time_hex'] == opt['window_time_hex'],
            members=equal_members(r0['replay'], opt['replay'])))
    return dict(pairs=pairs, exact=all(p['exact_timeline_equal'] and all(m['bytes_equal'] for m in p['members']) for p in pairs))


def terminal(rows):
    if len(rows) != 6 or any(r['status'] not in ('ORACLE_EXACT', 'PERFORMANCE_TIMEOUT') for r in rows):
        return 'STOPPED_ON_INTEGRITY_OR_PHYSICS_ERROR', 2
    return ('ORACLE_INCOMPLETE_TIMEOUTS', 1) if any(r['status'] == 'PERFORMANCE_TIMEOUT' for r in rows) else ('SIX_WINDOW_60SIM_ORACLE_EXACT', 0)


def opt_inputs(p, v):
    done, meta = read(OPT / 'completion.json'), read(OPT / 'matrix.json')
    require(done['status'] == 'FIXED_MATRIX_FINISHED' and done['error'] is None
        and len(done['rows']) == 18 and done['full_kernel_children_cpu_complete'], 'NOT_READY: sustained c not complete 18/18')
    require(read(OPT.with_name(OPT.name + '_launch') / 'numericexitcode.txt') == 0, 'OPT outer exit differs')
    require(read(OPT / 'source_finish.json')['source_unchanged'] is True
        and meta['mode'] == 'sustained' and meta['source_args']['inner_workers'] == 6
        and meta['source_args']['budget'] == BUDGET and meta['source_args']['implementation_commit'] == COMMIT,
        'sustained source/backend/budget differs')
    require([(r['window'], r['workers'], r['repeat']) for r in done['rows']] ==
        [(w, 6, rep) for w in IDS for rep in (1, 2, 3)], 'OPT fixed order/coverage differs')
    campaign = read(SERIAL / 'campaign.json')
    require(campaign['status'] == 'COMPLETED_PAIRED' and campaign['windows'] == list(IDS)
        and campaign['implementation_commit'] == COMMIT and campaign['corpus_sha256'] == CORPUS_SHA, 'serial source input differs')
    targets = {}
    for kernel in ('R0', 'OPT'):
        values = [row['target_identity'] for row in read(approval(kernel))['migrations']]
        require(len({v.digest(x) for x in values}) == 1, 'migration target not unique')
        targets[kernel] = values[0]['implementation_sha256']
    require(targets == campaign['target_execution_sha256'], 'migration/serial target differs')
    expected_source = {k: read(SERIAL / f'windows/W71-A/group-1/repeat-1/{k}/attempt/result.json')['source_identity'] for k in ('R0', 'OPT')}
    tracked = {str(OPT / n): sha(OPT / n) for n in ('completion.json', 'matrix.json', 'source_preflight.json', 'source_finish.json')}
    preflight = read(OPT / 'source_preflight.json')
    require(preflight['deployment']['actual_source_files_sha256'] == read(DEPLOYMENT)['actual_source_files_sha256']
        and preflight['reference'] == v._verify_export(Path(REFERENCE)), 'sustained original source evidence differs')
    for path, expected in meta['input_sha256'].items():
        require(sha(path) == expected, 'sustained input changed: ' + path)
        tracked[path] = expected
    require(sha(PROJECT / 'artifacts/rotating_slots/final_pass_window_matrix_20261019a/run_matrix.py') == meta['driver_sha256'], 'original matrix driver changed')
    opts = {w: [] for w in IDS}
    for row in done['rows']:
        expected_case = f"windows/{row['window']}/workers-6/repeat-{row['repeat']}/attempt"
        require(row['relative_case'] == expected_case, 'OPT case path differs')
        for path, expected in row['evidence_sha256'].items():
            require(Path(path).is_relative_to(OPT) and sha(path) == expected, 'OPT retained evidence changed: ' + path)
            tracked[path] = expected
        case = Path(OPT / row['relative_case'])
        record = checked_window(case, read(case.parent / 'collector/collector.json'), row['window'],
            'OPT', 6, campaign, targets, expected_source['OPT'], p)
        record['repeat'] = row['repeat']
        require(record['replay_sha256'] == row['replay_sha256'] and record['window_time_hex'] == row['window_time_hex'], 'OPT row/actual result differs')
        tracked.update(record['evidence_sha256'])
        opts[row['window']].append(record)
    for rows in opts.values():
        three_opt(rows)
    return campaign, targets, expected_source, opts, tracked


def run():
    require(sys.platform == 'linux' and sys.flags.isolated and sys.dont_write_bytecode, 'fresh Linux -I -B required')
    require(not Path(OUTPUT).exists() and 0 in os.sched_getaffinity(0), 'new fixed output and CPU0 required')
    for name in THREADS:
        os.environ[name] = '1'
    require(os.getpriority(os.PRIO_PROCESS, 0) <= 10, 'inherited nice exceeds10')
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    sys.path.insert(0, str(PROJECT / 'terminal_marl/core/rotating_slots'))
    from performance import paired as p, provenance as v
    initial = inventory(v)
    campaign, targets, sources, opts, tracked = opt_inputs(p, v)
    from performance.windows import validate_manifest
    for w in IDS:
        validate_manifest(read(CORPUS), CORPUS_SHA, w, 60)
    tracked.update({str(x): sha(x) for x in (CORPUS, DEPLOYMENT, approval('R0'), approval('OPT'), Path(__file__))})
    Path(OUTPUT).mkdir()
    save(OUTPUT / 'source_preflight.json', initial)
    save(OUTPUT / 'request.json', dict(jobs=jobs(), commands=[command(j) for j in jobs()],
        budget_seconds=BUDGET, cpu_affinity=[0], nice=10, thread_environment=dict.fromkeys(THREADS, '1'),
        python=PYTHON, started_utc=utc(), actual_argv=list(sys.orig_argv), retained_input_sha256=tracked))
    rows, error = [], None
    status, code = 'RUNNING', 2
    try:
        for job in jobs():
            case = Path(OUTPUT / job['relative_case'])
            case.parent.mkdir(parents=True)
            rec = p.collect(case.parent / 'collector', project=Path(PROJECT), logical_cpu=0,
                cli_argv=command(job), kind='window', budget=BUDGET, python=PYTHON)
            if rec['status'] == 'PERFORMANCE_TIMEOUT':
                row = dict(job, status='PERFORMANCE_TIMEOUT', full_cpu_complete=False,
                    observed_reaped_child_cpu_lower_bound_s=rec['observed_reaped_child_cpu_lower_bound_s'],
                    collector_record=str(case.parent / 'collector/collector.json'),
                    result_path=str(case / 'result.json'), terminal_checkpoint_claim=False,
                    evidence_sha256={str(case.parent / 'collector/collector.json'): sha(case.parent / 'collector/collector.json')})
            else:
                row = checked_window(case, rec, job['window'], 'R0', 0, campaign, targets, sources['R0'], p)
                row['comparison'] = compare_one(row, opts[job['window']])
                row['status'] = 'ORACLE_EXACT' if row['comparison']['exact'] else 'SEMANTIC_DIFFERENCE'
            save(case.parent / 'row.json', row)
            rows.append(row)
            require(row['status'] != 'SEMANTIC_DIFFERENCE', 'R0/three OPT replay or exact timeline differs')
        status, code = terminal(rows)
    except BaseException as exc:
        status, code = 'STOPPED_ON_INTEGRITY_OR_PHYSICS_ERROR', 2
        error = dict(type=type(exc).__name__, message=str(exc), job=job)
    finally:
        try:
            final = inventory(v)
            require(final == initial and all(sha(path) == value for path, value in tracked.items()), 'source or retained inputs changed')
            save(OUTPUT / 'source_finish.json', dict(source_unchanged=True, inventory=final))
        except BaseException as exc:
            status, code = 'STOPPED_ON_SOURCE_ERROR', 2
            error = dict(type=type(exc).__name__, message=str(exc))
        complete = status == 'SIX_WINDOW_60SIM_ORACLE_EXACT'
        save(OUTPUT / 'completion.json', dict(status=status, numeric_exit=code, error=error, rows=rows,
            completed_R0_windows=sum(r['status'] == 'ORACLE_EXACT' for r in rows),
            comparison_pairs=sum(len(r.get('comparison', {}).get('pairs', [])) for r in rows),
            human_status='HUMAN_PENDING', speed_qualification_claim=False, finished_utc=utc(),
            full_R0_kernel_cpu_s=sum(r['joined_kernel_cpu_s'] for r in rows) if complete else None,
            outer_collector_cpu_through_final_snapshot_s=time.process_time(),
            measurement_scope='R0 one sample/window; not a speedup table. Outer CPU overlaps per-job collector; excludes final write/print tail. RSS PID sweep/HWM estimates are not strict bounds.'))
        Path(OUTPUT / 'numericexitcode.txt').write_text(str(code) + '\n', encoding='ascii')
    print(json.dumps(dict(status=status, numeric_exit=code, output=str(OUTPUT))))
    return code


def self_check(output):
    output = Path(output).resolve()
    require(output.parent == Path(__file__).resolve().parent and not output.exists(), 'fresh local artifact self-check directory required')
    output.mkdir()
    require([j['window'] for j in jobs()] == list(IDS) and len(jobs()) == 6 and BUDGET == 7200, 'six fixed jobs/budget')
    recorded = THREADS[:-1]
    observed = dict.fromkeys(recorded, '1')
    require(check_thread_observation(observed, recorded), 'six observed fields rejected')
    thread_negatives = 0
    for name in recorded:
        for mutation in ('missing', 'non1'):
            changed = dict(observed)
            if mutation == 'missing':
                changed.pop(name)
            else:
                changed[name] = '2'
            try:
                check_thread_observation(changed, recorded)
            except ValueError:
                thread_negatives += 1
    require(thread_negatives == 12 and len(THREADS) == 7 and
        dict.fromkeys(THREADS, '1')['OMP_THREAD_LIMIT'] == '1', 'thread observation negatives/launch settings differ')
    for job in jobs():
        argv = command(job)
        for flag, value in (('--kernel', 'R0'), ('--duration', '60'), ('--inner-workers', '0'), ('--core-budget', '1'), ('--process-budget', '1')):
            require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, 'fixed R0 CLI differs')
    for name in ('R0', 'OPT1', 'OPT2', 'OPT3'):
        (output / name).mkdir()
        for member in MEMBERS:
            (output / name / member).write_bytes(('SYNTHETIC_BYTE_COMPARISON_ONLY:' + member + '\n').encode())
    times = dict(start_sim_time=(300.).hex(), actual_stop_sim_time=(360.2).hex(), actual_sim_delta_s=(60.2).hex())
    hashes = {n: sha(output / 'R0' / n) for n in MEMBERS}
    r0 = dict(replay=str(output / 'R0'), window_time_hex=times)
    opts = [dict(repeat=i, status='COMPLETED_WINDOW', replay=str(output / f'OPT{i}'),
        window_time_hex=times, replay_sha256=hashes) for i in (1, 2, 3)]
    require(compare_one(r0, opts)['exact'], 'synthetic equal comparison')
    changed = copy.deepcopy(opts)
    changed[2]['window_time_hex']['actual_stop_sim_time'] = math.nextafter(360.2, math.inf).hex()
    # Keep the three OPT fixtures mutually consistent to isolate the R0/OPT check.
    for opt in changed:
        opt['window_time_hex'] = changed[2]['window_time_hex']
    require(not compare_one(r0, changed)['exact'], 'exact one-ULP timeline difference accepted')
    (output / 'OPT2' / MEMBERS[0]).write_bytes(b'SYNTHETIC_DIFFERENT_BYTES\n')
    require(not compare_one(r0, opts)['exact'], 'member byte difference accepted')
    try:
        compare_one(r0, opts[:2])
    except ValueError:
        missing_rejected = True
    else:
        missing_rejected = False
    require(missing_rejected and terminal([dict(status='PERFORMANCE_TIMEOUT') for _ in IDS]) == ('ORACLE_INCOMPLETE_TIMEOUTS', 1), 'missing OPT/timeout accepted as exact')
    ast.parse(Path(__file__).read_text())
    save(output / 'self_check.json', dict(status='SELF_CHECK_PASSED', driver_sha256=sha(__file__),
        synthetic_only=True, replay_validation_executed=False, physics_executed=False, subprocess_executed=False,
        job_count=6, budget_seconds=BUDGET, commands=[command(j) for j in jobs()],
        observed_thread_names=list(recorded), launch_thread_environment=dict.fromkeys(THREADS, '1'),
        missing_or_non1_thread_rejections=thread_negatives,
        checks=['fixed R0 commands/order', 'equal six members', 'one-ULP timeline rejected',
            'member byte difference rejected', 'missing third OPT rejected', 'timeout cannot be exact',
            'six recorded thread fields accepted; each missing/non1 rejected; seven launch variables remain1']))
    print(json.dumps(dict(status='SELF_CHECK_PASSED', output=str(output), physics_executed=False)))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'self-check', 'run', 'summarize'), nargs='?', default='plan')
    parser.add_argument('--self-check-output', type=Path)
    args = parser.parse_args()
    if args.mode == 'plan':
        print(json.dumps(dict(jobs=jobs(), commands=[command(j) for j in jobs()], output=str(OUTPUT),
            budget_seconds=BUDGET, executed=False), indent=2))
        return 0
    if args.mode == 'self-check':
        return self_check(args.self_check_output or Path(__file__).with_name('self_check_20261019a'))
    if args.mode == 'summarize':
        done = read(OUTPUT / 'completion.json')
        for row in done['rows']:
            for path, expected in row['evidence_sha256'].items():
                require(sha(path) == expected, 'saved oracle evidence changed')
        print(json.dumps(done, indent=2))
        return done['numeric_exit']
    try:
        return run()
    except (ValueError, FileNotFoundError) as error:
        print(json.dumps(dict(status='NOT_READY_OR_PREFLIGHT_ERROR', error=str(error))))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
