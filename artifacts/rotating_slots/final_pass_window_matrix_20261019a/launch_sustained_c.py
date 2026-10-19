"""One fixed sustained-c launch; plan/self-check never start a process."""
import sys
sys.dont_write_bytecode = True

import argparse
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from datetime import datetime, timezone
from unittest.mock import Mock

PROJECT = PurePosixPath('/home/user/projects/terminal')
OUTPUT_BASE = PurePosixPath('/home/user/training_outputs/terminal/rotating_slots_perf_v1')
PARALLEL = OUTPUT_BASE / 'final_pass_parallel_matrix_20261019a'
OUTPUT = OUTPUT_BASE / 'final_pass_sustained_20261019c'
LAUNCH = OUTPUT.with_name(OUTPUT.name + '_launch')
PYTHON = '/home/user/envs/terminal_env/bin/python'
DRIVER = PROJECT / 'artifacts/rotating_slots/final_pass_window_matrix_20261019a/run_matrix.py'
DEPLOYMENT = PROJECT / 'artifacts/rotating_slots/final_pass_deployment_20261019a/deployment.json'
REFERENCE = PROJECT / 'artifacts/rotating_slots/performance_reference_r0'
CORPUS = OUTPUT_BASE / 'corpus_frozen_retention_20261018a/corpus_manifest.json'
COMMIT = '4623e38d9b13238f6ca886229c603a5626285698'
DRIVER_SHA = '2cbcc6c60b8b91ce630a525f697ced2ddcef314e2ac8fac6f92bbfc29ee4fa17'
DEPLOYMENT_SHA = '85298b6a7dbe8610abf97f2103a9009edbba1d192a5e72f1786b286c113f4f5b'
SOURCE_SHA = '7d5636412e78bc59803560aec68dff753f6c73215dd8617bc0263b1cb9b27ca7'
R0_COMMIT = '1c5afd37c891a0ed4776f64367fc7264db3c1c54'
R0_SHA = 'cd76b4c14b0a48d6d2a71ec7c63ce77010e6ea019b545807635b926eb2984891'
CORPUS_RAW_SHA = '3db067bedf1049bb110405673193c3443be89b817b0b6255f9e670f22b9b40be'
CORPUS_SHA = '91d35ee0e74d0b0082d3c30d5c5ce6134b1e8d37d4da213166f01221156ee9b2'
WRAPPER_SHA = 'c9ece3c6124f361c10e1c6ddbf2a93b1babca0adf7149b41a128e8ef09ae323b'
CPUS = [0, 2, 4, 6, 8, 10]
THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'), '1')
NOTE = ('T0 basis only: sum of 10sim workers6 rollout 514.6022873040056 * 6 * 3 '
        '+ 18 * 10 overhead; 60sim density unmeasured, not a reliable ETA')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def here(server_path):
    """Only the two existing Syncthing roots, for local read-only self-check."""
    path = PurePosixPath(server_path)
    if sys.platform == 'linux':
        return Path(path)
    local_project = Path(__file__).resolve().parents[3]
    for server, local in ((PROJECT, local_project),
            (OUTPUT_BASE, local_project.parent / 'training_outputs/terminal/rotating_slots_perf_v1')):
        if path.is_relative_to(server):
            return local / path.relative_to(server)
    raise ValueError('not a fixed synced input: ' + str(path))


def read(server_path):
    return json.loads(here(server_path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def prepare():
    old_launch = PARALLEL.with_name(PARALLEL.name + '_launch')
    old = read(old_launch / 'launch_manifest.json')
    require(old['numeric_exit_code'] == 0 and old['schema'] == 'final_pass_parallel_matrix_launch_v1', 'parallel launch not complete')
    require(here(old_launch / 'numericexitcode.txt').read_text().strip() == '0', 'parallel exit is not zero')
    require(read(PARALLEL / 'completion.json')['status'] == 'FIXED_MATRIX_FINISHED', 'parallel matrix incomplete')
    require(read(PARALLEL / 'source_finish.json')['source_unchanged'] is True, 'parallel source changed')
    options = ['--corpus-manifest', str(CORPUS), '--corpus-sha256', CORPUS_SHA,
        '--reference', str(REFERENCE), '--deployment-manifest', str(DEPLOYMENT),
        '--implementation-commit', COMMIT,
        '--r0-migration-approval', str(DEPLOYMENT.parent / 'R0_migration_approval.json'),
        '--opt-migration-approval', str(DEPLOYMENT.parent / 'OPT_migration_approval.json'),
        '--serial-campaign', str(OUTPUT_BASE / 'final_pass_paired_20261019a'),
        '--cpus', *map(str, CPUS), '--budget', '1800', '--output', str(PARALLEL)]
    original = old['cli_argv']
    require(original == [str(DRIVER), 'parallel', *options], 'original fixed CLI differs')
    matrix = read(PARALLEL / 'matrix.json')
    require(matrix['driver_sha256'] == DRIVER_SHA == sha(here(DRIVER)), 'driver SHA differs')
    for path, expected in matrix['input_sha256'].items():
        require(sha(here(path)) == expected, 'fixed input changed: ' + path)
    cli = list(original)
    cli[1] = 'sustained'
    cli[cli.index('--budget') + 1] = '7200'
    cli[cli.index('--output') + 1] = str(OUTPUT)
    cli += ['--inner-workers', '6']
    flags = [v for v in cli if v.startswith('--')]
    require(len(flags) == len(set(flags)), 'duplicate CLI option')
    wrapper = old['wrapper_source']
    require(hashlib.sha256(wrapper.encode()).hexdigest() == old['wrapper_source_sha256'] == WRAPPER_SHA, 'original wrapper differs')
    changes = [('final_pass_parallel_matrix_launch_v1', 'final_pass_sustained_launch_v1'),
        ('"kernel_budget_seconds":1800', '"kernel_budget_seconds":7200'),
        ('"t0_basis_seconds":3133.459001008072,',
         '"t0_basis_seconds":9442.8411714721,\n "t0_note":' + json.dumps(NOTE) + ',')]
    for before, after in changes:
        require(wrapper.count(before) == 1, 'wrapper constant differs: ' + before)
        wrapper = wrapper.replace(before, after, 1)
    ast.parse(wrapper)
    require(wrapper.startswith('import sys;sys.dont_write_bytecode=True\n'), 'wrapper dontwrite ordering')
    for index in (2, 3, 4):
        require('sys.argv[' + str(index) + ']' in wrapper, 'wrapper positional input missing')
    enc = lambda text: base64.b64encode(text.encode()).decode('ascii')
    argv = [PYTHON, '-I', '-B', '-c', wrapper, 'unused-position-1', str(LAUNCH),
            enc(json.dumps(cli, ensure_ascii=False)), enc(wrapper)]
    simulated_argv = ['-c', *argv[5:]]
    require(simulated_argv[2] == str(LAUNCH) and
        json.loads(base64.b64decode(simulated_argv[3])) == cli and
        base64.b64decode(simulated_argv[4]).decode() == wrapper, 'wrapper argv positions differ')
    require(sum(map(len, argv)) < 30000, 'argv unexpectedly long')
    require(not here(OUTPUT).exists() and not here(LAUNCH).exists(), 'c output or launch directory exists')
    return {'schema': 'fixed_sustained_c_request_v1', 'prepared_utc': utc(),
        'cli_argv': cli, 'popen_argv': argv, 'wrapper_source': wrapper,
        'wrapper_sha256': hashlib.sha256(wrapper.encode()).hexdigest(),
        'launcher_sha256': sha(__file__), 'output': str(OUTPUT), 'launch_dir': str(LAUNCH),
        'stdout': str(LAUNCH / 'stdout.log'), 'stderr': str(LAUNCH / 'stderr.log'),
        'thread_env': THREADS, 'nice': 10, 'budget_seconds': 7200,
        'fixed_job_count': 18, 'windows': ['W71-A', 'W71-B', 'W71-C', 'W60', 'W120', 'W180'],
        'repeats': [1, 2, 3], 'duration_sim_seconds': 60, 'inner_workers': 6,
        't0_basis_seconds': 9442.8411714721, 't0_note': NOTE,
        'no_source_or_deployment_identity_change': True}


def source_preflight():
    require(sha(here(DEPLOYMENT)) == DEPLOYMENT_SHA, 'deployment file SHA differs')
    deployment = read(DEPLOYMENT)
    files = deployment['actual_source_files_sha256']
    require(len(files) == 523 and digest(files) == deployment['implementation_sha256'] == SOURCE_SHA
        and deployment['implementation_commit'] == COMMIT, 'deployment map identity differs')
    for rel, expected in files.items():
        target = Path(PROJECT / rel).resolve()
        require(target.is_relative_to(Path(PROJECT).resolve()) and sha(target) == expected, 'source bytes differ: ' + rel)
    require(Path(__file__).resolve().relative_to(PROJECT).as_posix() not in files, 'launcher unexpectedly in deployment')
    reference = read(REFERENCE / '.reference_manifest.json')
    hashes = reference['actual_source_files_sha256']
    found = {f.relative_to(REFERENCE).as_posix(): sha(f) for f in Path(REFERENCE).rglob('*')
             if f.is_file() and f.name != '.reference_manifest.json'}
    require(reference['schema'] == 'ft_rs_r0_source_export_v1_0' and
        reference['policy_id'] == 'upstream_capacity_only_pad_arrival_v1_20261018' and
        reference['reference_commit'] == R0_COMMIT and len(hashes) == 24 and
        digest(hashes) == reference['source_sha256'] == R0_SHA and found == hashes, 'R0 exact export differs')
    corpus = read(CORPUS)
    require(sha(CORPUS) == CORPUS_RAW_SHA and digest({k: v for k, v in corpus.items()
        if k != 'manifest_sha256'}) == corpus['manifest_sha256'] == CORPUS_SHA, 'corpus raw/canonical SHA differs')
    return {'deployment_file_sha256': DEPLOYMENT_SHA, 'source_files_verified': 523,
        'source_map_sha256': SOURCE_SHA, 'reference_files_verified': 24,
        'reference_source_sha256': R0_SHA, 'corpus_raw_sha256': CORPUS_RAW_SHA,
        'corpus_canonical_sha256': CORPUS_SHA, 'driver_sha256': DRIVER_SHA}


def start_child(request, env, stdout, stderr, popen=subprocess.Popen):
    return popen(request['popen_argv'], shell=False, cwd=str(PROJECT), env=env,
        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--plan', action='store_true')
    group.add_argument('--self-check', action='store_true')
    group.add_argument('--launch', action='store_true')
    parser.add_argument('--self-check-output', type=Path)
    args = parser.parse_args()
    request = prepare()
    if not args.launch:
        if args.self_check:
            intents = [(w, rep, request['inner_workers'], request['duration_sim_seconds'])
                       for w in request['windows'] for rep in request['repeats']]
            require(len(intents) == len(set(intents)) == request['fixed_job_count'] == 18
                and all(workers == 6 and duration == 60 for _, _, workers, duration in intents),
                'fixed 18-job sustained intent differs')
            require(request['stdout'] == str(LAUNCH / 'stdout.log') and
                request['stderr'] == str(LAUNCH / 'stderr.log') and len(THREADS) == 7,
                'redirect paths or native environment differ')
            mock = Mock(return_value=object())
            env = dict(os.environ, **THREADS)
            with io.BytesIO() as stdout, io.BytesIO() as stderr:
                start_child(request, env, stdout, stderr, popen=mock)
                require(mock.call_count == 1 and mock.call_args.args[0] == request['popen_argv'], 'mock argv differs')
                kwargs = mock.call_args.kwargs
                require(kwargs['shell'] is False and kwargs['start_new_session'] is True
                    and kwargs['stdin'] == subprocess.DEVNULL and kwargs['env'] == env, 'mock Popen options differ')
            output = args.self_check_output or Path(__file__).with_suffix('.selfcheck.json')
            require(output.resolve().parent == Path(__file__).resolve().parent, 'self-check output outside artifact')
            write_new(output, dict(request, status='SELF_CHECK_PASSED', popen_executed=False,
                checks=['fixed CLI/unique options', '18 fixed jobs', 'wrapper AST/argv 2/3/4',
                        'single launch Path stdout/stderr', 'seven thread fields', 'mock Popen only']))
            print(json.dumps({'status': 'SELF_CHECK_PASSED', 'output': str(output), 'popen_executed': False}))
        else:
            print(json.dumps(dict(request, mode='PLAN_ONLY', popen_executed=False), indent=2))
        return
    require(sys.platform == 'linux' and sys.flags.isolated and sys.dont_write_bytecode, 'launch requires Linux -I -B')
    require(set(CPUS) <= os.sched_getaffinity(0), 'six required CPUs unavailable')
    env = dict(os.environ, **THREADS)
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    require(os.getpriority(os.PRIO_PROCESS, 0) == 10, 'nice is not 10')
    request['source_preflight'] = source_preflight()
    request['system'] = {'available_cpu_affinity': sorted(os.sched_getaffinity(0)),
        'loadavg': list(os.getloadavg()), 'memory': [line.strip() for line in
            Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))]}
    require(not Path(OUTPUT).exists() and not Path(LAUNCH).exists(), 'c paths appeared during preflight')
    Path(LAUNCH).mkdir()
    write_new(LAUNCH / 'request.json', request)
    try:
        with Path(LAUNCH / 'stdout.log').open('xb') as stdout, Path(LAUNCH / 'stderr.log').open('xb') as stderr:
            child = start_child(request, env, stdout, stderr)
    except OSError as error:
        write_new(LAUNCH / 'launcher_result.json', {'status': 'LAUNCH_ERROR', 'pid': None,
            'observed_utc': utc(), 'error': repr(error), 'no_child_exitcode_available': True})
        raise
    write_new(LAUNCH / 'launcher_result.json', {'status': 'CHILD_STARTED_NOT_COMPLETED',
        'pid': child.pid, 'started_utc': utc(), 'popen_returned': True,
        'launcher_sha256': request['launcher_sha256'], 'no_monitor_or_supervisor_retained': True})
    print(json.dumps({'status': 'CHILD_STARTED_NOT_COMPLETED', 'pid': child.pid, 'launch_dir': str(LAUNCH)}))


if __name__ == '__main__':
    main()
