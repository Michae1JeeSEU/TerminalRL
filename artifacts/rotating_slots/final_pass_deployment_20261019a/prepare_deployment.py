"""Prepare hash-only migration after primary review of the final Linux suite.

The default self-check reads inputs and parses Python only. It never issues a
migration approval, launches a server or restores/advances a corpus checkpoint.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = Path('D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1')
SERVER = '/home/user/training_outputs/terminal/rotating_slots_perf_v1'
SUITE = 'final_pass_linux_tests_20261019c'
REFERENCE = ROOT / 'artifacts/rotating_slots/performance_reference_r0'
CORPUS = BASE / 'corpus_frozen_retention_20261018a/corpus_manifest.json'
CORPUS_SHA = '91d35ee0e74d0b0082d3c30d5c5ce6134b1e8d37d4da213166f01221156ee9b2'
HELPER = 'terminal_marl/core/rotating_slots/performance/checkpoint.py'
HELPER_SHA = 'bf482dcb174150de23e0555191f08069ce6ca7311a1f293463edcd0ed979f3d5'
MATRIX_REL = 'artifacts/rotating_slots/final_pass_window_matrix_20261019a/run_matrix.py'
GENERATED = ('deployment.json', 'artifact_manifest.json',
             'R0_migration_approval.json', 'OPT_migration_approval.json', 'R0', 'OPT')
PREFLIGHT_CHILD = textwrap.dedent(r'''
    import sys
    sys.dont_write_bytecode=True
    import argparse,json,pathlib
    from dataclasses import asdict
    from datetime import datetime,timezone
    root,reference,deployment,output=map(pathlib.Path,sys.argv[1:5])
    kernel,commit=sys.argv[5:7]
    sys.path.insert(0,str(root/'terminal_marl/core/rotating_slots'))
    from performance import provenance,runner,windows
    from performance.checkpoint import atomic_write_json
    started=datetime.now(timezone.utc).isoformat()
    assert sys.flags.isolated and sys.dont_write_bytecode
    assert not runner._existing_core_modules()
    args=argparse.Namespace(kernel=kernel,reference=reference,deployment_manifest=deployment,implementation_commit=commit)
    runtime=runner.bootstrap(args,root)
    windows.bind_window_entrypoint(runtime,kernel)
    identity=runner._checkpoint_identity(runtime)
    assert len(runtime['reference']['actual_source_files_sha256'])==24
    imported=runner._verify_finish(runtime,args)
    result={'schema':'window_deployment_preflight_v1','kernel':kernel,'status':'PREFLIGHT_OK',
        'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'command':list(sys.orig_argv),'source_identity':runtime['source'],'checkpoint_identity':asdict(identity),
        'checkpoint_helper_sha256':windows._sha(root/'terminal_marl/core/rotating_slots/performance/checkpoint.py'),
        'observation_original_source_sha256':runtime['observed'].original_source_sha256,
        'observation_generated_source_sha256':runtime['observed'].generated_source_sha256,
        'reference_source_file_count':len(runtime['reference']['actual_source_files_sha256']),
        'final_imported_module_paths':imported,'isolated':bool(sys.flags.isolated),
        'dont_write_bytecode':sys.dont_write_bytecode,'advanced':False,'restored_real_source':False,
        'performance_status':'NOT_QUALIFIED','human_status':'HUMAN_PENDING'}
    atomic_write_json(output,result)
    print(json.dumps({'kernel':kernel,'status':result['status'],'execution_sha256':identity.implementation_sha256}))
''')


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def inputs():
    check(sys.flags.isolated and sys.dont_write_bytecode, 'use python -I -B')
    check(not any(n == 'terminal_marl' or n.startswith('terminal_marl.') for n in sys.modules),
          'core already imported in generator')
    sys.path.insert(0, str(ROOT / 'terminal_marl/core/rotating_slots'))
    from performance import provenance, windows
    reference = provenance._verify_export(REFERENCE)
    check(provenance.R0_COMMIT == '1c5afd37c891a0ed4776f64367fc7264db3c1c54', 'R0 commit differs')
    check(len(reference['actual_source_files_sha256']) == 24 and
          len([p for p in REFERENCE.rglob('*') if p.is_file()]) == 25, 'fixed R0 inventory differs')
    check(sha(ROOT / HELPER) == HELPER_SHA, 'checkpoint helper differs')
    corpus = read(CORPUS)
    windows.validate_manifest(corpus, CORPUS_SHA, 'W71-A', 10)
    return provenance, windows, reference, corpus


def fresh_suite(provenance, deployment, commit):
    """Read synced bytes; retain original Linux evidence paths in approvals."""
    folder = BASE / SUITE
    run = read(folder / 'run_metadata.json')
    check(run['schema'] == 'final_pass_linux_tests_run_v1' and
          run['exit_code'] == run['pytest_exit_code'] == run['pytest_session_exitstatus'] == 0,
          'final Linux suite did not exit successfully')
    check(run['implementation_commit'] == commit and run['preflight_passed'] is True and
          run['pytest_started'] is True and run['pytest_testsfailed'] == 0 and
          run['run_window_integration'] == '1' and run['failure'] is None and
          not run.get('source_finish_error'), 'final Linux metadata is incomplete or stale')
    check((folder / 'numericexitcode.txt').read_text().strip() == '0', 'numeric suite exit differs')
    check(run['project'] == '/home/user/projects/terminal' and run['output'] == SERVER + '/' + SUITE,
          'unexpected Linux evidence location')
    initial = read(folder / 'source_preflight.json')
    finish = read(folder / 'source_finish.json')
    gate = read(folder / 'source_gate.json')
    tested = initial['deployment']['actual_source_files_sha256']
    check(finish['source_unchanged'] is True and not finish.get('verification_error') and
          finish['source_files_sha256'] == tested and
          finish['source_manifest_sha256'] == provenance.digest(tested) == initial['deployment']['implementation_sha256'],
          'fresh Linux source finish verification differs')
    check(initial['deployment']['implementation_commit'] == commit and
          initial['reference'] == finish['reference'] == provenance._verify_export(REFERENCE),
          'fresh suite reference or label differs')
    request = gate['request']
    check(request['schema'] == 'final_pass_linux_tests_source_v1' and request['tests'] == run['tests'] and
          request['implementation_commit'] == commit and gate['sha_match'] is True and
          gate['actual_source_sha256'] == request['files'] == finish['source_gate_sha256'],
          'fresh Linux source gate differs')
    check(all(tested.get(rel) == value for rel, value in request['files'].items()),
          'fresh source gate files absent from tested inventory')
    for rel, value in tested.items():
        check(sha(ROOT / rel) == value, 'tested local source changed: ' + rel)
    # The Linux harness adds its own drivers/tests explicitly. Production
    # discovery remains the existing provenance inventory, without these extras.
    baseline = provenance.implementation_fingerprint(ROOT, modules={})['actual_source_files_sha256']
    fixture_before=read(folder/'runner_fixture_preflight.json')
    fixture_after=read(folder/'runner_fixture_finish.json')
    declared=request['runner_fixture']; fixture_path=(ROOT/declared['relative_path']).resolve()
    check(fixture_path.parent==(ROOT/'artifacts/rotating_slots/final_pass_linux_tests_20261019a').resolve() and
          fixture_path.name.startswith('runner_test_deployment_') and fixture_path.suffix=='.json', 'runner fixture path differs')
    fixture=read(fixture_path)
    check(fixture_before==fixture_after and fixture_before['fixture']==fixture and
          fixture_before['relative_path']==declared['relative_path'] and
          fixture_before['file_sha256']==declared['file_sha256']==sha(fixture_path) and
          fixture_before['production_map_equal'] is True and
          fixture_before['implementation_commit']==fixture['implementation_commit']==declared['implementation_commit']==commit and
          fixture_before['implementation_sha256']==fixture['implementation_sha256']==declared['implementation_sha256']==provenance.digest(baseline) and
          fixture['actual_source_files_sha256']==baseline and fixture['extra_paths']==[] and fixture['imported_module_paths']=={},
          'locally supplied runner fixture evidence differs')
    check(declared['relative_path'] not in initial['deployment']['extra_paths'] and declared['relative_path'] not in tested,
          'metadata fixture must not enter source inventory')
    harness_extras = set(initial['deployment']['extra_paths']) - set(baseline)
    allowed_extras = {'artifacts/rotating_slots/final_pass_linux_tests_20261019a/run_tests.py',
                      'artifacts/rotating_slots/final_pass_linux_tests_20261019a/run_linux_tests.ps1', *run['tests']}
    check(harness_extras == allowed_extras - set(baseline), 'unexpected harness-only extra source')
    check({k: v for k, v in tested.items() if k not in harness_extras} == baseline,
          'production source inventory differs from fresh Linux suite')
    tooling = {Path(__file__).resolve().relative_to(ROOT).as_posix(): sha(__file__), MATRIX_REL: sha(ROOT / MATRIX_REL)}
    check(deployment['actual_source_files_sha256'] == dict(baseline, **tooling),
          'deployment has untested production files or unlisted tooling extras')
    collection = read(folder / 'pytest_collection.json')
    nodes = collection['nodeids']
    check(isinstance(collection['collected_tests'], int) and collection['collected_tests'] > 0 and
          collection['collected_tests'] == len(nodes) == len(set(nodes)), 'invalid actual pytest collection')
    check(set(run['tests']) == {node.split('::')[0] for node in nodes}, 'collected test files differ')
    log = (folder / 'stdout.log').read_text(encoding='utf8')
    check(re.search(r'\b' + str(len(nodes)) + r' passed\b', log) and
          not re.search(r'\b\d+ (?:failed|skipped|error|errors|xfailed|xpassed)\b', log),
          'suite log is not an all-collected-tests pass; raw log tail:\n' + log[-2500:])
    required = [folder / n for n in ('run_metadata.json', 'stdout.log', 'stderr.log', 'numericexitcode.txt',
        'source_gate.json', 'source_preflight.json', 'source_finish.json', 'pytest_argv.json', 'pytest_collection.json',
        'runner_fixture_preflight.json', 'runner_fixture_finish.json')]
    patterns = ('bernstein_equivalence*/provenance.json', 'root_reuse*/evidence.json',
        'peak_reuse*/evidence.json', 'energy_queries*/provenance.json', 'equivalence*/comparison_provenance.json',
        'test_real_cold_window_fixture_*/fixture_evidence.json',
        'test_real_startup_all_workers_*/startup.json',
        'test_reconstruction_matches_in*/reconstruction.json',
        'test_real_owner_death_ex*/parent_death.json',
        'test_deadline_helper_identity_*/source_identity.json')
    anchors = []
    for pattern in patterns:
        found = sorted((folder / 'pytest').glob(pattern))
        check(bool(found), 'missing fresh oracle/lifecycle output: ' + pattern)
        anchors.extend(found)
    # Preserve the exact oracle files plus their sibling JSON/process/log data.
    for anchor in anchors:
        required.extend(p for p in anchor.parent.iterdir() if p.is_file() and p.suffix in {'.json', '.log', '.txt'})
        if anchor.name == 'fixture_evidence.json':
            fixture = read(anchor)
            variants = {'benchmark', 'profile', 'parallel', 'migration'}
            check(len(fixture) == len(variants) and
                  {PurePosixPath(row['output']).name for row in fixture} == variants and
                  all(row['output'] == SERVER + '/' + anchor.parent.relative_to(BASE).as_posix() + '/' +
                      PurePosixPath(row['output']).name and row['status'] == 'COMPLETED_WINDOW' and
                      row['source_unchanged'] is True and row['same_target_records_equal'] is True and
                      row['sim_delta'] >= 10 for row in fixture),
                  'fresh real-window fixture differs')
            for variant in sorted(variants):
                result_path = anchor.parent / variant / 'result.json'
                cold_path = anchor.parent / variant / 'cold_cache_report.json'
                result, cold = read(result_path), read(cold_path)
                check(result['status'] == 'COMPLETED_WINDOW' and
                      result['performance_status'] == 'NOT_QUALIFIED' and result['human_status'] == 'HUMAN_PENDING',
                      'fresh fixture result differs: ' + variant)
                check(cold['all_named_acceleration_caches_empty'] is True and cold['module_caches_cleared'] is False and
                      all(value == 0 for value in cold['world_entries'].values()) and
                      all(value.get('currsize', value.get('entries')) == 0 for value in cold['module_numerical_caches'].values()),
                      'fresh fixture cold-cache evidence differs: ' + variant)
                required.extend((result_path, cold_path))
                if variant == 'parallel':
                    parallel = result['parallel']
                    check(result['execution_backend'] == 'R0_PARALLEL' and result['inner_workers'] == 1 and
                          parallel['closed_and_joined'] is True and parallel['worker_startup_coverage_complete'] is True and
                          len(parallel['worker_pids']) == len(parallel['observed_worker_startups']) == 1 and
                          result['resources']['cpu_complete'] is True and result['timing']['worker_cpu_s'] >= 0,
                          'fresh parallel fixture startup/join evidence differs')
                    for startup in parallel['observed_worker_startups']:
                        for snapshot in (startup['cache_reset']['after'], startup['cache_after_world_construction']):
                            check(all(value.get('currsize', value.get('entries')) == 0 for value in snapshot.values()),
                                  'fresh parallel fixture worker startup cache not empty')
    evidence = {}
    for path in sorted(set(required)):
        check(path.is_file(), 'missing fresh evidence: ' + str(path))
        rel = path.relative_to(BASE).as_posix()
        evidence[rel] = {'local_path': str(path), 'server_path': SERVER + '/' + rel,
                         'sha256': sha(path), 'size_bytes': path.stat().st_size}
    return evidence, collection, sorted(harness_extras), provenance.digest(baseline)


def generate(note):
    check(not any((OUT / name).exists() for name in GENERATED), 'generated output exists; refusing overwrite')
    provenance, windows, reference, corpus = inputs()
    from performance.checkpoint import CheckpointIdentity, atomic_write_json
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    check(bool(re.fullmatch('[0-9a-f]{40}', commit)), 'exact local HEAD label required')
    started = stamp()
    check((ROOT / MATRIX_REL).is_file(), 'reviewed matrix orchestration script missing')
    ast.parse((ROOT / MATRIX_REL).read_text(encoding='utf8'))
    deployment = provenance.implementation_fingerprint(ROOT,
        extra_paths=(Path(__file__).resolve(), ROOT / MATRIX_REL), modules={})
    deployment['implementation_commit'] = commit
    evidence, collection, harness_extras, production_digest = fresh_suite(provenance, deployment, commit)
    identities = {}
    for row in corpus['windows']:
        metadata = corpus['sources'][row['case']]['source_identity']
        identity = CheckpointIdentity(**row['validation']['identity'])
        check(identity.implementation_sha256 == metadata['runner_execution_sha256'] and
              metadata['helper_execution_files_sha256'][HELPER] == HELPER_SHA, 'frozen identity/helper differs')
        entry = identities.setdefault(provenance.digest(asdict(identity)),
            {'identity': asdict(identity), 'metadata': metadata, 'windows': []})
        check(all(entry['metadata'][k] == metadata[k] for k in
              ('observation_original_source_sha256', 'observation_generated_source_sha256')), 'source shims differ')
        entry['windows'].append(row['window'])
    check(len(identities) == 2, 'expected two frozen implementation identities')
    # No approval or deployment writes occur before all fresh-suite checks above.
    atomic_write_json(OUT / 'deployment.json', deployment)
    targets = {}; commands = []
    for kernel in ('R0', 'OPT'):
        destination = OUT / kernel / 'preflight.json'
        command = [sys.executable, '-I', '-B', '-c', PREFLIGHT_CHILD, str(ROOT), str(REFERENCE),
                   str(OUT / 'deployment.json'), str(destination), kernel, commit]
        begin = stamp()
        try:
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired as error:
            def decoded(value):
                return value.decode('utf8', errors='replace') if isinstance(value, bytes) else value
            atomic_write_json(OUT / kernel / 'preflight_process.json',
                {'command': command, 'started_utc': begin, 'finished_utc': stamp(),
                 'returncode': None, 'timed_out_seconds': 120,
                 'stdout': decoded(error.stdout), 'stderr': decoded(error.stderr)})
            raise
        record = {'command': command, 'started_utc': begin, 'finished_utc': stamp(),
                  'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
        atomic_write_json(OUT / kernel / 'preflight_process.json', record)
        commands.append(record)
        check(process.returncode == 0, 'preflight failed: ' + kernel + ' ' + process.stdout + ' ' + process.stderr)
        targets[kernel] = read(destination)
    support = [{'path': entry['server_path'], 'sha256': entry['sha256']} for entry in evidence.values()]
    for kernel, target in targets.items():
        check(target['checkpoint_helper_sha256'] == HELPER_SHA, 'target helper differs')
        target_identity = target['checkpoint_identity']; pairs = []
        for source in identities.values():
            identity = source['identity']; metadata = source['metadata']
            check(identity['implementation_sha256'] != target_identity['implementation_sha256'] and
                  {k: v for k, v in identity.items() if k != 'implementation_sha256'} ==
                  {k: v for k, v in target_identity.items() if k != 'implementation_sha256'},
                  'migration would change semantics/config or has identical identity')
            check(all(metadata[k] == target[k] for k in
                  ('observation_original_source_sha256', 'observation_generated_source_sha256')), 'migration would change shim')
            pairs.append({'migration_id': 'final-pass-20261019a-' + commit[:12] + '-' + kernel.lower() + '-' + identity['implementation_sha256'][:12],
                'source_identity': identity, 'target_identity': target_identity,
                'source_observation_original_sha256': metadata['observation_original_source_sha256'],
                'source_observation_generated_sha256': metadata['observation_generated_source_sha256'],
                'frozen_windows': source['windows']})
        approval = {'schema': 'window_migration_approval_v1', 'qualified': True, 'registered_utc': stamp(),
            'corpus_sha256': CORPUS_SHA, 'target_execution_sha256': target_identity['implementation_sha256'],
            'target_observation_original_sha256': target['observation_original_source_sha256'],
            'target_observation_generated_sha256': target['observation_generated_source_sha256'],
            'checkpoint_helper_sha256': HELPER_SHA, 'supporting_evidence': support, 'migrations': pairs,
            'qualification_scope': 'Primary explicitly reviewed successful fresh Linux evidence and approved implementation-hash migration only. User amendment: speed targets are report-only in the single final pass. This does not establish performance acceptance or visual acceptance.',
            'authorization': {'primary_approval_note': note,
                              'review_scope': 'Fresh final Linux suite and finalized matrix orchestration reviewed by primary.',
                              'implementation_commit': commit,
                              'fresh_linux_suite': SERVER + '/' + SUITE, 'source_manifest_sha256': production_digest},
            'performance_status': 'NOT_QUALIFIED', 'human_status': 'HUMAN_PENDING'}
        atomic_write_json(OUT / (kernel + '_migration_approval.json'), approval)
    for entry in evidence.values():
        check(sha(entry['local_path']) == entry['sha256'], 'support evidence changed during generation')
    provenance.verify_deployment(ROOT, deployment, supplied_commit=commit)
    provenance._verify_export(REFERENCE, reference)
    files = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    manifest = {'schema': 'window_deployment_artifact_manifest_v1', 'status': 'PREPARED_LOCAL_ONLY',
        'started_utc': started, 'finished_utc': stamp(), 'implementation_commit': commit,
        'corpus_path_local': str(CORPUS), 'corpus_path_server': SERVER + '/corpus_frozen_retention_20261018a/corpus_manifest.json',
        'corpus_sha256': CORPUS_SHA, 'deployment_implementation_sha256': deployment['implementation_sha256'],
        'production_source_sha256': production_digest, 'excluded_linux_harness_extras': harness_extras,
        'deployment_tooling_extras_sha256': {p: sha(ROOT / p) for p in deployment['extra_paths']},
        'matrix_orchestration_linux_suite_coverage': 'Not part of 14-file Linux suite; separately reviewed preparation script.',
        'source_implementation_sha256': sorted(x['identity']['implementation_sha256'] for x in identities.values()),
        'target_execution_sha256': {k: v['checkpoint_identity']['implementation_sha256'] for k, v in targets.items()},
        'output_files_sha256': files, 'hash_scope': 'All directory files except this manifest; manifest_sha256 is canonical digest before that field.',
        'supporting_evidence_local_to_server': evidence, 'collection_evidence': collection,
        'checks': {'fixed_R0_source_files': 24, 'fresh_linux_suite_successful': True,
            'fresh_production_inventory_exact': True, 'source_unchanged': True,
            'same_checkpoint_helper': True, 'two_source_identity_pairs_per_kernel': True,
            'source_target_shims_identical': True, 'no_real_source_restore_or_advance': True,
            'linux_evidence_paths_preserved': True},
        'primary_approval_note': note, 'generator_command': list(sys.orig_argv), 'preflight_commands': commands,
        'user_amendment': 'Single final pass; speed targets are report-only.',
        'authorization_scope': 'Prepare implementation-hash migration artifacts only; real windows remain a separate launch.',
        'performance_status': 'NOT_QUALIFIED', 'human_status': 'HUMAN_PENDING'}
    manifest['manifest_sha256'] = provenance.digest(manifest)
    atomic_write_json(OUT / 'artifact_manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'output': str(OUT), 'manifest_sha256': manifest['manifest_sha256'],
                      'target_execution_sha256': manifest['target_execution_sha256'], 'server_executed': False}))


def self_check():
    ast.parse(Path(__file__).read_text(encoding='utf8'))
    ast.parse(PREFLIGHT_CHILD)
    provenance, _, reference, corpus = inputs()
    matrix = ROOT / MATRIX_REL
    if matrix.is_file():
        ast.parse(matrix.read_text(encoding='utf8'))
    print(json.dumps({'status': 'PREPARATION_SELF_CHECK_OK',
        'prepared_files_sha256': {name: sha(OUT / name) for name in ('prepare_deployment.py', 'README.md')},
        'reference_source_files': len(reference['actual_source_files_sha256']), 'corpus_sha256': corpus['manifest_sha256'],
        'generated_artifacts_present': [name for name in GENERATED if (OUT / name).exists()],
        'fresh_linux_output_exists': (BASE / SUITE).exists(), 'generation_executed': False, 'server_executed': False,
        'matrix_orchestration': {'path': MATRIX_REL, 'exists': matrix.is_file(),
                                 'sha256': sha(matrix) if matrix.is_file() else None,
                                 'AST_parsed': matrix.is_file(), 'part_of_linux_14_file_suite': False},
        'migration_decision': 'requires explicit primary note after successful final Linux evidence and finalized matrix review',
        'performance_status': 'NOT_QUALIFIED', 'human_status': 'HUMAN_PENDING'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--self-check', action='store_true')
    modes.add_argument('--generate-after-primary-approval', action='store_true')
    parser.add_argument('--primary-approval-note')
    args = parser.parse_args()
    if args.generate_after_primary_approval:
        if not args.primary_approval_note or not args.primary_approval_note.strip():
            parser.error('nonempty explicit primary approval note required after review of successful fresh Linux suite')
        generate(args.primary_approval_note.strip())
    else:
        self_check()
