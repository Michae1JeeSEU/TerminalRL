"""Fixed eight-query comparison, inactive modules, no new environment resets."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = {
    'mixed': ('dense_mixed/seed_893102', '3befcfb8f6e5520b88dc1b394e9504f1a1e3afb2fcc9c14fef9c5bbcef929019',
              '8401501008b969722b4555e0cc21dcb6ae13f572213aa261203985ed9f5a6d67'),
    'high': ('dense_high_launch2/seed_893101', '0f0f8313e3b908eaa585f401a189ed24b9ece1cd87386a4bd9810d8ef296cde4',
             '8bcecfbb3a97bc76f153692c34644bbcab9c68125befe6536c3c8b122c1deba1'),
}
CASES = (('mixed', 0), ('mixed', 19), ('mixed', 299), ('high', 177))
MODULE_SHA = {
    'baseline': '832a88f432f083010cbc34a9dbc2c14b8b9fe6cbe016292bc43c0e12695d0a9f',
    'candidate': '603a2997df35bac03771bc9d69d99f1fed48c899cce0ff1e93acf75908fea1e6',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def child(args):
    # Load the selected immutable source under its real name before env imports.
    # No production module is rewritten; only this subprocess uses the variant.
    module_path = ROOT / f'.tmp/p3_polynomial_{args.mode}_20261005.py'
    assert sha(module_path) == MODULE_SHA[args.mode]
    name = 'terminal_marl.core.polynomial_trajectory'
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    import terminal_marl.core
    setattr(terminal_marl.core, 'polynomial_trajectory', module)
    from terminal_marl.core.trajectory_time_parameterization import prime_quintic_bounds
    assert prime_quintic_bounds is module.prime_quintic_bounds
    import pickle
    from terminal_marl.scripts.collect_p3_v3_sparse_expert import query_snapshot
    from terminal_marl.scripts.diagnose_p3_direct_contract import canonical, execute_tape, state_record
    from terminal_marl.core.safety_trace import _digest
    scene, slot = CASES[args.case]
    rel, selected_sha, steps_sha = SOURCES[scene]
    source = args.source / rel
    assert sha(source/'selected.json') == selected_sha
    entry = json.loads((source/'selected.json').read_text())[slot]
    entry_path = source/'snapshots'/f'window_{slot:02d}.json'
    assert json.loads(entry_path.read_text()) == entry
    snapshot = source/'snapshots'/entry['snapshot']
    label_path = source/'labels'/f'window_{slot:02d}.json'
    original = json.loads(label_path.read_text())
    assert original['status'] == 'OK'
    before = {path: sha(path) for path in (snapshot, entry_path, label_path)}
    assert before[snapshot] == entry['snapshot_sha256']
    started = time.perf_counter()
    query = query_snapshot(entry_path)
    seconds = time.perf_counter() - started
    write(args.output/'query.json', query)
    assert all(key in original and value == original[key] for key, value in query.items()), 'original label/certificate differs'
    assert query['release_intents_checked'] <= 25
    report = dict(status='QUERY_OK', mode=args.mode, scene=scene, slot=slot, seconds=seconds,
                  source_sha256=MODULE_SHA[args.mode], source_snapshot_sha256=before[snapshot],
                  original_label_certificate_exact=True, teacher_queries=1,
                  reset_count=0, env_steps=0, partition='diagnostic_never_train', trained=False)
    import numpy as np
    report['runtime'] = dict(platform=platform.platform(), python=sys.version,
                             numpy=np.__version__, executable=sys.executable)
    # Two preregistered states, original/candidate once each: four taped steps.
    if args.case in (2, 3):
        assert sha(source/'steps.jsonl') == steps_sha
        row = entry['row']
        step = [json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines()][row['tick']]
        env = pickle.loads(snapshot.read_bytes())
        assert _digest(canonical(state_record(env))) == row['pre_state_hash'] == step['pre_state_hash']
        live = execute_tape(env, step['tape'])
        assert _digest(live['after']) == step['post_state_hash']
        assert live['outputs'][4]['action_traces'] == step['action_traces']
        assert canonical(env.last_release_telemetry) == step['release_telemetry']
        write(args.output/'step.json', live)
        report.update(env_steps=1, exact_tape_step=True)
    assert all(sha(path) == digest for path, digest in before.items())
    report['source_files_unchanged'] = True
    write(args.output/'result.json', report)
    print(json.dumps(report))


def parent(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    prior_seconds = 0.0
    result = dict(status='RUNNING', cases=[], teacher_queries_max=8, release_intents_max=200,
                  steps_max=4, reset_count=0, wall_seconds=1200,
                  partition='diagnostic_never_train', trained=False, PPO_ALLOWED=False)
    write(args.output/'protocol.json', dict(result, modules=MODULE_SHA, sources=SOURCES,
          planned_cases=CASES, per_child_wall_seconds=120, inactive_variants=True))
    try:
        indices = [0] if args.phase == 'smoke' else [1, 2, 3]
        if args.phase == 'full':
            assert args.smoke_output is not None
            smoke = json.loads((args.smoke_output/'result.json').read_text())
            assert smoke['status'] == 'SMOKE_OK' and len(smoke['cases']) == 2
            assert [item['mode'] for item in smoke['cases']] == ['baseline', 'candidate']
            assert all(item['scene'] == 'mixed' and item['slot'] == 0 and
                       item['source_sha256'] == MODULE_SHA[item['mode']] for item in smoke['cases'])
            result['cases'] = smoke['cases']
            result['smoke_reused'] = str(args.smoke_output)
            result['smoke_result_sha256'] = sha(args.smoke_output/'result.json')
            prior_seconds = float(smoke['elapsed_seconds'])
        for index in indices:
            scene, slot = CASES[index]
            # Smoke pair is also the first registered benchmark pair, never rerun.
            for mode in ('baseline', 'candidate'):
                remaining = 1200 - prior_seconds - (time.monotonic()-start)
                assert remaining > 0, 'total wall budget exhausted'
                directory = args.output/f'{index}_{scene}_{slot}_{mode}'
                directory.mkdir(exist_ok=False)
                with (directory/'run.log').open('x') as log:
                    completed = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                        '--source', str(args.source), '--output', str(directory),
                        '--case', str(index), '--mode', mode], stdout=log, stderr=subprocess.STDOUT,
                        timeout=min(120, remaining), check=False, env=dict(os.environ,
                        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                write(directory/'exit.json', {'exit_code': completed.returncode})
                assert completed.returncode == 0, f'child failed: {directory.name}'
                item = json.loads((directory/'result.json').read_text())
                assert item['status'] == 'QUERY_OK'
                result['cases'].append(item)
                write(args.output/'progress.json', result)
            pair = result['cases'][-2:]
            assert pair[0]['source_snapshot_sha256'] == pair[1]['source_snapshot_sha256']
            baseline = args.output/f'{index}_{scene}_{slot}_baseline'
            candidate = args.output/f'{index}_{scene}_{slot}_candidate'
            assert json.loads((baseline/'query.json').read_text()) == json.loads((candidate/'query.json').read_text())
            if index in (2, 3):
                assert json.loads((baseline/'step.json').read_text()) == json.loads((candidate/'step.json').read_text())
        result['status'] = 'SMOKE_OK' if args.phase == 'smoke' else 'COMPARISON_OK'
    except Exception as exc:
        result.update(status='FAILED', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    result['elapsed_seconds'] = time.monotonic()-start
    result['combined_elapsed_seconds'] = prior_seconds + result['elapsed_seconds']
    write(args.output/'result.json', result)
    print(json.dumps(result))
    return 0 if result['status'] in ('SMOKE_OK', 'COMPARISON_OK') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', type=int, choices=range(4))
    parser.add_argument('--mode', choices=MODULE_SHA)
    parser.add_argument('--phase', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--smoke-output', type=Path)
    args = parser.parse_args()
    if args.case is not None:
        assert args.mode is not None
        child(args)
    else:
        assert args.mode is None
        raise SystemExit(parent(args))
