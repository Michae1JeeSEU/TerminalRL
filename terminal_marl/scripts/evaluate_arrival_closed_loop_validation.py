"""Execute the registered same-authority validation through the existing native runner."""
from __future__ import annotations

import argparse
import json
from collections import Counter
import os
from pathlib import Path
import re
import time
import traceback

from terminal_marl.configs.arrival_closed_loop_validation import (
    PROFILE, PROTOCOL_VERSION, SEARCH_SETTINGS, build_specs, registration,
)
from terminal_marl.scripts.evaluate_arrival_restructure import SEARCH, execute, replay_run, write_json


def run_validation(mode, output, source_commit, workers):
    if not re.fullmatch(r'[0-9a-fA-F]{40}', source_commit):
        raise ValueError('source_commit must be a full 40-character hexadecimal SHA')
    if not 1 <= workers <= 18:
        raise ValueError('workers must be in 1..18')
    if SEARCH != SEARCH_SETTINGS:
        raise ValueError('native SEARCH differs from frozen validation settings')
    root = Path(output)
    specs = build_specs(mode, root, source_commit.lower())
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    base = dict(profile=PROFILE, protocol_version=PROTOCOL_VERSION, mode=mode,
                pid=os.getpid(), source_commit=source_commit.lower(), workers=workers)
    write_json(root/'manifest.json', dict(**base, registration=registration(), specs=specs))
    write_json(root/'status.json', dict(**base, status='RUNNING', elapsed_seconds=0.))
    results, replays, error = [], [], None
    try:
        results = execute(specs, workers)
        if mode == 'smoke':
            for result in results:
                replay = replay_run(result['output'], result['output'] + '_replay')
                replays.append(dict(source=result['output'], result=replay))
    except Exception:
        error = traceback.format_exc()
        # A pool failure can hide already-written arm results; retain those verbatim.
        known = {r.get('output') for r in results}
        for spec in specs:
            result_path = Path(spec['output'])/'result.json'
            if spec['output'] not in known and result_path.is_file():
                try:
                    results.append(json.loads(result_path.read_text(encoding='utf8')))
                except (OSError, ValueError):
                    error += '\nUnreadable arm result: ' + str(result_path)
    complete_count = len(results) == len(specs)
    if mode == 'smoke':
        passed = (not error and complete_count and len(replays) == len(specs)
            and all(r.get('status') == 'CENSORED' and r.get('stop_reason') == 'tick_cap'
                    and r.get('ticks') == 2 and r.get('integrity_ok') for r in results)
            and all(r['result'].get('status') == 'MATCH' and r['result'].get('matched_steps') == 2 for r in replays))
        status = 'SMOKE_PASS' if passed else 'ERROR'
    else:
        passed = not error and complete_count and all(r.get('status') == 'COMPLETED' for r in results)
        status = ('ERROR' if error or not complete_count or any(r.get('status') == 'ERROR' for r in results)
                  else 'COMPLETED' if passed else 'CENSORED')
    summary = dict(**base, status=status, exit_code=0 if passed else 1,
        elapsed_seconds=time.perf_counter()-started, runs=results, replays=replays,
        status_counts=dict(Counter(r.get('status') for r in results)), expected_arms=len(specs), error=error)
    write_json(root/'summary.json', summary)
    write_json(root/'status.json', {k:v for k,v in summary.items() if k not in ('runs', 'replays')})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'bridge', 'replicate', 'suite'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 19), default=1)
    args = parser.parse_args(argv)
    return run_validation(args.mode, args.output, args.source_commit, args.workers)['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
