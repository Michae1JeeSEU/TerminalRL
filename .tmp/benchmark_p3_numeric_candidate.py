"""Local bounds-only synthetic benchmark; no environment or teacher query."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

root = Path(__file__).resolve().parents[1]
modules = {}
for mode in ('baseline', 'candidate'):
    path = root / f'.tmp/p3_polynomial_{mode}_20261005.py'
    name = f'p3_numeric_{mode}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    modules[mode] = module
rng = np.random.default_rng(73103)
coefficients = rng.normal(size=(512, 6, 3)) * 100
coefficients[::7, 4:, :] = 0
coefficients[::11] *= 1e-12
durations = rng.uniform(0.5, 10.0, 512)
times, results = {'baseline': [], 'candidate': []}, {}
for repeat in range(4):
    order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
    for mode in order:
        module = modules[mode]
        segments = [module.QuinticSegment(0.0, float(dt), c) for c, dt in zip(coefficients, durations)]
        start = time.perf_counter()
        result = tuple(row for offset in range(0, len(segments), 64)
                       for row in module.prime_quintic_bounds(segments[offset:offset+64]))
        times[mode].append(time.perf_counter() - start)
        results[mode] = result
        if len(results) == 2:
            assert results['baseline'] == results['candidate'], 'bounds changed'
report = dict(status='BOUNDS_BENCHMARK_OK', scope='synthetic_bounds_only', segments_per_repeat=512,
              repeats=4, exact_bounds=True, seconds=times, reset_count=0, teacher_queries=0,
              median_speedup=float(np.median(times['baseline']) / np.median(times['candidate'])))
report['source_sha256'] = {mode: hashlib.sha256((root/f'.tmp/p3_polynomial_{mode}_20261005.py').read_bytes()).hexdigest()
                           for mode in modules}
output = root / '.tmp/p3_numeric_bounds_benchmark_20261005.json'
with output.open('x', encoding='utf-8') as f:
    json.dump(report, f, indent=2, allow_nan=False)
print(json.dumps(report))
