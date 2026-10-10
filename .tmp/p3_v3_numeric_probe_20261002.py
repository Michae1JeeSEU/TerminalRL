"""Bounded numeric and caller probes for the frozen V3 comparison failures."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path('/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/numeric_probe')
MANIFEST = ROOT / '.tmp/p3_v3_matrix_source_binding_20261002.json'
MANIFEST_SHA = '3a2a7f21354f16be1c76af18033919ac72546cb32332d05e97b594cd8afda9ba'
INPUT = OUTPUT.parent / 'failure_probe/failure_probe.json'
INPUT_SHA = 'dabeed4586a932d5468651340d1e6280f094516a7ee18f4f51112a2fd605c0dc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write_exclusive(path, value):
    # The directory may already contain the shell's log; each data file is
    # nevertheless exclusive and cannot overwrite an earlier observation.
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def guard():
    if sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('source-binding manifest changed')
    binding = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
    hashes = {name: sha(ROOT / name) for name in binding['source_paths']}
    if hashes != binding['local_sha256']:
        raise RuntimeError('frozen comparison source changed')
    checkpoint = binding['checkpoint']
    if sha(checkpoint['path']) != checkpoint['expected_sha256']:
        raise RuntimeError('fixed source checkpoint changed')
    return hashes


def output_preflight():
    example = {'vectors': np.array([[1., 2., 3.]]), 'trace': ['caller'], 'error': None}
    target = OUTPUT / 'output_contract_preflight.json'
    expected = plain(example)
    if not target.exists():
        write_exclusive(target, example)
    if json.loads(target.read_text(encoding='utf-8')) != expected:
        raise RuntimeError('output serialization roundtrip mismatch')


def bounds_probe(evaluator):
    from terminal_marl.core.trajectory_generator import TrajectoryGenerator
    from terminal_marl.core.trajectory_time_parameterization import GeometricPath, TrajectoryConstraints

    if sha(INPUT) != INPUT_SHA:
        raise RuntimeError('failure fixture changed')
    record = json.loads(INPUT.read_text(encoding='utf-8'))['parameterizer_failures'][-1]
    if len(record['path_points']) != 7:
        raise RuntimeError('expected the final seven-node departure fixture')
    parameterizer = TrajectoryGenerator(evaluator._base_config()).parameterizer
    original_candidate = parameterizer._v3_candidate
    original_segments = parameterizer._v3_segments_feasible
    original_speeds = parameterizer._v3_speed_profile
    trials = []
    current = {}

    def observe_candidate(*args, **kwargs):
        nonlocal current
        current = {'acceleration_reserve': kwargs['acceleration_reserve'],
                   'speed_cap_scale': kwargs['speed_cap_scale']}
        trials.append(current)
        result = original_candidate(*args, **kwargs)
        current['candidate_returned'] = result is not None
        return result

    def observe_speeds(caps, segments, start_speed, end_speed_cap, acceleration_reserve):
        result = original_speeds(caps, segments, start_speed, end_speed_cap, acceleration_reserve)
        current['speed_profile'] = {'start_speed': start_speed, 'caps': caps,
                                   'segment_lengths': segments, 'end_speed_cap': end_speed_cap,
                                   'returned_speeds': result}
        return result

    def observe_segments(segments):
        segments = tuple(segments)
        result = original_segments(segments)
        current['segments_feasible'] = result
        current['segment_bounds'] = [segment.bounds() for segment in segments]
        return result

    parameterizer._v3_candidate = observe_candidate
    parameterizer._v3_speed_profile = observe_speeds
    parameterizer._v3_segments_feasible = observe_segments
    constraints = dict(record['constraints'])
    for name in ('end_direction', 'end_curvature_vector'):
        if constraints.get(name) is not None:
            constraints[name] = np.asarray(constraints[name], dtype=float)
    result = {'fixture': record, 'environment_steps': 0,
              'profile': asdict(parameterizer.dynamics.profile), 'trials': trials}
    try:
        points = parameterizer.parameterize(
            GeometricPath(np.asarray(record['path_points'], dtype=float)),
            start_time=float(record['start_time']),
            initial_velocity=np.asarray(record['initial_velocity'], dtype=float),
            initial_acceleration=np.asarray(record['initial_acceleration'], dtype=float),
            target_horizontal_speed=float(record['target_horizontal_speed']),
            constraints=TrajectoryConstraints(**constraints),
        )
        result['returned_point_count'] = len(points)
        result['error'] = None
    except Exception as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error),
                           'traceback': traceback.format_exc()}
    return result


def ring_probe(evaluator):
    from terminal_marl.core.trajectory_generator import TrajectoryGenerator

    specs = [row for row in evaluator._arm_specs()
             if row['cell_id'] == 'cell_1' and row['arm_id'] == 'nominal_rule_repaired90']
    if len(specs) != 1 or specs[0]['kind'] != 'nominal_rule':
        raise RuntimeError('registered ring probe spec mismatch')
    events = []
    original = TrajectoryGenerator._v3_ring_hold_points

    def observe(self, ring, theta, current_speed, target_speed, start, duration,
                *, initial_acceleration=None):
        try:
            return original(self, ring, theta, current_speed, target_speed, start, duration,
                            initial_acceleration=initial_acceleration)
        except RuntimeError as error:
            if len(events) < 20:
                item = {'error': str(error), 'traceback': traceback.format_exc(),
                        'caller_stack': traceback.format_stack(),
                        'ring_radius': ring.radius, 'ring_height': ring.height,
                        'theta': theta, 'current_speed': current_speed, 'target_speed': target_speed,
                        'start': start, 'duration': duration,
                        'initial_acceleration': initial_acceleration}
                try:
                    values = self._v3_ring_transition_parameters(
                        ring, theta, current_speed, target_speed,
                        initial_acceleration=initial_acceleration)
                    item['transition'] = {'ease': values[0], 'minimum_ramp': values[1],
                                          'v_after_ease': values[2], 'alpha0': values[3]}
                except Exception as nested:
                    item['transition_error'] = str(nested)
                events.append(item)
            raise

    TrajectoryGenerator._v3_ring_hold_points = observe
    try:
        arm = evaluator._run_arm(specs[0], evaluator._base_config(), max_steps=10)
    finally:
        TrajectoryGenerator._v3_ring_hold_points = original
    return {'spec': specs[0], 'max_steps': 10, 'events': events,
            'arm': {name: arm.get(name) for name in
                    ('status', 'stop_reason', 'error', 'censored', 'safety', 'validity', 'frozen_flags')},
            'progress_length': len(arm.get('progress', [])),
            'last_progress': (arm.get('progress') or [None])[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('bounds', 'ring'), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check-only', action='store_true')
    action.add_argument('--run', action='store_true')
    args = parser.parse_args()
    hashes = guard()
    output_preflight()
    target = OUTPUT / (args.mode + '.json')
    if target.exists():
        raise RuntimeError('probe output already exists: ' + str(target))
    if args.check_only:
        print(json.dumps({'status': 'CHECK_ONLY_OK', 'mode': args.mode,
                          'output_roundtrip': True, 'environment_steps': 0}), flush=True)
        return 0
    from terminal_marl.scripts import evaluate_p3_foundation_repair_20261002 as evaluator

    print(json.dumps({'status': 'PROBE_START', 'mode': args.mode}), flush=True)
    result = bounds_probe(evaluator) if args.mode == 'bounds' else ring_probe(evaluator)
    result['source_hashes_before'] = hashes
    result['source_hashes_after'] = guard()
    write_exclusive(target, result)
    print(json.dumps({'status': 'PROBE_WRITTEN', 'path': str(target),
                      'sha256': sha(target)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
