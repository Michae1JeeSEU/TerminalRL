"""Recompute all train labels from a complete saved neutral-course rollout.

This is a new offline annotation attempt, not a resumed or upgraded collection.
The original INCOMPLETE result, raw states and partial labels remain untouched.
No environment is reset or stepped, and no Q/development data are accepted.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from . import p3_v25_course_data as data
from .p3_v3_bc_learnability import require, read, write

PROTOCOL = 'p3_v25_complete_saved_train_annotation_20261008_v2'
ANNOTATION_SECONDS = 21600
RAW_FILES = ('config.json', 'rows.jsonl', 'steps.jsonl', 'selected.json',
             'rollout_result.json', 'first_before.pkl')


def source_inputs(directory):
    directory = Path(directory)
    protocol, report = read(directory/'protocol.json'), read(directory/'result.json')
    require(protocol['protocol'] == report['protocol'] == data.PROTOCOL
            and protocol['profile'] == report['profile'] == data.PROFILE
            and protocol['partition'] == report['partition'] == 'train'
            and not protocol['smoke'], 'only current neutral train collection is accepted')
    index = protocol['index']
    require(type(index) is int and 1 <= index <= len(data.TRAIN_CASES), 'invalid train index')
    seed, scenario = data.TRAIN_CASES[index - 1]
    require(protocol['seed'] == report['seed'] == seed
            and protocol['scenario'] == report['scenario'] == scenario
            and protocol['task'] == data.task_settings(data.build_config(scenario)), 'wrong train identity/task')
    require(report['status'] == 'INCOMPLETE' and report['error'] is None
            and report['blockers'] == ['incomplete_annotation']
            and not data.rollout_blockers(report['rollout']), 'source must fail annotation only')
    episode = directory/f'seed_{seed}'
    require(read(episode/'rollout_result.json') == report['rollout'], 'source rollout differs')
    annotation = read(episode/'annotation_result.json')
    require(annotation == report['annotation'] and annotation['status'] == 'CENSORED',
            'source annotation must be a preserved budget failure')
    selected = read(episode/'selected.json')
    rows = [json.loads(line) for line in (episode/'rows.jsonl').read_text(encoding='utf8').splitlines()]
    actors = [row for row in rows if row['actor_mask'] is True]
    require(0 < len(selected) == len(actors) == annotation['selected'] <= 2048,
            'all actor states must be selected within the original cap')
    require([entry['row'] for entry in selected] == actors
            and [entry['window'] for entry in selected] == list(range(len(selected)))
            and [(r['tick'], int(r['uid'])) for r in actors]
                == sorted((r['tick'], int(r['uid'])) for r in actors), 'selection differs from ordered actor rows')
    require(all(row['teacher_label'] is None for row in rows), 'raw labels must remain null')
    for entry in selected:
        name = entry['snapshot']
        require(Path(name).name == name, 'invalid snapshot basename')
        require(read(episode/'snapshots'/f"window_{entry['window']:02d}.json") == entry,
                'selected snapshot entry differs')
    return protocol, report, episode, selected


def compare_original_labels(original, repaired, original_annotation, new_annotation):
    """Time is diagnostic metadata; every label/audit/certificate field is exact."""
    completed = {entry['window']: entry for entry in new_annotation['entries'] if entry['status'] == 'OK'}
    compared = 0
    for entry in original_annotation['entries']:
        if entry['status'] != 'OK' or entry['window'] not in completed:
            continue
        name = f"window_{entry['window']:02d}.json"
        old, new = read(original/'labels'/name), read(repaired/'labels'/name)
        old.pop('elapsed_seconds', None)
        new.pop('elapsed_seconds', None)
        require(old == new, f'original teacher label/certificate differs at {name}')
        compared += 1
    return compared


def run(source_directory, output):
    source_directory, output = Path(source_directory), Path(output)
    require(not output.exists(), 'annotation repair output already exists')
    protocol, original, source, selected = source_inputs(source_directory)
    source_files = {name: data.sha(source/name) for name in RAW_FILES}
    origin = dict(protocol=PROTOCOL, source_directory=str(source_directory),
        source_protocol_sha256=data.sha(source_directory/'protocol.json'),
        source_result_sha256=data.sha(source_directory/'result.json'),
        source_annotation_sha256=data.sha(source/'annotation_result.json'),
        raw_file_sha256=source_files, original_labelled=original['annotation']['labelled'],
        selection='all original actor rows, recomputed from first to last; no partial-label reuse',
        annotation_wall_seconds=ANNOTATION_SECONDS, reset_count=0, env_steps=0)
    output.mkdir(parents=True, exist_ok=False)
    revised = deepcopy(protocol)
    revised.update(annotation_wall_seconds=ANNOTATION_SECONDS, annotation_repair=origin)
    write(output/'protocol.json', revised)
    report = {key: original[key] for key in
              ('protocol', 'profile', 'seed', 'scenario', 'partition', 'rollout', 'I0_QUALIFIED', 'PPO_ALLOWED')}
    report.update(status='PREPARING', error=None, blockers=[], annotation_repair=origin)
    write(output/'result.json', report)
    episode = output/f"seed_{protocol['seed']}"
    try:
        episode.mkdir()
        (episode/'labels').mkdir()
        for name in RAW_FILES:
            shutil.copy2(source/name, episode/name)
            require(data.sha(episode/name) == source_files[name], 'raw copy differs')
        # Independent copies avoid mutable hard links to the preserved failure.
        shutil.copytree(source/'snapshots', episode/'snapshots')
        report['status'] = 'ANNOTATING'
        write(output/'result.json', report)
        annotation = data.sparse.annotate(episode, selected, wall_seconds=ANNOTATION_SECONDS)
        report['annotation'] = annotation
        report['original_labels_compared_exact'] = compare_original_labels(
            source, episode, original['annotation'], annotation)
        if annotation['status'] != 'COMPLETED' or annotation['labelled'] != len(selected):
            report['blockers'].append('incomplete_annotation')
        if not report['blockers']:
            require(report['original_labels_compared_exact'] == original['annotation']['labelled'],
                    'original completed label comparison incomplete')
            report['reconstruction'] = data.rebuild_context(episode, seed=protocol['seed'],
                partition='train', expected_gap_mode=data.GAP_MODE)
            report['coverage'] = data.coverage_report(episode)
        require(data.sha(source_directory/'result.json') == origin['source_result_sha256']
                and data.sha(source/'annotation_result.json') == origin['source_annotation_sha256']
                and all(data.sha(source/name) == digest for name, digest in source_files.items()),
                'original source changed')
        report['status'] = 'INCOMPLETE' if report['blockers'] else 'COLLECTION_OK'
    except Exception as error:
        report.update(status='ERROR', error=repr(error), blockers=['annotation_repair_error'])
    write(output/'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_directory, args.output)
    print(json.dumps(dict(status=result['status'], error=result['error'])))
    return 0 if result['status'] == 'COLLECTION_OK' else 2


if __name__ == '__main__':
    raise SystemExit(main())
