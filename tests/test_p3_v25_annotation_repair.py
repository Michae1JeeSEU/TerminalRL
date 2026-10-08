from copy import deepcopy
import json

import pytest

from terminal_marl.training import p3_v25_annotation_repair as repair


def fixture_source(tmp_path):
    data = repair.data
    seed, scenario = data.TRAIN_CASES[1]
    root = tmp_path/'source'
    episode = root/f'seed_{seed}'
    (episode/'labels').mkdir(parents=True)
    (episode/'snapshots').mkdir()
    identity = dict(protocol=data.PROTOCOL, profile=data.PROFILE, partition='train',
                    seed=seed, scenario=scenario)
    protocol = dict(identity, index=2, smoke=False, schema=data.DATA_SCHEMA,
        task=data.task_settings(data.build_config(scenario)), annotation_wall_seconds=7200)
    rows = [dict(actor_mask=True, teacher_label=None, tick=tick, uid='0',
                 decision_id=f'{data.DATA_SCHEMA}:{seed}:{tick}:0') for tick in (0, 1)]
    selected = []
    for index, row in enumerate(rows):
        name = f'tick_{index}.pkl'
        (episode/'snapshots'/name).write_bytes(b'test fixture only')
        entry = dict(window=index, row=row, snapshot=name,
                     snapshot_sha256=data.sha(episode/'snapshots'/name))
        repair.write(episode/'snapshots'/f'window_{index:02d}.json', entry)
        selected.append(entry)
    rows.append(dict(actor_mask=False, teacher_label=None, tick=2, uid='0', decision_id='forced'))
    (episode/'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    repair.write(episode/'selected.json', selected)
    repair.write(episode/'config.json', {})
    (episode/'steps.jsonl').write_text('{}\n')
    (episode/'first_before.pkl').write_bytes(b'test fixture only')
    flow = dict(status='COMPLETED', error=None, first_tick_replay_equal=True, selection_overflow=0,
        telemetry={key: 0 for key in data.sparse.SAFETY_KEYS}, terminal=True, censored=False,
        unresolved=0, active=0, queue=0, service=0, outcome_counts={'completed': 24}, requested=24)
    repair.write(episode/'rollout_result.json', flow)
    old_label = dict(status='OK', teacher_gate=0, teacher_proposal=None, selected_certificate=None,
                     decision_id=selected[0]['row']['decision_id'], teacher_audit=['original'], elapsed_seconds=5.)
    repair.write(episode/'labels/window_00.json', old_label)
    annotation = dict(status='CENSORED', selected=2, labelled=1, entries=[
        dict(window=0, status='OK'), dict(window=1, status='TIMEOUT')])
    repair.write(episode/'annotation_result.json', annotation)
    repair.write(root/'protocol.json', protocol)
    repair.write(root/'result.json', dict(identity, status='INCOMPLETE', error=None,
        blockers=['incomplete_annotation'], rollout=flow, annotation=annotation,
        I0_QUALIFIED=False, PPO_ALLOWED=False))
    return root, episode, old_label


def mock_annotation(monkeypatch, old_label, *, mismatch=False):
    def annotate(episode, selected, *, wall_seconds):
        assert wall_seconds == 21600
        assert len(selected) == 2 and not list((episode/'labels').iterdir())
        revised = repair.read(episode.parent/'protocol.json')
        assert revised['annotation_repair']['reset_count'] == revised['annotation_repair']['env_steps'] == 0
        entries = []
        for entry in selected:
            label = deepcopy(old_label)
            label.update(elapsed_seconds=1., decision_id=entry['row']['decision_id'])
            if mismatch:
                label['teacher_audit'] = ['changed']
            repair.write(episode/'labels'/f"window_{entry['window']:02d}.json", label)
            entries.append(dict(window=entry['window'], status='OK'))
        result = dict(status='COMPLETED', selected=2, labelled=2, entries=entries)
        repair.write(episode/'annotation_result.json', result)
        return result
    monkeypatch.setattr(repair.data.sparse, 'annotate', annotate)
    monkeypatch.setattr(repair.data, 'rebuild_context', lambda *args, **kwargs: {'rows': 2})
    monkeypatch.setattr(repair.data, 'coverage_report', lambda *args: {'labelled': 2})


def test_full_new_annotation_preserves_original_failure_and_all_raw_inputs(tmp_path, monkeypatch):
    root, episode, old_label = fixture_source(tmp_path)
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    mock_annotation(monkeypatch, old_label)
    result = repair.run(root, tmp_path/'repaired')
    assert result['status'] == 'COLLECTION_OK' and result['original_labels_compared_exact'] == 1
    assert result['reconstruction']['rows'] == 2 and result['PPO_ALLOWED'] is False
    assert result['annotation_repair']['env_steps'] == 0
    assert {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()} == before
    new_episode = tmp_path/'repaired'/episode.name
    assert (new_episode/'rows.jsonl').read_bytes() == (episode/'rows.jsonl').read_bytes()
    assert repair.read(root/'result.json')['status'] == 'INCOMPLETE'
    with pytest.raises(ValueError, match='already exists'):
        repair.run(root, tmp_path/'repaired')


def test_changed_label_stops_reconstruction_and_keeps_both_versions(tmp_path, monkeypatch):
    root, episode, old_label = fixture_source(tmp_path)
    mock_annotation(monkeypatch, old_label, mismatch=True)
    result = repair.run(root, tmp_path/'repaired')
    assert result['status'] == 'ERROR' and 'teacher label/certificate differs' in result['error']
    assert 'reconstruction' not in result
    assert repair.read(episode/'labels/window_00.json')['teacher_audit'] == ['original']
    assert repair.read(tmp_path/'repaired'/episode.name/'labels/window_00.json')['teacher_audit'] == ['changed']


def test_rejects_q_failed_rollout_and_partial_actor_selection(tmp_path):
    root, episode, _ = fixture_source(tmp_path)
    protocol = repair.read(root/'protocol.json')
    repair.write(root/'protocol.json', dict(protocol, partition='Q'))
    with pytest.raises(ValueError, match='only current neutral train'):
        repair.run(root, tmp_path/'bad_q')
    repair.write(root/'protocol.json', protocol)
    original = repair.read(root/'result.json')
    repair.write(root/'result.json', dict(original, blockers=['unresolved_cycle', 'incomplete_annotation']))
    with pytest.raises(ValueError, match='fail annotation only'):
        repair.run(root, tmp_path/'bad_rollout')
    repair.write(root/'result.json', original)
    repair.write(episode/'selected.json', repair.read(episode/'selected.json')[:1])
    with pytest.raises(ValueError, match='all actor states'):
        repair.run(root, tmp_path/'partial')
    assert not (tmp_path/'partial').exists()
