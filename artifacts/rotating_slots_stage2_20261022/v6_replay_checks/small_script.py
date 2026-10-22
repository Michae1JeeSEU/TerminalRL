"""Read-only v6 archive queries and production-viewer offscreen screenshots."""
from pathlib import Path
import collections
import gc
import json
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from terminal_marl.core.rotating_slots.replay import ReplayBundle, _plain
from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = Path('D:/7_codespace/training_outputs/terminal')
INVISIBLE = {'CLEARED', 'COMPLETED', 'REROUTED', 'UNRELEASED_AT_FAILURE'}
result = {'human_acceptance': 'HUMAN_PENDING', 'scope': 'Archive load/seek and existing platform render only',
          'python': sys.executable, 'bundles': [], 'visual_review': 'HUMAN_PENDING',
          'viewer_api': 'ReplayController(ReplayBundle.load(path)); seek(t); RotatingSlotsReplayViewer(controller, off_screen=True); set_view("3d"); draw(); _plotter.screenshot(path)',
          'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
          'viewer_diff_from_archive_commit': subprocess.check_output(['git', 'diff', '9cfb781', '--', 'terminal_marl/visualization/rotating_slots_replay.py', 'terminal_marl/visualization/pyvista_viewer.py', 'terminal_marl/core/rotating_slots/replay.py'], cwd=REPO, text=True),
          'normal_event_definition': 'type=maneuver_started, details.maneuver_kind=FINAL_NORMAL, time strictly after first emergency_declared',
          'launch_commands': []}

def save():
    (OUT / 'check.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

def declared(a):
    x = a.get('emergency', {})
    return bool(x.get('declared', False)) if isinstance(x, dict) else bool(x)

def describe(controller, requested):
    snapshot = controller.seek(requested)
    record = controller.bundle.sampled_state(controller.time)
    rows = _plain(record['aircraft'])
    aircraft = controller.aircraft_at()
    overlay = controller.overlays()
    emergency_ids = {str(a['id']) for a in aircraft if declared(a)}
    active = overlay['current_commits']
    soft = overlay['future_soft_dependencies']
    pads = controller.pads_at()
    expected_visible = sum(a.get('status') not in INVISIBLE for a in rows)
    expected_service = sum(a.get('status') in ('SERVICE', 'LANDED_SERVICE') for a in rows)
    expected_completed = sum(a.get('status') in ('COMPLETED', 'CLEARED') for a in rows)
    counts = {'record_aircraft': len(rows), 'controller_aircraft': len(aircraft),
              'record_visible_aircraft': expected_visible, 'snapshot_uavs': len(snapshot.uavs),
              'record_service': expected_service, 'snapshot_service': snapshot.service_count,
              'record_completed': expected_completed, 'snapshot_completed': snapshot.completed_count}
    checks = {'controller_count_matches_record': len(rows) == len(aircraft),
              'visible_count_matches_record': len(snapshot.uavs) == expected_visible,
              'service_count_matches_record': snapshot.service_count == expected_service,
              'completed_count_matches_record': snapshot.completed_count == expected_completed,
              'snapshot_time_matches_controller': snapshot.time == controller.time,
              'record_time_not_after_query': record['time'] <= controller.time}
    return {'requested_time': requested, 'controller_time': controller.time, 'record_sample_time': record['time'],
            'counts': counts, 'checks': checks,
            'status_counts': dict(collections.Counter(a.get('status') for a in aircraft)),
            'declared_aircraft_ids': sorted(emergency_ids),
            'active_rescue_ids': sorted(str(a['id']) for a in active if str(a['id']) in emergency_ids),
            'active_normal_ids': sorted(str(a['id']) for a in active if str(a['id']) not in emergency_ids),
            'soft_rescue_rows': sum(str(r['aircraft_id']) in emergency_ids for r in soft),
            'soft_normal_rows': sum(str(r['aircraft_id']) not in emergency_ids for r in soft),
            'physical_pad_occupants': [{'pad_id': p['id'], 'aircraft_id': p['occupant_id'], 'service_remaining': p['service_remaining']} for p in pads if p['occupant_id'] is not None],
            'future_pad_reservations': [{'pad_id': p['id'], 'owner_id': r['owner_id'], 'commitment': r['commitment'], 'maneuver_kind': r['maneuver_kind'], 'maneuver_start': r['maneuver_start'], 'interval_start': r['interval_start'], 'interval_end': r['interval_end']} for p in pads for r in p['reservations'] if r['future_interval']],
            'soft_classification': 'rescue means owner currently emergency.declared; rows are resource dependencies, not unique maneuvers'}

for scene in ('stress', 'diagnostic'):
    for method in ('energy_slack', 'rh_cp_sat'):
        path = ROOT / f'stage2_rescue_opportunity_v6_{scene}_480s_20261022' / method / 'replay_0480'
        entry = {'scene': scene, 'method': method, 'path': str(path), 'queries': [], 'screenshots': []}
        result['bundles'].append(entry)
        viewer = controller = bundle = None
        print('START', scene, method, flush=True)
        try:
            bundle = ReplayBundle.load(path)
            controller = ReplayController(bundle)
            events = _plain(bundle.data['events'])
            declarations = [e for e in events if e['type'] == 'emergency_declared']
            first = min(declarations, key=lambda e: e['time']) if declarations else None
            normals = [e for e in events if e['type'] == 'maneuver_started' and e.get('details', {}).get('maneuver_kind') == 'FINAL_NORMAL' and first and e['time'] > first['time']]
            normal = min(normals, key=lambda e: e['time']) if normals else None
            entry.update(load='PASS', time_bounds=bundle.time_bounds, states_count=len(bundle.data['states']),
                         code_commit=bundle.index.get('code_commit'), human_acceptance=bundle.index.get('human_acceptance'),
                         first_declaration=first, first_normal_start_after_declaration=normal)
            times = [0.] + ([first['time']] if first else []) + [420., 480.]
            entry['queries'] = [describe(controller, t) for t in times]
            save()
            if scene == 'stress':
                targets = [('first_declaration_plus1', first['time'] + 1 if first else 420., first),
                           ('normal_after_declaration_plus1' if normal else 'fallback420', normal['time'] + 1 if normal else 420., normal),
                           ('end480', 480., None)]
            elif method == 'energy_slack':
                targets = [('end480', 480., None)]
            else:
                targets = [('first_declaration_plus1', first['time'] + 1 if first else 420., first), ('end480', 480., None)]
            controller.seek(targets[0][1])
            viewer = RotatingSlotsReplayViewer(controller, off_screen=True)
            for label, t, event in targets:
                observation = describe(controller, t)
                viewer.draw()
                viewer.set_view('3d', vertical_exaggeration=1.)
                viewer.draw()
                image_path = OUT / f'{scene}_{method}_{label}.png'
                viewer._plotter.screenshot(str(image_path))
                with Image.open(image_path) as im:
                    im.load()
                    size, mode = im.size, im.mode
                shot = {'label': label, 'path': str(image_path), 'time': controller.time,
                        'selected_event': event, 'camera_api': 'set_view("3d", vertical_exaggeration=1.)',
                        'camera_position': [list(v) for v in viewer._plotter.camera_position],
                        'image_load': 'PASS', 'size': size, 'mode': mode, 'observation': observation}
                entry['screenshots'].append(shot)
                save()
                print('IMAGE', image_path.name, controller.time, size, flush=True)
            entry['render'] = 'PASS'
            result['launch_commands'].append(f'& .\\venv\\Scripts\\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; v=RotatingSlotsReplayViewer(ReplayController(r\'{path}\')); v.run()"')
        except Exception:
            entry['error'] = traceback.format_exc()
            print(entry['error'], flush=True)
        finally:
            if viewer is not None:
                viewer.close()
            viewer = controller = bundle = None
            gc.collect()
            save()

result['all_query_checks_pass'] = all(all(q['checks'].values()) for b in result['bundles'] for q in b['queries'])
result['image_count'] = sum(len(b['screenshots']) for b in result['bundles'])
save()
print(json.dumps({'all_query_checks_pass': result['all_query_checks_pass'], 'image_count': result['image_count'], 'check': str(OUT / 'check.json')}), flush=True)
