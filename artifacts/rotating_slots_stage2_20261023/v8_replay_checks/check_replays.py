"""Read-only v8 replay load/seek and production PyVista offscreen evidence."""
from pathlib import Path
import collections, gc, json, sys, traceback
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from PIL import Image
from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer
from terminal_marl.visualization.rotating_slots_controller import _recorded_rescue_status, _INVISIBLE
from terminal_marl.core.rotating_slots.replay import _plain
OUT = Path(__file__).resolve().parent
ROOT = Path('D:/7_codespace/training_outputs/terminal/stage2_downstream_service_v8_medium_600s_20261023_r1')
RESULT = {'human_acceptance': 'HUMAN_PENDING', 'visual_review': 'HUMAN_PENDING', 'python': sys.executable, 'root': str(ROOT), 'bundles': []}

def save():
    (OUT / 'check.json').write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding='utf-8')

for method in ('energy_slack', 'rh_cp_sat'):
    path = ROOT / method / 'replay_0600'
    row = {'method': method, 'bundle_path': str(path), 'queries': [], 'screenshots': []}
    RESULT['bundles'].append(row)
    viewer = controller = None
    try:
        controller = ReplayController(path)
        audit_path = path.parent / 'audit_0600.json'
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        row.update(load='PASS', time_bounds=controller.bundle.time_bounds, states_count=len(controller.bundle.data['states']),
                   archive_human_acceptance=controller.bundle.index.get('human_acceptance'), audit_path=str(audit_path),
                   audit_status=audit['status'], audit_issue_count=audit['issue_count'], audit_issues=audit['issues'])
        row['powershell_launch_command'] = ('& D:\\7_codespace\\terminal\\venv\\Scripts\\python.exe -c "'
            'from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; '
            f"RotatingSlotsReplayViewer(ReplayController(r'{path}')).run()\"")
        controller.seek(300.)
        viewer = RotatingSlotsReplayViewer(controller, off_screen=True)
        viewer._plotter.window_size = (1600, 860)
        for requested in (300., 450., 600.):
            snapshot = controller.seek(requested)
            frame = controller.sampled_state()
            aircraft = controller.aircraft_at()
            rescue = {str(a['id']): _recorded_rescue_status(a, frame, controller._plans) for a in aircraft}
            visible = [a for a in aircraft if a.get('status') not in _INVISIBLE]
            ring_counts = {ring: sum(a.get('ring') == ring for a in visible) for ring in ('outer', 'middle', 'inner')}
            rescue_counts = {state: sum(s == state for s in rescue.values()) for state in ('WAIT', 'RESERVED', 'ACTIVE')}
            obs = {'requested_time': requested, 'controller_time': controller.time, 'snapshot_time': snapshot.time,
                   'sampled_time': frame['time'], 'record_aircraft_count': len(frame['aircraft']),
                   'controller_aircraft_count': len(aircraft), 'visible_count': len(visible), 'snapshot_uavs': len(snapshot.uavs),
                   'ring_counts': ring_counts, 'status_counts': dict(collections.Counter(a.get('status') for a in aircraft)),
                   'rescue_counts': rescue_counts,
                   'visible_aircraft': [{'id': str(a['id']), 'ring': a.get('ring'), 'status': a.get('status'),
                                         'soc': a['soc'], 'rescue_label': rescue[str(a['id'])]} for a in visible],
                   'pads': _plain(controller.pads_at()), 'hud_lines': controller.hud_lines(),
                   'checks': {'requested_time_matches': controller.time == requested == snapshot.time,
                              'sample_not_after_query': frame['time'] <= controller.time,
                              'aircraft_count_matches': len(frame['aircraft']) == len(aircraft),
                              'visible_count_matches': len(visible) == len(snapshot.uavs)}}
            row['queries'].append(obs)
            viewer.draw(); viewer.set_view('3d', vertical_exaggeration=1.); viewer.draw()
            viewer._plotter.add_text(
                f'{method} | t={controller.time:g}s\n'
                f'O / M / I: {ring_counts["outer"]} / {ring_counts["middle"]} / {ring_counts["inner"]}\n'
                f'Rescue WAIT / RESERVED / ACTIVE: {rescue_counts["WAIT"]} / {rescue_counts["RESERVED"]} / {rescue_counts["ACTIVE"]}\n'
                f'Audit {audit["status"]}; issues={audit["issue_count"]}; HUMAN_PENDING',
                position=(1000, 115), font_size=9, color='black', name='v8-evidence-summary')
            image_path = OUT / f'{method}_{int(requested):04d}.png'
            viewer._plotter.screenshot(str(image_path))
            with Image.open(image_path) as im:
                im.load(); dimensions = im.size
            row['screenshots'].append({'path': str(image_path), 'time': controller.time, 'size': dimensions, 'image_load': 'PASS'})
            print('IMAGE', method, requested, dimensions, flush=True)
        row['render'] = 'PASS'
    except Exception:
        row['error'] = traceback.format_exc(); print(row['error'], flush=True); save(); raise
    finally:
        if viewer is not None: viewer.close()
        viewer = controller = None
        gc.collect(); save()
RESULT['all_query_checks_pass'] = all(all(q['checks'].values()) for b in RESULT['bundles'] for q in b['queries'])
RESULT['image_count'] = sum(len(b['screenshots']) for b in RESULT['bundles'])
save()
print(json.dumps({'all_query_checks_pass': RESULT['all_query_checks_pass'], 'image_count': RESULT['image_count']}), flush=True)
