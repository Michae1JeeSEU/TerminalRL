from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
sys.path.insert(0, r'D:\\7_codespace\\terminal')
import tempfile
import time
from pathlib import Path

import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

from terminal_marl.visualization.config import VisualizationConfig
from terminal_marl.visualization.pyvista_viewer import Terminal3DViewer as CurrentViewer
from terminal_marl.visualization.schema import (
    LandingZoneRenderState,
    RingRenderState,
    UAVRenderState,
    WorldRenderSnapshot,
)

OUT = Path('results/viewer_repair_20260924')
OUT.mkdir(parents=True, exist_ok=True)
BASELINE_REV = '9cd71b23dbbff48eff2992d719531b6e01972069'

def uav(uid, role, angle, radius, height, alert='normal'):
    position = (radius*np.cos(angle), radius*np.sin(angle), height)
    points = tuple((float(radius*np.cos(angle + 0.03*i)), float(radius*np.sin(angle + 0.03*i)), float(height)) for i in range(20))
    return UAVRenderState(
        uid=uid, arrival_id=uid, phase='ring_'+role, ring_role=role,
        position=tuple(float(v) for v in position), velocity=(0.0, 1.0, 0.0), speed=20.0,
        soc=0.32 if alert == 'emergency' else (0.44 if alert == 'low_soc' else 0.75),
        energy_remaining=750.0, energy_capacity=1000.0, mission_type='passenger',
        base_priority=1, assigned_gate=0, assigned_lz=None, trajectory_plan_id=f'plan-{uid}',
        merge_reservation_id=None, committed=True, trajectory_points=points,
        dynamic_priority=1.0, last_release_decision=None, last_safety_decision=None,
        emergency_landing=alert == 'emergency', energy_alert=alert,
    )

def snapshot(t=0.0):
    rings = (
        RingRenderState('outer', 600.0, 100.0, 600.0, 100.0, 30, 10),
        RingRenderState('middle', 400.0, 80.0, 400.0, 80.0, 20, 10),
        RingRenderState('inner', 200.0, 60.0, 200.0, 60.0, 10, 10),
    )
    uavs = []
    for i in range(10):
        uavs.append(uav(i, 'outer', i*2*np.pi/10, 600.0, 100.0, 'low_soc' if i == 2 else 'normal'))
    for i in range(10):
        uavs.append(uav(10+i, 'middle', i*2*np.pi/10+0.1, 400.0, 80.0, 'emergency' if i == 3 else 'normal'))
    for i in range(10):
        uavs.append(uav(20+i, 'inner', i*2*np.pi/10+0.2, 200.0, 60.0, 'low_soc' if i == 7 else 'normal'))
    lzs = tuple(LandingZoneRenderState(i, (float(100*np.cos(i*2*np.pi/6)), float(100*np.sin(i*2*np.pi/6)), 0.0), 'free', None, None, 0.0) for i in range(6))
    return WorldRenderSnapshot(
        time=t, step_count=int(t), rings=rings, gates=(), landing_zones=lzs,
        uavs=tuple(uavs), external_queue_size=0, service_count=0, completed_count=0,
        transition_active=False, transition_progress=0.0, telemetry={}, recent_conflict_uids=(),
    )

def load_before():
    path = Path(tempfile.gettempdir()) / 'pyvista_viewer_before_20260924.py'
    source = subprocess.check_output([
        'git', 'show', f'{BASELINE_REV}:terminal_marl/visualization/pyvista_viewer.py'
    ])
    path.write_bytes(source)
    name = 'terminal_marl.visualization.pyvista_viewer_before_20260924'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def screenshot(viewer, path):
    viewer._plotter.show(auto_close=False, interactive_update=True)
    viewer._plotter.reset_camera()
    viewer._plotter.render()
    viewer._plotter.screenshot(str(path), transparent_background=False)

def benchmark(Viewer, name):
    source = snapshot()
    viewer = Viewer(source, VisualizationConfig(show_labels=True, show_trajectories=True))
    try:
        viewer.update(source, render=True)
        for index in range(10):
            positions = tuple((u.position[0] + 0.5*index, u.position[1], u.position[2]) for u in source.uavs)
            viewer.update_interpolated_positions(source, positions)
        started = time.perf_counter()
        for index in range(60):
            positions = tuple((u.position[0] + 0.5*(index+10), u.position[1], u.position[2]) for u in source.uavs)
            viewer.update_interpolated_positions(source, positions)
        elapsed = time.perf_counter() - started
        return {'name': name, 'uavs': 30, 'warmup_frames': 10, 'timed_frames': 60, 'seconds': elapsed, 'fps': 60.0/elapsed}
    finally:
        viewer.close()

source = snapshot()
before_module = load_before()
before_viewer = before_module.Terminal3DViewer(source, VisualizationConfig(show_labels=True, show_trajectories=True))
try:
    before_viewer.update(source, render=True)
    screenshot(before_viewer, OUT / 'before.png')
finally:
    before_viewer.close()

after_viewer = CurrentViewer(source, VisualizationConfig(show_labels=True, show_trajectories=True))
try:
    after_viewer.update(source, render=True)
    screenshot(after_viewer, OUT / 'after.png')
finally:
    after_viewer.close()

benchmark_payload = {
    'generated_by': 'offscreen VTK renderer benchmark',
    'frames_are_not_hardware_fps': True,
    'baseline_revision': BASELINE_REV,
    'methodology': {
        'viewport': [1280, 860],
        'pyvista_version': pv.__version__,
        'off_screen': True,
        'render_path': 'Terminal3DViewer.update_interpolated_positions -> Plotter.update()',
        'uavs': 30,
        'warmup_frames': 10,
        'timed_frames': 60,
    },
    'before': benchmark(before_module.Terminal3DViewer, f'baseline {BASELINE_REV}'),
                     'after': benchmark(CurrentViewer, 'working-tree viewer')}
(OUT / 'renderer_frame_benchmark.json').write_text(json.dumps(benchmark_payload, indent=2), encoding='utf-8')
print(json.dumps(benchmark_payload, indent=2))
print('artifacts', sorted(str(p) for p in OUT.iterdir()))


