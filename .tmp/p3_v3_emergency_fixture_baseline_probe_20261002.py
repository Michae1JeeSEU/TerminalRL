"""Replay three stale V2 energy assertions against frozen and current code."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
bindings = (
    ('trajectory_time_parameterization', '.tmp/trajectory_time_parameterization_pre_numeric_repair_20261002.py'),
    ('trajectory_generator', '.tmp/trajectory_generator_pre_numeric_repair_20261002.py'),
    ('energy_reachability', '.tmp/energy_reachability_pre_numeric_repair_20261002.py'),
)
if args.baseline:
    import terminal_marl.core
    for name, relative in bindings:
        module_name = 'terminal_marl.core.' + name
        spec = importlib.util.spec_from_file_location(module_name, root / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        setattr(terminal_marl.core, name, module)

spec = importlib.util.spec_from_file_location('emergency_energy_fixture', root / 'tests/test_emergency_energy_v2.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
from terminal_marl.core import trajectory_generator, trajectory_time_parameterization, energy_reachability
from terminal_marl.core.arrival_process import ArrivalProcess
from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.envs.parallel_env import TerminalParallelEnv

floor_config = SimConfig(energy_model_mode='phase_motion_v2', phase_motion_nominal_solo_energy=1200.0)
try:
    process = ArrivalProcess(floor_config, seed=840000)
    bounds, floor_error = list(process._uniform_soc_bounds()), None
except ValueError as exc:
    bounds, floor_error = None, str(exc)
rows = []
for role, soc, step in ((RingRole.MIDDLE, .30, False), (RingRole.INNER, .249, True)):
    env = TerminalParallelEnv(fixture._config())
    uav = fixture._ring_uav(env, role, soc=soc)
    budgets = []
    for zone in env.lz_manager.zones:
        if uav.mission_type in zone.compatible_missions:
            plan = env.generator.emergency_landing(uav, zone, env.airspace, env.now)
            budgets.append({'lz_id': zone.lz_id,
                'required_soc': env.energy.emergency_required_energy(uav, plan) / uav.energy_capacity})
    if step:
        env.step({str(uav.uid): fixture.np.zeros(3)})
        accepted = bool(uav.emergency_landing)
    else:
        accepted = env._try_emergency_landing(uav)
    rows.append({'ring': role.value, 'initial_soc': soc, 'route_budgets': budgets,
        'accepted': accepted, 'active': uav.uid in env.active,
        'emergency_blocked_energy': env.telemetry['emergency_blocked_energy'],
        'emergency_committed': env.telemetry['emergency_committed'],
        'phase': uav.phase.value})
mode = 'baseline' if args.baseline else 'current'
record = {'mode': mode, 'schema': floor_config.trajectory_schema_version,
    'source_hashes': {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (trajectory_time_parameterization, trajectory_generator, energy_reachability)},
    'min_landing_soc': floor_config.min_landing_soc, 'energy_capacity': floor_config.energy_capacity,
    'nominal_solo_energy': floor_config.phase_motion_nominal_solo_energy,
    'default_soc_floor': floor_config.min_landing_soc + 1200.0 / floor_config.energy_capacity,
    'uniform_bounds': bounds, 'floor_error': floor_error, 'rows': rows}
output = Path('/home/user/training_outputs/terminal/trajectory_v3_repair_20261002') / ('emergency_fixture_' + mode + '.json')
with output.open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
print(json.dumps(record, indent=2, allow_nan=False))
