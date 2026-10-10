"""Compare the frozen pre-repair and current V2 warm-start energy helper."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from terminal_marl.core.config import SimConfig
from terminal_marl.core import energy_reachability as current
from terminal_marl.envs.parallel_env import TerminalParallelEnv

root = Path(__file__).resolve().parent.parent
baseline_path = root / '.tmp/energy_reachability_pre_numeric_repair_20261002.py'
spec = importlib.util.spec_from_file_location('terminal_marl.core._baseline_energy_probe', baseline_path)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
config = SimConfig(reset_mode='warm_start_rings', warm_start_uavs=1,
    warm_start_steady_motion=True, fixed_arrival_schedule=(0.0,), total_requested_uavs=1,
    initial_soc_mode='fixed', initial_soc=0.35, energy_model_mode='phase_motion_v2',
    emergency_landing_enabled=True, warm_start_energy_reachability_check_enabled=False)
env = TerminalParallelEnv(config)
uav = env.active[0]
args = (uav, uav.ring_role, env.now, env.airspace, env.lz_manager, env.generator, env.energy, config)
result = {
    'schema': config.trajectory_schema_version,
    'baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    'current_sha256': hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
    'current_required_soc': current.minimum_immediate_emergency_soc(*args),
    'baseline_required_soc': baseline.minimum_immediate_emergency_soc(*args),
    'supplied_soc': uav.soc,
    'ring': uav.ring_role.value,
    'per_zone': [],
}
for zone in env.lz_manager.zones:
    if uav.mission_type not in zone.compatible_missions:
        continue
    route = env.generator.emergency_landing(uav, zone, env.airspace, env.now)
    energy = env.energy.estimate_plan_energy(route, start=env.now, end=route.end_time)
    result['per_zone'].append({'lz_id': zone.lz_id, 'energy': energy,
        'required_soc': (energy + env.energy.min_landing_soc * uav.energy_capacity
            + config.emergency_energy_margin_seconds * config.power_reference) / uav.energy_capacity})
out = Path('/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/warm_start_soc_probe.json')
with out.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps(result, indent=2, allow_nan=False))
