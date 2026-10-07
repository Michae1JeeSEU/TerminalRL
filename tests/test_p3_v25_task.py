from dataclasses import replace

import numpy as np
import pytest

from terminal_marl.configs.p3_v25_task import build_config, SCENARIOS
from terminal_marl.configs.p3_integrated_repair import build_config as foundation
from terminal_marl.core.config import RingRole
from terminal_marl.core.observations import LOW_OBS_V2_4, build_low_observation_v2_4
from terminal_marl.core.rewards import low_level_urgency_penalty
from terminal_marl.envs.parallel_env import TerminalParallelEnv


@pytest.mark.parametrize('name,count,warm,horizon', [
    ('flow6',6,0,1210), ('dense24',24,12,1800), ('flow12',12,0,1570)])
def test_task_profile_keeps_repaired_physics_and_real_p3_attributes(name,count,warm,horizon):
    config = build_config(name)
    base = foundation(requests=count,warm_start=warm,forecast_merge_gap=True)
    for key in ('trajectory_schema_version','holding_speed_contract','v3_landing_path_mode',
                'merge_gap_prediction_mode','final_approach_geometry_mode','energy_model_mode',
                'route_aware_emergency_enabled','executable_admission_energy_enabled',
                'student_decision_mask_mode','runtime_proposal_shortlist_k','safe_distance',
                'final_queue_capacity','lz_assignment_mode','low_level_queue_cost_enabled'):
        assert getattr(config,key) == getattr(base,key)
    assert config.total_requested_uavs == count and config.warm_start_uavs == warm
    assert config.max_simulation_seconds == horizon
    assert config.task_priority_enabled and config.priority_mode == 'random_135'
    assert config.initial_priority is config.initial_soc is None
    assert (config.initial_soc_min,config.initial_soc_max)==(.55,.90)
    assert config.mission_probabilities == (.45,.25,.10,.20)
    assert config.release_arbitration_mode == 'downstream_priority'
    assert (config.low_level_priority_time_weight,config.low_level_soc_urgency_time_weight,
            config.low_level_soc_urgency_threshold)==(1.,2.,.65)
    if not warm:
        assert config.fixed_arrival_schedule == tuple(10.+60.*i for i in range(count))
        assert not config.warm_start_steady_motion and not config.warm_start_energy_reachability_check_enabled
    else:
        assert config.fixed_arrival_schedule == (0.,)*12+tuple(15.*i for i in range(1,13))
    assert not base.task_priority_enabled  # no historical profile mutation


def test_priority_changes_observation_arbitration_and_urgency_without_soc_only_ordering():
    env = TerminalParallelEnv(build_config('dense24'),seed=892905)  # test fixture only
    assert [sum(u.ring_role is r for u in env.active.values()) for r in RingRole] == [6,4,2]
    outer = [u for u in env.active.values() if u.ring_role is RingRole.OUTER]
    a,b = outer[:2]
    a.base_priority,b.base_priority=1,5
    a.energy_remaining=.60*a.energy_capacity; b.energy_remaining=.85*b.energy_capacity
    for u in env.active.values(): env.priority.update(u,env.now)
    order=[uid for uid,_ in env._release_processing_order()]
    assert order.index(b.uid)<order.index(a.uid)
    obs_a,obs_b=(build_low_observation_v2_4(u,env) for u in (a,b))
    assert obs_b[LOW_OBS_V2_4.slices['priority']][0]>obs_a[LOW_OBS_V2_4.slices['priority']][0]
    c=env.config
    def cost(priority):
        return low_level_urgency_penalty(c.low_level_time_penalty_per_second,c.dt,
            priority,.85,c.low_level_priority_time_weight,c.low_level_soc_urgency_time_weight,
            c.low_level_soc_urgency_threshold,c.soc_reserve)
    assert cost(5)<cost(1)
    for u in env.active.values():
        obs=env.trajectory_context_observations()[str(u.uid)]
        assert np.isfinite(obs.ego).all()


def test_natural_flow_has_empty_start_and_delayed_first_request():
    env=TerminalParallelEnv(build_config('flow6'),seed=892906)
    assert not env.active and env.arrivals.next_time == 10.
    assert not env.arrivals.exhausted()
    with pytest.raises(ValueError): build_config('warm_neutral')
