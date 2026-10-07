from dataclasses import replace

import pytest

from terminal_marl.configs.p3_v25_course import build_config, SCENARIOS
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def test_course_keeps_energy_service_safety_but_no_task_priority():
    for scenario in SCENARIOS:
        cfg=build_config(scenario)
        assert not cfg.task_priority_enabled and cfg.low_level_priority_time_weight==0
        assert cfg.low_level_soc_urgency_time_weight==2
        assert cfg.release_arbitration_mode=='downstream_soc_priority'
        assert cfg.mission_probabilities==(.45,.25,.10,.20)
        assert cfg.route_aware_emergency_enabled and cfg.energy_reachability_guard_enabled
        assert cfg.v3_landing_path_mode=='direct_c2' and cfg.merge_gap_prediction_mode=='committed_arrivals_v1'
        assert cfg.runtime_proposal_shortlist_k==3 and cfg.final_queue_capacity==3
    fixed=build_config('dense24_fixed');mixed=build_config('dense24_mixed')
    assert replace(fixed,initial_soc_mode=mixed.initial_soc_mode,initial_soc=mixed.initial_soc)==mixed
    assert fixed.initial_soc==.8 and fixed.initial_soc_mode=='fixed'
    assert mixed.initial_soc_mode=='uniform' and (mixed.initial_soc_min,mixed.initial_soc_max)==(.55,.90)
    flow=build_config('flow6_mixed')
    assert flow.warm_start_uavs==0 and flow.fixed_arrival_schedule==tuple(10.+60*i for i in range(6))
    with pytest.raises(ValueError):build_config('dense24')


def test_fixed_soc_and_neutral_attributes_reach_actual_reset():
    env=TerminalParallelEnv(build_config('dense24_fixed'),seed=892953)
    assert len(env.active)==12
    assert all(u.base_priority==1 and abs(u.soc-.8)<1e-12 for u in env.active.values())
    assert len({u.mission_type for u in env.active.values()})>1
    assert all(obs.ego[18:22].tolist()==[1.,0.,0.,0.] for obs in env.trajectory_context_observations().values())
