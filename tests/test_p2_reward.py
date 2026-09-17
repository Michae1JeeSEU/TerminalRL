from terminal_marl.training.run_v4_2_ppo_p2 import p2_config
from terminal_marl.training.run_v4_2_curriculum_pilot import config_for
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.run_v4_2_v2_1_student_qualification import scenario_config, protocol
from dataclasses import replace
import numpy as np


def test_p2_reward_change_is_explicit_and_does_not_change_physics():
    old,new=config_for(8),p2_config(8)
    differences=[k for k in vars(old) if getattr(old,k)!=getattr(new,k)]
    assert differences==['low_level_progress_reward_per_meter']
    assert old.low_level_progress_reward_per_meter==.001 and new.low_level_progress_reward_per_meter==0.


def test_waiting_in_a_ring_has_negative_reward_without_path_distance_bonus():
    c=replace(scenario_config('warm3',protocol()),low_level_progress_reward_per_meter=0.)
    env=TerminalParallelEnv(c);env.reset(752099)
    actions={str(uid):np.array([-1.,0.,0.]) for uid in env.active}
    _,rewards,_,_,info=env.step(actions)
    assert rewards and all(r<0 for r in rewards.values())
    assert all(r['progress']==0 for r in info['reward_components'].values())
