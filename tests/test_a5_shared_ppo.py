from dataclasses import replace
import numpy as np
import torch

from terminal_marl.algo import RolloutBuffer, SharedActorCentralCritic, SharedPPO, Transition
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
from terminal_marl.core.config import SimConfig
from terminal_marl.core.config import RingRole
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeScheduler
from terminal_marl.core.observations import A5_OBSERVATION_SCHEMA, pad_agent_sets
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.training.train_low_level_ppo import CurriculumStage, train_shared_low_level
from terminal_marl.training.evaluate_low_level import assess_a5_gate


def warm_config(n=1):
    return SimConfig(experiment_mode="fixed_low_rl", reset_mode="warm_start_rings", warm_start_uavs=n,
                     fixed_arrival_schedule=tuple(0.0 for _ in range(n)), total_requested_uavs=n,
                     max_simulation_seconds=300.0)


def test_a5_observation_schema_and_masked_set_critic_support_variable_n():
    schema=A5_OBSERVATION_SCHEMA
    batch, mask=pad_agent_sets([np.zeros((1,schema.dimension),np.float32),np.ones((5,schema.dimension),np.float32),np.ones((20,schema.dimension),np.float32)])
    assert batch.shape==(3,20,schema.dimension) and mask.sum(axis=1).tolist()==[1,5,20]
    critic=SharedActorCentralCritic(schema.dimension)
    value=critic.critic(torch.tensor(batch),torch.tensor(mask))
    assert value.shape==(3,) and torch.isfinite(value).all()


def test_a5_all_three_action_dimensions_change_canonical_semantics():
    env=TerminalParallelEnv(warm_config(1)); observations,_=env.reset(4); uid=env.agents[0]
    env.step({uid:np.array([0.,1.,0.])}); assert env.active[int(uid)].speed>20.0
    uav=env.active[int(uid)]; snapshot=WorldSnapshot(env.now,env.active,env.airspace); scheduler=MergeScheduler(env.config)
    early=scheduler.propose(uav,LowLevelIntent(1.,0.,-1.),snapshot,RingRole.MIDDLE)
    late=scheduler.propose(uav,LowLevelIntent(1.,0.,1.),snapshot,RingRole.MIDDLE)
    assert late.merge_time>early.merge_time
    _,_,_,_,info=env.step({uid:np.array([1.,0.,0.])})
    assert info["release_decisions"][uid] == "approved"


def test_a5_touchdown_reward_terminates_agent_before_service_completion():
    c=SimConfig(experiment_mode="fixed_low_rl", arrival_model="finite_cohort", total_requested_uavs=1, max_simulation_seconds=300.)
    env=TerminalParallelEnv(c); observations,_=env.reset(1); saw_touchdown=False
    for _ in range(300):
        actions={key:np.array([1.,0.,0.]) for key in observations}
        observations,reward,terminated,_,info=env.step(actions)
        touchdown=[x for x in info["reward_components"].values() if x["touchdown"]>0]
        if touchdown:
            saw_touchdown=True; assert any(terminated.values())
        if saw_touchdown and not reward and env.service_uavs:
            # Service remains global; no terminated actor receives its completion reward.
            assert all(x["touchdown"]==0 for x in info["reward_components"].values())
            break
    assert saw_touchdown


def test_a5_shared_ppo_update_and_checkpoint_metadata(tmp_path):
    d=A5_OBSERVATION_SCHEMA.dimension; policy=SharedPPO(d); buffer=RolloutBuffer(); world=np.zeros((2,d),np.float32)
    for index in range(5): buffer.add(Transition("u",world[0],world,np.zeros(3,np.float32),0.,0.,1.,index==4))
    result=policy.update(buffer,epochs=1,minibatch_size=4)
    assert np.isfinite(result["loss"])
    path=tmp_path/"a5.pt"; config=SimConfig(experiment_mode="fixed_low_rl"); save_checkpoint(path,policy,config); load_checkpoint(path,policy,config)


def test_a5_short_curriculum_training_has_no_nan(tmp_path):
    base=SimConfig(max_simulation_seconds=250., experiment_mode="fixed_low_rl")
    stages=(CurriculumStage("sanity",2,1,1.0), CurriculumStage("variable",2,3,.2))
    _, history=train_shared_low_level(base,curriculum=stages,output_dir=tmp_path,update_every=2)
    assert len(history)==4 and all(np.isfinite(row["reward"]) for row in history)


def test_a5_bursty_stage_generates_uncensored_requests():
    stage=CurriculumStage("bursty",1,5,.1,demand_model="bursty_wave")
    base=SimConfig(experiment_mode="fixed_low_rl")
    from terminal_marl.training.train_low_level_ppo import _stage_config
    env=TerminalParallelEnv(_stage_config(base,stage)); env.reset(2)
    for _ in range(20): env.step({})
    assert len(env.ledger.requested)>0


def test_a5_gate_requires_all_action_dimensions_and_a_primary_improvement():
    history=[{"reward":1., "action_release_std":.1, "action_speed_residual_std":.1, "action_merge_time_residual_std":.1}]
    comparison={"fixed_rule":{"delay":10.,"throughput":1.,"collisions":0.,"safety_interventions":2.},"fixed_low_rl":{"delay":9.,"throughput":1.,"collisions":0.,"safety_interventions":2.}}
    assert assess_a5_gate(comparison,history)["passed"]
