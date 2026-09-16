from dataclasses import replace

from terminal_marl.algo.joint_ppo_v4_2 import JointPPOV42
import terminal_marl.training.run_v4_2_v2_1_student_qualification as runner
from terminal_marl.training.student_rollout_v4_2 import rollout, validate_student_config


def test_frozen_protocol_and_teacher_free_natural_arrivals():
    p = runner.protocol()
    for name in p["scenarios"]:
        config = runner.scenario_config(name, p)
        validate_student_config(config)
        assert config.max_simulation_seconds == 900
        if name.startswith("flow"):
            assert config.warm_start_uavs == 0 and config.fixed_arrival_schedule[0] == 10


def test_deadline_alone_cannot_qualify_uncompleted_or_shield_dependent_student():
    p = runner.protocol()
    config = runner.scenario_config("warm3", p)
    summary = {"counts": {"actor_decisions": 100, "shield_actor": 6}, "completed": 0,
               "seconds": 900, "terminal": False, "queue": [0], "safety": {"nmac": 0},
               "replay_exact": True, "requested_logprob_exact": True}
    gates = runner.episode_gates(summary, config, p)
    assert not gates["all_requested_complete"] and not gates["terminal_and_queue_drained"]
    assert not gates["shield_burden"]


def test_logprob_replay_handles_empty_arrival_interval(tmp_path):
    config = replace(runner.scenario_config("flow3", runner.protocol()), max_simulation_seconds=2.)
    policy = JointPPOV42(88)
    _, summary = rollout(policy, config, seed=1, max_steps=2, output=tmp_path / "empty")
    assert summary["steps"] == 2 and summary["counts"]["actor_decisions"] == 0
    assert runner._policy_tape_logprob_exact(policy, config, summary["action_tape"])
