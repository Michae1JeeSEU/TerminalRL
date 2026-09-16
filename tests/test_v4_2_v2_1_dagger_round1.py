import copy

from terminal_marl.algo.joint_ppo_v4_2 import JointPPOV42
import terminal_marl.training.run_v4_2_v2_1_dagger_round1 as runner


def row(tick=0, phase="ring_outer", target=0):
    return {"seed": 1, "uav_id": "0", "phase": phase, "step": tick,
        "teacher_joint_index": target, "teacher_gate": int(target > 0),
        "teacher_proposal": target - 1 if target else None,
        "observation_v2_1": [0.] * 88, "decision_event_id": str(tick)}


def test_sampler_keeps_semantic_changes_without_mutating_or_duplicating():
    raw = [row(), row(1), row(2), row(3), row(4, target=11), row(5, "ring_middle", 11)]
    before = copy.deepcopy(raw)
    selected = runner.semantic_rows(raw)
    assert [r["step"] for r in selected] == [0, 3, 4, 5]
    assert raw == before


def test_frozen_protocol_and_finite_unweighted_finetune():
    assert runner._sha(runner.PROTOCOL) == runner.PROTOCOL_SHA
    policy = JointPPOV42(88)
    log = runner.fine_tune(policy, [row(), row(1, target=11)], {
        "training_seed": 1, "learning_rate": .0003, "epochs": 1, "batch_size": 1})
    assert len(log) == 1 and log[0]["finite_gradients"]


def test_incomplete_or_unsafe_collection_cannot_pass():
    summary = {"steps": 1, "terminal": False, "safety": {"nmac": 1},
               "replay_exact": True, "requested_logprob_exact": True}
    gates = runner.collection_gates([row()], [summary], [1, 2], 3)
    assert not gates["complete_episodes"] and not gates["safety"]
