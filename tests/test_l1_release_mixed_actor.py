from __future__ import annotations

import numpy as np

from terminal_marl.algo import RolloutBuffer, SharedPPO, Transition
from terminal_marl.core.observations import LOW_OBS_V2
from terminal_marl.training.l1_ppo_d3_bounded_distribution import _requested_action_provenance
from terminal_marl.training.l1_release_mixed_actor_diagnostic import release_decision_metrics


def _observation(*, eligible: bool, conflict: bool = False, reserved: bool = False) -> np.ndarray:
    row = np.zeros(LOW_OBS_V2.dimension, dtype=np.float32)
    row[LOW_OBS_V2.slices["release_eligible"]] = float(eligible)
    conflict_slice = LOW_OBS_V2.slices["nearest_predicted_conflict"]
    row[conflict_slice.start] = float(conflict)
    row[conflict_slice.start + 8] = float(reserved)
    return row


def test_release_targeted_metrics_cover_both_errors_and_contexts() -> None:
    policy = SharedPPO(LOW_OBS_V2.dimension)
    rows = [
        Transition("false-release", _observation(eligible=False, conflict=True, reserved=True),
                   np.zeros((1, LOW_OBS_V2.dimension), np.float32), np.asarray((.01, 0., 0.)), 0., 0., 0., True,
                   teacher_action=np.asarray((-1., 0., 0.)), teacher_head_mask=np.zeros(3)),
        Transition("false-wait", _observation(eligible=True),
                   np.zeros((1, LOW_OBS_V2.dimension), np.float32), np.asarray((-.01, 0., 0.)), 0., 0., 0., True,
                   teacher_action=np.asarray((1., 0., 0.)), teacher_head_mask=np.zeros(3)),
    ]
    metrics = release_decision_metrics(policy, rows)
    assert metrics["label_counts"] == {"WAIT": 1, "RELEASE": 1}
    assert metrics["false_release"] == metrics["false_wait"] == 1
    assert metrics["teacher_release_corrections"] == 2
    assert metrics["near_threshold_samples"] == 2
    assert metrics["busy_or_committed_samples"] >= 1 and metrics["safe_open_gap_samples"] >= 1


def test_mixed_release_binary_bc_keeps_requested_action_and_ratio_provenance() -> None:
    policy = SharedPPO(LOW_OBS_V2.dimension, policy_distribution="mixed_bernoulli_tanh", teacher_bc_weight=.05)
    observations = np.stack((_observation(eligible=False), _observation(eligible=True)))
    actions, log_probs, value, by_head = policy.act(observations, observations, return_head_log_probs=True)
    buffer = RolloutBuffer()
    for index, target in enumerate((-1., 1.)):
        teacher = actions[index].copy(); teacher[0] = target
        buffer.add(Transition(str(index), observations[index], observations, actions[index], float(log_probs[index]),
                              value, 1., True, teacher_action=teacher, applied_action=actions[index],
                              log_prob_by_head=by_head[index], teacher_head_mask=np.asarray((1., 0., 0.))))
    proof = _requested_action_provenance(policy, buffer)
    assert proof["passed"] and not proof["a_applied_enters_ratio"]
    metrics = policy.update(buffer, epochs=1, minibatch_size=2)
    assert all(np.isfinite(value) for value in metrics.values())
