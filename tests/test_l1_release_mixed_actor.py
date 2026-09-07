from __future__ import annotations

import copy
import numpy as np

from terminal_marl.algo import RolloutBuffer, SharedPPO, Transition
from terminal_marl.core.config import RingRole
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.core.observations import LOW_OBS_V2
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.training.l1_ppo_d3_bounded_distribution import _requested_action_provenance
from terminal_marl.training.l1_release_mixed_actor_diagnostic import (
    _release_probe_task, release_decision_metrics, release_objective_metrics,
)
from terminal_marl.training.l1_release_d5a_objective import _d5a_gate, run_d5a_runtime_preflight
from terminal_marl.training.l1_release_d5b_observation import analyze_release_observation_sufficiency
from terminal_marl.training.l1_release_d5c_merge_coupling import analyze_release_merge_coupling
from terminal_marl.training.train_low_v2_ppo import (
    _counterfactual_release_teacher_targets, l1_full_teacher_config,
)


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
    objective = release_objective_metrics(policy, buffer)
    assert objective["label_balance"] == {"WAIT": 1, "RELEASE": 1, "release_fraction": .5}
    assert objective["advantage_by_requested_decision"]["WAIT"]["samples"] + \
        objective["advantage_by_requested_decision"]["RELEASE"]["samples"] == 2
    assert np.isfinite(objective["release_binary_entropy_mean"])
    assert np.isfinite(objective["calibration"]["binary_nll"])
    metrics = policy.update(buffer, epochs=1, minibatch_size=2)
    assert all(np.isfinite(value) for value in metrics.values())


def test_release_magnitude_is_scheduler_aliased_within_each_binary_decision() -> None:
    config = l1_full_teacher_config(medium_rate=.001575, reset_mode="warm_start_rings",
                                    demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8,
                                    horizon_seconds=60.)
    env = TerminalParallelEnv(config); env.reset(1001)
    uav = next(item for item in env.active.values() if item.ring_role is not RingRole.INNER)
    target = RingRole.MIDDLE if uav.ring_role is RingRole.OUTER else RingRole.INNER
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    low = env.scheduler.propose(uav, LowLevelIntent(.01, .2, .3), snapshot, target)
    high = env.scheduler.propose(uav, LowLevelIntent(.99, .2, .3), snapshot, target)
    assert low == high


def test_counterfactual_release_labels_are_isolated_and_authoritative() -> None:
    config = l1_full_teacher_config(medium_rate=.001575, reset_mode="warm_start_rings",
                                    demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8,
                                    horizon_seconds=2.)
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(1001)
    state_before = (env.now, env.step_count, tuple(env.active), len(env.completed_uavs))
    actions = {uid: np.asarray((-.01, 0., 0.), dtype=np.float32) for uid in observations}
    targets, decisions = _counterfactual_release_teacher_targets(env, observations, actions)
    assert (env.now, env.step_count, tuple(env.active), len(env.completed_uavs)) == state_before
    assert set(targets) == set(observations) == set(decisions)
    assert set(targets.values()) == {-1., 1.}
    assert all((target > 0.) == (decisions[uid] in {"approved", "future_reserved", "future_reservation_wait"})
               for uid, target in targets.items())


def test_release_probe_has_aligned_fields_and_threshold_equivalence() -> None:
    config = l1_full_teacher_config(medium_rate=.001575, reset_mode="warm_start_rings",
                                    demand_model="finite_cohort", warm_start_uavs=8, cohort_uavs=8,
                                    horizon_seconds=2.)
    rows = [_release_probe_task((config, 1001, magnitude, 2)) for magnitude in (-.99, -.01, .01, .99)]
    by_magnitude = {row["requested_release_value"]: row for row in rows}
    comparable = ("teacher_target_counts", "false_release", "false_wait", "decision_counts", "completion", "safety")
    assert all(by_magnitude[.01][field] == by_magnitude[.99][field] for field in comparable)
    assert all(by_magnitude[-.01][field] == by_magnitude[-.99][field] for field in comparable)
    assert by_magnitude[.01]["near_threshold_samples"] > 0
    assert by_magnitude[-.01]["near_threshold_samples"] > 0
    assert all(set(row["safety"]) == {"collisions", "nmac", "hard_separation_violations"} for row in rows)


def test_d5a_gate_requires_targeted_probe_and_all_frozen_health_gates() -> None:
    control = {
        "release": {"teacher_release_correction_rate": 1.0},
        "teacher_burden_overall": 10., "teacher_burden_warm": 10.,
        "correction_p95": 1., "shield_burden": 1.,
        "completion_ratio": 1., "throughput_per_second": .01,
        "ppo_health": {"healthy": True}, "safety_passed": True, "provenance_passed": True,
    }
    candidate = copy.deepcopy(control)
    candidate["release"]["teacher_release_correction_rate"] = .5
    assert _d5a_gate(control, candidate, targeted_probe_passed=True)["window_passed"]
    assert not _d5a_gate(control, candidate, targeted_probe_passed=False)["window_passed"]
    candidate["safety_passed"] = False
    assert not _d5a_gate(control, candidate, targeted_probe_passed=True)["window_passed"]


def test_d5a_runtime_preflight_signature_exposes_output_closure() -> None:
    assert callable(run_d5a_runtime_preflight)


def test_d5b_detects_identical_observations_with_opposite_release_labels() -> None:
    observation = _observation(eligible=True)
    rows = []
    for index, (label, residual) in enumerate(((-1., -.8), (-1., -.4), (1., .4), (1., .8))):
        action = np.asarray((1., residual, residual), dtype=np.float32)
        teacher = action.copy(); teacher[0] = label
        rows.append(Transition(str(index), observation.copy(), observation[None].copy(), action,
                               0., 0., 0., True, teacher_action=teacher,
                               teacher_head_mask=np.asarray((1., 0., 0.))))
    result = analyze_release_observation_sufficiency(rows)
    assert result["observation_aliasing_detected"]
    assert result["requested_residual_coupling_hint"]
    assert result["quantized_label_conflicts"]["full_observation"][-1]["release_fraction_in_mixed_bins"] == 1.


def test_d5c_counts_release_merge_combined_corrections() -> None:
    observation = _observation(eligible=True)
    rows = []
    for index in range(6):
        action = np.asarray((1., 0., .8 if index < 4 else -.8), dtype=np.float32)
        teacher = action.copy()
        if index < 4:
            teacher[0] = -1.; teacher[2] = -.8
        rows.append(Transition(str(index), observation.copy(), observation[None].copy(), action,
                               0., 0., 0., True, teacher_action=teacher,
                               teacher_used=index < 4, teacher_head_mask=np.ones(3)))
    result = analyze_release_merge_coupling(rows)
    assert result["correction_category_counts"]["release_merge"] == 4
    assert result["correction_category_counts"]["none"] == 2
    assert result["coupling"]["release_merge_lift_over_independence"] > 1.
    assert result["strong_release_merge_coupling"]
