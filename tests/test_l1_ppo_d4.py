"""Small output-closure tests for D4 credit assignment variants."""
from __future__ import annotations

import numpy as np
import torch

from terminal_marl.algo import RolloutBuffer, SharedPPO, Transition
from terminal_marl.core.observations import LOW_OBS_V2
from terminal_marl.training.l1_ppo_d3_bounded_distribution import _requested_action_provenance
from terminal_marl.training.l1_ppo_d4_credit_assignment import _markdown, _variant_summary
from terminal_marl.training.train_low_v2_ppo import _release_teacher_target


def test_headwise_intervention_mask_keeps_requested_provenance_and_masks_only_corrected_head() -> None:
    torch.manual_seed(31)
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05, teacher_bc_loss="squashed_nll",
                       intervention_ppo_mask=True)
    observation = np.zeros((2, LOW_OBS_V2.dimension), dtype=np.float32)
    actions, log_probs, value, log_probs_by_head = policy.act(observation, observation, return_head_log_probs=True)
    buffer = RolloutBuffer()
    for index, reward in enumerate((1.0, -1.0)):
        teacher = actions[index].copy()
        teacher[0] = 1.0  # exact endpoint remains the external teacher semantic
        buffer.add(Transition(f"uav-{index}", observation[index], observation, actions[index], float(log_probs[index]),
                              value, reward, True, teacher_action=teacher, applied_action=teacher, teacher_used=True,
                              log_prob_by_head=log_probs_by_head[index]))
    snapshot = policy.diagnostic_snapshot(buffer, teacher_bc_weight=.05)
    credit = snapshot["heads"]["release_request"]["teacher_rescue_credit_assignment"]
    assert credit["corrected_samples"] == credit["ppo_masked_corrected_samples"] == 2
    assert credit["active_credit_leakage_samples"] == 0
    assert np.array_equal(buffer.by_agent["uav-0"][0].action, actions[0])
    assert np.allclose(log_probs, log_probs_by_head.sum(axis=-1))
    provenance = _requested_action_provenance(policy, buffer)
    assert provenance["passed"] and not provenance["a_applied_enters_ratio"]
    metrics = policy.update(buffer, epochs=1, minibatch_size=2)
    assert np.isclose(metrics["intervention_masked_head_fraction"], 1 / 3)
    assert all(np.isfinite(metric) for metric in metrics.values())


def test_provenance_allows_only_float32_log_prob_roundoff_not_a_real_density_mismatch() -> None:
    policy = SharedPPO(LOW_OBS_V2.dimension)
    observation = np.zeros((1, LOW_OBS_V2.dimension), dtype=np.float32)
    action, log_prob, value = policy.act(observation, observation)
    rounded = RolloutBuffer()
    rounded.add(Transition("roundoff", observation[0], observation, action[0], float(log_prob[0] + 8e-6),
                           value, 0.0, True))
    assert _requested_action_provenance(policy, rounded)["passed"]
    mismatched = RolloutBuffer()
    mismatched.add(Transition("wrong-density", observation[0], observation, action[0], float(log_prob[0] + 2e-4),
                              value, 0.0, True))
    assert not _requested_action_provenance(policy, mismatched)["passed"]


def test_release_teacher_target_covers_wait_release_and_scheduler_rejection() -> None:
    assert _release_teacher_target(release_eligible=False, requested_release=True, decision="plan_active") == -1.0
    assert _release_teacher_target(release_eligible=True, requested_release=False, decision="not_requested") == 1.0
    assert _release_teacher_target(release_eligible=True, requested_release=True, decision="approved") == 1.0
    assert _release_teacher_target(release_eligible=True, requested_release=True, decision="safety_wait") == -1.0


def test_d4_summary_and_required_markdown_are_closed_over_serialized_records() -> None:
    """Guard the final aggregation path before a multi-hour server diagnostic."""
    credit = {
        "corrected_samples": 2, "corrected_positive_advantage_samples": 1,
        "corrected_positive_advantage_negative_cosine_samples": 1,
        "active_credit_leakage_samples": 1, "ppo_masked_corrected_samples": 0,
        "samples": [{"ppo_bc_cosine_pre_mask": -0.8}],
    }
    head = {
        "teacher_rescue_credit_assignment": credit,
        "teacher_used_head_output_gradient_cosine": {"negative_fraction": .5},
        "saturation_rate": .1, "post_squash_actor_mean": {"mean": .2}, "std": .4,
    }
    summary = {
        "update": 0, "episode": 0, "seed": 1, "reset_mode": "warm_start_rings",
        "demand_model": "finite_cohort", "warm_start_uavs": 8, "cohort_uavs": 8,
        "rollout_rows": 2, "reward": 1.0, "arrivals": 8, "requested": 8, "admitted": 8,
        "completed": 8, "release_intents": 4, "teacher_interventions": 1,
        "shield_interventions": 0, "correction_magnitude_sum": .2,
        "correction_magnitudes": [.2], "teacher_correction_components": [["release"]],
        "requested_action_saturation_counts": [0, 0, 0], "requested_action_count": 2,
        "completion_delays": [1.0], "collisions": 0, "nmac": 0,
        "hard_separation_violations": 0, "steps": 10,
    }
    diagnostic = {
        "heads": {name: head for name in ("release_request", "speed_residual", "merge_time_residual")},
        "optimizer": {"learning_rate": 3e-4, "entropy_coefficient": .01, "clip_ratio": .2},
        "advantages": {"raw": {"mean": .1, "p90": .2, "p10": 0.0},
                       "normalized": {"mean": 0.0, "p90": 1.0, "p10": -1.0}},
    }
    record = {"update": 0, "summaries": [summary], "diagnostic": diagnostic,
              "ppo_update": {"approx_kl": .001, "clip_fraction": .01, "actor_loss": .1,
                             "value_loss": .2, "entropy": 1.0, "explained_variance": .3,
                             "teacher_bc_loss": .4, "loss": .5, "grad_norm": .6,
                             "intervention_masked_head_fraction": 0.0}}
    values = _variant_summary([record], [{"passed": True}])
    assert values["provenance_passed"] and values["safety_passed"]
    result = {"decision": {"teacher_rescue_credit_leakage_established": True,
                            "worst_head": "release_request",
                            "squashed_bc_improves_release_leakage": False,
                            "headwise_mask_improves_release_leakage": False,
                            "allow_resume_formal_l1": False,
                            "next_step": "release-head architecture / objective diagnostic"},
              "variants": {"A_current_tanh_current_bc": values,
                           "B_tanh_squashed_bc": values,
                           "C_tanh_squashed_bc_headwise_mask": values}}
    markdown = _markdown(result)
    assert "teacher-rescue credit leakage" in markdown
    assert "是否允许恢复正式 L1" in markdown
