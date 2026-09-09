from terminal_marl.training.run_v4_l1_short_pilot import V4EpisodeSummary, _by_scenario, _validation_line


def test_scenario_metrics_skip_episode_groups_without_joint_decisions():
    """An empty finite cohort must not abort a completed PPO pilot."""
    # The grouping path reads only these two fields before it short-circuits.
    summary = object.__new__(V4EpisodeSummary)
    object.__setattr__(summary, "reset_mode", "flow_from_empty")
    object.__setattr__(summary, "demand_model", "finite_cohort")
    # No model is constructed: the function should short-circuit before
    # categorical metrics are requested for this zero-decision group.
    assert _by_scenario([([], summary)], {}) == {}


def test_validation_markdown_handles_a_zero_decision_partition():
    line = _validation_line("warm-start", {"environment": {"completed": 0, "throughput_per_second": 0.0,
                                                        "external_queue_mean": 0.0, "shield_burden_per_100_joint_decisions": None}})
    assert "no joint decisions" in line


def test_validation_markdown_handles_the_static_joint_anchor_shape():
    joint = {"joint_policy_teacher_agreement": 0.5, "joint_teacher_intervention_rate": 0.5,
             "decoded": {"requested": {"WAIT": 2, "RELEASE": 3}, "teacher": {"WAIT": 3, "RELEASE": 2},
                         "release_disagreement_rate": 0.5, "speed_disagreement_rate": 0.2, "merge_disagreement_rate": 0.1}}
    assert "agreement=0.5000" in _validation_line("anchor", joint)
