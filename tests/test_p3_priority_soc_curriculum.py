from terminal_marl.training import run_v4_2_p3_priority_soc_curriculum_v1 as p3


def test_p3_protocol_changes_only_priority_and_soc_curriculum():
    value = p3.protocol()
    assert value["bindings"]["selected_candidate"] == "original_k3"
    assert value["bindings"]["runtime_proposal_shortlist_k"] == 3
    assert value["bindings"]["fixed_geometry"]
    assert value["scenario"]["priority_mode"] == "random_135"
    assert value["scenario"]["priority_values"] == [1, 3, 5]
    assert value["scenario"]["initial_soc_mode"] == "uniform"
    assert value["scenario"]["initial_soc_min"] == 0.55
    assert value["scenario"]["initial_soc_max"] == 0.90


def test_p3_config_preserves_p2_flow_and_enables_heterogeneity():
    value = p3.protocol()
    config = p3.p3_config(12, value)
    assert config.total_requested_uavs == 12
    assert config.fixed_arrival_schedule == tuple(10.0 + 60.0 * i for i in range(12))
    assert config.priority_mode == "random_135"
    assert config.initial_priority is None
    assert config.initial_soc_mode == "uniform"
    assert config.initial_soc_min == 0.55
    assert config.initial_soc_max == 0.90
    assert config.runtime_proposal_shortlist_k == 3
    assert not config.high_level_enabled
    assert not config.teacher_enabled


def test_p3_selection_review_validation_is_fail_closed():
    value = p3.protocol()
    report = {
        "stage": "V4_2_P2_RUNTIME_SHORTLIST_SELECTION_REVIEW_V1",
        "PASS": True, "gates": {"all": True},
        "SELECTED_LOW_LEVEL_CANDIDATE": "original_k3",
        "selected_checkpoint_sha256": value["bindings"]["initial_checkpoint_sha256"],
        "P2_PPO_STOP": True,
        "P3_PROTOCOL_PREREGISTRATION_ALLOWED": True,
        "P3_ALLOWED": False,
    }
    p3.validate_selection_review(report, value)
    report["SELECTED_LOW_LEVEL_CANDIDATE"] = "formal_k3"
    try:
        p3.validate_selection_review(report, value)
    except PermissionError:
        pass
    else:
        raise AssertionError("selection review must fail closed")


def test_p3_source_hashes_are_repository_relative():
    hashes = p3.source_hashes()
    assert "terminal_marl/core/arrival_process.py" in hashes
    assert "terminal_marl/core/priority.py" in hashes
    assert all(not key.startswith(("/", "\\")) and ":" not in key for key in hashes)
    assert all(len(value) == 64 for value in hashes.values())
